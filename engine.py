import os
import random
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import transforms as T
import utils
from data.dataloader_util import colllate_fn_custom


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name if torch.cuda.is_available() and str(device_name).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    return device


def model_cfg(use_lvmsf: bool = False, locate_ckpt: str = "", alpha: float = 0.5, use_localization: bool = False):
    return SimpleNamespace(
        coarse=SimpleNamespace(use_lvmsf=bool(use_lvmsf)),
        use_localization_guidance=bool(use_localization or locate_ckpt),
        locate_ckpt=locate_ckpt or "",
        alpha=float(alpha),
    )


def get_transform(img_size: int):
    return T.Compose(
        [
            T.Resize(img_size, img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_dataset(args, split: str):
    from data.refdataset import ReferDataset

    return ReferDataset(args, None, image_transforms=get_transform(int(args.img_size)), split=split)


def make_loader(dataset, batch_size: int, workers: int, pin_memory: bool, train: bool):
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=train and len(dataset) >= batch_size,
        collate_fn=colllate_fn_custom,
    )


def batch_to_device(data: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "image": data["image"].to(device, non_blocking=True),
        "target": data["target"].to(device, non_blocking=True),
        "text": data["tensor_embeddings"].to(device, non_blocking=True).squeeze(1),
        "l_mask": data["attention_mask"].to(device, non_blocking=True),
    }


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state.keys()):
        return {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def load_model_weights(model, ckpt_path: str, label: str = "Model"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(strip_module_prefix(state), strict=False)
    print(f"[{label}] loaded {ckpt_path}: {msg}")
    return msg


def save_training_checkpoint(path: str, model, optimizer, scheduler, epoch: int, args) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
    }
    torch.save(payload, path)


def parse_epochs(raw: str) -> Tuple[int, ...]:
    epochs = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            epochs.append(int(part[2:] if part.lower().startswith("ep") else part))
    return tuple(epochs)


def build_poly_scheduler(optimizer, steps_per_epoch: int, epochs: int):
    total_steps = max(int(steps_per_epoch) * int(epochs), 1)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1 - step / total_steps) ** 0.9)


def build_coarse_optimizer(model, args):
    backbone_no_decay = []
    backbone_decay = []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "absolute_pos_embed" in name or "relative_position_bias_table" in name:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    text_params = []
    for layer in model.text_encoder.encoder.layer[:10]:
        text_params.extend(p for p in layer.parameters() if p.requires_grad)

    param_groups = [
        {"params": backbone_no_decay, "weight_decay": 0.0},
        {"params": backbone_decay},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        {"params": text_params},
    ]
    return torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))


def train_segmentation_epoch(model, criterion, optimizer, scheduler, loader, device, epoch: int, print_freq: int):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("total_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("ce_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("dice_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))

    for data in metric_logger.log_every(loader, print_freq, f"Epoch: [{epoch}]"):
        batch = batch_to_device(data, device)
        out = model(batch["image"], batch["text"], l_mask=batch["l_mask"])
        loss_dict = criterion(pred=out["x"], targ=batch["target"])
        loss = loss_dict["total_loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate_segmentation(model, loader, device, header: str = "Test:"):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    eval_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(eval_thresholds), dtype=np.int32)
    seg_total = 0
    mean_iou = []
    cum_i = 0.0
    cum_u = 0.0

    for data in metric_logger.log_every(loader, 100, header):
        batch = batch_to_device(data, device)
        out = model(batch["image"], batch["text"], l_mask=batch["l_mask"])
        pred = out["x"].argmax(dim=1)

        pred_fg = pred == 1
        tgt_fg = batch["target"] == 1
        inter = (pred_fg & tgt_fg).float().flatten(1).sum(dim=1)
        union = (pred_fg | tgt_fg).float().flatten(1).sum(dim=1)
        iou = inter / (union + 1e-6)

        mean_iou.extend(iou.detach().cpu().tolist())
        cum_i += float(inter.sum().item())
        cum_u += float(union.sum().item())
        for value in iou.detach().cpu().tolist():
            for idx, threshold in enumerate(eval_thresholds):
                seg_correct[idx] += value >= threshold
        seg_total += pred.shape[0]

    miou = float(np.mean(np.asarray(mean_iou))) if mean_iou else 0.0
    giou = cum_i / (cum_u + 1e-6)
    print("Final results:")
    print("Mean IoU is %.2f" % (miou * 100.0))
    for idx, threshold in enumerate(eval_thresholds):
        print("    precision@%s = %.2f" % (threshold, seg_correct[idx] * 100.0 / max(seg_total, 1)))
    print("    overall IoU = %.2f" % (giou * 100.0))
    return miou * 100.0, giou * 100.0


def foreground_iou_from_logits(logits_120: torch.Tensor, target: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    pred = logits_120.argmax(dim=1)
    target_120 = F.interpolate(target.unsqueeze(1).float(), size=size_hw, mode="nearest").squeeze(1).long()
    pred_fg = pred == 1
    tgt_fg = target_120 == 1
    inter = (pred_fg & tgt_fg).float().flatten(1).sum(dim=1)
    union = (pred_fg | tgt_fg).float().flatten(1).sum(dim=1)
    return inter / (union + 1e-6)
