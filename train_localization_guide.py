import math
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

import utils
from args import localization_parser
from data.dataloader_util import colllate_fn_custom
from engine import (
    build_poly_scheduler,
    evaluate_segmentation,
    load_model_weights,
    make_dataset,
    make_loader,
    model_cfg,
    resolve_device,
    set_random_seed,
)
from offline_bank import DLGFeatureBank
from lib import segmentation
from lib.localization_guidance import build_localization_guidance, build_evidence_supervision, token_valid_mask, compute_localization_loss

def set_requires_grad(module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


class LocalizationAnnotations(Dataset):
    """Language and localization labels for selected training rows."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset.get_localization_item(self.indices[index])


class DLGBatches:
    """One epoch with one selected snapshot per training sample."""

    def __init__(self, data, batch_size: int):
        self.data = data
        self.batch_size = int(batch_size)

    def __len__(self):
        return math.ceil(self.data.samples / self.batch_size)

    def __iter__(self):
        bank = self.data.bank
        snapshot_indices = torch.randint(
            bank.num_snapshots,
            (self.data.samples,),
        )
        offsets = torch.tensor(bank.snapshot_offsets)
        rows = self.data.dataset_indices + offsets.index_select(0, snapshot_indices)
        row_shards = torch.div(rows, bank.shard_size, rounding_mode="floor")
        shard_indices = torch.unique(row_shards)
        shard_indices = shard_indices[
            torch.randperm(shard_indices.numel())
        ]

        pending = {}
        for shard_index in shard_indices.tolist():
            positions = (row_shards == shard_index).nonzero(as_tuple=False).flatten()
            positions = positions[torch.randperm(positions.numel())]
            local_rows = rows[positions] - shard_index * bank.shard_size
            selected = {name: value[positions] for name, value in self.data.prepared.items()}
            selected["feature_map"] = bank.load_shard(bank.shards[shard_index])[local_rows]

            for name, value in selected.items():
                pending[name] = torch.cat((pending[name], value)) if name in pending else value
            while len(pending["feature_map"]) >= self.batch_size:
                yield {name: value[:self.batch_size] for name, value in pending.items()}
                pending = {name: value[self.batch_size:] for name, value in pending.items()}

        if pending:
            yield pending


class PreparedDLGData:
    def __init__(self, bank, dataset_indices, prepared):
        self.bank = bank
        self.dataset_indices = dataset_indices
        self.prepared = prepared
        self.samples = int(dataset_indices.numel())

    def batches(self, batch_size: int):
        return DLGBatches(self, batch_size)


@torch.no_grad()
def prepare_training_data(dataset, bank, adapter, text_encoder, args, device):
    selected_indices = [
        index
        for index, item in enumerate(dataset.processed_data)
        if 0.0 < item["area_ratio"] < adapter.area_threshold
    ]
    if not selected_indices:
        raise ValueError("The training set contains no valid samples with area_ratio < 2%")

    annotations = LocalizationAnnotations(dataset, selected_indices)
    loader = DataLoader(
        annotations,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=colllate_fn_custom,
    )
    parts = {
        "dataset_index": [],
        "text_tokens": [],
        "token_valid": [],
        "pos": [],
        "far_bg": [],
        "sam_weak_neg": [],
        "center": [],
    }
    out_h, out_w = bank.feature_shape[-2:]
    text_encoder.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    for data in metric_logger.log_every(loader, 100, "Prepare DLG training data"):
        input_ids = data["tensor_embeddings"].to(device, non_blocking=True).squeeze(1)
        attention_mask = data["attention_mask"].to(device, non_blocking=True)
        text_tokens = text_encoder(input_ids, attention_mask=attention_mask.squeeze(-1))[0]
        supervision = build_evidence_supervision(
            target=data["target"].to(device, non_blocking=True),
            sam3_masks=data["sam3_masks"],
            out_h=out_h,
            out_w=out_w,
        )

        parts["dataset_index"].append(data["index"].long())
        parts["text_tokens"].append(text_tokens.to(dtype=torch.float16).cpu())
        parts["token_valid"].append(
            token_valid_mask(input_ids, attention_mask.permute(0, 2, 1)).cpu()
        )
        for name in ("pos", "far_bg", "sam_weak_neg"):
            parts[name].append(supervision[name].bool().cpu())
        parts["center"].append(supervision["center"].float().cpu())

    prepared = {
        name: torch.cat(values, dim=0)
        for name, values in parts.items()
    }
    dataset_indices = prepared.pop("dataset_index")
    print(
        f"[DLG] selected {dataset_indices.numel()}/{len(dataset)} training samples "
        f"with 0 < area_ratio < {adapter.area_threshold:.2%}; "
        f"snapshots={bank.num_snapshots}"
    )
    return PreparedDLGData(bank, dataset_indices, prepared)


def build_optimizer(adapter, args):
    evidence_params = list(adapter.module.evidence_head.parameters())
    ranker_params = list(adapter.module.ranker.parameters())
    print(
        f"[DLG] evidence parameters: "
        f"{sum(parameter.numel() for parameter in evidence_params)}"
    )
    print(
        f"[DLG] ranker parameters:   "
        f"{sum(parameter.numel() for parameter in ranker_params)}"
    )
    return torch.optim.AdamW(
        [
            {
                "params": evidence_params,
                "lr": args.evidence_lr,
                "weight_decay": args.evidence_weight_decay,
            },
            {
                "params": ranker_params,
                "lr": args.winner_lr,
                "weight_decay": args.winner_weight_decay,
            },
        ]
    )

def move_training_batch(data, device):
    return {
        "feature_map": data["feature_map"].to(device=device, dtype=torch.float32, non_blocking=True),
        "text_tokens": data["text_tokens"].to(device=device, dtype=torch.float32, non_blocking=True),
        "token_valid": data["token_valid"].to(device=device, non_blocking=True),
        "supervision": {
            name: data[name].to(device=device, non_blocking=True)
            for name in ("pos", "far_bg", "sam_weak_neg", "center")
        },
    }

def train_one_epoch(adapter, optimizer, scheduler, batches, device, epoch, print_freq):
    adapter.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("dlg_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("evidence_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("winner_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))

    for data in metric_logger.log_every(batches, print_freq, f"DLG Epoch: [{epoch}]"):
        batch = move_training_batch(data, device)
        guidance = adapter.module(
            feature_map=batch["feature_map"],
            text_tokens=batch["text_tokens"],
            token_valid=batch["token_valid"],
            generator=adapter.generator,
        )
        loss, loss_dict = compute_localization_loss(guidance, batch["supervision"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def save_guide(path: str, adapter) -> None:
    torch.save(adapter.state_dict(), path)


def main():
    args = localization_parser().parse_args()
    if args.workers > 0:
        # Avoid file-descriptor transfer failures for SAM3 tensors.
        torch.multiprocessing.set_sharing_strategy("file_system")
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = make_dataset(args, "train")
    bank = DLGFeatureBank(args.offline_bank_dir, sample_count=len(train_dataset))
    val_dataset = make_dataset(args, "val")

    model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(visual_fusion=args.visual_fusion),
    ).to(device)
    incompatible = load_model_weights(model, args.coarse_ckpt, label="DLG coarse model")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "The coarse checkpoint does not match the current model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    set_requires_grad(model, False)
    model.eval()

    adapter = build_localization_guidance(alpha=args.alpha, lambda_geo=args.lambda_geo)
    adapter = adapter.to(device)
    training_data = prepare_training_data(
        train_dataset,
        bank,
        adapter,
        model.text_encoder,
        args,
        device,
    )
    model.backbone.set_localization_guidance(adapter)

    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.workers,
        args.pin_mem,
        train=False,
    )
    optimizer = build_optimizer(adapter, args)
    steps_per_epoch = len(training_data.batches(args.batch_size))
    scheduler = build_poly_scheduler(optimizer, steps_per_epoch, args.epochs)

    print(f"[DLG] epochs={args.epochs}, samples={training_data.samples}")
    best_val_miou = -1.0
    start = time.time()
    for epoch in range(args.epochs):
        batches = training_data.batches(args.batch_size)
        train_one_epoch(adapter, optimizer, scheduler, batches, device, epoch, args.print_freq)
        val_miou, _ = evaluate_segmentation(model, val_loader, device, header=f"DLG Val Epoch [{epoch}]:")
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_guide(os.path.join(args.output_dir, "localization_guidance_best.pth"), adapter)
            print(f"[DLG] best val mIoU={best_val_miou:.2f}")

    save_guide(os.path.join(args.output_dir, "localization_guidance_last.pth"), adapter)
    print(f"[DLG] finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
