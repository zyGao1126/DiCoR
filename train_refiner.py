import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from args import refiner_parser
from data.dataloader_util import colllate_fn_custom
from engine import build_poly_scheduler, evaluate_segmentation, load_model_weights, make_dataset, model_cfg, resolve_device, set_random_seed
from offline_bank import LCRProbabilityBank
from lib._utils import DiCoRRefinerTrain
from lib.refiner import RefineUNet
from loss.loss import RefinerLoss


def train_one_epoch(model, criterion, optimizer, scheduler, loader, bank, device, epoch, print_freq):
    import utils

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("total_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("stage2_ce_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("stage2_dice_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("stage2_inhibit_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("valid_prompts", utils.SmoothedValue(window_size=20, fmt="{value:.0f}"))

    for data in metric_logger.log_every(loader, print_freq, f"Refiner Epoch: [{epoch}]"):
        image = data["image"].to(device, non_blocking=True)
        target = data["target"].to(device, non_blocking=True)
        indices = data["index"].long()
        snapshot_indices = torch.randint(0, bank.num_snapshots, (image.size(0),), dtype=torch.long)
        prompt = bank.load_probabilities(indices, snapshot_indices, device)

        valid_prompt = prompt.flatten(1).sum(dim=1) > 0
        metric_logger.update(valid_prompts=int(valid_prompt.sum().item()))
        if not valid_prompt.any():
            continue

        image = image[valid_prompt]
        target = target[valid_prompt]
        prompt = prompt[valid_prompt]

        out = model(image, prompt_override=prompt)
        loss_dict = criterion(
            pred=out["x"],
            targ=target,
            delta_logits_480=out["delta_logits_480"],
            focus_map=out["focus_map"],
        )
        loss = loss_dict["total_loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def build_evaluation_model(args, device):
    from lib import segmentation

    model = segmentation.dicor_refiner_test(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(visual_fusion=args.visual_fusion),
    ).to(device)
    incompatible = load_model_weights(model, args.coarse_ckpt, label="RefinerEval coarse")
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.startswith("refineHead.")]
    if unexpected or missing:
        raise RuntimeError(
            "The coarse checkpoint does not match the Refiner evaluation model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


def evaluate_refiner(model, refine_head_state, loader, device):
    model.refineHead.load_state_dict(refine_head_state, strict=True)
    return evaluate_segmentation(model, loader, device, header="Refiner Val:")


def main():
    args = refiner_parser().parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_base = make_dataset(args, "train")
    val_ds = make_dataset(args, "val")
    bank = LCRProbabilityBank(args.offline_bank_dir, map_size=args.img_size // 4)
    if len(train_base) != bank.sample_count:
        raise RuntimeError(f"LCR bank contains {bank.sample_count} samples, but the training set contains {len(train_base)}")

    train_loader = DataLoader(
        train_base,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_base),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=len(train_base) >= args.batch_size,
        collate_fn=colllate_fn_custom,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_ds),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=colllate_fn_custom,
    )

    model = DiCoRRefinerTrain(refineHead=RefineUNet(in_ch=4, base_ch=64)).to(device)
    model.PROMPT_PROCESSOR.aug_morph_prob = args.aug_morph_prob
    model.PROMPT_PROCESSOR.morph_max_radius = args.morph_max_radius
    evaluation_model = build_evaluation_model(args, device)
    print(
        "[Refiner Config] "
        f"dataset={args.dataset} base_ch=64 "
        f"batch_size={args.batch_size} epochs={args.epochs} "
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"seed={args.seed} num_vmsf_blocks={args.num_vmsf_blocks} "
        f"num_heads_fusion={args.num_heads_fusion} visual_fusion={args.visual_fusion} "
        f"window12={args.window12} aug_prob={model.PROMPT_PROCESSOR.aug_prob} "
        f"aug_morph_prob={args.aug_morph_prob} morph_max_radius={args.morph_max_radius} "
        f"train_samples={len(train_base)}"
    )

    criterion = RefinerLoss().to(device)
    optimizer = torch.optim.AdamW(model.refineHead.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_poly_scheduler(optimizer, len(train_loader), args.epochs)

    best_miou = -1.0
    best_miou_epoch = None

    start = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, scheduler, train_loader, bank, device, epoch, args.print_freq)
        miou, giou = evaluate_refiner(evaluation_model, model.refineHead.state_dict(), val_loader, device)
        is_best_miou = miou > best_miou

        if is_best_miou:
            best_miou = miou
            best_miou_epoch = epoch + 1
            torch.save(model.refineHead.state_dict(), os.path.join(args.output_dir, "refiner_best.pth"))

        print(
            f"[Refiner Metrics] epoch={epoch + 1} "
            f"val_gIoU={giou:.10f} val_mIoU={miou:.10f} "
            f"best_mIoU={best_miou:.10f}@{best_miou_epoch}"
        )

    elapsed_seconds = time.time() - start
    print(
        f"[Refiner] finished in {elapsed_seconds / 3600:.2f}h; "
        f"best_mIoU={best_miou:.10f}@{best_miou_epoch}"
    )


if __name__ == "__main__":
    main()
