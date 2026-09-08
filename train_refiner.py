import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from args import refiner_parser
from data.dataloader_util import colllate_fn_custom
from engine import build_poly_scheduler, evaluate_segmentation, load_model_weights, make_dataset, model_cfg, resolve_device, set_random_seed
from offline_bank import LCRProbabilityBank


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
        indices = data["index"].to(device, non_blocking=True).long()
        snapshot_indices = torch.randint(
            low=0,
            high=bank.num_snapshots,
            size=(image.size(0),),
            dtype=torch.long,
            device=device,
        )
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


def evaluate_refiner(args, refine_head_state, test_loader, device):
    from lib import segmentation

    model = segmentation.dicor_refiner_test(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(visual_fusion=args.visual_fusion),
    ).to(device)
    load_model_weights(model, args.coarse_ckpt, label="RefinerEval coarse")
    model.refineHead.load_state_dict(refine_head_state, strict=True)
    return evaluate_segmentation(model, test_loader, device, header="Refiner Test:")


def main():
    args = refiner_parser().parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    from lib._utils import DiCoRRefinerTrain
    from lib.refiner import RefineUNet
    from loss.loss import RefinerLoss

    train_base = make_dataset(args, "train")
    test_ds = make_dataset(args, "test")
    bank = LCRProbabilityBank(args.offline_bank_dir, map_size=args.img_size // 4)
    if len(train_base) != bank.sample_count:
        raise RuntimeError(
            f"LCR bank contains {bank.sample_count} samples, but the training set "
            f"contains {len(train_base)}"
        )
    if not bank.has_valid_probabilities():
        raise RuntimeError("The LCR bank contains no valid probability maps")

    train_loader = DataLoader(
        train_base,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_base),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=len(train_base) >= args.batch_size,
        collate_fn=colllate_fn_custom,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_ds),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=colllate_fn_custom,
    )

    model = DiCoRRefinerTrain(refineHead=RefineUNet(in_ch=4, base_ch=64)).to(device)
    criterion = RefinerLoss().to(device)
    optimizer = torch.optim.AdamW(model.refineHead.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_poly_scheduler(optimizer, len(train_loader), args.epochs)

    best_giou = -1.0
    start = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, scheduler, train_loader, bank, device, epoch, args.print_freq)
        _, giou = evaluate_refiner(args, model.refineHead.state_dict(), test_loader, device)
        if giou > best_giou:
            best_giou = giou
            torch.save(model.refineHead.state_dict(), os.path.join(args.output_dir, "refiner.pth"))
            print(f"[Refiner] best gIoU={best_giou:.2f}")
        torch.save(model.refineHead.state_dict(), os.path.join(args.output_dir, f"refiner_ep{epoch + 1}.pth"))

    print(f"[Refiner] finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
