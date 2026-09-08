import os
import time
import torch
from args import baseline_parser
from engine import (
    build_coarse_optimizer,
    build_poly_scheduler,
    evaluate_segmentation,
    make_dataset,
    make_loader,
    model_cfg,
    parse_epochs,
    resolve_device,
    save_training_checkpoint,
    set_random_seed,
    train_segmentation_epoch,
)
from lib import segmentation
from loss.loss import CoarseLoss

def main():
    args = baseline_parser().parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = make_dataset(args, "train")
    val_ds = make_dataset(args, "val")
    train_loader = make_loader(train_ds, args.batch_size, args.workers, args.pin_mem, train=True)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, args.pin_mem, train=False)

    model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(visual_fusion=args.visual_fusion),
    ).to(device)

    criterion = CoarseLoss().to(device)
    optimizer = build_coarse_optimizer(model, args)
    scheduler = build_poly_scheduler(optimizer, len(train_loader), args.epochs)
    snapshot_epochs = set(parse_epochs(args.snapshot_epochs))
    best_val_giou = -1.0

    start = time.time()
    for epoch in range(args.epochs):
        train_segmentation_epoch(model, criterion, optimizer, scheduler, train_loader, device, epoch, args.print_freq)
        _, val_giou = evaluate_segmentation(model, val_loader, device, header=f"Val Epoch [{epoch}]:")

        if val_giou > best_val_giou:
            best_val_giou = val_giou
            save_training_checkpoint(os.path.join(args.output_dir, "coarse_best.pth"), model, optimizer, scheduler, epoch, args)
            print(f"[Baseline] best val gIoU={best_val_giou:.2f}")

        if epoch + 1 in snapshot_epochs:
            path = os.path.join(args.output_dir, f"coarse_ep{epoch + 1}.pth")
            save_training_checkpoint(path, model, optimizer, scheduler, epoch, args)
            print(f"[Baseline] snapshot saved: {path}")

    print(f"[Baseline] finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
