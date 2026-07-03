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
    seed_everything,
    train_segmentation_epoch,
)


def main():
    args = baseline_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    from lib import segmentation
    from loss.loss import CoarseLoss

    train_ds = make_dataset(args, "train")
    test_ds = make_dataset(args, "test")
    train_loader = make_loader(train_ds, args.batch_size, args.workers, args.pin_mem, train=True)
    test_loader = make_loader(test_ds, args.batch_size, args.workers, args.pin_mem, train=False)

    model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(use_lvmsf=False),
    ).to(device)

    criterion = CoarseLoss().to(device)
    optimizer = build_coarse_optimizer(model, args)
    scheduler = build_poly_scheduler(optimizer, len(train_loader), args.epochs)
    snapshot_epochs = set(parse_epochs(args.snapshot_epochs))
    best_giou = -1.0

    start = time.time()
    for epoch in range(args.epochs):
        train_segmentation_epoch(model, criterion, optimizer, scheduler, train_loader, device, epoch, args.print_freq)
        _, giou = evaluate_segmentation(model, test_loader, device, header=f"Test Epoch [{epoch}]:")

        if giou > best_giou:
            best_giou = giou
            save_training_checkpoint(os.path.join(args.output_dir, "coarse_best.pth"), model, optimizer, scheduler, epoch, args)
            print(f"[Baseline] best gIoU={best_giou:.2f}")

        if epoch + 1 in snapshot_epochs:
            path = os.path.join(args.output_dir, f"coarse_ep{epoch + 1}.pth")
            save_training_checkpoint(path, model, optimizer, scheduler, epoch, args)
            print(f"[Baseline] snapshot saved: {path}")

    print(f"[Baseline] finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
