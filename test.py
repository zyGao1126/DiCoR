import os

import torch

from args import test_parser
from engine import evaluate_segmentation, load_model_weights, make_dataset, make_loader, model_cfg, resolve_device, seed_everything


def load_refiner_weights(model, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(key.startswith("refineHead.") for key in state.keys()):
        state = {key.replace("refineHead.", "", 1): value for key, value in state.items() if key.startswith("refineHead.")}
    msg = model.refineHead.load_state_dict(state, strict=False)
    print(f"[Test] loaded refiner {ckpt_path}: {msg}")


def main():
    args = test_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.visual_dir, exist_ok=True)

    use_refiner = bool(args.refiner_ckpt)
    use_locate = bool(args.locate_ckpt)
    cfg = model_cfg(use_lvmsf=use_locate, locate_ckpt=args.locate_ckpt, alpha=args.alpha, use_localization=use_locate)

    from lib import segmentation

    if use_refiner:
        model = segmentation.dicor_refiner_test(
            pretrained=args.pretrained_swin_weights,
            pretrained_refineHead="",
            args=args,
            cfg=cfg,
        ).to(device)
    else:
        model = segmentation.dicor_coarse(
            pretrained=args.pretrained_swin_weights,
            pretrained_refineHead="",
            args=args,
            cfg=cfg,
        ).to(device)

    load_model_weights(model, args.coarse_ckpt, label="Test coarse")
    if use_refiner:
        load_refiner_weights(model, args.refiner_ckpt)

    dataset = make_dataset(args, args.split)
    loader = make_loader(dataset, args.batch_size, args.workers, args.pin_mem, train=False)
    evaluate_segmentation(model, loader, device, header=f"Test {args.split}:")


if __name__ == "__main__":
    main()
