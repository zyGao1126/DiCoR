import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import utils
from args import localization_parser
from data.dataloader_util import colllate_fn_custom
from engine import (
    batch_to_device,
    build_poly_scheduler,
    evaluate_segmentation,
    load_model_weights,
    make_dataset,
    model_cfg,
    resolve_device,
    save_training_checkpoint,
    seed_everything,
)
from prompt_bank import PromptBank


def set_requires_grad(module, flag: bool):
    for param in module.parameters():
        param.requires_grad_(flag)


def build_pretrain_optimizer(adapter, args):
    evidence_params = [p for p in adapter.module.evidence_head.parameters() if p.requires_grad]
    winner_params = [p for p in adapter.module.ranker.parameters() if p.requires_grad]
    print(f"[GuidePretrain] evidence params: {sum(p.numel() for p in evidence_params)}")
    print(f"[GuidePretrain] winner params:   {sum(p.numel() for p in winner_params)}")
    return torch.optim.AdamW(
        [
            {
                "name": "evidence",
                "params": evidence_params,
                "lr": args.evidence_lr,
                "weight_decay": args.evidence_weight_decay,
            },
            {
                "name": "winner",
                "params": winner_params,
                "lr": args.winner_lr,
                "weight_decay": args.winner_weight_decay,
            },
        ]
    )


def configure_joint_trainable(model):
    set_requires_grad(model, False)
    set_requires_grad(model.backbone.localization_guidance.text_norm, True)
    set_requires_grad(model.backbone.VMSF, True)
    set_requires_grad(model.classifier, True)


def build_joint_optimizer(model, args):
    token_params = [p for p in model.backbone.localization_guidance.text_norm.parameters() if p.requires_grad]
    fusion_params = [p for p in model.backbone.VMSF.parameters() if p.requires_grad]
    decoder_params = [p for p in model.classifier.parameters() if p.requires_grad]

    print(f"[JointTune] token reweight params: {sum(p.numel() for p in token_params)}")
    print(f"[JointTune] multiscale params:    {sum(p.numel() for p in fusion_params)}")
    print(f"[JointTune] decoder params:       {sum(p.numel() for p in decoder_params)}")
    return torch.optim.AdamW(
        [
            {"name": "token_reweight", "params": token_params, "lr": args.guide_lr, "weight_decay": 1e-4},
            {"name": "multiscale_fusion", "params": fusion_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
            {"name": "decoder", "params": decoder_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        ]
    )


def set_joint_train_mode(model):
    model.eval()
    model.backbone.localization_guidance.eval()
    model.backbone.localization_guidance.text_norm.train()
    model.backbone.VMSF.train()
    model.classifier.train()


def train_guide_pretrain_epoch(feature_model, adapter, bank, optimizer, scheduler, loader, device, epoch, print_freq):
    from lib.localization_guidance import compute_localization_loss

    feature_model.eval()
    adapter.module.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loc_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("evidence_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("winner_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))
    metric_logger.add_meter("valid_prompts", utils.SmoothedValue(window_size=20, fmt="{value:.0f}"))

    for data in metric_logger.log_every(loader, print_freq, f"Guide Pretrain Epoch: [{epoch}]"):
        batch = batch_to_device(data, device)
        indices = data["index"].to(device, non_blocking=True).long()
        sids = torch.randint(low=0, high=len(bank), size=(batch["image"].size(0),), dtype=torch.long, device=device)
        prompt = bank.get_batch(indices, sids, device)

        valid_prompt = prompt.flatten(1).sum(dim=1) > 0
        metric_logger.update(valid_prompts=int(valid_prompt.sum().item()))
        if not valid_prompt.any():
            continue

        keep = valid_prompt.detach().cpu().tolist()
        batch = {
            "image": batch["image"][valid_prompt],
            "target": batch["target"][valid_prompt],
            "text": batch["text"][valid_prompt],
            "l_mask": batch["l_mask"][valid_prompt],
        }
        prompt = prompt[valid_prompt]
        sam3_masks = [masks for masks, is_valid in zip(data["sam3_masks"], keep) if is_valid]

        with torch.no_grad():
            feat_out = feature_model(batch["image"], batch["text"], l_mask=batch["l_mask"])
            feature_map = feat_out["x_pre_c3_star"].detach()
            text_tokens = feat_out["l_star"].detach()

        guidance = adapter.module(
            feature_map=feature_map,
            text_tokens=text_tokens,
            input_ids=batch["text"],
            l_mask=batch["l_mask"],
            generator=adapter.generator,
            candidate_prob=prompt,
        )
        loss, loss_dict = compute_localization_loss(guidance, batch["target"], sam3_masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(
            loc_loss=loss_dict["loc_loss"],
            evidence_loss=loss_dict["evidence_loss"],
            winner_loss=loss_dict["winner_loss"],
            lr=optimizer.param_groups[0]["lr"],
        )


def evaluate_pretrain_injection(eval_model, adapter, test_loader, device, epoch):
    eval_model.backbone.localization_guidance.load_state_dict(adapter.state_dict(), strict=True)
    set_requires_grad(eval_model, False)
    eval_model.eval()
    return evaluate_segmentation(
        eval_model,
        test_loader,
        device,
        header=f"Guide Inject Test Epoch [{epoch}]:",
    )


def train_joint_epoch(model, seg_criterion, optimizer, scheduler, loader, device, epoch, print_freq):
    set_joint_train_mode(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("token_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("main_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("seg_loss", utils.SmoothedValue(window_size=20, fmt="{value:.4f}"))

    for data in metric_logger.log_every(loader, print_freq, f"Joint Tune Epoch: [{epoch}]"):
        batch = batch_to_device(data, device)
        out = model(batch["image"], batch["text"], l_mask=batch["l_mask"])
        loss_dict = seg_criterion(pred=out["x"], targ=batch["target"])
        loss = loss_dict["total_loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(
            seg_loss=loss,
            token_lr=optimizer.param_groups[0]["lr"],
            main_lr=optimizer.param_groups[1]["lr"],
        )

def save_guide(path: str, model) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.backbone.localization_guidance.state_dict(), path)


def save_adapter(path: str, adapter) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(adapter.state_dict(), path)


def main():
    args = localization_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    from lib import segmentation
    from lib.localization_guidance import build_localization_guidance
    from loss.loss import CoarseLoss

    train_base = make_dataset(args, "train")
    test_ds = make_dataset(args, "test")
    bank = PromptBank(args.prompt_bank_dir, split="localization")
    if len(train_base) != bank.N:
        raise RuntimeError(f"PromptBank N={bank.N} does not match train dataset length={len(train_base)}.")
    if not bank.has_valid_prompts():
        raise RuntimeError("No valid localization records found. Check prompt bank IoU and area thresholds.")

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

    feature_model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(use_lvmsf=False),
    ).to(device)
    load_model_weights(feature_model, args.coarse_ckpt, label="Guide pretrain coarse")
    set_requires_grad(feature_model, False)
    feature_model.eval()

    adapter, _ = build_localization_guidance(alpha=args.alpha)
    adapter = adapter.to(device)

    start = time.time()
    if args.guide_pretrain_epochs > 0:
        print(f"[GuidePretrain] epochs={args.guide_pretrain_epochs}, samples={len(train_base)}")
        pretrain_optimizer = build_pretrain_optimizer(adapter, args)
        pretrain_scheduler = build_poly_scheduler(pretrain_optimizer, len(train_loader), args.guide_pretrain_epochs)
        eval_model = segmentation.dicor_coarse(
            pretrained=args.pretrained_swin_weights,
            pretrained_refineHead="",
            args=args,
            cfg=model_cfg(use_lvmsf=False, use_localization=True, alpha=args.alpha),
        ).to(device)
        load_model_weights(eval_model, args.coarse_ckpt, label="Guide inject eval coarse")
        set_requires_grad(eval_model, False)
        best_pretrain_giou = -1.0
        for epoch in range(args.guide_pretrain_epochs):
            train_guide_pretrain_epoch(
                feature_model=feature_model,
                adapter=adapter,
                bank=bank,
                optimizer=pretrain_optimizer,
                scheduler=pretrain_scheduler,
                loader=train_loader,
                device=device,
                epoch=epoch,
                print_freq=args.print_freq,
            )
            _, pretrain_giou = evaluate_pretrain_injection(eval_model, adapter, test_loader, device, epoch)
            if pretrain_giou > best_pretrain_giou:
                best_pretrain_giou = pretrain_giou
                save_adapter(os.path.join(args.output_dir, "localization_guidance_pretrained_best.pth"), adapter)
                print(f"[GuidePretrain] best inject gIoU={best_pretrain_giou:.2f}")
        del eval_model
        save_adapter(os.path.join(args.output_dir, "localization_guidance_pretrained.pth"), adapter)
        print("[GuidePretrain] saved localization_guidance_pretrained.pth")

    del feature_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(use_lvmsf=True, use_localization=True, alpha=args.alpha),
    ).to(device)
    load_model_weights(model, args.coarse_ckpt, label="Localization coarse")
    model.backbone.localization_guidance.load_state_dict(adapter.state_dict(), strict=True)
    configure_joint_trainable(model)

    seg_criterion = CoarseLoss().to(device)
    optimizer = build_joint_optimizer(model, args)
    scheduler = build_poly_scheduler(optimizer, len(train_loader), args.epochs)

    best_giou = -1.0
    print(f"[JointTune] epochs={args.epochs}, trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for epoch in range(args.epochs):
        train_joint_epoch(model, seg_criterion, optimizer, scheduler, train_loader, device, epoch, args.print_freq)
        _, giou = evaluate_segmentation(model, test_loader, device, header=f"Localization Test Epoch [{epoch}]:")
        if giou > best_giou:
            best_giou = giou
            save_training_checkpoint(os.path.join(args.output_dir, "joint_best.pth"), model, optimizer, scheduler, epoch, args)
            save_guide(os.path.join(args.output_dir, "localization_guidance_best.pth"), model)
            print(f"[Localization] best gIoU={best_giou:.2f}")

        save_guide(os.path.join(args.output_dir, f"localization_guidance_ep{epoch + 1}.pth"), model)

    print(f"[Localization] finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
