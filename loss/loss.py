from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary/semantic segmentation logits."""

    def __init__(
        self,
        axis: int = 1,
        smooth: float = 1e-6,
        reduction: str = "mean",
        square_in_union: bool = False,
    ):
        super().__init__()
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        targ = self._one_hot(targ, pred.shape[self.axis])
        assert pred.shape == targ.shape, "input and target dimensions differ, DiceLoss expects non one-hot targets"

        pred = F.softmax(pred, dim=self.axis)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        if self.square_in_union:
            union = torch.sum(pred**2, dim=sum_dims) + torch.sum(targ, dim=sum_dims)
        else:
            union = torch.sum(pred, dim=sum_dims) + torch.sum(targ, dim=sum_dims)

        loss = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

    @staticmethod
    def _one_hot(x: torch.Tensor, classes: int) -> torch.Tensor:
        one_hot_targ = torch.zeros(
            (x.size(0), classes, x.size(1), x.size(2)),
            device=x.device,
            dtype=torch.float32,
        )
        for c in range(classes):
            one_hot_targ[:, c, :, :] = (x == c)
        return one_hot_targ


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.register_buffer("ce_weight", torch.tensor([0.9, 1.1], dtype=torch.float))
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        self.dice_loss = DiceLoss(reduction="mean")

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> Dict[str, torch.Tensor]:
        targ = targ.long()
        ce_loss = self.ce_loss(pred, targ)
        dice_loss = self.dice_loss(pred, targ)
        total_loss = ce_loss + self.dice_weight * dice_loss
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "dice_loss": self.dice_weight * dice_loss,
        }


class CoarseLoss(SegmentationLoss):
    pass


class Stage2SegLoss(nn.Module):
    """Localized CE + Dice loss for final refiner logits."""

    def __init__(self, dice_weight: float = 1.0, ce_weight: torch.Tensor = None):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = ce_weight
        self.ce_loss_raw = nn.CrossEntropyLoss(weight=self.ce_weight, reduction="none")

    @staticmethod
    def _weighted_dice_per_sample(
        logits: torch.Tensor,
        target: torch.Tensor,
        w_pix: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        prob_fg = torch.softmax(logits, dim=1)[:, 1]
        gt_fg = (target == 1).float()
        w = w_pix.squeeze(1)

        inter = (w * prob_fg * gt_fg).flatten(1).sum(1)
        p_sum = (w * prob_fg).flatten(1).sum(1)
        g_sum = (w * gt_fg).flatten(1).sum(1)

        return 1.0 - (2.0 * inter + eps) / (p_sum + g_sum + eps)

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        focus_map: torch.Tensor,
        delta_logits: torch.Tensor,
        delta_inhibit_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = pred.shape[0]
        targ = targ.long()
        w_pix = focus_map.to(device=pred.device, dtype=pred.dtype).clamp_min(0.0)
        w_pix2 = w_pix.squeeze(1)

        ce_map = self.ce_loss_raw(pred, targ)
        ce_num = (ce_map * w_pix2).flatten(1).sum(1)
        ce_den = w_pix2.flatten(1).sum(1).clamp_min(1e-6)
        ce_per_sample = ce_num / ce_den

        dice_per_sample = self._weighted_dice_per_sample(pred, targ, w_pix)
        seg_per_sample = ce_per_sample + self.dice_weight * dice_per_sample

        inhibit_per_sample = pred.new_zeros((B,))
        if delta_inhibit_weight > 0.0:
            outside = (1.0 - w_pix2).clamp_min(0.0)
            out_den = outside.flatten(1).sum(1).clamp_min(1e-6)
            delta_mag = delta_logits.abs().mean(dim=1)
            inhibit_per_sample = (delta_mag * outside).flatten(1).sum(1) / out_den
            seg_per_sample = seg_per_sample + float(delta_inhibit_weight) * inhibit_per_sample

        total = seg_per_sample.mean()
        ce_loss = ce_per_sample.mean()
        dice_loss = dice_per_sample.mean()
        inhibit_loss = inhibit_per_sample.mean()

        return total, {
            "stage2_ce_loss": ce_loss,
            "stage2_dice_loss": self.dice_weight * dice_loss,
            "stage2_inhibit_loss": float(delta_inhibit_weight) * inhibit_loss,
        }


class RefinerLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, delta_inhibit_weight: float = 0.05):
        super().__init__()
        self.delta_inhibit_weight = float(delta_inhibit_weight)
        self.register_buffer("ce_weight", torch.tensor([0.9, 1.1], dtype=torch.float))
        self.stage2_loss = Stage2SegLoss(dice_weight=float(dice_weight), ce_weight=self.ce_weight)

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        delta_logits_480: torch.Tensor,
        focus_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        total_loss, loss_dict = self.stage2_loss(
            pred,
            targ.long(),
            focus_map=focus_map,
            delta_logits=delta_logits_480,
            delta_inhibit_weight=self.delta_inhibit_weight,
        )
        loss_dict["total_loss"] = total_loss
        return loss_dict
