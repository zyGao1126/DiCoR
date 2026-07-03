from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CandidateConfig:
    topk: int = 5
    peak_kernel: int = 3
    nms_radius: int = 2
    cand_sigma: float = 1.5
    min_peak_thr: float = 0.05
    gt_radius: float = 2.5
    winner_max_area_ratio: float = 0.05
    min_mass: float = 1e-8
    lambda_geo: float = 0.5
    prob_eps: float = 1e-6
    mask_special_tokens: bool = True
    special_token_ids: Tuple[int, ...] = (101, 102)


LOCALIZATION_FEATURE_KEY = "pre_x_c3"
LOCALIZATION_IN_CHANNELS = 512
EVIDENCE_HIDDEN_DIM = 128
WINNER_QUERY_DIM = 256
WINNER_ROUTER_HIDDEN_DIM = 256
WINNER_HIDDEN_DIM = 128


def spatial_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, width)
    return yy, xx


def valid_token_mask(l_mask: torch.Tensor) -> torch.Tensor:
    if l_mask.dim() == 3 and l_mask.shape[1] == 1:
        return l_mask[:, 0].bool()
    if l_mask.dim() == 3 and l_mask.shape[-1] == 1:
        return l_mask[:, :, 0].bool()
    return l_mask.bool()


def token_valid_mask(input_ids: torch.Tensor, l_mask: torch.Tensor, cfg: CandidateConfig) -> torch.Tensor:
    valid = valid_token_mask(l_mask)
    if cfg.mask_special_tokens:
        for token_id in cfg.special_token_ids:
            valid = valid & (input_ids != int(token_id))
    return valid


def masked_softmax(logits: torch.Tensor, valid: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked = logits.masked_fill(~valid, -1e4)
    weights = torch.softmax(masked, dim=dim) * valid.float()
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)


def candidate_weighted_pool(feat: torch.Tensor, cand_maps: torch.Tensor) -> torch.Tensor:
    if cand_maps.shape[-2:] != feat.shape[-2:]:
        batch_size, topk, _h, _w = cand_maps.shape
        cand_maps = F.interpolate(
            cand_maps.reshape(batch_size * topk, 1, *cand_maps.shape[-2:]),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, topk, *feat.shape[-2:])
        cand_maps = cand_maps / cand_maps.flatten(2).sum(dim=-1).view(batch_size, topk, 1, 1).clamp_min(1e-8)
    return torch.einsum("bchw,bkhw->bkc", feat, cand_maps)


def candidate_geometry(cand_maps: torch.Tensor, cand_centers: torch.Tensor, cand_valid: torch.Tensor) -> torch.Tensor:
    _batch_size, _topk, out_h, out_w = cand_maps.shape
    yy, xx = spatial_grid(out_h, out_w, cand_maps.device, cand_maps.dtype)
    yy_norm = (yy / max(out_h - 1, 1)) * 2.0 - 1.0
    xx_norm = (xx / max(out_w - 1, 1)) * 2.0 - 1.0

    mass = cand_maps.flatten(2).sum(dim=-1).clamp_min(1e-8)
    center_x = (cand_centers[..., 1] / max(out_w - 1, 1)) * 2.0 - 1.0
    center_y = (cand_centers[..., 0] / max(out_h - 1, 1)) * 2.0 - 1.0
    mean_x = (cand_maps * xx_norm.view(1, 1, 1, out_w)).flatten(2).sum(dim=-1) / mass
    mean_y = (cand_maps * yy_norm.view(1, 1, out_h, 1)).flatten(2).sum(dim=-1) / mass
    std_x = torch.sqrt(
        (cand_maps * (xx_norm.view(1, 1, 1, out_w) - mean_x[:, :, None, None]) ** 2)
        .flatten(2)
        .sum(dim=-1)
        .div(mass)
        .clamp_min(1e-12)
    )
    std_y = torch.sqrt(
        (cand_maps * (yy_norm.view(1, 1, out_h, 1) - mean_y[:, :, None, None]) ** 2)
        .flatten(2)
        .sum(dim=-1)
        .div(mass)
        .clamp_min(1e-12)
    )
    effective_area = (1.0 / cand_maps.square().flatten(2).sum(dim=-1).clamp_min(1e-8)) / float(max(out_h * out_w, 1))
    geom = torch.stack([center_x, center_y, effective_area, std_x, std_y], dim=-1)
    return geom * cand_valid.unsqueeze(-1).float()


def gather_candidate_maps(cand_maps: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _batch_size, _topk, out_h, out_w = cand_maps.shape
    gather_idx = indices.view(-1, 1, 1, 1).expand(-1, 1, out_h, out_w)
    return torch.gather(cand_maps, 1, gather_idx)


def gather_candidate_centers(cand_centers: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_idx = indices.view(-1, 1, 1).expand(-1, 1, 2)
    return torch.gather(cand_centers, 1, gather_idx).squeeze(1)


def grid_distance(out_h: int, out_w: int, center_y: torch.Tensor, center_x: torch.Tensor, device: torch.device):
    yy = torch.arange(out_h, device=device, dtype=torch.float32).view(out_h, 1)
    xx = torch.arange(out_w, device=device, dtype=torch.float32).view(1, out_w)
    return torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2 + 1e-12)


def mask_centroid(mask_hw: torch.Tensor):
    ys, xs = torch.where(mask_hw > 0.5)
    if ys.numel() == 0:
        return None
    return ys.float().mean(), xs.float().mean(), mask_hw.new_tensor(float(ys.numel()))


def project_centroid(center_y, center_x, in_h: int, in_w: int, out_h: int, out_w: int):
    proj_y = (center_y + 0.5) * float(out_h) / float(in_h) - 0.5
    proj_x = (center_x + 0.5) * float(out_w) / float(in_w) - 0.5
    return proj_y.clamp(0, out_h - 1), proj_x.clamp(0, out_w - 1)


def dilate01(mask_hw: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask_hw
    kernel = 2 * radius + 1
    return F.max_pool2d(mask_hw[None, None].float(), kernel_size=kernel, stride=1, padding=radius)[0, 0]


def build_sam_ignore(sam_masks: Sequence[torch.Tensor], out_h: int, out_w: int, device: torch.device) -> torch.Tensor:
    sam_ignore = torch.zeros((out_h, out_w), device=device)
    for raw_mask in sam_masks:
        mask_hw = raw_mask.to(device=device).float()
        soft = F.interpolate(mask_hw[None, None], size=(out_h, out_w), mode="area")[0, 0]
        sam_ignore = torch.maximum(sam_ignore, dilate01((soft > 1e-6).float(), radius=1))
    return sam_ignore.clamp(0, 1)


def build_evidence_supervision(target: torch.Tensor, sam3_masks: Sequence[Sequence[torch.Tensor]], out_h: int, out_w: int):
    pos_list: List[torch.Tensor] = []
    far_bg_list: List[torch.Tensor] = []
    sam_neg_list: List[torch.Tensor] = []
    center_list: List[torch.Tensor] = []
    area_ratio_list: List[torch.Tensor] = []
    valid_list: List[torch.Tensor] = []

    for batch_idx in range(target.shape[0]):
        gt_hw = (target[batch_idx].float() > 0).float()
        in_h, in_w = gt_hw.shape
        centroid = mask_centroid(gt_hw)
        if centroid is None:
            empty = gt_hw.new_zeros((out_h, out_w))
            pos_list.append(empty)
            far_bg_list.append(empty)
            sam_neg_list.append(empty)
            center_list.append(gt_hw.new_zeros(2))
            area_ratio_list.append(gt_hw.new_zeros(()))
            valid_list.append(torch.tensor(False, device=gt_hw.device))
            continue

        center_y, center_x, area = centroid
        area_ratio = area / float(max(in_h * in_w, 1))
        proj_y, proj_x = project_centroid(center_y, center_x, in_h, in_w, out_h, out_w)
        area_proj = area * float(out_h * out_w) / float(max(in_h * in_w, 1))
        radius = torch.maximum(torch.sqrt(area_proj.clamp_min(1e-6) / math.pi), gt_hw.new_tensor(1.0))
        dist = grid_distance(out_h, out_w, proj_y, proj_x, gt_hw.device)

        pos = (dist <= radius).float()
        gt_support = (F.interpolate(gt_hw[None, None], size=(out_h, out_w), mode="area")[0, 0] > 1e-6).float()
        sam_ignore = build_sam_ignore(sam3_masks[batch_idx], out_h, out_w, gt_hw.device)
        sam_neg = ((sam_ignore > 0.5) & (gt_support < 0.5) & (pos < 0.5)).float()
        far_bg = ((dist >= radius * 2.0) & (gt_support < 0.5) & (sam_ignore < 0.5) & (pos < 0.5)).float()

        pos_list.append(pos)
        far_bg_list.append(far_bg)
        sam_neg_list.append(sam_neg)
        center_list.append(torch.stack([proj_y, proj_x]))
        area_ratio_list.append(area_ratio)
        valid_list.append(torch.tensor(True, device=gt_hw.device))

    return {
        "pos": torch.stack(pos_list, dim=0).unsqueeze(1),
        "far_bg": torch.stack(far_bg_list, dim=0).unsqueeze(1),
        "sam_weak_neg": torch.stack(sam_neg_list, dim=0).unsqueeze(1),
        "center": torch.stack(center_list, dim=0),
        "area_ratio": torch.stack(area_ratio_list, dim=0),
        "valid": torch.stack(valid_list, dim=0),
    }


def region_mean_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for batch_idx in range(values.shape[0]):
        denom = mask[batch_idx].sum()
        if float(denom.item()) > 1e-6:
            losses.append((values[batch_idx] * mask[batch_idx]).sum() / denom.clamp_min(1.0))
    return torch.stack(losses).mean() if losses else values.sum() * 0.0


def compute_evidence_loss(logits: torch.Tensor, sup: Dict[str, torch.Tensor]) -> torch.Tensor:
    bce_pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    bce_neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    return (
        region_mean_loss(bce_pos, sup["pos"])
        + region_mean_loss(bce_neg, sup["far_bg"])
        + 0.5 * region_mean_loss(bce_neg, sup["sam_weak_neg"])
    )


def gather_candidate_map_value(cand_maps: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    batch_size, _topk, out_h, out_w = cand_maps.shape
    center_y = centers[:, 0].round().long().clamp(0, out_h - 1)
    center_x = centers[:, 1].round().long().clamp(0, out_w - 1)
    values = []
    for batch_idx in range(batch_size):
        values.append(cand_maps[batch_idx, :, center_y[batch_idx], center_x[batch_idx]])
    return torch.stack(values, dim=0)


def build_candidate_targets(candidates: Dict[str, torch.Tensor], sup: Dict[str, torch.Tensor], cfg: CandidateConfig):
    cand_maps = candidates["cand_maps"]
    cand_centers = candidates["cand_centers"]
    cand_valid = candidates["cand_valid"]
    gt_center = sup["center"]

    dist = torch.sqrt(((cand_centers - gt_center[:, None, :]) ** 2).sum(dim=-1).clamp_min(1e-12))
    center_values = gather_candidate_map_value(cand_maps, gt_center).masked_fill(~cand_valid, -1.0)
    target_idx = center_values.argmax(dim=1)
    target_dist = dist.gather(1, target_idx[:, None]).squeeze(1)
    target_valid = cand_valid.gather(1, target_idx[:, None]).squeeze(1)
    target_valid = target_valid & (target_dist <= float(cfg.gt_radius)) & sup["valid"].bool()
    target_valid = target_valid & (sup["area_ratio"] <= float(cfg.winner_max_area_ratio))
    return {"target_idx": target_idx, "target_valid": target_valid, "candidate_gt_dist": dist}


def compute_winner_loss(scores: torch.Tensor, candidate_sup: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_valid = candidate_sup["target_valid"]
    if int(target_valid.sum().item()) == 0:
        return scores.sum() * 0.0
    return F.cross_entropy(scores[target_valid], candidate_sup["target_idx"][target_valid])


def compute_localization_loss(out: Dict[str, torch.Tensor], target: torch.Tensor, sam3_masks) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    sup = build_evidence_supervision(
        target=target,
        sam3_masks=sam3_masks,
        out_h=out["evidence_logit"].shape[-2],
        out_w=out["evidence_logit"].shape[-1],
    )
    candidate_cfg = out.get("candidate_cfg", CandidateConfig())
    candidate_sup = build_candidate_targets(out, sup, candidate_cfg)
    evidence_loss = compute_evidence_loss(out["evidence_logit"], sup)
    winner_loss = compute_winner_loss(out["scores"], candidate_sup)
    total = evidence_loss + 1.2 * winner_loss
    return total, {
        "loc_loss": total,
        "evidence_loss": evidence_loss,
        "winner_loss": winner_loss,
    }


def normalize_guidance_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    denom = flat.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    return x / denom.view(x.shape[0], x.shape[1], 1, 1)


def winner_token_scale(
    token_alpha: torch.Tensor,
    winner_idx: torch.Tensor,
    winner_top1_map: torch.Tensor,
    token_valid: torch.Tensor,
    prob_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_f = token_valid.float()
    gather_idx = winner_idx.view(-1, 1, 1).expand(-1, 1, token_alpha.shape[-1])
    token_weight = torch.gather(token_alpha, 1, gather_idx).squeeze(1)
    token_weight = token_weight * valid_f

    valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    uniform_weight = valid_f / valid_count
    has_valid = (valid_f.sum(dim=-1, keepdim=True) > 0).float()
    weight_sum = token_weight.sum(dim=-1, keepdim=True)
    token_weight = torch.where(
        weight_sum > float(prob_eps),
        token_weight / weight_sum.clamp_min(float(prob_eps)),
        uniform_weight,
    )
    token_weight = token_weight * has_valid + uniform_weight * (1.0 - has_valid)

    token_scale = token_weight * valid_count
    spatial_scale = normalize_guidance_map(winner_top1_map).flatten(2).mean(dim=-1)
    token_scale = 1.0 + (token_scale - 1.0) * spatial_scale
    token_scale = torch.where(token_valid, token_scale, torch.ones_like(token_scale))
    return token_weight, token_scale


def apply_text_guidance(
    l_feats: torch.Tensor,
    token_scale: torch.Tensor,
    norm: nn.LayerNorm,
) -> torch.Tensor:
    guided = l_feats * token_scale.unsqueeze(1)
    guided = norm(guided.transpose(1, 2)).transpose(1, 2).contiguous()
    return guided


class EvidenceHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            self._conv_block(in_channels, hidden_channels),
            self._conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _group_count(channels: int) -> int:
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return groups

    @classmethod
    def _conv_block(cls, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(cls._group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CandidateConditionedTokenRouter(nn.Module):
    def __init__(self, candidate_dim: int, text_dim: int, query_dim: int, route_dim: int, hidden_dim: int):
        super().__init__()
        self.cand_route = nn.Sequential(
            nn.Linear(candidate_dim, route_dim),
            nn.LayerNorm(route_dim),
            nn.GELU(),
        )
        self.text_route = nn.Sequential(
            nn.Linear(text_dim, route_dim),
            nn.LayerNorm(route_dim),
            nn.GELU(),
        )
        self.text_query = nn.Sequential(
            nn.Linear(text_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(route_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cand_feat: torch.Tensor,
        text_tokens: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _bsz, num_candidates, _ = cand_feat.shape
        num_tokens = text_tokens.shape[1]

        cand_route = self.cand_route(cand_feat).unsqueeze(2).expand(-1, -1, num_tokens, -1)
        text_route = self.text_route(text_tokens).unsqueeze(1).expand(-1, num_candidates, -1, -1)
        router_input = torch.cat([text_route, cand_route], dim=-1)
        token_logits = self.score_mlp(router_input).squeeze(-1)
        valid = token_valid.unsqueeze(1).expand(-1, num_candidates, -1)
        token_alpha = masked_softmax(token_logits, valid, dim=-1)
        query_tokens = self.text_query(text_tokens)
        candidate_query = torch.einsum("bkn,bnq->bkq", token_alpha, query_tokens)
        return candidate_query, token_alpha, token_logits


class CandidateRanker(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        query_dim: int,
        router_hidden_dim: int,
        hidden_dim: int,
        lambda_geo: float,
        text_dim: int = 768,
    ):
        super().__init__()
        self.geom_dim = 5
        candidate_dim = visual_dim + self.geom_dim
        self.lambda_geo = float(lambda_geo)
        self.router = CandidateConditionedTokenRouter(
            candidate_dim=candidate_dim,
            text_dim=text_dim,
            query_dim=query_dim,
            route_dim=router_hidden_dim,
            hidden_dim=router_hidden_dim,
        )
        self.visual_content = nn.Sequential(
            nn.Linear(visual_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.query_content = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.geom_dim + query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cand_visual: torch.Tensor,
        geom: torch.Tensor,
        text_tokens: torch.Tensor,
        token_valid: torch.Tensor,
        cand_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cand_feat = torch.cat([cand_visual, geom], dim=-1)
        query, token_alpha, token_logits = self.router(cand_feat, text_tokens, token_valid)

        visual_score = F.normalize(self.visual_content(cand_visual), p=2, dim=-1)
        query_score = F.normalize(self.query_content(query), p=2, dim=-1)
        content_score = (visual_score * query_score).sum(dim=-1)

        geo_score = self.geo_mlp(torch.cat([geom, query], dim=-1)).squeeze(-1)
        scores = content_score + self.lambda_geo * geo_score
        scores = scores.masked_fill(~cand_valid, -1e4)
        cand_probs = masked_softmax(scores, cand_valid, dim=-1)

        return {
            "scores": scores,
            "cand_probs": cand_probs,
            "content_score": content_score,
            "geo_score": geo_score,
            "token_alpha": token_alpha,
            "token_logits": token_logits,
        }


class CandidateGenerator:
    def __init__(self, cfg: CandidateConfig):
        self.cfg = cfg

    def _empty_outputs(self, prob_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _channels, out_h, out_w = prob_map.shape
        device = prob_map.device
        dtype = prob_map.dtype
        topk = int(self.cfg.topk)
        return {
            "cand_maps": torch.zeros((batch_size, topk, out_h, out_w), device=device, dtype=dtype),
            "cand_centers": torch.zeros((batch_size, topk, 2), device=device, dtype=dtype),
            "cand_valid": torch.zeros((batch_size, topk), device=device, dtype=torch.bool),
            "peak_values": torch.zeros((batch_size, topk), device=device, dtype=dtype),
            "raw_masses": torch.zeros((batch_size, topk), device=device, dtype=dtype),
            "peak_rel": torch.zeros((batch_size, topk), device=device, dtype=dtype),
        }

    def _fill_candidate(
        self,
        outputs: Dict[str, torch.Tensor],
        prob_map: torch.Tensor,
        batch_idx: int,
        cand_idx: int,
        center_y: float,
        center_x: float,
        yy: torch.Tensor,
        xx: torch.Tensor,
    ) -> None:
        score_map = prob_map[batch_idx, 0]
        out_h, out_w = score_map.shape
        center_y = max(0, min(out_h - 1, int(round(float(center_y)))))
        center_x = max(0, min(out_w - 1, int(round(float(center_x)))))

        outputs["cand_centers"][batch_idx, cand_idx] = torch.tensor(
            [float(center_y), float(center_x)],
            device=prob_map.device,
            dtype=prob_map.dtype,
        )
        outputs["peak_values"][batch_idx, cand_idx] = score_map[center_y, center_x]

        sigma = max(float(self.cfg.cand_sigma), 1e-6)
        dist2 = (yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2
        raw = score_map * torch.exp(-0.5 * dist2 / (sigma ** 2))
        mass = raw.sum()
        if float(mass.item()) <= float(self.cfg.min_mass):
            return

        outputs["cand_maps"][batch_idx, cand_idx] = raw / mass.clamp_min(float(self.cfg.min_mass))
        outputs["raw_masses"][batch_idx, cand_idx] = mass
        outputs["cand_valid"][batch_idx, cand_idx] = True

    def __call__(self, prob_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _channels, out_h, out_w = prob_map.shape
        peak_kernel = max(int(self.cfg.peak_kernel), 1)
        if peak_kernel % 2 == 0:
            peak_kernel += 1

        pooled = F.max_pool2d(prob_map, kernel_size=peak_kernel, stride=1, padding=peak_kernel // 2)
        peak_mask = (prob_map >= pooled - 1e-12) & (prob_map >= float(self.cfg.min_peak_thr))

        outputs = self._empty_outputs(prob_map)
        yy, xx = spatial_grid(out_h, out_w, prob_map.device, prob_map.dtype)

        for batch_idx in range(batch_size):
            score_map = prob_map[batch_idx, 0]
            work = torch.where(
                peak_mask[batch_idx, 0],
                score_map,
                score_map.new_full((out_h, out_w), float("-inf")),
            ).clone()
            if not torch.isfinite(work).any():
                work = score_map.clone()

            for cand_idx in range(int(self.cfg.topk)):
                flat_idx = int(torch.argmax(work).item())
                value = work.flatten()[flat_idx]
                if not torch.isfinite(value):
                    break

                center_y = flat_idx // out_w
                center_x = flat_idx % out_w
                self._fill_candidate(outputs, prob_map, batch_idx, cand_idx, center_y, center_x, yy, xx)

                dist2 = (yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2
                suppress = dist2 <= float(self.cfg.nms_radius) ** 2
                work = work.masked_fill(suppress, float("-inf"))

        top_peak = outputs["peak_values"].max(dim=1, keepdim=True).values.clamp_min(float(self.cfg.prob_eps))
        outputs["peak_rel"] = outputs["peak_values"] / top_peak
        return outputs


class LocalizationGuidanceModule(nn.Module):
    def __init__(self, args: SimpleNamespace, in_channels: int, cand_cfg: CandidateConfig):
        super().__init__()
        self.candidate_cfg = cand_cfg
        self.evidence_head = EvidenceHead(
            in_channels=in_channels,
            hidden_channels=int(args.evidence_hidden_dim),
        )
        self.ranker = CandidateRanker(
            visual_dim=in_channels,
            query_dim=int(args.winner_query_dim),
            router_hidden_dim=int(args.winner_router_hidden_dim),
            hidden_dim=int(args.winner_hidden_dim),
            lambda_geo=float(cand_cfg.lambda_geo),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        text_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        l_mask: torch.Tensor,
        generator: Optional[CandidateGenerator] = None,
        candidate_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        generator = generator or CandidateGenerator(self.candidate_cfg)
        evidence_logit = self.evidence_head(feature_map)
        prob_map = torch.sigmoid(evidence_logit)

        candidate_source = prob_map
        if candidate_prob is not None:
            candidate_source = candidate_prob.to(device=feature_map.device, dtype=feature_map.dtype)
            if candidate_source.shape[-2:] != prob_map.shape[-2:]:
                candidate_source = F.interpolate(candidate_source, size=prob_map.shape[-2:], mode="bilinear", align_corners=False)
            candidate_source = candidate_source.clamp(0.0, 1.0)

        candidates = generator(candidate_source)
        cand_maps = candidates["cand_maps"]
        cand_valid = candidates["cand_valid"]

        cand_visual = candidate_weighted_pool(feature_map, cand_maps)
        geom = candidate_geometry(cand_maps, candidates["cand_centers"], cand_valid)
        token_valid = token_valid_mask(input_ids, l_mask, self.candidate_cfg)
        ranker_out = self.ranker(cand_visual, geom, text_tokens, token_valid, cand_valid)

        winner_prior = torch.einsum("bk,bkhw->bhw", ranker_out["cand_probs"], cand_maps).unsqueeze(1)
        winner_prior = winner_prior / winner_prior.flatten(2).sum(dim=-1).view(feature_map.shape[0], 1, 1, 1).clamp_min(
            float(self.candidate_cfg.prob_eps)
        )

        pred_idx = ranker_out["scores"].argmax(dim=1)
        winner_top1_map = gather_candidate_maps(cand_maps, pred_idx)
        winner_top1_center = gather_candidate_centers(candidates["cand_centers"], pred_idx)
        cand_up = normalize_guidance_map(winner_top1_map)

        return {
            "evidence_logit": evidence_logit,
            "prob_map": prob_map,
            "candidate_source": candidate_source,
            "candidate_cfg": self.candidate_cfg,
            "winner_prior": winner_prior,
            "winner_top1_map": winner_top1_map,
            "winner_top1_center": winner_top1_center,
            "cand_up": cand_up,
            "token_valid": token_valid,
            **candidates,
            **ranker_out,
        }


class LocalizationGuidanceAdapter(nn.Module):
    def __init__(
        self,
        module: LocalizationGuidanceModule,
        generator: CandidateGenerator,
        alpha: float,
    ):
        super().__init__()
        self.module = module
        self.generator = generator
        self.alpha = float(alpha)

        text_dim = int(self.module.ranker.router.text_route[0].in_features)
        self.text_norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        feature_map: torch.Tensor,
        input_ids: torch.Tensor,
        l_mask: torch.Tensor,
        updated_l_feats: torch.Tensor,
        guide_text: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        text_tokens = updated_l_feats.permute(0, 2, 1).contiguous()
        guidance = self.module(
            feature_map=feature_map,
            text_tokens=text_tokens,
            input_ids=input_ids,
            l_mask=l_mask,
            generator=self.generator,
        )
        enhanced = feature_map * (1.0 + self.alpha * guidance["cand_up"])

        if not guide_text:
            return enhanced, updated_l_feats, guidance

        winner_idx = guidance["scores"].argmax(dim=1)

        token_weight, token_scale = winner_token_scale(
            token_alpha=guidance["token_alpha"],
            winner_idx=winner_idx,
            winner_top1_map=guidance["winner_top1_map"],
            token_valid=guidance["token_valid"],
            prob_eps=float(self.module.candidate_cfg.prob_eps),
        )
        l_feats_guided = apply_text_guidance(updated_l_feats, token_scale, self.text_norm)
        guidance["token_weight"] = token_weight
        guidance["token_scale"] = token_scale
        return enhanced, l_feats_guided, guidance


def build_localization_guidance(alpha: float = 0.5) -> Tuple[LocalizationGuidanceAdapter, Dict[str, object]]:
    module_args = SimpleNamespace(
        evidence_hidden_dim=EVIDENCE_HIDDEN_DIM,
        winner_query_dim=WINNER_QUERY_DIM,
        winner_router_hidden_dim=WINNER_ROUTER_HIDDEN_DIM,
        winner_hidden_dim=WINNER_HIDDEN_DIM,
    )
    cand_cfg = CandidateConfig()
    module = LocalizationGuidanceModule(module_args, in_channels=LOCALIZATION_IN_CHANNELS, cand_cfg=cand_cfg)
    adapter = LocalizationGuidanceAdapter(module=module, generator=CandidateGenerator(cand_cfg), alpha=float(alpha))
    return adapter, {
        "feature_key": LOCALIZATION_FEATURE_KEY,
        "in_channels": LOCALIZATION_IN_CHANNELS,
        "candidate_cfg": cand_cfg.__dict__,
        "args": module_args.__dict__,
    }


def load_localization_guidance(
    ckpt_path: str,
    alpha: float = 0.5,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[LocalizationGuidanceAdapter, Dict[str, object]]:
    state = torch.load(ckpt_path, map_location=map_location)
    adapter, payload = build_localization_guidance(alpha=alpha)
    if any(key.startswith("module.") or key.startswith("text_norm.") for key in state.keys()):
        adapter.load_state_dict(state, strict=True)
    else:
        adapter.module.load_state_dict(state, strict=True)
    return adapter, payload
