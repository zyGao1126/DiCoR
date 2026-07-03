import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class VisualFusionBlock(nn.Module):
    """
    纯视觉的融合块，使用自注意力机制在聚合后的多尺度特征上进行信息交互。
    取代了原有的 TMEMBlock。
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Self-Attention + Residual Connection
        x_norm1 = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x_seq = x_seq + attn_out

        # FFN + Residual Connection
        x_norm2 = self.norm2(x_seq)
        ffn_out = self.ffn(x_norm2)
        x_seq = x_seq + ffn_out

        out = x_seq.transpose(1, 2).reshape(B, C, H, W)
        return out

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class ScaleAwareGate(nn.Module):
    def __init__(self, inp, oup):
        super(ScaleAwareGate, self).__init__()
        self.local_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(oup)
        self.global_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(oup)
        self.global_act = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(oup)
        # h_sigmoid is a custom hard sigmoid, which is fine
        self.act = h_sigmoid() 
    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.bn1(self.local_embedding(x_l))
        global_feat = self.bn2(self.global_embedding(x_g))
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        global_act = self.bn3(self.global_act(x_g))
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * sig_act + global_feat
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class VMSF(nn.Module):
    def __init__(self, in_channels_list, num_blocks=1, downsample_stride=1):
        super().__init__()
        self.channels = in_channels_list
        self.total_in_dim = sum(in_channels_list)
        
        self.hidden_dim = self.total_in_dim // 4
        
        self.pool = PyramidPoolAgg(stride=downsample_stride)
        self.down_channel = nn.Conv2d(self.total_in_dim, self.hidden_dim, 1)

        self.blocks = nn.ModuleList([
            VisualFusionBlock(self.hidden_dim) for _ in range(num_blocks)
        ])
        
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.up_channel = nn.Conv2d(self.hidden_dim, self.total_in_dim, 1)

        self.fusion_gates = nn.ModuleList([
            ScaleAwareGate(channels, channels) for channels in self.channels
        ])

    def forward(self, multi_scale_inputs):
        # multi_scale_inputs: a list of feature maps [c1, c2, c3, c4]
        
        aggregated_feat = self.pool(multi_scale_inputs)
        
        aggregated_feat = self.down_channel(aggregated_feat)
        for block in self.blocks:
            aggregated_feat = block(aggregated_feat)
        aggregated_feat = self.bn(aggregated_feat)
        
        processed_multi_scale_feat = self.up_channel(aggregated_feat)

        split_processed_feats = processed_multi_scale_feat.split(self.channels, dim=1)
        
        results = []
        for i in range(len(self.channels)):
            original_feat = multi_scale_inputs[i]
            processed_feat = split_processed_feats[i] 
            fused_output = self.fusion_gates[i](original_feat, processed_feat)
            results.append(fused_output)
            
        return tuple(results)


class _LVMSFBlock(nn.Module):
    """One interaction step: l2v -> v2l -> ms_attn."""

    def __init__(self,
                 hidden_dim: int,
                 text_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.l2v = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )
        self.v2l = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=text_dim,
            vdim=text_dim,
        )

        self.lnL1 = nn.LayerNorm(text_dim)
        self.lnL2 = nn.LayerNorm(text_dim)
        self.lnV1 = nn.LayerNorm(hidden_dim)
        self.lnV2 = nn.LayerNorm(hidden_dim)
        self.ffnL = FeedForward(text_dim, int(text_dim * 4), dropout=dropout)
        self.ffnV = FeedForward(hidden_dim, int(hidden_dim * 4), dropout=dropout)

        self.ms_attn = VisualFusionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self,
                v_feat: torch.Tensor,
                l_seq: torch.Tensor,
                l_key_padding: torch.Tensor):
        B, C, H, W = v_feat.shape
        v_seq = v_feat.flatten(2).transpose(1, 2).contiguous()

        delta_l, _ = self.l2v(l_seq, v_seq, v_seq, need_weights=False)
        l_seq = self.lnL1(l_seq + delta_l)
        l_seq = self.lnL2(l_seq + self.ffnL(l_seq))

        delta_v, _ = self.v2l(v_seq, l_seq, l_seq, key_padding_mask=l_key_padding, need_weights=False)
        v_seq = self.lnV1(v_seq + delta_v)
        v_seq = self.lnV2(v_seq + self.ffnV(v_seq))

        v_feat = v_seq.transpose(1, 2).reshape(B, C, H, W).contiguous()
        v_feat = self.ms_attn(v_feat).contiguous()
        return v_feat, l_seq


class LVMSF(nn.Module):
    """Language-guided VMSF with per-block l2v->v2l->ms_attn interaction."""

    def __init__(self,
                 in_channels_list,
                 num_blocks=1,
                 downsample_stride=1,
                 text_dim: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.channels = in_channels_list
        self.total_in_dim = sum(in_channels_list)
        self.hidden_dim = self.total_in_dim // 4
        self.text_dim = text_dim

        self.pool = PyramidPoolAgg(stride=downsample_stride)
        self.down_channel = nn.Conv2d(self.total_in_dim, self.hidden_dim, 1)

        self.blocks = nn.ModuleList([
            _LVMSFBlock(
                hidden_dim=self.hidden_dim,
                text_dim=self.text_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_blocks)
        ])

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.up_channel = nn.Conv2d(self.hidden_dim, self.total_in_dim, 1)

        self.fusion_gates = nn.ModuleList([
            ScaleAwareGate(channels, channels) for channels in self.channels
        ])

    def forward(self,
                multi_scale_inputs: List[torch.Tensor],
                l_feats: Optional[torch.Tensor] = None,
                l_mask: Optional[torch.Tensor] = None):
        aggregated_feat = self.pool(multi_scale_inputs)
        aggregated_feat = self.down_channel(aggregated_feat)

        assert l_feats is not None and l_mask is not None, "LVMSF requires l_feats and l_mask."
        l_seq = l_feats.permute(0, 2, 1).contiguous()
        l_key_padding = (l_mask.squeeze(-1) == 0)
        for block in self.blocks:
            aggregated_feat, l_seq = block(aggregated_feat, l_seq, l_key_padding)

        aggregated_feat = self.bn(aggregated_feat).contiguous()
        processed_multi_scale_feat = self.up_channel(aggregated_feat).contiguous()
        split_processed_feats = processed_multi_scale_feat.split(self.channels, dim=1)

        results = []
        for i in range(len(self.channels)):
            original_feat = multi_scale_inputs[i]
            processed_feat = split_processed_feats[i].contiguous()
            fused_output = self.fusion_gates[i](original_feat, processed_feat)
            results.append(fused_output)

        l_feats_out = l_seq.permute(0, 2, 1).contiguous()
        return tuple(results), l_feats_out

class TextGuideFusionGate(nn.Module):
    """Per-scale TCSA-style gate with optional position guide map."""

    def __init__(self,
                 channels: int,
                 text_dim: int = 768,
                 reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.text_proj_c = nn.Linear(text_dim, channels, bias=False)

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.text_proj_s = nn.Linear(text_dim, 1, bias=False)

    @staticmethod
    def pool_text(l_feats: torch.Tensor, l_mask: torch.Tensor) -> torch.Tensor:
        m = l_mask.permute(0, 2, 1).float()
        den = m.sum(dim=2).clamp_min(1e-6)
        num = (l_feats * m).sum(dim=2)
        return num / den

    def forward(self,
                orig: torch.Tensor,
                proc: torch.Tensor,
                text_vec: torch.Tensor) -> torch.Tensor:
        B, C, H, W = orig.shape
        if proc.shape[-2:] != (H, W):
            proc = F.interpolate(proc, size=(H, W), mode="bilinear", align_corners=False)

        delta = proc - orig

        avg_pool = orig.mean(dim=(2, 3))
        ch_att = torch.sigmoid(self.channel_mlp(avg_pool) + self.text_proj_c(text_vec)).view(B, C, 1, 1)

        delta_c = delta * ch_att
        avg = delta_c.mean(dim=1, keepdim=True)
        mx, _ = delta_c.max(dim=1, keepdim=True)
        spatial_feat = torch.cat([avg, mx], dim=1)
        sp_att = torch.sigmoid(self.spatial_conv(spatial_feat) + self.text_proj_s(text_vec).view(B, 1, 1, 1))

        alpha = ch_att * sp_att

        return orig + alpha * delta


class TextGuidedVMSF(nn.Module):
    """VMSF with text-conditioned gates."""

    def __init__(self,
                 in_channels_list: List[int],
                 num_blocks: int = 1,
                 downsample_stride: int = 1,
                 text_dim: int = 768,
                 reduction: int = 4):
        super().__init__()
        self.channels = in_channels_list
        self.total_in_dim = sum(in_channels_list)
        self.hidden_dim = self.total_in_dim // 4

        self.pool = PyramidPoolAgg(stride=downsample_stride)
        self.down_channel = nn.Conv2d(self.total_in_dim, self.hidden_dim, 1)

        self.blocks = nn.ModuleList([VisualFusionBlock(self.hidden_dim) for _ in range(num_blocks)])
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.up_channel = nn.Conv2d(self.hidden_dim, self.total_in_dim, 1)

        self.fusion_gates = nn.ModuleList([
            TextGuideFusionGate(
                channels=c,
                text_dim=text_dim,
                reduction=reduction,
            ) for c in self.channels
        ])

    def forward(self,
                feats: List[torch.Tensor],
                l_feats: Optional[torch.Tensor] = None,
                l_mask: Optional[torch.Tensor] = None):
        text_vec = TextGuideFusionGate.pool_text(l_feats, l_mask)
        agg = self.pool(feats)
        agg = self.down_channel(agg)
        for blk in self.blocks:
            agg = blk(agg)
        agg = self.bn(agg)
        proc = self.up_channel(agg)
        proc_splits = proc.split(self.channels, dim=1)

        outs = []
        for idx, (orig, p) in enumerate(zip(feats, proc_splits)):
            outs.append(self.fusion_gates[idx](orig, p, text_vec=text_vec))

        return tuple(outs)
