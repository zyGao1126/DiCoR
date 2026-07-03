import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .mmcv_custom import load_checkpoint
from typing import Optional
from .text_aware_multiscale_enhancement import TMEM
from .visual_multiscale_enhancement import VMSF, LVMSF
from typing import List
from .backbone_util import *

# Orthogonality (A)
def get_ortho_loss(q):
    if q.shape[1] <= 1: return 0.0
    q_norm = F.normalize(q, p=2, dim=-1)
    gram = torch.bmm(q_norm, q_norm.transpose(1, 2))
    I = torch.eye(q.shape[1], device=q.device).unsqueeze(0)
    return ((gram - I) ** 2).mean()

def get_div_loss(w):
    if w.shape[1] <= 1: return 0.0
    gram = torch.bmm(w, w.transpose(1, 2))
    off_diag = gram.sum(dim=(1, 2)) - gram.diagonal(dim1=1, dim2=2).sum(dim=1)
    return off_diag.mean()


def masked_softmax_1d(scores_bt: torch.Tensor, mask_bt: torch.Tensor, eps: float = 1e-6):
    if mask_bt.dtype != scores_bt.dtype:
        mask_bt = mask_bt.to(dtype=scores_bt.dtype)
    scores_bt = scores_bt.masked_fill(mask_bt <= 0, -1e9)
    probs = F.softmax(scores_bt, dim=-1)
    probs = probs * mask_bt
    den = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / den


def entropy_1d(prob_bt: torch.Tensor, mask_bt: torch.Tensor, eps: float = 1e-6):
    if mask_bt.dtype != prob_bt.dtype:
        mask_bt = mask_bt.to(dtype=prob_bt.dtype)
    p = prob_bt.clamp_min(eps) * mask_bt
    ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
    denom = mask_bt.sum(dim=-1).clamp_min(1.0)
    return (ent / denom).mean()

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN feed-forward network
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class MultiModalSwinTransformerV2(nn.Module):
    def __init__(self, **swin_kwargs):
        super().__init__()

        self.embed_dim      = swin_kwargs.get('embed_dim', 96)
        self.depths         = swin_kwargs.get('depths', [2, 2, 6, 2])
        self.swin_num_heads = swin_kwargs.get('swin_num_heads', [4, 8, 16, 32])
        self.window_size    = swin_kwargs.get('window_size', 7)
        self.out_indices    = tuple(swin_kwargs.get('out_indices', (0, 1, 2, 3)))
        self.drop_path_rate = swin_kwargs.get('drop_path_rate', 0.3)
        self.patch_norm     = swin_kwargs.get('patch_norm', True)
        self.num_tmem       = swin_kwargs.get('num_tmem', 3)
        self.num_heads_fusion = swin_kwargs.get('num_heads_fusion', 1)
        self.use_checkpoint = swin_kwargs.get('use_checkpoint', False)
        self.use_lvmsf = swin_kwargs.get('use_lvmsf', False)

        self.text_feat_dim  = 768
        self.num_layers = len(self.depths)        

        # ---- Patch embed ----
        self.patch_embed = PatchEmbed(
            patch_size=4, in_chans=3, embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=0.0)

        # ---- Build stage layers ----
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            layer_kwargs = {
                'dim': int(self.embed_dim * 2 ** i),
                'swin_depth': self.depths[i],
                'swin_num_heads': self.swin_num_heads[i],
                'num_heads_fusion': self.num_heads_fusion,
                'window_size': self.window_size,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'qk_scale': None,
                'drop': 0.0,
                'drop_path': dpr[sum(self.depths[:i]):sum(self.depths[:i+1])],
                'fusion_drop': 0.0,
                'norm_layer': nn.LayerNorm,
                'downsample': PatchMerging if (i < self.num_layers - 1) else None,
                'use_checkpoint': self.use_checkpoint,
                'text_feat_dim': self.text_feat_dim,
            }
            self.layers.append(CascadedMMBasicLayer(**layer_kwargs))

        num_features = [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
        for i_layer in self.out_indices:
            self.add_module(f'norm{i_layer}', nn.LayerNorm(num_features[i_layer]))
        if self.use_lvmsf:
            self.VMSF = LVMSF(num_features, num_blocks=self.num_tmem, text_dim=self.text_feat_dim)
        else:
            self.VMSF = VMSF(num_features, num_blocks=self.num_tmem)
        self.localization_guidance = None

    def set_localization_guidance(self, module):
        self.localization_guidance = module

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=None)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l_feats, l_mask, input_ids=None):
        # image embed
        x = self.patch_embed(x)
        B, _, Wh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        l_feats_cur = l_feats
        for i, layer in enumerate(self.layers):
            layer_kwargs = {
                'x': x, 'Wh': Wh, 'Ww': Ww,
                'l_feat': l_feats_cur, 'l_mask': l_mask, 'stage_idx': i,
            }
            H, W = Wh, Ww

            x_result = layer(**layer_kwargs)
            x_out, Wh, Ww, x = x_result['x_out'], x_result['Wh'], x_result['Ww'], x_result['x']
            l_feats_cur = x_result.get('l_feat', l_feats_cur)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)  # output of a Block has shape (B, H*W, dim)
                out = x_out.view(-1, H, W, x_out.shape[-1]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        features_pre_vmsf = [o for o in outs]
        localization_out = None
        guided_l_feats = l_feats_cur
        if self.localization_guidance is not None:
            if input_ids is None:
                raise ValueError("Localization guidance requires raw input_ids during backbone forward.")
            outs = list(outs)
            outs[2], guided_l_feats, localization_out = self.localization_guidance(
                feature_map=outs[2],
                input_ids=input_ids,
                l_mask=l_mask,
                updated_l_feats=l_feats_cur,
                guide_text=self.use_lvmsf,
            )
        
        if self.use_lvmsf:
            outs, l_feats_cur = self.VMSF(outs, l_feats=guided_l_feats, l_mask=l_mask)
        else:
            outs = self.VMSF(outs)
            l_feats_cur = guided_l_feats
        result = {
            'features': outs,
            'features_pre_vmsf': features_pre_vmsf,
            'l_feats': l_feats_cur,
        }
        if localization_out is not None:
            result['localization_guidance'] = localization_out

        return result

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MultiModalSwinTransformerV2, self).train(mode)
        # self._freeze_stages()

class CascadedMMBasicLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs.get("dim") # int(embed_dim * 2 ** i)
        self.swin_depth = kwargs.get("swin_depth") # [2, 2, 6, 2][i]
        self.swin_num_heads = kwargs.get("swin_num_heads", [3, 6, 12, 24])
        self.num_heads_fusion = kwargs.get("num_heads_fusion", 1)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)   
        self.window_size = kwargs.get("window_size", 7)
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.qk_scale = kwargs.get("qk_scale", None)
        self.drop = kwargs.get("drop", 0.0)
        self.drop_path = kwargs.get("drop_path")
        self.fusion_drop = kwargs.get("fusion_drop", 0.0)          
        self.use_checkpoint = kwargs.get("use_checkpoint", False)
        self.text_feat_dim = kwargs.get("text_feat_dim", 768)
        downsample = kwargs.get("downsample", None)
        self.downsample = downsample(dim=self.dim, norm_layer=nn.LayerNorm) if downsample is not None else None

        # Swin blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=self.dim,
                num_heads=self.swin_num_heads,
                window_size=self.window_size,
                shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop,
                attn_drop=0.0,
                drop_path=self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path,
                norm_layer=nn.LayerNorm)
            for i in range(self.swin_depth)])

        # === Fusion ===
        self.fusion_global = self._build_fusion_block()

        self.res_gate = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.Tanh()
        )

    def _build_fusion_block(self):
        common_args = (self.dim, self.dim, self.text_feat_dim, self.dim, self.dim)
        common_kwargs = dict(
            num_heads=self.num_heads_fusion,
            dropout=self.fusion_drop,
        )
        return PWAM(*common_args, **common_kwargs)

    @staticmethod
    def _forward_fusion_block(module, x, l_feats, mask, H, W, **extra):
        return module(x, l_feats, mask)

    def create_attention_mask(self, H, W, device):
        """创建Swin Transformer的attention mask"""
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask

    def _fuse_single_global(self, x_swin, l_feats, l_mask, H, W):
        x_global, aux = self._forward_fusion_block(
            self.fusion_global, x_swin, l_feats, l_mask, H, W
        )

        x_out_gate = self.res_gate(x_global) * x_global
        l_next = aux.get('l_new', l_feats)
        return x_global, x_swin + x_out_gate, l_next

    def forward(self, **kwargs):
        x = kwargs['x']
        H, W = kwargs['Wh'], kwargs['Ww']
        l_feats = kwargs['l_feat']
        l_mask = kwargs['l_mask']

        attn_mask = self.create_attention_mask(H, W, x.device)
        x_swin = self.apply_swin_blocks(x, H, W, attn_mask)
        _, HW, _ = x_swin.shape
        assert HW == H * W

        x_out, x_next, l_next = self._fuse_single_global(
            x_swin, l_feats, l_mask, H, W
        )

        if self.downsample is not None:
            x_down = self.downsample(x_next, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            result = {'x_out': x_out, 'Wh': Wh, 'Ww': Ww, 'x': x_down, 'l_feat': l_next}
        else:
            Wh, Ww = H, W
            result = {'x_out': x_out, 'Wh': Wh, 'Ww': Ww, 'x': x_next, 'l_feat': l_next}
        return result

    def apply_swin_blocks(self, x, H, W, attn_mask):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        return x

class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels,
                 num_heads=0, dropout=0.0, enable_text_update=False, l2v_attn_dim=256, l2v_heads=4):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

        self.enable_text_update = enable_text_update
        if self.enable_text_update:
            attn_dim = min(l2v_attn_dim, dim)
            self.l_q = nn.Linear(l_in_channels, attn_dim, bias=False)
            self.v_kv = nn.Linear(dim, attn_dim, bias=False)
            self.l2v_attn = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=l2v_heads,
                dropout=dropout,
                batch_first=False,
            )
            self.l_out = nn.Sequential(
                nn.Linear(attn_dim, l_in_channels, bias=False),
                nn.LayerNorm(l_in_channels)
            )
            self.l_gate = nn.Sequential(
                nn.Linear(l_in_channels, l_in_channels, bias=False),
                nn.Tanh()
            )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        if not self.enable_text_update:
            return mm, {}

        l_tok = l.permute(0, 2, 1).contiguous()
        l_valid = (l_mask > 0)
        query = self.l_q(l_tok).transpose(0, 1)
        kv = self.v_kv(x).transpose(0, 1)
        delta, _ = self.l2v_attn(query, kv, kv, need_weights=False)
        delta = delta.transpose(0, 1)
        delta = self.l_out(delta)
        gate = self.l_gate(delta)
        l_new = l_tok + gate * delta
        l_new = torch.where(l_valid, l_new, l_tok)
        l_new = l_new.permute(0, 2, 1).contiguous()

        aux = {'l_new': l_new}

        return mm, aux

class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1).float()  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key_full = self.f_key(l)  # (B, key_channels, N_l)
        value_full = self.f_value(l)  # (B, self.value_channels, N_l)
        key_full = key_full * l_mask  # (B, key_channels, N_l)
        value_full = value_full * l_mask  # (B, self.value_channels, N_l)
        n_l = value_full.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key_full.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value_full.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        pad_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product
        sim_map = sim_map + (1e4 * pad_mask - 1e4)  # assign a very small number to padding positions

        attn1 = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(attn1, value.permute(0, 1, 3, 2))

        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)
        return out
