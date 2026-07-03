import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class DensePromptHead(nn.Module):
    def __init__(self, in_dim, mid_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim), nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, 1)  # logits
        )

    def forward(self, feat_4d):  # (B,C,H,W)
        return self.proj(feat_4d)  # (B,1,H,W)

class QueryToSoftMask(nn.Module):
    """
    将 learnable queries 与 token 级 l_feats 对齐，生成 soft mask (B,T,1)。
    """
    def __init__(self, dim: int, tau: float = 1.0, reduce: str = "mean"):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.tau = tau
        assert reduce in ("mean", "max"), "reduce must be mean or max"
        self.reduce = reduce

    def forward(self, queries: torch.Tensor, l_feats_btC: torch.Tensor):
        """
        Args:
            queries (torch.Tensor): The learnable queries, shape (B, Nq, C).
            l_feats_btC (torch.Tensor): Language features, shape (B, T, C).

        Returns:
            soft_mask (torch.Tensor): L1-normalized weights, shape (B, T, 1).
            attn (torch.Tensor): Per-query attention weights for visualization, shape (B, Nq, T).
        """
        B, Nq, C = queries.shape
        _, T, C2 = l_feats_btC.shape
        assert C == C2

        Q = self.q_proj(queries)      # (B, Nq, C)
        K = self.k_proj(l_feats_btC)  # (B, T, C)
        
        # 计算相似度
        sim = torch.einsum('bnc,btc->bnt', Q, K) / (math.sqrt(C) * max(self.tau, 1e-6))
        A = F.softmax(sim, dim=-1)    # (B, Nq, T)

        if self.reduce == "mean":
            m = A.mean(dim=1)         # (B, T)
        else: # max
            m, _ = A.max(dim=1)       # (B, T)

        m = m / (m.sum(dim=1, keepdim=True) + 1e-8)  # (B, T)
        m = m.clamp_min(1e-8) 
        return m.unsqueeze(-1), A

class MaskedQueryAttn(nn.Module):
    def __init__(self, num_queries, text_dim):
        super().__init__()
        self.summary_queries = nn.Parameter(torch.randn(1, num_queries, text_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=text_dim, 
            num_heads=8, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, text_feats, mask):
        B = text_feats.shape[0]
        key_padding_mask = (mask.squeeze(-1) == 0)
        queries, _ = self.attention(
            query=self.summary_queries.expand(B, -1, -1),
            key=text_feats,
            value=text_feats,
            key_padding_mask=key_padding_mask
        )
        return self.norm(queries)    

class MaskedQueryPooler(nn.Module):
    def __init__(self, in_dim: int, num_queries: int, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_queries, bias=False)
        self.temperature = temperature

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        B, T, C = tokens.shape
        logits = self.proj(tokens) / max(self.temperature, 1e-6)  # (B, T, Nq)
        if mask is not None:
            m = (mask.squeeze(-1) > 0)  # (B, T) bool
            minus_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~m.unsqueeze(-1), minus_inf)
        attn = F.softmax(logits, dim=1)                          # (B, T, Nq)
        queries = torch.einsum('btn,btc->bnc', attn, tokens)     # (B, Nq, C)
        return queries, attn.permute(0, 2, 1)                    # (B,Nq,T)

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # cat op
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def masked_token_softmax(scores: torch.Tensor, mask01: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    """
    scores: (B, N) raw logits
    mask01: (B, N) in {0,1}  -> 1 where valid tokens
    returns prob: (B, N), zero where mask==0. safe when all zeros.
    """
    # zero-out invalid positions before max
    very_neg = torch.finfo(scores.dtype).min / 4
    masked_scores = scores.masked_fill(mask01 == 0, very_neg)
    # stability: subtract max on valid tokens
    maxv, _ = masked_scores.max(dim=dim, keepdim=True)
    exps = torch.exp(masked_scores - maxv)
    exps = exps * mask01
    denom = exps.sum(dim=dim, keepdim=True) + eps
    prob = exps / denom
    # keep zeros on masked positions
    prob = prob * mask01
    return prob
    
class QuerySelectTextContext(nn.Module):
    """
    inputs:
      queries: (B, Nq, C_t)
      l_feats: (B, C_t, T)
      l_mask : (B, T, 1)   1=有效
    outputs:
      ctx: (B, Nq, D)      # 每个 query 的上下文向量
      A  : (B, Nq, T)      # 文本 token 权重（可视化）
    """
    def __init__(self, text_dim: int, ctx_dim: int = None, dropout: float = 0.0):
        super().__init__()
        D = ctx_dim or text_dim
        self.q_proj = nn.Linear(text_dim, D, bias=False)
        self.k_proj = nn.Linear(text_dim, D, bias=False)
        self.v_proj = nn.Linear(text_dim, D, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = D ** 0.5

    def forward(self, queries, l_feats, l_mask=None):
        B, Nq, Ct = queries.shape
        _, Ct2, T = l_feats.shape
        assert Ct == Ct2
        l_btc = l_feats.permute(0, 2, 1).contiguous()   # (B,T,Ct)

        Q = self.q_proj(queries)                        # (B,Nq,D)
        K = self.k_proj(l_btc)                          # (B,T,D)
        V = self.v_proj(l_btc)                          # (B,T,D)

        logits = torch.einsum('bnd,btd->bnt', Q, K) / self.scale  # (B,Nq,T)
        if l_mask is not None:
            logits = logits + torch.log(l_mask.squeeze(-1).clamp_min(1e-8)).unsqueeze(1)
        A = F.softmax(logits, dim=-1)                   # (B,Nq,T)
        A = self.dropout(A)

        ctx = torch.einsum('bnt,btd->bnd', A, V)        # (B,Nq,D)
        return ctx, A