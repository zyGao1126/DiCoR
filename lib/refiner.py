import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RefineUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, base_ch),
            ConvBNReLU(base_ch, base_ch)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            ConvBNReLU(base_ch, base_ch * 2),
            ConvBNReLU(base_ch * 2, base_ch * 2)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            ConvBNReLU(base_ch * 2, base_ch * 4),
            ConvBNReLU(base_ch * 4, base_ch * 4)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = nn.Sequential(
            ConvBNReLU(base_ch * 4, base_ch * 8),
            ConvBNReLU(base_ch * 8, base_ch * 8)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            ConvBNReLU(base_ch * 8, base_ch * 16),
            ConvBNReLU(base_ch * 16, base_ch * 16)
        )

        self.up4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 2, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.out_conv(d1)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out


class RefinerPromptProcessor:
    FOCUS_AREA_SMALL = 0.002
    FOCUS_AREA_LARGE = 0.02
    FOCUS_R_SMALL = 6
    FOCUS_R_MED = 12
    FOCUS_R_LARGE = 18
    FOCUS_R_BAND = 3
    FOCUS_W_FOCUS = 0.7
    FOCUS_W_BAND = 0.2
    FOCUS_W_UNC = 0.1

    @staticmethod
    def _dilate01(x01: torch.Tensor, r: int):
        if r <= 0:
            return x01
        k = 2 * r + 1
        return F.max_pool2d(x01, kernel_size=k, stride=1, padding=r)

    @staticmethod
    def _erode01(x01: torch.Tensor, r: int):
        if r <= 0:
            return x01
        return 1.0 - RefinerPromptProcessor._dilate01(1.0 - x01, r)

    def build_focus_map(self, prompt_prob: torch.Tensor) -> torch.Tensor:
        pred_fg = (prompt_prob > 0.5).float()
        area = pred_fg.mean(dim=(2, 3), keepdim=True)
        r = torch.where(
            area < self.FOCUS_AREA_SMALL,
            torch.full_like(area, float(self.FOCUS_R_SMALL)),
            torch.where(
                area > self.FOCUS_AREA_LARGE,
                torch.full_like(area, float(self.FOCUS_R_LARGE)),
                torch.full_like(area, float(self.FOCUS_R_MED)),
            ),
        ).view(-1)

        focus = torch.cat([
            self._dilate01(pred_fg[i:i + 1], int(r[i].item()))
            for i in range(prompt_prob.size(0))
        ], dim=0)

        edge = (self._dilate01(pred_fg, 1) - self._erode01(pred_fg, 1)).clamp(0, 1)
        band = self._dilate01(edge, self.FOCUS_R_BAND)
        unc = (1.0 - (2.0 * prompt_prob - 1.0).abs()).clamp(0, 1)

        focus_map = (
            self.FOCUS_W_FOCUS * focus +
            self.FOCUS_W_BAND * band +
            self.FOCUS_W_UNC * unc
        ).clamp(0, 1)
        return focus_map

    @staticmethod
    def logits_from_prob_fg(prob_fg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = prob_fg.clamp(eps, 1.0 - eps)
        log_fg = torch.log(p)
        log_bg = torch.log(1.0 - p)
        return torch.cat([log_bg, log_fg], dim=1)
