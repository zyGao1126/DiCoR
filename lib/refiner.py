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
    aug_prob = 0.6
    aug_morph_prob = 0.6
    morph_max_radius = 4
    area_thresholds = (0.002, 0.02)

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

    def augment_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        """Apply the best ablation's area-adaptive morphology policy."""
        if torch.rand(1).item() > self.aug_prob:
            return prompt

        area_ratios = prompt.mean(dim=(2, 3)).squeeze(1)
        augmented = []
        for index in range(prompt.shape[0]):
            sample = prompt[index:index + 1]
            if torch.rand(1).item() < self.aug_morph_prob:
                area = area_ratios[index].item()
                if area < self.area_thresholds[0]:
                    max_radius = max(1, self.morph_max_radius // 2)
                elif area > self.area_thresholds[1]:
                    max_radius = int(self.morph_max_radius * 1.5)
                else:
                    max_radius = self.morph_max_radius

                radius = torch.randint(-max_radius, max_radius + 1, (1,)).item()
                if radius > 0:
                    sample = self._dilate01(sample, radius)
                elif radius < 0:
                    sample = self._erode01(sample, -radius)
            augmented.append(sample)
        return torch.cat(augmented, dim=0)

    def build_focus_map(self, prompt_prob: torch.Tensor) -> torch.Tensor:
        dilation_radii = (6, 12, 18)
        focus_weights = (0.7, 0.2, 0.1)
        band_radius = 3

        pred_fg = (prompt_prob > 0.5).float()
        area = pred_fg.mean(dim=(2, 3), keepdim=True)
        r = torch.where(
            area < self.area_thresholds[0],
            torch.full_like(area, float(dilation_radii[0])),
            torch.where(
                area > self.area_thresholds[1],
                torch.full_like(area, float(dilation_radii[2])),
                torch.full_like(area, float(dilation_radii[1])),
            ),
        ).view(-1)

        focus = torch.cat([
            self._dilate01(pred_fg[i:i + 1], int(r[i].item()))
            for i in range(prompt_prob.size(0))
        ], dim=0)

        edge = (self._dilate01(pred_fg, 1) - self._erode01(pred_fg, 1)).clamp(0, 1)
        band = self._dilate01(edge, band_radius)
        unc = (1.0 - (2.0 * prompt_prob - 1.0).abs()).clamp(0, 1)

        focus_map = (
            focus_weights[0] * focus + focus_weights[1] * band + focus_weights[2] * unc
        ).clamp(0, 1)
        return 0.05 + 0.95 * focus_map

    @staticmethod
    def logits_from_prob_fg(prob_fg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = prob_fg.clamp(eps, 1.0 - eps)
        log_fg = torch.log(p)
        log_bg = torch.log(1.0 - p)
        return torch.cat([log_bg, log_fg], dim=1)
