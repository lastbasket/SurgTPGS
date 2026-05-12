"""
Fuse CLIP patch features (x / res3) with mid/deep ViT features (res4, res5) on the 24×24 grid,
then upsample with **ConvTranspose2d only** to 192×192 (512 channels). A single bilinear
``interpolate`` maps 192×192 to the padded input size (height×width).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelevancyAggregator(nn.Module):
    """
    - Fuse: ``[x; res4; res5]`` at H0×W0 (e.g. 24×24) → 512 channels via 1×1 conv.
    - Upsample: three stride-2 ConvTranspose2d (24 → 48 → 96 → 192), no ``interpolate``.
    - Last step: ``F.interpolate`` to arbitrary ``(height, width)`` only.
    """

    def __init__(self, proj_dim: int = 768, out_ch: int = 512):
        super().__init__()
        fused_in = 512 + proj_dim + proj_dim
        self.fuse = nn.Conv2d(fused_in, out_ch, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.fuse.weight, mode="fan_out", nonlinearity="relu")
        if self.fuse.bias is not None:
            nn.init.zeros_(self.fuse.bias)

        # 24×24 → 48 → 96 → 192 (×8 spatial); channels stay ``out_ch`` (512)
        self.up_to_192 = nn.Sequential(
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
        )
        for m in self.up_to_192:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        vis_guidance: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        x: [B, 512, H0, W0] — res3 / img_feat.
        vis_guidance: ``res4``, ``res5`` at [B, proj_dim, H0, W0].
        Returns [B, 512, height, width].
        """
        if vis_guidance is None:
            raise ValueError("vis_guidance is required for RelevancyAggregator")
        res4 = vis_guidance["res4"]
        res5 = vis_guidance["res5"]
        if x.shape[-2:] != res4.shape[-2:] or x.shape[-2:] != res5.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch: x {x.shape}, res4 {res4.shape}, res5 {res5.shape}"
            )
        z = torch.cat([x, res4, res5], dim=1)
        z = F.relu(self.fuse(z), inplace=True)
        # print('z', z.shape)
        z = self.up_to_192(z)
        # print('z after up_to_192', z.shape)
        # Expected [B, 512, 192, 192] when H0=W0=24
        return z
