"""
Decision Agent (DA) — Enhanced Agent 4 for UCOD-ADF.

Combines signals from QAA (reliability), RAA (regions), and Memory Agent
(temporal stability & temporal variance) via Spatial & Channel Attention Gating.

Inputs (5 channels):
  1. Reliability Map R_i (from QAA)
  2. Region Map (from RAA)
  3. Temporal Stability Map S_temp (from MA)
  4. Temporal Variance Map σ²_temp (from MA)
  5. Epoch Progress t/T

Outputs:
  - Spatial mixing map W_i(x,y) ∈ [0, 1]
  - Look-Twice trigger map LT_flag(x,y) ∈ [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ChannelAttention(nn.Module):
    """Channel Attention Gate."""
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Gate."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(scale))


class DecisionAgent(nn.Module):
    """
    Enhanced Decision Agent with CBAM Attention Gating.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mid_dim: int = 32,
        lt_threshold: float = 0.5,
    ):
        super().__init__()
        self.lt_threshold = lt_threshold

        # 5 input channels: [R_i, region_map, S_temp, V_temp, progress_map]
        self.in_conv = nn.Sequential(
            nn.Conv2d(5, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Attention Gating
        self.ca = ChannelAttention(hidden_dim)
        self.sa = SpatialAttention()

        self.mid_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )

        # Output heads
        self.mixing_head = nn.Conv2d(mid_dim, 1, kernel_size=1)
        self.lt_head = nn.Conv2d(mid_dim, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.mixing_head.bias, 0.0)
        nn.init.constant_(self.lt_head.bias, -2.0)

    def forward(
        self,
        reliability_map: torch.Tensor,
        region_map: torch.Tensor,
        stability_map: torch.Tensor,
        variance_map: torch.Tensor,
        epoch_progress: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce spatial mixing map and Look-Twice trigger map.
        """
        B, _, H, W = reliability_map.shape
        device = reliability_map.device

        progress_map = torch.full(
            (B, 1, H, W), epoch_progress, device=device, dtype=reliability_map.dtype
        )

        x = torch.cat([
            reliability_map,
            region_map,
            stability_map,
            variance_map,
            progress_map,
        ], dim=1)

        feat = self.in_conv(x)

        # Apply Attention Gating
        feat = feat * self.ca(feat)
        feat = feat * self.sa(feat)

        mid = self.mid_conv(feat)

        mixing_map = torch.sigmoid(self.mixing_head(mid))
        lt_trigger_map = torch.sigmoid(self.lt_head(mid))

        return mixing_map, lt_trigger_map

    @torch.no_grad()
    def get_lt_trigger_flags(
        self,
        lt_trigger_map: torch.Tensor,
        region_info_batch: list,
    ) -> list:
        trigger_flags = []
        for b, regions in enumerate(region_info_batch):
            batch_triggers = []
            for idx, region in enumerate(regions):
                mask = region['pixel_mask']
                region_lt_values = lt_trigger_map[b, 0, mask]
                if region_lt_values.max().item() > self.lt_threshold:
                    batch_triggers.append(idx)
            trigger_flags.append(batch_triggers)
        return trigger_flags


def construct_spatial_pseudo_label(
    mixing_map: torch.Tensor,
    teacher_logits: torch.Tensor,
    fixed_strategy_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Construct spatial pseudo-label via Decision Agent mixing map.
    """
    teacher_probs = torch.sigmoid(teacher_logits)
    pseudo_label = mixing_map * teacher_probs + (1 - mixing_map) * fixed_strategy_mask
    return pseudo_label
