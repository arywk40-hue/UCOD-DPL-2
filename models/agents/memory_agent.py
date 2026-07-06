"""
Memory Agent (MA) — Enhanced Agent 3 for UCOD-ADF.

Tracks both Temporal Mean Prediction (EMA) and Running Temporal Variance (Variance EMA)
for each training sample across epochs.

Outputs:
  1. Mean stability signal (1 - |pred - EMA_mean|)
  2. Temporal prediction variance σ²_temp(x,y) — signals unstable / oscillating regions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MemoryAgent(nn.Module):
    """
    Enhanced Per-image temporal stability and variance tracker.
    """

    def __init__(
        self,
        dataset_size: int,
        spatial_h: int = 68,
        spatial_w: int = 68,
        alpha: float = 0.95,
        warmup_visits: int = 3,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.warmup_visits = warmup_visits
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Register CPU-side memory buffers
        self.register_buffer(
            'ema_masks',
            torch.full((dataset_size, 1, spatial_h, spatial_w), 0.5),
            persistent=True,
        )
        self.register_buffer(
            'ema_vars',
            torch.zeros((dataset_size, 1, spatial_h, spatial_w)),
            persistent=True,
        )
        self.register_buffer(
            'visit_counts',
            torch.zeros(dataset_size, dtype=torch.long),
            persistent=True,
        )

        self.ema_masks = self.ema_masks.cpu()
        self.ema_vars = self.ema_vars.cpu()
        self.visit_counts = self.visit_counts.cpu()

    @torch.no_grad()
    def get_stability(
        self,
        image_ids: torch.Tensor,
        current_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal stability map AND temporal variance map.

        Returns:
            stability_map: (B, 1, H, W)
            variance_map: (B, 1, H, W)
        """
        device = current_prediction.device
        B = image_ids.shape[0]
        H, W = current_prediction.shape[-2:]

        ema_batch = self.ema_masks[image_ids].to(device)
        var_batch = self.ema_vars[image_ids].to(device)

        if ema_batch.shape[-2:] != (H, W):
            ema_batch = torch.nn.functional.interpolate(ema_batch, size=(H, W), mode='bilinear', align_corners=False)
            var_batch = torch.nn.functional.interpolate(var_batch, size=(H, W), mode='bilinear', align_corners=False)

        raw_stability = 1.0 - torch.abs(current_prediction - ema_batch)

        visits = self.visit_counts[image_ids].float().to(device)
        warmup_weight = torch.clamp(visits / self.warmup_visits, max=1.0).view(B, 1, 1, 1)

        stability = warmup_weight * raw_stability + (1 - warmup_weight) * 0.5
        variance = warmup_weight * var_batch + (1 - warmup_weight) * 0.25

        return stability, variance

    @torch.no_grad()
    def get_ema(
        self,
        image_ids: torch.Tensor,
        target_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        ema = self.ema_masks[image_ids]
        if target_size is not None and ema.shape[-2:] != target_size:
            ema = torch.nn.functional.interpolate(ema, size=target_size, mode='bilinear', align_corners=False)
        return ema

    @torch.no_grad()
    def update(
        self,
        image_ids: torch.Tensor,
        current_prediction: torch.Tensor,
    ) -> None:
        """
        Update EMA mean and variance buffers.
        """
        pred_cpu = current_prediction.detach().cpu()

        if pred_cpu.shape[-2:] != (self.spatial_h, self.spatial_w):
            pred_cpu = torch.nn.functional.interpolate(
                pred_cpu,
                size=(self.spatial_h, self.spatial_w),
                mode='bilinear',
                align_corners=False,
            )

        for i, img_id in enumerate(image_ids):
            idx = img_id.item()
            old_mean = self.ema_masks[idx].clone()

            # Update Mean EMA
            self.ema_masks[idx] = self.alpha * old_mean + (1 - self.alpha) * pred_cpu[i]

            # Update Variance EMA (running squared error)
            diff_sq = (pred_cpu[i] - old_mean) ** 2
            self.ema_vars[idx] = self.alpha * self.ema_vars[idx] + (1 - self.alpha) * diff_sq

            self.visit_counts[idx] += 1

    def compute_temporal_loss(
        self,
        image_ids: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Temporal consistency loss: L1 between current prediction and EMA.
        """
        device = student_logits.device
        H, W = student_logits.shape[-2:]

        student_probs = torch.sigmoid(student_logits)
        ema = self.get_ema(image_ids, target_size=(H, W)).to(device)

        loss = torch.nn.functional.l1_loss(student_probs, ema)
        return loss

    def reset(self) -> None:
        self.ema_masks.fill_(0.5)
        self.ema_vars.zero_()
        self.visit_counts.zero_()
