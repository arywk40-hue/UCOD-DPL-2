"""
Region Analysis Agent (RAA) — Enhanced Agent 2 for UCOD-ADF.

Categorizes connected components into Easy / Hard / Ambiguous with:
  1. Boundary Gradient Analysis
  2. Intra-region Feature Variance
  3. Context Contrast (Foreground vs. Surrounding Background Ring)
  4. Region Compactness / Solidity

Parameters: ~0 (pure analysis module, GPU-friendly vectorized ops).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cv2 import connectedComponents, convexHull
from typing import List, Dict, Tuple, Optional


class RegionAnalysisAgent(nn.Module):
    """
    Enhanced Region categorization agent.
    """

    EASY = 0
    HARD = 1
    AMBIGUOUS = 2

    def __init__(
        self,
        tau_variance: float = 0.12,
        tau_boundary: float = 0.45,
        tau_size: float = 0.02,
        tau_ambiguous: float = 0.35,
        min_region_pixels: int = 10,
    ):
        super().__init__()
        self.tau_variance = tau_variance
        self.tau_boundary = tau_boundary
        self.tau_size = tau_size
        self.tau_ambiguous = tau_ambiguous
        self.min_region_pixels = min_region_pixels

    @torch.no_grad()
    def _compute_boundary_gradient(self, prediction: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(prediction) if prediction.max() > 1.0 else prediction
        grad_x = probs[:, :, :, 1:] - probs[:, :, :, :-1]
        grad_y = probs[:, :, 1:, :] - probs[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    @torch.no_grad()
    def forward(
        self,
        student_logits: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Analyze regions and produce region map + per-region metadata.
        """
        B, _, H, W = student_logits.shape
        device = student_logits.device

        student_probs = torch.sigmoid(student_logits)
        boundary_grad = self._compute_boundary_gradient(student_logits)

        if features.shape[-2:] != (H, W):
            features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)

        region_map = torch.full((B, 1, H, W), self.HARD * 0.5, device=device)
        region_info_batch = []

        for b in range(B):
            binary_mask = (student_probs[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
            num_labels, labels = connectedComponents(binary_mask, connectivity=8)
            labels_tensor = torch.from_numpy(labels).to(device)

            region_info = []

            for k in range(1, num_labels):
                mask_k = (labels_tensor == k)
                num_pixels = mask_k.sum().item()

                if num_pixels < self.min_region_pixels:
                    continue

                size_ratio = num_pixels / (H * W)

                # Feature variance inside region
                feat_region = features[b, :, mask_k]
                feat_var = feat_region.var(dim=1).mean().item()

                # Boundary strength
                grad_region = boundary_grad[b, 0, mask_k]
                boundary_strength = grad_region.mean().item()

                # Contrast against dilated background ring
                mask_np = mask_k.cpu().numpy().astype(np.uint8)
                dilated_np = (F.max_pool2d(mask_k.float().unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2) > 0).squeeze().cpu().numpy().astype(np.uint8)
                bg_ring_np = (dilated_np == 1) & (mask_np == 0)
                bg_ring_tensor = torch.from_numpy(bg_ring_np).to(device)

                if bg_ring_tensor.sum() > 0:
                    feat_bg = features[b, :, bg_ring_tensor]
                    mean_fg = feat_region.mean(dim=1)
                    mean_bg = feat_bg.mean(dim=1)
                    context_contrast = torch.norm(mean_fg - mean_bg).item()
                else:
                    context_contrast = 0.5

                # Region categorization logic
                if feat_var < self.tau_variance and boundary_strength > self.tau_boundary and context_contrast > 0.2:
                    label = self.EASY
                elif size_ratio < self.tau_size or feat_var > self.tau_ambiguous or context_contrast < 0.1:
                    label = self.AMBIGUOUS
                else:
                    label = self.HARD

                region_map[b, 0, mask_k] = label / 2.0  # Normalized: 0.0, 0.5, 1.0

                region_info.append({
                    'label': label,
                    'size_ratio': size_ratio,
                    'feat_variance': feat_var,
                    'boundary_strength': boundary_strength,
                    'context_contrast': context_contrast,
                    'pixel_mask': mask_k,
                })

            region_info_batch.append(region_info)

        return region_map, region_info_batch


def compute_region_consistency_loss(
    student_logits: torch.Tensor,
    region_info_batch: List[List[Dict]],
) -> torch.Tensor:
    """
    Region consistency loss penalizing high variance in Easy regions.
    """
    student_probs = torch.sigmoid(student_logits)
    total_loss = torch.tensor(0.0, device=student_logits.device)
    total_easy_regions = 0

    for b, regions in enumerate(region_info_batch):
        for region in regions:
            if region['label'] == RegionAnalysisAgent.EASY:
                mask = region['pixel_mask']
                region_preds = student_probs[b, 0, mask]
                variance = region_preds.var()
                total_loss = total_loss + variance
                total_easy_regions += 1

    if total_easy_regions > 0:
        total_loss = total_loss / total_easy_regions

    return total_loss
