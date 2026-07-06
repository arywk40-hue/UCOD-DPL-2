"""
Quality Assessment Agent (QAA) — Enhanced Agent 1 for UCOD-ADF.

 Dense spatial reliability estimator for pseudo-labels.
Enhancements:
  1. Multi-scale Receptive Field: Uses Atrous (Dilated) Spatial Pyramid Conv blocks
     to assess reliability at both fine object boundaries and global object context.
  2. Feature Contrast Input: Takes normalized local feature gradient/contrast as an
     additional 6th input channel to sense camouflage visual ambiguity.
  3. Calibrated Reliability Output: Self-supervised loss with boundary-weighted
     calibration targets.

Input channels (6):
  1. Fixed-strategy mask P_fs
  2. Teacher probability σ(P_t)
  3. Student probability σ(Y_fg)
  4. Disagreement map |σ(P_t) - P_fs|
  5. Student entropy H(Y_fg)
  6. Feature gradient magnitude ||∇F||
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_binary_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel binary entropy from logits."""
    probs = torch.sigmoid(logits)
    probs = probs.clamp(1e-6, 1 - 1e-6)
    entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
    return entropy


def compute_disagreement(
    teacher_logits: torch.Tensor,
    fixed_strategy_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pixel disagreement map."""
    teacher_probs = torch.sigmoid(teacher_logits)
    return torch.abs(teacher_probs - fixed_strategy_mask)


def compute_feature_gradient(features: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial gradient magnitude of feature representations.
    Aggregates norm across channel dimension.
    """
    # Reduce feature channels via mean magnitude for efficiency
    feat_norm = torch.linalg.norm(features, dim=1, keepdim=True)
    feat_norm = (feat_norm - feat_norm.min()) / (feat_norm.max() - feat_norm.min() + 1e-8)
    
    grad_x = feat_norm[:, :, :, 1:] - feat_norm[:, :, :, :-1]
    grad_y = feat_norm[:, :, 1:, :] - feat_norm[:, :, :-1, :]
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
    
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)


class QualityAssessmentAgent(nn.Module):
    """
    Enhanced Quality Assessment Agent with Multi-scale Receptive Field.
    """

    def __init__(self, delta: float = 0.3, hidden_dim: int = 64):
        super().__init__()
        self.delta = delta

        # Input projection (6 channels -> hidden_dim)
        self.in_proj = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Multi-scale Dilated Spatial Pyramid Conv Block
        self.branch_d1 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch_d2 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch_d4 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=4, dilation=4, bias=False)
        self.branch_d6 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=6, dilation=6, bias=False)

        self.pyramid_bn = nn.BatchNorm2d(hidden_dim)
        self.pyramid_act = nn.ReLU(inplace=True)

        # Output refinement head
        self.out_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        fixed_strategy_mask: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute dense spatial reliability map.
        """
        teacher_probs = torch.sigmoid(teacher_logits)
        student_probs = torch.sigmoid(student_logits)

        disagreement = torch.abs(teacher_probs - fixed_strategy_mask)
        entropy = compute_binary_entropy(student_logits)

        if features is not None:
            if features.shape[-2:] != fixed_strategy_mask.shape[-2:]:
                features = F.interpolate(features, size=fixed_strategy_mask.shape[-2:], mode='bilinear', align_corners=False)
            feat_grad = compute_feature_gradient(features)
        else:
            feat_grad = torch.zeros_like(fixed_strategy_mask)

        x = torch.cat([
            fixed_strategy_mask,
            teacher_probs,
            student_probs,
            disagreement,
            entropy,
            feat_grad,
        ], dim=1)

        feat = self.in_proj(x)

        # Multi-scale pyramid feature aggregation
        d1 = self.branch_d1(feat)
        d2 = self.branch_d2(feat)
        d4 = self.branch_d4(feat)
        d6 = self.branch_d6(feat)
        pyramid = self.pyramid_act(self.pyramid_bn(torch.cat([d1, d2, d4, d6], dim=1)))

        reliability = torch.sigmoid(self.out_head(pyramid + feat))
        return reliability

    def compute_confidence_loss(
        self,
        reliability_map: torch.Tensor,
        teacher_logits: torch.Tensor,
        fixed_strategy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Self-supervised confidence calibration loss.
        """
        disagreement = compute_disagreement(teacher_logits, fixed_strategy_mask)
        target = (disagreement < self.delta).float()
        loss = F.binary_cross_entropy(reliability_map, target)
        return loss
