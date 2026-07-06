"""
Unit tests for Enhanced UCOD-ADF agent modules.

Tests forward pass shapes, multi-scale feature inputs, loss computation,
and memory agent behavior for all 4 agents in the Agentic Decision Framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import pytest
from models.agents.quality_assessment_agent import (
    QualityAssessmentAgent,
    compute_binary_entropy,
    compute_disagreement,
    compute_feature_gradient,
)
from models.agents.region_analysis_agent import (
    RegionAnalysisAgent,
    compute_region_consistency_loss,
)
from models.agents.memory_agent import MemoryAgent
from models.agents.decision_agent import (
    DecisionAgent,
    construct_spatial_pseudo_label,
)


B, C, H, W = 2, 768, 68, 68


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def dummy_inputs(device):
    return {
        'pseudo_labels': torch.rand(B, 1, H, W, device=device),
        'teacher_logits': torch.randn(B, 1, H, W, device=device),
        'student_logits': torch.randn(B, 1, H, W, device=device),
        'features': torch.randn(B, C, H, W, device=device),
    }


# ─────────────────────────────────────────
# Agent 1: Enhanced QAA
# ─────────────────────────────────────────

class TestQAA:
    def test_forward_shape_with_features(self, device, dummy_inputs):
        qaa = QualityAssessmentAgent(delta=0.3, hidden_dim=64).to(device)

        reliability = qaa(
            fixed_strategy_mask=dummy_inputs['pseudo_labels'],
            teacher_logits=dummy_inputs['teacher_logits'],
            student_logits=dummy_inputs['student_logits'],
            features=dummy_inputs['features'],
        )

        assert reliability.shape == (B, 1, H, W)
        assert reliability.min() >= 0.0
        assert reliability.max() <= 1.0

    def test_confidence_loss(self, device, dummy_inputs):
        qaa = QualityAssessmentAgent(delta=0.3).to(device)

        reliability = qaa(
            fixed_strategy_mask=dummy_inputs['pseudo_labels'],
            teacher_logits=dummy_inputs['teacher_logits'],
            student_logits=dummy_inputs['student_logits'],
            features=dummy_inputs['features'],
        )

        loss = qaa.compute_confidence_loss(
            reliability_map=reliability,
            teacher_logits=dummy_inputs['teacher_logits'],
            fixed_strategy_mask=dummy_inputs['pseudo_labels'],
        )

        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_feature_gradient(self, dummy_inputs):
        feat_grad = compute_feature_gradient(dummy_inputs['features'])
        assert feat_grad.shape == (B, 1, H, W)
        assert feat_grad.min() >= 0.0


# ─────────────────────────────────────────
# Agent 2: Enhanced RAA
# ─────────────────────────────────────────

class TestRAA:
    def test_forward_shape(self, device, dummy_inputs):
        raa = RegionAnalysisAgent()

        region_map, region_info = raa(
            student_logits=dummy_inputs['student_logits'],
            features=dummy_inputs['features'],
        )

        assert region_map.shape == (B, 1, H, W)
        assert len(region_info) == B
        assert isinstance(region_info[0], list)

    def test_region_map_values(self, device, dummy_inputs):
        raa = RegionAnalysisAgent()

        region_map, _ = raa(
            student_logits=dummy_inputs['student_logits'],
            features=dummy_inputs['features'],
        )

        unique_vals = region_map.unique()
        for v in unique_vals:
            assert v.item() in [0.0, 0.5, 1.0], f"Unexpected region map value: {v.item()}"


# ─────────────────────────────────────────
# Agent 3: Enhanced Memory Agent
# ─────────────────────────────────────────

class TestMemoryAgent:
    def test_initial_stability_and_variance(self):
        ma = MemoryAgent(dataset_size=100, spatial_h=H, spatial_w=W, warmup_visits=3)

        image_ids = torch.tensor([0, 1])
        current_pred = torch.rand(2, 1, H, W)

        stability, variance = ma.get_stability(image_ids, current_pred)

        assert stability.shape == (2, 1, H, W)
        assert variance.shape == (2, 1, H, W)

    def test_update_and_stability(self):
        ma = MemoryAgent(dataset_size=100, spatial_h=H, spatial_w=W, warmup_visits=2)

        image_ids = torch.tensor([5])
        fixed_pred = torch.ones(1, 1, H, W) * 0.8

        for _ in range(3):
            ma.update(image_ids, fixed_pred)

        stability, variance = ma.get_stability(image_ids, fixed_pred)
        assert stability.mean().item() > 0.7


# ─────────────────────────────────────────
# Agent 4: Enhanced Decision Agent (with Attention Gating)
# ─────────────────────────────────────────

class TestDecisionAgent:
    def test_forward_shape(self, device):
        da = DecisionAgent(hidden_dim=64, mid_dim=32).to(device)

        reliability = torch.rand(B, 1, H, W, device=device)
        region_map = torch.ones(B, 1, H, W, device=device) * 0.5
        stability = torch.rand(B, 1, H, W, device=device)
        variance = torch.rand(B, 1, H, W, device=device) * 0.1

        mixing, lt_trigger = da(reliability, region_map, stability, variance, epoch_progress=0.5)

        assert mixing.shape == (B, 1, H, W)
        assert lt_trigger.shape == (B, 1, H, W)
        assert mixing.min() >= 0.0 and mixing.max() <= 1.0
        assert lt_trigger.min() >= 0.0 and lt_trigger.max() <= 1.0


# ─────────────────────────────────────────
# Full End-to-End Integration
# ─────────────────────────────────────────

class TestFullPipeline:
    def test_end_to_end_forward(self, device, dummy_inputs):
        qaa = QualityAssessmentAgent().to(device)
        raa = RegionAnalysisAgent()
        ma = MemoryAgent(dataset_size=100, spatial_h=H, spatial_w=W)
        da = DecisionAgent().to(device)

        reliability = qaa(
            dummy_inputs['pseudo_labels'],
            dummy_inputs['teacher_logits'],
            dummy_inputs['student_logits'],
            dummy_inputs['features'],
        )

        region_map, region_info = raa(
            dummy_inputs['student_logits'],
            dummy_inputs['features'],
        )

        image_ids = torch.tensor([0, 1])
        student_probs = torch.sigmoid(dummy_inputs['student_logits'].detach())
        stability, variance = ma.get_stability(image_ids, student_probs)

        mixing, lt_trigger = da(
            reliability, region_map, stability, variance, epoch_progress=0.4
        )

        pseudo = construct_spatial_pseudo_label(
            mixing, dummy_inputs['teacher_logits'], dummy_inputs['pseudo_labels']
        )

        loss_conf = qaa.compute_confidence_loss(
            reliability, dummy_inputs['teacher_logits'], dummy_inputs['pseudo_labels']
        )
        loss_region = compute_region_consistency_loss(
            dummy_inputs['student_logits'], region_info
        )
        loss_temporal = ma.compute_temporal_loss(
            image_ids, dummy_inputs['student_logits']
        )

        ma.update(image_ids, student_probs)

        assert pseudo.shape == (B, 1, H, W)
        assert loss_conf.dim() == 0
        assert loss_region.dim() == 0
        assert loss_temporal.dim() == 0

        total_loss = loss_conf + loss_temporal
        total_loss.backward()

        qaa_has_grad = any(p.grad is not None for p in qaa.parameters())
        assert qaa_has_grad, "QAA should receive gradients"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
