"""
ADF Training Loop — Enhanced Agentic Decision Framework for UCOD.

Integrates the multi-scale QAA, context-aware RAA, mean+variance Memory Agent,
and Attention-Gated Decision Agent.
"""

from abc import ABCMeta, abstractmethod
from engine.config.config import CfgNode
from engine.utils.metrics.metric import calculate_cod_metrics, statistics
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from cv2 import connectedComponents, boundingRect
from torchvision import transforms
from PIL import Image
from data.utils.feature_extractor import backbone
from .utils import ProgressManager
import numpy as np
import os
from tqdm import tqdm

from models.agents.quality_assessment_agent import QualityAssessmentAgent
from models.agents.region_analysis_agent import (
    RegionAnalysisAgent,
    compute_region_consistency_loss,
)
from models.agents.memory_agent import MemoryAgent
from models.agents.decision_agent import DecisionAgent, construct_spatial_pseudo_label


Image.MAX_IMAGE_PIXELS = None


class ADFTrainLoop:
    def __init__(self, config: CfgNode, runner):
        self.cfg = config
        self._runner = runner
        self._dist_train = self.cfg.train_cfg.dist_train

        self._mode = 'train'
        self._start_epoch = self.cfg.train_cfg.start_epoch
        self._max_epoch = self.cfg.train_cfg.max_epoch
        self.global_step = 0
        self._cur_epoch = 0
        self._start_finetune = self.cfg.train_cfg.start_finetune
        self.finetune = False

        self.criterion = nn.BCEWithLogitsLoss()
        self.ema_alpha = self.cfg.model_cfg.ema_weight

        adf_cfg = self.cfg.train_cfg.adf_cfg
        self.lambda_conf = adf_cfg.lambda_conf
        self.lambda_region = adf_cfg.lambda_region
        self.lambda_temporal = adf_cfg.lambda_temporal
        self.empty_pseudo_label_policy = getattr(adf_cfg, 'empty_pseudo_label_policy', 'error')

        feature_size = self.cfg.model_cfg.feature_size
        dataset_size = len(self.runner.train_dataloader.dataset)
        self._validate_pseudo_label_cache_if_needed()

        # Initialize Enhanced Agents
        self.qaa = QualityAssessmentAgent(
            delta=adf_cfg.qaa_delta,
            hidden_dim=adf_cfg.qaa_hidden_dim,
        ).to(self.runner.accelerator.device)

        self.raa = RegionAnalysisAgent(
            tau_variance=adf_cfg.raa_tau_variance,
            tau_boundary=adf_cfg.raa_tau_boundary,
            tau_size=adf_cfg.raa_tau_size,
            tau_ambiguous=adf_cfg.raa_tau_ambiguous,
        )

        self.memory = MemoryAgent(
            dataset_size=dataset_size,
            spatial_h=feature_size,
            spatial_w=feature_size,
            alpha=adf_cfg.memory_alpha,
            warmup_visits=adf_cfg.memory_warmup,
        )

        self.da = DecisionAgent(
            hidden_dim=adf_cfg.da_hidden_dim,
            mid_dim=adf_cfg.da_mid_dim,
            lt_threshold=adf_cfg.da_lt_threshold,
        ).to(self.runner.accelerator.device)

        self._setup_agent_optimizer()
        self._setup_validation_config()
        self._setup_logging_config()
        self._setup_progress_manager()

        self.best_mae = 1000.0
        self.best_result = None
        self._batch_start_idx = 0

    @property
    def runner(self):
        return self._runner

    def _validate_pseudo_label_cache_if_needed(self) -> None:
        if self.empty_pseudo_label_policy != 'error':
            return

        dataset = self.runner.train_dataloader.dataset
        pseudo_cache = dataset.cache_manager.get_pseudo_label_cache()
        expected = len(dataset)

        if pseudo_cache is None or pseudo_cache.mode != 'r' or pseudo_cache.length() != expected:
            cache_path = getattr(pseudo_cache, 'base_path', os.path.join(
                self.cfg.dataset_cfg.cache_dir,
                'pseudo_label_cache',
                self.cfg.dataset_cfg.trainset_cfg.DATASET,
            ))
            raise RuntimeError(
                "\n[ADF Pseudo Label Error] Fixed-strategy pseudo labels are required for stable ADF training.\n"
                f"  Expected cache: {cache_path}\n"
                f"  Expected entries: {expected}\n"
                "  Generate the cache with:\n"
                f"    python generate_pseudo_label.py --dataset '{self.cfg.dataset_cfg.trainset_cfg.DATASET}'\n"
                "  For a debugging-only run, set:\n"
                "    --opts train_cfg.adf_cfg.empty_pseudo_label_policy zeros\n"
            )

    def _setup_agent_optimizer(self):
        agent_params = list(self.qaa.parameters()) + list(self.da.parameters())
        self.runner.optimizer.add_param_group({
            'params': agent_params,
            'lr': self.cfg.train_cfg.lr0,
        })

    def _setup_progress_manager(self) -> None:
        self.progress_manager = ProgressManager(self.runner.accelerator)
        self.progress_manager.setup_progress()
        self.progress_manager.add_task("Train Iteration", total=len(self.runner.train_dataloader))
        self.progress_manager.add_task("Validation Iteration", total=len(self.runner.val_dataloader))
        self.progress_manager.add_task("Train Epoch", total=self.cfg.train_cfg.max_epoch)

    def _setup_validation_config(self) -> None:
        self.enable_val = self.cfg.val_cfg.enable_val
        self.val_interval = self.cfg.val_cfg.val_interval
        self.val_start = (self._max_epoch + self.cfg.val_cfg.start_val) \
                            if self.cfg.val_cfg.start_val < 0 else self.cfg.val_cfg.start_val

        self.save_start = (self._max_epoch + self.cfg.train_cfg.save_cfg.start_save) \
                            if self.cfg.train_cfg.save_cfg.start_save < 0 else self.cfg.train_cfg.save_cfg.start_save
        self.save_interval = self.cfg.train_cfg.save_cfg.save_interval

    def _setup_logging_config(self) -> None:
        self.log_interval = self.cfg.log_cfg.log_interval

    def run(self):
        self.runner.logger.log(self.cfg)
        self.runner.logger.log("=" * 60)
        self.runner.logger.log("UCOD-ADF: Enhanced Agentic Decision Framework Training")
        self.runner.logger.log("=" * 60)

        with self.progress_manager:
            self.progress_manager.start_task('Train Epoch')

            while self._cur_epoch < self._max_epoch:
                if self.decide_to_finetune():
                    self.runner.start_finetune()
                    self._setup_agent_optimizer()
                    self.global_step = 0

                self.run_epoch()
                self._cur_epoch += 1

                if self.decide_to_save():
                    self.runner.save_checkpoint(self._cur_epoch)

                if self.decide_to_val():
                    result = self.runner.launch_val_look_twice()
                    self._update_best_result(result)

                self.progress_manager.update_task('Train Epoch')

    def _update_best_result(self, result: Dict[str, float]) -> None:
        mae = result["MAE"]
        if mae < self.best_mae:
            self.best_mae = mae
            self.best_result = result
            result_table = {key: [round(result[key], 4)] for key in result.keys()}
            self.runner.logger.log("best result:")
            self.runner.logger.log_table(result_table)

    def run_epoch(self):
        self.runner.model.train()
        self.qaa.train()
        self.da.train()
        self.progress_manager.start_task("Train Iteration")

        self._batch_start_idx = 0

        for batch_data in self.runner.train_dataloader:
            loss = self._process_batch_adf(batch_data)
            if self._cur_epoch % self.log_interval == 0:
                self.runner.logger.log(f"iter{self.global_step}:loss:{loss:.4f}")
            self.global_step += 1
            self.progress_manager.update_task("Train Iteration")

        self.progress_manager.reset_task("Train Iteration")

    def _process_batch_adf(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        pseudo_labels, _, features, _ = batch_data.values()
        self.runner.optimizer.zero_grad()

        h = w = self.cfg.model_cfg.feature_size
        features = F.interpolate(features, size=(h, w), mode='bilinear')
        
        # Debug-only fallback for unpopulated pseudo-label cache.
        if isinstance(pseudo_labels, list):
            if self.empty_pseudo_label_policy != 'zeros':
                raise RuntimeError(
                    "Pseudo labels are missing. Generate them with "
                    f"`python generate_pseudo_label.py --dataset '{self.cfg.dataset_cfg.trainset_cfg.DATASET}'`."
                )
            batch_size = features.shape[0]
            pseudo_labels = torch.zeros((batch_size, 1, h, w), device=features.device).float()
        else:
            pseudo_labels = F.interpolate(pseudo_labels, size=(h, w), mode='bilinear').float()

        batch_size = features.shape[0]

        image_ids = torch.arange(
            self._batch_start_idx,
            self._batch_start_idx + batch_size,
            device='cpu'
        )
        dataset_size = self.memory.dataset_size
        image_ids = image_ids % dataset_size
        self._batch_start_idx += batch_size

        with torch.no_grad():
            preds_ema = self.runner.model(features, ema=True)

        preds, preds_rev, extra_loss = self.runner.model(features)

        # Agent 1: Enhanced QAA (with multi-scale features & feature gradient)
        reliability_map = self.qaa(
            fixed_strategy_mask=pseudo_labels,
            teacher_logits=preds_ema,
            student_logits=preds,
            features=features.detach(),
        )

        # Agent 2: Enhanced RAA
        with torch.no_grad():
            region_map, region_info_batch = self.raa(
                student_logits=preds.detach(),
                features=features.detach(),
            )

        # Agent 3: Enhanced Memory Agent (Stability + Variance)
        with torch.no_grad():
            student_probs = torch.sigmoid(preds.detach())
            stability_map, variance_map = self.memory.get_stability(image_ids, student_probs)

        # Agent 4: Enhanced Decision Agent (with CBAM Attention Gating)
        epoch_progress = self._cur_epoch / max(self._max_epoch, 1)
        mixing_map, lt_trigger_map = self.da(
            reliability_map=reliability_map,
            region_map=region_map,
            stability_map=stability_map,
            variance_map=variance_map,
            epoch_progress=epoch_progress,
        )

        spatial_pseudo_label = construct_spatial_pseudo_label(
            mixing_map=mixing_map,
            teacher_logits=preds_ema,
            fixed_strategy_mask=pseudo_labels,
        )

        flat_pseudo = spatial_pseudo_label.permute(0, 2, 3, 1).reshape(-1, 1)
        flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)
        flat_preds_rev = preds_rev.permute(0, 2, 3, 1).reshape(-1, 1)

        loss_seg = self.criterion(flat_preds, flat_pseudo)
        loss_seg += self.criterion(flat_preds_rev, (1 - flat_pseudo))

        loss_total = loss_seg
        if extra_loss is not None:
            loss_total = loss_total + extra_loss

        loss_conf = self.qaa.compute_confidence_loss(
            reliability_map=reliability_map,
            teacher_logits=preds_ema,
            fixed_strategy_mask=pseudo_labels,
        )
        loss_total = loss_total + self.lambda_conf * loss_conf

        loss_region = compute_region_consistency_loss(preds, region_info_batch)
        loss_total = loss_total + self.lambda_region * loss_region

        loss_temporal = self.memory.compute_temporal_loss(image_ids, preds)
        loss_total = loss_total + self.lambda_temporal * loss_temporal

        if self.global_step % 50 == 0:
            self.runner.logger.log(
                f"[ADF] epoch={self._cur_epoch} step={self.global_step} "
                f"L_total={loss_total.item():.4f} L_seg={loss_seg.item():.4f} "
                f"L_conf={loss_conf.item():.4f} L_region={loss_region.item():.4f} "
                f"L_temporal={loss_temporal.item():.4f} "
                f"W_mean={mixing_map.mean().item():.3f} "
                f"R_mean={reliability_map.mean().item():.3f}"
            )

        self.runner.accelerator.backward(loss_total)
        self.runner.optimizer.step()
        self.runner.lr_scheduler.step()

        self.update_ema_decoder()

        with torch.no_grad():
            self.memory.update(image_ids, torch.sigmoid(preds.detach()))

        self.global_step += 1

        return loss_total

    def update_ema_decoder(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(
            self.runner.model.decoder_ema.parameters(),
            self.runner.model.decoder.parameters()
        ):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        for ema_buffer, buffer in zip(
            self.runner.model.decoder_ema.buffers(),
            self.runner.model.decoder.buffers()
        ):
            ema_buffer.data.copy_(buffer.data)

    def decide_to_finetune(self):
        if self._cur_epoch == self._max_epoch + self._start_finetune:
            self.finetune = True
            return True
        return False

    def decide_to_save(self) -> bool:
        return (self._cur_epoch >= self.save_start and 
                self._cur_epoch % self.save_interval == 0)

    def decide_to_val(self) -> bool:
        return (self.enable_val and 
                self._cur_epoch >= self.val_start and 
                self._cur_epoch % self.val_interval == 0)
