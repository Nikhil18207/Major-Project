"""
XR2Text Trainer Module with HAQT-ARR Support

Handles training loop, validation, checkpointing, and evaluation
for the XR2Text model with HAQT-ARR (Hierarchical Anatomical Query Tokens
with Adaptive Region Routing) projection layer.

Features:
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large effective batch sizes
- HAQT-ARR region weight tracking and visualization
- Anatomical attention monitoring
- Scheduled sampling for improved generation (Novel)
- Label smoothing for regularization
- Region weight regularization for balanced attention

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
# Fix: Use non-deprecated amp imports (PyTorch 2.4+)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Tuple, Any
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from ..models.xr2text import XR2TextModel
from ..utils.metrics import compute_metrics, MetricsTracker
from ..utils.device import (
    get_device, 
    setup_cuda_optimizations, 
    GPUMemoryMonitor,
    GPUTemperatureMonitor,
    check_gpu_health,
    get_gpu_temperature,
)
from ..utils.logger import TrainingLogger
from .scheduler import get_scheduler
from .losses import CombinedNovelLoss
from .curriculum import AnatomicalCurriculumScheduler, CurriculumDataset, create_curriculum_dataloader
from ..utils.clinical_validator import ClinicalValidator
from ..data.dataset import IGNORE_INDEX
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average for model parameters.

    IMPROVED: Provides more stable model weights for evaluation and final checkpoint.
    EMA helps smooth out noisy gradients and often leads to better generalization.

    Usage:
        ema = EMA(model, decay=0.999)
        # During training:
        ema.update()
        # For evaluation:
        ema.apply_shadow()  # Use EMA weights
        # Evaluate...
        ema.restore()  # Restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


class XR2TextTrainer:
    """
    Trainer class for XR2Text model with HAQT-ARR support.

    Handles the complete training pipeline including:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Evaluation
    - Early stopping
    - HAQT-ARR region weight tracking (Novel)

    Args:
        model: XR2TextModel instance (with or without HAQT-ARR)
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
    """

    def __init__(
        self,
        model: XR2TextModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ):
        self.config = config
        self.device = get_device()

        # Setup CUDA optimizations
        setup_cuda_optimizations()

        # Move model to device
        self.model = model.to(self.device)

        # Enable gradient checkpointing if configured (RTX 4060 memory optimization)
        if config.get("gradient_checkpointing", False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for memory efficiency")

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Store base dataset and loader config for curriculum learning
        self.base_train_dataset = train_loader.dataset
        self.train_batch_size = train_loader.batch_size
        self.train_num_workers = train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
        self.train_collate_fn = train_loader.collate_fn  # Store collate_fn for curriculum

        # Training settings
        self.epochs = config.get("epochs", 50)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("use_amp", True)

        # Early stopping
        self.patience = config.get("patience", 15)  # Increased from 5
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None  # Store best model in memory, save only at end

        # Label smoothing for regularization
        self.label_smoothing = config.get("label_smoothing", 0.1)
        
        # Scheduled sampling parameters
        self.use_scheduled_sampling = config.get("use_scheduled_sampling", True)
        self.ss_start = config.get("scheduled_sampling_start", 1.0)  # 100% teacher forcing initially
        self.ss_end = config.get("scheduled_sampling_end", 0.5)  # 50% at the end
        self.ss_warmup_epochs = config.get("scheduled_sampling_warmup", 5)  # Warmup before decay
        self.current_ss_ratio = self.ss_start
        
        # Region weight regularization (for balanced anatomical attention)
        self.use_region_regularization = config.get("use_region_regularization", True)
        self.region_reg_weight = config.get("region_regularization_weight", 0.01)
        
        # NOVEL: Combined novel loss functions
        self.use_novel_losses = config.get("use_novel_losses", True)
        if self.use_novel_losses:
            self.novel_loss = CombinedNovelLoss(
                use_anatomical_consistency=config.get("use_anatomical_consistency_loss", True),
                use_clinical_entity=config.get("use_clinical_entity_loss", True),
                use_region_focal=config.get("use_region_focal_loss", True),
                use_cross_modal=config.get("use_cross_modal_loss", False),  # Requires decoder hidden states
                anatomical_weight=config.get("anatomical_loss_weight", 0.1),
                clinical_weight=config.get("clinical_loss_weight", 0.2),
                focal_weight=config.get("focal_loss_weight", 0.15),
                alignment_weight=config.get("alignment_loss_weight", 0.1),
            )
            logger.info("Novel loss functions enabled")
        
        # NOVEL: Curriculum learning
        self.use_curriculum = config.get("use_curriculum_learning", True)
        if self.use_curriculum:
            # FIXED: Pass curriculum stages from config instead of using hardcoded defaults
            curriculum_stages = config.get("curriculum_stages", None)
            self.curriculum_scheduler = AnatomicalCurriculumScheduler(stages=curriculum_stages)
            logger.info("Curriculum learning enabled")
        
        # NOVEL: Clinical validation
        self.use_clinical_validation = config.get("use_clinical_validation", True)
        if self.use_clinical_validation:
            self.clinical_validator = ClinicalValidator()
            logger.info("Clinical validation enabled")

        # IMPROVED: R-Drop regularization for better generation
        self.use_rdrop = config.get("use_rdrop", False)  # Disabled by default for faster training
        self.rdrop_alpha = config.get("rdrop_alpha", 0.7)
        if self.use_rdrop:
            logger.info(f"R-Drop regularization enabled with alpha={self.rdrop_alpha}")

        # GPU Temperature Monitoring - controlled by config.enable_temp_monitoring
        # When enabled, training will pause if GPU temperature exceeds thresholds
        self.enable_temp_monitoring = config.get("enable_temp_monitoring", False)
        self.temp_monitor = None
        if self.enable_temp_monitoring:
            self.max_gpu_temp = config.get("max_gpu_temp", 83)
            self.warning_gpu_temp = config.get("warning_gpu_temp", 75)
            self.pause_gpu_temp = config.get("pause_gpu_temp", 80)
            self.temp_check_interval = config.get("temp_check_interval", 10)
            self.gpu_cooldown_time = config.get("gpu_cooldown_time", 30)
            logger.info(
                f"GPU temperature monitoring enabled: "
                f"pause at {self.pause_gpu_temp}°C, max {self.max_gpu_temp}°C"
            )

        # CUBLAS Error Recovery - retry on random CUDA crashes
        self.cublas_retry_enabled = config.get("cublas_retry_enabled", True)
        self.cublas_max_retries = config.get("cublas_max_retries", 3)
        self.cublas_retry_delay = config.get("cublas_retry_delay", 10)
        if self.cublas_retry_enabled:
            logger.info(f"CUBLAS error recovery enabled: {self.cublas_max_retries} retries with {self.cublas_retry_delay}s delay")

        # Checkpointing - ONLY best model saved at the end of training
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: Periodic checkpoints removed - best model tracked in memory and saved at end

        # Logging - reduced frequency to avoid cluttering output
        self.log_every = config.get("log_every", 500)
        self.training_logger = TrainingLogger(config.get("experiment_name", "xr2text"))

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        # FIX: Use FULL dataset size for total_steps calculation, not curriculum-filtered size
        # This prevents LR from decaying too fast when curriculum starts with small subsets
        # The scheduler should be based on the maximum number of steps across all epochs
        if self.use_curriculum and hasattr(self, 'base_train_dataset'):
            # Calculate based on full dataset (hard stage uses all samples)
            full_dataset_size = len(self.base_train_dataset)
            steps_per_epoch = full_dataset_size // self.train_batch_size
        else:
            steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.epochs // self.gradient_accumulation_steps
        logger.info(f"Scheduler initialized with total_steps={total_steps} (steps_per_epoch={steps_per_epoch})")

        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.get("scheduler", "cosine"),
            num_training_steps=total_steps,
            num_warmup_steps=self.warmup_steps,
            min_lr_ratio=config.get("min_lr_ratio", 0.1),
            num_cycles=config.get("num_cycles", 2),
        )

        # Mixed precision scaler (Fix: specify device for PyTorch 2.4+)
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # OOM Recovery settings - read from config (defaults match default.yaml)
        self.oom_recovery_enabled = True
        self.oom_batch_reduction = 0.5  # Reduce batch by 50% on OOM
        self.max_oom_retries = config.get("max_oom_retries", 10)  # Max OOM errors before stopping
        self.clear_cache_every_steps = config.get("clear_cache_every_steps", 5)  # Clear cache every N steps
        logger.info(f"OOM recovery enabled: max {self.max_oom_retries} retries, cache clear every {self.clear_cache_every_steps} steps")

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # State
        self.current_epoch = 0
        self.global_step = 0

        # IMPROVED: EMA (Exponential Moving Average) for stable training
        self.use_ema = config.get("use_ema", True)
        self.ema_decay = config.get("ema_decay", 0.9999)
        self.ema = None
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.ema_decay)
            logger.info(f"EMA enabled with decay={self.ema_decay}")

        # HAQT-ARR tracking
        self.use_anatomical_attention = getattr(model, 'use_anatomical_attention', False)
        if self.use_anatomical_attention:
            self.region_names = model.get_anatomical_regions()
            self.region_weight_history = []
            logger.info(f"HAQT-ARR enabled with regions: {self.region_names}")

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Training for {self.epochs} epochs")
        logger.info(f"Total optimization steps: {total_steps}")
        logger.info(f"Label smoothing: {self.label_smoothing}")
        logger.info(f"Scheduled sampling: {self.use_scheduled_sampling} (start={self.ss_start}, end={self.ss_end})")
        logger.info(f"Region regularization: {self.use_region_regularization} (weight={self.region_reg_weight})")
        logger.info(f"Early stopping patience: {self.patience}")

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with layer-wise learning rates for better convergence."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        # Layer-wise learning rates (IMPROVED for faster convergence)
        encoder_lr = self.config.get("encoder_lr", self.learning_rate * 0.5)
        decoder_lr = self.config.get("decoder_lr", self.learning_rate)
        projection_lr = self.config.get("projection_lr", self.learning_rate * 1.5)

        optimizer_grouped_parameters = []
        assigned_params = set()  # Track assigned parameters to avoid duplicates

        # Helper to check if parameter is already assigned
        def is_assigned(p):
            return id(p) in assigned_params

        def mark_assigned(params):
            for p in params:
                assigned_params.add(id(p))

        # 1. Projection/HAQT-ARR parameters FIRST (highest priority - new layers)
        # These get highest learning rate
        projection_params_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p)
            and ("projection" in n or "haqt" in n.lower() or "anatomical" in n.lower())
            and not any(nd in n for nd in no_decay)
        ]
        projection_params_no_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p)
            and ("projection" in n or "haqt" in n.lower() or "anatomical" in n.lower())
            and any(nd in n for nd in no_decay)
        ]

        if projection_params_decay:
            mark_assigned(projection_params_decay)
            optimizer_grouped_parameters.append({
                "params": projection_params_decay,
                "lr": projection_lr,
                "weight_decay": self.weight_decay,
                "name": "projection_decay"
            })
        if projection_params_no_decay:
            mark_assigned(projection_params_no_decay)
            optimizer_grouped_parameters.append({
                "params": projection_params_no_decay,
                "lr": projection_lr,
                "weight_decay": 0.0,
                "name": "projection_no_decay"
            })

        # 2. Encoder parameters (lower learning rate - pretrained)
        encoder_params_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p)
            and "encoder" in n and not any(nd in n for nd in no_decay)
        ]
        encoder_params_no_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p)
            and "encoder" in n and any(nd in n for nd in no_decay)
        ]

        if encoder_params_decay:
            mark_assigned(encoder_params_decay)
            optimizer_grouped_parameters.append({
                "params": encoder_params_decay,
                "lr": encoder_lr,
                "weight_decay": self.weight_decay,
                "name": "encoder_decay"
            })
        if encoder_params_no_decay:
            mark_assigned(encoder_params_no_decay)
            optimizer_grouped_parameters.append({
                "params": encoder_params_no_decay,
                "lr": encoder_lr,
                "weight_decay": 0.0,
                "name": "encoder_no_decay"
            })

        # 3. Decoder parameters (standard learning rate)
        decoder_params_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p)
            and "decoder" in n and not any(nd in n for nd in no_decay)
        ]
        decoder_params_no_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p)
            and "decoder" in n and any(nd in n for nd in no_decay)
        ]

        if decoder_params_decay:
            mark_assigned(decoder_params_decay)
            optimizer_grouped_parameters.append({
                "params": decoder_params_decay,
                "lr": decoder_lr,
                "weight_decay": self.weight_decay,
                "name": "decoder_decay"
            })
        if decoder_params_no_decay:
            mark_assigned(decoder_params_no_decay)
            optimizer_grouped_parameters.append({
                "params": decoder_params_no_decay,
                "lr": decoder_lr,
                "weight_decay": 0.0,
                "name": "decoder_no_decay"
            })

        # 4. Any remaining parameters (fallback)
        other_params_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p) and not any(nd in n for nd in no_decay)
        ]
        other_params_no_decay = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not is_assigned(p) and any(nd in n for nd in no_decay)
        ]

        if other_params_decay:
            mark_assigned(other_params_decay)
            optimizer_grouped_parameters.append({
                "params": other_params_decay,
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
                "name": "other_decay"
            })
        if other_params_no_decay:
            mark_assigned(other_params_no_decay)
            optimizer_grouped_parameters.append({
                "params": other_params_no_decay,
                "lr": self.learning_rate,
                "weight_decay": 0.0,
                "name": "other_no_decay"
            })

        # Log parameter groups
        total_params = 0
        for group in optimizer_grouped_parameters:
            n_params = sum(p.numel() for p in group["params"])
            total_params += n_params
            logger.info(f"Optimizer group '{group.get('name', 'unnamed')}': {n_params:,} params, lr={group['lr']:.2e}")
        logger.info(f"Total trainable parameters: {total_params:,}")

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _update_scheduled_sampling_ratio(self, epoch: int) -> None:
        """Update scheduled sampling ratio based on current epoch."""
        if not self.use_scheduled_sampling:
            self.current_ss_ratio = 1.0
            return

        if epoch < self.ss_warmup_epochs:
            self.current_ss_ratio = self.ss_start
        else:
            decay_epochs = self.epochs - self.ss_warmup_epochs
            progress = (epoch - self.ss_warmup_epochs) / max(decay_epochs, 1)
            self.current_ss_ratio = self.ss_start - (self.ss_start - self.ss_end) * progress
            self.current_ss_ratio = max(self.ss_end, self.current_ss_ratio)

    def _update_focal_gamma(self, epoch: int) -> None:
        """
        IMPROVED: Update focal loss gamma with warmup.

        Gradually increases focal gamma from 0 to target value during warmup.
        This prevents early training instability when the model hasn't learned
        basic patterns yet.

        Args:
            epoch: Current training epoch
        """
        # Get focal loss config from decoder section
        decoder_config = self.config.get("decoder", {})
        target_gamma = decoder_config.get("focal_gamma", 2.0)
        warmup_epochs = decoder_config.get("focal_gamma_warmup_epochs", 5)

        if warmup_epochs <= 0:
            # No warmup, use full gamma immediately
            current_gamma = target_gamma
        elif epoch < warmup_epochs:
            # Linear warmup from 0 to target_gamma
            current_gamma = target_gamma * (epoch / warmup_epochs)
        else:
            # After warmup, use full gamma
            current_gamma = target_gamma

        # Apply to decoder
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'set_focal_gamma'):
            self.model.decoder.set_focal_gamma(current_gamma)
            if epoch == 0 or (epoch == warmup_epochs and warmup_epochs > 0):
                logger.info(f"Focal gamma: {current_gamma:.2f} (target: {target_gamma})")

    def _compute_region_regularization_loss(self, region_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute region weight regularization loss to encourage balanced attention.
        
        Uses entropy maximization to prevent single region dominance.
        
        Args:
            region_weights: Tensor of shape (batch, num_regions)
            
        Returns:
            Regularization loss (scalar)
        """
        if region_weights is None or not self.use_region_regularization:
            return torch.tensor(0.0, device=self.device)
        
        # Normalize to ensure valid probability distribution
        weights = region_weights / (region_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        
        # Maximum entropy for num_regions uniform distribution
        num_regions = region_weights.shape[-1]
        max_entropy = torch.log(torch.tensor(float(num_regions), device=self.device))
        
        # Regularization loss = negative entropy (we want to maximize entropy)
        # Normalize by max entropy so loss is in [0, 1] range
        reg_loss = (max_entropy - entropy) / max_entropy

        return self.region_reg_weight * reg_loss

    def _compute_rdrop_loss(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        IMPROVED: R-Drop Regularization Loss (with proper error handling)

        R-Drop (Regularized Dropout) performs two forward passes with dropout
        and minimizes the bidirectional KL divergence between the two output
        distributions. This regularization significantly improves NLG tasks.

        Paper: "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)

        Args:
            logits1: Logits from first forward pass (B, seq_len, vocab_size)
            logits2: Logits from second forward pass (B, seq_len, vocab_size)
            labels: Target labels for masking padding (B, seq_len)

        Returns:
            R-Drop KL divergence loss
        """
        try:
            # Create mask for non-padding positions
            mask = (labels != IGNORE_INDEX).float()

            # Check if mask has any valid positions
            mask_sum = mask.sum()
            if mask_sum == 0:
                return torch.tensor(0.0, device=logits1.device, requires_grad=True)

            # Compute log probabilities with numerical stability
            log_probs1 = F.log_softmax(logits1, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)

            # Bidirectional KL divergence: KL(p1||p2) + KL(p2||p1)
            # Use detach to prevent double backprop through probs
            kl_loss1 = F.kl_div(log_probs1, probs2.detach(), reduction='none').sum(dim=-1)
            kl_loss2 = F.kl_div(log_probs2, probs1.detach(), reduction='none').sum(dim=-1)

            # Apply mask and average
            kl_loss = (kl_loss1 + kl_loss2) * mask
            kl_loss = kl_loss.sum() / (mask_sum + 1e-8)

            # Clamp to prevent NaN/Inf
            kl_loss = torch.clamp(kl_loss, min=0.0, max=100.0)

            return self.rdrop_alpha * kl_loss

        except Exception as e:
            logger.warning(f"R-Drop loss computation failed: {e}, returning 0")
            return torch.tensor(0.0, device=logits1.device, requires_grad=True)

    def train(self, start_epoch: int = 0) -> Dict[str, float]:
        """
        Run the complete training loop.

        Args:
            start_epoch: Starting epoch (for resume functionality)

        Returns:
            Dictionary with final metrics
        """
        logger.info("Starting training...")

        # Support start_epoch parameter for resume functionality
        if start_epoch > 0:
            self.current_epoch = start_epoch
            logger.info(f"Resuming from epoch {start_epoch}")

        # Track previous curriculum stage for transition detection
        prev_stage_name = None

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # NOVEL: Update curriculum learning - filter samples based on epoch
            if self.use_curriculum:
                stage_name = self.curriculum_scheduler.get_stage_name(epoch)

                # CRITICAL: Aggressive memory cleanup on curriculum stage transitions
                # This prevents OOM when transitioning from easy (9k samples) to medium (27k samples)
                if prev_stage_name is not None and stage_name != prev_stage_name:
                    logger.info(f"Curriculum stage transition: {prev_stage_name} -> {stage_name}")
                    logger.info("Performing aggressive memory cleanup before stage transition...")

                    # FIX: Delete old curriculum dataset wrapper explicitly to prevent memory leak
                    # The CurriculumDataset holds references that can prevent garbage collection
                    if hasattr(self, '_current_curriculum_dataset'):
                        del self._current_curriculum_dataset
                        self._current_curriculum_dataset = None

                    # Delete old dataloader to free memory
                    if hasattr(self, 'train_loader'):
                        del self.train_loader

                    # Aggressive cleanup sequence
                    self.optimizer.zero_grad(set_to_none=True)
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # Force a full garbage collection cycle
                        gc.collect()
                        torch.cuda.empty_cache()

                        # Log memory state after cleanup
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"GPU memory after cleanup - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

                    # Brief pause to let GPU stabilize
                    time.sleep(2)

                prev_stage_name = stage_name

                # FIX: Store reference to curriculum dataset for proper cleanup on next transition
                self._current_curriculum_dataset = CurriculumDataset(
                    base_dataset=self.base_train_dataset,
                    curriculum_scheduler=self.curriculum_scheduler,
                    current_epoch=epoch,
                )
                self.train_loader = DataLoader(
                    self._current_curriculum_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=self.train_num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=self.train_collate_fn,  # Use original collate function
                )
                logger.info(f"Curriculum stage: {stage_name} ({len(self._current_curriculum_dataset)}/{len(self.base_train_dataset)} samples)")

            # Update scheduled sampling ratio (silent)
            self._update_scheduled_sampling_ratio(epoch)

            # IMPROVED: Update focal loss gamma with warmup
            self._update_focal_gamma(epoch)

            # Training phase
            train_loss = self._train_epoch()

            # Validation phase - configurable frequency (default every 2 epochs)
            validate_every = self.config.get('validate_every', 2)
            if (epoch + 1) % validate_every == 0 or (epoch + 1) == self.epochs:
                val_loss, val_metrics = self._validate()

                # Log epoch summary (single line)
                current_lr = self.scheduler.get_last_lr()[0]
                bleu4 = val_metrics.get('bleu_4', 0)
                rougel = val_metrics.get('rouge_l', 0)
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"BLEU-4: {bleu4:.4f} | ROUGE-L: {rougel:.4f}"
                )

                # Track metrics
                self.metrics_tracker.update(val_metrics, epoch + 1)
                self.metrics_tracker.update({"train_loss": train_loss, "val_loss": val_loss}, epoch + 1)
            else:
                # Skip validation, just log training loss
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train: {train_loss:.4f} | Val: SKIPPED (validating every {validate_every} epochs)"
                )
                val_metrics = {}
                val_loss = 0.0

            # Check for improvement (only when we have validation metrics)
            current_metric = val_metrics.get("bleu_4", 0) + val_metrics.get("rouge_l", 0)

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch + 1
                self.patience_counter = 0

                # Store best model state in memory (NO DISK SAVE during training)
                # We'll save only at the very end after all 50 epochs
                logger.info(f"New best model at epoch {epoch + 1}: BLEU-4 + ROUGE-L = {current_metric:.4f}")
                self.best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_metric': self.best_metric,
                    'metrics': val_metrics.copy(),
                }
                self.training_logger.log_best_model(epoch + 1, "BLEU-4 + ROUGE-L", current_metric)
            else:
                self.patience_counter += 1

            # NOTE: No mid-training checkpoints - only best_model.pt saved at end of training

            # Early stopping
            if self.patience_counter >= self.patience:
                self.training_logger.log_early_stop(epoch + 1, self.patience)
                break

            # Clear CUDA cache periodically to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Training complete - NOW save the best model to disk
        logger.info(f"Training completed! Saving best model from epoch {self.best_epoch}...")
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(self.best_model_state, checkpoint_path)
            logger.info(f"Best model saved to {checkpoint_path}")
            logger.info(f"Best epoch: {self.best_epoch} | BLEU-4 + ROUGE-L = {self.best_metric:.4f}")
        else:
            logger.warning("No best model state found - saving current model")
            self._save_checkpoint("best_model.pt", is_best=True)
        return self._get_final_metrics()

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        # Calculate number of optimization steps (accounts for gradient accumulation)
        num_optimization_steps = num_batches // self.gradient_accumulation_steps
        if num_batches % self.gradient_accumulation_steps != 0:
            num_optimization_steps += 1

        progress_bar = tqdm(
            total=num_optimization_steps,
            desc=f"Epoch {self.current_epoch + 1}",
            unit="step",
        )

        self.optimizer.zero_grad()

        oom_count = 0  # Track OOM occurrences
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move batch to device
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass with mixed precision (Fix: use device_type for PyTorch 2.4+)
                with autocast('cuda', enabled=self.use_amp):
                    # For BART: shift labels to create decoder_input_ids
                    # decoder_input_ids should be labels shifted right with decoder_start_token_id
                    decoder_start_token_id = self.model.decoder.model.config.decoder_start_token_id
                    if decoder_start_token_id is None:
                        decoder_start_token_id = self.model.decoder.bos_token_id

                    # Create shifted decoder input ids
                    shifted_input_ids = labels.new_zeros(labels.shape)
                    shifted_input_ids[:, 1:] = labels[:, :-1].clone()
                    shifted_input_ids[:, 0] = decoder_start_token_id

                    # Replace IGNORE_INDEX with pad_token_id in decoder_input_ids
                    shifted_input_ids = shifted_input_ids.masked_fill(
                        shifted_input_ids == IGNORE_INDEX,
                        self.model.decoder.pad_token_id
                    )

                    # First forward pass
                    outputs = self.model(
                        images=images,
                        decoder_input_ids=shifted_input_ids,
                        decoder_attention_mask=attention_mask,
                        labels=labels,
                    )

                    # Main loss (with label smoothing applied in model)
                    loss = outputs["loss"]

                    # IMPROVED: R-Drop regularization - second forward pass
                    if self.use_rdrop and self.model.training:
                        outputs2 = self.model(
                            images=images,
                            decoder_input_ids=shifted_input_ids,
                            decoder_attention_mask=attention_mask,
                            labels=labels,
                        )
                        # Average the two losses
                        loss = (loss + outputs2["loss"]) / 2.0
                        # Add R-Drop KL divergence loss
                        rdrop_loss = self._compute_rdrop_loss(
                            outputs["logits"],
                            outputs2["logits"],
                            labels,
                        )
                        loss = loss + rdrop_loss

                    # Add region regularization loss for balanced anatomical attention
                    if self.use_anatomical_attention and outputs.get("region_weights") is not None:
                        reg_loss = self._compute_region_regularization_loss(outputs["region_weights"])
                        loss = loss + reg_loss

                    # NOVEL: Add combined novel losses
                    if self.use_novel_losses:
                        novel_losses = self.novel_loss(
                            outputs=outputs,
                            labels=labels,
                            pad_token_id=self.model.decoder.pad_token_id,
                        )
                        total_novel_loss = novel_losses.get("total_novel", torch.tensor(0.0, device=self.device))
                        loss = loss + total_novel_loss

                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps

                # Track HAQT-ARR region weights (periodically)
                if self.use_anatomical_attention and batch_idx % 50 == 0:
                    if outputs.get("region_weights") is not None:
                        region_weights = outputs["region_weights"].mean(dim=0).detach().cpu().numpy()
                        self.region_weight_history.append(region_weights)

            except RuntimeError as e:
                error_str = str(e).lower()

                # OOM Recovery - more robust handling with graceful degradation
                if "out of memory" in error_str and self.oom_recovery_enabled:
                    oom_count += 1
                    logger.warning(f"OOM at batch {batch_idx}, attempting recovery (OOM count: {oom_count})")

                    # Aggressive memory cleanup sequence
                    try:
                        # Step 1: Clear gradients aggressively
                        self.optimizer.zero_grad(set_to_none=True)

                        # Step 2: Delete any cached tensors
                        if 'images' in dir():
                            del images
                        if 'input_ids' in dir():
                            del input_ids
                        if 'attention_mask' in dir():
                            del attention_mask
                        if 'labels' in dir():
                            del labels
                        if 'outputs' in dir():
                            del outputs

                        # Step 3: Force garbage collection
                        gc.collect()

                        # Step 4: CUDA cleanup
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                            # Step 5: Reset CUDA memory stats
                            torch.cuda.reset_peak_memory_stats()

                            # Log memory after cleanup
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            logger.info(f"GPU memory after OOM cleanup: {allocated:.2f} GB")

                        # Step 6: Progressive wait based on OOM count
                        wait_time = min(5 * oom_count, 60)  # 5s, 10s, 15s... up to 60s
                        logger.warning(f"Waiting {wait_time}s for GPU to stabilize...")
                        time.sleep(wait_time)

                        # Additional cleanup after wait
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as cache_error:
                        logger.error(f"Cache clearing failed: {cache_error}")
                        # Even if cleanup fails, try to continue with a longer wait
                        time.sleep(60)
                        gc.collect()

                    # Skip this batch if OOM count is reasonable
                    if oom_count <= self.max_oom_retries:
                        logger.info(f"Skipping batch {batch_idx} and continuing...")
                        continue
                    else:
                        # Save emergency checkpoint before failing
                        logger.error(f"Too many OOMs ({oom_count}/{self.max_oom_retries})")
                        try:
                            self._save_checkpoint(f"emergency_checkpoint_epoch_{self.current_epoch+1}_oom.pt")
                            logger.info("Emergency checkpoint saved")
                        except:
                            pass
                        raise e

                # CUBLAS Error Recovery - retry on CUDA crashes
                elif ("cublas" in error_str or "cuda error" in error_str) and self.cublas_retry_enabled:
                    if not hasattr(self, '_cublas_retry_count'):
                        self._cublas_retry_count = 0

                    self._cublas_retry_count += 1

                    if self._cublas_retry_count <= self.cublas_max_retries:
                        logger.warning(
                            f"CUBLAS/CUDA error at batch {batch_idx}. "
                            f"Retry {self._cublas_retry_count}/{self.cublas_max_retries}. "
                            f"Waiting {self.cublas_retry_delay}s..."
                        )

                        # Clear GPU memory and wait
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()

                        time.sleep(self.cublas_retry_delay)

                        # Zero gradients and retry
                        self.optimizer.zero_grad()
                        continue
                    else:
                        logger.error(f"CUBLAS error persists after {self.cublas_max_retries} retries. Failing.")
                        self._cublas_retry_count = 0
                        raise e
                else:
                    raise e

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # IMPROVED: Update EMA after each optimization step
                if self.use_ema and self.ema is not None:
                    self.ema.update()

                # Reset CUBLAS retry counter on successful step
                if hasattr(self, '_cublas_retry_count'):
                    self._cublas_retry_count = 0

                # Periodic cache clearing to prevent memory fragmentation
                if self.global_step % self.clear_cache_every_steps == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                # GPU Temperature Check - pause if overheating
                if self.enable_temp_monitoring and self.global_step % self.temp_check_interval == 0:
                    current_temp = get_gpu_temperature()
                    if current_temp is not None:
                        if current_temp >= self.max_gpu_temp:
                            logger.error(f"GPU temp {current_temp}°C >= max {self.max_gpu_temp}°C! Stopping.")
                            raise RuntimeError(f"GPU overheated: {current_temp}°C")
                        elif current_temp >= self.pause_gpu_temp:
                            logger.warning(f"GPU temp {current_temp}°C >= pause threshold. Cooling down for {self.gpu_cooldown_time}s...")
                            torch.cuda.empty_cache()
                            time.sleep(self.gpu_cooldown_time)
                        elif current_temp >= self.warning_gpu_temp:
                            logger.warning(f"GPU temp {current_temp}°C - approaching limit")

                # Update progress bar only on optimization steps
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                })

        progress_bar.close()

        # Clear cache after training epoch to free memory for validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self) -> tuple:
        """Run validation and compute metrics."""
        # IMPROVED: Use EMA weights for validation (more stable)
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []

        # Use 25% of validation data for speed (configurable)
        val_fraction = self.config.get('val_fraction', 0.25)
        max_val_batches = max(1, int(len(self.val_loader) * val_fraction))

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            total=max_val_batches,
        )

        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= max_val_batches:
                break
            # Move batch to device
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            raw_texts = batch["raw_texts"]

            # Forward pass (Fix: use device_type for PyTorch 2.4+)
            with autocast('cuda', enabled=self.use_amp):
                # Create shifted decoder input ids (same as in training)
                decoder_start_token_id = self.model.decoder.model.config.decoder_start_token_id
                if decoder_start_token_id is None:
                    decoder_start_token_id = self.model.decoder.bos_token_id
                
                shifted_input_ids = labels.new_zeros(labels.shape)
                shifted_input_ids[:, 1:] = labels[:, :-1].clone()
                shifted_input_ids[:, 0] = decoder_start_token_id
                shifted_input_ids = shifted_input_ids.masked_fill(
                    shifted_input_ids == IGNORE_INDEX,
                    self.model.decoder.pad_token_id
                )
                
                outputs = self.model(
                    images=images,
                    decoder_input_ids=shifted_input_ids,
                    decoder_attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs["loss"].item()

            # Generate predictions for metrics - OPTIMIZED for faster validation
            # Use fewer beams during training validation (2 instead of 5)
            # Full beam search can be used during final evaluation
            gen_config = self.config.get('generation', {})
            val_num_beams = gen_config.get('val_num_beams', 2)  # Faster validation

            # NEW: Diverse beam search support (memory-safe)
            use_diverse = gen_config.get('use_diverse_beam_search', False)
            num_beam_groups = gen_config.get('num_beam_groups', 1) if use_diverse else 1
            diversity_penalty = gen_config.get('diversity_penalty', 0.0) if use_diverse else 0.0

            _, generated_texts, _ = self.model.generate(
                images=images,
                max_length=gen_config.get('max_length', 256),
                min_length=gen_config.get('min_length', 20),
                num_beams=val_num_beams,  # Use fewer beams for speed
                length_penalty=gen_config.get('length_penalty', 1.0),
                repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
                early_stopping=True,  # Stop when all beams finish
                # NEW: Diverse beam search (memory-safe, ~5% more VRAM)
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
            )

            all_predictions.extend(generated_texts)
            all_references.extend(raw_texts)

        progress_bar.close()

        # Clear cache after validation generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute metrics (use actual number of batches processed)
        val_loss = total_loss / max_val_batches
        metrics = compute_metrics(all_predictions, all_references, include_all=False)
        metrics["val_loss"] = val_loss
        
        # NOVEL: Clinical validation
        if self.use_clinical_validation:
            clinical_results = self.clinical_validator.batch_validate(
                all_predictions,
                all_references,
            )
            metrics["clinical_accuracy"] = clinical_results["average_clinical_accuracy"]
            metrics["clinical_f1"] = clinical_results["average_f1"]
            metrics["clinical_precision"] = clinical_results["average_precision"]
            metrics["clinical_recall"] = clinical_results["average_recall"]
            metrics["critical_errors"] = clinical_results["total_critical_errors"]

        # IMPROVED: Restore original weights after validation
        if self.use_ema and self.ema is not None:
            self.ema.restore()

        return val_loss, metrics

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint with robust permission handling for RunPod."""
        # Ensure checkpoint directory exists with proper permissions
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.chmod(self.checkpoint_dir, 0o777)
        except:
            pass  # Ignore permission errors on some systems

        checkpoint_path = self.checkpoint_dir / filename

        # FIX: Save epoch_completed flag to handle resume edge case
        # If training crashes at epoch boundary, we know this epoch was completed
        checkpoint = {
            "epoch": self.current_epoch,
            "epoch_completed": True,  # FIX: Track if epoch was fully completed
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "ema_state_dict": self.ema.state_dict() if self.use_ema and self.ema else None,  # IMPROVED: Save EMA
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,  # FIX: Also save patience counter
            "config": self.config,
        }

        # Save with retry logic for RunPod permission issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                torch.save(checkpoint, checkpoint_path)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Permission error saving checkpoint (attempt {attempt+1}), retrying...")
                    time.sleep(1)
                    try:
                        os.chmod(self.checkpoint_dir, 0o777)
                    except:
                        pass
                else:
                    logger.error(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                    raise

        self.training_logger.log_checkpoint(
            str(checkpoint_path),
            self.current_epoch + 1,
            self.best_metric if is_best else None,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint with proper resume handling.

        FIX: Properly handles the edge case where training crashes at epoch boundary.
        Uses epoch_completed flag to determine correct resume epoch.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # IMPROVED: Restore EMA state if available
        if self.use_ema and self.ema and checkpoint.get("ema_state_dict"):
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Restored EMA state from checkpoint")

        # FIX: Properly determine resume epoch based on completion status
        saved_epoch = checkpoint["epoch"]
        epoch_completed = checkpoint.get("epoch_completed", True)  # Default True for backwards compatibility

        if epoch_completed:
            # Epoch was completed, start from next epoch
            self.current_epoch = saved_epoch + 1
        else:
            # Epoch was not completed, resume from same epoch
            self.current_epoch = saved_epoch
            logger.warning(f"Resuming incomplete epoch {saved_epoch}")

        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric", 0.0)

        # FIX: Restore patience counter if available
        if "patience_counter" in checkpoint:
            self.patience_counter = checkpoint["patience_counter"]
            logger.info(f"Restored patience counter: {self.patience_counter}/{self.patience}")

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch} (saved at epoch {saved_epoch}, completed: {epoch_completed})")

    def evaluate(
        self,
        test_loader: DataLoader,
        generation_config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            generation_config: Optional generation config for better inference

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Use provided generation config or defaults
        gen_config = generation_config or {}

        all_predictions = []
        all_references = []
        total_loss = 0.0

        logger.info("Running final evaluation on test set...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                raw_texts = batch.get("raw_text", [""] * len(images))

                # Generate text using model's generate method
                generated_texts = self.model.generate(
                    images,
                    max_length=gen_config.get('max_length', 256),
                    min_length=gen_config.get('min_length', 30),
                    num_beams=gen_config.get('num_beams', 4),
                    length_penalty=gen_config.get('length_penalty', 1.2),
                    repetition_penalty=gen_config.get('repetition_penalty', 1.5),
                    no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 4),
                    early_stopping=gen_config.get('early_stopping', True),
                )

                all_predictions.extend(generated_texts)
                all_references.extend(raw_texts)

        # Compute all metrics
        metrics = compute_metrics(all_predictions, all_references, include_all=True)

        # Clinical validation
        if self.use_clinical_validation:
            clinical_results = self.clinical_validator.batch_validate(
                all_predictions,
                all_references,
            )
            metrics["clinical_accuracy"] = clinical_results["average_clinical_accuracy"]
            metrics["clinical_f1"] = clinical_results["average_f1"]
            metrics["clinical_precision"] = clinical_results["average_precision"]
            metrics["clinical_recall"] = clinical_results["average_recall"]

        return metrics

    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        best_bleu, bleu_epoch = self.metrics_tracker.get_best("bleu_4")
        best_rouge, rouge_epoch = self.metrics_tracker.get_best("rouge_l")

        metrics = {
            "best_bleu_4": best_bleu,
            "best_bleu_4_epoch": bleu_epoch,
            "best_rouge_l": best_rouge,
            "best_rouge_l_epoch": rouge_epoch,
            "final_epoch": self.current_epoch + 1,
            "total_steps": self.global_step,
        }

        # Add HAQT-ARR region weight summary
        if self.use_anatomical_attention and len(self.region_weight_history) > 0:
            import numpy as np
            region_weights_array = np.array(self.region_weight_history)
            mean_weights = region_weights_array.mean(axis=0)
            for i, region in enumerate(self.region_names):
                metrics[f"avg_weight_{region}"] = float(mean_weights[i])

        return metrics


def train_xr2text(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: Optional[XR2TextModel] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, float]:
    """
    Convenience function to train XR2Text model.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Optional pre-created model
        resume_from: Optional checkpoint path to resume from

    Returns:
        Final training metrics
    """
    # Create model if not provided
    if model is None:
        model = XR2TextModel.from_config(config.get("model", {}))

    # Create trainer
    trainer = XR2TextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume from checkpoint if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)

    # Train
    return trainer.train()


if __name__ == "__main__":
    # Test trainer setup
    from ..models.xr2text import DEFAULT_CONFIG

    print("Testing trainer initialization...")

    config = {
        "epochs": 2,
        "learning_rate": 1e-4,
        "use_amp": True,
        "model": DEFAULT_CONFIG,
    }

    # This would need actual data loaders to run
    print("Trainer module loaded successfully!")
