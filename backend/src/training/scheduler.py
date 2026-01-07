"""
Learning Rate Schedulers for XR2Text Training

Implements various learning rate scheduling strategies
with warmup support.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
from typing import Optional


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 10000,
    num_warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Get learning rate scheduler with warmup.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("linear", "cosine", "cosine_restarts", "polynomial")
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum learning rate ratio (for cosine)
        num_cycles: Number of cycles for cosine with restarts

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )

    elif scheduler_type == "cosine_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

    elif scheduler_type == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=2.0,
        )

    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by linear decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay to min_lr_ratio.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1.0,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay with hard restarts.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * ((progress * num_cycles) % 1.0)))
        )

    return LambdaLR(optimizer, lr_lambda)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 2.0,
    lr_end: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup followed by polynomial decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        lr_range = 1.0 - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps

        return lr_range * pct_remaining ** power + lr_end

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by constant learning rate.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineScheduler:
    """
    Custom scheduler with warmup and cosine decay.

    Provides more control over the scheduling behavior.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = last_epoch + 1
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, [lr] * len(self.optimizer.param_groups)):
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (self.current_step / self.warmup_steps)

        # Cosine decay
        progress = (self.current_step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay

    def get_last_lr(self) -> list:
        """Get current learning rate."""
        return [self.get_lr()]


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt

    # Test schedulers
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_steps = 10000
    warmup_steps = 1000

    schedulers = {
        "linear": get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps),
        "cosine": get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps),
        "polynomial": get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, num_steps),
    }

    print("Scheduler test completed!")
