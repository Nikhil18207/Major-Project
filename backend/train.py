#!/usr/bin/env python
"""
XR2Text Training Script

Main entry point for training the XR2Text model.
Supports configuration via YAML files and command-line arguments.

Usage:
    # Activate virtual environment first
    # Windows: swin\\Scripts\\activate
    # Linux/Mac: source swin/bin/activate

    # Train with default config
    python train.py

    # Train with custom config
    python train.py --config configs/training_rtx4060.yaml

    # Resume training from checkpoint
    python train.py --resume checkpoints/checkpoint_epoch_10.pt
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.xr2text import XR2TextModel
from src.data.dataloader import get_dataloaders
from src.training.trainer import XR2TextTrainer
from src.utils.device import get_device, print_gpu_info, setup_cuda_optimizations
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge two config dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="Train XR2Text Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use subset of data for debugging",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config

    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Apply command-line overrides
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.no_amp:
        config.setdefault("training", {})["use_amp"] = False

    # Setup logging
    log_config = config.get("logging", {})
    setup_logger(
        log_dir=log_config.get("log_dir", "logs"),
        log_level=log_config.get("level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("XR2Text Training")
    logger.info("=" * 60)

    # Print GPU info
    print_gpu_info()
    setup_cuda_optimizations()

    # Create model
    logger.info("Creating model...")
    model_config = config.get("model", {})
    model = XR2TextModel.from_config(model_config)

    # Get tokenizer
    tokenizer = model.get_tokenizer()

    # Create data loaders
    logger.info("Creating data loaders...")
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=training_config.get("batch_size", 8),
        num_workers=data_config.get("num_workers", 4),
        image_size=data_config.get("image_size", 384),
        max_length=data_config.get("max_length", 256),
        target_type=data_config.get("target_type", "both"),
        train_subset=args.subset,
        val_subset=args.subset // 5 if args.subset else None,
        pin_memory=data_config.get("pin_memory", True),
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = XR2TextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    final_metrics = trainer.train()

    # Print final metrics
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value}")

    return final_metrics


if __name__ == "__main__":
    main()
