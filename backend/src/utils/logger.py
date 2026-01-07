"""
Logging Configuration for XR2Text

Sets up structured logging with Loguru for training and inference.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure the Loguru logger for XR2Text.

    Args:
        log_dir: Directory for log files (default: ./logs)
        log_level: Minimum log level
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        rotation: Log file rotation size
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()

    # Console format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # File format (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Add console handler
    if log_to_console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
        )

    # Add file handlers
    if log_to_file:
        log_dir = Path(log_dir or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        logger.add(
            log_dir / "xr2text_{time:YYYY-MM-DD}.log",
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        # Error log file (errors only)
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        # Training log file
        logger.add(
            log_dir / "training_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            compression="zip",
            filter=lambda record: "training" in record["extra"],
        )

    logger.info("Logger initialized")


class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides structured logging for training metrics and progress.
    """

    def __init__(self, experiment_name: str = "xr2text"):
        self.experiment_name = experiment_name
        self.logger = logger.bind(training=True)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.logger.info(f"Starting Epoch {epoch}/{total_epochs}")

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[dict] = None,
        lr: Optional[float] = None,
    ):
        """Log epoch summary."""
        msg = f"Epoch {epoch} complete | Train Loss: {train_loss:.4f}"

        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"

        if lr is not None:
            msg += f" | LR: {lr:.2e}"

        self.logger.info(msg)

        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Metrics: {metrics_str}")

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        lr: Optional[float] = None,
        log_every: int = 100,
    ):
        """Log batch progress."""
        if batch % log_every == 0 or batch == total_batches - 1:
            progress = 100 * batch / total_batches
            msg = f"Epoch {epoch} | Batch {batch}/{total_batches} ({progress:.1f}%) | Loss: {loss:.4f}"

            if lr is not None:
                msg += f" | LR: {lr:.2e}"

            self.logger.debug(msg)

    def log_checkpoint(self, path: str, epoch: int, metric: Optional[float] = None):
        """Log checkpoint save."""
        msg = f"Checkpoint saved: {path} (Epoch {epoch}"
        if metric is not None:
            msg += f", Metric: {metric:.4f}"
        msg += ")"
        self.logger.info(msg)

    def log_early_stop(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.warning(
            f"Early stopping triggered at epoch {epoch} (patience: {patience})"
        )

    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log new best model."""
        self.logger.success(
            f"New best model at epoch {epoch}! {metric_name}: {metric_value:.4f}"
        )


if __name__ == "__main__":
    # Test logger
    setup_logger(log_dir="./test_logs", log_level="DEBUG")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test training logger
    training_logger = TrainingLogger("test_experiment")
    training_logger.log_epoch_start(1, 10)
    training_logger.log_batch(1, 50, 100, 0.5432, lr=1e-4)
    training_logger.log_epoch_end(
        1,
        train_loss=0.5,
        val_loss=0.45,
        metrics={"bleu_4": 0.25, "rouge_l": 0.40},
        lr=1e-4,
    )
