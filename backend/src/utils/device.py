"""
Device and GPU Utilities

Handles device selection, GPU information, and CUDA optimization
for RTX 4060 GPU.
"""

import torch
from typing import Optional
from loguru import logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for computation.

    Prefers CUDA if available, otherwise falls back to CPU.

    Args:
        prefer_cuda: Whether to prefer CUDA over CPU

    Returns:
        torch.device instance
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def print_gpu_info():
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    print("\n" + "=" * 50)
    print("GPU INFORMATION")
    print("=" * 50)

    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")

        # Memory information
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(i) / (1024**3)

        print(f"Total Memory: {total_memory:.2f} GB")
        print(f"Allocated Memory: {allocated_memory:.2f} GB")
        print(f"Cached Memory: {cached_memory:.2f} GB")
        print(f"Free Memory: {total_memory - allocated_memory:.2f} GB")

        # Compute capability
        major, minor = torch.cuda.get_device_capability(i)
        print(f"Compute Capability: {major}.{minor}")

    # CUDA version
    print(f"\nCUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 50 + "\n")


def setup_cuda_optimizations():
    """
    Set up CUDA optimizations for better performance.

    Optimized for RTX 4060 GPU.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping optimizations")
        return

    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    logger.info("Enabled cuDNN benchmark mode")

    # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
    # RTX 4060 supports TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("Enabled TF32 for matrix operations")

    # Set memory allocation strategy
    torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache")


def get_optimal_batch_size(
    model: torch.nn.Module,
    image_size: int = 384,
    min_batch: int = 1,
    max_batch: int = 32,
) -> int:
    """
    Find optimal batch size that fits in GPU memory.

    Uses binary search to find the largest batch size that works.

    Args:
        model: The model to test
        image_size: Input image size
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return min_batch

    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    optimal_batch = min_batch

    for batch_size in range(min_batch, max_batch + 1, 2):
        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Try forward pass
            with torch.no_grad():
                dummy_input = torch.randn(
                    batch_size, 3, image_size, image_size,
                    device=device
                )
                _ = model.encoder(dummy_input)

            optimal_batch = batch_size
            logger.debug(f"Batch size {batch_size} works")

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.debug(f"Batch size {batch_size} OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    logger.info(f"Optimal batch size: {optimal_batch}")
    return optimal_batch


def memory_efficient_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """
    Memory-efficient forward pass using gradient checkpointing.

    Processes inputs in chunks to reduce memory usage.

    Args:
        model: Model to run
        inputs: Input tensor
        chunk_size: Size of each chunk

    Returns:
        Concatenated outputs
    """
    outputs = []
    for i in range(0, inputs.size(0), chunk_size):
        chunk = inputs[i:i + chunk_size]
        with torch.no_grad():
            output = model(chunk)
        outputs.append(output)
        torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0)


class GPUMemoryMonitor:
    """Context manager for monitoring GPU memory usage."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_memory = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            diff = (end_memory - self.start_memory) / (1024**2)
            logger.debug(f"[{self.name}] Memory change: {diff:+.2f} MB")


def get_gpu_temperature(device_id: int = 0) -> Optional[int]:
    """
    Get GPU temperature in Celsius.

    Args:
        device_id: GPU device ID

    Returns:
        Temperature in Celsius, or None if unavailable
    """
    if not torch.cuda.is_available():
        return None

    # Method 1: Try pynvml (most reliable for NVIDIA GPUs)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return temp
    except Exception:
        pass

    # Method 2: Try nvidia-smi command (fallback)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits', f'--id={device_id}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            temp = int(result.stdout.strip())
            return temp
    except Exception:
        pass

    # Temperature monitoring not available - this is fine, training will continue
    return None


# Default temperature thresholds (can be overridden via config)
DEFAULT_MAX_TEMP = 95  # RTX 4060 max safe temp
DEFAULT_WARNING_TEMP = 85
DEFAULT_PAUSE_TEMP = 90

# GPU-specific temperature presets
GPU_TEMP_PRESETS = {
    'rtx_4060': {'max': 95, 'warning': 85, 'pause': 90},  # RTX 4060 can handle up to 95°C
    'rtx_4070': {'max': 95, 'warning': 85, 'pause': 90},
    'rtx_4080': {'max': 95, 'warning': 85, 'pause': 90},
    'rtx_4090': {'max': 95, 'warning': 85, 'pause': 90},
    'rtx_3080': {'max': 93, 'warning': 83, 'pause': 88},
    'rtx_3090': {'max': 93, 'warning': 83, 'pause': 88},
    'default': {'max': 95, 'warning': 85, 'pause': 90},
}


def get_gpu_temp_preset(gpu_name: str = None) -> dict:
    """
    Get temperature preset for GPU.

    Args:
        gpu_name: GPU name (auto-detected if None)

    Returns:
        Dict with 'max', 'warning', 'pause' temperatures
    """
    if gpu_name is None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()

    if gpu_name:
        for key in GPU_TEMP_PRESETS:
            if key != 'default' and key.replace('_', ' ') in gpu_name.lower():
                return GPU_TEMP_PRESETS[key]

    return GPU_TEMP_PRESETS['default']


def check_gpu_health(
    max_temp: int = None,
    warning_temp: int = None,
    device_id: int = 0
) -> tuple:
    """
    Check GPU health and temperature.

    Args:
        max_temp: Maximum safe temperature (Celsius). Auto-detected if None.
        warning_temp: Warning temperature (Celsius). Auto-detected if None.
        device_id: GPU device ID

    Returns:
        (is_safe, message) tuple
    """
    # Auto-detect temperature thresholds if not provided
    if max_temp is None or warning_temp is None:
        preset = get_gpu_temp_preset()
        max_temp = max_temp or preset['max']
        warning_temp = warning_temp or preset['warning']
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    
    temp = get_gpu_temperature(device_id)
    if temp is None:
        return True, "Temperature monitoring unavailable"
    
    if temp >= max_temp:
        return False, f"CRITICAL: GPU temperature {temp}°C exceeds maximum {max_temp}°C"
    elif temp >= warning_temp:
        return True, f"WARNING: GPU temperature {temp}°C is high (max: {max_temp}°C)"
    else:
        return True, f"OK: GPU temperature {temp}°C"


class GPUTemperatureMonitor:
    """
    Monitor GPU temperature and automatically throttle/pause training if too hot.
    
    Designed for RTX 4060 laptop GPU to prevent overheating.
    """
    
    def __init__(
        self,
        max_temp: int = 83,  # RTX 4060 throttle temperature
        warning_temp: int = 75,  # Warning threshold
        pause_temp: int = 80,  # Pause training threshold
        check_interval: int = 10,  # Check every N batches
        cooldown_time: int = 30,  # Seconds to wait when paused
    ):
        self.max_temp = max_temp
        self.warning_temp = warning_temp
        self.pause_temp = pause_temp
        self.check_interval = check_interval
        self.cooldown_time = cooldown_time
        self.batch_count = 0
        self.pause_count = 0
        
    def check(self) -> tuple[bool, str]:
        """
        Check GPU temperature and return whether training should continue.
        
        Returns:
            (should_continue, message) tuple
        """
        self.batch_count += 1
        
        # Only check every N batches to avoid overhead
        if self.batch_count % self.check_interval != 0:
            return True, ""
        
        temp = get_gpu_temperature()
        if temp is None:
            return True, ""
        
        if temp >= self.max_temp:
            self.pause_count += 1
            logger.critical(
                f"GPU OVERHEATING: {temp}°C >= {self.max_temp}°C. "
                f"Training paused. Pause #{self.pause_count}"
            )
            return False, f"CRITICAL: {temp}°C (max: {self.max_temp}°C)"
        elif temp >= self.pause_temp:
            self.pause_count += 1
            logger.warning(
                f"GPU HOT: {temp}°C >= {self.pause_temp}°C. "
                f"Pausing training for {self.cooldown_time}s. Pause #{self.pause_count}"
            )
            return False, f"HOT: {temp}°C (pause at: {self.pause_temp}°C)"
        elif temp >= self.warning_temp:
            # Don't log warnings every check - too noisy
            return True, f"WARNING: {temp}°C"
        else:
            return True, f"OK: {temp}°C"
    
    def wait_for_cooldown(self):
        """Wait for GPU to cool down."""
        import time
        logger.info(f"Waiting {self.cooldown_time}s for GPU to cool down...")
        time.sleep(self.cooldown_time)
        
        # Check temperature after cooldown
        temp = get_gpu_temperature()
        if temp:
            logger.info(f"GPU temperature after cooldown: {temp}°C")
    
    def reset(self):
        """Reset batch counter."""
        self.batch_count = 0


if __name__ == "__main__":
    print_gpu_info()
    setup_cuda_optimizations()

    device = get_device()
    print(f"\nSelected device: {device}")
