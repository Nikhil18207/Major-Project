"""
Computational Efficiency Analysis Module

This module provides comprehensive tools for analyzing model efficiency:
- FLOPs calculation
- Memory profiling
- Inference latency analysis
- Throughput benchmarking

Essential for research publications comparing model efficiency.

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from loguru import logger


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""
    flops: int  # Floating point operations
    macs: int  # Multiply-accumulate operations
    params: int  # Number of parameters
    trainable_params: int
    memory_mb: float  # Peak memory in MB
    inference_time_ms: float  # Average inference time
    throughput_fps: float  # Frames per second


class FLOPsCounter:
    """
    Count FLOPs (Floating Point Operations) for PyTorch models.
    
    Provides accurate FLOPs counting for common operations.
    """
    
    def __init__(self):
        self.flops = 0
        self.macs = 0
        self.hooks = []
        
    def _conv_flops(self, module, input, output):
        """Count FLOPs for convolution layers."""
        batch_size = input[0].size(0)
        output_height, output_width = output.size(2), output.size(3)
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        bias_ops = 1 if module.bias is not None else 0
        
        flops = batch_size * module.out_channels * output_height * output_width * (kernel_ops + bias_ops)
        self.flops += flops
        self.macs += flops // 2
        
    def _linear_flops(self, module, input, output):
        """Count FLOPs for linear layers."""
        batch_size = input[0].size(0) if input[0].dim() > 1 else 1
        
        # Handle multi-dimensional inputs (e.g., from transformers)
        if input[0].dim() > 2:
            batch_size = input[0].numel() // input[0].size(-1)
        
        flops = batch_size * module.in_features * module.out_features
        if module.bias is not None:
            flops += batch_size * module.out_features
            
        self.flops += flops
        self.macs += (batch_size * module.in_features * module.out_features) // 2
        
    def _attention_flops(self, module, input, output):
        """Count FLOPs for multi-head attention."""
        # Approximate: Q*K^T + softmax + attention*V
        if hasattr(module, 'embed_dim') and hasattr(module, 'num_heads'):
            batch_size = input[0].size(0)
            seq_len = input[0].size(1)
            embed_dim = module.embed_dim
            
            # Q*K^T: (B, H, S, D/H) x (B, H, D/H, S) = (B, H, S, S)
            qk_flops = batch_size * module.num_heads * seq_len * seq_len * (embed_dim // module.num_heads)
            
            # Softmax: approximately 5 ops per element
            softmax_flops = batch_size * module.num_heads * seq_len * seq_len * 5
            
            # Attention * V: (B, H, S, S) x (B, H, S, D/H) = (B, H, S, D/H)
            attn_v_flops = batch_size * module.num_heads * seq_len * seq_len * (embed_dim // module.num_heads)
            
            self.flops += qk_flops + softmax_flops + attn_v_flops
            self.macs += (qk_flops + attn_v_flops) // 2
    
    def _layernorm_flops(self, module, input, output):
        """Count FLOPs for layer normalization."""
        # Mean, var, normalize: approximately 5 ops per element
        flops = input[0].numel() * 5
        self.flops += flops
        
    def _gelu_flops(self, module, input, output):
        """Count FLOPs for GELU activation."""
        # Approximate: 8 ops per element
        flops = input[0].numel() * 8
        self.flops += flops
        
    def count(self, model: nn.Module, input_tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Count FLOPs for a forward pass.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor
            
        Returns:
            Tuple of (FLOPs, MACs)
        """
        self.flops = 0
        self.macs = 0
        self.hooks = []
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(self._conv_flops)
                self.hooks.append(hook)
            elif isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._linear_flops)
                self.hooks.append(hook)
            elif isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_flops)
                self.hooks.append(hook)
            elif isinstance(module, nn.LayerNorm):
                hook = module.register_forward_hook(self._layernorm_flops)
                self.hooks.append(hook)
            elif isinstance(module, nn.GELU):
                hook = module.register_forward_hook(self._gelu_flops)
                self.hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            model(input_tensor)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        return self.flops, self.macs


class MemoryProfiler:
    """Profile GPU memory usage during model execution."""
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {
                'allocated_mb': 0,
                'reserved_mb': 0,
                'max_allocated_mb': 0,
            }
        
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        }
    
    @staticmethod
    @contextmanager
    def profile():
        """Context manager for memory profiling."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    @staticmethod
    def profile_model(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Profile memory usage for a model forward pass.
        
        Args:
            model: PyTorch model
            input_tensor: Example input
            
        Returns:
            Dictionary with memory statistics
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return MemoryProfiler.get_memory_stats()


class InferenceBenchmark:
    """Benchmark model inference performance."""
    
    def __init__(self, warmup_runs: int = 10, num_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
    def benchmark(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """
        Benchmark inference latency and throughput.
        
        Args:
            model: PyTorch model
            input_tensor: Example input
            use_amp: Whether to use automatic mixed precision
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
                
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                else:
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
        
        batch_size = input_tensor.size(0)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_fps': 1000 / np.mean(times) * batch_size,
            'per_sample_ms': np.mean(times) / batch_size,
        }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by module type
    by_module = {}
    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters(recurse=False))
        if module_params > 0:
            module_type = type(module).__name__
            by_module[module_type] = by_module.get(module_type, 0) + module_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'by_module': by_module,
    }


def comprehensive_efficiency_analysis(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 384, 384),
    device: str = 'cuda',
    use_amp: bool = False,
) -> Dict:
    """
    Perform comprehensive efficiency analysis.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on
        use_amp: Whether to use AMP
        
    Returns:
        Dictionary with all efficiency metrics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    results = {}
    
    # 1. Parameter count
    logger.info("Counting parameters...")
    param_info = count_parameters(model)
    results['parameters'] = param_info
    
    # 2. FLOPs count
    logger.info("Counting FLOPs...")
    try:
        flops_counter = FLOPsCounter()
        flops, macs = flops_counter.count(model, dummy_input)
        results['flops'] = flops
        results['macs'] = macs
        results['gflops'] = flops / 1e9
        results['gmacs'] = macs / 1e9
    except Exception as e:
        logger.warning(f"FLOPs counting failed: {e}")
        results['flops'] = 0
        results['macs'] = 0
    
    # 3. Memory profiling
    logger.info("Profiling memory...")
    memory_stats = MemoryProfiler.profile_model(model, dummy_input)
    results['memory'] = memory_stats
    
    # 4. Inference benchmark
    logger.info("Benchmarking inference...")
    benchmark = InferenceBenchmark(warmup_runs=10, num_runs=100)
    timing_stats = benchmark.benchmark(model, dummy_input, use_amp=use_amp)
    results['timing'] = timing_stats
    
    # 5. Summary
    results['summary'] = {
        'total_params_M': param_info['total'] / 1e6,
        'trainable_params_M': param_info['trainable'] / 1e6,
        'gflops': results.get('gflops', 0),
        'peak_memory_mb': memory_stats.get('max_allocated_mb', 0),
        'inference_ms': timing_stats['mean_ms'],
        'throughput_fps': timing_stats['throughput_fps'],
    }
    
    return results


def compare_model_efficiency(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...] = (1, 3, 384, 384),
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Compare efficiency of multiple models.
    
    Args:
        models: Dictionary mapping model names to models
        input_shape: Input tensor shape
        device: Device to run on
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for name, model in models.items():
        logger.info(f"Analyzing {name}...")
        results[name] = comprehensive_efficiency_analysis(model, input_shape, device)
    
    return results


def generate_efficiency_table(results: Dict[str, Dict]) -> str:
    """Generate LaTeX table for efficiency comparison."""
    latex = """
\\begin{table}[h]
\\centering
\\caption{Model Efficiency Comparison}
\\label{tab:efficiency}
\\begin{tabular}{l|cccc}
\\hline
\\textbf{Model} & \\textbf{Params (M)} & \\textbf{GFLOPs} & \\textbf{Memory (MB)} & \\textbf{Latency (ms)} \\\\
\\hline
"""
    
    for name, metrics in results.items():
        summary = metrics['summary']
        latex += f"{name} & {summary['total_params_M']:.1f} & {summary['gflops']:.1f} & "
        latex += f"{summary['peak_memory_mb']:.0f} & {summary['inference_ms']:.1f} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex
