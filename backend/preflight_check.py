#!/usr/bin/env python3
"""
XR2Text Pre-Flight Check Script
Run this before starting training to verify everything is ready.

Usage:
    python preflight_check.py

Authors: S. Nikhil, Dadhania Omkumar
"""

import sys
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def check_pass(name):
    print(f"  [PASS] {name}")
    return True

def check_fail(name, reason=""):
    print(f"  [FAIL] {name}")
    if reason:
        print(f"         {reason}")
    return False

def main():
    print_header("XR2Text Pre-Flight Check")

    all_passed = True

    # 1. Check Python version
    print("\n1. PYTHON VERSION")
    if sys.version_info >= (3, 9):
        check_pass(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        check_fail(f"Python {sys.version_info.major}.{sys.version_info.minor}", "Need Python 3.9+")
        all_passed = False

    # 2. Check PyTorch and CUDA
    print("\n2. PYTORCH & CUDA")
    try:
        import torch
        check_pass(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            check_pass(f"CUDA available - {gpu_name} ({gpu_mem:.1f} GB)")

            if gpu_mem >= 40:
                check_pass("GPU memory sufficient for batch_size=16")
            elif gpu_mem >= 8:
                check_pass("GPU memory sufficient for batch_size=1-4")
            else:
                check_fail("GPU memory", "Need at least 8GB VRAM")
                all_passed = False
        else:
            check_fail("CUDA not available", "Training requires GPU")
            all_passed = False
    except ImportError:
        check_fail("PyTorch", "Not installed")
        all_passed = False

    # 3. Check critical packages
    print("\n3. CRITICAL PACKAGES")
    packages = [
        ("transformers", "transformers"),
        ("timm", "timm"),
        ("datasets", "datasets"),
        ("PIL", "Pillow"),
        ("albumentations", "albumentations"),
        ("nltk", "nltk"),
        ("rouge_score", "rouge-score"),
        ("loguru", "loguru"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("yaml", "pyyaml"),
    ]

    for import_name, package_name in packages:
        try:
            __import__(import_name)
            check_pass(package_name)
        except ImportError:
            check_fail(package_name, f"pip install {package_name}")
            all_passed = False

    # 4. Check source files
    print("\n4. SOURCE FILES")
    critical_files = [
        "src/models/xr2text.py",
        "src/models/anatomical_attention.py",
        "src/training/trainer.py",
        "src/training/losses.py",
        "src/data/dataloader.py",
        "src/utils/metrics.py",
    ]

    for f in critical_files:
        if os.path.exists(f):
            check_pass(f)
        else:
            check_fail(f, "File not found")
            all_passed = False

    # 5. Check directories and WRITE PERMISSIONS (Critical for RunPod!)
    print("\n5. DIRECTORIES & WRITE PERMISSIONS")
    dirs = [
        "checkpoints",
        "logs",
        "data",
        "data/figures",
        "data/statistics",
        "data/human_evaluation",
        "data/ablation_results",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Try to fix permissions
        try:
            os.chmod(d, 0o777)
        except:
            pass

        # Test write permission
        test_file = os.path.join(d, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            check_pass(f"{d} (writable)")
        except Exception as e:
            check_fail(d, f"NOT WRITABLE: {str(e)[:30]}")
            all_passed = False

    # 6. Check notebooks
    print("\n6. NOTEBOOKS")
    notebooks = [
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_model_training.ipynb",
        "notebooks/03_evaluation_metrics.ipynb",
        "notebooks/04_ablation_study.ipynb",
        "notebooks/05_cross_dataset_evaluation.ipynb",
        "notebooks/06_radiologist_evaluation.ipynb",
    ]

    for nb in notebooks:
        if os.path.exists(nb):
            check_pass(nb.split("/")[-1])
        else:
            check_fail(nb.split("/")[-1], "Not found")
            all_passed = False

    # 7. Test model import
    print("\n7. MODEL IMPORT TEST")
    try:
        from src.models.xr2text import XR2TextModel, DEFAULT_CONFIG
        check_pass("XR2TextModel imports successfully")
    except Exception as e:
        check_fail("XR2TextModel import", str(e)[:50])
        all_passed = False

    # 8. Test dataloader import
    print("\n8. DATALOADER IMPORT TEST")
    try:
        from src.data.dataloader import get_dataloaders
        check_pass("get_dataloaders imports successfully")
    except Exception as e:
        check_fail("get_dataloaders import", str(e)[:50])
        all_passed = False

    # 9. NLTK data
    print("\n9. NLTK DATA")
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        check_pass("NLTK punkt tokenizer")
    except LookupError:
        print("  [WARN] NLTK punkt not found - downloading...")
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        check_pass("NLTK data downloaded")
    except Exception as e:
        check_fail("NLTK data", str(e)[:50])

    # Final summary
    print_header("PRE-FLIGHT CHECK SUMMARY")

    if all_passed:
        print("\n  ALL CHECKS PASSED!")
        print("\n  You are ready to start training.")
        print("  Run: jupyter notebook notebooks/02_model_training.ipynb")
        return 0
    else:
        print("\n  SOME CHECKS FAILED!")
        print("\n  Please fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
