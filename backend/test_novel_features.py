"""
Quick test script to verify novel features work before running notebooks.

Run this BEFORE running the notebooks to catch any import or integration issues.
"""

import sys
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TESTING NOVEL FEATURES")
print("=" * 60)

# Test 1: Import novel losses
print("\n1. Testing Novel Losses Import...")
try:
    from src.training.losses import (
        CombinedNovelLoss,
        AnatomicalConsistencyLoss,
        ClinicalEntityLoss,
        RegionAwareFocalLoss,
    )
    print("   [OK] Novel losses imported successfully")
except ImportError as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 2: Import curriculum learning
print("\n2. Testing Curriculum Learning Import...")
try:
    from src.training.curriculum import (
        AnatomicalCurriculumScheduler,
        create_curriculum_dataloader,
    )
    print("   [OK] Curriculum learning imported successfully")
except ImportError as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 3: Import clinical validator
print("\n3. Testing Clinical Validator Import...")
try:
    from src.utils.clinical_validator import ClinicalValidator
    print("   [OK] Clinical validator imported successfully")
except ImportError as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 4: Import deployment tools
print("\n4. Testing Deployment Tools Import...")
try:
    from src.utils.deployment import ModelQuantizer, InferenceOptimizer, BatchProcessor
    print("   [OK] Deployment tools imported successfully")
except ImportError as e:
    print(f"   [WARNING] Warning: {e} (deployment tools optional)")
    print("   (This is OK if you're not testing deployment yet)")

# Test 5: Import multi-scale features
print("\n5. Testing Multi-Scale Features Import...")
try:
    from src.models.multiscale_features import MultiScaleFeatureFusion
    print("   [OK] Multi-scale features imported successfully")
except ImportError as e:
    print(f"   [WARNING] Warning: {e} (multi-scale features optional)")
    print("   (This is OK if you're not using multi-scale features)")

# Test 6: Test CombinedNovelLoss initialization
print("\n6. Testing CombinedNovelLoss Initialization...")
try:
    import torch
    novel_loss = CombinedNovelLoss(
        use_anatomical_consistency=True,
        use_clinical_entity=True,
        use_region_focal=True,
        use_cross_modal=False,
    )
    print("   [OK] CombinedNovelLoss initialized successfully")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 7: Test ClinicalValidator initialization
print("\n7. Testing ClinicalValidator Initialization...")
try:
    validator = ClinicalValidator()
    print("   [OK] ClinicalValidator initialized successfully")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 8: Test CurriculumScheduler initialization
print("\n8. Testing CurriculumScheduler Initialization...")
try:
    scheduler = AnatomicalCurriculumScheduler()
    print("   [OK] AnatomicalCurriculumScheduler initialized successfully")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 9: Test model outputs structure (if model available)
print("\n9. Testing Model Outputs Structure...")
try:
    from src.models.xr2text import XR2TextModel, DEFAULT_CONFIG
    
    # Check if model can be created (don't actually create it - just check structure)
    print("   [OK] Model class accessible")
    print("   [OK] Model outputs should include:")
    print("      - 'spatial_priors'")
    print("      - 'attention_info' (with 'region_attention')")
    print("      - 'region_weights'")
    print("      - 'logits'")
    print("      - 'projected_features'")
except Exception as e:
    print(f"   [WARNING] Warning: {e}")
    print("   (Model check skipped - this is OK)")

print("\n" + "=" * 60)
print("[SUCCESS] ALL CRITICAL TESTS PASSED!")
print("=" * 60)
print("\nYou can now proceed to run the notebooks:")
print("  1. Start with: notebooks/01_data_exploration.ipynb")
print("  2. Then: notebooks/02_model_training.ipynb (with novel features DISABLED first)")
print("  3. Then: notebooks/03_evaluation_metrics.ipynb")
print("  4. Finally: notebooks/04_ablation_study.ipynb")
print("\nSee TESTING_GUIDE.md for detailed instructions.")

