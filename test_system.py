#!/usr/bin/env python
"""
Test script to verify the reflection removal system
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import load_config
from app.controllers.pipeline import run_one
from app.utils.io import get_image_files

def test_single_image():
    """Test processing a single image"""
    print("=" * 60)
    print("Testing Single Image Processing")
    print("=" * 60)
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Override paths for test
    config['paths']['out_mask'] = 'out/test/mask'
    config['paths']['out_result'] = 'out/test/result'
    config['paths']['out_eval'] = 'out/test/eval'
    
    # Get a sample image
    sample_images = get_image_files('data/samples')
    
    if not sample_images:
        print("ERROR: No sample images found")
        return False
    
    test_image = sample_images[0]
    print(f"Testing with image: {test_image}")
    
    # Process the image
    metrics = run_one(test_image, config)
    
    if metrics:
        print("SUCCESS: Image processed successfully")
        print(f"Metrics: SPP reduction = {metrics['spp_reduction']:.4f}")
        print(f"         EPR = {metrics['epr']:.3f}")
        print(f"         Colorfulness ratio = {metrics['colorfulness_ratio']:.3f}")
        return True
    else:
        print("ERROR: Failed to process image")
        return False

def test_configuration():
    """Test configuration loading"""
    print("=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)
    
    try:
        config = load_config('configs/config.yaml')
        print("SUCCESS: Configuration loaded")
        print(f"Input directory: {config['paths']['input_dir']}")
        print(f"Container threshold: {config['container']['a_threshold']}")
        print(f"Detection Z-threshold: {config['detect']['z_thresh']}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return False

def test_imports():
    """Test all module imports"""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    try:
        from app.models import (
            get_container_mask, detect_candidates,
            split_thin_blob, inpaint_lab, compute_metrics
        )
        from app.views import save_mask, save_image, save_metrics
        from app.controllers import run_one, run_batch
        
        print("SUCCESS: All modules imported successfully")
        return True
    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("REFLECTION REMOVAL SYSTEM TEST")
    print("=" * 60 + "\n")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Single Image Processing", test_single_image)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
        print()
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:30} {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please check the errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
