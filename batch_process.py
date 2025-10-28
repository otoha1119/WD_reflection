#!/usr/bin/env python
"""
Example script for batch processing with custom settings
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.config import load_config
from app.controllers.pipeline import run_batch

def main():
    """
    Example of batch processing with custom configuration
    """
    # Load base configuration
    config = load_config('configs/config.yaml')
    
    # Customize settings for specific use case
    # For darker images, relax detection thresholds
    config['detect']['z_thresh'] = 1.8  # More sensitive detection
    config['detect']['s_thresh'] = 50   # Allow higher saturation
    
    # For stronger reflections, increase dilation
    config['shape']['dilate_blob'] = 3
    
    # Process different input directories
    input_dirs = [
        'data/samples',  # Test with sample images first
        # 'data/images',   # Uncomment for full dataset
    ]
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Skipping {input_dir} (not found)")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing: {input_dir}")
        print(f"{'='*60}")
        
        # Update paths for this batch
        batch_name = os.path.basename(input_dir)
        config['paths']['input_dir'] = input_dir
        config['paths']['out_mask'] = f'out/{batch_name}/mask'
        config['paths']['out_result'] = f'out/{batch_name}/result'
        config['paths']['out_eval'] = f'out/{batch_name}/eval'
        config['paths']['out_logs'] = f'out/{batch_name}/logs'
        
        # Run batch processing
        try:
            run_batch(input_dir, config)
            print(f"✓ Successfully processed {input_dir}")
        except Exception as e:
            print(f"✗ Error processing {input_dir}: {e}")
    
    print(f"\n{'='*60}")
    print("Batch processing completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
