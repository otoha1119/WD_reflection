"""
Main processing pipeline controller
"""

import os
import cv2
import numpy as np
import time
import logging
import traceback
from typing import Dict, Optional, Any
from pathlib import Path

from app.utils.io import (
    get_image_files, read_image_with_exif, validate_image, 
    get_output_path, ensure_dir
)
from app.models.container_mask import get_container_mask
from app.models.highlight_detect import detect_candidates
from app.models.shape_classify import split_thin_blob
from app.models.inpaint import inpaint_lab
from app.models.metrics import compute_metrics, format_metrics_for_csv
from app.views.writers import (
    save_mask, save_image, save_metrics, save_panel,
    save_detailed_panel, save_debug_visualization
)
from app.views.viz import (
    create_debug_panel, overlay_mask, create_before_after_comparison
)

logger = logging.getLogger(__name__)


def run_one(path: str, cfg: Dict) -> Optional[Dict]:
    """
    Process a single image through the pipeline
    
    Args:
        path: Path to input image
        cfg: Configuration dictionary
        
    Returns:
        Dictionary of metrics or None if failed
    """
    start_time = time.time()
    filename = os.path.basename(path)
    
    try:
        logger.info(f"Processing: {filename}")
        
        # Step 1: Load image
        img_bgr = read_image_with_exif(path)
        if img_bgr is None or not validate_image(img_bgr):
            logger.error(f"Failed to load or validate image: {path}")
            return None
        
        logger.debug(f"Image shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
        
        # Step 2: Extract container mask
        logger.debug("Extracting container mask...")
        M_container, M_interior = get_container_mask(img_bgr, cfg)
        
        if np.sum(M_interior) == 0:
            logger.error("Failed to extract container region")
            return None
        
        # Step 3: Detect reflection candidates
        logger.debug("Detecting reflection candidates...")
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        candidates = detect_candidates(img_bgr, hsv, M_interior, cfg)
        
        if np.sum(candidates) == 0:
            logger.warning("No reflections detected")
            # Still save empty mask and original image
            mask_path = get_output_path(path, cfg['paths']['out_mask'], '.png')
            save_mask(mask_path, candidates)
            
            result_path = get_output_path(path, cfg['paths']['out_result'], '.png')
            save_image(result_path, img_bgr)
            
            # Compute metrics with no changes
            metrics = compute_metrics(img_bgr, img_bgr, candidates, cfg)
            formatted = format_metrics_for_csv(metrics, filename)
            csv_path = os.path.join(cfg['paths']['out_eval'], 'metrics.csv')
            save_metrics(formatted, csv_path)
            
            return metrics
        
        # Step 4: Shape classification
        logger.debug("Classifying reflection shapes...")
        mask_thin, mask_blob = split_thin_blob(candidates, cfg)
        
        # Step 5: Inpainting
        logger.debug("Performing inpainting...")
        img_result, mask_union = inpaint_lab(img_bgr, mask_thin, mask_blob, cfg)
        
        # Step 6: Save outputs
        # Save mask
        mask_path = get_output_path(path, cfg['paths']['out_mask'], '.png')
        save_mask(mask_path, mask_union)
        logger.info(f"Saved mask: {mask_path}")
        
        # Save result image
        result_path = get_output_path(path, cfg['paths']['out_result'], '.png')
        save_image(result_path, img_result)
        logger.info(f"Saved result: {result_path}")
        
        # Step 7: Compute metrics
        logger.debug("Computing metrics...")
        metrics = compute_metrics(img_bgr, img_result, mask_union, cfg)
        
        # Save metrics to CSV
        formatted = format_metrics_for_csv(metrics, filename)
        csv_path = os.path.join(cfg['paths']['out_eval'], 'metrics.csv')
        save_metrics(formatted, csv_path)
        
        # Step 8: Optional visualizations
        if cfg.get('debug', {}).get('save_panels', True):
            panel_path = os.path.join(cfg['paths']['out_eval'], 'panels', 
                                     os.path.splitext(filename)[0] + '_panel.png')
            ensure_dir(os.path.dirname(panel_path))
            save_panel(img_bgr, mask_union, img_result, panel_path)
            logger.debug(f"Saved panel: {panel_path}")
            
            # Save detailed panel if in debug mode
            if cfg.get('debug', {}).get('verbose', False):
                detailed_masks = {
                    'Container': M_container * 255,
                    'Candidates': candidates * 255,
                    'Thin': mask_thin * 255,
                    'Blob': mask_blob * 255,
                    'Union': mask_union * 255
                }
                detailed_path = os.path.join(cfg['paths']['out_eval'], 'debug', 
                                           os.path.splitext(filename)[0] + '_detailed.png')
                ensure_dir(os.path.dirname(detailed_path))
                save_detailed_panel(img_bgr, detailed_masks, img_result, detailed_path)
        
        # Log processing time
        elapsed = time.time() - start_time
        logger.info(f"Completed {filename} in {elapsed:.2f} seconds")
        
        # Log key metrics
        logger.info(f"  SPP: {metrics['spp_before']:.3%} → {metrics['spp_after']:.3%}")
        logger.info(f"  EPR: {metrics['epr']:.3f}")
        logger.info(f"  Colorfulness ratio: {metrics['colorfulness_ratio']:.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


def run_batch(input_dir: str, cfg: Dict) -> None:
    """
    Process all images in directory
    
    Args:
        input_dir: Input directory path
        cfg: Configuration dictionary
    """
    # Get list of images
    image_files = get_image_files(input_dir)
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process statistics
    successful = 0
    failed = 0
    skipped = 0
    total_time = 0
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        logger.info(f"[{idx}/{len(image_files)}] Processing: {filename}")
        
        # Check if already processed (optional skip)
        if cfg.get('skip_existing', False):
            result_path = get_output_path(image_path, cfg['paths']['out_result'], '.png')
            if os.path.exists(result_path):
                logger.info(f"  Skipping (already processed)")
                skipped += 1
                continue
        
        # Process image
        start_time = time.time()
        metrics = run_one(image_path, cfg)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if metrics is not None:
            successful += 1
            logger.info(f"  Success ({elapsed:.2f}s)")
        else:
            failed += 1
            logger.error(f"  Failed ({elapsed:.2f}s)")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("Batch Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    if successful > 0:
        logger.info(f"Average time: {total_time/successful:.2f} seconds/image")
    logger.info("=" * 60)
    
    # Create summary plots if metrics exist
    csv_path = os.path.join(cfg['paths']['out_eval'], 'metrics.csv')
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            from app.views.viz import plot_metrics_summary
            
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                plot_path = os.path.join(cfg['paths']['out_eval'], 'metrics_summary.png')
                plot_metrics_summary(df, plot_path)
                logger.info(f"Created metrics summary plot: {plot_path}")
        except Exception as e:
            logger.warning(f"Could not create summary plots: {e}")


def process_with_fallback(img_path: str, cfg: Dict, max_retries: int = 2) -> Optional[Dict]:
    """
    Process image with automatic parameter adjustment on failure
    
    Args:
        img_path: Path to image
        cfg: Configuration dictionary
        max_retries: Maximum number of retries with adjusted parameters
        
    Returns:
        Metrics dictionary or None
    """
    original_cfg = cfg.copy()
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"Retry {attempt} with adjusted parameters")
            
            # Relax detection thresholds
            cfg['detect']['z_thresh'] = max(1.5, cfg['detect']['z_thresh'] - 0.3)
            cfg['detect']['s_thresh'] = min(60, cfg['detect']['s_thresh'] + 10)
            cfg['detect']['min_area'] = max(5, cfg['detect']['min_area'] - 5)
            
            # Adjust container detection
            cfg['container']['a_threshold'] = cfg['container']['a_threshold'] + 2
            
            logger.debug(f"Adjusted z_thresh: {cfg['detect']['z_thresh']}")
            logger.debug(f"Adjusted s_thresh: {cfg['detect']['s_thresh']}")
        
        metrics = run_one(img_path, cfg)
        
        if metrics is not None:
            if attempt > 0:
                logger.info(f"Successfully processed after {attempt} retries")
            return metrics
    
    logger.error(f"Failed to process after {max_retries} retries")
    return None


def validate_configuration(cfg: Dict) -> bool:
    """
    Validate configuration parameters
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['paths', 'container', 'detect', 'shape', 'inpaint', 'eval']
    
    for section in required_sections:
        if section not in cfg:
            logger.error(f"Missing configuration section: {section}")
            return False
    
    # Validate paths
    required_paths = ['input_dir', 'out_mask', 'out_result', 'out_eval']
    for path_key in required_paths:
        if path_key not in cfg['paths']:
            logger.error(f"Missing path configuration: {path_key}")
            return False
    
    # Validate numeric parameters
    numeric_checks = [
        ('container', 'a_threshold', 0, 255),
        ('detect', 'z_thresh', 0.5, 5.0),
        ('detect', 's_thresh', 0, 255),
        ('shape', 'thin_aspect_min', 1.0, 10.0),
        ('inpaint', 'radius', 1, 10)
    ]
    
    for section, key, min_val, max_val in numeric_checks:
        if key in cfg.get(section, {}):
            value = cfg[section][key]
            if not (min_val <= value <= max_val):
                logger.warning(f"Parameter {section}.{key}={value} outside range [{min_val}, {max_val}]")
    
    return True
