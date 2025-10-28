"""
Output writers for saving masks, images, and metrics
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def save_mask(path: str, mask01: np.ndarray) -> None:
    """
    Save binary mask as PNG image
    
    Args:
        path: Output file path
        mask01: Binary mask with values 0 or 1
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert to 8-bit (0 or 255)
    mask255 = (mask01 * 255).astype(np.uint8)
    
    # Save as PNG
    success = cv2.imwrite(path, mask255)
    
    if success:
        logger.debug(f"Saved mask: {path}")
    else:
        logger.error(f"Failed to save mask: {path}")


def save_image(path: str, bgr: np.ndarray, quality: int = 95) -> None:
    """
    Save BGR image with appropriate format
    
    Args:
        path: Output file path
        bgr: BGR image array
        quality: JPEG quality (1-100) if saving as JPEG
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Determine format from extension
    ext = os.path.splitext(path)[1].lower()
    
    # Set encoding parameters
    if ext in ['.jpg', '.jpeg']:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    else:
        encode_params = []
    
    # Save image
    success = cv2.imwrite(path, bgr, encode_params)
    
    if success:
        logger.debug(f"Saved image: {path}")
    else:
        logger.error(f"Failed to save image: {path}")


def save_metrics(row: Dict[str, Any], csv_path: str) -> None:
    """
    Save or append metrics to CSV file
    
    Args:
        row: Dictionary containing metrics for one image
        csv_path: Path to CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Create DataFrame from row
    df_new = pd.DataFrame([row])
    
    # Check if CSV exists
    if os.path.exists(csv_path):
        # Read existing CSV
        df_existing = pd.read_csv(csv_path)
        
        # Check if this filename already exists
        if 'filename' in df_existing.columns:
            # Remove existing entry for this filename
            df_existing = df_existing[df_existing['filename'] != row.get('filename', '')]
        
        # Append new row
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Save to CSV
    df_combined.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to: {csv_path}")


def save_panel(before: np.ndarray, mask: np.ndarray, after: np.ndarray,
               out_path: str, add_labels: bool = True) -> None:
    """
    Save comparison panel showing before/mask/after
    
    Args:
        before: Original image
        mask: Binary mask (0/1 or 0/255)
        after: Processed image
        out_path: Output path for panel image
        add_labels: Whether to add text labels
    """
    h, w = before.shape[:2]
    
    # Ensure mask is 3-channel for visualization
    if len(mask.shape) == 2:
        # Normalize mask to 0-255 range
        if mask.max() <= 1:
            mask = mask * 255
        mask_vis = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        mask_vis = mask
    
    # Create panel with 3 images side by side
    panel = np.zeros((h, w * 3, 3), dtype=np.uint8)
    panel[:, :w] = before
    panel[:, w:w*2] = mask_vis
    panel[:, w*2:] = after
    
    # Add vertical separator lines
    panel[:, w-1:w+1] = [128, 128, 128]
    panel[:, w*2-1:w*2+1] = [128, 128, 128]
    
    # Add labels if requested
    if add_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        # Add background rectangles for better text visibility
        cv2.rectangle(panel, (5, 5), (150, 40), (0, 0, 0), -1)
        cv2.rectangle(panel, (w+5, 5), (w+150, 40), (0, 0, 0), -1)
        cv2.rectangle(panel, (w*2+5, 5), (w*2+150, 40), (0, 0, 0), -1)
        
        # Add text labels
        cv2.putText(panel, "Before", (10, 30), font, font_scale, color, thickness)
        cv2.putText(panel, "Mask", (w+10, 30), font, font_scale, color, thickness)
        cv2.putText(panel, "After", (w*2+10, 30), font, font_scale, color, thickness)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save panel
    success = cv2.imwrite(out_path, panel)
    
    if success:
        logger.debug(f"Saved panel: {out_path}")
    else:
        logger.error(f"Failed to save panel: {out_path}")


def save_detailed_panel(before: np.ndarray, masks: Dict[str, np.ndarray], 
                       after: np.ndarray, out_path: str) -> None:
    """
    Save detailed comparison panel with multiple masks
    
    Args:
        before: Original image
        masks: Dictionary of mask names and arrays
        after: Processed image
        out_path: Output path
    """
    h, w = before.shape[:2]
    
    # Calculate panel dimensions
    n_images = 2 + len(masks)  # before + masks + after
    panel_width = w * n_images
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    
    # Add before image
    panel[:, :w] = before
    
    # Add masks
    x_offset = w
    for mask_name, mask_array in masks.items():
        # Convert mask to 3-channel
        if len(mask_array.shape) == 2:
            if mask_array.max() <= 1:
                mask_array = mask_array * 255
            mask_vis = cv2.cvtColor(mask_array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            mask_vis = mask_array
        
        panel[:, x_offset:x_offset+w] = mask_vis
        
        # Add label
        cv2.rectangle(panel, (x_offset+5, 5), (x_offset+150, 40), (0, 0, 0), -1)
        cv2.putText(panel, mask_name, (x_offset+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        x_offset += w
    
    # Add after image
    panel[:, x_offset:] = after
    cv2.rectangle(panel, (x_offset+5, 5), (x_offset+150, 40), (0, 0, 0), -1)
    cv2.putText(panel, "After", (x_offset+10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save panel
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, panel)
    logger.debug(f"Saved detailed panel: {out_path}")


def create_summary_report(csv_path: str, output_dir: str) -> None:
    """
    Create summary report from metrics CSV
    
    Args:
        csv_path: Path to metrics CSV
        output_dir: Output directory for report
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Metrics CSV not found: {csv_path}")
        return
    
    # Read metrics
    df = pd.read_csv(csv_path)
    
    # Create summary statistics
    summary = {
        'Total Images': len(df),
        'Avg SPP Reduction': f"{df['spp_reduction'].mean():.4f}" if 'spp_reduction' in df else 'N/A',
        'Avg EPR': f"{df['epr'].mean():.3f}" if 'epr' in df else 'N/A',
        'Avg Colorfulness Ratio': f"{df['colorfulness_ratio'].mean():.3f}" if 'colorfulness_ratio' in df else 'N/A',
        'Avg SSIM': f"{df['ssim'].mean():.3f}" if 'ssim' in df else 'N/A'
    }
    
    # Save summary as text file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Reflection Removal Summary Report\n")
        f.write("=" * 40 + "\n\n")
        
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 40 + "\n")
        f.write("Detailed metrics available in: metrics.csv\n")
    
    logger.info(f"Created summary report: {summary_path}")


def save_debug_visualization(data: Dict[str, np.ndarray], out_path: str) -> None:
    """
    Save debug visualization with multiple intermediate results
    
    Args:
        data: Dictionary of stage names and images
        out_path: Output path
    """
    if not data:
        return
    
    # Get image dimensions from first item
    first_img = next(iter(data.values()))
    h, w = first_img.shape[:2]
    
    # Calculate grid dimensions
    n_images = len(data)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    # Create grid
    grid_h = h * rows
    grid_w = w * cols
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill grid
    for idx, (name, img) in enumerate(data.items()):
        row = idx // cols
        col = idx % cols
        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w
        
        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        grid[y1:y2, x1:x2] = img
        
        # Add label
        cv2.rectangle(grid, (x1+5, y1+5), (x1+200, y1+35), (0, 0, 0), -1)
        cv2.putText(grid, name, (x1+10, y1+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save grid
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)
    logger.debug(f"Saved debug visualization: {out_path}")
