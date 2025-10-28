"""
Visualization utilities for debugging and analysis
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def create_debug_panel(images: Dict[str, np.ndarray], title: str = "Debug Panel") -> np.ndarray:
    """
    Create a debug panel with multiple images
    
    Args:
        images: Dictionary of image names and arrays
        title: Panel title
        
    Returns:
        Combined panel image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Convert all images to BGR and same size
    processed = []
    target_h, target_w = 0, 0
    
    for name, img in images.items():
        if img is None:
            continue
            
        # Convert grayscale to BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        processed.append((name, img))
        target_h = max(target_h, img.shape[0])
        target_w = max(target_w, img.shape[1])
    
    if not processed:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Create grid layout
    n_images = len(processed)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    panel_h = target_h * rows + 50  # Extra space for title
    panel_w = target_w * cols
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255, 255, 255), 2)
    
    # Add images
    for idx, (name, img) in enumerate(processed):
        row = idx // cols
        col = idx % cols
        y1 = row * target_h + 50
        y2 = y1 + img.shape[0]
        x1 = col * target_w
        x2 = x1 + img.shape[1]
        
        panel[y1:y2, x1:x2] = img
        
        # Add image label
        label_y = y1 + 25
        label_x = x1 + 10
        cv2.rectangle(panel, (label_x-5, label_y-20), 
                     (label_x+len(name)*10+5, label_y+5), (0, 0, 0), -1)
        cv2.putText(panel, name, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return panel


def visualize_histogram(img_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create histogram visualization
    
    Args:
        img_bgr: BGR image
        mask: Optional mask to limit histogram region
        
    Returns:
        Histogram visualization image
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['blue', 'green', 'red']
    channel_names = ['B Channel', 'G Channel', 'R Channel']
    
    for i, (ax, color, name) in enumerate(zip(axes, colors, channel_names)):
        # Calculate histogram
        if mask is not None:
            hist = cv2.calcHist([img_bgr], [i], mask, [256], [0, 256])
        else:
            hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        
        # Plot
        ax.plot(hist, color=color)
        ax.set_title(name)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_hist = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_hist = cv2.cvtColor(img_hist, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return img_hist


def visualize_heatmap(data: np.ndarray, title: str = "Heatmap", 
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Create heatmap visualization
    
    Args:
        data: 2D array to visualize
        title: Heatmap title
        colormap: OpenCV colormap
        
    Returns:
        Heatmap image
    """
    # Normalize to 0-255
    normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(normalized, colormap)
    
    # Add title
    h, w = heatmap.shape[:2]
    title_area = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(title_area, title, (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine
    result = np.vstack([title_area, heatmap])
    
    return result


def overlay_mask(img: np.ndarray, mask: np.ndarray, 
                 color: Tuple[int, int, int] = (0, 0, 255), 
                 alpha: float = 0.3) -> np.ndarray:
    """
    Overlay mask on image with transparency
    
    Args:
        img: Base image
        mask: Binary mask
        color: Overlay color (BGR)
        alpha: Transparency (0-1)
        
    Returns:
        Image with overlay
    """
    overlay = img.copy()
    
    # Create colored mask
    mask_colored = np.zeros_like(img)
    mask_colored[mask > 0] = color
    
    # Blend
    result = cv2.addWeighted(img, 1-alpha, mask_colored, alpha, 0)
    
    return result


def draw_component_boxes(img: np.ndarray, mask: np.ndarray, 
                        min_area: int = 10) -> np.ndarray:
    """
    Draw bounding boxes around mask components
    
    Args:
        img: Base image
        mask: Binary mask
        min_area: Minimum component area to draw
        
    Returns:
        Image with bounding boxes
    """
    result = img.copy()
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Draw boxes for each component
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add label with area
        label_text = f"#{label} A:{area}"
        cv2.putText(result, label_text, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return result


def create_before_after_comparison(before: np.ndarray, after: np.ndarray, 
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create side-by-side before/after comparison
    
    Args:
        before: Original image
        after: Processed image
        mask: Optional mask to highlight
        
    Returns:
        Comparison image
    """
    h, w = before.shape[:2]
    
    # Create panel
    panel = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # Add images
    if mask is not None:
        # Overlay mask on before image
        before_with_mask = overlay_mask(before, mask, (0, 0, 255), 0.3)
        panel[:, :w] = before_with_mask
    else:
        panel[:, :w] = before
    
    panel[:, w:] = after
    
    # Add separator
    panel[:, w-2:w+2] = [128, 128, 128]
    
    # Add labels
    cv2.rectangle(panel, (5, 5), (100, 35), (0, 0, 0), -1)
    cv2.rectangle(panel, (w+5, 5), (w+100, 35), (0, 0, 0), -1)
    cv2.putText(panel, "Before", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "After", (w+10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return panel


def create_difference_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Create difference map between two images
    
    Args:
        before: First image
        after: Second image
        
    Returns:
        Difference map visualization
    """
    # Convert to float for accurate difference
    before_float = before.astype(np.float32)
    after_float = after.astype(np.float32)
    
    # Compute absolute difference
    diff = np.abs(after_float - before_float)
    
    # Convert to grayscale for better visualization
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    diff_enhanced = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    diff_enhanced = diff_enhanced.astype(np.uint8)
    
    # Apply colormap
    diff_colored = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_HOT)
    
    return diff_colored


def save_debug_images(debug_dict: Dict[str, np.ndarray], output_dir: str, 
                     base_name: str) -> None:
    """
    Save all debug images to directory
    
    Args:
        debug_dict: Dictionary of debug images
        output_dir: Output directory
        base_name: Base filename
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, img in debug_dict.items():
        if img is None:
            continue
        
        filename = f"{base_name}_{name}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, img)
        logger.debug(f"Saved debug image: {filepath}")


def plot_metrics_summary(metrics_df: pd.DataFrame, output_path: str) -> None:
    """
    Create summary plots for metrics
    
    Args:
        metrics_df: DataFrame with metrics
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SPP reduction
    if 'spp_before' in metrics_df and 'spp_after' in metrics_df:
        ax = axes[0, 0]
        x = range(len(metrics_df))
        ax.plot(x, metrics_df['spp_before'], 'r-', label='Before', alpha=0.7)
        ax.plot(x, metrics_df['spp_after'], 'g-', label='After', alpha=0.7)
        ax.set_title('Saturated Pixel Percentage')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('SPP')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # EPR distribution
    if 'epr' in metrics_df:
        ax = axes[0, 1]
        ax.hist(metrics_df['epr'], bins=20, edgecolor='black')
        ax.axvline(0.9, color='r', linestyle='--', label='Target (0.9)')
        ax.set_title('Edge Preservation Ratio Distribution')
        ax.set_xlabel('EPR')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Colorfulness ratio
    if 'colorfulness_ratio' in metrics_df:
        ax = axes[1, 0]
        ax.hist(metrics_df['colorfulness_ratio'], bins=20, edgecolor='black')
        ax.axvline(1.0, color='r', linestyle='--', label='No change')
        ax.set_title('Colorfulness Ratio Distribution')
        ax.set_xlabel('Ratio (After/Before)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # SSIM distribution
    if 'ssim' in metrics_df:
        ax = axes[1, 1]
        ax.hist(metrics_df['ssim'], bins=20, edgecolor='black')
        ax.set_title('SSIM Distribution')
        ax.set_xlabel('SSIM')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics summary plot: {output_path}")
