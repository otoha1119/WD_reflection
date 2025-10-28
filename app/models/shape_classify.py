"""
Shape classification for reflection components (thin/linear vs blob/area)
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def split_thin_blob(cand: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split candidates into thin (linear) and blob (area) components
    
    Args:
        cand: Binary candidate mask [0/1]
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (mask_thin[0/1], mask_blob[0/1])
    """
    # Get configuration parameters
    thin_min_short = cfg.get('shape', {}).get('thin_min_short', 8)
    thin_aspect_min = cfg.get('shape', {}).get('thin_aspect_min', 4.0)
    thin_area_max = cfg.get('shape', {}).get('thin_area_max', 400)
    dilate_thin = cfg.get('shape', {}).get('dilate_thin', 1)
    dilate_blob = cfg.get('shape', {}).get('dilate_blob', 2)
    
    # Initialize output masks
    mask_thin = np.zeros_like(cand, dtype=np.uint8)
    mask_blob = np.zeros_like(cand, dtype=np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand, connectivity=8)
    
    thin_count = 0
    blob_count = 0
    
    # Analyze each component
    for label in range(1, num_labels):  # Skip background
        # Get component statistics
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        
        # Calculate shape metrics
        min_dim = min(w, h)
        max_dim = max(w, h)
        aspect_ratio = max_dim / max(min_dim, 1)
        
        # Create component mask
        component_mask = (labels == label).astype(np.uint8)
        
        # Classify as thin (linear) or blob (area)
        is_thin = (min_dim < thin_min_short and 
                  aspect_ratio > thin_aspect_min and 
                  area < thin_area_max)
        
        if is_thin:
            mask_thin |= component_mask
            thin_count += 1
            logger.debug(f"Component {label}: THIN (w={w}, h={h}, aspect={aspect_ratio:.2f}, area={area})")
        else:
            mask_blob |= component_mask
            blob_count += 1
            logger.debug(f"Component {label}: BLOB (w={w}, h={h}, aspect={aspect_ratio:.2f}, area={area})")
    
    logger.info(f"Classified {thin_count} thin and {blob_count} blob components")
    
    # Apply variable dilation based on type
    mask_thin_dilated = dilate_mask(mask_thin, dilate_thin, shape_type='thin')
    mask_blob_dilated = dilate_mask(mask_blob, dilate_blob, shape_type='blob')
    
    # Optional: Apply distance-based dilation for stronger reflections
    mask_blob_dilated = apply_intensity_based_dilation(mask_blob_dilated, cand, cfg)
    
    return mask_thin_dilated, mask_blob_dilated


def dilate_mask(mask: np.ndarray, iterations: int, shape_type: str = 'blob') -> np.ndarray:
    """
    Apply morphological dilation with appropriate kernel
    
    Args:
        mask: Binary mask to dilate
        iterations: Number of dilation iterations
        shape_type: 'thin' or 'blob' to select kernel type
        
    Returns:
        Dilated mask
    """
    if iterations <= 0:
        return mask
    
    if shape_type == 'thin':
        # Use smaller kernel for thin components to avoid over-dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    else:
        # Use larger kernel for blob components
        kernel_size = 5 if iterations <= 2 else 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    
    return dilated.astype(np.uint8)


def apply_intensity_based_dilation(mask_blob: np.ndarray, original_cand: np.ndarray, 
                                   cfg: Dict) -> np.ndarray:
    """
    Apply additional dilation for high-intensity blob components
    
    Args:
        mask_blob: Blob mask
        original_cand: Original candidate mask before dilation
        cfg: Configuration dictionary
        
    Returns:
        Enhanced dilated mask
    """
    # Find connected components in blob mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_blob, connectivity=8)
    
    enhanced_mask = mask_blob.copy()
    
    for label in range(1, num_labels):
        # Get component mask
        component_mask = (labels == label).astype(np.uint8)
        
        # Check overlap with original strong detections
        overlap = component_mask & original_cand
        overlap_ratio = np.sum(overlap) / max(np.sum(component_mask), 1)
        
        # If high overlap with original detection, apply extra dilation
        if overlap_ratio > 0.7:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_component = cv2.dilate(component_mask, kernel, iterations=1)
            enhanced_mask |= dilated_component
            logger.debug(f"Applied extra dilation to strong blob component {label}")
    
    return enhanced_mask.astype(np.uint8)


def refine_classification(mask_thin: np.ndarray, mask_blob: np.ndarray, 
                         img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine classification using gradient information
    
    Args:
        mask_thin: Initial thin mask
        mask_blob: Initial blob mask
        img_bgr: Original BGR image
        
    Returns:
        Refined (mask_thin, mask_blob)
    """
    # Convert to grayscale for gradient computation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Analyze gradient density for blob components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_blob, connectivity=8)
    
    refined_thin = mask_thin.copy()
    refined_blob = mask_blob.copy()
    
    for label in range(1, num_labels):
        component_mask = (labels == label)
        area = stats[label, cv2.CC_STAT_AREA]
        
        # Large area with low gradient density might be false positive (e.g., white label)
        if area > 500:
            mean_gradient = np.mean(grad_mag[component_mask])
            if mean_gradient < 10:  # Low texture
                logger.info(f"Removing low-gradient large blob (area={area}, grad={mean_gradient:.2f})")
                refined_blob[component_mask] = 0
    
    return refined_thin, refined_blob


def merge_close_components(mask: np.ndarray, distance_thresh: int = 5) -> np.ndarray:
    """
    Merge components that are very close to each other
    
    Args:
        mask: Binary mask
        distance_thresh: Maximum distance for merging
        
    Returns:
        Mask with merged components
    """
    # Dilate to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance_thresh, distance_thresh))
    merged = cv2.dilate(mask, kernel, iterations=1)
    
    # Erode back to approximate original size
    merged = cv2.erode(merged, kernel, iterations=1)
    
    return merged.astype(np.uint8)


def get_shape_statistics(mask_thin: np.ndarray, mask_blob: np.ndarray) -> Dict:
    """
    Compute statistics about detected shapes
    
    Args:
        mask_thin: Thin components mask
        mask_blob: Blob components mask
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'thin_pixels': int(np.sum(mask_thin)),
        'blob_pixels': int(np.sum(mask_blob)),
        'total_pixels': int(np.sum(mask_thin | mask_blob)),
        'thin_components': cv2.connectedComponents(mask_thin)[0] - 1,
        'blob_components': cv2.connectedComponents(mask_blob)[0] - 1
    }
    
    return stats


def visualize_shape_classification(img_bgr: np.ndarray, mask_thin: np.ndarray, 
                                   mask_blob: np.ndarray) -> np.ndarray:
    """
    Visualize shape classification results
    
    Args:
        img_bgr: Original image
        mask_thin: Thin components mask
        mask_blob: Blob components mask
        
    Returns:
        Visualization image
    """
    vis = img_bgr.copy()
    
    # Create colored overlay
    overlay = np.zeros_like(img_bgr)
    overlay[mask_thin == 1] = [255, 0, 0]  # Blue for thin
    overlay[mask_blob == 1] = [0, 0, 255]  # Red for blob
    
    # Blend with original
    result = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    
    # Add contours
    thin_contours, _ = cv2.findContours(mask_thin * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_contours, _ = cv2.findContours(mask_blob * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(result, thin_contours, -1, (255, 255, 0), 1)  # Cyan for thin contours
    cv2.drawContours(result, blob_contours, -1, (255, 0, 255), 1)  # Magenta for blob contours
    
    return result
