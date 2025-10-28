"""
Container region extraction using Lab color space and rotated rectangle fitting
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def get_container_mask(img_bgr: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract container mask from green plastic container image
    
    Args:
        img_bgr: Input BGR image
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (M_container[0/1], M_interior[0/1])
    """
    h, w = img_bgr.shape[:2]
    total_pixels = h * w
    
    # Get parameters
    a_threshold = cfg.get('container', {}).get('a_threshold', 126)
    erode_iter = cfg.get('container', {}).get('erode_iter', 1)
    
    # Convert BGR to Lab color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    # Initial green extraction using a* channel
    # In OpenCV Lab, a* ranges from 0-255 where 128 is neutral
    M0 = (a_ch < a_threshold).astype(np.uint8) * 255
    
    # Morphological opening to remove noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    M0 = cv2.morphologyEx(M0, cv2.MORPH_OPEN, kernel_small)
    
    # Find the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(M0, connectivity=8)
    
    if num_labels <= 1:
        logger.warning("No container detected with initial threshold")
        # Try adaptive fallback
        M0 = adaptive_container_detection(img_bgr, lab, a_threshold)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(M0, connectivity=8)
    
    # Get the largest component (excluding background at label 0)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        M_largest = (labels == largest_label).astype(np.uint8) * 255
    else:
        logger.error("Failed to detect container")
        # Return full image masks as fallback
        M_container = np.ones((h, w), dtype=np.uint8)
        M_interior = np.ones((h, w), dtype=np.uint8)
        return M_container, M_interior
    
    # Check if detected region is too small
    container_ratio = np.sum(M_largest > 0) / total_pixels
    if container_ratio < 0.08:
        logger.warning(f"Container too small: {container_ratio:.2%} of image")
        # Try relaxing threshold
        for retry in range(3):
            a_threshold += 1
            logger.info(f"Relaxing a* threshold to {a_threshold}")
            M0 = (a_ch < a_threshold).astype(np.uint8) * 255
            M0 = cv2.morphologyEx(M0, cv2.MORPH_OPEN, kernel_small)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(M0, connectivity=8)
            
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label = np.argmax(areas) + 1
                M_largest = (labels == largest_label).astype(np.uint8) * 255
                container_ratio = np.sum(M_largest > 0) / total_pixels
                
                if container_ratio >= 0.08:
                    break
    
    # Get contours and fit rotated rectangle
    contours, _ = cv2.findContours(M_largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.error("No contours found")
        M_container = np.ones((h, w), dtype=np.uint8)
        M_interior = np.ones((h, w), dtype=np.uint8)
        return M_container, M_interior
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit minimum area rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Create filled rotated rectangle mask
    M_container = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(M_container, box, 1)
    
    # Create interior mask with safety band
    M_interior = M_container.copy()
    if erode_iter > 0:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        M_interior = cv2.erode(M_interior, kernel_erode, iterations=erode_iter)
    
    # Log statistics
    container_pixels = np.sum(M_container)
    interior_pixels = np.sum(M_interior)
    logger.info(f"Container mask: {container_pixels} pixels ({container_pixels/total_pixels:.1%})")
    logger.info(f"Interior mask: {interior_pixels} pixels ({interior_pixels/total_pixels:.1%})")
    
    return M_container.astype(np.uint8), M_interior.astype(np.uint8)


def adaptive_container_detection(img_bgr: np.ndarray, lab: np.ndarray, 
                                initial_threshold: int) -> np.ndarray:
    """
    Adaptive fallback for container detection when initial method fails
    
    Args:
        img_bgr: Input BGR image
        lab: Lab color space image
        initial_threshold: Initial a* threshold that failed
        
    Returns:
        Binary mask of detected container
    """
    h, w = img_bgr.shape[:2]
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    # Try combining multiple cues
    # 1. Relaxed a* threshold
    mask_a = (a_ch < initial_threshold + 3).astype(np.uint8) * 255
    
    # 2. Low saturation in HSV (grayish-green)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mask_sat = (s_ch < 100).astype(np.uint8) * 255
    
    # 3. Mid-range brightness (not too dark, not too bright)
    mask_val = ((v_ch > 30) & (v_ch < 200)).astype(np.uint8) * 255
    
    # Combine masks
    combined = cv2.bitwise_and(mask_a, mask_sat)
    combined = cv2.bitwise_and(combined, mask_val)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    logger.warning("Using adaptive container detection fallback")
    
    return combined


def visualize_container_mask(img_bgr: np.ndarray, M_container: np.ndarray, 
                            M_interior: np.ndarray) -> np.ndarray:
    """
    Visualize container masks for debugging
    
    Args:
        img_bgr: Original BGR image
        M_container: Container mask
        M_interior: Interior mask
        
    Returns:
        Visualization image
    """
    vis = img_bgr.copy()
    
    # Draw container boundary in green
    container_contours, _ = cv2.findContours(M_container * 255, 
                                            cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, container_contours, -1, (0, 255, 0), 2)
    
    # Draw interior boundary in yellow
    interior_contours, _ = cv2.findContours(M_interior * 255, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, interior_contours, -1, (0, 255, 255), 1)
    
    # Add semi-transparent overlay
    overlay = img_bgr.copy()
    overlay[M_interior == 0] = overlay[M_interior == 0] * 0.3
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    return vis
