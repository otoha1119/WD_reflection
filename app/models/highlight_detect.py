"""
Reflection candidate detection using local Z-score and color analysis
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def detect_candidates(img_bgr: np.ndarray, hsv: np.ndarray,
                     m_interior: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Detect reflection candidates using local Z-score and whiteness criteria
    
    Args:
        img_bgr: Input BGR image
        hsv: HSV color space image
        m_interior: Interior mask [0/1]
        cfg: Configuration dictionary
        
    Returns:
        Binary mask of candidates [0/1]
    """
    # Get configuration parameters
    z_sigma = cfg.get('detect', {}).get('z_sigma', 11)
    z_thresh = cfg.get('detect', {}).get('z_thresh', 2.0)
    sat_cut = cfg.get('detect', {}).get('sat_cut', 245)
    s_thresh = cfg.get('detect', {}).get('s_thresh', 40)
    rgb_range_thresh = cfg.get('detect', {}).get('rgb_range_thresh', 25)
    min_area = cfg.get('detect', {}).get('min_area', 20)
    
    # Extract channels
    h_ch, s_ch, v_ch = cv2.split(hsv)
    b_ch, g_ch, r_ch = cv2.split(img_bgr)
    
    # Compute local Z-score for V channel
    z_score = compute_local_zscore(v_ch, sigma=z_sigma)
    
    # Whiteness gate: low saturation OR low RGB range
    rgb_max = np.maximum(r_ch, np.maximum(g_ch, b_ch))
    rgb_min = np.minimum(r_ch, np.minimum(g_ch, b_ch))
    rgb_range = rgb_max - rgb_min
    
    whiteish = (s_ch < s_thresh) | (rgb_range < rgb_range_thresh)
    
    # White saturation: any channel near maximum
    saturated = (rgb_max >= sat_cut)
    
    # Combine conditions
    # High Z-score AND whitish AND inside container
    z_candidates = (z_score > z_thresh) & whiteish & (m_interior == 1)
    
    # OR saturated pixels inside container
    sat_candidates = saturated & (m_interior == 1)
    
    # Combine both conditions
    candidates = z_candidates | sat_candidates
    
    # Convert to uint8 binary mask
    cand_mask = candidates.astype(np.uint8)
    
    # Remove small noise with morphological opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Remove small connected components
    cand_mask = remove_small_components(cand_mask, min_area)
    
    # Check if detection is too weak and adjust if needed
    detected_pixels = np.sum(cand_mask)
    total_interior = np.sum(m_interior)
    detection_ratio = detected_pixels / max(total_interior, 1)
    
    if detection_ratio < 0.001:  # Less than 0.1% detected
        logger.warning(f"Weak detection: {detection_ratio:.3%}, relaxing thresholds")
        cand_mask = adaptive_detection(img_bgr, hsv, m_interior, z_score, cfg)
    
    # Log detection statistics
    final_pixels = np.sum(cand_mask)
    final_ratio = final_pixels / max(total_interior, 1)
    logger.info(f"Detected {final_pixels} reflection pixels ({final_ratio:.2%} of container)")
    
    return cand_mask


def compute_local_zscore(v_channel: np.ndarray, sigma: int = 11) -> np.ndarray:
    """
    Compute local Z-score for brightness channel
    
    Args:
        v_channel: Value/brightness channel
        sigma: Gaussian blur sigma for local statistics
        
    Returns:
        Z-score array
    """
    # Ensure sigma is odd
    if sigma % 2 == 0:
        sigma += 1
    
    # Convert to float for precision
    v_float = v_channel.astype(np.float32)
    
    # Compute local mean
    ksize = (sigma, sigma)
    mu = cv2.GaussianBlur(v_float, ksize, sigma)
    
    # Compute local standard deviation
    v_squared = v_float ** 2
    mu_squared = cv2.GaussianBlur(v_squared, ksize, sigma)
    variance = mu_squared - mu ** 2
    
    # Avoid negative variance due to numerical errors
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance + 1e-6)  # Add epsilon to avoid division by zero
    
    # Compute Z-score
    z_score = (v_float - mu) / (std + 1e-6)
    
    return z_score


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components smaller than min_area
    
    Args:
        mask: Binary mask [0/1]
        min_area: Minimum area threshold
        
    Returns:
        Cleaned mask
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create output mask
    clean_mask = np.zeros_like(mask)
    
    # Keep only components larger than min_area
    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == label] = 1
    
    return clean_mask.astype(np.uint8)


def adaptive_detection(img_bgr: np.ndarray, hsv: np.ndarray, 
                       m_interior: np.ndarray, z_score: np.ndarray, 
                       cfg: Dict) -> np.ndarray:
    """
    Adaptive detection with relaxed thresholds when initial detection is weak
    
    Args:
        img_bgr: Input BGR image
        hsv: HSV color space image
        m_interior: Interior mask
        z_score: Pre-computed Z-score
        cfg: Configuration dictionary
        
    Returns:
        Adjusted candidate mask
    """
    # Relax thresholds
    z_thresh_relaxed = max(1.8, cfg.get('detect', {}).get('z_thresh', 2.0) - 0.2)
    s_thresh_relaxed = min(50, cfg.get('detect', {}).get('s_thresh', 40) + 10)
    
    logger.info(f"Relaxed thresholds: z_thresh={z_thresh_relaxed}, s_thresh={s_thresh_relaxed}")
    
    # Re-detect with relaxed thresholds
    h_ch, s_ch, v_ch = cv2.split(hsv)
    b_ch, g_ch, r_ch = cv2.split(img_bgr)
    
    rgb_max = np.maximum(r_ch, np.maximum(g_ch, b_ch))
    rgb_min = np.minimum(r_ch, np.minimum(g_ch, b_ch))
    rgb_range = rgb_max - rgb_min
    
    # More permissive whiteness
    whiteish = (s_ch < s_thresh_relaxed) | (rgb_range < 30)
    
    # Lower Z-score threshold
    candidates = (z_score > z_thresh_relaxed) & whiteish & (m_interior == 1)
    
    # Still include saturated pixels
    saturated = (rgb_max >= 240)  # Slightly lower threshold
    candidates = candidates | (saturated & (m_interior == 1))
    
    cand_mask = candidates.astype(np.uint8)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_OPEN, kernel)
    cand_mask = remove_small_components(cand_mask, max(10, cfg.get('detect', {}).get('min_area', 20) // 2))
    
    return cand_mask


def filter_edge_components(cand_mask: np.ndarray, m_interior: np.ndarray, 
                          distance_thresh: int = 3) -> np.ndarray:
    """
    Remove components too close to container edges
    
    Args:
        cand_mask: Candidate mask
        m_interior: Interior mask
        distance_thresh: Minimum distance from edge
        
    Returns:
        Filtered mask
    """
    # Compute distance transform from edges
    dist_transform = cv2.distanceTransform(m_interior, cv2.DIST_L2, 5)
    
    # Remove candidates too close to edges
    edge_safe = dist_transform >= distance_thresh
    filtered = cand_mask & edge_safe.astype(np.uint8)
    
    return filtered


def visualize_detection(img_bgr: np.ndarray, cand_mask: np.ndarray, 
                       z_score: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualize detection results for debugging
    
    Args:
        img_bgr: Original image
        cand_mask: Detected candidates
        z_score: Optional Z-score map
        
    Returns:
        Visualization image
    """
    vis = img_bgr.copy()
    
    # Highlight detected regions in red
    vis[cand_mask == 1] = [0, 0, 255]
    
    # Blend with original
    result = cv2.addWeighted(img_bgr, 0.7, vis, 0.3, 0)
    
    # Add contours
    contours, _ = cv2.findContours(cand_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)
    
    return result
