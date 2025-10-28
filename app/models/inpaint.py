"""
Inpainting module for reflection removal using Lab color space
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def inpaint_lab(img_bgr: np.ndarray, mask_thin: np.ndarray, mask_blob: np.ndarray,
                cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove reflections using inpainting in Lab color space
    
    Args:
        img_bgr: Input BGR image
        mask_thin: Binary mask for thin reflections [0/1]
        mask_blob: Binary mask for blob reflections [0/1]
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (img_out_bgr, mask_union[0/1])
    """
    # Get configuration parameters
    radius = cfg.get('inpaint', {}).get('radius', 3)
    feather_width = cfg.get('inpaint', {}).get('feather', 2)
    
    # Create union mask for all reflections
    mask_union = (mask_thin | mask_blob).astype(np.uint8)
    
    # Convert masks to 8-bit (0 or 255) for OpenCV inpaint
    mask_thin_255 = mask_thin * 255
    mask_blob_255 = mask_blob * 255
    mask_union_255 = mask_union * 255
    
    # Convert BGR to Lab color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    # Store original L channel for feathering
    l_original = l_ch.copy()
    
    # Process L channel (brightness) - most important for reflection removal
    # Step 1: Inpaint thin reflections with Navier-Stokes
    if np.any(mask_thin):
        l_ch = cv2.inpaint(l_ch, mask_thin_255, radius, flags=cv2.INPAINT_NS)
        logger.debug(f"Inpainted {np.sum(mask_thin)} thin pixels in L channel")
    
    # Step 2: Inpaint blob reflections with Telea on the result
    if np.any(mask_blob):
        l_ch = cv2.inpaint(l_ch, mask_blob_255, radius, flags=cv2.INPAINT_TELEA)
        logger.debug(f"Inpainted {np.sum(mask_blob)} blob pixels in L channel")
    
    # Process a and b channels (color) - lighter touch to preserve color
    # Only process blob areas to avoid color bleeding
    if np.any(mask_blob):
        # Use smaller radius for color channels
        color_radius = max(1, radius - 1)
        a_ch = cv2.inpaint(a_ch, mask_blob_255, color_radius, flags=cv2.INPAINT_TELEA)
        b_ch = cv2.inpaint(b_ch, mask_blob_255, color_radius, flags=cv2.INPAINT_TELEA)
        logger.debug("Inpainted color channels for blob areas")
    
    # Apply boundary feathering for smoother transitions
    if feather_width > 0:
        l_ch = apply_feathering(l_original, l_ch, mask_union, feather_width)
    
    # Reconstruct Lab image
    lab_inpainted = cv2.merge([l_ch, a_ch, b_ch])
    
    # Convert back to BGR
    img_out_bgr = cv2.cvtColor(lab_inpainted, cv2.COLOR_Lab2BGR)
    
    # Post-processing: reduce any remaining artifacts
    img_out_bgr = post_process_inpaint(img_bgr, img_out_bgr, mask_union, cfg)
    
    # Log statistics
    total_pixels = mask_union.shape[0] * mask_union.shape[1]
    inpainted_pixels = np.sum(mask_union)
    logger.info(f"Inpainted {inpainted_pixels} pixels ({inpainted_pixels/total_pixels:.2%} of image)")
    
    return img_out_bgr, mask_union


def apply_feathering(original: np.ndarray, inpainted: np.ndarray, 
                    mask: np.ndarray, feather_width: int) -> np.ndarray:
    """
    Apply feathering at mask boundaries for smooth transition
    
    Args:
        original: Original channel values
        inpainted: Inpainted channel values
        mask: Binary mask of inpainted regions
        feather_width: Width of feather band in pixels
        
    Returns:
        Feathered result
    """
    # Create boundary band
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=feather_width)
    boundary = dilated - mask
    
    # Compute distance transform for smooth weights
    dist_transform = cv2.distanceTransform(boundary, cv2.DIST_L2, 5)
    
    # Normalize to 0-1 range
    if dist_transform.max() > 0:
        weights = dist_transform / dist_transform.max()
    else:
        weights = dist_transform
    
    # Apply weighted blending in boundary region
    result = inpainted.copy()
    boundary_mask = boundary == 1
    result[boundary_mask] = (weights[boundary_mask] * inpainted[boundary_mask] + 
                             (1 - weights[boundary_mask]) * original[boundary_mask])
    
    return result.astype(np.uint8)


def post_process_inpaint(original: np.ndarray, inpainted: np.ndarray, 
                        mask: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Post-process inpainted image to reduce artifacts
    
    Args:
        original: Original BGR image
        inpainted: Inpainted BGR image
        mask: Mask of inpainted regions
        cfg: Configuration dictionary
        
    Returns:
        Post-processed image
    """
    result = inpainted.copy()
    
    # Apply slight Gaussian blur only to inpainted regions
    if np.any(mask):
        # Create slightly dilated mask for smoother edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Blur the inpainted regions
        blurred = cv2.GaussianBlur(inpainted, (3, 3), 0.5)
        
        # Blend only in masked regions
        mask_3ch = np.stack([mask_dilated] * 3, axis=2)
        result = np.where(mask_3ch, blurred, inpainted)
    
    # Preserve image statistics
    result = preserve_global_statistics(original, result, mask)
    
    return result.astype(np.uint8)


def preserve_global_statistics(original: np.ndarray, processed: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
    """
    Preserve global image statistics after inpainting
    
    Args:
        original: Original image
        processed: Processed image
        mask: Inpainted regions mask
        
    Returns:
        Adjusted image
    """
    # Get regions outside mask
    valid_mask = (mask == 0)
    
    if np.sum(valid_mask) < 100:  # Not enough valid pixels
        return processed
    
    result = processed.copy().astype(np.float32)
    
    for c in range(3):  # For each color channel
        # Compute statistics in non-masked regions
        orig_mean = np.mean(original[:, :, c][valid_mask])
        orig_std = np.std(original[:, :, c][valid_mask])
        
        proc_mean = np.mean(result[:, :, c][valid_mask])
        proc_std = np.std(result[:, :, c][valid_mask]) + 1e-6
        
        # Adjust to match original statistics
        if proc_std > 0:
            alpha = orig_std / proc_std
            beta = orig_mean - alpha * proc_mean
            
            # Apply linear transformation
            result[:, :, c] = alpha * result[:, :, c] + beta
    
    # Clip values to valid range
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def adaptive_inpaint(img_bgr: np.ndarray, mask: np.ndarray, 
                    method: str = 'mixed') -> np.ndarray:
    """
    Adaptive inpainting with method selection based on mask characteristics
    
    Args:
        img_bgr: Input BGR image
        mask: Binary mask [0/255]
        method: 'ns', 'telea', or 'mixed'
        
    Returns:
        Inpainted image
    """
    if method == 'mixed':
        # Analyze mask to choose method
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask // 255, connectivity=8)
        
        result = img_bgr.copy()
        
        for label in range(1, num_labels):
            component_mask = ((labels == label) * 255).astype(np.uint8)
            area = stats[label, cv2.CC_STAT_AREA]
            
            # Use NS for small areas, Telea for large
            if area < 100:
                result = cv2.inpaint(result, component_mask, 3, cv2.INPAINT_NS)
            else:
                result = cv2.inpaint(result, component_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    elif method == 'ns':
        return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_NS)
    else:  # telea
        return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)


def visualize_inpaint_result(original: np.ndarray, inpainted: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
    """
    Create visualization comparing original and inpainted
    
    Args:
        original: Original image
        inpainted: Inpainted result
        mask: Inpaint mask
        
    Returns:
        Side-by-side comparison image
    """
    h, w = original.shape[:2]
    
    # Create side-by-side view
    vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
    vis[:, :w] = original
    vis[:, w:] = inpainted
    
    # Draw mask overlay on left side
    mask_colored = np.zeros_like(original)
    mask_colored[:, :, 2] = mask * 255  # Red channel
    vis[:, :w] = cv2.addWeighted(vis[:, :w], 0.7, mask_colored, 0.3, 0)
    
    # Add text labels
    cv2.putText(vis, "Original + Mask", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Inpainted", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis
