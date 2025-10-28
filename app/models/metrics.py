"""
Quantitative metrics for reflection removal evaluation
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Any
from scipy import ndimage
from skimage.color import rgb2gray
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def compute_metrics(before_bgr: np.ndarray, after_bgr: np.ndarray,
                   mask_union: np.ndarray, cfg: Dict) -> Dict[str, float]:
    """
    Compute evaluation metrics for reflection removal
    
    Args:
        before_bgr: Original image with reflections
        after_bgr: Image after reflection removal
        mask_union: Binary mask of processed regions
        cfg: Configuration dictionary
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # 1. Saturated Pixel Percentage (SPP)
    spp_before, spp_after = compute_spp(before_bgr, after_bgr)
    metrics['spp_before'] = spp_before
    metrics['spp_after'] = spp_after
    metrics['spp_reduction'] = spp_before - spp_after
    
    # 2. Edge Preservation Ratio (EPR)
    epr_band = cfg.get('eval', {}).get('epr_band', 3)
    epr = compute_epr(before_bgr, after_bgr, mask_union, epr_band)
    metrics['epr'] = epr
    
    # 3. Colorfulness (Hasler-Süsstrunk)
    color_before = compute_colorfulness(before_bgr)
    color_after = compute_colorfulness(after_bgr)
    metrics['colorfulness_before'] = color_before
    metrics['colorfulness_after'] = color_after
    metrics['colorfulness_ratio'] = color_after / max(color_before, 0.001)
    
    # 4. PSNR in masked regions
    psnr_mask = compute_psnr_in_mask(before_bgr, after_bgr, mask_union)
    metrics['psnr_masked'] = psnr_mask
    
    # 5. SSIM (Structural Similarity)
    ssim = compute_ssim(before_bgr, after_bgr)
    metrics['ssim'] = ssim
    
    # 6. Mean brightness change
    brightness_change = compute_brightness_change(before_bgr, after_bgr, mask_union)
    metrics['brightness_change'] = brightness_change
    
    # 7. Contrast preservation
    contrast_ratio = compute_contrast_preservation(before_bgr, after_bgr, mask_union)
    metrics['contrast_preservation'] = contrast_ratio
    
    # Optional: Add BRISQUE/NIQE if libraries available
    try:
        brisque_score = compute_brisque(after_bgr)
        if brisque_score is not None:
            metrics['brisque'] = brisque_score
    except:
        logger.debug("BRISQUE not available")
    
    # Log key metrics
    logger.info(f"SPP: {spp_before:.3%} → {spp_after:.3%} (reduction: {spp_before - spp_after:.3%})")
    logger.info(f"EPR: {epr:.3f}, Colorfulness ratio: {metrics['colorfulness_ratio']:.3f}")
    
    return metrics


def compute_spp(before: np.ndarray, after: np.ndarray, threshold: int = 245) -> tuple:
    """
    Compute Saturated Pixel Percentage
    
    Args:
        before: Original image
        after: Processed image
        threshold: Saturation threshold
        
    Returns:
        Tuple of (spp_before, spp_after)
    """
    # Get maximum value across color channels
    max_before = np.max(before, axis=2)
    max_after = np.max(after, axis=2)
    
    # Count saturated pixels
    total_pixels = before.shape[0] * before.shape[1]
    sat_before = np.sum(max_before >= threshold)
    sat_after = np.sum(max_after >= threshold)
    
    # Calculate percentages
    spp_before = sat_before / total_pixels
    spp_after = sat_after / total_pixels
    
    return spp_before, spp_after


def compute_epr(before: np.ndarray, after: np.ndarray, 
               mask: np.ndarray, band_width: int = 3) -> float:
    """
    Compute Edge Preservation Ratio
    
    Args:
        before: Original image
        after: Processed image
        mask: Binary mask of processed regions
        band_width: Width of boundary band
        
    Returns:
        Edge preservation ratio (ideal ≥ 0.9)
    """
    # Convert to grayscale for edge detection
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    # Create boundary band around masked regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=band_width)
    band = dilated & (~mask)  # XOR to get band only
    
    if np.sum(band) == 0:
        return 1.0  # No boundary to evaluate
    
    # Compute gradients using Sobel
    grad_x_before = cv2.Sobel(gray_before, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_before = cv2.Sobel(gray_before, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_before = np.abs(grad_x_before) + np.abs(grad_y_before)
    
    grad_x_after = cv2.Sobel(gray_after, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_after = cv2.Sobel(gray_after, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_after = np.abs(grad_x_after) + np.abs(grad_y_after)
    
    # Compute edge strength in boundary band
    edge_before = np.sum(grad_mag_before[band == 1])
    edge_after = np.sum(grad_mag_after[band == 1])
    
    # Calculate preservation ratio
    epr = edge_after / max(edge_before, 1.0)
    epr = min(epr, 2.0)  # Cap at 2.0 to handle edge enhancement
    
    return epr


def compute_colorfulness(img_bgr: np.ndarray) -> float:
    """
    Compute Hasler-Süsstrunk colorfulness metric
    
    Args:
        img_bgr: BGR image
        
    Returns:
        Colorfulness score
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    
    # Compute opponent color space
    rg = R.astype(np.float32) - G.astype(np.float32)
    yb = 0.5 * (R.astype(np.float32) + G.astype(np.float32)) - B.astype(np.float32)
    
    # Compute mean and standard deviation
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_std = np.std(rg)
    yb_std = np.std(yb)
    
    # Compute colorfulness
    std_root = np.sqrt(rg_std**2 + yb_std**2)
    mean_root = np.sqrt(rg_mean**2 + yb_mean**2)
    colorfulness = std_root + 0.3 * mean_root
    
    return colorfulness


def compute_psnr_in_mask(before: np.ndarray, after: np.ndarray, 
                         mask: np.ndarray) -> float:
    """
    Compute PSNR only in masked regions
    
    Args:
        before: Original image
        after: Processed image
        mask: Binary mask
        
    Returns:
        PSNR value in dB
    """
    if np.sum(mask) == 0:
        return float('inf')
    
    # Extract masked regions
    masked_before = before[mask == 1]
    masked_after = after[mask == 1]
    
    # Compute MSE
    mse = mean_squared_error(masked_before.flatten(), masked_after.flatten())
    
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def compute_ssim(before: np.ndarray, after: np.ndarray) -> float:
    """
    Compute Structural Similarity Index
    
    Args:
        before: Original image
        after: Processed image
        
    Returns:
        SSIM value (0-1, higher is better)
    """
    # Convert to grayscale
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    # Constants for SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Compute means
    mu1 = cv2.GaussianBlur(gray_before.astype(np.float32), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray_after.astype(np.float32), (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = cv2.GaussianBlur(gray_before.astype(np.float32) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray_after.astype(np.float32) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray_before.astype(np.float32) * gray_after.astype(np.float32), 
                               (11, 11), 1.5) - mu1_mu2
    
    # Compute SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    return np.mean(ssim_map)


def compute_brightness_change(before: np.ndarray, after: np.ndarray, 
                             mask: np.ndarray) -> float:
    """
    Compute mean brightness change in processed regions
    
    Args:
        before: Original image
        after: Processed image
        mask: Binary mask
        
    Returns:
        Mean brightness change
    """
    # Convert to LAB for perceptual brightness
    lab_before = cv2.cvtColor(before, cv2.COLOR_BGR2Lab)
    lab_after = cv2.cvtColor(after, cv2.COLOR_BGR2Lab)
    
    # Extract L channel
    l_before = lab_before[:, :, 0]
    l_after = lab_after[:, :, 0]
    
    # Compute change in masked regions
    if np.sum(mask) > 0:
        mean_change = np.mean(l_after[mask == 1]) - np.mean(l_before[mask == 1])
    else:
        mean_change = 0.0
    
    return mean_change


def compute_contrast_preservation(before: np.ndarray, after: np.ndarray,
                                 mask: np.ndarray) -> float:
    """
    Compute contrast preservation ratio
    
    Args:
        before: Original image
        after: Processed image
        mask: Binary mask
        
    Returns:
        Contrast preservation ratio
    """
    # Convert to grayscale
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    
    # Create region around mask for contrast computation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    region = cv2.dilate(mask, kernel, iterations=1)
    
    if np.sum(region) == 0:
        return 1.0
    
    # Compute local contrast using standard deviation
    contrast_before = np.std(gray_before[region == 1])
    contrast_after = np.std(gray_after[region == 1])
    
    # Compute preservation ratio
    if contrast_before > 0:
        ratio = contrast_after / contrast_before
    else:
        ratio = 1.0
    
    return min(ratio, 2.0)  # Cap at 2.0


def compute_brisque(img_bgr: np.ndarray) -> Optional[float]:
    """
    Compute BRISQUE no-reference image quality score
    
    Args:
        img_bgr: BGR image
        
    Returns:
        BRISQUE score (lower is better) or None if not available
    """
    try:
        from imquality import brisque
        score = brisque(img_bgr)
        return score
    except ImportError:
        logger.debug("BRISQUE library not available")
        return None
    except Exception as e:
        logger.debug(f"BRISQUE computation failed: {e}")
        return None


def format_metrics_for_csv(metrics: Dict[str, float], filename: str) -> Dict[str, Any]:
    """
    Format metrics dictionary for CSV output
    
    Args:
        metrics: Raw metrics dictionary
        filename: Image filename
        
    Returns:
        Formatted dictionary for CSV
    """
    formatted = {'filename': filename}
    
    # Add metrics with proper formatting
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'spp' in key or 'ratio' in key or 'epr' in key:
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = value
    
    return formatted
