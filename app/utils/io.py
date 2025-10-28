"""
Image I/O utilities with EXIF rotation support
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def get_image_files(input_dir: str) -> List[str]:
    """
    Get list of image files from directory
    
    Args:
        input_dir: Directory containing images
        
    Returns:
        List of absolute paths to image files
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = []
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return image_files
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1]
            if ext in valid_extensions:
                image_files.append(os.path.abspath(file_path))
    
    image_files.sort()
    logger.info(f"Found {len(image_files)} image files in {input_dir}")
    return image_files


def read_image_with_exif(image_path: str) -> Optional[np.ndarray]:
    """
    Read image with EXIF rotation correction
    
    Args:
        image_path: Path to image file
        
    Returns:
        BGR image array or None if failed
    """
    try:
        # First try to read EXIF data using PIL
        pil_img = Image.open(image_path)
        
        # Check for EXIF orientation
        orientation = None
        if hasattr(pil_img, '_getexif'):
            exif = pil_img._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        orientation = value
                        break
        
        # Read image with OpenCV
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        # Apply rotation based on EXIF orientation
        if orientation:
            img_bgr = apply_exif_rotation(img_bgr, orientation)
        
        logger.debug(f"Loaded image: {image_path}, shape: {img_bgr.shape}")
        return img_bgr
        
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return None


def apply_exif_rotation(img: np.ndarray, orientation: int) -> np.ndarray:
    """
    Apply rotation based on EXIF orientation tag
    
    Args:
        img: Input image array
        orientation: EXIF orientation value (1-8)
        
    Returns:
        Rotated image array
    """
    if orientation == 1:
        return img
    elif orientation == 2:  # Mirror horizontal
        return cv2.flip(img, 1)
    elif orientation == 3:  # Rotate 180
        return cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 4:  # Mirror vertical
        return cv2.flip(img, 0)
    elif orientation == 5:  # Mirror horizontal and rotate 270
        img = cv2.flip(img, 1)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:  # Rotate 90
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:  # Mirror horizontal and rotate 90
        img = cv2.flip(img, 1)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:  # Rotate 270
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def get_output_path(input_path: str, output_dir: str, extension: Optional[str] = None) -> str:
    """
    Generate output path based on input filename
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        extension: New extension (optional)
        
    Returns:
        Output file path
    """
    filename = os.path.basename(input_path)
    if extension:
        filename = os.path.splitext(filename)[0] + extension
    
    return os.path.join(output_dir, filename)


def validate_image(img: np.ndarray) -> bool:
    """
    Validate image array
    
    Args:
        img: Image array to validate
        
    Returns:
        True if valid, False otherwise
    """
    if img is None:
        return False
    
    if img.size == 0:
        logger.warning("Image has zero size")
        return False
    
    if len(img.shape) != 3 or img.shape[2] != 3:
        logger.warning(f"Invalid image shape: {img.shape}")
        return False
    
    if img.shape[0] < 10 or img.shape[1] < 10:
        logger.warning(f"Image too small: {img.shape}")
        return False
    
    return True


def resize_if_needed(img: np.ndarray, max_size: int = 2048) -> Tuple[np.ndarray, float]:
    """
    Resize image if larger than max_size while maintaining aspect ratio
    
    Args:
        img: Input image
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > max_size:
        scale = max_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {(w,h)} to {(new_w,new_h)}")
        return resized, scale
    
    return img, 1.0
