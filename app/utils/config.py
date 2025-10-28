"""
Configuration loader and validator
"""

import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Validate required sections
        required_sections = ['paths', 'container', 'detect', 'shape', 'inpaint', 'eval']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")
                config[section] = {}
        
        # Set defaults for critical values
        _set_defaults(config)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def _set_defaults(config: Dict[str, Any]) -> None:
    """Set default values for missing configuration parameters"""
    
    # Path defaults
    if 'paths' not in config:
        config['paths'] = {}
    path_defaults = {
        'input_dir': 'data/images',
        'out_mask': 'out/mask',
        'out_result': 'out/result',
        'out_eval': 'out/eval',
        'out_logs': 'out/logs'
    }
    for key, value in path_defaults.items():
        config['paths'].setdefault(key, value)
    
    # Container defaults
    if 'container' not in config:
        config['container'] = {}
    config['container'].setdefault('a_threshold', 126)
    config['container'].setdefault('erode_iter', 1)
    
    # Detection defaults
    if 'detect' not in config:
        config['detect'] = {}
    detect_defaults = {
        'z_sigma': 11,
        'z_thresh': 2.0,
        'sat_cut': 245,
        's_thresh': 40,
        'rgb_range_thresh': 25,
        'min_area': 20
    }
    for key, value in detect_defaults.items():
        config['detect'].setdefault(key, value)
    
    # Shape defaults
    if 'shape' not in config:
        config['shape'] = {}
    shape_defaults = {
        'thin_min_short': 8,
        'thin_aspect_min': 4.0,
        'thin_area_max': 400,
        'dilate_thin': 1,
        'dilate_blob': 2
    }
    for key, value in shape_defaults.items():
        config['shape'].setdefault(key, value)
    
    # Inpaint defaults
    if 'inpaint' not in config:
        config['inpaint'] = {}
    config['inpaint'].setdefault('radius', 3)
    config['inpaint'].setdefault('feather', 2)
    
    # Eval defaults
    if 'eval' not in config:
        config['eval'] = {}
    config['eval'].setdefault('epr_band', 3)
    
    # Debug defaults
    if 'debug' not in config:
        config['debug'] = {}
    config['debug'].setdefault('save_panels', True)


def update_config(config: Dict[str, Any], overrides: Dict[str, str]) -> Dict[str, Any]:
    """
    Update configuration with command-line overrides
    
    Args:
        config: Base configuration dictionary
        overrides: Dictionary of path overrides from CLI
        
    Returns:
        Updated configuration dictionary
    """
    if overrides:
        if 'paths' not in config:
            config['paths'] = {}
        
        for key, value in overrides.items():
            if value is not None:
                config['paths'][key] = value
                logger.info(f"Override config.paths.{key} = {value}")
    
    return config
