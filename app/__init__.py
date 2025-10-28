"""
Green Container Reflection Detection and Removal System
MVC Architecture for processing water droplet and specular reflections
"""

__version__ = "1.0.0"
__author__ = "Reflection Removal System"

import logging

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set random seed for reproducibility
import numpy as np
np.random.seed(0)
