"""
Command-line interface controller
"""

import argparse
import logging
import sys
import os
from typing import Optional

from app.utils.config import load_config, update_config
from app.controllers.pipeline import run_batch

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Green Container Reflection Detection and Removal System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app --input data/images --config configs/config.yaml
  python -m app --input data/samples --outmask out/mask --outresult out/result
        """
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory containing images (default: from config)',
        default=None
    )
    
    parser.add_argument(
        '--outmask', '-m',
        type=str,
        help='Output directory for masks (default: from config)',
        default=None
    )
    
    parser.add_argument(
        '--outresult', '-r',
        type=str,
        help='Output directory for processed images (default: from config)',
        default=None
    )
    
    parser.add_argument(
        '--outeval', '-e',
        type=str,
        help='Output directory for evaluation metrics (default: from config)',
        default=None
    )
    
    parser.add_argument(
        '--outlogs', '-l',
        type=str,
        help='Output directory for logs (default: from config)',
        default=None
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration YAML file (default: configs/config.yaml)',
        default='configs/config.yaml'
    )
    
    # Processing options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose output and visualizations'
    )
    
    parser.add_argument(
        '--no-panels',
        action='store_true',
        help='Disable saving comparison panels'
    )
    
    parser.add_argument(
        '--single', '-s',
        type=str,
        help='Process single image file instead of directory',
        default=None
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser.parse_args()


def setup_logging(args, log_dir: Optional[str] = None):
    """
    Setup logging configuration based on CLI arguments
    
    Args:
        args: Parsed command-line arguments
        log_dir: Optional log directory
    """
    # Determine log level
    if args.quiet:
        level = logging.ERROR
    elif args.verbose or args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log directory specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'processing.log')
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def validate_paths(config):
    """
    Validate and create necessary directories
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    # Check input directory exists
    input_dir = config.get('paths', {}).get('input_dir')
    if not input_dir or not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return False
    
    # Create output directories
    output_dirs = [
        config.get('paths', {}).get('out_mask'),
        config.get('paths', {}).get('out_result'),
        config.get('paths', {}).get('out_eval'),
        config.get('paths', {}).get('out_logs')
    ]
    
    for dir_path in output_dirs:
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    return True


def main():
    """
    Main entry point for CLI
    """
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Update config with CLI overrides
        overrides = {
            'input_dir': args.input,
            'out_mask': args.outmask,
            'out_result': args.outresult,
            'out_eval': args.outeval,
            'out_logs': args.outlogs
        }
        config = update_config(config, overrides)
        
        # Setup logging
        log_dir = config.get('paths', {}).get('out_logs')
        setup_logging(args, log_dir)
        
        # Update debug settings
        if args.debug:
            config.setdefault('debug', {})['save_panels'] = True
            config['debug']['verbose'] = True
        
        if args.no_panels:
            config.setdefault('debug', {})['save_panels'] = False
        
        # Validate paths
        if not validate_paths(config):
            logger.error("Path validation failed")
            sys.exit(1)
        
        # Log configuration
        logger.info("=" * 60)
        logger.info("Green Container Reflection Removal System")
        logger.info("=" * 60)
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Input directory: {config['paths']['input_dir']}")
        logger.info(f"Output mask: {config['paths']['out_mask']}")
        logger.info(f"Output result: {config['paths']['out_result']}")
        logger.info(f"Output eval: {config['paths']['out_eval']}")
        logger.info("=" * 60)
        
        # Process single image or batch
        if args.single:
            # Import single image processor
            from app.controllers.pipeline import run_one
            
            logger.info(f"Processing single image: {args.single}")
            metrics = run_one(args.single, config)
            
            if metrics:
                logger.info("Processing completed successfully")
                logger.info(f"Metrics: {metrics}")
            else:
                logger.error("Processing failed")
                sys.exit(1)
        else:
            # Process batch
            input_dir = config['paths']['input_dir']
            logger.info(f"Processing batch from: {input_dir}")
            
            run_batch(input_dir, config)
            
            logger.info("Batch processing completed successfully")
        
        # Create summary report
        from app.views.writers import create_summary_report
        csv_path = os.path.join(config['paths']['out_eval'], 'metrics.csv')
        create_summary_report(csv_path, config['paths']['out_eval'])
        
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
