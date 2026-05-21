import logging
import sys
from .config import LOGGING_LEVEL

def setup_logger(name: str = "forest_audio"):
    logger = logging.getLogger(name)
    
    # Set logging level from config
    level = getattr(logging, LOGGING_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # Prevent adding handlers multiple times if setup is called twice
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger

# Default logger instance
logger = setup_logger()
