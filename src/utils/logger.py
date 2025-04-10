# src/utils/logger.py

import logging

def get_logger(name=__name__):
    """
    Returns a standardized logger instance.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)
