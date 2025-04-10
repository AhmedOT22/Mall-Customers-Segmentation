import pandas as pd
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(filepath):
    """
    Load data from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If file not found.
        pd.errors.ParserError: If the file cannot be parsed.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {filepath}")
        raise fnf
    except pd.errors.ParserError as pe:
        logger.error(f"Parser error in file: {filepath}")
        raise pe
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise e
