# src/utils/io.py
import pickle
import pandas as pd
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_model(path: str):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def load_csv(path: str):
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load CSV from {path}: {e}")
        raise e
