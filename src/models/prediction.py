import pandas as pd
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(model, input_data: dict, features: list):
    """
    Predicts the cluster for a single input.

    Parameters:
        model: Trained clustering model.
        input_data (dict): Dictionary with input features (e.g., {'Age': 32, ...})
        features (list): List of features to use for prediction.

    Returns:
        int: Predicted cluster label.
    """
    try:
        input_df = pd.DataFrame([input_data])[features]
        prediction = model.predict(input_df)[0]
        logger.info(f"Prediction made successfully: Cluster {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e
