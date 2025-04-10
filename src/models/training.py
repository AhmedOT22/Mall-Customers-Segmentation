import os
import pandas as pd
import pickle
from src.data_processing.data_loader import load_data
from src.models.evaluation import evaluate_k_range, compute_wss
from src.models.clustering import train_kmeans
from src.utils.visualization import plot_metric
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, FIGURE_DIR, DEFAULT_K, K_RANGE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline(
    raw_data_path: str = RAW_DATA_PATH,
    processed_data_path: str = PROCESSED_DATA_PATH,
    model_path: str = MODEL_PATH,
    figure_dir: str = FIGURE_DIR,
    optimal_k: int = DEFAULT_K
):
    """
    Run the full KMeans training pipeline:
    - Load raw customer data
    - Evaluate WSS and silhouette scores
    - Train final KMeans model with selected k
    - Save clustered dataset and serialized model

    Parameters:
        raw_data_path (str): Path to raw input CSV
        processed_data_path (str): Where to store clustered data
        model_path (str): Where to store trained model
        figure_dir (str): Directory for saving plots
        optimal_k (int): Number of clusters to use for final model

    Returns:
        model (KMeans): Trained model
        df (pd.DataFrame): Data with assigned clusters
    """
    try:
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        # Load data
        df = load_data(raw_data_path)
        features = df[['Age', 'Annual_Income', 'Spending_Score']]

        # Evaluate cluster count
        wss_result = compute_wss(features, K_RANGE)
        sil_result = evaluate_k_range(features, K_RANGE)

        # Visualize evaluation
        plot_metric(wss_result['k'], wss_result['wss'], metric_name='WSS Score')
        plot_metric(sil_result['k'], sil_result['silhouette_scores'], metric_name='Silhouette Score')

        # Train model
        model, labels = train_kmeans(features, n_clusters=optimal_k)
        df['Cluster'] = labels

        # Save clustered dataset
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

        return model, df

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise e
