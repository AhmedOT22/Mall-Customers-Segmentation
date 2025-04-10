from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_wss(data, k_range, init_method='k-means++', random_state=42):
    """
    Compute Within-Cluster Sum of Squares (WSS) for a range of k.

    Parameters:
        data (pd.DataFrame): Feature set.
        k_range (range): Range of cluster values.
        init_method (str): Centroid initialization.
        random_state (int): Seed.

    Returns:
        dict: {'k': [...], 'wss': [...]}
    """
    wss = []
    try:
        for k in k_range:
            model = KMeans(n_clusters=k, init=init_method, random_state=random_state)
            model.fit(data)
            wss.append(model.inertia_)
            logger.info(f"WSS for k={k}: {model.inertia_:.2f}")
    except Exception as e:
        logger.error(f"Error during WSS computation: {e}")
        raise e
    return {'k': list(k_range), 'wss': wss}

def evaluate_k_range(data, k_range, init_method='k-means++', random_state=42):
    """
    Compute Silhouette Score for a range of k.

    Parameters:
        data (pd.DataFrame): Feature set.
        k_range (range): Range of cluster values.
        init_method (str): Centroid initialization.
        random_state (int): Seed.

    Returns:
        dict: {'k': [...], 'silhouette_scores': [...]}
    """
    silhouette_scores = []
    try:
        for k in k_range:
            model = KMeans(n_clusters=k, init=init_method, random_state=random_state)
            labels = model.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
            logger.info(f"Silhouette Score for k={k}: {score:.4f}")
    except Exception as e:
        logger.error(f"Error during silhouette evaluation: {e}")
        raise e
    return {'k': list(k_range), 'silhouette_scores': silhouette_scores}
