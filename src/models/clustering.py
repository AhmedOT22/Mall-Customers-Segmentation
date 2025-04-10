from sklearn.cluster import KMeans
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_kmeans(data, n_clusters=5, init_method='k-means++', random_state=42):
    """
    Train a KMeans clustering model.

    Parameters:
        data (pd.DataFrame): Features to cluster.
        n_clusters (int): Number of clusters.
        init_method (str): Centroid initialization method.
        random_state (int): Random seed.

    Returns:
        model (KMeans): Trained KMeans model.
        labels (np.ndarray): Cluster labels for each sample.
    """
    try:
        model = KMeans(n_clusters=n_clusters, init=init_method, random_state=random_state)
        labels = model.fit_predict(data)
        logger.info(f"KMeans trained successfully with k={n_clusters}")
        return model, labels
    except Exception as e:
        logger.error(f"Failed to train KMeans: {e}")
        raise e
