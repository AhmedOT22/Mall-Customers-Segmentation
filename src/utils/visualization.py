import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)

def plot_metric(k_values, scores, metric_name='Silhouette Score', save_path=None):
    """
    Plot a clustering metric (e.g. WSS, silhouette) over different k values.

    Parameters:
        k_values (list): List of k values.
        scores (list): Corresponding scores.
        metric_name (str): Y-axis label and plot title.
        save_path (str, optional): Path to save the figure.
    
    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
    """
    try:
        df = pd.DataFrame({'k': k_values, metric_name: scores})
        fig, ax = plt.subplots()
        df.plot(x='k', y=metric_name, marker='o', ax=ax)
        ax.set_title(f'{metric_name} vs Number of Clusters')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel(metric_name)
        ax.grid(True)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            logger.info(f"{metric_name} plot saved to {save_path}")

        plt.show()
        logger.info(f"{metric_name} plot generated.")
        return fig

    except Exception as e:
        logger.error(f"Failed to plot {metric_name}: {e}")
        raise e

def plot_scatter_clusters(df, x, y, label_col='Cluster', title='Cluster Scatter', save_path=None):
    """
    Plot a 2D scatter plot of clusters.

    Parameters:
        df (pd.DataFrame): Data with features and cluster labels.
        x (str): x-axis feature.
        y (str): y-axis feature.
        label_col (str): Column for hue.
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
    """
    try:
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue=label_col, data=df, palette='colorblind', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            logger.info(f"{title} saved to {save_path}")

        plt.show()
        logger.info(f"{title} generated.")
        return fig

    except Exception as e:
        logger.error(f"Error in scatter plot: {e}")
        raise e
