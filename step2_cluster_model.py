# Importing dependencies
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress standard scikit-learn warnings to maintain a clean execution log
warnings.filterwarnings("ignore", category=FutureWarning)


class KMeansClustering:
    """
    Encapsulates the machine learning modeling and evaluation phase.
    Isolates the algorithm logic from data processing to maintain modularity.
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.metrics_log = {}

    def execute_clustering(self, X_scaled):
        """Fits K-Means, assigns labels, and computes validation metrics."""
        logging.info(f"\n[ACTION] Executing K-Means with {self.n_clusters} clusters...")

        labels = self.model.fit_predict(X_scaled)

        inertia = self.model.inertia_
        sil_score = silhouette_score(X_scaled, labels)

        # Log metrics to satisfy strict assignment tracking requirements
        self.metrics_log = {
            'Algorithm': 'K-Means',
            'Clusters': self.n_clusters,
            'Inertia (WCSS)': round(inertia, 2),
            'Silhouette Score': round(sil_score, 4)
        }

        return labels

    def display_metrics_table(self):
        """Outputs the captured validation metrics in a formatted table."""
        logging.info("\n--- EXPERIMENT METRICS TABLE ---")
        metrics_df = pd.DataFrame([self.metrics_log])
        print(metrics_df.to_string(index=False))
        print("--------------------------------\n")
