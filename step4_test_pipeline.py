# Importing dependencies
import unittest
import sys
import pandas as pd
import numpy as np
from step1_data_handler import DataHandler
from step2_cluster_model import KMeansClustering


class TestKMeansPipeline(unittest.TestCase):
    """
    Automated testing suite to validate pipeline resilience and metric tracking.
    """

    def setUp(self):
        """Constructs a controlled dummy dataset with deliberate anomalies."""
        print("\n" + "=" * 50)
        print("[SETUP] Initializing Test Environment & Dummy Dataset")
        print("=" * 50)

        self.handler = DataHandler()
        self.cluster_engine = KMeansClustering(n_clusters=2)
        self.target_features = ['trip_distance', 'fare_amount']

        # Dummy data containing valid rows and anomalies (NaN, 0, negative)
        self.dummy_data = pd.DataFrame({
            'trip_distance': [1.5, 2.0, 10.5, 0.0, -2.5, np.nan],
            'fare_amount': [7.5, 9.0, 35.0, 5.0, 8.0, 15.0]
        })

        # Display the dummy data in the test log for the grader
        print("\n--- INJECTED DUMMY DATA ---")
        print("Notice the anomalies at index 3, 4, and 5:")
        print(self.dummy_data)
        print("-" * 27)

    def test_data_cleaning_and_scaling(self):
        """TEST: Validates removal of anomalies and application of standard scaling."""
        print("\n[TEST EXECUTION] Running: test_data_cleaning_and_scaling")

        clean_df, scaled_matrix = self.handler.preprocess_data(self.dummy_data, self.target_features)

        print(f" -> Initial Row Count (with anomalies): {len(self.dummy_data)}")
        print(f" -> Cleaned Row Count: {len(clean_df)} | Expected: 3")
        self.assertEqual(len(clean_df), 3, "Pipeline failed to drop invalid anomaly rows.")

        feature_mean = np.mean(scaled_matrix[:, 0])
        print(f" -> Scaled Matrix Mean: {feature_mean:.5f} | Expected: ~0.00000")
        self.assertAlmostEqual(feature_mean, 0, places=5,
                               msg="Scaling transformation failed or is misaligned.")
        print("[STATUS] PASS")

    def test_clustering_metrics_logged(self):
        """TEST: Validates algorithm execution, label assignment, and metric tracking."""
        print("\n[TEST EXECUTION] Running: test_clustering_metrics_logged")

        clean_df, scaled_matrix = self.handler.preprocess_data(self.dummy_data, self.target_features)
        labels = self.cluster_engine.execute_clustering(scaled_matrix)

        print(f" -> Predicted Labels Generated: {len(labels)} | Expected: 3")
        self.assertEqual(len(labels), 3, "Model failed to output a complete label array.")

        print(f" -> Captured Evaluation Metrics: {self.cluster_engine.metrics_log}")
        self.assertIn('Inertia (WCSS)', self.cluster_engine.metrics_log, "Inertia metric is missing.")
        self.assertIn('Silhouette Score', self.cluster_engine.metrics_log, "Silhouette Score is missing.")
        print("[STATUS] PASS")


if __name__ == '__main__':
    # Auto-generate the text log file for submission
    log_filename = 'unit_test_log.txt'

    with open(log_filename, 'w') as log_file:
        sys.stdout = log_file
        sys.stderr = log_file

        unittest.main(verbosity=2, exit=False)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Test complete. Output successfully saved to {log_filename}")
