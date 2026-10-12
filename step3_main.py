# Importing dependencies
import logging
from step1_data_handler import DataHandler
from step2_cluster_model import KMeansClustering

# Configure root logger to output execution steps to the console
logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    """
    Orchestrates the end-to-end machine learning pipeline:
    Data Ingestion -> EDA -> Transformation -> Modeling -> Evaluation.
    """
    # Define AWS dataset endpoint and select continuous numerical features for clustering
    aws_dataset_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    target_features = ['trip_distance', 'fare_amount']

    # Initialize modular pipeline components
    handler = DataHandler()
    cluster_engine = KMeansClustering(n_clusters=3)

    # Pipeline Stage 1: Data Ingestion
    raw_dataframe = handler.load_data(aws_dataset_url)

    # Pipeline Stage 2: Exploratory Data Analysis
    handler.explore_data(raw_dataframe, target_features)

    # Pipeline Stage 3: Data Cleansing and Feature Scaling
    clean_dataframe, scaled_matrix = handler.preprocess_data(raw_dataframe, target_features)

    # Pipeline Stage 4: Algorithm Execution and Label Assignment
    clean_dataframe['Behavior_Cluster'] = cluster_engine.execute_clustering(scaled_matrix)

    # Pipeline Stage 5: Validation and Final Output Display
    cluster_engine.display_metrics_table()
    logging.info("--- FINAL CLUSTERED DATA (Sample) ---")
    print(clean_dataframe[target_features + ['Behavior_Cluster']].head(10))


if __name__ == "__main__":
    main()