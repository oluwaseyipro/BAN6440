Project Overview
This repository contains a modular, enterprise-grade Python application developed to segment transportation transactions using machine learning. Following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, the application ingests raw cloud data, performs Exploratory Data Analysis (EDA), sanitizes anomalies, and applies the K-Means clustering algorithm to uncover distinct customer behavioral patterns based on trip distance and financial yield.

Data Source
The dataset is sourced directly from the AWS Registry of Open Data.

Dataset: New York City Taxi and Limousine Commission (TLC) Trip Record Data (January 2023).

Format: Highly optimized Parquet format, retrieved dynamically via public HTTPS distribution.

Repository Architecture
The application is structured into four sequential, object-oriented modules to ensure maintainability and separation of concerns:

step1_data_handler.py: Manages data ingestion, visual EDA, programmatic anomaly purging (e.g., negative fares and zero-distance trips), and standard feature scaling.

step2_cluster_model.py: Encapsulates the K-Means machine learning logic, assigns behavioral labels, and captures validation metrics (Silhouette Score and Inertia).

step3_main.py: The central orchestrator that executes the end-to-end data pipeline from ingestion to final metric output.

step4_test_pipeline.py: An automated unittest suite that validates pipeline resilience using controlled dummy datasets injected with edge-case anomalies.

Prerequisites & Installation
This project was developed using PyCharm. To execute the code locally, ensure you have Python 3.8+ installed, then install the required dependencies via the terminal:

Bash
pip install pandas scikit-learn matplotlib seaborn pyarrow fastparquet
(Note: pyarrow and fastparquet are required for efficient Parquet file decoding, particularly on Apple Silicon architectures).

Execution Guide
1. Running the Main Pipeline
Execute the orchestrator script to run the full clustering pipeline:

Bash
python step3_main.py
Note on Execution: The script will pause during the pipeline to display a scatter plot for Exploratory Data Analysis. You must close the plot window for the script to resume data cleaning and algorithm execution.

2. Running the Unit Tests
To validate the mathematical integrity of the preprocessing and scaling logic, execute the test suite:

Bash
python -m unittest step4_test_pipeline.py -v
Business Insights & Results
The algorithm successfully segmented the transactions with a highly optimized Silhouette Score of 0.7188, mathematically validating the presence of three tightly defined behavioral clusters:

Cluster 0 (Micro-Transactions): High-frequency, short-distance trips (~1 mile) with standard baseline fares representing daily convenience travel.

Cluster 1 (Standard Commutes): Mid-range travel (~4.8 miles) with higher financial yields representing consistent commuter behavior.

Cluster 2 (Premium Outliers): Long-distance, premium-fare trips that represent lucrative edge-case usage.
