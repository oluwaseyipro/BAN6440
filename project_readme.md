# K-Means Transactional Behavior Clustering

## Project Overview
This project implements a machine learning application to segment transactional transportation data using the **K-Means clustering algorithm**. Built in Python, the application ingests the New York City Taxi and Limousine Commission (TLC) Trip Record Data directly from the **AWS Registry of Open Data**. 

By applying the CRISP-DM methodology, the pipeline performs Exploratory Data Analysis (EDA), rigorous anomaly filtering, standard feature scaling, and behavioral clustering to identify distinct customer usage patterns based on trip distance and financial yield.

## Project Structure
The application is built using a modular, object-oriented architecture to ensure enterprise-grade reliability and maintainability.

* `step1_data_handler.py`: Manages data ingestion (Parquet format via AWS), Exploratory Data Analysis (EDA), anomaly filtering (removing negative fares and zero-distance trips), and feature scaling.
* `step2_cluster_model.py`: Isolates the machine learning logic, executing the K-Means algorithm and calculating validation metrics (Inertia and Silhouette Score).
* `step3_main.py`: The primary orchestrator script that executes the end-to-end data pipeline.
* `step4_test_pipeline.py`: The automated unit testing suite utilizing the `unittest` framework to mathematically validate data cleaning and algorithm execution.

## Installation and Prerequisites

1. Ensure Python 3.8+ is installed.
2. Clone this repository or open the project folder in your preferred IDE (e.g., PyCharm).
3. Install the required dependencies using the PyCharm terminal:

```bash
pip install pandas scikit-learn matplotlib seaborn pyarrow fastparquet
```
*(Note: `pyarrow` and `fastparquet` are required for efficient ingestion of the AWS Parquet dataset, especially on Apple Silicon processors).*

## Usage Instructions

### 1. Running the Main Application
To execute the clustering pipeline, run the main orchestrator script:
```bash
python step3_main.py
```
**Expected Output:**
* The script will download the dataset and display a statistical summary.
* An EDA scatter plot will pop up. **You must close this image window** for the pipeline to continue.
* The terminal will output explicit before-and-after states of the data cleaning and scaling process.
* The final clustering metrics (Silhouette Score) and a sample of the clustered dataset will be displayed.

### 2. Running the Unit Tests
To validate the pipeline's resilience against anomalous data (like negative currency values or missing data), run the testing suite:
```bash
python -m unittest step4_test_pipeline.py -v
```
**Expected Output:**
* The terminal will display step-by-step mathematical proofs of the scaling logic and row-count validations, resulting in a successful `OK` status.

## Data Source
* **Dataset:** NYC TLC Trip Record Data (January 2023)
* **Provider:** Registry of Open Data on AWS
* **Format:** Apache Parquet