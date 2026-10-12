# Importing dependencies
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class DataHandler:
    """
    Manages the initial stages of the data science pipeline: Ingestion,
    Exploratory Data Analysis (EDA), and Feature Transformation.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, url):
        """
        Ingests the dataset from the AWS Open Data Registry.
        """
        logging.info("\n[ACTION] Downloading dataset from AWS Open Data...")
        df = pd.read_parquet(url)
        logging.info(f"--- RAW DATA LOADED: {len(df)} rows ---")
        return df

    def explore_data(self, df, features):
        """
        Executes Exploratory Data Analysis (EDA) to identify structural anomalies.
        """
        print("\n" + "=" * 60)
        print("             EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)

        print("\n--- DATASET OVERVIEW ---")
        df.info()

        logging.info("\n[ACTION] Generating EDA Scatter Plot. Close image window to continue...")
        plt.figure(figsize=(10, 6))

        sample_df = df.sample(min(5000, len(df)), random_state=42)
        sns.scatterplot(x='trip_distance', y='fare_amount', data=sample_df, alpha=0.5)
        plt.title('EDA: Trip Distance vs. Fare Amount (Raw Data)')
        plt.xlabel('Trip Distance (Miles)')
        plt.ylabel('Fare Amount ($)')
        plt.show()

    def preprocess_data(self, df, features):
        """
        Executes and explicitly logs the cleaning and transformation transformations.
        """
        print("\n" + "=" * 60)
        print("          DATA CLEANING & SCALING TRANSFORMATION")
        print("=" * 60)

        # --- BEFORE CLEANING ---
        print("\n[STATE 1: BEFORE CLEANING]")
        print(f"Row Count: {len(df)}")
        print("Min/Max Values (Notice the negative/zero anomalies):")
        print(df[features].describe().loc[['min', 'max']])

        # Action: Clean Data
        df_clean = df.dropna(subset=features).copy()
        df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['fare_amount'] > 0)]

        if len(df_clean) > 10000:
            df_clean = df_clean.sample(10000, random_state=42)

        # --- AFTER CLEANING ---
        print("\n[STATE 2: AFTER CLEANING]")
        print(f"Row Count: {len(df_clean)}")
        print("Min/Max Values (Anomalies successfully purged):")
        print(df_clean[features].describe().loc[['min', 'max']])

        # --- BEFORE SCALING ---
        print("\n[STATE 3: BEFORE SCALING (Raw Features)]")
        X = df_clean[features]
        print(X.head())

        # Action: Scale Features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

        # --- AFTER SCALING ---
        print("\n[STATE 4: AFTER SCALING (Transformed Features)]")
        print(X_scaled_df.head())
        print("\nMathematical Proof of Scaling (Mean ≈ 0, Std ≈ 1):")
        print(X_scaled_df.describe().loc[['mean', 'std']].round(5))

        return df_clean, X_scaled