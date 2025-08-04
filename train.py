import os
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import urllib.request

warnings.filterwarnings("ignore")

def prepare_iris_data(file_path):
    """
    Checks if the Iris dataset exists. If not, it downloads and formats it.
    
    Args:
        file_path (str): The expected path to the iris.csv file.
    
    Returns:
        bool: True if the data is ready, False otherwise.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found. Creating it.")
        os.makedirs(directory)

    if not os.path.exists(file_path):
        print(f"File not found at '{file_path}'. Attempting to download...")
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            urllib.request.urlretrieve(url, file_path)
            print("Download complete.")
            column_names = ["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"]
            df = pd.read_csv(file_path, header=None, names=column_names)
            df['variety'] = df['variety'].str.replace('Iris-', '')
            df.to_csv(file_path, index=False)
            print(f"Data formatted and saved to '{file_path}'.")
        except Exception as e:
            print(f"Failed to download or process the data. Error: {e}")
            return False
    return True


def poison_dataframe(df, percentage):
    """Applies random noise poisoning to a given percentage of a DataFrame."""
    if percentage == 0:
        return df.copy()
    X_processed = df.copy()
    num_to_poison = int(len(X_processed) * (percentage / 100))
    poisoned_indices = np.random.choice(X_processed.index, size=num_to_poison, replace=False)
    for col in X_processed.columns:
        min_val = X_processed[col].min()
        max_val = X_processed[col].max()
        X_processed.loc[poisoned_indices, col] = np.random.uniform(min_val, max_val, size=num_to_poison)
    return X_processed


def run_and_log_experiment(X_train, y_train, X_val, y_val, X_test, y_test, level, output_dir="./data"):
    """
    A standard function to run a training experiment, generate predictions,
    and log all relevant information to MLflow.
    """
    with mlflow.start_run(run_name=f"Poisoning_Level_{level}%"):
        print(f"\n--- Logging experiment for {level}% poisoned data ---")
        
        # --- Artifact Preparation ---
        os.makedirs(output_dir, exist_ok=True)
        train_artifact_path = os.path.join(output_dir, f"train_p{level}.csv")
        val_artifact_path = os.path.join(output_dir, f"validation_p{level}.csv")
        X_train.join(y_train).to_csv(train_artifact_path, index=False)
        X_val.join(y_val).to_csv(val_artifact_path, index=False)
        
        # --- MLflow Logging: Parameters and Datasets ---
        mlflow.log_param("poisoning_level", level)
        mlflow.log_artifact(train_artifact_path, "datasets")
        mlflow.log_artifact(val_artifact_path, "datasets")

        # --- Model Training ---
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        # --- Evaluation and Prediction ---
        # Evaluate on poisoned validation set
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Evaluate and generate predictions on the clean test set
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # --- NEW: Save and log test predictions for analysis ---
        predictions_df = X_test.copy()
        predictions_df['true_label'] = y_test
        predictions_df['predicted_label'] = y_test_pred
        
        predictions_artifact_path = os.path.join(output_dir, f"test_predictions_p{level}.csv")
        predictions_df.to_csv(predictions_artifact_path, index=False)
        
        mlflow.log_artifact(predictions_artifact_path, "predictions")
        print(f"  Saved test predictions artifact: 'predictions/test_predictions_p{level}.csv'")
        
        # --- MLflow Logging: Metrics and Model ---
        mlflow.log_metric("validation_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        print(f"  Validation Set (Poisoned): Accuracy={val_accuracy:.4f}")
        print(f"  Test Set (Clean):          Accuracy={test_accuracy:.4f}")

        mlflow.sklearn.log_model(model, "logistic_regression_model")
        print(f"--- Run for level {level}% successfully logged. ---")


def main():
    """Main function to orchestrate the data poisoning experiment."""
    csv_file_path = "./data/iris.csv"
    
    if not prepare_iris_data(csv_file_path):
        return

    tracking_uri = "http://34.63.124.126:8100/"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"\nMLflow tracking server set to: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment("Iris Poisoning - Full Pipeline with Predictions")

    print(f"Reading base data from '{csv_file_path}'...")
    try:
        iris_df = pd.read_csv(csv_file_path)
        X = iris_df.drop("variety", axis=1)
        y = iris_df["variety"]
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    X_train_val, X_test_clean, y_train_val, y_test_clean = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_clean, X_val_clean, y_train_clean, y_val_clean = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print(f"Data split complete: {len(X_train_clean)} train, {len(X_val_clean)} validation, {len(X_test_clean)} test samples.")

    poisoning_levels = [0, 5, 10, 50]
    for level in poisoning_levels:
        X_train_poisoned = poison_dataframe(X_train_clean, level)
        X_val_poisoned = poison_dataframe(X_val_clean, level)
        run_and_log_experiment(
            X_train=X_train_poisoned, y_train=y_train_clean,
            X_val=X_val_poisoned, y_val=y_val_clean,
            X_test=X_test_clean, y_test=y_test_clean,
            level=level
        )

if __name__ == "__main__":
    main()
