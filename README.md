# Investigating the Impact of Data Poisoning on a Classification Model

This repository contains a **complete, reproducible pipeline** that demonstrates how different levels of data poisoning affect the performance of a logistic‑regression classifier trained on the classic *Iris* dataset.
All experiments are logged to **MLflow**, making it easy to compare runs, inspect artifacts, and reproduce results.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Data‑Poisoning Strategy](#data-poisoning-strategy)
7. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
8. [Mitigation Strategy](#Mitigation-Strategy)

---

## Project Overview

We simulate a common security scenario: **malicious corruption of training data**.
Starting from pristine *Iris* measurements, a configurable *poisoning function* injects random noise into a chosen percentage of the training and validation sets.
We then:

1. Train a **Logistic Regression** classifier on the (potentially) tainted data.
2. Evaluate the model on **two separate splits**

   * a *validation* set that may also be poisoned, and
   * a *clean* hold‑out *test* set.
3. Log every artifact, parameter, and metric to an MLflow tracking server for later analysis.

This side‑by‑side comparison clearly illustrates how poisoning levels of `0 %`, `5 %`, `10 %`, and `50 %` degrade model accuracy.

---

## Features

* **Automatic data fetching** – Downloads the Iris dataset if it does not already exist locally.
* **Configurable data poisoning** – Inject random‑uniform noise into a user‑defined fraction of rows.
* **Full training pipeline** – Splits data, trains a model, evaluates on poisoned & clean splits.
* **Comprehensive MLflow logging** – Parameters, metrics, data snapshots, predictions, and the fitted model are all versioned automatically.
* **Reusable artifacts** – CSV copies of every dataset split and model predictions are stored for deep‑dive analysis.

---

## Requirements

| Package        | Version (≥) |
| -------------- | ----------- |
| `pandas`       | 1.5.0       |
| `numpy`        | 1.23.0      |
| `scikit‑learn` | 1.2.0       |
| `mlflow`       | 2.0.0       |

> **Tip:** A ready‑made `requirements.txt` is provided for convenience.

---


## Usage

1. **Start / configure an MLflow Tracking Server**

   The script assumes a remote server is reachable at:

   ```
   http://0.0.0.0:8100
   ```

   Change the URI inside `main()` if you have a different endpoint:

   ```python
   mlflow.set_tracking_uri("http://<your‑server>:<port>")
   ```

2. **Run the experiment**

   ```bash
   python train.py
   ```

   The script will:

   * Ensure `./data/iris.csv` exists (downloads if missing).
   * Split the data into **train / validation / test**.
   * Iterate over the poisoning levels **0 %, 5 %, 10 %, 50 %**.
   * For each level:

     * Corrupt the appropriate rows.
     * Fit a logistic‑regression model.
     * Log everything to MLflow.

3. **Inspect results**

   Launch the MLflow UI:

   ```bash
   mlflow server --host 0.0.0.0 --port 8100
   ```

   Compare runs by sorting on `validation_accuracy` or `test_accuracy` and drill into the stored artifacts.

---


## Dataset

The *Iris* dataset contains 150 flower samples labelled as **setosa**, **versicolor**, or **virginica** with four numeric features:

| Feature      | Units |
| ------------ | ----- |
| sepal length | cm    |
| sepal width  | cm    |
| petal length | cm    |
| petal width  | cm    |

The script downloads it directly from the UCI Machine‑Learning Repository if it is not present locally.

---

## Data‑Poisoning Strategy

The function `poison_dataframe(df, poisoning_level)`:

1. Selects a *random* subset of rows equal to `poisoning_level * len(df)`.
2. For each selected row and each numerical feature, replaces the value with a sample drawn **uniformly** from the column’s min–max range.

This mimics a simple but effective **feature noise attack** that removes the intrinsic separability of the classes.

---

## Experiment Tracking with MLflow

Every run is stored under the experiment name

```
"Iris Poisoning Data – Full Pipeline with Predictions"
```

### Logged items

| Category       | Contents                                                  |
| -------------- | --------------------------------------------------------- |
| **Parameters** | `poisoning_level` (0.0 → 0.5)                             |
| **Metrics**    | `validation_accuracy`, `test_accuracy`                    |


## Mitigation Strategy
- It is always good to have data versions in order to restore the dataset before corruption
- It is always good to admin control on the data access internally and also while end-user using the apis.
---


