# 🏠 California Housing Price Prediction

A machine learning project that trains a **Random Forest Regressor** to predict California housing prices based on the classic California Housing dataset. The script handles the full ML lifecycle — data loading, stratified splitting, feature engineering, pipeline construction, model training, serialization, and batch inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Output](#output)

---

## Overview

This project uses **scikit-learn** to build a robust preprocessing pipeline and train a Random Forest model on housing data from California. It is designed to be run repeatedly — on the first run it trains and saves the model, and on every subsequent run it loads the saved model and runs inference on the input data.

---

## 📁 Project Structure

```
project_gurgaon/
│
├── housing.csv              # Raw California housing dataset (source data)
├── housing_main.py          # Main script: training + inference
├── main_old.py              # Old experimental script (model comparison)
│
├── housing_model.pkl        # Saved trained model (generated, not tracked by Git)
├── housing_pipeline.pkl     # Saved preprocessing pipeline (generated, not tracked by Git)
├── input_data.csv           # Test split used for inference (generated, not tracked by Git)
└── output_data.csv          # Prediction results (generated, not tracked by Git)
```

> **Note:** `.pkl`, `input_data.csv`, and `output_data.csv` are generated at runtime and excluded from version control via `.gitignore`.

---

## 📊 Dataset

The project uses the **California Housing dataset** (`housing.csv`), which contains data from the 1990 California census. Each row represents a census block group.

| Feature | Description |
|---|---|
| `longitude` | Longitude of the block |
| `latitude` | Latitude of the block |
| `housing_median_age` | Median age of houses in block |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Block population |
| `households` | Number of households |
| `median_income` | Median household income (in tens of thousands) |
| `ocean_proximity` | Proximity to the ocean (categorical) |
| `median_house_value` | **Target variable** — Median house value in USD |

---

## ⚙️ How It Works

The script operates in two modes automatically:

### Mode 1 — Training (first run)
If `housing_model.pkl` does **not** exist:

1. **Load** `housing.csv`
2. **Stratified Split** — Splits data into 80% train / 20% test using `StratifiedShuffleSplit` based on income category to preserve the income distribution
3. **Feature Engineering** — Separates numerical and categorical attributes
4. **Preprocessing Pipeline** — Applies:
   - `SimpleImputer(strategy="median")` → fills missing values
   - `StandardScaler()` → normalizes numerical features
   - `OneHotEncoder()` → encodes `ocean_proximity`
5. **Train** a `RandomForestRegressor` on the prepared training data
6. **Save** the model (`housing_model.pkl`) and pipeline (`housing_pipeline.pkl`) to disk

### Mode 2 — Inference (subsequent runs)
If `housing_model.pkl` **exists**:

1. **Load** the saved model and pipeline
2. **Read** `input_data.csv` (the 20% test split saved during training)
3. **Transform** the input using the saved pipeline
4. **Predict** `median_house_value` for each row
5. **Save** predictions to `output_data.csv`

```
First Run:
housing.csv ──► Stratified Split ──► Pipeline ──► RandomForestRegressor ──► housing_model.pkl
                                                                          └──► housing_pipeline.pkl

Subsequent Runs:
input_data.csv ──► housing_pipeline.pkl ──► housing_model.pkl ──► output_data.csv
```

---

## 📦 Dependencies

Install all required packages using pip:

```bash
pip install numpy pandas scikit-learn joblib
```

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `scikit-learn` | ≥ 1.3 | ML pipelines, models, and metrics |
| `joblib` | ≥ 1.3 | Model serialization and deserialization |

---

## 🚀 Usage

**1. Clone the repository:**

```bash
git clone https://github.com/Karanbyte07/california-housing-predictor.git
cd california-housing-predictor
```

**2. Install dependencies:**

```bash
pip install numpy pandas scikit-learn joblib
```

**3. Run the script:**

```bash
python housing_main.py
```

- **First run** → trains and saves the model.
- **Subsequent runs** → loads the model and runs inference.

---

## 📤 Output

After inference, predictions are written to `output_data.csv`, which contains all original columns from `input_data.csv` plus a new column:

| Column | Description |
|---|---|
| *(all original features)* | Original input features |
| `median_house_value` | Predicted housing price in USD |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | `RandomForestRegressor` |
| `random_state` | `42` |
| Train/Test Split | 80% / 20% |
| Split Strategy | `StratifiedShuffleSplit` on income category |
| Imputation | Median strategy for missing values |
| Scaling | `StandardScaler` for numerical features |
| Encoding | `OneHotEncoder` for `ocean_proximity` |
