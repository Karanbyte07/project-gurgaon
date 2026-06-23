import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

MODEL_FILE = "housing_model.pkl"
PIPELINE_FILE = "housing_pipeline.pkl"


def build_pipeline(numerical_attributes, categorical_attributes):
    # Pipeline for numerical attributes
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for categorical attributes
    categorical_pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Construct the full preprocessing pipeline
    preprocessing_pipeline = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_attributes),
            ("cat", categorical_pipeline, categorical_attributes),
        ]
    )
    return preprocessing_pipeline


if not os.path.exists(MODEL_FILE):
    # Let's train the model and save it
    housing_df = pd.read_csv("housing.csv")

    # Create stratified shuffle split based on income category
    housing_df["income_category"] = pd.cut(
        housing_df["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split_strategy.split(
        housing_df, housing_df["income_category"]
    ):
        # Save the test partition as input_data.csv for future inference
        housing_df.loc[test_index].drop("income_category", axis=1).to_csv(
            "input_data.csv", index=False
        )
        housing = housing_df.loc[train_index].drop("income_category", axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    numerical_attributes = (
        housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    )
    categorical_attributes = ["ocean_proximity"]

    pipeline = build_pipeline(numerical_attributes, categorical_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save the trained model and processing pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model and pipeline saved successfully and trained on the data")

else:
    # Let's run inference using the saved model and pipeline
    print("Model and pipeline already exists, loading them")
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input_data.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output_data.csv", index=False)
    print("Predictions saved to output_data.csv")