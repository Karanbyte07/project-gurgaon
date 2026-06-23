import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


# 1 -> Load the data

housing_df = pd.read_csv("housing.csv")


# 2-> create stratified shuffle split
housing_df['income_category'] = pd.cut(housing_df['median_income'], 
                                       bins=[0,1.5,3.0,4.5,6.0, np.inf],
                                       labels=[1,2,3,4,5])

split_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split_strategy.split(housing_df, housing_df['income_category']):
    train_set = housing_df.loc[train_index].drop('income_category', axis = 1)
    test_set = housing_df.loc[test_index].drop('income_category', axis = 1)


# we will work on the copy of train set
train_housing_df = train_set.copy()

# 3 -> Separating features and labels
housing_labels = train_housing_df["median_house_value"].copy()
housing_features = train_housing_df.drop("median_house_value", axis = 1)


# 4 -> Separate numerical and categorical columns
numerical_attributes = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
categorical_attributes = ["ocean_proximity"]

# 5 -> Pipeline 
# for numerical colm

numerical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    [
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# construct the full pipeline
preprocessing_pipeline = ColumnTransformer(
    [
        ("num", numerical_pipeline, numerical_attributes),
        ("cat", categorical_pipeline, categorical_attributes),
    ]
)

# 6 -> transform the data
prepared_housing = preprocessing_pipeline.fit_transform(housing_features)
# print(prepared_housing.shape)

# 7 -> print the data after transformation
# final_housing = pd.DataFrame(
#     prepared_housing,
#     columns=preprocessing_pipeline.get_feature_names_out(),
#     index=train_housing_df.index,
# )
# print(final_housing)

# 8 -> Train the model (Selecting the best model using cross validation)

#linear regression
lin_reg = LinearRegression()
lin_reg.fit(prepared_housing, housing_labels)
lin_pred = lin_reg.predict(prepared_housing)
#lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
#cross validation 
lin_rmses = -cross_val_score(lin_reg, prepared_housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10) 
print(f"Linear Regression root mean squared error is : {lin_rmses.mean()}")


#Decision Tree Regressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(prepared_housing, housing_labels)
dt_pred = dt_reg.predict(prepared_housing)
#dt_rmse = root_mean_squared_error(housing_labels, dt_pred)
dt_rmses = -cross_val_score(dt_reg, prepared_housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(f"Decision Tree Regression root mean squared error is : {dt_rmses.mean()}")


#Random Forest Regressor
rf_reg = RandomForestRegressor()
rf_reg.fit(prepared_housing, housing_labels)
rf_pred = rf_reg.predict(prepared_housing)
#rf_rmse = root_mean_squared_error(housing_labels, rf_pred)
rf_rmses = -cross_val_score(rf_reg, prepared_housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(f"Random Forest Regression root mean squared error is : {rf_rmses.mean()}")


#so the best model is Random Forest Regressor with the lowest root mean squared error. We will use this model to make predictions on the test set.

