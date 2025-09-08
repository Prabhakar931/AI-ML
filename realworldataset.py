import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
# from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

california_housing = fetch_california_housing()
# print("California housing: ", california_housing)
california_housing_df = pd.DataFrame(california_housing['data'], columns=california_housing['feature_names'])
california_housing_df["target"] = california_housing['target']
# print("Housing dataframe: ", california_housing_df)
# print("Target: ", california_housing_df['target'].head())
# california_housing_df = california_housing_df.drop('MedHouseVal', axis=1)
# print("California DF X: ", california_housing_df)
x = california_housing_df.drop('target', axis=1)
y = california_housing_df['target']
# print("X: ", x)
# print("y", y)
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(x_train, y_train)
model_score = model.score(x_test, y_test)
print("Model Scr: ", model_score)
