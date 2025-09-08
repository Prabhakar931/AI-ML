import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.datasets import _california_housing
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("car-sales.csv")
print(df.head())

# Clean target (remove $ and , then convert to float)
df["Price"] = df["Price"].replace('[\$,]', '', regex=True).astype(float)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# One-hot encode categorical features
categorical_features = ["Make", "Colour", "Doors"]
transformer = ColumnTransformer(
    [("one_hot", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

# Transform features
transformed_X = transformer.fit_transform(X)
print("Transformed shape:", transformed_X.shape)

# Train-test split (correct order + random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    transformed_X, y, test_size=0.2, random_state=42
)

# Train Random Forest
# model = RandomForestRegressor(random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
print("Training R²:", model.score(X_train, y_train))
print("Testing R²:", model.score(X_test, y_test))
