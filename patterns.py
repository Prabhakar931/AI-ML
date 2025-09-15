import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


heart_disease_df = pd.read_csv("heart-disease.csv")
x = heart_disease_df.drop("target", axis=1)
y = heart_disease_df["target"]

# print(x)
# print(y)

pd.crosstab(heart_disease_df["sex"], heart_disease_df["target"]).plot(
    kind="bar", figsize=(6,4), color=["salmon", "lightblue"]
)
plt.title("Heart Disease Frequency by Sex")
plt.xlabel("0 = No heart disease, 1 = Heart disease")
plt.legend(["Female", "Male"])
plt.ylabel("Count")
plt.show()
