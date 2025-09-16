import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.helper import modelScore


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
# plt.show())
# print(pd.crosstab(heart_disease_df["cp"], heart_disease_df["target"]))
corr_matrix = heart_disease_df.corr()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}


model_score = modelScore(models, X_train, X_test, y_train, y_test)

# print("Models Score: ", model_score)
model_compare = pd.DataFrame(model_score, index=["accuracy"])
plot = model_compare.T.plot.bar()
# plt.show()