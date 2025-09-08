import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

heart_disease = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
heart_disease.head()
# print(heart_disease)
x = heart_disease.drop('target', axis = 1)
y = heart_disease['target']
clf = RandomForestClassifier(n_estimators=100)
clf.get_params()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf.fit(x_train, y_train)
y_label = clf.predict(x_test)
# print(y_label)
score_train = clf.score(x_train, y_train)
print("Score trained data: ", score_train)
score_test = clf.score(x_test, y_test)
print("Score test: ", score_test)
print(classification_report(y_test, y_label))
# new_patient = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
# prediction = clf.predict(new_patient)
# print("Prediction:", prediction)  # 1 = Heart disease, 0 = No heart disease
