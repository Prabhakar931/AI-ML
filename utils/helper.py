from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def modelScore(models, X_train, X_test, y_train, y_test):
    model_scores = {}
    np.random.seed(42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    
    return model_scores
