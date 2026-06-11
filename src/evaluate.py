import joblib
import numpy as np

from preprocessing import load_and_preprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

X, y = load_and_preprocess()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Phân bố dữ liệu:")
print(y.value_counts())

# =========================
# LOGISTIC REGRESSION
# =========================

w, b = joblib.load(
    "models/logistic_model.pkl"
)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.dot(X_test, w) + b

y_pred_logistic = (
    sigmoid(z) > 0.5
).astype(int)

# =========================
# RANDOM FOREST
# =========================

rf_model = joblib.load(
    "models/randomforest_model.pkl"
)

y_pred_rf = rf_model.predict(X_test)

# =========================
# KẾT QUẢ LOGISTIC
# =========================

print("\n========== Logistic Regression ==========")

print("Accuracy :", accuracy_score(y_test, y_pred_logistic))
print("Precision:", precision_score(y_test, y_pred_logistic, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_logistic, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred_logistic, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))

# =========================
# KẾT QUẢ RANDOM FOREST
# =========================

print("\n========== Random Forest ==========")

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_rf, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred_rf, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))