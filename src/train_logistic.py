import joblib
import numpy as np

from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split

X, y = load_and_preprocess()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, learning_rate=0.01, num_epochs=1000):

    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(num_epochs):

        z = np.dot(X, w) + b

        y_pred = sigmoid(z)

        dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
        db = (1 / len(X)) * np.sum(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

w, b = train(X_train, y_train)

joblib.dump((w, b), "models/logistic_model.pkl")

print("Đã lưu Logistic Model")