import joblib
import numpy as np
import pandas as pd

from preprocessing import load_and_preprocess
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Đọc dữ liệu đã tiền xử lý
X, y = load_and_preprocess()

# Hàm sigmoid để chuyển đầu ra thành xác suất
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm huấn luyện Logistic Regression
def train(X, y, learning_rate, num_epochs):

    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(num_epochs):

        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
        db = (1 / len(X)) * np.sum(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        if (epoch + 1) % 100 == 0:

            loss = -np.mean(
                y * np.log(y_pred + 1e-10)
                + (1 - y) * np.log(1 - y_pred + 1e-10)
            )

            print(
                f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}"
            )

    return w, b


# Hàm dự đoán
def predict(X, w, b):

    z = np.dot(X, w) + b

    y_pred = sigmoid(z)

    return (y_pred > 0.5).astype(int), y_pred


# Huấn luyện mô hình
learning_rate = 0.01
num_epochs = 1000

w, b = train(
    X,
    y,
    learning_rate,
    num_epochs
)

# In trọng số
print("\nWeights:")
print(w)

print("\nBias:")
print(b)

# Dự đoán
predictions, probabilities = predict(
    X,
    w,
    b
)

print("\nPredictions:")
print(predictions)

# Accuracy
accuracy = np.mean(predictions == y)

print(f"\nAccuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(
    y,
    predictions
)

print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(
    y,
    predictions,
    zero_division=0
)

print("\nClassification Report:")
print(class_report)

# 10 dòng kết quả đầu
result_df = pd.DataFrame({
    'Actual': y,
    'Predicted': predictions,
    'Probability': probabilities
})

print("\n10 kết quả đầu tiên:")
print(result_df.head(10))

# Lưu model
joblib.dump(
    (w, b),
    "models/logistic_model.pkl"
)

print("\nĐã lưu Logistic Model")