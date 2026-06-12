import joblib
import pandas as pd

from preprocessing import load_and_preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 5. CHUẨN BỊ DỮ LIỆU CHO RANDOM FOREST

X, y = load_and_preprocess()

# Chia dữ liệu với tỷ lệ 70-30
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("Kích thước của tập huấn luyện:")
print(X_train.shape)

print("Kích thước của tập kiểm tra:")
print(X_test.shape)

# 6. HUẤN LUYỆN RANDOM FOREST

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

# Ma trận nhầm lẫn
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Kết quả dự đoán đầu tiên
results_df = pd.DataFrame({
    'Thực tế': y_test,
    'Dự đoán': y_pred
})

print("\n10 kết quả đầu tiên:")
print(results_df.head(10))

# Báo cáo phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=1
    )
)

# Lưu model
joblib.dump(
    model,
    "models/randomforest_model.pkl"
)

print("\nĐã lưu Random Forest Model")