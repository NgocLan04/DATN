# import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 23

#1. KHÁM PHÁ DỮ LIỆU 
# Đường dẫn đến tập dữ liệu
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Kiểm tra kích thước của dữ liệu
df.shape

# Hiển thị các dòng đầu tiên của dữ liệu để kiểm tra
df.head()

# kiểm tra số lượng dữ liệu nan
df.isna().sum()

# xóa tất cả dữ liệu nan
df = df.dropna()
df.shape

# xem thông tin bộ dữ liệu
df.info()

# xem mô tả
df.describe()

#2. TRỰC QUAN HÓA DỮ LIỆU
# Biểu đồ phân phối tuổi theo nhãn đột quỵ
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, color='blue')
plt.title('Phân phối tuổi')
plt.show()

# Biểu đồ phân phối mức đường huyết trung bình theo nhãn đột quỵ
plt.figure(figsize=(10, 6))
sns.histplot(df['avg_glucose_level'], bins=30, kde=True, color='green')
plt.title('Phân phối mức đường huyết trung bình')
plt.show()

# Biểu đồ thanh so sánh tỷ lệ đột quỵ theo giới tính
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='gender', hue='stroke', palette='viridis')
plt.title('Tỷ lệ đột quỵ theo giới tính')
plt.show()

# Biểu đồ thanh so sánh tỷ lệ đột quỵ theo tình trạng hôn nhân
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='ever_married', hue='stroke', palette='magma')
plt.title('Tỷ lệ đột quỵ theo tình trạng hôn nhân')
plt.show()

# Chọn các cột dạng số để vẽ biểu đồ nhiệt
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Vẽ biểu đồ nhiệt (heatmap)
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Feature Correlations')
plt.show()

#3. CHUẨN HÓA DỮ LIỆU
# Mã hóa biến phân loại
lb_make = LabelEncoder()
df["gender"] = lb_make.fit_transform(df["gender"])
df["ever_married"] = lb_make.fit_transform(df["ever_married"])
df["work_type"] = lb_make.fit_transform(df["work_type"])
df["Residence_type"] = lb_make.fit_transform(df["Residence_type"])
df["smoking_status"] = lb_make.fit_transform(df["smoking_status"])

# Chia dữ liệu thành features và target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Chuẩn hóa các đặc trưng số học
# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Kiểm tra dữ liệu sau khi mã hóa
df.head()

#4. XÂY DỰNG HÀM MODEL LOGISTIC REGRESSION
# Hàm sigmoid để chuyển đầu ra thành xác suất
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm huấn luyện mô hình hồi quy logistic
def train(X, y, learning_rate, num_epochs):
    # Khởi tạo trọng số w và bias b
    w = np.zeros(X.shape[1])
    b = 0
    
    # Huấn luyện mô hình
    for epoch in range(num_epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        # Tính đạo hàm cho trọng số w và bias b
        dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
        db = (1 / len(X)) * np.sum(y_pred - y)
        
        # Cập nhật trọng số w và bias b
        w -= learning_rate * dw
        b -= learning_rate * db

        # In ra quá trình huấn luyện
        if (epoch + 1) % 100 == 0:
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}')
        
    return w, b

#Dự đoán kết quả
def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred > 0.5).astype(int), y_pred

# Xây dựng mô hình và huấn luyện
learning_rate = 0.01
num_epochs = 1000

w, b = train(X, y, learning_rate, num_epochs)

# In trọng số và bias sau khi huấn luyện
print(f"Weights: {w}")
print(f"Bias: {b}")

# Dự đoán kết quả
predictions, probabilities = predict(X, w, b)
print("Predictions:")
print(predictions)

# Kiểm tra độ chính xác của mô hình
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")

# In ra ma trận nhầm lẫn (confusion matrix)
conf_matrix = confusion_matrix(y, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# In báo cáo phân loại chi tiết (classification report)
class_report = classification_report(y, predictions, zero_division=0)
print("Classification Report:")
print(class_report)

# In kết quả dự đoán so với giá trị thực tế
result_df = pd.DataFrame({'Actual': y, 'Predicted': predictions, 'Probability': probabilities})
result_df.head(10)

#5. CÂN BẰNG VÀ CHUẨN BỊ DỮ LIỆU CHO SVM
# Sử dụng Random Oversampling để cân bằng dữ liệu
# Tách dữ liệu thành hai nhóm: lớp chiếm đa số và lớp chiếm thiểu số
df_majority = df[df.stroke == 0]
df_minority = df[df.stroke == 1]

# Tăng số lượng mẫu của lớp chiếm thiểu số
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # lấy mẫu với thay thế
                                 n_samples=len(df_majority),  # số lượng mẫu sau khi tăng
                                 random_state=42)  # số ngẫu nhiên để tái tạo

# Kết hợp lại hai nhóm
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Kiểm tra phân phối của các lớp sau khi cân bằng
df_upsampled.stroke.value_counts()

# Chuẩn bị dữ liệu
X = df_upsampled.drop(['id', 'stroke'], axis=1)
y = df_upsampled['stroke']

# Chia dữ liệu với tỷ lệ 70-30 cho tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Kiểm tra kích thước của tập huấn luyện và tập kiểm tra
print("Kích thước của tập huấn luyện:", X_train.shape)
print("Kích thước của tập kiểm tra:", X_test.shape)

#6. HUẤN LUYÊN MODEL SVM
# Huấn luyện mô hình SVM với trọng số lớp
model = SVC(kernel='linear', C=1, class_weight='balanced')
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# In ra kết quả dự đoán so với giá trị thực tế
results_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred})
print(results_df.head(10))

# Tạo báo cáo phân loại chi tiết
print("Báo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred, zero_division=1))