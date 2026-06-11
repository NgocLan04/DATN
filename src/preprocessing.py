import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
from sklearn.utils import resample

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 16

if __name__ == "__main__":
    #1. KHÁM PHÁ DỮ LIỆU

    # Đường dẫn đến tập dữ liệu
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

    print("Kích thước dữ liệu:")
    print(df.shape)

    print("\n5 dòng đầu tiên:")
    print(df.head())

    print("\nSố lượng dữ liệu thiếu:")
    print(df.isna().sum())

    # xóa tất cả dữ liệu nan
    df = df.dropna()

    print("\nKích thước sau khi xử lý dữ liệu thiếu:")
    print(df.shape)

    print("\nThông tin bộ dữ liệu:")
    df.info()

    print("\nMô tả dữ liệu:")
    print(df.describe())

    #2. TRỰC QUAN HÓA DỮ LIỆU

    # Biểu đồ phân phối tuổi
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=30, kde=True, color='blue')
    plt.title('Phân phối tuổi')
    plt.show()

    # Biểu đồ phân phối mức đường huyết trung bình
    plt.figure(figsize=(10, 6))
    sns.histplot(df['avg_glucose_level'], bins=30, kde=True, color='green')
    plt.title('Phân phối mức đường huyết trung bình')
    plt.show()

    # Biểu đồ tỷ lệ đột quỵ theo giới tính
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        x='gender',
        hue='stroke',
        palette='viridis'
    )
    plt.title('Tỷ lệ đột quỵ theo giới tính')
    plt.show()

    # Biểu đồ tỷ lệ đột quỵ theo tình trạng hôn nhân
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        x='ever_married',
        hue='stroke',
        palette='magma'
    )
    plt.title('Tỷ lệ đột quỵ theo tình trạng hôn nhân')
    plt.show()

    # Heatmap tương quan

    df_numeric = df.select_dtypes(
        include=['float64', 'int64']
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_numeric.corr(),
        annot=True,
        cmap='coolwarm',
        fmt=".2f"
    )
    plt.title('Heatmap of Feature Correlations')
    plt.show()

    #3. CHUẨN HÓA DỮ LIỆU

    # Mã hóa biến phân loại
    encoder = LabelEncoder()

    df["gender"] = encoder.fit_transform(df["gender"])
    df["ever_married"] = encoder.fit_transform(df["ever_married"])
    df["work_type"] = encoder.fit_transform(df["work_type"])
    df["Residence_type"] = encoder.fit_transform(df["Residence_type"])
    df["smoking_status"] = encoder.fit_transform(df["smoking_status"])

    # Chia dữ liệu thành features và target
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    print("\nDữ liệu sau khi mã hóa:")
    print(df.head())

    print("\nKích thước X:")
    print(X.shape)

    print("\nKích thước y:")
    print(y.shape)

def load_and_preprocess():

    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

    # Xóa dữ liệu thiếu
    df = df.dropna()

    # Mã hóa dữ liệu phân loại
    encoder = LabelEncoder()

    df["gender"] = encoder.fit_transform(df["gender"])
    df["ever_married"] = encoder.fit_transform(df["ever_married"])
    df["work_type"] = encoder.fit_transform(df["work_type"])
    df["Residence_type"] = encoder.fit_transform(df["Residence_type"])
    df["smoking_status"] = encoder.fit_transform(df["smoking_status"])

    df_majority = df[df.stroke == 0]
    df_minority = df[df.stroke == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat(
        [df_majority, df_minority_upsampled]
    )

    X = df_balanced.drop('stroke', axis=1)
    y = df_balanced['stroke']

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y