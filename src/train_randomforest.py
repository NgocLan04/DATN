import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocessing import load_and_preprocess

X, y = load_and_preprocess()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/randomforest_model.pkl"
)

print("Đã lưu Random Forest Model")