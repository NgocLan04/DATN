import joblib

def predict_patient(data):

    model = joblib.load(
        "models/randomforest_model.pkl"
    )

    result = model.predict([data])

    return result[0]