import streamlit as st
from src.predict import predict_patient

st.set_page_config(
    page_title="Dự đoán đột quỵ",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Hệ thống dự đoán nguy cơ đột quỵ")
st.write("Nhập thông tin bệnh nhân để dự đoán nguy cơ mắc bệnh đột quỵ.")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Giới tính",
        ["Nam", "Nữ"]
    )

    age = st.number_input(
        "Tuổi",
        min_value=0,
        max_value=120,
        value=30
    )

    hypertension = st.selectbox(
        "Tăng huyết áp",
        ["Không", "Có"]
    )

    heart_disease = st.selectbox(
        "Bệnh tim",
        ["Không", "Có"]
    )

    ever_married = st.selectbox(
        "Đã kết hôn",
        ["Chưa", "Rồi"]
    )

with col2:

    work_type = st.selectbox(
        "Loại công việc",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

    residence = st.selectbox(
        "Nơi ở",
        ["Urban", "Rural"]
    )

    glucose = st.number_input(
        "Mức đường huyết trung bình",
        min_value=0.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    smoking = st.selectbox(
        "Tình trạng hút thuốc",
        [
            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown"
        ]
    )

st.divider()

if st.button("🔍 Dự đoán nguy cơ đột quỵ", use_container_width=True):

    gender = 1 if gender == "Nam" else 0
    hypertension = 1 if hypertension == "Có" else 0
    heart_disease = 1 if heart_disease == "Có" else 0
    ever_married = 1 if ever_married == "Rồi" else 0
    residence = 1 if residence == "Urban" else 0

    work_mapping = {
        "Govt_job": 0,
        "Never_worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "children": 4
    }

    smoking_mapping = {
        "Unknown": 0,
        "formerly smoked": 1,
        "never smoked": 2,
        "smokes": 3
    }

    data = [
        1,
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_mapping[work_type],
        residence,
        glucose,
        bmi,
        smoking_mapping[smoking]
    ]

    result = predict_patient(data)

    st.subheader("Kết quả dự đoán")

    if result == 1:
        st.error("⚠️ Bệnh nhân có nguy cơ đột quỵ")
    else:
        st.success("✅ Bệnh nhân không có nguy cơ đột quỵ")