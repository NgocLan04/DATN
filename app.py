import streamlit as st
from src.predict import predict_patient

# ==========================
# CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Dự đoán đột quỵ",
    page_icon="🩺",
    layout="wide"
)

# ==========================
# CSS
# ==========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    text-align: center;
}

.stButton > button {
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"]{
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# TIÊU ĐỀ
# ==========================
st.markdown("""
<h1 style='color:#1565C0'>
🩺 HỆ THỐNG DỰ ĐOÁN NGUY CƠ ĐỘT QUỴ
</h1>
""", unsafe_allow_html=True)

st.write("")

# ==========================
# THÔNG TIN CÁ NHÂN
# ==========================
with st.container(border=True):

    st.subheader("👤 Thông tin cá nhân")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Giới tính",
            ["Nam", "Nữ"]
        )

    with col2:
        age = st.number_input(
            "Tuổi",
            min_value=0,
            max_value=120,
            value=30
        )

    with col3:
        ever_married = st.selectbox(
            "Tình trạng hôn nhân",
            ["Chưa kết hôn", "Đã kết hôn"]
        )

# ==========================
# TIỀN SỬ BỆNH
# ==========================
with st.container(border=True):

    st.subheader("❤️ Tiền sử bệnh")

    col1, col2 = st.columns(2)

    with col1:
        hypertension = st.selectbox(
            "Tăng huyết áp",
            ["Không", "Có"]
        )

    with col2:
        heart_disease = st.selectbox(
            "Bệnh tim",
            ["Không", "Có"]
        )

# ==========================
# THÔNG TIN SINH HOẠT
# ==========================
with st.container(border=True):

    st.subheader("🏢 Thông tin sinh hoạt")

    col1, col2, col3 = st.columns(3)

    with col1:
        work_type = st.selectbox(
            "Nghề nghiệp",
            [
                "Tư nhân",
                "Tự kinh doanh",
                "Công chức",
                "Trẻ em",
                "Chưa từng làm việc"
            ]
        )

    with col2:
        residence = st.selectbox(
            "Nơi ở",
            ["Thành thị", "Nông thôn"]
        )

    with col3:
        smoking = st.selectbox(
            "Tình trạng hút thuốc",
            [
                "Không hút thuốc",
                "Đã từng hút",
                "Đang hút thuốc",
                "Không rõ"
            ]
        )

# ==========================
# CHỈ SỐ SỨC KHỎE
# ==========================
with st.container(border=True):

    st.subheader("📊 Chỉ số sức khỏe")

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input(
            "Đường huyết trung bình",
            min_value=0.0,
            value=100.0
        )

    with col2:
        bmi = st.number_input(
            "Chỉ số BMI",
            min_value=0.0,
            value=25.0
        )

st.write("")

# ==========================
# NÚT CHỨC NĂNG
# ==========================
btn1, btn2 = st.columns([4, 1])

with btn1:
    predict_btn = st.button(
        "🔍 DỰ ĐOÁN",
        use_container_width=True
    )

with btn2:
    refresh_btn = st.button(
        "🔄 LÀM MỚI",
        use_container_width=True
    )

# ==========================
# LÀM MỚI
# ==========================
if refresh_btn:
    st.rerun()

# ==========================
# DỰ ĐOÁN
# ==========================
if predict_btn:

    gender = 1 if gender == "Nam" else 0
    hypertension = 1 if hypertension == "Có" else 0
    heart_disease = 1 if heart_disease == "Có" else 0
    ever_married = 1 if ever_married == "Đã kết hôn" else 0
    residence = 1 if residence == "Thành thị" else 0

    work_mapping = {
        "Công chức": 0,
        "Chưa từng làm việc": 1,
        "Tư nhân": 2,
        "Tự kinh doanh": 3,
        "Trẻ em": 4
    }

    smoking_mapping = {
        "Không rõ": 0,
        "Đã từng hút": 1,
        "Không hút thuốc": 2,
        "Đang hút thuốc": 3
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

    st.write("")
    st.subheader("📋 Kết quả dự đoán")

    if result == 1:
        st.markdown("""
        <div style="
            padding:20px;
            border-radius:10px;
            background:#ffebee;
            border-left:8px solid red;
        ">
            <h3>⚠️ Nguy cơ đột quỵ CAO</h3>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            padding:20px;
            border-radius:10px;
            background:#e8f5e9;
            border-left:8px solid green;
        ">
            <h3>✅ Nguy cơ đột quỵ THẤP</h3>
        </div>
        """, unsafe_allow_html=True)
