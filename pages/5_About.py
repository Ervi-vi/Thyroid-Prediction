import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ",
    layout="wide"
)

st.title("Tentang Aplikasi")

st.markdown("---")

st.header("🩺 Thyroid Cancer Recurrence Prediction System")

st.write("""
Aplikasi ini dikembangkan untuk membantu memprediksi kemungkinan
kekambuhan kanker tiroid menggunakan algoritma Machine Learning
(Random Forest Classifier).

Aplikasi dibuat menggunakan framework Streamlit sehingga dapat
diakses melalui browser.
""")

st.subheader("Dataset")

st.info("""
Dataset : Thyroid_Diff.csv

Jumlah Data : 383

Jumlah Fitur : 17

Target : Recurred
""")

st.subheader("Teknologi")

st.success("""
• Python

• Streamlit

• Pandas

• Scikit-Learn

• Plotly

• Joblib
""")

st.subheader("Machine Learning")

st.write("""
Algoritma :

✔ Random Forest Classifier
""")

st.subheader("Developer")

st.write("""
Nama : Ervi

Program Studi : Informatika

Mata Kuliah : Data Science For Health

Universitas Respati Yogyakarta

2026
""")

st.markdown("---")

st.caption("© 2026 | Ervi | Thyroid Prediction System")
