import streamlit as st

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Thyroid Cancer Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CSS
# =====================================

st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#1565C0;
}

.subtitle{
    font-size:22px;
    color:#424242;
}

.box{
    background-color:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966480.png",
        width=120
    )

    st.title("Navigation")

    st.success("Gunakan menu di bawah ini")

    st.page_link("app.py", label="🏠 Home")

    st.divider()

    st.info("""
📂 Dataset

📊 Visualization

🤖 Prediction

📈 Model Evaluation

ℹ About
""")

# =====================================
# HALAMAN UTAMA
# =====================================

st.markdown(
    "<p class='title'>🩺 Thyroid Cancer Recurrence Prediction</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning menggunakan Random Forest</p>",
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns([2,1])

with col1:

    st.markdown("""
<div class='box'>

### Selamat Datang 👋

Aplikasi ini digunakan untuk memprediksi kemungkinan
terjadinya **kekambuhan kanker tiroid (Thyroid Cancer Recurrence)**.

Tahapan sistem meliputi:

- 📂 Menampilkan Dataset
- 📊 Visualisasi Data
- 🤖 Prediksi Pasien
- 📈 Evaluasi Model Machine Learning

Model yang digunakan adalah **Random Forest Classifier**.

</div>
""", unsafe_allow_html=True)

with col2:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
        use_container_width=True
    )

st.write("")

# =====================================
# METRIC
# =====================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Dataset",
        "383"
    )

with c2:
    st.metric(
        "Features",
        "17"
    )

with c3:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with c4:
    st.metric(
        "Target",
        "Recurred"
    )

st.write("")

# =====================================
# FITUR
# =====================================

st.subheader("✨ Fitur Sistem")

f1,f2,f3 = st.columns(3)

with f1:
    st.info("""
### 📂 Dataset

Melihat isi dataset
dan statistik dasar.
""")

with f2:
    st.success("""
### 🤖 Prediction

Melakukan prediksi
kemungkinan kekambuhan.
""")

with f3:
    st.warning("""
### 📈 Evaluation

Melihat akurasi model,
confusion matrix,
dan metrik evaluasi.
""")

st.write("")

# =====================================
# FOOTER
# =====================================

st.markdown("""
<hr>
<p class='footer'>
© 2026 | Ervi | Thyroid Prediction System
</p>
""", unsafe_allow_html=True)
