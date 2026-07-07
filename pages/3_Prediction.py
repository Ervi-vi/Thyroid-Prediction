import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Prediksi Kanker Tiroid",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

.block-container{
    padding-top:2rem;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#1565C0;
}

.subtitle{
    font-size:18px;
    color:#555555;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.10);
    margin-bottom:15px;
}

.section{
    font-size:24px;
    font-weight:bold;
    color:#1565C0;
    margin-top:15px;
    margin-bottom:10px;
}

hr{
    margin-top:10px;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = joblib.load("models/model.pkl")
    encoder = joblib.load("models/encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Opsional, jika sudah dibuat saat training
    try:
        feature_names = joblib.load("models/features.pkl")
    except:
        feature_names = None

    try:
        target_encoder = joblib.load("models/target_encoder.pkl")
    except:
        target_encoder = None

except Exception as e:

    st.error(f"Gagal memuat model : {e}")
    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🩺 Informasi Model")

    st.success("Model berhasil dimuat")

    st.markdown("---")

    st.write("**Algoritma**")
    st.info("Random Forest")

    st.write("**Target Prediksi**")
    st.info("Recurred")

    st.write("**Dataset**")
    st.info("Thyroid_Diff.csv")

    st.markdown("---")

    st.caption("Universitas Respati Yogyakarta")

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<p class="title">🩺 Sistem Prediksi Kekambuhan Kanker Tiroid</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Masukkan data pasien untuk memperoleh hasil prediksi menggunakan Machine Learning.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================================
# FORM INPUT
# ==========================================================

st.markdown(
    '<p class="section">📋 Data Pasien</p>',
    unsafe_allow_html=True
)

kiri, kanan = st.columns(2)

# ==========================================================
# KOLOM KIRI
# ==========================================================

with kiri:

    age = st.number_input(
        "Usia",
        10,
        100,
        40
    )

    gender = st.selectbox(
        "Jenis Kelamin",
        ["F","M"]
    )

    smoking = st.selectbox(
        "Merokok",
        ["No","Yes"]
    )

    hx_smoking = st.selectbox(
        "Riwayat Merokok",
        ["No","Yes"]
    )

    hx_radiotherapy = st.selectbox(
        "Riwayat Radioterapi",
        ["No","Yes"]
    )

    thyroid_function = st.selectbox(
        "Fungsi Tiroid",
        [
            "Normal",
            "Clinical Hyperthyroidism",
            "Subclinical Hyperthyroidism",
            "Clinical Hypothyroidism",
            "Subclinical Hypothyroidism"
        ]
    )

    physical_exam = st.selectbox(
        "Pemeriksaan Fisik",
        [
            "Single nodular goiter-left",
            "Single nodular goiter-right",
            "Multinodular goiter",
            "Diffuse goiter",
            "Normal"
        ]
    )

# ==========================================================
# KOLOM KANAN
# ==========================================================

with kanan:

    adenopathy = st.selectbox(
        "Adenopathy",
        [
            "No",
            "Left",
            "Right",
            "Bilateral",
            "Posterior"
        ]
    )

    pathology = st.selectbox(
        "Patologi",
        [
            "Micropapillary",
            "Papillary",
            "Follicular",
            "Hurthle cell"
        ]
    )

    focality = st.selectbox(
        "Focality",
        [
            "Uni-Focal",
            "Multi-Focal"
        ]
    )

    risk = st.selectbox(
        "Kategori Risiko",
        [
            "Low",
            "Intermediate",
            "High"
        ]
    )

    T = st.selectbox(
        "Tumor (T)",
        [
            "T1a",
            "T1b",
            "T2",
            "T3a",
            "T3b",
            "T4a",
            "T4b"
        ]
    )

    N = st.selectbox(
        "Node (N)",
        [
            "N0",
            "N1a",
            "N1b"
        ]
    )

    M = st.selectbox(
        "Metastasis (M)",
        [
            "M0",
            "M1"
        ]
    )

    stage = st.selectbox(
        "Stadium",
        [
            "I",
            "II",
            "III",
            "IVA",
            "IVB"
        ]
    )

    response = st.selectbox(
        "Respon Terapi",
        [
            "Excellent",
            "Indeterminate",
            "Biochemical Incomplete",
            "Structural Incomplete"
        ]
    )

st.markdown("---")

prediksi = st.button(
    "🔍 Prediksi Risiko Kekambuhan",
    use_container_width=True,
    type="primary"
)

# ==========================================================
# PROSES PREDIKSI
# ==========================================================

if prediksi:

    input_df = pd.DataFrame({

        "Age":[age],
        "Gender":[gender],
        "Smoking":[smoking],
        "Hx Smoking":[hx_smoking],
        "Hx Radiothreapy":[hx_radiotherapy],   # Sesuaikan dengan nama kolom dataset
        "Thyroid Function":[thyroid_function],
        "Physical Examination":[physical_exam],
        "Adenopathy":[adenopathy],
        "Pathology":[pathology],
        "Focality":[focality],
        "Risk":[risk],
        "T":[T],
        "N":[N],
        "M":[M],
        "Stage":[stage],
        "Response":[response]

    })

    # ===============================
    # Encoding
    # ===============================

    for col in input_df.columns:

        if col in encoder:

            try:

                input_df[col] = encoder[col].transform(
                    input_df[col].astype(str)
                )

            except Exception:

                st.error(
                    f"Nilai pada kolom '{col}' tidak sesuai dengan data training."
                )

                st.stop()

    # ===============================
    # Menyamakan urutan fitur
    # ===============================

    if feature_names is not None:

        input_df = input_df[feature_names]

    # ===============================
    # Normalisasi
    # ===============================

    input_scaled = scaler.transform(input_df)

    # ===============================
    # Prediksi
    # ===============================

    prediksi_model = model.predict(input_scaled)

    probabilitas = model.predict_proba(input_scaled)[0]

    st.session_state["hasil"] = hasil
    st.session_state["probabilitas"] = probabilitas
    st.session_state["input_pasien"] = input_df.copy()

    st.success("✅ Prediksi berhasil dilakukan.")

# ==========================================================
# TAMPILKAN HASIL
# ==========================================================

if "hasil" in st.session_state:

    hasil = st.session_state["hasil"]

    probabilitas = st.session_state["probabilitas"]

    st.markdown("---")

    st.header("📊 Hasil Prediksi")

if hasil == "No" or hasil == 0:

    st.success("## 🟢 Risiko Rendah")

    st.write(
        "Model memprediksi pasien **tidak mengalami kekambuhan kanker tiroid**."
    )

else:

    st.error("## 🔴 Risiko Tinggi")

    st.write(
        "Model memprediksi pasien **berpotensi mengalami kekambuhan kanker tiroid**."
    )

confidence = max(probabilitas)

st.metric(

    "Tingkat Keyakinan Model",

    f"{confidence*100:.2f}%"

)

import plotly.express as px

prob_df = pd.DataFrame({

    "Kategori":[

        "Tidak Kambuh",

        "Kambuh"

    ],

    "Probabilitas":[

        probabilitas[0]*100,

        probabilitas[1]*100

    ]

})

fig = px.bar(

    prob_df,

    x="Kategori",

    y="Probabilitas",

    color="Kategori",

    text="Probabilitas"

)

fig.update_traces(

    texttemplate="%{text:.2f}%",

    textposition="outside"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.subheader("📋 Ringkasan Data Pasien")

ringkasan = pd.DataFrame({

    "Parameter":[

        "Usia",

        "Jenis Kelamin",

        "Merokok",

        "Riwayat Merokok",

        "Riwayat Radioterapi",

        "Fungsi Tiroid",

        "Pemeriksaan Fisik",

        "Adenopathy",

        "Patologi",

        "Focality",

        "Kategori Risiko",

        "Tumor",

        "Node",

        "Metastasis",

        "Stadium",

        "Response"

    ],

    "Nilai":[

        age,

        "Perempuan" if gender=="F" else "Laki-laki",

        "Ya" if smoking=="Yes" else "Tidak",

        "Ya" if hx_smoking=="Yes" else "Tidak",

        "Ya" if hx_radiotherapy=="Yes" else "Tidak",

        thyroid_function,

        physical_exam,

        adenopathy,

        pathology,

        focality,

        risk,

        T,

        N,

        M,

        stage,

        response

    ]

})

st.dataframe(

    ringkasan,

    use_container_width=True,

    hide_index=True

)
