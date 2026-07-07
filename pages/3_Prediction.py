import streamlit as st
import pandas as pd
import joblib

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Prediksi Kekambuhan Kanker Tiroid",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sistem Prediksi Kekambuhan Kanker Tiroid")
st.caption("Silakan masukkan data pasien untuk melakukan prediksi.")

st.markdown("---")

# ==========================================
# LOAD MODEL
# ==========================================

try:
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model belum tersedia. Jalankan train_model.py terlebih dahulu.")
    st.stop()

# ==========================================
# FORM INPUT
# ==========================================

st.subheader("Input Data Pasien")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Usia",
        min_value=10,
        max_value=100,
        value=40
    )

    gender = st.selectbox(
        "Jenis Kelamin",
        ["F", "M"]
    )

    smoking = st.selectbox(
        "Apakah Anda Perokok?",
        ["No", "Yes"]
    )

    hx_smoking = st.selectbox(
        "Apakah Anda Pernah Merokok?",
        ["No", "Yes"]
    )

    hx_radiotherapy = st.selectbox(
        "Apakah Anda Pernah Radioterapi?",
        ["No", "Yes"]
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

with col2:

    adenopathy = st.selectbox(
        "Adenopati",
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
        "Risk",
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
        "Stage",
        [
            "I",
            "II",
            "III",
            "IVA",
            "IVB"
        ]
    )

    response = st.selectbox(
        "Response",
        [
            "Excellent",
            "Indeterminate",
            "Biochemical Incomplete",
            "Structural Incomplete"
        ]
    )

st.markdown("---")

# ==========================================
# TOMBOL PREDIKSI
# ==========================================

predict = st.button(
    "🔍 Predict",
    use_container_width=True
)

# ==========================================
# PREPROCESSING
# ==========================================

if predict:

    input_df = pd.DataFrame({
        "Age":[age],
        "Gender":[gender],
        "Smoking":[smoking],
        "Hx Smoking":[hx_smoking],
        "Hx Radiothreapy":[hx_radiotherapy],
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

    # Encoding sesuai encoder saat training
    for col in input_df.columns:

        if col in encoder:

            try:
                input_df[col] = encoder[col].transform(input_df[col])
            except:

                st.error(
                    f"Nilai '{input_df[col][0]}' pada kolom '{col}' tidak dikenali."
                )
                st.stop()

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Simpan agar dipakai pada Tahap 5B
    st.session_state["input_scaled"] = input_scaled
    st.session_state["input_df"] = input_df

    st.success("Data pasien berhasil diproses dan siap diprediksi.")
# ==========================================
# HASIL PREDIKSI
# ==========================================

if predict:

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")

    st.subheader("📋 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 0:

            st.success("🟢 Prediksi : No Recurrence")

        else:

            st.error("🔴 Prediksi : Recurrence")

    with col2:

        st.metric(
            "Confidence",
            f"{max(probability)*100:.2f}%"
        )

# ==========================================
# PROBABILITAS
# ==========================================

    st.subheader("📊 Probabilitas Prediksi")

    prob_df = pd.DataFrame({

        "Kategori":[
            "No Recurrence",
            "Recurrence"
        ],

        "Probabilitas (%)":[
            probability[0]*100,
            probability[1]*100
        ]

    })

    st.bar_chart(
        prob_df.set_index("Kategori")
    )

# ==========================================
# DATA PASIEN
# ==========================================

    st.subheader("🧾 Ringkasan Data Pasien")

    tampil = pd.DataFrame({

        "Variabel":[
            "Age",
            "Gender",
            "Smoking",
            "Hx Smoking",
            "Hx Radiotherapy",
            "Thyroid Function",
            "Physical Examination",
            "Adenopathy",
            "Pathology",
            "Focality",
            "Risk",
            "T",
            "N",
            "M",
            "Stage",
            "Response"
        ],

        "Nilai":[
            age,
            gender,
            smoking,
            hx_smoking,
            hx_radiotherapy,
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

    st.table(tampil)

# ==========================================
# REKOMENDASI
# ==========================================

    st.subheader("💡 Rekomendasi")

    if prediction == 0:

        st.success("""

Pasien diprediksi **tidak mengalami kekambuhan kanker tiroid**.

Tetap disarankan untuk:

- Melakukan kontrol rutin sesuai jadwal dokter.
- Menjaga pola hidup sehat.
- Mengonsumsi obat sesuai anjuran.
- Melakukan pemeriksaan laboratorium secara berkala.

""")

    else:

        st.warning("""

Pasien diprediksi **berpotensi mengalami kekambuhan kanker tiroid**.

Disarankan untuk:

- Segera berkonsultasi dengan dokter spesialis.
- Melakukan pemeriksaan lanjutan.
- Menjalani evaluasi terapi.
- Melakukan pemantauan secara berkala.

""")

# ==========================================
# FOOTER
# ==========================================

    st.markdown("---")

    st.caption(
        "Prediction generated using Random Forest Classifier"
    )
