import streamlit as st
import pandas as pd

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Dataset",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Dataset Thyroid Cancer")

# =====================================
# MEMBACA DATASET
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("Thyroid_Diff.csv")

df = load_data()

# =====================================
# INFORMASI DATASET
# =====================================

st.subheader("Informasi Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", df.shape[0])

with col2:
    st.metric("Jumlah Fitur", df.shape[1])

with col3:
    st.metric("Missing Value", int(df.isnull().sum().sum()))

# =====================================
# PREVIEW DATASET
# =====================================

st.subheader("Preview Dataset")

jumlah_data = st.slider(
    "Jumlah data yang ditampilkan",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(
    df.head(jumlah_data),
    use_container_width=True
)

# =====================================
# TIPE DATA
# =====================================

st.subheader("Tipe Data")

dtype = pd.DataFrame({
    "Kolom": df.columns,
    "Tipe Data": df.dtypes.astype(str)
})

st.dataframe(dtype, use_container_width=True)

# =====================================
# MISSING VALUE
# =====================================

st.subheader("Missing Value")

missing = pd.DataFrame({
    "Kolom": df.columns,
    "Jumlah Missing": df.isnull().sum(),
    "Persentase (%)": round(df.isnull().mean()*100,2)
})

st.dataframe(missing, use_container_width=True)

# =====================================
# STATISTIK DESKRIPTIF
# =====================================

st.subheader("Statistik Deskriptif")

st.dataframe(
    df.describe(include="all"),
    use_container_width=True
)

# =====================================
# DOWNLOAD DATASET
# =====================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="Thyroid_Diff.csv",
    mime="text/csv"
)

# =====================================
# INFORMASI KOLOM
# =====================================

st.subheader("Daftar Kolom Dataset")

st.write(list(df.columns))

# =====================================
# FOOTER
# =====================================

st.markdown("---")
st.caption("© 2026 | Ervi | Thyroid Prediction System")
