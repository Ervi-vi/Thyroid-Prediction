import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Visualisasi Data",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualisasi Dataset Thyroid Cancer")

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("Thyroid_Diff.csv")

df = load_data()

st.markdown("---")

# ==========================================
# PILIH KOLOM
# ==========================================

st.sidebar.header("Filter Visualisasi")

kategori = df.select_dtypes(include="object").columns.tolist()

pilihan = st.sidebar.selectbox(
    "Pilih Kolom Kategori",
    kategori
)

# ==========================================
# BAR CHART
# ==========================================

st.subheader(f"Distribusi {pilihan}")

count_df = (
    df[pilihan]
    .value_counts()
    .reset_index(name="Jumlah")
)

count_df.columns = [pilihan, "Jumlah"]

bar = px.bar(
    count_df,
    x=pilihan,
    y="Jumlah",
    color=pilihan,
    text="Jumlah"
)

bar.update_layout(showlegend=False)

st.plotly_chart(bar, use_container_width=True)

# ==========================================
# PIE CHART
# ==========================================

st.subheader(f"Pie Chart {pilihan}")

pie = px.pie(
    count_df,
    names=pilihan,
    values="Jumlah",
    hole=0.45
)

st.plotly_chart(pie, use_container_width=True)

# ==========================================
# HISTOGRAM AGE
# ==========================================

st.subheader("Distribusi Umur Pasien")

hist = px.histogram(
    df,
    x="Age",
    nbins=20,
    color_discrete_sequence=["royalblue"]
)

st.plotly_chart(hist, use_container_width=True)

# ==========================================
# BOXPLOT AGE
# ==========================================

st.subheader("Boxplot Umur")

box = px.box(
    df,
    y="Age",
    color_discrete_sequence=["orange"]
)

st.plotly_chart(box, use_container_width=True)

# ==========================================
# TARGET RECURRED
# ==========================================

st.subheader("Distribusi Target (Recurred)")

target_df = (
    df["Recurred"]
    .value_counts()
    .reset_index(name="Jumlah")
)

target_df.columns = ["Recurred", "Jumlah"]

target = px.bar(
    target_df,
    x="Recurred",
    y="Jumlah",
    color="Recurred",
    text="Jumlah"
)

target.update_layout(showlegend=False)

st.plotly_chart(target, use_container_width=True)

# ==========================================
# KORELASI
# ==========================================

st.subheader("Heatmap Korelasi")

data_corr = df.copy()

# Encoding otomatis
for col in data_corr.columns:
    if data_corr[col].dtype == "object":
        data_corr[col] = data_corr[col].astype("category").cat.codes

corr = data_corr.corr()

heatmap = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues"
)

st.plotly_chart(heatmap, use_container_width=True)

# ==========================================
# SCATTER PLOT
# ==========================================

st.subheader("Scatter Plot")

kolom_numerik = df.select_dtypes(include=["int64","float64"]).columns

x = st.selectbox("Sumbu X", kolom_numerik)

y = st.selectbox(
    "Sumbu Y",
    kolom_numerik,
    index=0
)

scatter = px.scatter(
    df,
    x=x,
    y=y,
    color="Recurred",
    size="Age",
    hover_data=df.columns
)

st.plotly_chart(scatter, use_container_width=True)

# ==========================================
# TABEL FREKUENSI
# ==========================================

st.subheader("Tabel Frekuensi")

st.dataframe(
    df[pilihan].value_counts().reset_index(),
    use_container_width=True
)

# ==========================================
# INFORMASI
# ==========================================

with st.expander("ℹ Informasi Visualisasi"):

    st.write("""
    Visualisasi ini digunakan untuk memahami karakteristik dataset, seperti:

    - Distribusi umur pasien
    - Distribusi jenis kelamin
    - Distribusi kekambuhan kanker
    - Korelasi antar fitur
    - Persebaran data numerik
    """)

st.markdown("---")

st.caption("Developed by Ervi | Universitas Respati Yogyakarta")
