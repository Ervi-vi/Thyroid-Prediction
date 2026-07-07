import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI
# ==========================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Evaluasi Model Machine Learning")

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("Thyroid_Diff.csv")

df = load_data()

# ==========================================
# ENCODING
# ==========================================

encoder = {}

for col in df.columns:

    # Semua kolom selain numerik akan di-encode
    if not pd.api.types.is_numeric_dtype(df[col]):

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoder[col] = le

# ==========================================
# SPLIT DATA
# ==========================================

X = df.drop("Recurred", axis=1)

y = df["Recurred"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("model.pkl")

# ==========================================
# PREDIKSI
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# METRIK
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

# ==========================================
# METRIC CARD
# ==========================================

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.metric("Accuracy",f"{accuracy*100:.2f}%")

with c2:

    st.metric("Precision",f"{precision*100:.2f}%")

with c3:

    st.metric("Recall",f"{recall*100:.2f}%")

with c4:

    st.metric("F1 Score",f"{f1*100:.2f}%")

st.markdown("---")

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(5,5))

ConfusionMatrixDisplay.from_predictions(

    y_test,

    y_pred,

    cmap="Blues",

    ax=ax

)

st.pyplot(fig)

# ==========================================
# ROC CURVE
# ==========================================

st.subheader("ROC Curve")

prob = model.predict_proba(X_test)[:,1]

fpr,tpr,_ = roc_curve(y_test,prob)

roc_auc = auc(fpr,tpr)

fig2,ax2 = plt.subplots(figsize=(6,5))

ax2.plot(fpr,tpr,label=f"AUC = {roc_auc:.3f}")

ax2.plot([0,1],[0,1],'--')

ax2.set_xlabel("False Positive Rate")

ax2.set_ylabel("True Positive Rate")

ax2.set_title("ROC Curve")

ax2.legend()

st.pyplot(fig2)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("Feature Importance")

importance = pd.DataFrame({

    "Feature":df.drop("Recurred",axis=1).columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

fig3 = px.bar(

    importance,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance"

)

st.plotly_chart(fig3,use_container_width=True)

# ==========================================
# HASIL EVALUASI
# ==========================================

st.subheader("Interpretasi")

if accuracy >= 0.90:

    st.success("""

Model memiliki performa yang sangat baik.

Model layak digunakan sebagai sistem pendukung keputusan
untuk prediksi kekambuhan kanker tiroid.

""")

elif accuracy >=0.80:

    st.warning("""

Model cukup baik namun masih dapat ditingkatkan.

""")

else:

    st.error("""

Model masih kurang baik sehingga perlu dilakukan tuning
parameter atau menggunakan algoritma lain.

""")

# ==========================================
# TABEL FEATURE IMPORTANCE
# ==========================================

st.subheader("Ranking Feature")

st.dataframe(

    importance,

    use_container_width=True

)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption("Random Forest Evaluation | Developed by Ervi")
