import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# MEMBACA DATASET
# =====================================================

df = pd.read_csv("Thyroid_Diff.csv")

print("=" * 50)
print("Dataset berhasil dibaca")
print(df.head())
print("=" * 50)

# =====================================================
# CEK MISSING VALUE
# =====================================================

print("\nMissing Value")
print(df.isnull().sum())

# =====================================================
# ENCODING DATA KATEGORIKAL
# =====================================================

encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

        encoders[column] = le

# =====================================================
# MEMISAHKAN FITUR DAN TARGET
# =====================================================

X = df.drop("Recurred", axis=1)
y = df["Recurred"]

# =====================================================
# NORMALISASI
# =====================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nJumlah Data Training :", len(X_train))
print("Jumlah Data Testing  :", len(X_test))

# =====================================================
# MEMBUAT MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# =====================================================
# TRAINING
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# PREDIKSI
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUASI
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# SIMPAN MODEL
# =====================================================

joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel berhasil disimpan")
print("model.pkl")
print("encoder.pkl")
print("scaler.pkl")