import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from logger import log

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_PATH = "data/train_FD001.csv"
MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

# Colonnes capteurs NASA CMAPSS
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"] + SENSOR_COLS

# Contamination : proportion d'anomalies attendues (5%)
CONTAMINATION = 0.05

# ---------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------

def load_data():
    log("📁 Chargement des données NASA CMAPSS...")
    df = pd.read_csv(DATA_PATH)
    log(f"✔️ {len(df)} lignes chargées")
    return df

# ---------------------------------------------------------
# 2. Préparation des features
# ---------------------------------------------------------

def prepare_features(df):
    log("🔧 Préparation des features...")

    # Sélection des colonnes pertinentes
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].copy()

    # Suppression des valeurs manquantes
    X = X.dropna()

    log(f"✔️ {len(available_cols)} features sélectionnées")
    log(f"✔️ Shape : {X.shape}")
    return X

# ---------------------------------------------------------
# 3. Normalisation
# ---------------------------------------------------------

def normalize_features(X, fit=True):
    log("📐 Normalisation des features...")
    scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
        log(f"💾 Scaler sauvegardé : {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    return X_scaled, scaler

# ---------------------------------------------------------
# 4. Entraînement du modèle
# ---------------------------------------------------------

def train_model(X_scaled):
    log("🤖 Entraînement du modèle Isolation Forest...")
    log(f"   Contamination : {CONTAMINATION} ({int(CONTAMINATION*100)}% d'anomalies attendues)")

    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    # Statistiques d'entraînement
    predictions = model.predict(X_scaled)
    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()

    log(f"✔️ Modèle entraîné sur {len(X_scaled)} échantillons")
    log(f"   Normaux   : {n_normal} ({round(n_normal/len(X_scaled)*100, 1)}%)")
    log(f"   Anomalies : {n_anomalies} ({round(n_anomalies/len(X_scaled)*100, 1)}%)")

    return model

# ---------------------------------------------------------
# 5. Sauvegarde du modèle
# ---------------------------------------------------------

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log(f"💾 Modèle sauvegardé : {MODEL_PATH}")

# ---------------------------------------------------------
# POINT D'ENTRÉE
# ---------------------------------------------------------

if __name__ == "__main__":
    log("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT — Détection d'anomalies Safran")

    df = load_data()
    X = prepare_features(df)
    X_scaled, scaler = normalize_features(X, fit=True)
    model = train_model(X_scaled)
    save_model(model)

    log("🎉 Entraînement terminé avec succès !")