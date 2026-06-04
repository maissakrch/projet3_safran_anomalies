import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from logger import log

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/train_FD001.csv"
OUTPUT_PATH = "data/predictions.csv"

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"] + SENSOR_COLS

# ---------------------------------------------------------
# 1. Chargement du modèle
# ---------------------------------------------------------

def load_model():
    log("🤖 Chargement du modèle Isolation Forest...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    log("✔️ Modèle chargé avec succès")
    return model, scaler

# ---------------------------------------------------------
# 2. Prédiction sur un DataFrame
# ---------------------------------------------------------

def predict(model, scaler, df):
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0)
    X_scaled = scaler.transform(X)

    # -1 = anomalie, 1 = normal
    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df = df.copy()
    df["anomalie"] = (predictions == -1).astype(int)
    df["score_anomalie"] = scores
    df["statut"] = df["anomalie"].map({0: "NORMAL", 1: "ANOMALIE"})

    return df

# ---------------------------------------------------------
# 3. Prédiction sur un seul point
# ---------------------------------------------------------

def predict_single(model, scaler, features: dict):
    df_input = pd.DataFrame([features])
    available_cols = [c for c in FEATURE_COLS if c in df_input.columns]

    if not available_cols:
        raise ValueError("Aucune feature valide fournie")

    X = df_input[available_cols].fillna(0)

    # Aligner avec les features du scaler
    expected_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else available_cols
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]

    return {
        "anomalie": int(prediction == -1),
        "score_anomalie": round(float(score), 6),
        "statut": "ANOMALIE" if prediction == -1 else "NORMAL"
    }

# ---------------------------------------------------------
# 4. Statistiques
# ---------------------------------------------------------

def print_stats(df):
    total = len(df)
    anomalies = df["anomalie"].sum()
    normaux = total - anomalies

    log(f"\n📊 RÉSULTATS DE DÉTECTION D'ANOMALIES")
    log(f"   Total échantillons : {total}")
    log(f"   Normaux            : {normaux} ({round(normaux/total*100, 1)}%)")
    log(f"   Anomalies          : {anomalies} ({round(anomalies/total*100, 1)}%)")
    log(f"   Score moyen        : {round(df['score_anomalie'].mean(), 4)}")
    log(f"   Score min          : {round(df['score_anomalie'].min(), 4)}")
    log(f"   Score max          : {round(df['score_anomalie'].max(), 4)}")

# ---------------------------------------------------------
# POINT D'ENTRÉE
# ---------------------------------------------------------

if __name__ == "__main__":
    log("🚀 DÉMARRAGE DE LA DÉTECTION D'ANOMALIES — Safran")

    model, scaler = load_model()

    log("📁 Chargement des données de test...")
    df = pd.read_csv(DATA_PATH)

    df_result = predict(model, scaler, df)

    os.makedirs("data", exist_ok=True)
    df_result.to_csv(OUTPUT_PATH, index=False)

    print_stats(df_result)

    log(f"\n💾 Résultats sauvegardés : {OUTPUT_PATH}")
    log("🎉 Détection terminée.")