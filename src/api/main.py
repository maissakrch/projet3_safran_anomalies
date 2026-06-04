import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from logger import log
from src.predict import load_model, predict_single, predict

load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

app = FastAPI(
    title="Safran Anomaly Detection API",
    description="API REST de détection d'anomalies sur signaux capteurs moteurs — Safran Data Systems",
    version="1.0.0"
)

API_KEY = os.getenv("API_KEY", "safran-anomaly-key-2024")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# ---------------------------------------------------------
# AUTHENTIFICATION
# ---------------------------------------------------------

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        log("❌ Tentative d'accès non autorisé")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante"
        )
    return api_key

# ---------------------------------------------------------
# CHARGEMENT MODÈLE AU DÉMARRAGE
# ---------------------------------------------------------

model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global model, scaler
    log("🚀 Démarrage API — chargement du modèle...")
    model, scaler = load_model()
    log("✔️ API prête")

# ---------------------------------------------------------
# SCHÉMAS
# ---------------------------------------------------------

class SensorInput(BaseModel):
    unit: Optional[int] = 1
    time: Optional[int] = 1
    op_setting_1: Optional[float] = 0.0
    op_setting_2: Optional[float] = 0.0
    op_setting_3: Optional[float] = 0.0
    sensor_1: Optional[float] = 0.0
    sensor_2: Optional[float] = 0.0
    sensor_3: Optional[float] = 0.0
    sensor_4: Optional[float] = 0.0
    sensor_5: Optional[float] = 0.0
    sensor_6: Optional[float] = 0.0
    sensor_7: Optional[float] = 0.0
    sensor_8: Optional[float] = 0.0
    sensor_9: Optional[float] = 0.0
    sensor_10: Optional[float] = 0.0
    sensor_11: Optional[float] = 0.0
    sensor_12: Optional[float] = 0.0
    sensor_13: Optional[float] = 0.0
    sensor_14: Optional[float] = 0.0
    sensor_15: Optional[float] = 0.0
    sensor_16: Optional[float] = 0.0
    sensor_17: Optional[float] = 0.0
    sensor_18: Optional[float] = 0.0
    sensor_19: Optional[float] = 0.0
    sensor_20: Optional[float] = 0.0
    sensor_21: Optional[float] = 0.0

class PredictionOutput(BaseModel):
    anomalie: int
    score_anomalie: float
    statut: str
    interpretation: str

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.get("/", tags=["Status"])
def root():
    """Vérifie que l'API est opérationnelle."""
    return {"message": "Safran Anomaly Detection API opérationnelle", "version": "1.0.0"}

@app.post(
    "/predict",
    tags=["Détection"],
    summary="Détecter une anomalie sur un point capteur",
    description="Prend les valeurs des capteurs d'un moteur et retourne si c'est une anomalie ou non.",
    response_model=PredictionOutput
)
def predict_anomaly(
    data: SensorInput,
    api_key: str = Depends(verify_api_key)
):
    log(f"🔍 Prédiction — unit={data.unit} time={data.time}")
    features = data.model_dump()
    result = predict_single(model, scaler, features)

    interpretation = (
        f"Score {result['score_anomalie']:.4f} — "
        + ("Comportement anormal détecté sur les capteurs moteur." if result["anomalie"] == 1
           else "Comportement normal des capteurs moteur.")
    )

    log(f"✔️ Résultat : {result['statut']} (score: {result['score_anomalie']})")

    return PredictionOutput(
        anomalie=result["anomalie"],
        score_anomalie=result["score_anomalie"],
        statut=result["statut"],
        interpretation=interpretation
    )

@app.get(
    "/stats",
    tags=["Statistiques"],
    summary="Statistiques globales de détection",
    description="Retourne les statistiques du dernier batch de prédictions."
)
def get_stats(api_key: str = Depends(verify_api_key)):
    if not os.path.exists("data/predictions.csv"):
        raise HTTPException(status_code=404, detail="Aucune prédiction disponible. Lancez predict.py d'abord.")
    df = pd.read_csv("data/predictions.csv")
    total = len(df)
    anomalies = int(df["anomalie"].sum())
    return {
        "total_echantillons": total,
        "normaux": total - anomalies,
        "anomalies": anomalies,
        "taux_anomalies": round(anomalies / total * 100, 2),
        "score_moyen": round(df["score_anomalie"].mean(), 4),
        "score_min": round(df["score_anomalie"].min(), 4),
        "score_max": round(df["score_anomalie"].max(), 4),
    }

@app.get(
    "/health",
    tags=["Status"],
    summary="Santé du modèle",
    description="Vérifie que le modèle est chargé et opérationnel."
)
def health(api_key: str = Depends(verify_api_key)):
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_type": "IsolationForest",
        "status": "healthy" if model is not None else "error"
    }