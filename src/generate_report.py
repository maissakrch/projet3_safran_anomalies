import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from datetime import datetime
from logger import log

MODEL_PATH = "models/isolation_forest.pkl"
PREDICTIONS_PATH = "data/predictions.csv"
REPORT_PATH = "evaluation_report.md"

def generate_report():
    log("📊 Génération du rapport d'évaluation...")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PREDICTIONS_PATH)

    total = len(df)
    anomalies = int(df["anomalie"].sum())
    normaux = total - anomalies
    taux = round(anomalies / total * 100, 2)
    score_moyen = round(df["score_anomalie"].mean(), 4)
    score_min = round(df["score_anomalie"].min(), 4)
    score_max = round(df["score_anomalie"].max(), 4)

    report = f"""# Rapport d'évaluation — Modèle de détection d'anomalies

**Date de génération :** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Hyperparamètres du modèle
- n_estimators : {model.n_estimators}
- contamination : {model.contamination}
- random_state : {model.random_state}

## Résultats sur le jeu de données
| Métrique | Valeur |
|---|---|
| Total échantillons | {total} |
| Normaux | {normaux} |
| Anomalies | {anomalies} |
| Taux d'anomalies | {taux}% |
| Score moyen | {score_moyen} |
| Score minimum | {score_min} |
| Score maximum | {score_max} |

## Note méthodologique
Ce modèle étant non supervisé, il n'existe pas de vraie catégorie connue
à l'avance pour calculer une matrice de confusion classique (accuracy,
precision, recall). L'évaluation se fait donc sur la cohérence entre le
taux d'anomalies observé et le taux de contamination fixé à l'entraînement,
ainsi que sur la distribution des scores produits.
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    log(f"✔️ Rapport généré : {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()