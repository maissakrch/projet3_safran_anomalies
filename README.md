# Projet 3 — Détection d'Anomalies Moteurs Safran (MLOps)

## Contexte

Safran Data Systems surveille en continu des moteurs avioniques via des capteurs.
Ce projet développe un pipeline MLOps complet : entraînement d'un modèle de
détection d'anomalies, exposition via API REST, monitoring en temps réel,
tests automatisés et CI/CD.

Bloc de compétences validé : C9 à C13
Dataset : NASA CMAPSS (Turbofan Engine Degradation Simulation)

---

## Architecture du projet

projet3_safran_anomalies/
├── data/
│   ├── train_FD001.csv
│   └── predictions.csv
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── api/
│   │   └── main.py
│   ├── monitoring/
│   │   └── dashboard.py
│   └── tests/
│       └── test_model.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── specifications.md
├── .env
├── logger.py
├── pipeline.log
├── requirements.txt
└── README.md

---

## Installation

Prérequis : Python 3.13, pip, Git

Cloner le projet :
git clone https://github.com/maissakrch/projet3_safran_anomalies.git
cd projet3_safran_anomalies

Installer les dépendances :
pip install -r requirements.txt

Configurer le .env :
API_KEY=safran-anomaly-key-2024

---

## Utilisation

1. Entraîner le modèle :
python src/train_model.py

2. Lancer les prédictions :
python src/predict.py

3. Lancer l'API :
uvicorn src.api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

4. Lancer le dashboard de monitoring :
python src/monitoring/dashboard.py
Dashboard : http://127.0.0.1:8050

5. Lancer les tests :
pytest src/tests/test_model.py -v
pytest src/tests/test_model.py --cov=src --cov-report=term-missing

---

## Authentification API

Toutes les routes sauf / requièrent une clé API :
X-API-Key: safran-anomaly-key-2024

Exemple avec curl :
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: safran-anomaly-key-2024" \
  -H "Content-Type: application/json" \
  -d '{"unit": 1, "time": 1, "sensor_1": 518.67, "sensor_2": 641.82}'

curl -H "X-API-Key: safran-anomaly-key-2024" http://127.0.0.1:8000/stats
curl -H "X-API-Key: safran-anomaly-key-2024" http://127.0.0.1:8000/health

---

## Endpoints API

Méthode | Route     | Auth | Description
GET     | /         | Non  | Status de l'API
POST    | /predict  | Oui  | Détecter anomalie sur un point capteur
GET     | /stats    | Oui  | Statistiques globales de détection
GET     | /health   | Oui  | Santé du modèle

---

## Modèle IA

Algorithme            : Isolation Forest
Features              : 24 (3 settings + 21 capteurs)
Contamination         : 5%
Échantillons          : 20 631
Anomalies détectées   : 1 032 (5%)
Score moyen           : 0.1027

---

## Tests

Classe                | Tests | Résultat
TestModelLoading      | 3     | 3/3 PASSED
TestSinglePrediction  | 6     | 6/6 PASSED
TestBatchPrediction   | 4     | 4/4 PASSED
TestDataValidation    | 4     | 4/4 PASSED
Total                 | 17    | 17/17 PASSED

---

## CI/CD GitHub Actions

Pipeline automatique déclenché sur chaque push sur main :
1. Checkout du code
2. Installation Python 3.11
3. Installation des dépendances
4. Entraînement du modèle
5. Prédiction batch
6. Tests automatisés
7. Rapport de couverture

Repo : https://github.com/maissakrch/projet3_safran_anomalies

---

## Compétences validées

C9  | API REST exposant un modèle IA de détection d'anomalies | OK
C10 | Intégration API dans application existante              | Projet 4
C11 | Monitoring modèle avec dashboard Dash/Plotly            | OK
C12 | Tests automatisés pytest 17/17 avec couverture          | OK
C13 | CI/CD GitHub Actions pipeline complet                   | OK

---

## Dépendances

pandas
numpy
scikit-learn
fastapi
uvicorn
python-dotenv
dash
plotly
pytest
pytest-cov
joblib
requests