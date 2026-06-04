# Spécifications Techniques — Détection d'Anomalies Moteurs Safran

## 1. Contexte et objectifs

**Projet :** API IA + MLOps pour la détection d'anomalies sur signaux capteurs
**Bloc de compétences :** C9 → C13
**Dataset :** NASA CMAPSS (Turbofan Engine Degradation Simulation)

### Acteurs
- Équipes qualité et ingénierie Safran (consommateurs de l'API)
- Développeur IA (réalisateur du projet)

### Contraintes techniques
- Python 3.13
- Environnement local (macOS)
- Budget : 0€
- Modèle entraînable sans GPU
- CI/CD via GitHub Actions

---

## 2. Modèle IA (C9)

### Algorithme : Isolation Forest

**Justification du choix :**
- Algorithme standard pour la détection d'anomalies non supervisée
- Pas besoin de données labellisées (anomalies/normaux)
- Efficace sur des données de capteurs multivariées
- Rapide à entraîner sur CPU
- Intégré dans scikit-learn (open source, gratuit)

### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| n_estimators | 100 | Nombre d'arbres, bon compromis vitesse/précision |
| contamination | 0.05 | 5% d'anomalies attendues dans les données |
| random_state | 42 | Reproductibilité des résultats |
| n_jobs | -1 | Utilisation de tous les CPUs disponibles |

### Features utilisées
- 3 paramètres opérationnels (op_setting_1, op_setting_2, op_setting_3)
- 21 capteurs (sensor_1 à sensor_21)
- Total : 24 features

### Résultats d'entraînement

| Métrique | Valeur |
|----------|--------|
| Échantillons entraînement | 20 631 |
| Normaux détectés | 19 599 (95%) |
| Anomalies détectées | 1 032 (5%) |
| Score moyen | 0.1027 |

### Normalisation
StandardScaler appliqué avant l'entraînement.
Scaler sauvegardé dans models/scaler.pkl pour cohérence train/predict.

---

## 3. API REST (C9)

### Stack technique
- Framework : FastAPI
- Documentation : Swagger UI (OpenAPI 3.1)
- Authentification : APIKeyHeader (X-API-Key)
- Sérialisation : Pydantic

### Endpoints

| Méthode | Route | Auth | Description |
|---------|-------|------|-------------|
| GET | `/` | Non | Status API |
| POST | `/predict` | Oui | Détecter anomalie sur un point capteur |
| GET | `/stats` | Oui | Statistiques globales détection |
| GET | `/health` | Oui | Santé du modèle |

### Authentification
X-API-Key: safran-anomaly-key-2024
Clé stockée dans .env, chargée via python-dotenv.

### Schéma entrée POST /predict
{
  "unit": 1,
  "time": 1,
  "op_setting_1": -0.0007,
  "sensor_1": 518.67,
  "...": "..."
}

### Schéma sortie POST /predict
{
  "anomalie": 0,
  "score_anomalie": 0.1234,
  "statut": "NORMAL",
  "interpretation": "Score 0.1234 — Comportement normal des capteurs moteur."
}

### Sécurité
- APIKeyHeader sur toutes les routes protégées
- HTTP 401 si clé invalide
- Pydantic pour validation des entrées
- Recommandations OWASP API Top 10 appliquées

---

## 4. Monitoring (C11)

### Outil : Dash/Plotly

**Justification :**
- 100% Python, pas de configuration serveur externe
- Dashboard interactif dans le navigateur
- Refresh automatique toutes les 30 secondes
- Gratuit et open source

### Métriques surveillées

| Métrique | Seuil d'alerte | Action recommandée |
|----------|---------------|-------------------|
| Taux anomalies > 10% | Alerte orange | Vérification manuelle |
| Taux anomalies > 20% | Alerte rouge | Inspection moteur urgente |
| Score moyen < 0.05 | Attention | Modèle potentiellement dégradé |

### Visualisations disponibles
- Timeline anomalies par cycle et unité moteur
- Pie chart répartition Normal/Anomalie
- Distribution des scores d'anomalie
- Top 20 unités par nombre d'anomalies
- 5 KPIs temps réel

### Accès
http://127.0.0.1:8050

---

## 5. Tests automatisés (C12)

### Framework : pytest + pytest-cov

### Couverture des tests

| Classe de tests | Nombre | Description |
|-----------------|--------|-------------|
| TestModelLoading | 3 | Chargement modèle et scaler |
| TestSinglePrediction | 6 | Prédiction sur un point |
| TestBatchPrediction | 4 | Prédiction sur DataFrame |
| TestDataValidation | 4 | Validation des données |
| Total | 17 | 17/17 passés |

### Exécution
pytest src/tests/test_model.py -v
pytest src/tests/test_model.py --cov=src --cov-report=term-missing

### Stratégie de test
- Fixtures pour modèle, données normales et anomalies
- Tests de régression sur valeurs extrêmes
- Validation des types de retour
- Validation de la cohérence des prédictions

---

## 6. CI/CD MLOps (C13)

### Outil : GitHub Actions

**Justification :**
- Intégration native avec GitHub
- Gratuit pour les repos publics et privés (2000 min/mois)
- Standard industrie pour les projets Python
- Déclenchement automatique sur push et pull request

### Pipeline CI/CD

Push/PR sur main
      ↓
Checkout code
      ↓
Installation Python 3.11
      ↓
Installation dépendances
      ↓
Entraînement modèle
      ↓
Prédiction batch
      ↓
Tests automatisés (17 tests)
      ↓
Rapport couverture

### Fichier de configuration
.github/workflows/ci.yml

### Déclencheurs
- Push sur branche main
- Pull request vers main

---

## 7. RGPD

Les données utilisées sont des données techniques de capteurs moteurs NASA.
Aucune donnée personnelle n'est collectée ou traitée.
Le registre des traitements est vide pour ce projet.

---

## 8. Dépendances

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