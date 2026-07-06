import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

API_KEY = os.getenv("API_KEY", "safran-anomaly-key-2024")

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# ---------------------------------------------------------
# TESTS ENDPOINTS RÉELS (via HTTP, pas d'appel direct aux fonctions)
# ---------------------------------------------------------

class TestAPIEndpoints:

    def test_root(self, client):
        """L'API répond sur la racine, sans authentification."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_requires_auth(self, client):
        """GET /health refuse l'accès sans clé API."""
        response = client.get("/health")
        assert response.status_code == 401

    def test_health_with_auth(self, client):
        """GET /health confirme que le modèle est bien chargé."""
        response = client.get("/health", headers={"X-API-Key": API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data

    def test_predict_requires_auth(self, client):
        """POST /predict refuse l'accès sans clé API."""
        response = client.post("/predict", json={"unit": 1, "time": 1})
        assert response.status_code == 401

    def test_predict_with_auth_returns_valid_response(self, client):
        """POST /predict retourne une prédiction complète et valide."""
        payload = {
            "unit": 1, "time": 1,
            "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0,
            "sensor_1": 518.67, "sensor_2": 641.82, "sensor_3": 1589.7,
            "sensor_4": 1400.6, "sensor_5": 14.62, "sensor_6": 21.61,
            "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19,
            "sensor_10": 1.3, "sensor_11": 47.47, "sensor_12": 521.66,
            "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195,
            "sensor_16": 0.03, "sensor_17": 392.0, "sensor_18": 2388.0,
            "sensor_19": 100.0, "sensor_20": 39.06, "sensor_21": 23.419
        }
        response = client.post("/predict", json=payload, headers={"X-API-Key": API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "anomalie" in data
        assert "score_anomalie" in data
        assert "statut" in data
        assert "interpretation" in data

    def test_stats_with_auth(self, client):
        """GET /stats retourne les statistiques du modèle."""
        response = client.get("/stats", headers={"X-API-Key": API_KEY})
        assert response.status_code == 200