import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import pandas as pd
import numpy as np
import joblib
from src.predict import load_model, predict_single, predict

# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------

@pytest.fixture
def model_and_scaler():
    """Charge le modèle et le scaler pour les tests."""
    model, scaler = load_model()
    return model, scaler

@pytest.fixture
def sample_normal():
    """Exemple de données capteurs normales (valeurs moyennes NASA)."""
    return {
        "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0,
        "sensor_1": 518.67, "sensor_2": 641.82, "sensor_3": 1589.7,
        "sensor_4": 1400.6, "sensor_5": 14.62, "sensor_6": 21.61,
        "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19,
        "sensor_10": 1.3, "sensor_11": 47.47, "sensor_12": 521.66,
        "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195,
        "sensor_16": 0.03, "sensor_17": 392.0, "sensor_18": 2388.0,
        "sensor_19": 100.0, "sensor_20": 39.06, "sensor_21": 23.419
    }

@pytest.fixture
def sample_anomaly():
    """Exemple de données capteurs anormales (valeurs extremes)."""
    return {
        "op_setting_1": 99.9, "op_setting_2": 99.9, "op_setting_3": 99.9,
        "sensor_1": 9999.0, "sensor_2": 9999.0, "sensor_3": 9999.0,
        "sensor_4": 9999.0, "sensor_5": 9999.0, "sensor_6": 9999.0,
        "sensor_7": 9999.0, "sensor_8": 9999.0, "sensor_9": 9999.0,
        "sensor_10": 9999.0, "sensor_11": 9999.0, "sensor_12": 9999.0,
        "sensor_13": 9999.0, "sensor_14": 9999.0, "sensor_15": 9999.0,
        "sensor_16": 9999.0, "sensor_17": 9999.0, "sensor_18": 9999.0,
        "sensor_19": 9999.0, "sensor_20": 9999.0, "sensor_21": 9999.0
    }

@pytest.fixture
def sample_dataframe():
    """DataFrame de test avec 10 lignes."""
    return pd.read_csv("data/train_FD001.csv").head(10)

# ---------------------------------------------------------
# TESTS CHARGEMENT MODÈLE
# ---------------------------------------------------------

class TestModelLoading:

    def test_model_loads_successfully(self, model_and_scaler):
        """Le modèle et le scaler se chargent sans erreur."""
        model, scaler = model_and_scaler
        assert model is not None
        assert scaler is not None

    def test_model_files_exist(self):
        """Les fichiers modèle et scaler existent bien."""
        assert os.path.exists("models/isolation_forest.pkl"), "Fichier modèle manquant"
        assert os.path.exists("models/scaler.pkl"), "Fichier scaler manquant"

    def test_model_type(self, model_and_scaler):
        """Le modèle est bien un IsolationForest."""
        from sklearn.ensemble import IsolationForest
        model, _ = model_and_scaler
        assert isinstance(model, IsolationForest)

# ---------------------------------------------------------
# TESTS PRÉDICTION SINGLE
# ---------------------------------------------------------

class TestSinglePrediction:

    def test_prediction_returns_correct_keys(self, model_and_scaler, sample_normal):
        """La prédiction retourne les bonnes clés."""
        model, scaler = model_and_scaler
        result = predict_single(model, scaler, sample_normal)
        assert "anomalie" in result
        assert "score_anomalie" in result
        assert "statut" in result

    def test_prediction_anomalie_is_binary(self, model_and_scaler, sample_normal):
        """Le champ anomalie est 0 ou 1."""
        model, scaler = model_and_scaler
        result = predict_single(model, scaler, sample_normal)
        assert result["anomalie"] in [0, 1]

    def test_prediction_statut_values(self, model_and_scaler, sample_normal):
        """Le statut est NORMAL ou ANOMALIE."""
        model, scaler = model_and_scaler
        result = predict_single(model, scaler, sample_normal)
        assert result["statut"] in ["NORMAL", "ANOMALIE"]

    def test_prediction_score_is_float(self, model_and_scaler, sample_normal):
        """Le score est un nombre décimal."""
        model, scaler = model_and_scaler
        result = predict_single(model, scaler, sample_normal)
        assert isinstance(result["score_anomalie"], float)

    def test_anomaly_detected_on_extreme_values(self, model_and_scaler, sample_anomaly):
        """Des valeurs extrêmes doivent être détectées comme anomalies."""
        model, scaler = model_and_scaler
        result = predict_single(model, scaler, sample_anomaly)
        assert result["anomalie"] == 1, "Valeurs extrêmes doivent être des anomalies"
        assert result["statut"] == "ANOMALIE"

    def test_normal_values_score_higher(self, model_and_scaler, sample_normal, sample_anomaly):
        """Les données normales ont un score plus élevé que les anomalies."""
        model, scaler = model_and_scaler
        result_normal = predict_single(model, scaler, sample_normal)
        result_anomaly = predict_single(model, scaler, sample_anomaly)
        assert result_normal["score_anomalie"] > result_anomaly["score_anomalie"]

# ---------------------------------------------------------
# TESTS PRÉDICTION BATCH
# ---------------------------------------------------------

class TestBatchPrediction:

    def test_batch_prediction_returns_dataframe(self, model_and_scaler, sample_dataframe):
        """La prédiction batch retourne un DataFrame."""
        model, scaler = model_and_scaler
        result = predict(model, scaler, sample_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_batch_prediction_adds_columns(self, model_and_scaler, sample_dataframe):
        """Les colonnes anomalie, score et statut sont ajoutées."""
        model, scaler = model_and_scaler
        result = predict(model, scaler, sample_dataframe)
        assert "anomalie" in result.columns
        assert "score_anomalie" in result.columns
        assert "statut" in result.columns

    def test_batch_prediction_correct_length(self, model_and_scaler, sample_dataframe):
        """Le nombre de lignes est conservé."""
        model, scaler = model_and_scaler
        result = predict(model, scaler, sample_dataframe)
        assert len(result) == len(sample_dataframe)

    def test_batch_anomalie_values(self, model_and_scaler, sample_dataframe):
        """Tous les résultats sont 0 ou 1."""
        model, scaler = model_and_scaler
        result = predict(model, scaler, sample_dataframe)
        assert result["anomalie"].isin([0, 1]).all()

# ---------------------------------------------------------
# TESTS DONNÉES
# ---------------------------------------------------------

class TestDataValidation:

    def test_data_file_exists(self):
        """Le fichier de données existe."""
        assert os.path.exists("data/train_FD001.csv")

    def test_data_has_correct_columns(self):
        """Le fichier contient les colonnes attendues."""
        df = pd.read_csv("data/train_FD001.csv")
        expected = ["unit", "time", "sensor_1", "sensor_2"]
        for col in expected:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_data_not_empty(self):
        """Le fichier n'est pas vide."""
        df = pd.read_csv("data/train_FD001.csv")
        assert len(df) > 0

    def test_predictions_file_exists(self):
        """Le fichier de prédictions existe."""
        assert os.path.exists("data/predictions.csv")