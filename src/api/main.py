from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Métriques Prometheus
REQUEST_COUNT = Counter('prediction_requests_total', 'Nombre total de requêtes de prédiction')
REQUEST_LATENCY = Histogram('prediction_request_duration_seconds', 'Latence des prédictions')

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Classification API", version="1.0.0")

# Modèles globaux
model = None
scaler = None
feature_names = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    confidence: float

@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage"""
    global model, scaler, feature_names
    
    try:
        # Charger le modèle
        with open("models/classifier.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Charger le scaler
        with open("data/processed/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Charger les noms de features depuis les données d'entraînement
        with open("data/processed/train.pkl", 'rb') as f:
            train_data = pickle.load(f)
            feature_names = train_data['feature_names']
        
        logger.info("Modèles chargés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        raise

@app.get("/health")
async def health_check():
    """Point de contrôle de santé"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint de prédiction"""
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé")
        
        # Validation des features
        if len(request.features) != len(feature_names):
            raise HTTPException(
                status_code=400, 
                detail=f"Nombre de features incorrect. Attendu: {len(feature_names)}, reçu: {len(request.features)}"
            )
        
        # Préparation des données
        features_array = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        response = PredictionResponse(
            prediction=int(prediction),
            probability=probabilities.tolist(),
            confidence=confidence
        )
        
        # Enregistrer la latence
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Endpoint pour les métriques Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
async def model_info():
    """Informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "feature_names": feature_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)