import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ------------------------------
# CONFIG
# ------------------------------

DATA_FOLDER = "data"
MODEL_PATH = "modelo_riesgo.pkl"

os.makedirs(DATA_FOLDER, exist_ok=True)


# ------------------------------
# FASTAPI APP
# ------------------------------

app = FastAPI(title="Modelo Riesgo Accidentes Ecuador")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# MODELOS PYDANTIC
# ------------------------------

class PuntoRuta(BaseModel):
    lat: float
    lng: float

class RutaRequest(BaseModel):
    ruta: List[PuntoRuta]


# ------------------------------
# CREAR TARGET (alto riesgo)
# ------------------------------

def create_target(df: pd.DataFrame):
    """
    Regla simple:
    - gravedad == fatal OR victimas >= 3
    """
    df["numero_victimas"] = pd.to_numeric(df["numero_victimas"], errors="coerce").fillna(0)
    cond_fatal = df["gravedad"].astype(str).str.lower() == "fatal"
    cond_many = df["numero_victimas"] >= 3
    return (cond_fatal | cond_many).astype(int)


# ------------------------------
# ENTRENAR MODELO
# ------------------------------

def train_model(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8")

    # generar etiqueta
    df["alto_riesgo"] = create_target(df)

    # features
    X = df[[
        "latitud_accidente", "longitud_accidente",
        "tipo_accidente", "gravedad", "clima",
        "numero_victimas", "cantón", "provincia"
    ]].copy()

    y = df["alto_riesgo"]

    # conversión numérica
    X["latitud_accidente"] = pd.to_numeric(X["latitud_accidente"])
    X["longitud_accidente"] = pd.to_numeric(X["longitud_accidente"])
    X["numero_victimas"] = pd.to_numeric(X["numero_victimas"])

    num_cols = ["latitud_accidente", "longitud_accidente", "numero_victimas"]
    cat_cols = ["tipo_accidente", "gravedad", "clima", "cantón", "provincia"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    joblib.dump(pipe, MODEL_PATH)

    return {"status": "ok", "message": "Modelo entrenado y guardado."}


# ------------------------------
# PREDICT
# ------------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise Exception("Modelo no existe. Entrena primero.")
    return joblib.load(MODEL_PATH)


def predict_row(payload: dict):
    model = load_model()
    df = pd.DataFrame([payload])

    # asegurar numéricos
    df["latitud_accidente"] = pd.to_numeric(df["latitud_accidente"])
    df["longitud_accidente"] = pd.to_numeric(df["longitud_accidente"])
    df["numero_victimas"] = pd.to_numeric(df.get("numero_victimas", 0))

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    return {"prob_riesgo": float(prob), "alto_riesgo": bool(pred)}


# ------------------------------
# HEATMAP PNG
# ------------------------------

def generate_heatmap(df: pd.DataFrame, bins=100):
    plt.figure(figsize=(10, 8))

    lats = df["latitud_accidente"].values
    lons = df["longitud_accidente"].values

    plt.hist2d(lons, lats, bins=bins, norm=LogNorm(), cmap="hot")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Mapa de calor de accidentes")
    plt.colorbar(label="Densidad (log)")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


# ------------------------------
# GRID DE RIESGO (JSON)
# ------------------------------

def risk_grid(df: pd.DataFrame, cell_size=0.03):
    df["alto_riesgo"] = create_target(df)

    min_lat, max_lat = df["latitud_accidente"].min(), df["latitud_accidente"].max()
    min_lon, max_lon = df["longitud_accidente"].min(), df["longitud_accidente"].max()

    # grilla
    grid = []

    lat_bins = np.arange(min_lat, max_lat + cell_size, cell_size)
    lon_bins = np.arange(min_lon, max_lon + cell_size, cell_size)

    for lat_min in lat_bins:
        for lon_min in lon_bins:
            lat_max = lat_min + cell_size
            lon_max = lon_min + cell_size

            cell = df[
                (df["latitud_accidente"] >= lat_min) &
                (df["latitud_accidente"] < lat_max) &
                (df["longitud_accidente"] >= lon_min) &
                (df["longitud_accidente"] < lon_max)
            ]

            if len(cell) == 0:
                continue

            risk_rate = cell["alto_riesgo"].mean()

            grid.append({
                "centroid": [float(lon_min + cell_size/2), float(lat_min + cell_size/2)],
                "count": int(len(cell)),
                "risk_rate": float(risk_rate)
            })

    return grid


# ======================================================
# FASTAPI ENDPOINTS
# ======================================================

@app.get("/")
def root():
    return {
        "message": "API de Predicción de Riesgo de Accidentes",
        "version": "1.0",
        "endpoints": {
            "upload": "/upload-csv/",
            "train": "/train/",
            "predict": "/predict/",
            "predict_ruta": "/predict-ruta/",
            "model_status": "/model-status/",
            "heatmap": "/heatmap/",
            "risk_grid": "/risk-grid/"
        }
    }


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Solo CSV permitido.")

    path = os.path.join(DATA_FOLDER, file.filename)

    with open(path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    return {"msg": "CSV subido", "filename": file.filename}


@app.post("/train/")
def train(csv_filename: str):
    path = os.path.join(DATA_FOLDER, csv_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Archivo no encontrado")

    return train_model(path)


@app.post("/predict/")
def predict(payload: dict):
    try:
        return predict_row(payload)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict-ruta/")
def predict_ruta(request: RutaRequest):
    """
    Recibe una ruta como lista de puntos con lat/lng
    """
    # VERIFICAR SI EL MODELO EXISTE
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=400, 
            detail="El modelo no ha sido entrenado aún. Por favor, entrena el modelo primero desde la sección 'Entrenar IA'."
        )
    
    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al cargar el modelo: {str(e)}"
        )
    
    resultados = []
    
    for punto in request.ruta:
        payload = {
            "latitud_accidente": punto.lat,
            "longitud_accidente": punto.lng,
            "numero_victimas": 0,
            "tipo_accidente": "ruta",
            "gravedad": "leve",
            "clima": "normal",
            "cantón": "Cuenca",
            "provincia": "Azuay"
        }
        try:
            resultado = predict_row(payload)
            resultados.append(resultado)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error en la predicción: {str(e)}"
            )
    
    return {"resultados": resultados}


@app.get("/model-status/")
def model_status():
    """
    Retorna el estado del modelo
    """
    exists = os.path.exists(MODEL_PATH)
    return {
        "model_trained": exists,
        "model_path": MODEL_PATH,
        "message": "Modelo entrenado y listo" if exists else "Modelo no entrenado. Entrena el modelo primero."
    }


@app.get("/heatmap/")
def heatmap(csv_filename: str, bins: int = 100):
    path = os.path.join(DATA_FOLDER, csv_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "CSV no encontrado")

    df = pd.read_csv(path)
    buf = generate_heatmap(df, bins)

    tmp = os.path.join(DATA_FOLDER, "heatmap.png")
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())

    return FileResponse(tmp, media_type="image/png", filename="heatmap.png")


@app.get("/risk-grid/")
def riskgrid(csv_filename: str):
    path = os.path.join(DATA_FOLDER, csv_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "CSV no encontrado")

    df = pd.read_csv(path)
    grid = risk_grid(df)

    return JSONResponse({"cells": grid})


# ------------------------------
# Rutas seguras (opcional)
# ------------------------------
@app.get("/rutas_seguras/")
def rutas_seguras(destino: str, csv_filename: str = "accidentes_sinteticos_ecuador_cuen.csv"):
    path = os.path.join(DATA_FOLDER, csv_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "CSV no encontrado")
    df = pd.read_csv(path)
    df["alto_riesgo"] = create_target(df)
    df_seguro = df[df["alto_riesgo"] == 0]

    if df_seguro.empty:
        raise HTTPException(404, "No hay rutas seguras disponibles")

    rutas = []
    for i in range(3):
        rutas.append({
            "nombre": f"Ruta segura {i+1}",
            "camino": [
                {"lat": -0.170653 + i*0.01, "lng": -78.457834 + i*0.01}
            ]
        })
    return JSONResponse(rutas)


# ------------------------------
# Iniciar servidor
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)