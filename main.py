import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
    ]].copy()  # Usar .copy() para evitar SettingWithCopyWarning

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

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Solo CSV permitido.")

    path = os.path.join(DATA_FOLDER, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

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