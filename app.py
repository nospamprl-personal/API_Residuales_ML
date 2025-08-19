#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

OUTDIR = "model_ratio_mm"

app = FastAPI(title="API Ratio-MM Residual", version="3.0.0")

class PredictIn(BaseModel):
    Marca_Modelo: str
    Version: str
    Transmision: str
    Location: str
    Antiguedad: float
    Kilometraje: float
    listPrice: float  # requerido

# ========== Carga de artefactos ==========
try:
    model   = joblib.load(os.path.join(OUTDIR, 'model_residual.pkl'))
    encs    = joblib.load(os.path.join(OUTDIR, 'encoders.pkl'))
    uplifts = joblib.load(os.path.join(OUTDIR, 'uplifts.pkl'))
    with open(os.path.join(OUTDIR, 'age_calibration.json'), 'r', encoding='utf-8') as f:
        AGE_CAL = json.load(f)
    with open(os.path.join(OUTDIR, 'km_calibration.json'), 'r', encoding='utf-8') as f:
        KM_CAL = json.load(f)
    catalog_path = os.path.join(OUTDIR, 'catalog.json')
    if os.path.exists(catalog_path):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            CATALOG = json.load(f)
    else:
        CATALOG = []
except Exception as e:
    raise RuntimeError(f"No pude cargar artefactos desde {OUTDIR}: {e}")

pre   = encs['pre']
te_mm = encs.get('te_mm', {})

# ========== Prior ==========
def prior_age(age: float) -> float:
    a = float(max(0.0, min(age, 20.0)))
    if a <= 1.0:
        return 0.95 - (0.95 - 0.78) * a
    val = 0.78 * (0.96 ** (a - 1.0))
    if a > 10.0:
        val *= (0.93 ** (a - 10.0))
    return max(val, 0.12)

def prior_km(km: float) -> float:
    MAX_KM = 300_000.0
    SALV = 0.15
    k = float(max(0.0, min(km, MAX_KM)))
    x = k / MAX_KM
    exponent = 0.35 * x + 0.15 * max(0.0, x - 0.25)
    return max(SALV ** exponent, 0.12)

def make_prior(age: float, km: float) -> float:
    return prior_age(age) * prior_km(km)

# ========== Calibraciones ==========
def apply_age_calibration(age: float) -> float:
    b = str(int(round(min(max(age, 0), 20))))
    return float(AGE_CAL.get(b, 1.0))

def apply_km_calibration(km: float) -> float:
    b = str(int(round(min(max(km / 20000.0, 0), 15))))
    return float(KM_CAL.get(b, 1.0))

# ========== Features ==========
def build_features(inp: PredictIn):
    mm = str(inp.Marca_Modelo).strip()
    ver = str(inp.Version).strip()
    u = uplifts.get(mm, {}).get(ver, 1.0)

    X_num = pd.DataFrame([{
        'Antiguedad': float(inp.Antiguedad),
        'Kilometraje': float(inp.Kilometraje),
        'pseudo_listPrice': float(inp.listPrice),
        'log_listPrice': np.log1p(float(inp.listPrice))
    }])

    X_cat = pd.DataFrame([{
        'Transmision': str(inp.Transmision).strip(),
        'Location': str(inp.Location).strip()
    }])

    X_basic = pre.transform(pd.concat([X_cat, X_num], axis=1))

    mm_te = float(te_mm.get(mm, np.mean(list(te_mm.values()))))
    mm_te_sparse = sp.csr_matrix(np.array([[mm_te]]))
    uplift_sparse = sp.csr_matrix(np.array([[u]]))

    return sp.hstack([X_basic, mm_te_sparse, uplift_sparse], format='csr')

# ========== Predicción ==========
@app.post("/predecir")
def predict(inp: PredictIn):
    try:
        prior_val = make_prior(inp.Antiguedad, inp.Kilometraje)
        X = build_features(inp)
        res_pred = float(model.predict(X)[0])

        ratio = prior_val * np.exp(res_pred)
        ratio *= apply_age_calibration(inp.Antiguedad)
        ratio *= apply_km_calibration(inp.Kilometraje)

        precio_estimado = float(inp.listPrice) * ratio

        return {
            "precio_estimado": precio_estimado,
            "ratio": ratio,
            "listPrice": inp.listPrice,
            "Marca_Modelo": inp.Marca_Modelo,
            "Version": inp.Version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

# ========== Catálogo ==========
@app.get("/catalogo")
def get_catalog() -> List[Dict[str, str]]:
    return CATALOG