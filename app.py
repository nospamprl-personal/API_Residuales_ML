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

# ---------- Reglas de negocio ----------
# Se mantienen solo las reglas lógicas.
DEPREC_PRIMER_ANIO = 0.22

app = FastAPI(title="API Ratio-MM Residual", version="6.0.0")

class PredictIn(BaseModel):
    Marca_Modelo: str
    Version: str
    Transmision: str
    Location: str
    Antiguedad: float
    Kilometraje: float
    listPrice: float

# ========== Carga de artefactos =========
try:
    model = joblib.load(os.path.join(OUTDIR, 'model_residual.pkl'))
    encs = joblib.load(os.path.join(OUTDIR, 'encoders.pkl'))
    te_mm = encs['te_mm']
    te_mm_age = encs['te_mm_age']
    pre = encs['pre']
    uplifts = joblib.load(os.path.join(OUTDIR, 'uplifts.pkl'))
    with open(os.path.join(OUTDIR, 'age_calibration.json'), 'r', encoding='utf-8') as f:
        AGE_CAL = json.load(f)
    with open(os.path.join(OUTDIR, 'km_calibration.json'), 'r', encoding='utf-8') as f:
        KM_CAL = json.load(f)
    with open(os.path.join(OUTDIR, 'catalog.json'), 'r', encoding='utf-8') as f:
        CATALOG = json.load(f)

    print("Artefactos del modelo cargados correctamente.")

except Exception as e:
    print(f"Error al cargar los artefactos del modelo: {e}")
    raise RuntimeError("No se pudieron cargar los archivos del modelo. Asegúrese de ejecutar el script de entrenamiento primero.")

# ========== Prior ==========
def make_prior(age: float, km: float) -> float:
    a = float(max(0.0, min(age, 20.0)))
    if a <= 1.0:
        val_age = 0.95 - (0.95 - 0.78) * a
    else:
        val_age = 0.78 * (0.96 ** (a - 1.0))
        if a > 10.0:
            val_age *= (0.93 ** (a - 10.0))
    val_age = max(val_age, 0.12)
    
    MAX_KM = 300_000.0
    SALV = 0.15
    k = float(max(0.0, min(km, MAX_KM)))
    x = k / MAX_KM
    exponent = 0.35 * x + 0.15 * max(0.0, x - 0.25)
    val_km = max(SALV ** exponent, 0.12)
    
    return val_age * val_km

# ========== Calibración ==========
def apply_age_calibration(age: float) -> float:
    age_str = str(int(np.floor(age)))
    return AGE_CAL.get(age_str, 1.0)

def apply_km_calibration(km: float) -> float:
    km_bin = np.floor(km / 20000) * 20000
    km_str = str(int(km_bin))
    return KM_CAL.get(km_str, 1.0)

# ========== Feature Engineering ==========
def build_features(inp: PredictIn) -> sp.csr_matrix:
    df = pd.DataFrame([{
        'Transmision': inp.Transmision,
        'Location': inp.Location,
        'Antiguedad': inp.Antiguedad,
        'Kilometraje': inp.Kilometraje,
        'pseudo_listPrice': inp.listPrice,
        'Marca_Modelo': inp.Marca_Modelo
    }])
    
    bins_age = [0, 1, 2, 3, 5, 10, 20]
    labels_age = ['0-1', '1-2', '2-3', '3-5', '5-10', '10+']
    df['age_group'] = pd.cut(df['Antiguedad'], bins=bins_age, labels=labels_age, right=False)
    df['mm_age_group'] = df['Marca_Modelo'] + '_' + df['age_group'].astype(str)

    X_basic = pre.transform(df)

    mm_te_val = float(te_mm.get(df['Marca_Modelo'].iloc[0], np.mean(list(te_mm.values()))))
    mm_te_sparse = sp.csr_matrix(np.array([[mm_te_val]]))
    
    mm_age_te_val = float(te_mm_age.get(df['mm_age_group'].iloc[0], np.mean(list(te_mm_age.values()))))
    mm_age_te_sparse = sp.csr_matrix(np.array([[mm_age_te_val]]))

    u = uplifts.get(df['Marca_Modelo'].iloc[0], {}).get(inp.Version, 1.0)
    uplift_sparse = sp.csr_matrix(np.array([[u]]))

    return sp.hstack([X_basic, mm_te_sparse, mm_age_te_sparse, uplift_sparse], format='csr')

# ========== Endpoints =========
@app.get("/catalogo")
def get_catalogo():
    return CATALOG

@app.post("/predecir")
def predict(inp: PredictIn):
    try:
        prior_val = make_prior(inp.Antiguedad, inp.Kilometraje)
        X = build_features(inp)
        res_pred = float(model.predict(X)[0])

        ratio = prior_val * np.exp(res_pred)
        ratio *= apply_age_calibration(inp.Antiguedad)
        ratio *= apply_km_calibration(inp.Kilometraje)

        # Se aplica la única regla de negocio lógica: no puede tener una depreciación menor a la del primer año
        if inp.Antiguedad > 0:
            ratio = min(ratio, 1.0 - DEPREC_PRIMER_ANIO)

        precio_estimado = float(inp.listPrice) * ratio

        return {
            "precio_estimado": precio_estimado,
            "depreciacion_porcentaje": (1 - ratio) * 100
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))