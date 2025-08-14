#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

OUTDIR = "model_ratio_mm"

# ---------- Reglas de negocio ----------
DEPREC_PRIMER_ANIO    = 0.22   # año 1: al menos -22% vs listPrice (cap en 0.78)
MIN_DEPREC_YRS_1_5    = 0.07   # años 2–5: al menos -7% vs precio FINAL del año previo
MAX_RATIO_VS_PREV_GT5 = 0.98   # años >5: <=98% del precio FINAL del año previo

app = FastAPI(title="API Ratio-MM Residual", version="2.2.0")

class PredictIn(BaseModel):
    Marca_Modelo: str = Field(..., description="Marca y Modelo, ej. 'Mazda Mazda 3'")
    Version: str
    Transmision: str
    Location: str
    Antiguedad: float
    Kilometraje: float
    listPrice: float  # requerido

# ----- Cargar artefactos -----
try:
    model   = joblib.load(os.path.join(OUTDIR, 'model_residual.pkl'))
    encs    = joblib.load(os.path.join(OUTDIR, 'encoders.pkl'))     # {'pre':..., 'te_mm':...}
    uplifts = joblib.load(os.path.join(OUTDIR, 'uplifts.pkl'))      # dict MM -> {Version: uplift}
    with open(os.path.join(OUTDIR, 'age_calibration.json'), 'r', encoding='utf-8') as f:
        AGE_CAL = json.load(f)
    with open(os.path.join(OUTDIR, 'km_calibration.json'), 'r', encoding='utf-8') as f:
        KM_CAL = json.load(f)
    # catálogo plano con EXACTAMENTE Marca/Modelo/Version
    catalog_path = os.path.join(OUTDIR, 'catalog.json')
    if os.path.exists(catalog_path):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            CATALOG = [
                {"Marca": str(it.get("Marca","")).strip(),
                 "Modelo": str(it.get("Modelo","")).strip(),
                 "Version": str(it.get("Version","")).strip()}
                for it in json.load(f)
            ]
    else:
        CATALOG = []
except Exception as e:
    raise RuntimeError(f"No pude cargar artefactos desde {OUTDIR}: {e}")

pre   = encs['pre']
te_mm = encs.get('te_mm', {})

# ----- Prior (idéntico al trainer) -----
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
    SALV   = 0.15
    k = float(max(0.0, min(km, MAX_KM)))
    x = k / MAX_KM
    exponent = 0.35 * x + 0.15 * max(0.0, x - 0.25)  # mayor pendiente > ~75k
    return max(SALV ** exponent, 0.12)

def make_prior(age: float, km: float) -> float:
    return prior_age(age) * prior_km(km)

def age_multiplier(age: float) -> float:
    b = int(round(max(0.0, min(age, 20.0))))
    return float(AGE_CAL.get(str(b), 1.0))

def km_multiplier(km: float) -> float:
    b = int(round(max(0.0, min(km, 300000.0)) / 20000.0))
    return float(KM_CAL.get(str(b), 1.0))

# ----- Features (igual que en training) -----
def build_features(inp: PredictIn):
    mm  = inp.Marca_Modelo.strip()
    ver = inp.Version.strip()
    mm_upl_dict = uplifts.get(mm, {})
    ver_uplift  = float(mm_upl_dict.get(ver, 1.0))

    X_cat = pd.DataFrame([{'Transmision': inp.Transmision, 'Location': inp.Location}])
    X_num = pd.DataFrame([{'Antiguedad': float(inp.Antiguedad), 'Kilometraje': float(inp.Kilometraje)}])

    X_basic = pre.transform(pd.concat([X_cat, X_num], axis=1))
    if not sp.issparse(X_basic):
        X_basic = sp.csr_matrix(X_basic)

    mm_te_default = float(np.mean(list(te_mm.values()))) if len(te_mm) else 0.0
    mm_te_val = float(te_mm.get(mm, mm_te_default))

    mm_te_sparse  = sp.csr_matrix([[mm_te_val]])
    uplift_sparse = sp.csr_matrix([[ver_uplift]])
    X = sp.hstack([X_basic, mm_te_sparse, uplift_sparse], format='csr')
    return X

def predict_raw_ratio(inp: PredictIn) -> float:
    """ prior × exp(residual̂) × calib_edad × calib_km (sin reglas de negocio) """
    X = build_features(inp)
    res_hat = float(model.predict(X)[0])  # residuo en log
    prior   = make_prior(inp.Antiguedad, inp.Kilometraje)
    ratio   = prior * float(np.exp(res_hat))
    ratio  *= age_multiplier(inp.Antiguedad)
    ratio  *= km_multiplier(inp.Kilometraje)
    return float(max(0.1, min(ratio, 1.2)))  # guard-rails suaves

# ----- Lógica de trayectoria monótona -----
def price_raw_for(age: float, total_km: float, data: PredictIn) -> float:
    """Precio crudo del modelo para una edad dada (sin reglas), con km anual consistente."""
    age = float(max(0.0, age))
    if age <= 0.0:
        km_at_age = 0.0
    else:
        km_per_year = max(0.0, total_km) / age
        km_at_age = km_per_year * age
    tmp = PredictIn(
        Marca_Modelo=data.Marca_Modelo, Version=data.Version,
        Transmision=data.Transmision, Location=data.Location,
        Antiguedad=age, Kilometraje=km_at_age, listPrice=data.listPrice
    )
    raw_ratio = predict_raw_ratio(tmp)
    return float(data.listPrice) * raw_ratio

def price_final_for(target_age: float, total_km: float, data: PredictIn) -> float:
    """
    Construye la trayectoria año por año (1..target_age) aplicando reglas:
      - Año 1: cap a (1 - 0.22)
      - Años 2–5: al menos -7% vs precio FINAL del año previo
      - Años >5:   <=98% vs precio FINAL del año previo
    Devuelve el precio FINAL en target_age.
    """
    age = float(target_age)
    if age <= 0.5:  # 0 años => precio crudo (sin reglas)
        return price_raw_for(0.0, total_km, data)

    # Año 1: precio FINAL = min(crudo, 0.78 × listPrice)
    p1_raw = price_raw_for(1.0, total_km, data)
    p1_cap = float(data.listPrice) * (1.0 - DEPREC_PRIMER_ANIO)
    p_prev = min(p1_raw, p1_cap)

    # Años 2..N
    y = 2
    target_int = int(round(age))
    while y <= target_int:
        p_y_raw = price_raw_for(float(y), total_km, data)
        p_y = p_y_raw

        if 2 <= y <= 5:
            max_allowed = p_prev * (1.0 - MIN_DEPREC_YRS_1_5)  # 93% del final previo
            if p_y > max_allowed:
                p_y = max_allowed
        elif y > 5:
            max_allowed = p_prev * MAX_RATIO_VS_PREV_GT5       # 98% del final previo
            if p_y > max_allowed:
                p_y = max_allowed

        p_prev = p_y
        y += 1

    return p_prev

# ----- Endpoints -----
@app.post("/predecir")
def predecir(data: PredictIn):
    try:
        if data.listPrice is None:
            raise HTTPException(status_code=400, detail="Falta listPrice en el payload.")

        # Precio crudo del modelo (compatibilidad / debugging)
        ratio_raw  = predict_raw_ratio(data)
        price_raw  = float(data.listPrice) * ratio_raw

        # Precio FINAL con reglas de negocio y km anual consistente
        price_final = price_final_for(float(data.Antiguedad), float(data.Kilometraje), data)
        ratio_final = price_final / float(data.listPrice)

        return {
            "precio_estimado": round(price_final, 2),
            "ratio": round(ratio_final, 6),
            "moneda": "MXN",
            "version_api": app.version
            # "debug": {"ratio_raw": ratio_raw, "price_raw": price_raw}  # opcional
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/catalogo")
def catalogo():
    """
    Devuelve un ARRAY de objetos con EXACTAMENTE:
      - Marca (str)
      - Modelo (str)
      - Version (str)
    """
    try:
        if CATALOG:
            return CATALOG
        # Fallback si no hay catalog.json: derivarlo de uplifts (split MM en "Marca Modelo")
        items: List[Dict[str, str]] = []
        for mm_key, vers in uplifts.items():
            parts = mm_key.split(" ", 1)
            marca = parts[0].strip()
            modelo = parts[1].strip() if len(parts) > 1 else ""
            for v in sorted(vers.keys()):
                items.append({"Marca": marca, "Modelo": modelo, "Version": str(v).strip()})
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"ok": True, "version": app.version, "requires": ["listPrice"], "endpoints": ["/predecir","/catalogo"]}

# uvicorn app_ratio_api_residual:app --host 0.0.0.0 --port 8000
