from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import os, json, io
from typing import Optional, Literal, Tuple, List
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from unify_exoplanet_catalogs import unify

# SE PONE EL MODELO RAIZ
MODEL_PATH = "exoplanet_stacking_model.pkl"
FEATURE_ORDER_PATH = "feature_order.json"   # OPCIONAL
LABEL_ENCODER_PATH = "label_encoder.pkl"    # OPCIONAL

# COLUMNAS CRUDAS
TOI_COLUMNS = ['tfopwg_disp','pl_orbper','pl_trandurh','pl_trandep','pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid']
K2_COLUMNS  = ['disposition','pl_orbper','pl_trandur','pl_trandep','pl_rade','pl_eqt','pl_insol','st_teff','st_logg','st_rad','ra','dec','pl_tranmid']
KOI_COLUMNS = ['koi_disposition','koi_period','koi_duration','koi_depth','koi_prad','koi_teq','koi_insol','koi_steff','koi_slogg','koi_srad','ra','dec','koi_time0']

DEFAULT_LABELS = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
forced = os.getenv("EXO_LABELS")
if forced:
    DEFAULT_LABELS = [x.strip() for x in forced.split(",") if x.strip()]

app = FastAPI(title="Unified Exoplanet Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No pude cargar el modelo '{MODEL_PATH}': {e}")

FEATURE_ORDER = None
if os.path.exists(FEATURE_ORDER_PATH):
    try:
        with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
            FEATURE_ORDER = json.load(f).get("feature_order")
    except Exception:
        FEATURE_ORDER = None

CLASSES = None
if os.path.exists(LABEL_ENCODER_PATH):
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        if hasattr(label_encoder, "classes_"):
            CLASSES = list(label_encoder.classes_)
    except Exception:
        CLASSES = None

if CLASSES is None and hasattr(model, "classes_"):
    try:
        CLASSES = list(model.classes_)
    except Exception:
        CLASSES = None

MODEL_FEATURES = None
for attr in ("feature_names_in_", "X_columns_", "feature_order_"):
    if hasattr(model, attr):
        try:
            MODEL_FEATURES = list(getattr(model, attr))
            break
        except Exception:
            pass

# FUNCIONES AUXILIARES PARA UNIFY + PREDICCIÓN
def detect_tipo(df: pd.DataFrame) -> Literal["KOI","TOI","K2","UNKNOWN"]:
    if any(c in df.columns for c in KOI_COLUMNS): return "KOI"
    if any(c in df.columns for c in TOI_COLUMNS): return "TOI"
    if any(c in df.columns for c in K2_COLUMNS): return "K2"
    return "UNKNOWN"

# PRIORIDAD PARA UNIFY
def _priority_from_str(s: str) -> Tuple[str, str, str]:
    parts = [x.strip().lower() for x in s.split(",") if x.strip()]
    valid = [p for p in parts if p in ("toi","koi","k2")]
    base = ["toi","koi","k2"]
    for b in base:
        if b not in valid:
            valid.append(b)
    return tuple(valid[:3])  

# FUNCIONES DE UNIFY
def preprocess_with_unify(df: pd.DataFrame, tipo: str, priority: Tuple[str,str,str]) -> pd.DataFrame:
    df_koi = df if tipo == "KOI" else None
    df_toi = df if tipo == "TOI" else None
    df_k2  = df if tipo == "K2"  else None

    unified = unify(df_koi, df_toi, df_k2, priority=priority)
    if unified is None or unified.empty:
        return pd.DataFrame()

    for c in unified.columns:
        unified[c] = pd.to_numeric(unified[c], errors="coerce")
    unified = unified.loc[:, unified.notna().any(axis=0)]

    if unified.shape[1] > 0:
        imputer = IterativeImputer(max_iter=10, random_state=0)
        X = pd.DataFrame(imputer.fit_transform(unified), columns=unified.columns)
    else:
        X = unified.copy()

    target_order = FEATURE_ORDER or MODEL_FEATURES
    if target_order:
        for c in target_order:
            if c not in X.columns:
                X[c] = 0
        X = X[[c for c in target_order]]
    return X

# ELECCIÓN DE ID
def pick_id(df, tipo, i):
    try:
        if tipo == "TOI" and "toi" in df.columns and pd.notna(df.iloc[i].get("toi")):
            return str(df.iloc[i]["toi"])
        if tipo == "KOI" and "kepoi_name" in df.columns and pd.notna(df.iloc[i].get("kepoi_name")):
            return str(df.iloc[i]["kepoi_name"])
        if tipo == "K2":
            for cand in ("pl_name","tic_id","epic_id"):
                if cand in df.columns and pd.notna(df.iloc[i].get(cand)):
                    return str(df.iloc[i][cand])
    except Exception:
        pass
    return f"row_{i+1}"

# DECODIFICACIÓN DE ETIQUETAS
def decode_labels(preds) -> List[str]:
    if CLASSES is not None and len(CLASSES) > 0 and isinstance(CLASSES[0], str):
        try:
            return [str(CLASSES[int(i)]) for i in preds]
        except Exception:
            pass
    out = []
    for i in preds:
        try:
            idx = int(i)
        except Exception:
            idx = 0
        out.append(DEFAULT_LABELS[idx] if 0 <= idx < len(DEFAULT_LABELS) else str(idx))
    return out

# VALIDACIÓN DE COLUMNAS Y PREDICCIÓN
def validate_feature_count(X: pd.DataFrame):
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Incompatibilidad de columnas: el modelo espera {expected} features, "
                f"pero recibí {X.shape[1]}. Alinea columnas como en entrenamiento."
            )
        )

# PREDICCIÓN CON CONFIANZA
def predict_with_confidence(X: pd.DataFrame):
    """Devuelve (pred_idx, confidence_list). Si no hay predict_proba, confidence=None por fila."""
    y = model.predict(X)
    conf = [None] * len(y)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X) 
            for i, yi in enumerate(y):
                try:
                    conf[i] = float(proba[i][int(yi)])
                except Exception:
                    conf[i] = float(max(proba[i]))  
        except Exception:
            pass
    return y, conf

# ENDPOINT PARA HACER LA PREDICCIÓN
@app.post("/predict-excel")
async def predict_excel(
    file: UploadFile = File(...),
    tipo: Optional[str] = Query(default="auto"),
    priority: str = Query(default="toi,koi,k2", description="prioridad para unify, ej: toi,koi,k2"),
    limit: Optional[int] = Query(default=None, description="máx filas (paginación)"),
    offset: int = Query(default=0, description="desplazamiento (paginación)"),
    output: Literal["json","csv"] = Query(default="json", description="formato de salida"),
    include_cols: Optional[str] = Query(default=None, description="eco de columnas del input, separadas por coma"),
    min_conf: Optional[float] = Query(default=None, description="filtra por confianza mínima (0-1)"),
    only_label: Optional[str] = Query(default=None, description="filtra por etiqueta exacta"),
):
    try:
        fname = (file.filename or "").lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif fname.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado (.csv o .xlsx)")

        if df.empty:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        detected = detect_tipo(df) if tipo == "auto" else tipo.upper()
        if detected == "UNKNOWN":
            raise HTTPException(status_code=400, detail="No se detectó el tipo de catálogo (KOI/TOI/K2)")

        prio = _priority_from_str(priority)


        df_for_ids = df.copy()

        X = preprocess_with_unify(df.copy(), detected, prio)
        if X.empty:
            raise HTTPException(status_code=400, detail="No quedaron columnas útiles tras unify/preprocesamiento.")

        validate_feature_count(X)

        raw_preds, confidence = predict_with_confidence(X)
        labels_text = decode_labels(raw_preds)

        cols_to_echo = []
        if include_cols:
            cols_to_echo = [c.strip() for c in include_cols.split(",") if c.strip() in df_for_ids.columns]

        rows = []
        n = len(labels_text)
        for i in range(n):
            pid = pick_id(df_for_ids, detected, i)
            item = {"id": pid, "prediction": labels_text[i]}
            if confidence[i] is not None:
                item["confidence"] = confidence[i]
            if cols_to_echo:
                item["echo"] = {c: (None if pd.isna(df_for_ids.iloc[i].get(c)) else df_for_ids.iloc[i].get(c)) for c in cols_to_echo}
            rows.append(item)

        if only_label:
            rows = [r for r in rows if r["prediction"] == only_label]
        if min_conf is not None:
            rows = [r for r in rows if ("confidence" in r and r["confidence"] >= float(min_conf))]

        total_rows = len(rows)
        if offset < 0: offset = 0
        end = None if limit is None else offset + max(0, int(limit))
        rows = rows[offset:end]

        if output == "csv":
            df_out = pd.DataFrame(rows)
            buf = io.StringIO()
            df_out.to_csv(buf, index=False)
            buf.seek(0)
            return StreamingResponse(buf, media_type="text/csv", headers={
                "Content-Disposition": 'attachment; filename="predictions.csv"'
            })

        return {
            "catalog_type": detected,
            "priority": prio,
            "total_rows": total_rows,
            "features_in": list(X.columns),
            "offset": offset,
            "limit": (None if limit is None else int(limit)),
            "include_cols": cols_to_echo,
            "has_confidence": any("confidence" in r for r in rows),
            "results": rows
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ENDPOINTS DE DIAGNÓSTICO
@app.get("/health")
def health():
    return {
        "model_loaded": True,
        "feature_order_present": FEATURE_ORDER is not None,
        "model_features_present": MODEL_FEATURES is not None,
        "classes_present": CLASSES is not None,
        "model_n_features_in": getattr(model, "n_features_in_", None),
        "default_labels": DEFAULT_LABELS,
    }

# OBTENER COLUMNAS
@app.get("/debug-columns")
def debug_columns():
    return {
        "feature_order_json": FEATURE_ORDER,
        "model_feature_names_in": getattr(model, "feature_names_in_", None),
    }

# PARA PODER CORRERLO
# uvicorn api:app --reload --host 0.0.0.0 --port 8000
