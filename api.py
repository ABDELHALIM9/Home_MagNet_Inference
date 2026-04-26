"""
CNNv3 FastAPI — multi-material inference API
=============================================
Automatically discovers all material models from a local `models/` folder:

  models/
    N87/
      cnnv3_best.pth
      stats.json
    N49/
      cnnv3_best.pth
      stats.json

Endpoints
---------
  GET  /health                    – liveness + list of loaded materials
  GET  /models                    – detail info for every loaded model
  POST /predict/bh/{material}     – predict from pre-computed B/H waveforms
  

Run
---
  uvicorn api:app --host 0.0.0.0 --port 8000

Environment variables (all optional):
  MODELS_DIR    path to the models folder     (default: ./models)
  CONFIG_PATH   path to a shared config.yaml  (default: None)
  DEVICE        auto | cpu | cuda             (default: auto)
  BATCH_SIZE    forward-pass batch size        (default: 256)
"""

from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from cnnv3_inference import CNNv3Inferencer  

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("cnnv3-api")

# ─────────────────────────────────────────────────────────────────────────────
# Model registry  { material_name -> CNNv3Inferencer }
# ─────────────────────────────────────────────────────────────────────────────
_registry: Dict[str, CNNv3Inferencer] = {}


def _discover_models(models_dir: str, device: str, batch_size: int, config: Optional[str]):
    """
    Walk `models_dir`. For every sub-folder that contains both
    `cnnv3_best.pth` and `stats.json`, load a CNNv3Inferencer and
    register it under the sub-folder name (= material name).
    """
    if not os.path.isdir(models_dir):
        raise RuntimeError(
            f"Models directory not found: '{models_dir}'\n"
            f"Set the MODELS_DIR env var or create the folder."
        )

    loaded, skipped = [], []

    for entry in sorted(os.scandir(models_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue

        material   = entry.name
        ckpt_path  = os.path.join(entry.path, "cnnv3_best.pth")
        stats_path = os.path.join(entry.path, "stats.json")

        missing = [f for f in (ckpt_path, stats_path) if not os.path.isfile(f)]
        if missing:
            log.warning(f"  [{material}] skipped — missing: {[os.path.basename(f) for f in missing]}")
            skipped.append(material)
            continue

        log.info(f"  [{material}] loading …")
        try:
            _registry[material] = CNNv3Inferencer(
                checkpoint_path = ckpt_path,
                stats_path      = stats_path,
                config_path     = config,
                device          = device,
                batch_size      = batch_size,
            )
            loaded.append(material)
            log.info(f"  [{material}] ready ✓")
        except Exception as exc:
            log.error(f"  [{material}] FAILED to load: {exc}")
            skipped.append(material)

    if not _registry:
        raise RuntimeError(
            f"No valid models found in '{models_dir}'. "
            "Each material must have its own sub-folder containing "
            "cnnv3_best.pth and stats.json."
        )

    log.info(f"Loaded {len(loaded)} model(s): {loaded}")
    if skipped:
        log.warning(f"Skipped {len(skipped)} folder(s): {skipped}")


def _get_inferencer(material: str) -> CNNv3Inferencer:
    inf = _registry.get(material)
    if inf is None:
        available = sorted(_registry.keys())
        raise HTTPException(
            status_code = 404,
            detail      = f"Material '{material}' not found. Available: {available}",
        )
    return inf


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — discover + load all models on startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
    config     = os.getenv("CONFIG_PATH", None) or None
    device     = os.getenv("DEVICE",     "auto")
    batch_size = int(os.getenv("BATCH_SIZE", "256"))

    log.info(f"Discovering models in '{models_dir}' …")
    _discover_models(models_dir, device, batch_size, config)
    yield
    _registry.clear()
    log.info("Shutting down — registry cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "CNNv3 Multi-Material Inference API",
    description = (
        "Predicts core-loss (W/m³) for different magnetic materials.\n\n"
        "Models are auto-discovered from the `models/` folder — one sub-folder per material."
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
)
# need to edit 
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class BHSample(BaseModel):
    """One sample with pre-computed B and H waveforms."""
    B:           List[float] = Field(..., description="Flux-density waveform (T), length T")
    H:           List[float] = Field(..., description="Magnetic-field waveform (A/m), length T")
    Frequency:   float       = Field(..., gt=0, description="Excitation frequency (Hz)")
    Temperature: float       = Field(..., description="Operating temperature (°C)")

    @model_validator(mode="after")
    def lengths_match(self):
        if len(self.B) != len(self.H):
            raise ValueError(f"B and H length mismatch ({len(self.B)} vs {len(self.H)})")
        if len(self.B) < 4:
            raise ValueError("Waveform must have at least 4 points")
        return self


class BHRequest(BaseModel):
    samples:    List[BHSample] = Field(..., min_length=1)
    return_log: bool           = Field(False, description="Return log10(Loss) instead of W/m³")



class PredictionResult(BaseModel):
    sample_index:   int
    predicted_loss: float
    unit:           str = "W/m³"


class PredictResponse(BaseModel):
    material:          str
    results:           List[PredictionResult]
    count:             int
    inference_time_ms: float


class ModelInfo(BaseModel):
    material:   str
    device:     str
    batch_size: int


class HealthResponse(BaseModel):
    status:           str
    loaded_materials: List[str]
    count:            int


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_response(
    material: str,
    preds: np.ndarray,
    return_log: bool,
    t0: float,
) -> PredictResponse:
    unit = "log10(W/m³)" if return_log else "W/m³"
    return PredictResponse(
        material          = material,
        results           = [
            PredictionResult(sample_index=i, predicted_loss=float(p), unit=unit)
            for i, p in enumerate(preds)
        ],
        count             = len(preds),
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Liveness check — lists all loaded materials."""
    materials = sorted(_registry.keys())
    return HealthResponse(
        status           = "ok",
        loaded_materials = materials,
        count            = len(materials),
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Meta"])
def list_models():
    """Return info for every loaded material model."""
    return [
        ModelInfo(
            material   = mat,
            device     = str(inf.device),
            batch_size = inf.batch_size,
        )
        for mat, inf in sorted(_registry.items())
    ]


@app.post("/predict/bh/{material}", response_model=PredictResponse, tags=["Inference"])
def predict_bh(
    req:      BHRequest,
    material: str = Path(..., description="Material name, e.g. N87 or N49"),
):
    """
    Predict core loss from **pre-computed B / H waveforms** for a given material.

    - `material` in the URL must match a sub-folder name inside `models/`.
    - Set `return_log=true` to get log₁₀(W/m³) instead of W/m³.
    """
    inf = _get_inferencer(material)
    t0  = time.perf_counter()
    try:
        B    = np.array([s.B           for s in req.samples], dtype=np.float32)
        H    = np.array([s.H           for s in req.samples], dtype=np.float32)
        freq = np.array([s.Frequency   for s in req.samples], dtype=np.float32)
        temp = np.array([s.Temperature for s in req.samples], dtype=np.float32)
        preds = inf.predict_from_bh(B, H, freq, temp, return_log=req.return_log)
    except Exception as exc:
        log.exception(f"[{material}] inference error")
        raise HTTPException(status_code=422, detail=str(exc))

    return _build_response(material, preds, req.return_log, t0)


