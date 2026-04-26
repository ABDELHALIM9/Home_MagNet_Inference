"""
infer_cnnv3.py — CNNv3 Inference Script
=========================================
Self-contained. No imports from src/ needed.

Pipeline
--------
  Input  ──►  Physics (B, H)  ──►  Normalize  ──►  CNNv3  ──►  Denormalize  ──►  W/m³

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT MODES  (auto-detected from --data extension)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  .mat          Raw MATLAB dataset (all samples or --indices subset)
  .json         Single sample OR list of samples  (see JSON FORMAT below)
  .csv          Table of samples — one row per sample  (see CSV FORMAT below)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single sample (dict):
  {
    "B":           [0.12, 0.15, ...],   ← waveform list, length T
    "H":           [42.1, 45.3, ...],   ← waveform list, length T
    "Frequency":   50000,
    "Temperature": 25.0
  }

  OR supply raw waveforms + meta to compute B/H internally:
  {
    "voltage":     [...],
    "current":     [...],
    "Frequency":   50000,
    "Temperature": 25.0,
    "meta": { "N_prim":22, "N_sec":4, "Ae":1.46e-4, "Le":0.1, "dt":2e-7 }
  }

Multiple samples (list of dicts):
  [
    { "B": [...], "H": [...], "Frequency": 50000, "Temperature": 25 },
    { "B": [...], "H": [...], "Frequency": 80000, "Temperature": 60 },
    ...
  ]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSV FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Required columns (B/H pre-computed):
    Frequency, Temperature,
    B_0  B_1  ...  B_{T-1},
    H_0  H_1  ...  H_{T-1}

  OR (raw waveforms — B/H computed internally):
    Frequency, Temperature, N_prim, N_sec, Ae, Le, dt,
    voltage_0 ... voltage_{T-1},
    current_0 ... current_{T-1}

OUTPUT FORMAT  (auto-detected from --output extension)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  .csv          sample_index, predicted_Loss_W_per_m3  [+ gt + error if --evaluate]
  .json         list of result objects  OR  single result object for single input

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # .mat → CSV
  python infer_cnnv3.py --checkpoint cnnv3_best.pth --stats stats.json \\
      --data data.mat --output predictions.csv

  # single JSON sample → single JSON result object
  python infer_cnnv3.py --checkpoint cnnv3_best.pth --stats stats.json \\
      --data sample.json --output result.json

  # list of JSON samples → JSON array of results
  python infer_cnnv3.py --checkpoint cnnv3_best.pth --stats stats.json \\
      --data samples.json --output results.json

  # CSV input → CSV output
  python infer_cnnv3.py --checkpoint cnnv3_best.pth --stats stats.json \\
      --data samples.csv --output predictions.csv

  # .mat with evaluation metrics vs physics ground truth
  python infer_cnnv3.py --checkpoint cnnv3_best.pth --stats stats.json \\
      --data data.mat --output predictions.csv --evaluate

  # subset of .mat samples → JSON
  python infer_cnnv3.py --checkpoint cnnv3_best.pth --stats stats.json \\
      --data data.mat --indices 0 1 2 100 --output predictions.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PYTHON API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  from infer_cnnv3 import CNNv3Inferencer

  inf = CNNv3Inferencer('cnnv3_best.pth', 'stats.json', 'config.yaml')

  # Single sample dict → scalar float W/m3
  pred = inf.predict_single({'B': [...], 'H': [...],
                              'Frequency': 50000, 'Temperature': 25})

  # List of sample dicts → (N,) ndarray W/m3
  preds = inf.predict_samples([{...}, {...}])

  # Pre-computed B/H arrays → (N,) ndarray
  preds = inf.predict_from_bh(B, H, frequency, temperature)

  # Raw voltage/current arrays → (N,) ndarray
  preds = inf.predict_from_raw(voltage, current, frequency, temperature, meta)

  # .mat file → (N,) ndarray
  preds = inf.predict_from_mat('data.mat')

  # JSON file → float or (N,) ndarray
  preds = inf.predict_from_json('samples.json')

  # CSV file → (N,) ndarray
  preds = inf.predict_from_csv('samples.csv')
"""

from __future__ import annotations

import os
import csv
import json
import argparse
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import cumulative_trapezoid, trapezoid


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL  (self-contained, no src/ imports)
# ══════════════════════════════════════════════════════════════════════════════

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class FiLM(nn.Module):
    def __init__(self, scalar_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(scalar_dim, num_channels * 2),
            nn.GELU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(scalars).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)


class ResConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.se  = SqueezeExcite(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.block(x) + x))


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels=(3, 7, 15)):
        super().__init__()
        branch_ch = out_channels // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_ch, k, padding=k // 2),
                nn.BatchNorm1d(branch_ch),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1)


class CNNv3Network(nn.Module):
    """
    CNNv3: MultiScaleConv + FiLM conditioning + residual conv blocks + MLP head.

    forward(B, H, scalars) → log10(Loss)
      B, H    : (batch, T, 1)
      scalars : (batch, 2)  — [normalized_freq, normalized_temp]
    """

    def __init__(
        self,
        input_dim:    int   = 3,
        num_channels: int   = 96,
        num_layers:   int   = 4,
        scalar_dim:   int   = 2,
        dropout:      float = 0.15,
        stats:        dict  = None,
    ):
        super().__init__()
        self.stats    = stats or {}
        kernels       = (3, 7, 15)
        branch_ch     = num_channels // len(kernels)
        actual_ch     = branch_ch * len(kernels)

        self.ms_conv    = MultiScaleConv(input_dim, num_channels, kernels=kernels)
        self.film1      = FiLM(scalar_dim, actual_ch)
        self.res_blocks = nn.ModuleList([
            ResConvBlock(actual_ch, kernel_size=7, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.film2 = FiLM(scalar_dim, actual_ch)

        head_in = actual_ch * 2 + scalar_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        B:       torch.Tensor,
        H:       torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        # Per-sample peak normalization (matches training behaviour)
        B  = B / (B.abs().max(dim=1, keepdim=True)[0] + 1e-6)
        H  = H / (H.abs().max(dim=1, keepdim=True)[0] + 1e-6)
        BH = B * H

        x = torch.cat([B, H, BH], dim=-1).permute(0, 2, 1)   # (batch, 3, T)
        x = self.ms_conv(x)
        x = self.film1(x, scalars)
        for block in self.res_blocks:
            x = block(x)
        x = self.film2(x, scalars)

        pooled = torch.cat([x.mean(-1), x.max(-1).values], dim=-1)
        return self.head(torch.cat([pooled, scalars], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PHYSICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _calc_B(voltage: np.ndarray, dt: float, N_sec: float, Ae: float) -> np.ndarray:
    """Flux density B, shape (N, T) float32."""
    v    = voltage - voltage.mean(axis=1, keepdims=True)
    flux = cumulative_trapezoid(v, axis=-1, initial=0) * dt
    B    = flux / (N_sec * Ae)
    B   -= B.mean(axis=1, keepdims=True)
    return B.astype(np.float32)


def _calc_H(current: np.ndarray, N_prim: float, Le: float) -> np.ndarray:
    """Magnetizing force H, shape (N, T) float32."""
    return (N_prim * current / Le).astype(np.float32)


def _calc_Loss(B: np.ndarray, H: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Volumetric power loss, shape (N,) float32."""
    energy = np.abs(trapezoid(y=H, x=B, axis=-1))
    return (energy * freq).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(data: np.ndarray, method: str, stats: dict) -> np.ndarray:
    data = data.astype(np.float32)
    if method == 'none':
        return data
    if method == 'log10':
        return np.log10(np.abs(data) + 1e-6).astype(np.float32)
    if method == 'standard':
        return ((data - stats['mean']) / stats['std']).astype(np.float32)
    if method == 'minmax':
        denom = stats['max'] - stats['min']
        if denom == 0:
            denom = 1.0
        return ((data - stats['min']) / denom).astype(np.float32)
    raise ValueError(f"Unknown normalization method: '{method}'")


def _denormalize_loss(pred: np.ndarray, method: str, stats: dict) -> np.ndarray:
    pred = pred.astype(np.float64)
    if method == 'log10':
        return (10.0 ** pred).astype(np.float32)
    if method == 'standard':
        return (pred * stats['std'] + stats['mean']).astype(np.float32)
    return pred.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — .MAT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_mat(file_path: str) -> dict:
    """Load MagNet .mat file. Handles scipy (v7) and h5py (v7.3 HDF5)."""
    import scipy.io
    import h5py

    def _scipy():
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        D   = mat['Data']
        return {
            'voltage': D.Voltage.astype(np.float32),
            'current': D.Current.astype(np.float32),
            'freq':    D.Frequency_command.astype(np.float32),
            'temp':    D.Temperature_command.astype(np.float32),
            'meta': {
                'N_prim': float(D.Primary_Turns),
                'N_sec':  float(D.Secondary_Turns),
                'Ae':     float(D.Effective_Area),
                'Le':     float(D.Effective_Length),
                'dt':     float(np.array(D.Sampling_Time).flat[0]),
            },
        }

    def _h5():
        with h5py.File(file_path, 'r') as f:
            D   = f['Data']
            arr = lambda k: np.array(D[k]).flatten().astype(np.float32)
            return {
                'voltage': np.array(D['Voltage']).T.astype(np.float32),
                'current': np.array(D['Current']).T.astype(np.float32),
                'freq':    arr('Frequency_command'),
                'temp':    arr('Temperature_command'),
                'meta': {
                    'N_prim': float(np.array(D['Primary_Turns']).item()),
                    'N_sec':  float(np.array(D['Secondary_Turns']).item()),
                    'Ae':     float(np.array(D['Effective_Area']).item()),
                    'Le':     float(np.array(D['Effective_Length']).item()),
                    'dt':     float(np.array(D['Sampling_Time']).flat[0]),
                },
            }

    try:
        return _scipy()
    except NotImplementedError:
        print("  [Loader] scipy failed — falling back to h5py ...")
        return _h5()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — JSON / CSV PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_sample_dict(d: dict) -> dict:
    """
    Parse one input dict into internal arrays.

    Accepts two flavours:

    A) Pre-computed B/H:
       { "B": [...], "H": [...], "Frequency": float, "Temperature": float }

    B) Raw voltage/current:
       { "voltage": [...], "current": [...],
         "Frequency": float, "Temperature": float,
         "meta": {"N_prim":..., "N_sec":..., "Ae":..., "Le":..., "dt":...} }

    Returns dict: { B:(1,T), H:(1,T), freq:(1,), temp:(1,) }
    """
    freq = float(d['Frequency'])
    temp = float(d['Temperature'])

    if 'B' in d and 'H' in d:
        B = np.array(d['B'], dtype=np.float32).reshape(1, -1)
        H = np.array(d['H'], dtype=np.float32).reshape(1, -1)

    elif 'voltage' in d and 'current' in d:
        if 'meta' not in d:
            raise KeyError(
                "JSON sample with 'voltage'/'current' fields must also include "
                "'meta': {N_prim, N_sec, Ae, Le, dt}"
            )
        meta    = d['meta']
        voltage = np.array(d['voltage'], dtype=np.float32).reshape(1, -1)
        current = np.array(d['current'], dtype=np.float32).reshape(1, -1)
        B = _calc_B(voltage, float(meta['dt']),
                    float(meta['N_sec']), float(meta['Ae']))
        H = _calc_H(current, float(meta['N_prim']), float(meta['Le']))

    else:
        raise KeyError(
            "Each sample dict must have either:\n"
            "  'B' + 'H'                          (pre-computed waveforms)\n"
            "  'voltage' + 'current' + 'meta'      (raw waveforms)"
        )

    return {
        'B':    B,
        'H':    H,
        'freq': np.array([freq], dtype=np.float32),
        'temp': np.array([temp], dtype=np.float32),
    }


def _load_json_input(path: str):
    """
    Load a JSON input file.

    Returns
    -------
    samples   : list of parsed sample dicts
    is_single : True if the root JSON was a plain dict (not a list)
    """
    with open(path, 'r') as f:
        raw = json.load(f)

    is_single = isinstance(raw, dict)
    items     = [raw] if is_single else list(raw)

    if not items:
        raise ValueError(f"JSON file is empty: {path}")

    samples = [_parse_sample_dict(item) for item in items]
    return samples, is_single


def _load_csv_input(path: str) -> list:
    """
    Load a CSV input file. Returns list of parsed sample dicts.

    Expected column naming
    ----------------------
    Waveform columns  : B_0, B_1, ..., B_{T-1}   and  H_0, H_1, ...
                        OR  voltage_0, ..., current_0, ...
    Scalar columns    : Frequency, Temperature
    Meta cols (opt.)  : N_prim, N_sec, Ae, Le, dt  (needed with voltage/current)
    """
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    cols        = list(rows[0].keys())
    has_B       = any(c.startswith('B_')       for c in cols)
    has_H       = any(c.startswith('H_')       for c in cols)
    has_voltage = any(c.startswith('voltage_') for c in cols)
    has_current = any(c.startswith('current_') for c in cols)

    if not ((has_B and has_H) or (has_voltage and has_current)):
        raise ValueError(
            "CSV must have either:\n"
            "  B_0…B_T  and  H_0…H_T   (pre-computed waveforms)\n"
            "  voltage_0…  and  current_0…  plus N_prim,N_sec,Ae,Le,dt"
        )

    def extract_waveform(row: dict, prefix: str) -> np.ndarray:
        wave_cols = sorted(
            [c for c in row if c.startswith(prefix + '_')],
            key=lambda c: int(c.split('_', 1)[-1])
        )
        if not wave_cols:
            raise KeyError(f"No CSV columns with prefix '{prefix}_'")
        return np.array([float(row[c]) for c in wave_cols], dtype=np.float32)

    samples = []
    for row_idx, row in enumerate(rows):
        freq = float(row['Frequency'])
        temp = float(row['Temperature'])

        if has_B and has_H:
            B = extract_waveform(row, 'B').reshape(1, -1)
            H = extract_waveform(row, 'H').reshape(1, -1)
        else:
            voltage = extract_waveform(row, 'voltage').reshape(1, -1)
            current = extract_waveform(row, 'current').reshape(1, -1)
            for meta_key in ('N_prim', 'N_sec', 'Ae', 'Le', 'dt'):
                if meta_key not in row:
                    raise KeyError(
                        f"CSV row {row_idx}: missing meta column '{meta_key}'. "
                        f"Required when using voltage/current columns."
                    )
            B = _calc_B(voltage, float(row['dt']),
                        float(row['N_sec']), float(row['Ae']))
            H = _calc_H(current, float(row['N_prim']), float(row['Le']))

        samples.append({
            'B':    B,
            'H':    H,
            'freq': np.array([freq], dtype=np.float32),
            'temp': np.array([temp], dtype=np.float32),
        })

    return samples


def _stack_samples(samples: list) -> tuple:
    """Stack a list of per-sample dicts → batch arrays (N,T), (N,T), (N,), (N,)."""
    B    = np.concatenate([s['B']    for s in samples], axis=0)
    H    = np.concatenate([s['H']    for s in samples], axis=0)
    freq = np.concatenate([s['freq'] for s in samples], axis=0)
    temp = np.concatenate([s['temp'] for s in samples], axis=0)
    return B, H, freq, temp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OUTPUT WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _write_csv_output(
    path:         str,
    predictions:  np.ndarray,
    ground_truth: np.ndarray = None,
    label:        str        = 'Loss_W_per_m3',
):
    """Write predictions (+ optional ground truth) to CSV."""
    _ensure_dir(path)
    eps = 1e-8
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        if ground_truth is not None:
            w.writerow(['sample_index', f'predicted_{label}',
                        f'gt_{label}', 'abs_relative_error'])
            rel = np.abs((predictions - ground_truth) /
                         (np.abs(ground_truth) + eps))
            for i, (p, g, r) in enumerate(zip(predictions, ground_truth, rel)):
                w.writerow([i, float(p), float(g), float(r)])
        else:
            w.writerow(['sample_index', f'predicted_{label}'])
            for i, p in enumerate(predictions):
                w.writerow([i, float(p)])


def _write_json_output(
    path:         str,
    predictions:  np.ndarray,
    ground_truth: np.ndarray = None,
    is_single:    bool       = False,
    label:        str        = 'Loss_W_per_m3',
):
    """Write predictions to JSON. Single-input → plain object, batch → list."""
    _ensure_dir(path)
    eps = 1e-8

    def make_record(i, p, g=None):
        rec = {'sample_index': i, f'predicted_{label}': float(p)}
        if g is not None:
            rec[f'gt_{label}']        = float(g)
            rec['abs_relative_error'] = float(abs(p - g) / (abs(g) + eps))
        return rec

    if ground_truth is not None:
        records = [make_record(i, p, g)
                   for i, (p, g) in enumerate(zip(predictions, ground_truth))]
    else:
        records = [make_record(i, p) for i, p in enumerate(predictions)]

    # A single-sample JSON input → return a plain object (not a one-element list)
    output = records[0] if (is_single and len(records) == 1) else records

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — INFERENCER
# ══════════════════════════════════════════════════════════════════════════════

class CNNv3Inferencer:
    """
    Full inference pipeline for CNNv3.

    Parameters
    ----------
    checkpoint_path : str
    stats_path      : str   stats.json produced by prepare_datasets.py
    config_path     : str   config.yaml (optional; uses defaults if absent)
    device          : 'auto' | 'cpu' | 'cuda'
    batch_size      : int   GPU mini-batch size (default 256)
    """

    _DEFAULT_INPUTS  = {'B': 'minmax', 'H': 'standard',
                        'Frequency': 'standard', 'Temperature': 'standard'}
    _DEFAULT_TARGETS = {'Loss': 'log10'}

    def __init__(
        self,
        checkpoint_path: str,
        stats_path:      str,
        config_path:     str = None,
        device:          str = 'auto',
        batch_size:      int = 256,
    ):
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size

        # Config (optional)
        self._cfg = self._load_config(config_path) if config_path else {}

        # Feature norm maps — config overrides defaults
        self.input_features  = dict(self._DEFAULT_INPUTS)
        self.target_features = dict(self._DEFAULT_TARGETS)
        if 'models' in self._cfg and 'cnnv3' in self._cfg['models']:
            feat = self._cfg['models']['cnnv3'].get('features', {})
            if feat.get('inputs'):
                self.input_features  = dict(feat['inputs'])
            if feat.get('targets'):
                self.target_features = dict(feat['targets'])

        # Normalization stats
        self.stats = self._load_stats(stats_path)

        # Build + load model weights
        self.model = self._build_model(checkpoint_path)
        self.model.eval()

        # print(f"\n[CNNv3Inferencer]")
        # print(f"  checkpoint : {os.path.basename(checkpoint_path)}")
        # print(f"  device     : {self.device}")
        # print(f"  batch_size : {self.batch_size}")
        # print(f"  inputs     : {self.input_features}")
        # print(f"  targets    : {self.target_features}")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> dict:
        import yaml
        if not path or not os.path.exists(path):
            print(f"  [config] '{path}' not found — using defaults.")
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_stats(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"stats.json not found: '{path}'\n"
                f"Run prepare_datasets.py first to generate it."
            )
        with open(path, 'r') as f:
            return json.load(f)

    def _build_model(self, ckpt_path: str) -> CNNv3Network:
        model_cfg = {}
        if 'models' in self._cfg and 'cnnv3' in self._cfg['models']:
            model_cfg = self._cfg['models']['cnnv3']

        model = CNNv3Network(
            input_dim    = 3,
            num_channels = int(model_cfg.get('num_channels', 96)),
            num_layers   = int(model_cfg.get('num_layers',    4)),
            scalar_dim   = int(model_cfg.get('scalar_dim',    2)),
            dropout      = float(model_cfg.get('dropout',  0.15)),
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")

        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if key in state:
                    state = state[key]
                    break

        model.load_state_dict(state)
        model.to(self.device)
        return model

    def _norm(self, data: np.ndarray, name: str) -> np.ndarray:
        method = self.input_features.get(name, 'none')
        return _normalize(data, method, self.stats.get(name, {}))

    def _denorm_loss(self, pred: np.ndarray) -> np.ndarray:
        method = self.target_features.get('Loss', 'log10')
        return _denormalize_loss(pred, method, self.stats.get('Loss', {}))

    def _run_batched(
        self,
        B_norm:    np.ndarray,
        H_norm:    np.ndarray,
        freq_norm: np.ndarray,
        temp_norm: np.ndarray,
    ) -> np.ndarray:
        """Run CNNv3 forward pass in mini-batches. Returns log10-space output."""
        N, all_preds = B_norm.shape[0], []
        with torch.no_grad():
            for start in range(0, N, self.batch_size):
                end  = min(start + self.batch_size, N)
                B_t  = torch.tensor(
                    B_norm[start:end, :, np.newaxis], dtype=torch.float32
                ).to(self.device)
                H_t  = torch.tensor(
                    H_norm[start:end, :, np.newaxis], dtype=torch.float32
                ).to(self.device)
                sc_t = torch.tensor(
                    np.stack([freq_norm[start:end],
                               temp_norm[start:end]], axis=1),
                    dtype=torch.float32,
                ).to(self.device)
                out = self.model(B_t, H_t, sc_t)
                all_preds.append(out.cpu().numpy().flatten())
        return np.concatenate(all_preds, axis=0)

    # ── Shared normalization + inference kernel ───────────────────────────────

    def _infer_bh(
        self,
        B:          np.ndarray,
        H:          np.ndarray,
        frequency:  np.ndarray,
        temperature:np.ndarray,
        return_log: bool,
    ) -> np.ndarray:
        B_n    = self._norm(B,                       'B')
        H_n    = self._norm(H,                       'H')
        freq_n = self._norm(frequency.reshape(-1,1), 'Frequency').flatten()
        temp_n = self._norm(temperature.reshape(-1,1),'Temperature').flatten()
        raw    = self._run_batched(B_n, H_n, freq_n, temp_n)
        return raw if return_log else self._denorm_loss(raw)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def predict_single(self, sample: dict, return_log: bool = False) -> float:
        """
        Predict for one sample supplied as a Python dict.

        Parameters
        ----------
        sample : dict
            Pre-computed  → {'B':[...], 'H':[...], 'Frequency':f, 'Temperature':t}
            Raw waveforms → {'voltage':[...], 'current':[...],
                             'Frequency':f, 'Temperature':t,
                             'meta':{N_prim,N_sec,Ae,Le,dt}}

        Returns float  W/m³  (or log10 if return_log=True)
        """
        parsed = _parse_sample_dict(sample)
        B, H, freq, temp = _stack_samples([parsed])
        preds = self._infer_bh(B, H, freq, temp, return_log)
        return float(preds[0])

    def predict_samples(
        self,
        samples:    List[dict],
        return_log: bool = False,
    ) -> np.ndarray:
        """
        Predict for a list of sample dicts.
        Returns (N,) float32 array  W/m³.
        """
        parsed = [_parse_sample_dict(s) for s in samples]
        B, H, freq, temp = _stack_samples(parsed)
        return self._infer_bh(B, H, freq, temp, return_log)

    def predict_from_bh(
        self,
        B:           np.ndarray,
        H:           np.ndarray,
        frequency:   np.ndarray,
        temperature: np.ndarray,
        return_log:  bool = False,
    ) -> np.ndarray:
        """
        Predict from pre-computed B/H waveforms.
        B, H : (N, T) float32.  frequency, temperature : (N,).
        Returns (N,) float32  W/m³.
        """
        return self._infer_bh(B, H, frequency, temperature, return_log)

    def predict_from_raw(
        self,
        voltage:     np.ndarray,
        current:     np.ndarray,
        frequency:   np.ndarray,
        temperature: np.ndarray,
        meta:        dict,
        return_log:  bool = False,
    ) -> np.ndarray:
        """
        Predict from raw voltage / current waveforms.
        meta must have: N_prim, N_sec, Ae, Le, dt.
        Returns (N,) float32  W/m³.
        """
        dt = float(np.array(meta['dt']).flat[0])
        B  = _calc_B(voltage, dt, float(meta['N_sec']), float(meta['Ae']))
        H  = _calc_H(current, float(meta['N_prim']), float(meta['Le']))
        return self._infer_bh(B, H, frequency, temperature, return_log)

    def predict_from_mat(
        self,
        mat_path:   str,
        indices:    np.ndarray = None,
        return_log: bool       = False,
    ) -> np.ndarray:
        """
        Predict for all (or selected) samples in a .mat file.
        Returns (N,) float32  W/m³.
        """
        print(f"  Loading .mat: {mat_path}")
        raw = _load_mat(mat_path)

        voltage = raw['voltage']
        current = raw['current']
        freq    = raw['freq']
        temp    = raw['temp']
        meta    = raw['meta']

        if indices is not None:
            idx     = np.asarray(indices)
            voltage = voltage[idx]
            current = current[idx]
            freq    = freq[idx]
            temp    = temp[idx]

        print(f"  {voltage.shape[0]:,} samples  |  T={voltage.shape[1]}")
        return self.predict_from_raw(voltage, current, freq, temp, meta, return_log)

    def predict_from_json(
        self,
        json_path:  str,
        return_log: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Predict from a JSON file.

        Returns
        -------
        float       — if the file was a single sample dict
        np.ndarray  — if the file was a list of sample dicts
        """
        print(f"  Loading JSON: {json_path}")
        samples, is_single = _load_json_input(json_path)
        print(f"  {len(samples):,} sample(s)")

        B, H, freq, temp = _stack_samples(samples)
        preds = self._infer_bh(B, H, freq, temp, return_log)

        return float(preds[0]) if is_single else preds

    def predict_from_csv(
        self,
        csv_path:   str,
        return_log: bool = False,
    ) -> np.ndarray:
        """
        Predict from a CSV input file.
        Returns (N,) float32  W/m³.
        """
        print(f"  Loading CSV: {csv_path}")
        samples = _load_csv_input(csv_path)
        print(f"  {len(samples):,} sample(s)")

        B, H, freq, temp = _stack_samples(samples)
        return self._infer_bh(B, H, freq, temp, return_log)

    def evaluate(
        self,
        mat_path: str,
        indices:  np.ndarray = None,
    ) -> dict:
        """
        Predict AND compare to physics ground truth from a .mat file.

        Returns
        -------
        dict with keys:
          predictions, ground_truth — (N,) float32  W/m³
          mse, mae, rmse, p95_relative_error, max_relative_error
        """
        print(f"  Loading .mat for evaluation: {mat_path}")
        raw = _load_mat(mat_path)

        voltage = raw['voltage']
        current = raw['current']
        freq    = raw['freq']
        temp    = raw['temp']
        meta    = raw['meta']

        if indices is not None:
            idx     = np.asarray(indices)
            voltage = voltage[idx]
            current = current[idx]
            freq    = freq[idx]
            temp    = temp[idx]

        dt      = float(np.array(meta['dt']).flat[0])
        B       = _calc_B(voltage, dt, float(meta['N_sec']), float(meta['Ae']))
        H       = _calc_H(current, float(meta['N_prim']), float(meta['Le']))
        loss_gt = _calc_Loss(B, H, freq)

        predictions = self._infer_bh(B, H, freq, temp, return_log=False)

        eps    = 1e-8
        rel    = np.abs((predictions - loss_gt) / (np.abs(loss_gt) + eps))
        mse    = float(np.mean((predictions - loss_gt) ** 2))
        mae    = float(np.mean(np.abs(predictions - loss_gt)))
        rmse   = float(np.sqrt(mse))
        p95_re = float(np.percentile(rel, 95))
        max_re = float(np.max(rel))

        print(
            f"\n  ── Evaluation ────────────────────────────────\n"
            f"  Samples           : {len(predictions):,}\n"
            f"  MSE               : {mse:.4e}\n"
            f"  MAE               : {mae:.4e}\n"
            f"  RMSE              : {rmse:.4e}\n"
            f"  P95 Relative Error: {p95_re:.4f}\n"
            f"  Max Relative Error: {max_re:.4f}\n"
            f"  ─────────────────────────────────────────────"
        )

        return dict(
            predictions        = predictions,
            ground_truth       = loss_gt,
            mse                = mse,
            mae                = mae,
            rmse               = rmse,
            p95_relative_error = p95_re,
            max_relative_error = max_re,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — AUTO-ROUTE: detect input/output format, run, save
# ══════════════════════════════════════════════════════════════════════════════

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _run_inference(
    inferencer:  CNNv3Inferencer,
    data_path:   str,
    output_path: str,
    indices:     list = None,
    evaluate:    bool = False,
    return_log:  bool = False,
):
    """
    Detect input extension → call the right predict_* method.
    Detect output extension → call the right writer.

    Input:  .mat | .json | .csv
    Output: .csv | .json
    """
    idx_arr   = np.array(indices) if indices else None
    label     = 'log10_Loss' if return_log else 'Loss_W_per_m3'
    in_ext    = _ext(data_path)
    out_ext   = _ext(output_path)
    is_single = False
    gt        = None

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    if in_ext == '.mat':
        if evaluate:
            results = inferencer.evaluate(data_path, indices=idx_arr)
            preds   = results['predictions']
            gt      = results['ground_truth']
        else:
            preds = inferencer.predict_from_mat(
                data_path, indices=idx_arr, return_log=return_log
            )

    elif in_ext == '.json':
        if evaluate:
            print("  [WARNING] --evaluate needs a .mat file (physics GT requires "
                  "raw voltage/current). Evaluation skipped.")
        samples, is_single = _load_json_input(data_path)
        n = len(samples)
        print(f"  {n} sample{'s' if n != 1 else ''} loaded from JSON")
        B, H, freq, temp = _stack_samples(samples)
        preds = inferencer.predict_from_bh(B, H, freq, temp, return_log)

    elif in_ext == '.csv':
        if evaluate:
            print("  [WARNING] --evaluate needs a .mat file. Evaluation skipped.")
        samples = _load_csv_input(data_path)
        n = len(samples)
        print(f"  {n} sample{'s' if n != 1 else ''} loaded from CSV")
        B, H, freq, temp = _stack_samples(samples)
        preds = inferencer.predict_from_bh(B, H, freq, temp, return_log)

    else:
        raise ValueError(
            f"Unsupported input format: '{in_ext}'\n"
            f"Supported: .mat  .json  .csv"
        )

    # ── SAVE OUTPUT ───────────────────────────────────────────────────────────
    if out_ext == '.csv':
        _write_csv_output(output_path, preds, gt, label)
    elif out_ext == '.json':
        _write_json_output(output_path, preds, gt, is_single, label)
    else:
        raise ValueError(
            f"Unsupported output format: '{out_ext}'\n"
            f"Supported: .csv  .json"
        )

    print(f"\n  {len(preds):,} prediction(s) saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "CNNv3 Inference  —  single / batch  ·  .mat / .json / .csv  →  .csv / .json\n"
            "\n"
            "Input formats\n"
            "  .mat   MATLAB dataset (all samples or --indices subset)\n"
            "  .json  Single sample dict  OR  list of sample dicts\n"
            "  .csv   Table with B_0…B_T, H_0…H_T (or voltage_ / current_) columns\n"
            "\n"
            "Output formats  (auto-detected from --output extension)\n"
            "  .csv   sample_index, predicted_Loss_W_per_m3  [+ gt + rel_error if --evaluate]\n"
            "  .json  result object  OR  list of result objects\n"
            "\n"
            "Examples\n"
            "  # .mat → CSV\n"
            "  python infer_cnnv3.py --checkpoint ckpt.pth --stats stats.json \\\n"
            "      --data data.mat --output preds.csv\n"
            "\n"
            "  # single JSON → single JSON result object\n"
            "  python infer_cnnv3.py --checkpoint ckpt.pth --stats stats.json \\\n"
            "      --data sample.json --output result.json\n"
            "\n"
            "  # batch JSON → JSON list of results\n"
            "  python infer_cnnv3.py --checkpoint ckpt.pth --stats stats.json \\\n"
            "      --data samples.json --output results.json\n"
            "\n"
            "  # CSV → CSV\n"
            "  python infer_cnnv3.py --checkpoint ckpt.pth --stats stats.json \\\n"
            "      --data samples.csv --output preds.csv\n"
            "\n"
            "  # .mat with evaluation metrics\n"
            "  python infer_cnnv3.py --checkpoint ckpt.pth --stats stats.json \\\n"
            "      --data data.mat --output preds.csv --evaluate\n"
            "\n"
            "  # subset of .mat indices → JSON\n"
            "  python infer_cnnv3.py --checkpoint ckpt.pth --stats stats.json \\\n"
            "      --data data.mat --indices 0 1 5 100 --output preds.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to cnnv3_best.pth checkpoint')
    p.add_argument('--stats',      required=True,
                   help='Path to stats.json  (from prepare_datasets.py)')
    p.add_argument('--data',       required=True,
                   help='Input: .mat | .json | .csv')
    p.add_argument('--output',     default='predictions.csv',
                   help='Output: .csv | .json  (default: predictions.csv)')
    p.add_argument('--config',     default=None,
                   help='Path to config.yaml  (optional)')
    p.add_argument('--indices',    nargs='+', type=int, default=None,
                   help='Sample indices to run  (.mat only; default: all)')
    p.add_argument('--batch-size', type=int, default=256,
                   help='Forward-pass mini-batch size  (default: 256)')
    p.add_argument('--device',     default='auto',
                   choices=['auto', 'cpu', 'cuda'],
                   help='Compute device  (default: auto)')
    p.add_argument('--evaluate',   action='store_true',
                   help='Compare to physics ground truth  (.mat only)')
    p.add_argument('--return-log', action='store_true',
                   help='Output log10(Loss) instead of W/m³')
    return p


def main():
    args = _build_parser().parse_args()

    print("\n" + "═" * 64)
    print("  CNNv3 Inference Pipeline")
    print("═" * 64)

    inferencer = CNNv3Inferencer(
        checkpoint_path = args.checkpoint,
        stats_path      = args.stats,
        config_path     = args.config,
        device          = args.device,
        batch_size      = args.batch_size,
    )

    _run_inference(
        inferencer  = inferencer,
        data_path   = args.data,
        output_path = args.output,
        indices     = args.indices,
        evaluate    = args.evaluate,
        return_log  = args.return_log,
    )

   


if __name__ == '__main__':
    main()