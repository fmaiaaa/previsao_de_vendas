"""Pesos de amostra: recência + magnitude do alvo (reduz regressão à média em VGV alto)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weights(n: int, exp_scale: float = 2.45) -> np.ndarray:
    """
    Mais peso nas observações recentes: curva exponencial em t∈[0,1]
    (t=0 mais antigo, t=1 mais recente). exp_scale maior = contraste mais forte.
    """
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.ones(1)
    t = np.linspace(0.0, 1.0, n)
    w = np.exp(exp_scale * t)
    return (w * (n / w.sum())).astype(float)


def sample_weights(
    y: pd.Series,
    use_recency: bool,
    target_name: str,
    horizon: int | None = None,
) -> np.ndarray:
    yv = np.asarray(y, dtype=float).ravel()
    n = len(yv)
    base = np.ones(n)
    long_h = horizon is not None and int(horizon) >= 21
    if use_recency:
        base = recency_weights(n, exp_scale=2.75 if long_h else 2.45)
    if target_name == "valor":
        exp = 0.5 if long_h else 0.42
        hi = 9.0 if long_h else 6.0
        lo = 0.3 if long_h else 0.35
        scale = np.power(np.maximum(yv, np.percentile(yv, 15)), exp)
        scale = scale / (np.median(scale) + 1e-9)
        base = base * np.clip(scale, lo, hi)
    elif target_name == "qtd":
        exp = 0.62 if long_h else 0.55
        hi = 6.8 if long_h else 5.0
        scale = np.power(np.maximum(yv, 0.35), exp)
        scale = scale / (np.median(scale) + 1e-9)
        base = base * np.clip(scale, 0.38, hi)
    return (base * (n / (base.sum() + 1e-12))).astype(float)
