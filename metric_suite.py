"""Métricas de regressão + tarefa binária auxiliar (alto/baixo vs mediana do treino) para ROC/F1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    out: dict[str, float] = {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "MedAE": float(median_absolute_error(yt, yp)),
        "ExplainedVar": float(explained_variance_score(yt, yp)),
        "MaxError": float(np.max(np.abs(yt - yp))),
    }
    return out


def binary_from_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_y: float,
) -> dict[str, float]:
    """
    y alto = valor real acima da mediana de referência (só do treino, sem leakage).
    Classe predita: predição acima do mesmo limiar em escala de y.
    Score para ROC: valor predito (ordenável).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    y_true_bin = (yt > threshold_y).astype(int)
    y_pred_bin = (yp > threshold_y).astype(int)
    out: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "Precision": float(
            precision_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "Recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "F1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }
    if len(np.unique(y_true_bin)) > 1:
        try:
            out["ROC_AUC"] = float(roc_auc_score(y_true_bin, yp))
        except ValueError:
            out["ROC_AUC"] = float("nan")
    else:
        out["ROC_AUC"] = float("nan")
    return out


def roc_curve_data(
    y_true: np.ndarray, y_pred: np.ndarray, threshold_y: float
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    y_true_bin = (yt > threshold_y).astype(int)
    if len(np.unique(y_true_bin)) < 2:
        return {"fpr": [], "tpr": [], "auc": float("nan")}
    fpr, tpr, _ = roc_curve(y_true_bin, yp)
    auc = float(roc_auc_score(y_true_bin, yp))
    return {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "auc": auc,
    }
