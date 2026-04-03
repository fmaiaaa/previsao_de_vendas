"""Benchmark de todos os candidatos + ensembles; escolha por MAE na validação (sem leakage no limiar)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

from .metric_suite import (
    binary_from_regression,
    regression_metrics,
    roc_curve_data,
)


def _clip_pos(arr: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap)


def run_model_benchmark(
    specs: list[tuple[str, Any, bool]],
    fit_fn: Callable[[Any, pd.DataFrame, pd.Series, bool], None],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Ajusta cada modelo em X_train, mede MAE em X_val; reajusta em X_fit para métricas de teste.
    Limiar binário = mediana(y_train) (só treino interno).
    """
    y_tr = np.asarray(y_train, dtype=float).ravel()
    thr = float(np.median(y_tr))
    cap = float(max(np.percentile(y_tr, 99.5) * 5.0, 1.0))

    preds_val: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}
    val_mae: dict[str, float] = {}
    spec_by_name: dict[str, tuple[Any, bool]] = {}

    it = specs
    if show_progress:
        it = tqdm(specs, desc="Benchmark modelos", leave=False, unit="m")

    for name, tpl, use_sw in it:
        spec_by_name[name] = (tpl, use_sw)
        try:
            p = clone(tpl)
            fit_fn(p, X_train, y_train, use_sw)
            pv = _clip_pos(np.asarray(p.predict(X_val), dtype=float).ravel(), cap)
            preds_val[name] = pv
            val_mae[name] = float(mean_absolute_error(np.asarray(y_val, dtype=float), pv))

            p2 = clone(tpl)
            fit_fn(p2, X_fit, y_fit, use_sw)
            pt = _clip_pos(np.asarray(p2.predict(X_test), dtype=float).ravel(), cap)
            preds_test[name] = pt
        except Exception:
            preds_val.pop(name, None)
            preds_test.pop(name, None)
            val_mae.pop(name, None)
            spec_by_name.pop(name, None)

    if not val_mae:
        return {
            "rows": [],
            "roc_traces": [],
            "winner_names": [],
            "winner_weights": [],
            "winner_label": "nenhum",
            "threshold_y": thr,
            "spec_by_name": {},
        }

    ranked = sorted(val_mae.items(), key=lambda x: x[1])
    names_ord = [n for n, _ in ranked]

    y_va = np.asarray(y_val, dtype=float).ravel()
    y_te = np.asarray(y_test, dtype=float).ravel()
    y_fi = np.asarray(y_fit, dtype=float).ravel()

    rows: list[dict[str, Any]] = []

    def add_synthetic_row(
        label: str,
        pv: np.ndarray,
        pt: np.ndarray,
    ) -> None:
        rv = regression_metrics(y_va, pv)
        rt = regression_metrics(y_te, pt)
        bv = binary_from_regression(y_va, pv, thr)
        bt = binary_from_regression(y_te, pt, thr)
        roc_t = roc_curve_data(y_te, pt, thr)
        rows.append(
            {
                "name": label,
                "mae_val": float(mean_absolute_error(y_va, pv)),
                "reg_val": rv,
                "reg_test": rt,
                "bin_val": bv,
                "bin_test": bt,
                "roc_test": roc_t,
            }
        )
        preds_val[label] = pv
        preds_test[label] = pt
        val_mae[label] = float(mean_absolute_error(y_va, pv))

    # Ensembles a partir dos melhores na validação
    top3 = names_ord[: min(3, len(names_ord))]
    top5 = names_ord[: min(5, len(names_ord))]
    top7 = names_ord[: min(7, len(names_ord))]

    if len(top3) >= 2:
        pv = np.mean([preds_val[n] for n in top3], axis=0)
        pt = np.mean([preds_test[n] for n in top3], axis=0)
        add_synthetic_row("Ens.Média-Top3", pv, pt)

    if len(top5) >= 3:
        pv = np.mean([preds_val[n] for n in top5], axis=0)
        pt = np.mean([preds_test[n] for n in top5], axis=0)
        add_synthetic_row("Ens.Média-Top5", pv, pt)

    if len(top5) >= 2:
        ws = np.array([1.0 / (val_mae[n] + 1e-8) for n in top5], dtype=float)
        ws = ws / (ws.sum() + 1e-12)
        pv = sum(ws[i] * preds_val[top5[i]] for i in range(len(top5)))
        pt = sum(ws[i] * preds_test[top5[i]] for i in range(len(top5)))
        add_synthetic_row("Ens.PesoInvMAE-Top5", pv, pt)

    if len(top7) >= 4:
        ws = np.array([1.0 / (val_mae[n] + 1e-8) for n in top7], dtype=float)
        ws = ws / (ws.sum() + 1e-12)
        pv = sum(ws[i] * preds_val[top7[i]] for i in range(len(top7)))
        pt = sum(ws[i] * preds_test[top7[i]] for i in range(len(top7)))
        add_synthetic_row("Ens.PesoInvMAE-Top7", pv, pt)

    if len(top7) >= 5:
        pv = np.mean([preds_val[n] for n in top7], axis=0)
        pt = np.mean([preds_test[n] for n in top7], axis=0)
        add_synthetic_row("Ens.Média-Top7", pv, pt)

    # Linhas dos modelos base
    for name in names_ord:
        if name not in preds_val or name not in preds_test:
            continue
        pv = preds_val[name]
        pt = preds_test[name]
        rv = regression_metrics(y_va, pv)
        rt = regression_metrics(y_te, pt)
        bv = binary_from_regression(y_va, pv, thr)
        bt = binary_from_regression(y_te, pt, thr)
        roc_t = roc_curve_data(y_te, pt, thr)
        rows.append(
            {
                "name": name,
                "mae_val": val_mae[name],
                "reg_val": rv,
                "reg_test": rt,
                "bin_val": bv,
                "bin_test": bt,
                "roc_test": roc_t,
            }
        )

    # Ordenar rows por mae_val
    rows.sort(key=lambda r: r["mae_val"])
    best = rows[0]
    best_name = str(best["name"])

    winner_names: list[str] = []
    winner_weights: list[float] = []

    if best_name.startswith("Ens."):
        if best_name == "Ens.Média-Top3":
            winner_names = top3
            winner_weights = [1.0 / len(top3)] * len(top3)
        elif best_name == "Ens.Média-Top5":
            winner_names = top5
            winner_weights = [1.0 / len(top5)] * len(top5)
        elif best_name == "Ens.PesoInvMAE-Top5":
            winner_names = top5
            ws = np.array([1.0 / (val_mae[n] + 1e-8) for n in top5], dtype=float)
            ws = ws / (ws.sum() + 1e-12)
            winner_weights = ws.tolist()
        elif best_name == "Ens.PesoInvMAE-Top7":
            winner_names = top7
            ws = np.array([1.0 / (val_mae[n] + 1e-8) for n in top7], dtype=float)
            ws = ws / (ws.sum() + 1e-12)
            winner_weights = ws.tolist()
        elif best_name == "Ens.Média-Top7":
            winner_names = top7
            winner_weights = [1.0 / len(top7)] * len(top7)
    else:
        winner_names = [best_name]
        winner_weights = [1.0]

    # Curvas ROC (teste) para todos os modelos com AUC finito
    roc_traces: list[dict[str, Any]] = []
    for r in rows:
        roc = r.get("roc_test") or {}
        auc = roc.get("auc")
        if isinstance(auc, float) and np.isfinite(auc) and roc.get("fpr"):
            roc_traces.append(
                {
                    "name": r["name"],
                    "fpr": roc["fpr"],
                    "tpr": roc["tpr"],
                    "auc": float(auc),
                }
            )

    return {
        "rows": rows,
        "roc_traces": roc_traces,
        "winner_names": winner_names,
        "winner_weights": winner_weights,
        "winner_label": best_name,
        "threshold_y": thr,
        "spec_by_name": spec_by_name,
    }
