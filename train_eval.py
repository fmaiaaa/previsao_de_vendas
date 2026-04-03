"""Split 80/20, Optuna (TSS) com trials adaptativos, super-ensemble e zoo de modelos."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .ts_components import build_median_quantile_blend_pipe, build_ts_linear_bridge_pipe
from .weights import sample_weights

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None  # type: ignore[misc, assignment]

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore[misc, assignment]

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None  # type: ignore[misc, assignment]

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except ImportError:
    NGBRegressor = None  # type: ignore[misc, assignment]
    Normal = None  # type: ignore[misc, assignment]


@dataclass
class TrainResult:
    pipeline: Any
    metrics_val: dict[str, float]
    metrics_test: dict[str, float]
    importance: pd.Series
    y_val: pd.Series
    y_val_pred: np.ndarray
    y_test: pd.Series
    y_test_pred: np.ndarray
    best_params: dict[str, Any]
    dates_val: pd.Index
    dates_test: pd.Index
    model_label: str
    full_period_train: bool = False
    chart_dates: list[str] = field(default_factory=list)
    chart_y_real: list[float] = field(default_factory=list)
    chart_y_pred: list[float] = field(default_factory=list)
    chart_split_index: int = 0
    benchmark_appendix: dict[str, Any] | None = None


class _FittedBlend:
    """Combinação linear de pipelines já ajustados (pesos normalizados)."""

    def __init__(self, fitted_pipes: list[Any], weights: np.ndarray) -> None:
        self.fitted_pipes = fitted_pipes
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / (self.weights.sum() + 1e-12)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        P = np.column_stack([p.predict(X) for p in self.fitted_pipes])
        out = P @ self.weights
        return np.asarray(out, dtype=float).ravel()


def _make_fitted_blend(
    templates: list[Any],
    weights: np.ndarray,
    sw_flags: list[bool],
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    horizon: int = 7,
) -> _FittedBlend:
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    fitted: list[Any] = []
    for tpl, sw in zip(templates, sw_flags):
        p = clone(tpl)
        _safe_fit_pipeline(p, X, y, sw, target_name, horizon)
        fitted.append(p)
    return _FittedBlend(fitted, w)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-6
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))


def _mape_nz(y_true: np.ndarray, y_pred: np.ndarray, min_y: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = y_true >= min_y
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    return float(np.mean(np.abs(yt - yp) / np.maximum(yt, min_y)) * 100.0)


def _metrics_dict(
    y_true: np.ndarray, y_pred: np.ndarray, target_name: str
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[str, float] = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "sMAPE": _smape(y_true, y_pred),
    }
    if target_name == "qtd":
        out["MAPE"] = _mape_nz(y_true, y_pred, min_y=0.5)
    else:
        pos = y_true[y_true > 0]
        p15 = float(np.percentile(pos, 15)) if len(pos) >= 5 else 1e5
        floor = max(50_000.0, p15)
        out["MAPE"] = _mape_nz(y_true, y_pred, min_y=floor)
    return out


def temporal_train_test_indices(n: int, train_frac: float = 0.8) -> tuple[slice, slice]:
    """Primeiros train_frac para treino; resto para teste out-of-sample."""
    i_tr = max(1, int(n * train_frac))
    if i_tr >= n:
        raise ValueError(
            "Série muito curta para 80%/20%. Aumente o histórico diário ou reduza o horizonte."
        )
    return slice(0, i_tr), slice(i_tr, n)


def _ensemble_selection_splits(n: int) -> tuple[slice, slice]:
    """
    Primeiros (~92%) para ajustar candidatos; últimos (~8%) só para MAE e escolha do blend.
    Em modo 80/20, n é o comprimento só do bloco de treino.
    """
    if n < 15:
        n_va = max(1, min(3, n // 5))
    else:
        n_va = max(7, int(n * 0.08))
    n_va = min(n_va, max(1, n - max(10, n // 2)))
    i_tr = n - n_va
    if i_tr < 1:
        i_tr, n_va = n - 1, 1
    return slice(0, i_tr), slice(i_tr, n)


def _safe_fit_pipeline(
    pipe: Any,
    X: pd.DataFrame,
    y: pd.Series,
    use_sw: bool,
    target_name: str,
    horizon: int = 7,
) -> None:
    sw = sample_weights(y, True, target_name, horizon) if use_sw else None
    if sw is None:
        pipe.fit(X, y)
        return
    try:
        pipe.fit(X, y, model__sample_weight=sw)
    except TypeError:
        pipe.fit(X, y)


def _optuna_objective_lgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    random_state: int,
    target_name: str,
    horizon: int = 7,
) -> float:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm não instalado")

    is_qtd = target_name == "qtd"
    long_h = int(horizon) >= 21
    if long_h:
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 160, 520),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.032, 0.16, log=True),
            "subsample": trial.suggest_float("subsample", 0.58, 0.92),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.58, 0.92),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 12.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.03, 22.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 22, 84),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 16 if is_qtd else 10, 95 if is_qtd else 72
            ),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0008, 0.95, log=True),
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        pen_std = 0.055
    else:
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 90, 420),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.035, 0.14, log=True),
            "subsample": trial.suggest_float("subsample", 0.62, 0.92),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.62, 0.92),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.02, 18.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 28.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 56),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 28 if is_qtd else 18, 130 if is_qtd else 95
            ),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.002, 1.2, log=True),
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        pen_std = 0.10
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for tr_idx, va_idx in tscv.split(X_train):
        model = LGBMRegressor(**param)
        y_tr = y_train.iloc[tr_idx]
        sw = sample_weights(y_tr, True, target_name, horizon)
        model.fit(X_train.iloc[tr_idx], y_tr, sample_weight=sw)
        p = model.predict(X_train.iloc[va_idx])
        scores.append(mean_absolute_error(y_train.iloc[va_idx], p))
    mean_mae = float(np.mean(scores))
    if len(scores) > 1:
        mean_mae += pen_std * float(np.std(scores))
    return mean_mae


def _wrap_log(reg: Any) -> Any:
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)


def _mm() -> MinMaxScaler:
    # clip=True: valores fora do min/max vistos no fit ficam em [0,1].
    # Sem isso, Ridge/ElasticNet explodem no teste quando há deriva de distribuição.
    return MinMaxScaler(feature_range=(0, 1), clip=True)


def build_ridge_pipe(rs: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", _mm()),
            ("model", Ridge(alpha=24.0, random_state=rs)),
        ]
    )


def build_elastic_net_pipe(rs: int, target_name: str) -> Pipeline:
    a = 0.14 if target_name == "qtd" else 72000.0
    return Pipeline(
        [
            ("scaler", _mm()),
            (
                "model",
                ElasticNet(
                    alpha=a,
                    l1_ratio=0.48 if target_name == "qtd" else 0.42,
                    random_state=rs,
                    max_iter=8000,
                ),
            ),
        ]
    )


def build_random_forest_pipe(rs: int, horizon: int) -> Pipeline:
    long_h = int(horizon) >= 21
    return Pipeline(
        [
            ("scaler", _mm()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=450 if long_h else 340,
                    max_depth=18 if long_h else 14,
                    min_samples_leaf=3,
                    random_state=rs,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_lgbm_only_pipe(best_params: dict[str, Any], use_log: bool, rs: int) -> Pipeline:
    lgb = LGBMRegressor(
        **{**best_params, "random_state": rs, "verbosity": -1, "n_jobs": -1}
    )
    inner: Any = _wrap_log(lgb) if use_log else lgb
    return Pipeline([("scaler", _mm()), ("model", inner)])


def build_lgbm_fair_pipe(best_params: dict[str, Any], rs: int) -> Pipeline:
    p = {
        **best_params,
        "objective": "fair",
        "fair_c": 52.0,
        "random_state": rs,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return Pipeline([("scaler", _mm()), ("model", LGBMRegressor(**p))])


def build_lgbm_quantile_pipe(
    best_params: dict[str, Any], alpha: float, rs: int
) -> Pipeline:
    p = {
        **best_params,
        "objective": "quantile",
        "alpha": float(alpha),
        "random_state": rs,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return Pipeline([("scaler", _mm()), ("model", LGBMRegressor(**p))])


def build_hgb_pipe(rs: int, target_name: str) -> Pipeline:
    if target_name == "qtd":
        hgb = HistGradientBoostingRegressor(
            max_iter=480,
            max_depth=4,
            min_samples_leaf=42,
            learning_rate=0.06,
            l2_regularization=4.8,
            early_stopping=True,
            validation_fraction=0.14,
            n_iter_no_change=22,
            random_state=rs,
        )
    else:
        hgb = HistGradientBoostingRegressor(
            max_iter=560,
            max_depth=5,
            min_samples_leaf=28,
            learning_rate=0.04,
            l2_regularization=3.4,
            early_stopping=True,
            validation_fraction=0.14,
            n_iter_no_change=26,
            random_state=rs,
        )
    return Pipeline([("scaler", _mm()), ("model", hgb)])


def build_xgb_only_pipe(
    best_params: dict[str, Any],
    use_log: bool,
    rs: int,
    horizon: int = 7,
) -> Pipeline | None:
    if XGBRegressor is None:
        return None
    long_h = int(horizon) >= 21
    nfac = 0.95 if long_h else 0.82
    dcap = 8 if long_h else 6
    kw = {
        "n_estimators": min(420, int(nfac * float(best_params.get("n_estimators", 240)))),
        "max_depth": min(dcap, int(best_params.get("max_depth", 6))),
        "learning_rate": float(best_params.get("learning_rate", 0.055)),
        "subsample": float(min(0.9, best_params.get("subsample", 0.82) + 0.02)),
        "colsample_bytree": float(min(0.9, best_params.get("colsample_bytree", 0.82) + 0.02)),
        "reg_alpha": float(max(0.05, best_params.get("reg_alpha", 0.18) * 1.15)),
        "reg_lambda": float(max(0.5, best_params.get("reg_lambda", 1.4) * 1.35)),
        "random_state": rs,
        "n_jobs": -1,
        "verbosity": 0,
    }
    xgb = XGBRegressor(**kw)
    inner: Any = _wrap_log(xgb) if use_log else xgb
    return Pipeline([("scaler", _mm()), ("model", inner)])


def build_catboost_pipe(use_log: bool, rs: int, target_name: str) -> Pipeline | None:
    if CatBoostRegressor is None:
        return None
    iters = int(0.82 * (1000 if target_name == "valor" else 750))
    cb = CatBoostRegressor(
        iterations=max(200, iters),
        depth=5,
        learning_rate=0.032,
        l2_leaf_reg=14.0,
        random_strength=0.45,
        bagging_temperature=0.35,
        rsm=0.82,
        border_count=254,
        random_seed=rs,
        verbose=False,
        loss_function="RMSE",
        allow_writing_files=False,
        thread_count=-1,
    )
    inner: Any = _wrap_log(cb) if use_log else cb
    return Pipeline([("scaler", _mm()), ("model", inner)])


def build_extratrees_pipe(
    best_params: dict[str, Any], use_log: bool, rs: int, horizon: int = 7
) -> Pipeline:
    long_h = int(horizon) >= 21
    n_cap = 520 if long_h else 420
    d_cap = 14 if long_h else 12
    n_est = min(n_cap, max(260 if long_h else 240, int(best_params.get("n_estimators", 400))))
    depth = min(d_cap, max(7 if long_h else 6, int(best_params.get("max_depth", 8))))
    et = ExtraTreesRegressor(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_leaf=8,
        min_samples_split=16,
        max_features="sqrt",
        random_state=rs,
        n_jobs=-1,
    )
    inner: Any = _wrap_log(et) if use_log else et
    return Pipeline([("scaler", _mm()), ("model", inner)])


def build_ngboost_pipe(rs: int) -> Pipeline | None:
    if NGBRegressor is None or Normal is None:
        return None
    ngb = NGBRegressor(
        Dist=Normal,
        n_estimators=240,
        learning_rate=0.045,
        natural_gradient=True,
        random_state=rs,
        verbose=False,
    )
    return Pipeline([("scaler", _mm()), ("model", ngb)])


def build_light_stack_pipe(
    best_params: dict[str, Any], use_log: bool, rs: int, target_name: str
) -> Pipeline | None:
    if LGBMRegressor is None:
        return None
    lgb_p = {**best_params, "random_state": rs, "verbosity": -1, "n_jobs": -1}
    lgb_p["n_estimators"] = min(int(lgb_p.get("n_estimators", 200)), 200)
    lgb_p["min_child_samples"] = max(int(lgb_p.get("min_child_samples", 20)), 22)
    lgb = LGBMRegressor(**lgb_p)
    hgb = HistGradientBoostingRegressor(
        max_iter=320,
        max_depth=4,
        min_samples_leaf=32 if target_name == "qtd" else 22,
        learning_rate=0.05,
        l2_regularization=5.5 if target_name == "qtd" else 3.8,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=rs,
    )
    stack = StackingRegressor(
        estimators=[("lgb", lgb), ("hgb", hgb)],
        final_estimator=Ridge(alpha=32.0),
        passthrough=False,
        n_jobs=-1,
    )
    inner: Any = _wrap_log(stack) if use_log else stack
    return Pipeline([("scaler", _mm()), ("model", inner)])


def build_dummy_median_pipe() -> Pipeline:
    return Pipeline(
        [("scaler", _mm()), ("model", DummyRegressor(strategy="median"))]
    )


def _score_val_mae(
    pipe: Any,
    X_tr,
    y_tr,
    X_va,
    y_va,
    use_sw: bool,
    target_name: str,
    horizon: int = 7,
) -> float:
    p = clone(pipe)
    _safe_fit_pipeline(p, X_tr, y_tr, use_sw, target_name, horizon)
    pred = p.predict(X_va)
    pred = np.maximum(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return float(mean_absolute_error(y_va, pred))


def _collect_candidate_specs(
    target_name: str, best_params: dict[str, Any], rs: int, horizon: int = 7
) -> list[tuple[str, Pipeline, bool]]:
    """Nome, pipeline (template), usar sample_weight recency."""
    out: list[tuple[str, Pipeline, bool]] = []
    long_h = int(horizon) >= 21

    out.append(("Ridge", build_ridge_pipe(rs), False))
    out.append(("ElasticNet", build_elastic_net_pipe(rs, target_name), False))
    out.append(("RandomForest", build_random_forest_pipe(rs, horizon), False))

    if LGBMRegressor is not None:
        out.append(("LGBM", build_lgbm_only_pipe(best_params, False, rs), True))
        out.append(("LGBM+fair", build_lgbm_fair_pipe(best_params, rs), True))
        out.append(("LGBM+q0.78", build_lgbm_quantile_pipe(best_params, 0.78, rs), True))
        if long_h:
            out.append(
                ("LGBM+q0.85", build_lgbm_quantile_pipe(best_params, 0.85, rs), True)
            )
        if target_name == "valor":
            out.append(("LGBM+q0.82", build_lgbm_quantile_pipe(best_params, 0.82, rs), True))
            if long_h:
                out.append(
                    ("LGBM+q0.90", build_lgbm_quantile_pipe(best_params, 0.90, rs), True)
                )
            out.append(("LGBM+log1p", build_lgbm_only_pipe(best_params, True, rs), True))

    out.append(("TS-linear", build_ts_linear_bridge_pipe(), False))
    if LGBMRegressor is not None:
        out.append(
            (
                "Med+High-LGBM",
                build_median_quantile_blend_pipe(best_params, rs, target_name, horizon),
                False,
            )
        )

    out.append(("HGB", build_hgb_pipe(rs, target_name), False))

    xgbp = build_xgb_only_pipe(best_params, False, rs, horizon)
    if xgbp is not None:
        out.append(("XGB", xgbp, True))
        if target_name == "valor":
            xl = build_xgb_only_pipe(best_params, True, rs, horizon)
            if xl is not None:
                out.append(("XGB+log1p", xl, True))

    cb = build_catboost_pipe(False, rs, target_name)
    if cb is not None:
        out.append(("CatBoost", cb, False))
        if target_name == "valor":
            cb_log = build_catboost_pipe(True, rs, target_name)
            if cb_log is not None:
                out.append(("CatBoost+log1p", cb_log, False))

    out.append(("ExtraTrees", build_extratrees_pipe(best_params, False, rs, horizon), True))
    if target_name == "valor":
        out.append(
            ("ExtraTrees+log1p", build_extratrees_pipe(best_params, True, rs, horizon), True)
        )
        ngbp = build_ngboost_pipe(rs)
        if ngbp is not None:
            out.append(("NGBoost", ngbp, False))

    lstack = build_light_stack_pipe(best_params, False, rs, target_name)
    if lstack is not None:
        out.append(("Stack(LGBM+HGB)", lstack, False))
        if target_name == "valor":
            ls2 = build_light_stack_pipe(best_params, True, rs, target_name)
            if ls2 is not None:
                out.append(("Stack+log1p", ls2, False))

    out.append(("Baseline mediana", build_dummy_median_pipe(), False))
    return out


def _pick_super_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_name: str,
    best_params: dict[str, Any],
    rs: int,
    blend_top_k: int,
    show_progress: bool = True,
    horizon: int = 7,
) -> tuple[list[Pipeline], np.ndarray, list[bool], str]:
    long_h = int(horizon) >= 21
    specs = _collect_candidate_specs(target_name, best_params, rs, horizon)
    scored: list[tuple[float, str, Pipeline, bool]] = []
    it = specs
    if show_progress:
        it = tqdm(
            specs,
            desc=f"Candidatos ({target_name})",
            leave=False,
            unit="modelo",
        )
    for name, pipe, use_sw in it:
        try:
            mae = _score_val_mae(
                pipe, X_train, y_train, X_val, y_val, use_sw, target_name, horizon
            )
            scored.append((mae, name, pipe, use_sw))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0])
    if not scored:
        p = build_dummy_median_pipe()
        return [clone(p)], np.array([1.0]), [False], "Baseline mediana (fallback)"

    best_mae, best_name, best_pipe, best_sw = scored[0]
    k = min(blend_top_k, len(scored))
    if k < 2:
        return [clone(best_pipe)], np.array([1.0]), [best_sw], best_name
    top = scored[:k]
    templates = [clone(t[2]) for t in top]
    maes = np.array([t[0] for t in top], dtype=float)
    sws = [t[3] for t in top]
    names = [t[1] for t in top]

    tau = max(float(np.median(maes)), 1e-9) * (0.62 if long_h else 0.78)
    w = np.exp(-(maes - maes.min()) / tau)
    w = w / (w.sum() + 1e-12)

    blend = _make_fitted_blend(
        templates, w, sws, X_train, y_train, target_name, horizon
    )
    pred_b = blend.predict(X_val)
    pred_b = np.maximum(np.nan_to_num(pred_b, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    mae_blend = float(mean_absolute_error(y_val, pred_b))

    rel_gain = (best_mae - mae_blend) / (best_mae + 1e-9)
    rel_min = 0.011 if long_h else 0.015
    if mae_blend < best_mae and k >= 2 and rel_gain > rel_min:
        short = "+".join(names[: min(4, len(names))])
        if len(names) > 4:
            short += "+…"
        return templates, w, sws, f"Super-blend ({k}): {short}"

    return [clone(best_pipe)], np.array([1.0]), [best_sw], best_name


def _sanitize_benchmark_appendix(d: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly: converte numpy e substitui NaN/Inf por null."""

    def fix(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return None
            return xf
        if isinstance(x, np.ndarray):
            return fix(x.tolist())
        return x

    return fix(d)


def _resolve_n_trials(n_trials: int, n_samples: int, horizon: int = 7) -> int:
    """n_trials <= 0 → escala com log do tamanho da amostra (faixa ~55–140)."""
    if n_trials > 0:
        return n_trials
    base = int(max(55, min(140, int(24 * np.log1p(n_samples)))))
    if int(horizon) >= 21:
        base = min(158, base + 24)
    return base


def train_one_target(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    target_name: str,
    n_trials: int = 0,
    n_splits_tss: int = 7,
    random_state: int = 42,
    optuna_seed: int = 42,
    blend_top_k: int = 5,
    show_progress: bool = True,
    full_period_train: bool = False,
) -> TrainResult:
    n = len(X)
    hz = int(horizon)

    if full_period_train:
        X_fit = X
        y_fit = y
        sl_e_tr, sl_e_va = _ensemble_selection_splits(n)
        X_train, y_train = X.iloc[sl_e_tr], y.iloc[sl_e_tr]
        X_val, y_val = X.iloc[sl_e_va], y.iloc[sl_e_va]
        dates_val = X_val.index
        dates_test = X_fit.index
        X_opt, y_opt = X_fit, y_fit
        X_imp = X_fit
        y_imp = y_fit
    else:
        sl_tr, sl_te = temporal_train_test_indices(n, 0.8)
        X_80 = X.iloc[sl_tr]
        y_80 = y.iloc[sl_tr]
        X_test = X.iloc[sl_te]
        y_test = y.iloc[sl_te]
        sl_e_tr, sl_e_va = _ensemble_selection_splits(len(X_80))
        X_train = X_80.iloc[sl_e_tr]
        y_train = y_80.iloc[sl_e_tr]
        X_val = X_80.iloc[sl_e_va]
        y_val = y_80.iloc[sl_e_va]
        dates_val = X_val.index
        dates_test = X_test.index
        X_fit = X_80
        y_fit = y_80
        X_opt, y_opt = X_80, y_80
        X_imp = X
        y_imp = y

    n_splits = min(n_splits_tss, max(3, len(X_opt) // 22))
    n_splits = max(3, n_splits)
    nt = _resolve_n_trials(n_trials, len(X_opt), hz)

    if LGBMRegressor is None:
        if hz >= 21:
            best_params = {
                "n_estimators": 300,
                "max_depth": 7,
                "learning_rate": 0.062,
                "subsample": 0.74,
                "colsample_bytree": 0.72,
                "reg_alpha": 0.14,
                "reg_lambda": 1.55,
                "num_leaves": 54,
                "min_child_samples": 22,
                "min_split_gain": 0.018,
            }
        else:
            best_params = {
                "n_estimators": 220,
                "max_depth": 6,
                "learning_rate": 0.07,
                "subsample": 0.78,
                "colsample_bytree": 0.78,
                "reg_alpha": 0.35,
                "reg_lambda": 2.2,
                "num_leaves": 40,
                "min_child_samples": 36,
                "min_split_gain": 0.045,
            }
    else:

        def objective(trial: optuna.Trial) -> float:
            return _optuna_objective_lgbm(
                trial,
                X_opt,
                y_opt,
                n_splits=n_splits,
                random_state=random_state,
                target_name=target_name,
                horizon=hz,
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=optuna_seed,
                multivariate=True,
                n_startup_trials=max(10, min(28, nt // 4)),
            ),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=nt,
            show_progress_bar=show_progress,
        )
        best_params = study.best_params

    from .benchmark_runner import run_model_benchmark

    specs_bm = _collect_candidate_specs(target_name, best_params, random_state, hz)

    def _fit_bm(pipe: Any, X: pd.DataFrame, y: pd.Series, sw: bool) -> None:
        _safe_fit_pipeline(pipe, X, y, sw, target_name, hz)

    Xt_eval = X_test if not full_period_train else X_fit
    yt_eval = y_test if not full_period_train else y_fit
    bench_raw = run_model_benchmark(
        specs_bm,
        _fit_bm,
        X_train,
        y_train,
        X_val,
        y_val,
        X_fit,
        y_fit,
        Xt_eval,
        yt_eval,
        show_progress=show_progress,
    )
    spec_map: dict[str, tuple[Any, bool]] = dict(bench_raw.get("spec_by_name") or {})
    benchmark_appendix = _sanitize_benchmark_appendix(
        {k: v for k, v in bench_raw.items() if k != "spec_by_name"}
    )

    wn = list(bench_raw.get("winner_names") or [])
    ww = np.asarray(bench_raw.get("winner_weights") or [], dtype=float)
    if (
        wn
        and len(ww) == len(wn)
        and float(ww.sum()) > 1e-12
        and all(n in spec_map for n in wn)
    ):
        ww = ww / (ww.sum() + 1e-12)
        templates = [clone(spec_map[n][0]) for n in wn]
        sws = [spec_map[n][1] for n in wn]
        weights = ww
        model_label = f"Benchmark: {bench_raw.get('winner_label', '?')}"
    else:
        templates, weights, sws, model_label = _pick_super_ensemble(
            X_train,
            y_train,
            X_val,
            y_val,
            target_name,
            best_params,
            random_state,
            blend_top_k=blend_top_k,
            show_progress=show_progress,
            horizon=hz,
        )

    pipe_val = _make_fitted_blend(
        templates, weights, sws, X_train, y_train, target_name, hz
    )
    pred_val = pipe_val.predict(X_val)
    pipe_eval = _make_fitted_blend(
        templates, weights, sws, X_fit, y_fit, target_name, hz
    )
    if full_period_train:
        pred_test = pipe_eval.predict(X_fit)
        y_test = y_fit
        pipe = pipe_eval
    else:
        pred_test = pipe_eval.predict(X_test)
        pipe = _make_fitted_blend(
            templates, weights, sws, X_imp, y_imp, target_name, hz
        )

    y_max_tr = float(np.percentile(y_train, 99.5)) if len(y_train) else 0.0
    long_hz = hz >= 21
    if target_name == "qtd":
        cap_mult = 2.75 if long_hz else 2.35
    else:
        cap_mult = 4.45 if long_hz else 3.55
    cap = max(y_max_tr * cap_mult, 1.0)
    pred_val = np.clip(np.nan_to_num(pred_val, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap)
    pred_test = np.clip(np.nan_to_num(pred_test, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap)

    if full_period_train:
        pred_tr_chart = np.clip(
            np.nan_to_num(pipe_val.predict(X_train), nan=0.0, posinf=cap, neginf=0.0),
            0.0,
            cap,
        )
        chart_y_pred = np.concatenate(
            [np.asarray(pred_tr_chart, dtype=float).ravel(), pred_val.ravel()]
        ).tolist()
        chart_y_real = (
            np.asarray(y_train, dtype=float).ravel().tolist()
            + np.asarray(y_val, dtype=float).ravel().tolist()
        )
        chart_dates = [d.strftime("%Y-%m-%d") for d in X_train.index] + [
            d.strftime("%Y-%m-%d") for d in X_val.index
        ]
        chart_split_index = int(len(X_train))
    else:
        pred_tr_chart = np.clip(
            np.nan_to_num(pipe_eval.predict(X_fit), nan=0.0, posinf=cap, neginf=0.0),
            0.0,
            cap,
        )
        chart_y_pred = np.concatenate(
            [
                np.asarray(pred_tr_chart, dtype=float).ravel(),
                np.asarray(pred_test, dtype=float).ravel(),
            ]
        ).tolist()
        chart_y_real = (
            np.asarray(y_fit, dtype=float).ravel().tolist()
            + np.asarray(y_test, dtype=float).ravel().tolist()
        )
        chart_dates = [d.strftime("%Y-%m-%d") for d in X_fit.index] + [
            d.strftime("%Y-%m-%d") for d in X_test.index
        ]
        chart_split_index = int(len(X_fit))

    metrics_val = _metrics_dict(y_val.values, pred_val, target_name)
    metrics_test = _metrics_dict(np.asarray(y_test, dtype=float), pred_test, target_name)
    metrics_val["n_features"] = float(X.shape[1])
    metrics_test["n_features"] = float(X.shape[1])

    importance = pd.Series(dtype=float)
    if LGBMRegressor is not None:
        imp_model = LGBMRegressor(
            **{**best_params, "random_state": random_state, "verbosity": -1, "n_jobs": -1}
        )
        scaler = _mm()
        Xs = scaler.fit_transform(X_imp)
        sw_imp = sample_weights(
            pd.Series(np.asarray(y_imp, dtype=float)),
            True,
            target_name,
            hz,
        )
        imp_model.fit(
            Xs, np.asarray(y_imp, dtype=float), sample_weight=sw_imp
        )
        importance = (
            pd.Series(imp_model.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
            .head(20)
        )

    return TrainResult(
        pipeline=pipe,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        importance=importance,
        y_val=y_val,
        y_val_pred=pred_val,
        y_test=y_test,
        y_test_pred=pred_test,
        best_params=best_params,
        dates_val=dates_val,
        dates_test=dates_test,
        model_label=model_label,
        full_period_train=full_period_train,
        chart_dates=chart_dates,
        chart_y_real=chart_y_real,
        chart_y_pred=chart_y_pred,
        chart_split_index=chart_split_index,
        benchmark_appendix=benchmark_appendix,
    )
