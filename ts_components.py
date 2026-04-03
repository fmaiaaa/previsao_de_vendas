"""Componentes de séries temporais (sklearn-compatible) para o ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class TSLinearBridgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge só em colunas `ts_*` (sinais de ST, sazonalidade explícita, etc.).
    Captura tendência linear nos componentes temporais sem suavizar tanto quanto árvores puras.
    """

    def __init__(self, alphas: np.ndarray | None = None) -> None:
        self.alphas = alphas if alphas is not None else np.logspace(-1.5, 3.5, 18)

    def fit(self, X: pd.DataFrame, y: Any) -> TSLinearBridgeRegressor:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        cols = [c for c in X.columns if str(c).startswith("ts_")]
        if len(cols) < 2:
            cols = list(X.columns)[: min(12, X.shape[1])]
        self.cols_: list[str] = cols
        self.scaler_ = MinMaxScaler(feature_range=(0, 1), clip=True)
        Z = self.scaler_.fit_transform(X[cols].astype(float))
        self.ridge_ = RidgeCV(alphas=self.alphas, cv=3)
        self.ridge_.fit(Z, np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        Z = self.scaler_.transform(X[self.cols_].astype(float))
        return np.maximum(self.ridge_.predict(Z), 0.0)


def build_ts_linear_bridge_pipe() -> Pipeline:
    return Pipeline(
        [
            ("ts_lin", TSLinearBridgeRegressor()),
        ]
    )


class MedianQuantileBlendRegressor(BaseEstimator, RegressorMixin):
    """
    LGBM médio (L2) + LGBM quantil alto: mistura calibrada no fim do treino para
    não colapsar na média quando o VGV (ou volume) tem cauda pesada.
    """

    def __init__(
        self,
        best_params: dict[str, Any] | None,
        random_state: int = 42,
        target_name: str = "valor",
        q_high: float = 0.82,
        horizon: int = 7,
    ) -> None:
        self.best_params = dict(best_params or {})
        self.random_state = random_state
        self.target_name = target_name
        self.q_high = q_high
        self.horizon = int(horizon)

    def fit(self, X: pd.DataFrame, y: Any) -> MedianQuantileBlendRegressor:
        from lightgbm import LGBMRegressor

        from .weights import sample_weights

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        y_s = pd.Series(np.asarray(y, dtype=float).ravel())
        n = len(y_s)
        cut = max(int(n * 0.87), n - 30)
        X_tr, y_tr = X.iloc[:cut], y_s.iloc[:cut]
        X_ho, y_ho = X.iloc[cut:], y_s.iloc[cut:]

        self.scaler_ = MinMaxScaler(feature_range=(0, 1), clip=True)
        X_tr_s = self.scaler_.fit_transform(X_tr)
        sw_tr = sample_weights(y_tr, True, self.target_name, self.horizon)

        base = {
            **self.best_params,
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        self.med_ = LGBMRegressor(**base)
        self.med_.fit(X_tr_s, y_tr, sample_weight=sw_tr)

        q_params = {**base, "objective": "quantile", "alpha": self.q_high}
        self.q_ = LGBMRegressor(**q_params)
        self.q_.fit(X_tr_s, y_tr, sample_weight=sw_tr)

        long_h = self.horizon >= 21
        self.gamma_ = 0.42 if long_h else 0.34
        if len(y_ho) > 6:
            Xh = self.scaler_.transform(X_ho)
            pm = self.med_.predict(Xh)
            pq = self.q_.predict(Xh)
            best_mae = 1e30
            best_g = self.gamma_
            g_lo, g_hi, n_g = (0.22, 0.78, 15) if long_h else (0.12, 0.62, 12)
            for g in np.linspace(g_lo, g_hi, n_g):
                pred = (1.0 - g) * pm + g * pq
                mae = mean_absolute_error(
                    y_ho, np.maximum(pred, 0.0)
                )
                if mae < best_mae:
                    best_mae, best_g = mae, float(g)
            self.gamma_ = best_g

        X_full_s = self.scaler_.transform(X)
        sw_full = sample_weights(y_s, True, self.target_name, self.horizon)
        self.med_.fit(X_full_s, y_s, sample_weight=sw_full)
        self.q_.fit(X_full_s, y_s, sample_weight=sw_full)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        Xs = self.scaler_.transform(X)
        pm = self.med_.predict(Xs)
        pq = self.q_.predict(Xs)
        out = (1.0 - self.gamma_) * pm + self.gamma_ * pq
        return np.maximum(np.asarray(out, dtype=float), 0.0)


def build_median_quantile_blend_pipe(
    best_params: dict[str, Any], rs: int, target_name: str, horizon: int = 7
) -> Pipeline:
    long_h = int(horizon) >= 21
    q_hi = 0.88 if long_h else 0.84
    inner = MedianQuantileBlendRegressor(
        best_params, rs, target_name, q_high=q_hi, horizon=int(horizon)
    )
    return Pipeline([("model", inner)])
