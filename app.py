"""
Previsão de vendas — Streamlit (Direcional). Ficheiro único para deploy (ex.: Streamlit Cloud).
"""

from __future__ import annotations

# ----- inlined: load.py -----

import io
import re
import unicodedata
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Union

import pandas as pd

PathLike = Union[str, Path, BinaryIO]


def normalize_header(name: str) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_header(c) for c in out.columns]
    # colisões raras: sufixo numérico
    seen: dict[str, int] = {}
    new_cols = []
    for c in out.columns:
        base = c or "col"
        if base in seen:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            new_cols.append(base)
    out.columns = new_cols
    return out


def _score_header_df(df: pd.DataFrame) -> int:
    """Prioriza tabelas com muitas colunas e nomes que parecem cabeçalho de negócio."""
    if df is None or df.shape[1] == 0:
        return -1
    names = [normalize_header(c) for c in df.columns]
    non_empty = sum(1 for n in names if n and not n.startswith("unnamed"))
    data_hits = sum(1 for n in names if "data" in n)
    return int(df.shape[1] * 3 + non_empty * 2 + data_hits * 15)


def read_excel(source: PathLike, **kwargs: Any) -> pd.DataFrame:
    """
    Lê .xlsx com tolerância a: folha não predefinida, título nas primeiras linhas,
    cabeçalho na linha 1–7, ou ficheiros onde a 1.ª linha está vazia.
    """
    engine = kwargs.pop("engine", "openpyxl")
    sheet_kw = kwargs.pop("sheet_name", None)

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if isinstance(source, io.BufferedIOBase) or hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "tell") else None
        raw = source.read()
        if pos is not None and pos == 0:
            try:
                source.seek(0)
            except Exception:
                pass
        buf = io.BytesIO(raw)
        xl: pd.ExcelFile = pd.ExcelFile(buf, engine=engine)
    else:
        xl = pd.ExcelFile(source, engine=engine)

    if sheet_kw is not None:
        for hdr in range(0, 10):
            try:
                df_try = pd.read_excel(
                    xl, sheet_name=sheet_kw, header=hdr, engine=engine, **kwargs
                )
            except Exception:
                continue
            if df_try.shape[1] > 0:
                return normalize_dataframe_columns(df_try)

    best_df: pd.DataFrame | None = None
    best_score = -1
    for sheet in xl.sheet_names:
        s_low = str(sheet).strip().lower()
        if s_low.startswith("graf") or "chart" in s_low:
            continue
        for hdr in range(0, 10):
            try:
                df = pd.read_excel(
                    xl, sheet_name=sheet, header=hdr, engine=engine, **kwargs
                )
            except Exception:
                continue
            if df.shape[1] == 0:
                continue
            sc = _score_header_df(df)
            if sc > best_score:
                best_score = sc
                best_df = df

    if best_df is None or best_df.shape[1] == 0:
        raise ValueError(
            "Não foi possível ler nenhuma folha com colunas neste Excel. "
            "Confirme que há dados numa folha e que o cabeçalho está nas primeiras 10 linhas."
        )

    return normalize_dataframe_columns(best_df)


def _match_file_role(filename: str) -> Optional[str]:
    u = normalize_header(filename).replace(" ", "")
    u_sp = normalize_header(filename)
    # Base de respostas do formulário (previsão humana) — antes de "vendas" genérico
    if "esboco" in u_sp or "esboço" in filename.lower():
        if "previsao" in u or "predicao" in u or "vendas" in u_sp:
            return "formulario_previsao"
    if "resposta" in u_sp and (
        "formulario" in u_sp or "formulário" in filename.lower()
    ):
        return "formulario_previsao"
    if ("previsao" in u or "predicao" in u) and (
        "resposta" in u_sp or "formulario" in u_sp
    ):
        return "formulario_previsao"
    if "vendas" in u and "predicao" in u.replace("ç", "c"):
        return "vendas"
    if "leads" in u:
        return "leads"
    if "pastas" in u:
        return "pastas"
    if "agendamento" in u or "visitas" in u or "visita" in u:
        return "agendamentos"
    # fallback por palavra solta
    if "vendas" in u:
        return "vendas"
    if "leads" in u:
        return "leads"
    return None


def classify_upload(name: str) -> str:
    role = _match_file_role(name)
    if role:
        return role
    u = normalize_header(name)
    if "formulario" in u and "resposta" in u:
        return "formulario_previsao"
    if "venda" in u:
        return "vendas"
    if "lead" in u:
        return "leads"
    if "pasta" in u:
        return "pastas"
    return "agendamentos"


def read_excel_formulario_previsao(source: PathLike) -> pd.DataFrame:
    """
    Lê o Excel tipo *ESBOÇO* / respostas ao formulário.
    Prioriza folha cujo nome normalizado contenha 'resposta' e 'formulario'.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if isinstance(source, io.BufferedIOBase) or hasattr(source, "read"):
        raw = source.read()
        buf = io.BytesIO(raw)
        xl: pd.ExcelFile = pd.ExcelFile(buf, engine="openpyxl")
    else:
        xl = pd.ExcelFile(source, engine="openpyxl")

    sheet_order: list[str] = []
    for sheet in xl.sheet_names:
        ns = normalize_header(sheet)
        if "resposta" in ns and "formulario" in ns:
            sheet_order.insert(0, sheet)
        elif "resposta" in ns:
            sheet_order.append(sheet)
    if not sheet_order:
        sheet_order = list(xl.sheet_names)

    best_df: pd.DataFrame | None = None
    best_score = -1
    for sheet in sheet_order:
        if str(sheet).lower().startswith("graf") or "chart" in str(sheet).lower():
            continue
        for hdr in range(0, 10):
            try:
                df = pd.read_excel(
                    xl, sheet_name=sheet, header=hdr, engine="openpyxl"
                )
            except Exception:
                continue
            if df.shape[1] < 5:
                continue
            sc = _score_header_df(df)
            if sc > best_score:
                best_score = sc
                best_df = df

    if best_df is None or best_df.shape[1] == 0:
        raise ValueError(
            "Não foi possível ler folha útil do Excel do formulário de previsão."
        )
    return normalize_dataframe_columns(best_df)


def load_four_files(
    files: Mapping[str, PathLike],
) -> dict[str, pd.DataFrame]:
    """files: keys must be leads, agendamentos, pastas, vendas; opcional formulario_previsao."""
    required = {"leads", "agendamentos", "pastas", "vendas"}
    missing = required - set(files)
    if missing:
        raise ValueError(f"Faltam bases: {sorted(missing)}")
    out: dict[str, pd.DataFrame] = {}
    for k in sorted(required):
        src = files[k]
        label = str(src) if not hasattr(src, "read") else "(stream)"
        try:
            df = read_excel(src)
        except Exception as e:
            raise ValueError(
                f"Erro ao ler a base '{k}' ({label}): {e}"
            ) from e
        if df.shape[1] == 0:
            raise ValueError(
                f"A base '{k}' ficou sem colunas após a leitura ({label}). "
                "Pode ser folha em branco, ficheiro errado na pasta ou Excel corrompido."
            )
        out[k] = df
    if "formulario_previsao" in files and files["formulario_previsao"] is not None:
        fp = files["formulario_previsao"]
        label = str(fp) if not hasattr(fp, "read") else "(stream)"
        try:
            out["formulario_previsao"] = read_excel_formulario_previsao(fp)
        except Exception as e:
            raise ValueError(
                f"Erro ao ler formulário de previsão ({label}): {e}"
            ) from e
    return out


def find_column(df: pd.DataFrame, must_contain: list[str]) -> Optional[str]:
    for col in df.columns:
        c = str(col)
        if all(part in c for part in must_contain):
            return col
    return None


def find_column_any(df: pd.DataFrame, alternatives: list[list[str]]) -> Optional[str]:
    for parts in alternatives:
        hit = find_column(df, parts)
        if hit:
            return hit
    return None


def eda_dataframe(df: pd.DataFrame, name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "nome": name,
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "colunas_lista": list(df.columns),
        "tipos": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "nulos_por_coluna": df.isna().sum().sort_values(ascending=False).head(25).to_dict(),
        "amostra_head": df.head(5).to_html(classes="table", border=0, index=False),
    }
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]" or "data" in str(col):
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().sum() > max(10, len(df) * 0.05):
                vmin, vmax = s.min(), s.max()
                date_cols.append(
                    {
                        "coluna": col,
                        "nao_nulos": int(s.notna().sum()),
                        "min": vmin.isoformat() if pd.notna(vmin) else None,
                        "max": vmax.isoformat() if pd.notna(vmax) else None,
                    }
                )
    summary["colunas_data_candidatas"] = date_cols[:20]
    return summary

# ----- inlined: weights.py -----

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

# ----- inlined: metric_suite.py -----

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

# ----- inlined: benchmark_runner.py -----

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

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

# ----- inlined: ts_components.py -----

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

# ----- inlined: train_eval.py -----

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

# ----- inlined: macro_features.py -----
import json as _macro_json
import logging as _macro_logging
import os as _macro_os
import urllib.error as _macro_urllib_error
import urllib.request as _macro_urllib_request

_macro_logger = _macro_logging.getLogger(__name__)


def _parse_bcb_valor(vs: str) -> float:
    """Converte string `valor` da API SGS (ponto decimal ou formato 1.234,56)."""
    s = str(vs).strip()
    if not s:
        return float("nan")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


MACRO_BCB_SPECS: list[tuple[str, int, int, str]] = [
    ("macro_ptax_usd", 1, 3, "d"),
    ("macro_tr_dia", 11, 3, "d"),
    ("macro_selic_aa", 432, 3, "d"),
    ("macro_ipca_mom", 13522, 45, "m"),
    ("macro_ipca_12m", 433, 45, "m"),
    ("macro_ibc_br_dessaz", 24364, 45, "m"),
    ("macro_igpm_mom", 189, 45, "m"),
    ("macro_pmc_materiais_constr", 21859, 45, "m"),
    ("macro_pmc_indice_20786", 20786, 45, "m"),
    ("macro_pmc_volume_20539", 20539, 45, "m"),
    ("macro_setor_const_7459", 7459, 45, "m"),
]


def _macro_disabled() -> bool:
    v = (_macro_os.environ.get("PREVISAO_MACRO_BCB") or "").strip().lower()
    return v in ("0", "false", "no", "off")


def _fetch_bcb_sgs(codigo: int, data_inicial: str, data_final: str) -> pd.Series:
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
        f"?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )
    req = _macro_urllib_request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PrevisaoVendas/1.3)"},
    )
    try:
        with _macro_urllib_request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
    except (
        _macro_urllib_error.URLError,
        _macro_urllib_error.HTTPError,
        TimeoutError,
        OSError,
    ) as e:
        _macro_logger.warning("BCB SGS %s: falha de rede — %s", codigo, e)
        return pd.Series(dtype=float)
    if not raw or raw[0] not in "[{":
        return pd.Series(dtype=float)
    try:
        data = _macro_json.loads(raw)
    except _macro_json.JSONDecodeError:
        return pd.Series(dtype=float)
    if not isinstance(data, list) or not data:
        return pd.Series(dtype=float)
    dates: list[pd.Timestamp] = []
    vals: list[float] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("data", "")).strip()
        vs = str(row.get("valor", "")).strip()
        if not ds:
            continue
        try:
            dt = pd.to_datetime(ds, dayfirst=True, errors="coerce")
            if pd.isna(dt):
                continue
            v = _parse_bcb_valor(vs)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            dates.append(pd.Timestamp(dt).normalize())
            vals.append(v)
    if not dates:
        return pd.Series(dtype=float)
    s = pd.Series(vals, index=pd.DatetimeIndex(dates))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _align_series_to_index(
    s: pd.Series,
    idx: pd.DatetimeIndex,
    lag_days: int,
) -> pd.Series:
    if s.empty or len(idx) == 0:
        return pd.Series(0.0, index=idx)
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=int(lag_days))
    start = min(s.index.min(), idx.min())
    end = idx.max()
    daily = pd.date_range(start, end, freq="D")
    expanded = s.reindex(daily).ffill()
    out = expanded.reindex(idx.normalize()).ffill()
    return out.fillna(0.0)


def merge_macro_bcb_into_daily(
    df_master: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> None:
    if _macro_disabled():
        for name, *_ in MACRO_BCB_SPECS:
            df_master[name] = 0.0
        df_master["macro_ptax_ret20d"] = 0.0
        return

    idx = pd.DatetimeIndex(df_master.index).normalize()
    if len(idx) == 0:
        return

    buf = pd.Timedelta(days=800)
    d0 = (idx.min() - buf).strftime("%d/%m/%Y")
    d1 = idx.max().strftime("%d/%m/%Y")

    for col, codigo, lag, _freq in MACRO_BCB_SPECS:
        s = _fetch_bcb_sgs(codigo, d0, d1)
        aligned = _align_series_to_index(s, idx, lag)
        df_master[col] = aligned.values

    if "macro_ptax_usd" in df_master.columns:
        px = pd.to_numeric(df_master["macro_ptax_usd"], errors="coerce").replace(
            0.0, np.nan
        )
        chg = px.pct_change(periods=20).shift(1).fillna(0.0).clip(-0.5, 0.5)
        df_master["macro_ptax_ret20d"] = chg.values
    else:
        df_master["macro_ptax_ret20d"] = 0.0

    if show_progress:
        nz = sum(
            1 for c, *_ in MACRO_BCB_SPECS if float(df_master[c].abs().sum()) > 0
        )
        print(f"Macro BCB: {nz}/{len(MACRO_BCB_SPECS)} séries com valores não nulos.")


# ----- inlined: features.py -----

from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Após o sábado de referência, os totais do formulário só entram como feature (evita vazamento).
FORMULARIO_FEATURE_LAG_DIAS = 7

# Feriados nacionais + estaduais (RJ) — pacote `holidays`; se ausente, colunas ficam a zero.
CAL_SUBDIV_BR = "RJ"


def _br_holiday_dates_for_index(idx: pd.DatetimeIndex) -> frozenset:
    """Conjunto de datas (date) de feriado no Brasil (RJ). Vazio se `holidays` não estiver instalado."""
    try:
        import holidays
    except ImportError:
        return frozenset()
    try:
        y0 = int(pd.Timestamp(idx.min()).year) - 1
        y1 = int(pd.Timestamp(idx.max()).year) + 2
        years = range(y0, y1 + 1)
        cal = holidays.Brazil(subdiv=CAL_SUBDIV_BR, years=years)
        return frozenset(cal.keys())
    except Exception:
        return frozenset()


def _inject_calendario_feriados_br(
    df_master: pd.DataFrame, idx: pd.DatetimeIndex, H: frozenset
) -> None:
    """
    Flags de calendário (sem leakage: só o dia t).
    H: datas (date) de feriado — Brasil subdiv RJ (nacional + estaduais).
    """
    from bisect import bisect_left, bisect_right
    from datetime import timedelta

    n = len(idx)
    if not H:
        df_master["cal_feriado_br"] = 0
        df_master["cal_vespera_feriado_br"] = 0
        df_master["cal_pos_feriado_br"] = 0
        df_master["cal_feriado_pontu"] = 0
        df_master["cal_dias_ate_proximo_feriado"] = 0.0
        df_master["cal_dias_desde_feriado"] = 0.0
        return

    sorted_h = sorted(H)
    fer = np.zeros(n, dtype=np.float64)
    ves = np.zeros(n, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    pont = np.zeros(n, dtype=np.float64)
    ate = np.full(n, 30.0, dtype=np.float64)
    desde = np.full(n, 30.0, dtype=np.float64)

    def _feriado_em_ponte(d) -> bool:
        """Feriado em sexta/sáb/dom ou segunda (padrão típico de emenda)."""
        if d not in H:
            return False
        wd = d.weekday()
        return wd in (0, 4, 5, 6)

    for i, ts in enumerate(idx):
        d = pd.Timestamp(ts).normalize().date()
        if d in H:
            fer[i] = 1.0
            if _feriado_em_ponte(d):
                pont[i] = 1.0
        if (d + timedelta(days=1)) in H:
            ves[i] = 1.0
        if (d - timedelta(days=1)) in H:
            pos[i] = 1.0
        j = bisect_left(sorted_h, d)
        if j < len(sorted_h):
            ate[i] = min(30.0, float((sorted_h[j] - d).days))
        k = bisect_right(sorted_h, d) - 1
        if k >= 0:
            desde[i] = min(30.0, float((d - sorted_h[k]).days))

    df_master["cal_feriado_br"] = fer
    df_master["cal_vespera_feriado_br"] = ves
    df_master["cal_pos_feriado_br"] = pos
    df_master["cal_feriado_pontu"] = pont
    df_master["cal_dias_ate_proximo_feriado"] = ate
    df_master["cal_dias_desde_feriado"] = desde


def _to_day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _resolve_leads(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    date_c = find_column_any(
        df,
        [
            ["data", "criacao"],
            ["data", "criacao", "lead"],
            ["data", "atendimento"],
            ["data", "lead"],
        ],
    )
    if date_c is None:
        date_c = find_column_any(df, [["data", "criacao"]])
    if date_c is None:
        raise ValueError(
            "Base Leads: não encontrei coluna de data (ex.: 'Data de criação'). "
            f"Colunas: {list(df.columns)}"
        )
    out = df[[date_c]].copy()
    out["_d"] = _to_day(out[date_c])
    return out, date_c


def _resolve_agendamentos(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    c_ag = find_column_any(df, [["data", "agendamento"]])
    c_vis = find_column_any(df, [["data", "visita"]])
    if c_ag is None:
        raise ValueError(
            "Base Agendamentos: falta 'Data Agendamento'. "
            f"Colunas: {list(df.columns)}"
        )
    if c_vis is None:
        raise ValueError(
            "Base Agendamentos: falta 'Data da visita'. "
            f"Colunas: {list(df.columns)}"
        )
    out = df[[c_ag, c_vis]].copy()
    out["_dag"] = _to_day(out[c_ag])
    out["_dvi"] = _to_day(out[c_vis])
    return out, c_ag, c_vis


def _resolve_pastas(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    c = find_column_any(
        df,
        [
            ["data", "criacao", "pasta"],
            ["data", "criacao"],
        ],
    )
    if c is None:
        raise ValueError(
            "Base Pastas: não encontrei 'Data Criação Pasta'. "
            f"Colunas: {list(df.columns)}"
        )
    out = df[[c]].copy()
    out["_d"] = _to_day(out[c])
    return out, c


def _resolve_vendas(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    c_data = find_column_any(
        df,
        [
            ["data", "venda"],
            ["data", "fechamento"],
        ],
    )
    if c_data is None:
        raise ValueError(
            "Base Vendas: não encontrei 'Data da venda'. "
            f"Colunas: {list(df.columns)}"
        )
    c_val = find_column_any(
        df,
        [
            ["valor", "real", "venda"],
            ["valor", "real"],
        ],
    )
    if c_val is None:
        c_val = find_column_any(df, [["valor"]])
    if c_val is None:
        raise ValueError(
            "Base Vendas: não encontrei 'Valor Real de Venda'. "
            f"Colunas: {list(df.columns)}"
        )
    out = df[[c_data, c_val]].copy()
    out["_d"] = _to_day(out[c_data])
    out["_valor"] = pd.to_numeric(out[c_val], errors="coerce").fillna(0.0)
    return out, c_data, c_val


def forward_calendar_features(index: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """
    Por cada dia t: contagens no calendário dos dias t+1..t+H.
    Não usa vendas — apenas datas (permitido para previsão).
    Inclui feriados BR (RJ) no horizonte futuro.
    """
    from datetime import timedelta

    H = _br_holiday_dates_for_index(index)

    def _ponte(d) -> bool:
        if d not in H:
            return False
        return d.weekday() in (0, 4, 5, 6)

    fwd_wknd = np.zeros(len(index), dtype=float)
    fwd_bday = np.zeros(len(index), dtype=float)
    fwd_fer = np.zeros(len(index), dtype=float)
    fwd_ves = np.zeros(len(index), dtype=float)
    fwd_pos = np.zeros(len(index), dtype=float)
    fwd_pont = np.zeros(len(index), dtype=float)
    for i, ts in enumerate(index):
        dr = pd.date_range(ts + pd.Timedelta(days=1), periods=horizon, freq="D")
        dow = dr.dayofweek.to_numpy()
        fwd_wknd[i] = float((dow >= 5).sum())
        fwd_bday[i] = float((dow < 5).sum())
        if H:
            nf = nv = npos = npont = 0
            for dt in dr:
                d = pd.Timestamp(dt).normalize().date()
                if d in H:
                    nf += 1
                    if _ponte(d):
                        npont += 1
                if (d + timedelta(days=1)) in H:
                    nv += 1
                if (d - timedelta(days=1)) in H:
                    npos += 1
            fwd_fer[i] = float(nf)
            fwd_ves[i] = float(nv)
            fwd_pos[i] = float(npos)
            fwd_pont[i] = float(npont)
    h = float(horizon)
    return pd.DataFrame(
        {
            "fwd_h": h,
            "fwd_wknd_h": fwd_wknd,
            "fwd_bday_h": fwd_bday,
            "fwd_wknd_ratio": fwd_wknd / (h + 1e-9),
            "fwd_feriados_h": fwd_fer,
            "fwd_vesperas_h": fwd_ves,
            "fwd_pos_feriado_h": fwd_pos,
            "fwd_feriado_pontu_h": fwd_pont,
        },
        index=index,
    )


def _inject_ts_and_stl_features(
    df_master: pd.DataFrame, show_progress: bool = False
) -> None:
    """Picos, quantis, eco semanal, ticket e STL rolante (statsmodels) — só passado."""
    for col, short in (("target_valor", "vgv"), ("target_qtd", "qtd")):
        s = df_master[col].astype(float)
        df_master[f"ts_rollmax7_{short}"] = s.shift(1).rolling(7, min_periods=1).max()
        df_master[f"ts_rollmax30_{short}"] = s.shift(1).rolling(30, min_periods=1).max()
        df_master[f"ts_q85_30_{short}"] = (
            s.shift(1).rolling(30, min_periods=10).quantile(0.85)
        )
        df_master[f"ts_q90_60_{short}"] = (
            s.shift(1).rolling(60, min_periods=15).quantile(0.90)
        )
        df_master[f"ts_ewm_short_{short}"] = s.shift(1).ewm(span=5, adjust=False).mean()
        roll_m = s.shift(1).rolling(14, min_periods=3).mean()
        df_master[f"ts_burst_{short}"] = (s.shift(1) / (roll_m + 1e-6)).fillna(0.0)

    tv = df_master["target_valor"]
    tq = df_master["target_qtd"]
    df_master["ts_weekly_echo_vgv"] = (
        0.45 * tv.shift(7) + 0.30 * tv.shift(14) + 0.25 * tv.shift(21)
    ).fillna(0.0)
    df_master["ts_weekly_echo_qtd"] = (
        0.45 * tq.shift(7) + 0.30 * tq.shift(14) + 0.25 * tq.shift(21)
    ).fillna(0.0)
    df_master["ts_ticket_lag1"] = (tv.shift(1) / (tq.shift(1) + 1e-6)).fillna(0.0)

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        for short in ("vgv", "qtd"):
            df_master[f"ts_stl_trend_{short}"] = 0.0
            df_master[f"ts_stl_seasamp_{short}"] = 0.0
        return

    win, period = 91, 7
    for col, short in (("target_valor", "vgv"), ("target_qtd", "qtd")):
        s = df_master[col].astype(float)
        n = len(s)
        arr = s.to_numpy(dtype=float)
        trend = np.zeros(n)
        seas_amp = np.zeros(n)
        idx_iter = range(win, n)
        if show_progress:
            idx_iter = tqdm(
                idx_iter,
                desc=f"STL {short}",
                leave=False,
                unit="dia",
            )
        for i in idx_iter:
            seg = arr[i - win : i]
            if np.nanmax(seg) - np.nanmin(seg) < 1e-11:
                trend[i] = seg[-1]
                seas_amp[i] = 0.0
                continue
            try:
                r = STL(seg, period=period, robust=True).fit()
                trend[i] = float(r.trend[-1])
                seas_amp[i] = float(np.std(r.seasonal))
            except Exception:
                trend[i] = float(np.nanmean(seg[-period:]))
                seas_amp[i] = 0.0
        df_master[f"ts_stl_trend_{short}"] = trend
        df_master[f"ts_stl_seasamp_{short}"] = seas_amp


def _find_val_col_no_qtd(df: pd.DataFrame, parts: list[str]) -> str | None:
    """Coluna de valor (VGV) nas colunas 'Vendas ...' sem 'qtd' no nome."""
    for col in df.columns:
        c = str(col)
        if "qtd" in c:
            continue
        if all(p in c for p in parts):
            return col
    return None


def _find_qtd_col(df: pd.DataFrame, parts: list[str]) -> str | None:
    for col in df.columns:
        c = str(col)
        if "qtd" not in c:
            continue
        if all(p in c for p in parts):
            return col
    return None


def _week_end_saturday(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts).normalize()
    try:
        p = t.to_period("W-SAT")
        return pd.Timestamp(p.end_time).normalize()
    except Exception:
        days = (5 - t.weekday()) % 7
        return (t + pd.Timedelta(days=days)).normalize()


def build_formulario_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por 'Data de referência' (sábado): soma VGV e QTD previstas/reais
    (facilitadas vs normais), como no formulário ESBOÇO 2.
    Índice = data normalizada do sábado de referência.
    """
    ref_c = find_column_any(
        df,
        [
            ["data", "referencia", "sabado"],
            ["referencia", "sabado"],
            ["data", "referencia"],
        ],
    )
    if ref_c is None:
        raise ValueError(
            "Formulário: não encontrei coluna de data de referência (sábado). "
            f"Colunas disponíveis: {list(df.columns)[:35]}"
        )

    val_prev_fac = _find_val_col_no_qtd(df, ["vendas", "facilitadas", "previstas"])
    val_prev_norm = _find_val_col_no_qtd(df, ["vendas", "normais", "previstas"])
    val_real_fac = _find_val_col_no_qtd(df, ["vendas", "facilitadas", "reais"])
    val_real_norm = _find_val_col_no_qtd(df, ["vendas", "normais", "reais"])
    q_prev_fac = _find_qtd_col(df, ["facilitadas", "previstas"])
    q_prev_norm = _find_qtd_col(df, ["normais", "previstas"])
    q_real_fac = _find_qtd_col(df, ["facilitadas", "reais"])
    q_real_norm = _find_qtd_col(df, ["normais", "reais"])

    dref = _to_day(df[ref_c])
    base = pd.DataFrame({"_ref": dref})
    specs: list[tuple[str, str | None]] = [
        ("fb_vgv_prev_fac_sum", val_prev_fac),
        ("fb_vgv_prev_norm_sum", val_prev_norm),
        ("fb_vgv_real_fac_sum", val_real_fac),
        ("fb_vgv_real_norm_sum", val_real_norm),
        ("fb_qtd_prev_fac_sum", q_prev_fac),
        ("fb_qtd_prev_norm_sum", q_prev_norm),
        ("fb_qtd_real_fac_sum", q_real_fac),
        ("fb_qtd_real_norm_sum", q_real_norm),
    ]
    for out_name, col in specs:
        if col is None:
            base[out_name] = 0.0
        else:
            base[out_name] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    base = base.dropna(subset=["_ref"])
    weekly = base.groupby("_ref", sort=True).sum()
    weekly.index = pd.DatetimeIndex(weekly.index).normalize()
    weekly = weekly.sort_index()
    return weekly


def merge_formulario_weekly_into_daily(
    df_master: pd.DataFrame, weekly: pd.DataFrame | None
) -> pd.DataFrame:
    """Alinha cada dia ao sábado da semana (W-SAT) e aplica defasagem temporal."""
    if weekly is None or weekly.empty:
        return df_master

    idx = df_master.index
    keys = [_week_end_saturday(pd.Timestamp(d)) for d in idx]
    cols = list(weekly.columns)
    data = {c: [] for c in cols}
    for i, _d in enumerate(idx):
        k = keys[i]
        if k in weekly.index:
            row = weekly.loc[k]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        else:
            row = pd.Series(0.0, index=cols)
        for c in cols:
            data[c].append(float(row[c]) if c in row.index else 0.0)
    fb = pd.DataFrame(data, index=idx)
    fb = fb.shift(FORMULARIO_FEATURE_LAG_DIAS).fillna(0.0)
    for c in fb.columns:
        df_master[c] = fb[c].astype(float)
    return df_master


def forward_sum(series: pd.Series, horizon: int) -> pd.Series:
    """Soma dos valores nos próximos `horizon` dias (t+1 .. t+H), sem incluir t."""
    arr = series.fillna(0).to_numpy(dtype=float)
    n = len(arr)
    y = np.full(n, np.nan, dtype=float)
    for i in range(n):
        j1 = i + 1
        j2 = i + 1 + horizon
        if j2 <= n:
            y[i] = float(arr[j1:j2].sum())
    return pd.Series(y, index=series.index)


def build_daily_master(
    dict_dfs: dict[str, pd.DataFrame], show_progress: bool = True
) -> pd.DataFrame:
    """
    dict_dfs keys: leads, agendamentos, pastas, vendas; opcional formulario_previsao.
    Retorna DataFrame diário ordenado com volumes e alvos diários (para derivar H).
    """
    df_l = dict_dfs["leads"]
    df_a = dict_dfs["agendamentos"]
    df_p = dict_dfs["pastas"]
    df_v = dict_dfs["vendas"]

    l_df, _ = _resolve_leads(df_l)
    a_df, _, _ = _resolve_agendamentos(df_a)
    p_df, _ = _resolve_pastas(df_p)
    v_df, _, _ = _resolve_vendas(df_v)

    leads_day = l_df.groupby("_d").size().to_frame("vol_leads")

    agend_day = a_df.groupby("_dag").size().to_frame("vol_agend")
    visit_mask = a_df["_dvi"].notna()
    visit_day = a_df.loc[visit_mask].groupby("_dvi").size().to_frame("vol_visit")

    pastas_day = p_df.groupby("_d").size().to_frame("vol_pastas")

    vendas_day = v_df.groupby("_d").agg(
        target_qtd=("_d", "count"),
        target_valor=("_valor", "sum"),
    )

    df_master = (
        pd.concat([vendas_day, leads_day, agend_day, visit_day, pastas_day], axis=1)
        .fillna(0.0)
        .sort_index()
    )

    df_form = dict_dfs.get("formulario_previsao")
    if df_form is not None and len(df_form) > 0:
        try:
            wk = build_formulario_weekly_aggregates(df_form)
            df_master = merge_formulario_weekly_into_daily(df_master, wk)
        except Exception as e:
            if show_progress:
                print(f"Aviso: não integrei o formulário de previsão: {e}")

    colunas_base = [
        "vol_leads",
        "vol_agend",
        "vol_visit",
        "vol_pastas",
        "target_qtd",
        "target_valor",
    ]
    colunas_fb = [c for c in df_master.columns if c.startswith("fb_")]
    for col in colunas_base + colunas_fb:
        s = df_master[col]
        for lag in (1, 3, 7, 14, 30):
            df_master[f"{col}_lag_{lag}"] = s.shift(lag)
        df_master[f"{col}_ma_7d"] = s.shift(1).rolling(7, min_periods=1).mean()
        df_master[f"{col}_ma_15d"] = s.shift(1).rolling(15, min_periods=1).mean()
        df_master[f"{col}_ma_30d"] = s.shift(1).rolling(30, min_periods=1).mean()
        df_master[f"{col}_std_15d"] = s.shift(1).rolling(15, min_periods=1).std()
        df_master[f"{col}_momentum"] = df_master[f"{col}_ma_7d"] / (
            df_master[f"{col}_ma_30d"] + 1e-5
        )

    df_master["tx_agend_lead"] = (
        df_master["vol_agend"].shift(1).rolling(7, min_periods=1).sum()
        / (df_master["vol_leads"].shift(7).rolling(7, min_periods=1).sum() + 1e-5)
    )
    df_master["tx_visit_agend"] = (
        df_master["vol_visit"].shift(1).rolling(7, min_periods=1).sum()
        / (df_master["vol_agend"].shift(3).rolling(7, min_periods=1).sum() + 1e-5)
    )
    df_master["tx_pasta_visit"] = (
        df_master["vol_pastas"].shift(1).rolling(7, min_periods=1).sum()
        / (df_master["vol_visit"].shift(3).rolling(7, min_periods=1).sum() + 1e-5)
    )

    idx = df_master.index
    day_of_month = pd.Series(idx.day, index=idx)
    day_of_week = pd.Series(idx.dayofweek, index=idx)
    df_master["sin_day"] = np.sin(2 * np.pi * day_of_month / 31.0)
    df_master["cos_day"] = np.cos(2 * np.pi * day_of_month / 31.0)
    df_master["sin_weekday"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df_master["cos_weekday"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df_master["time_idx"] = np.arange(len(df_master), dtype=float)
    ti = df_master["time_idx"].to_numpy(dtype=float)
    for period, tag in ((7.0, "w"), (30.44, "m"), (365.25, "y")):
        for k in (1, 2):
            df_master[f"fourier_{tag}_sin_{k}"] = np.sin(2 * np.pi * k * ti / period)
            df_master[f"fourier_{tag}_cos_{k}"] = np.cos(2 * np.pi * k * ti / period)
    df_master["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    dias_fim = (idx + pd.offsets.MonthEnd(0) - idx).days
    df_master["dias_para_fim_mes"] = dias_fim
    df_master["is_month_end"] = (df_master["dias_para_fim_mes"] <= 3).astype(int)
    df_master["distancia_pagamento"] = idx.day.map(lambda x: min(abs(x - 5), abs(x - 20)))

    _holiday_dates = _br_holiday_dates_for_index(idx)
    _inject_calendario_feriados_br(df_master, idx, _holiday_dates)

    merge_macro_bcb_into_daily(df_master, show_progress=show_progress)

    _inject_ts_and_stl_features(df_master, show_progress=show_progress)

    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_master = df_master.dropna(how="any")
    return df_master


def build_feature_matrix(df_master: pd.DataFrame) -> pd.DataFrame:
    """Matriz X alinhada ao índice diário (sem colunas de alvo bruto)."""
    feature_cols = [c for c in df_master.columns if c not in ("target_qtd", "target_valor")]
    X = df_master[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X.fillna(0.0)


def build_xy_for_horizon(
    df_master: pd.DataFrame,
    target_daily_col: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """X sem colunas de alvo bruto; y = soma futura em H dias; inclui calendário futuro (H)."""
    y = forward_sum(df_master[target_daily_col], horizon)
    feature_cols = [c for c in df_master.columns if c not in ("target_qtd", "target_valor")]
    X = df_master[feature_cols].copy()
    cal = forward_calendar_features(df_master.index, horizon)
    X = pd.concat([X, cal], axis=1)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0.0)
    return X, y


def predict_last_row(df_master: pd.DataFrame, pipeline: Any, horizon: int) -> float:
    """Previsão no último dia; `horizon` deve ser o mesmo usado no treino (colunas fwd_*)."""
    Xb = build_feature_matrix(df_master)
    cal = forward_calendar_features(df_master.index, horizon)
    X = pd.concat([Xb, cal], axis=1)
    X_last = X.iloc[[-1]]
    p = float(np.asarray(pipeline.predict(X_last)).ravel()[0])
    if not np.isfinite(p):
        p = 0.0
    return max(0.0, p)

# ----- inlined: gsheets_loader.py -----

import io
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

EXPORT_URL = "https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"

ROLE_LABELS: dict[str, str] = {
    "leads": "BD Leads",
    "agendamentos": "BD Agendamentos e visitas",
    "pastas": "BD Pastas",
    "vendas": "BD Vendas",
    "formulario_previsao": "Esboço — formulário",
}


def _validate_formulario_columns(df: pd.DataFrame) -> None:
    """Confirma colunas de referência + pelo menos uma métrica prevista/real (facilitada ou normal)."""
    fn = normalize_dataframe_columns(df.copy())
    ref = find_column_any(
        fn,
        [
            ["data", "referencia", "sabado"],
            ["referencia", "sabado"],
            ["data", "referencia"],
        ],
    )
    if ref is None:
        raise ValueError("sem coluna de data de referência (sábado)")
    hits = 0
    for c in fn.columns:
        cl = str(c)
        if ("previstas" in cl or "reais" in cl or "previst" in cl) and (
            "facilitadas" in cl or "normais" in cl or "facilitada" in cl or "normal" in cl
        ):
            hits += 1
    if hits < 2:
        raise ValueError("poucas colunas de previsão/realização facilitada|normal")


_SCORERS: dict[str, Callable[[pd.DataFrame], None]] = {
    "leads": lambda df: _resolve_leads(df),
    "agendamentos": lambda df: _resolve_agendamentos(df),
    "pastas": lambda df: _resolve_pastas(df),
    "vendas": lambda df: _resolve_vendas(df),
    "formulario_previsao": lambda df: _validate_formulario_columns(df),
}


def score_dataframe_for_role(df: pd.DataFrame, role: str) -> float:
    """Pontuação maior = melhor correspondência ao papel esperado."""
    if df is None or df.shape[0] < 3 or df.shape[1] < 2:
        return -1.0
    fn = normalize_dataframe_columns(df.copy())
    scorer = _SCORERS.get(role)
    if scorer is None:
        return -1.0
    try:
        scorer(fn)
    except Exception:
        return -1.0
    n = min(len(fn), 50_000)
    ncol = fn.shape[1]
    return 10_000.0 + ncol * 15.0 + n * 0.02


def _dataframe_from_values(
    rows: list[list[Any]], header_row: int
) -> pd.DataFrame | None:
    if not rows or header_row >= len(rows):
        return None
    header = [str(x).strip() if x is not None else "" for x in rows[header_row]]
    if sum(1 for h in header if h) < 2:
        return None
    data_rows = rows[header_row + 1 :]
    if not data_rows:
        return None
    nc = len(header)
    grid: list[list[Any]] = []
    for r in data_rows:
        r = list(r)
        if len(r) < nc:
            r = r + [""] * (nc - len(r))
        grid.append(r[:nc])
    return pd.DataFrame(grid, columns=header)


def _best_df_from_values(rows: list[list[Any]], role: str) -> tuple[pd.DataFrame, float, int]:
    """Varre linhas de cabeçalho candidatas (0..9)."""
    best: tuple[pd.DataFrame, float, int] | None = None
    max_h = min(10, max(0, len(rows) - 1))
    for h in range(max_h + 1):
        df = _dataframe_from_values(rows, h)
        if df is None:
            continue
        sc = score_dataframe_for_role(df, role)
        if sc < 0:
            continue
        if best is None or sc > best[1]:
            best = (df, sc, h)
    if best is None:
        raise ValueError("Nenhuma disposição de cabeçalho válida.")
    return best


def try_gspread_client() -> Any | None:
    """Instancia cliente gspread a partir de credenciais no ambiente (sem Streamlit)."""
    try:
        import gspread
    except ImportError:
        return None
    info = _service_account_info_from_env()
    if not info:
        return None
    try:
        return gspread.service_account_from_dict(info)
    except Exception as e:
        logger.warning("gspread service_account_from_dict falhou: %s", e)
        return None


def _service_account_info_from_env() -> dict[str, Any] | None:
    import os

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def service_account_info_from_streamlit_secrets() -> dict[str, Any] | None:
    """Lê JSON completo ou secção TOML gerada a partir das chaves do service account."""
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
            raw = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
            if isinstance(raw, str) and raw.strip():
                return json.loads(raw.strip())
        for key in ("google_service_account", "gcp_service_account", "service_account"):
            if key in st.secrets:
                sec = st.secrets[key]
                if isinstance(sec, str) and sec.strip().startswith("{"):
                    return json.loads(sec.strip())
                if hasattr(sec, "keys"):
                    return {str(k): sec[k] for k in sec.keys()}
    except Exception as e:
        logger.warning("Secrets Streamlit (service account): %s", e)
    return None


def gspread_client_from_streamlit() -> Any | None:
    try:
        import gspread
    except ImportError:
        return None
    info = service_account_info_from_streamlit_secrets()
    if not info:
        info = _service_account_info_from_env()
    if not info:
        return None
    try:
        return gspread.service_account_from_dict(info)
    except Exception as e:
        logger.warning("gspread (Streamlit): %s", e)
        return None


def load_best_worksheet_gspread(
    gc: Any, spreadsheet_id: str, role: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Abre a planilha, percorre **todas** as abas e escolhe a melhor pontuação para o papel.
    """
    sh = gc.open_by_key(spreadsheet_id)
    meta: dict[str, Any] = {
        "spreadsheet_title": sh.title,
        "spreadsheet_id": spreadsheet_id,
        "role": role,
    }
    best_df: pd.DataFrame | None = None
    best_score = -1.0
    best_sheet = ""
    best_header_row = -1
    errors: list[str] = []

    for ws in sh.worksheets():
        title = ws.title
        if str(title).lower().startswith("graf") or "chart" in str(title).lower():
            continue
        try:
            rows = ws.get_all_values()
        except Exception as e:
            errors.append(f"{title}: leitura {e}")
            continue
        try:
            df, sc, hdr = _best_df_from_values(rows, role)
        except Exception as e:
            errors.append(f"{title}: {e}")
            continue
        if sc > best_score:
            best_score = sc
            best_df = normalize_dataframe_columns(df)
            best_sheet = title
            best_header_row = hdr

    if best_df is None:
        raise ValueError(
            f"Nenhuma aba compatível com «{ROLE_LABELS.get(role, role)}» em «{sh.title}». "
            f"Resumo: {'; '.join(errors[:8])}"
        )
    meta.update(
        {
            "worksheet_title": best_sheet,
            "header_row_index": best_header_row,
            "score": round(best_score, 2),
            "method": "gspread",
        }
    )
    return best_df, meta


def _fetch_csv_bytes(sheet_id: str, gid: str) -> bytes | None:
    url = EXPORT_URL.format(sid=sheet_id, gid=gid)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PrevisaoVendas/1.1)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.read()
    except urllib.error.HTTPError:
        return None
    except Exception:
        return None


def load_best_worksheet_csv_public(
    spreadsheet_id: str,
    role: str,
    gid_hints: list[str] | None = None,
    max_gid_scan: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sem API: tenta export CSV por gid (sugestões primeiro, depois 0..max_gid_scan).
    """
    tried: list[str] = []
    gids_ordered: list[str] = []
    if gid_hints:
        gids_ordered.extend([g for g in gid_hints if g not in gids_ordered])
    for g in range(max_gid_scan + 1):
        s = str(g)
        if s not in gids_ordered:
            gids_ordered.append(s)

    best_df: pd.DataFrame | None = None
    best_score = -1.0
    best_gid = ""
    best_hdr = -1

    for gid in gids_ordered:
        raw = _fetch_csv_bytes(spreadsheet_id, gid)
        if raw is None:
            tried.append(f"{gid}:HTTP-?")
            continue
        tried.append(gid)
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                buf = io.BytesIO(raw)
                df0 = pd.read_csv(
                    buf, encoding=enc, on_bad_lines="skip", low_memory=False
                )
                break
            except Exception:
                df0 = None
        if df0 is None or df0.shape[1] < 2:
            continue
        rows = [df0.columns.tolist()] + df0.astype(str).values.tolist()
        try:
            df, sc, hdr = _best_df_from_values(rows, role)
        except Exception:
            continue
        if sc > best_score:
            best_score = sc
            best_df = normalize_dataframe_columns(df)
            best_gid = gid
            best_hdr = hdr

    if best_df is None:
        raise ValueError(
            f"CSV público: nenhum gid válido para {spreadsheet_id} "
            f"(tentados: {', '.join(tried[:20])}…). "
            "Use conta de serviço nas secrets ou torne a planilha pública."
        )
    meta = {
        "spreadsheet_id": spreadsheet_id,
        "role": role,
        "gid": best_gid,
        "header_row_index": best_hdr,
        "score": round(best_score, 2),
        "method": "csv_export",
    }
    return best_df, meta


def load_role_dataframe(
    gc: Any | None,
    spreadsheet_id: str,
    role: str,
    csv_gid_hints: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if gc is not None:
        return load_best_worksheet_gspread(gc, spreadsheet_id, role)
    return load_best_worksheet_csv_public(spreadsheet_id, role, gid_hints=csv_gid_hints)

# ----- inlined: report_html.py -----

import html as html_lib
import json
import math
from datetime import datetime
from typing import Any


def _json_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _cell_metric(
    x: Any, *, prec: int = 4, scale_million: bool = False
) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(xf) or math.isinf(xf):
        return "—"
    if scale_million:
        xf /= 1e6
    return f"{xf:.{prec}f}"


def _appendix_for_horizon_target(
    ba: dict[str, Any] | None,
    h: int,
    title: str,
    slug: str,
    vgv: bool,
    full_period_train: bool,
) -> tuple[str, str]:
    """
    Gera HTML do apêndice + script Plotly (ROC + barras MAE teste).
    slug: 'qtd' | 'valor'
    """
    hold_lbl = "in-sample (100%)" if full_period_train else "holdout (20%)"
    if not ba or not ba.get("rows"):
        empty = (
            f'<div class="glass-card p-5 mb-6"><h4 class="font-bold text-slate-800 mb-2">{title} — H={h}d</h4>'
            f'<p class="text-sm text-amber-800">Benchmark indisponível para este alvo.</p></div>'
        )
        return empty, ""

    rows_raw = list(ba["rows"])
    rows_raw.sort(key=lambda r: float(r.get("mae_val") or 9e99))
    thr = ba.get("threshold_y")
    thr_s = _cell_metric(thr, prec=2, scale_million=vgv)
    if vgv and thr is not None and thr_s != "—":
        thr_s = f"{thr_s} mi R$"

    win = html_lib.escape(str(ba.get("winner_label") or "—"))
    wnames = ba.get("winner_names") or []
    wweights = ba.get("winner_weights") or []
    wparts = []
    for i, n in enumerate(wnames):
        ww = wweights[i] if i < len(wweights) else None
        wparts.append(
            f"{html_lib.escape(str(n))}"
            + (f" ({float(ww):.3f})" if ww is not None else "")
        )
    blend_txt = " · ".join(wparts) if wparts else "—"

    reg_heads = (
        "<tr class='bg-slate-100 text-left'>"
        "<th class='p-2 border'>Modelo</th>"
        "<th class='p-2 border'>MAE sel. (val)</th>"
        "<th class='p-2 border'>MAE val</th>"
        "<th class='p-2 border'>MAE teste</th>"
        "<th class='p-2 border'>RMSE teste</th>"
        "<th class='p-2 border'>R² teste</th>"
        "<th class='p-2 border'>MedAE teste</th>"
        "<th class='p-2 border'>Var.expl. teste</th>"
        "<th class='p-2 border'>Erro máx. teste</th>"
        "</tr>"
    )
    reg_rows_html = [reg_heads]
    for r in rows_raw:
        nm = html_lib.escape(str(r.get("name", "")))
        rv = r.get("reg_val") or {}
        rt = r.get("reg_test") or {}
        reg_rows_html.append(
            "<tr>"
            f"<td class='p-2 border font-medium text-slate-800'>{nm}</td>"
            f"<td class='p-2 border'>{_cell_metric(r.get('mae_val'), prec=3, scale_million=vgv)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rv.get('MAE'), prec=3, scale_million=vgv)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rt.get('MAE'), prec=3, scale_million=vgv)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rt.get('RMSE'), prec=3, scale_million=vgv)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rt.get('R2'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rt.get('MedAE'), prec=3, scale_million=vgv)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rt.get('ExplainedVar'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(rt.get('MaxError'), prec=3, scale_million=vgv)}</td>"
            "</tr>"
        )

    bin_heads = (
        "<tr class='bg-slate-100'>"
        "<th class='p-2 border' rowspan='2'>Modelo</th>"
        "<th class='p-2 border text-center' colspan='5'>Validação (binária aux.)</th>"
        "<th class='p-2 border text-center' colspan='5'>Teste (binária aux.)</th>"
        "</tr>"
        "<tr class='bg-slate-50 text-xs'>"
        "<th class='p-2 border'>Acc</th><th class='p-2 border'>Prec</th><th class='p-2 border'>Rec</th>"
        "<th class='p-2 border'>F1</th><th class='p-2 border'>AUC</th>"
        "<th class='p-2 border'>Acc</th><th class='p-2 border'>Prec</th><th class='p-2 border'>Rec</th>"
        "<th class='p-2 border'>F1</th><th class='p-2 border'>AUC</th>"
        "</tr>"
    )
    bin_rows_html = [bin_heads]
    for r in rows_raw:
        nm = html_lib.escape(str(r.get("name", "")))
        bv = r.get("bin_val") or {}
        bt = r.get("bin_test") or {}
        bin_rows_html.append(
            "<tr>"
            f"<td class='p-2 border font-medium'>{nm}</td>"
            f"<td class='p-2 border'>{_cell_metric(bv.get('Accuracy'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bv.get('Precision'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bv.get('Recall'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bv.get('F1'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bv.get('ROC_AUC'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bt.get('Accuracy'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bt.get('Precision'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bt.get('Recall'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bt.get('F1'), prec=3)}</td>"
            f"<td class='p-2 border'>{_cell_metric(bt.get('ROC_AUC'), prec=3)}</td>"
            "</tr>"
        )

    unit_note = (
        "Valores de erro em milhões de R$ (VGV)."
        if vgv
        else "Valores de erro em unidades (quantidade)."
    )
    vgv_js = "true" if vgv else "false"

    roc_traces = ba.get("roc_traces") or []
    mae_bar: list[dict[str, Any]] = []
    for r in rows_raw:
        rt = r.get("reg_test") or {}
        mae_bar.append(
            {
                "name": str(r.get("name", "")),
                "mae": rt.get("MAE"),
            }
        )

    roc_json = _json_safe(roc_traces)
    mae_json = _json_safe(mae_bar)

    html_frag = f"""
    <div class="glass-card p-6 mb-8">
      <h4 class="text-lg font-bold text-slate-900 mb-2">{html_lib.escape(title)} — horizonte {h} dias</h4>
      <p class="text-xs text-slate-600 mb-3">
        Vencedor do benchmark (menor MAE na validação interna): <b>{win}</b><br/>
        Composição: {blend_txt}<br/>
        Limiar da tarefa binária auxiliar (mediana de <code>y</code> só no treino interno do benchmark): <b>{thr_s}</b>.
        Métricas de teste referem-se ao conjunto <b>{hold_lbl}</b>. {unit_note}
      </p>
      <div class="overflow-x-auto mb-4">
        <p class="text-xs font-bold text-slate-500 uppercase mb-1">Regressão (validação / teste)</p>
        <table class="text-xs border-collapse w-full min-w-[720px]">{"".join(reg_rows_html)}</table>
      </div>
      <div class="overflow-x-auto mb-4">
        <p class="text-xs font-bold text-slate-500 uppercase mb-1">
          Classificação auxiliar (real e predito &gt; limiar vs ≤ limiar; score ROC = valor predito contínuo)
        </p>
        <table class="text-xs border-collapse w-full min-w-[900px]">{"".join(bin_rows_html)}</table>
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <p class="text-xs font-bold text-slate-600 mb-1">Curvas ROC (teste — tarefa auxiliar)</p>
          <div id="roc_{slug}_{h}" style="height:400px;width:100%;"></div>
        </div>
        <div>
          <p class="text-xs font-bold text-slate-600 mb-1">MAE no teste — comparação entre modelos</p>
          <div id="maebar_{slug}_{h}" style="height:400px;width:100%;"></div>
        </div>
      </div>
    </div>
    """

    script = f"""
    (function() {{
      var roc = {roc_json};
      var palette = ['#6366f1','#10b981','#f59e0b','#ec4899','#8b5cf6','#06b6d4','#84cc16','#f97316','#64748b','#0ea5e9'];
      var traces = roc.map(function(t, i) {{
        return {{
          x: t.fpr || [], y: t.tpr || [], mode: 'lines',
          name: (t.name || '') + (t.auc != null ? ' (AUC=' + Number(t.auc).toFixed(3) + ')' : ''),
          line: {{ width: 1.5, color: palette[i % palette.length] }}
        }};
      }});
      traces.push({{ x: [0,1], y: [0,1], mode: 'lines', name: 'aleatório', line: {{ dash: 'dot', color: '#94a3b8' }} }});
      Plotly.newPlot('roc_{slug}_{h}', traces, {{
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(250,250,250,1)',
        font: {{ family: 'Plus Jakarta Sans', size: 11, color: '#475569' }},
        margin: {{ t: 36, l: 48, r: 12, b: 48 }},
        xaxis: {{ title: 'Taxa de falsos positivos', gridcolor: '#e2e8f0', zeroline: false }},
        yaxis: {{ title: 'Taxa de verdadeiros positivos', gridcolor: '#e2e8f0', zeroline: false }},
        showlegend: true, legend: {{ orientation: 'v', x: 1.02, y: 1 }}
      }}, {{ responsive: true }});

      var mb = {mae_json};
      var vgv = {vgv_js};
      var yv = mb.map(function(d) {{
        var v = d.mae;
        if (v == null || isNaN(v)) return 0;
        return vgv ? v / 1e6 : v;
      }});
      var ytitle = vgv ? 'MAE teste (mi R$)' : 'MAE teste (unid.)';
      Plotly.newPlot('maebar_{slug}_{h}', [{{
        type: 'bar',
        x: mb.map(function(d) {{ return d.name; }}),
        y: yv,
        marker: {{ color: '#4f46e5', opacity: 0.85 }}
      }}], {{
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(255,255,255,1)',
        font: {{ family: 'Plus Jakarta Sans', size: 11 }},
        margin: {{ t: 28, l: 52, r: 12, b: 120 }},
        xaxis: {{ tickangle: -55, automargin: true }},
        yaxis: {{ title: ytitle, gridcolor: '#f1f5f9' }}
      }}, {{ responsive: true }});
    }})();
    """
    return html_frag, script


def _build_appendix_html(
    horizontes: list[int],
    por_horizonte: dict[int, dict[str, Any]],
    full_period_train: bool,
) -> tuple[str, str]:
    parts: list[str] = []
    scripts: list[str] = []
    for h in horizontes:
        ph = por_horizonte[h]
        hq, sq = _appendix_for_horizon_target(
            ph["qtd"].get("benchmark_appendix"),
            h,
            "Volume (quantidade)",
            "qtd",
            False,
            full_period_train,
        )
        hv, sv = _appendix_for_horizon_target(
            ph["valor"].get("benchmark_appendix"),
            h,
            "VGV (valor)",
            "valor",
            True,
            full_period_train,
        )
        parts.append(
            f'<div class="mb-4"><h3 class="text-xl font-black text-slate-800 mb-3">Horizonte {h} dias</h3>{hq}{hv}</div>'
        )
        if sq:
            scripts.append(sq)
        if sv:
            scripts.append(sv)
    intro = f"""
    <div class="glass-card p-6 mb-8 border-l-4 border-indigo-500">
      <h2 class="text-xl font-black text-slate-900 mb-3">Apêndice técnico — benchmark de modelos</h2>
      <p class="text-sm text-slate-600 leading-relaxed">
        Todas as pipelines numéricas usam <b>MinMaxScaler (0, 1) com <code>clip=True</code></b>:
        na inferência, valores acima/abaixo do min–max do treino são <b>recortados</b> para o intervalo,
        evitando extrapolação absurda em modelos lineares (Ridge/ElasticNet) no holdout.
        A escolha do modelo operacional é feita pelo <b>menor MAE na validação temporal interna</b> (~últimos 8% antes do teste),
        incluindo <b>ensembles por média</b> e <b>ponderação inversa do MAE</b>, sem reutilizar o conjunto de teste nessa escolha.
        Os indicadores <b>acurácia, precisão, recall, F1 e ROC-AUC</b> referem-se a uma <b>tarefa binária auxiliar</b>:
        “alto” vs “baixo” relativamente à <b>mediana de <code>y</code> calculada apenas no treino</b> do benchmark
        (sem leakage do teste no limiar). O score da ROC é o valor predito contínuo da regressão.
        Gráficos de ROC e barras de MAE são calculados no mesmo período de <b>teste</b> indicado nas tabelas.
      </p>
    </div>
    """
    return intro + "\n".join(parts), "\n".join(scripts)


def render_dashboard(
    stats_base: dict[str, float],
    ticket_medio: float,
    conversao_funil: float,
    horizontes: list[int],
    por_horizonte: dict[int, dict[str, Any]],
    best_params_preview: dict[int, dict[str, dict[str, Any]]],
    out_path: str | None = None,
    full_period_train: bool = False,
) -> str:
    """
    por_horizonte[h]['qtd'|'valor']: dict com keys:
      metrics_val, metrics_test, importance_names, importance_vals,
      y_test, pred_test, dates_test (list str),
      pred_ultimo_dia, y_test_last (opcional)
    """
    funnel = [
        {"etapa": "Leads", "valor": int(stats_base["leads"])},
        {"etapa": "Agendamentos", "valor": int(stats_base["agend"])},
        {"etapa": "Visitas", "valor": int(stats_base["visit"])},
        {"etapa": "Pastas", "valor": int(stats_base["pastas"])},
        {"etapa": "Vendas", "valor": int(stats_base["vendas"])},
    ]
    funnel_json = _json_safe(funnel)
    hoje = datetime.now().strftime("%d/%m/%Y")
    subtreino = (
        "Treino em 100% do histórico · última fatia (~8%) só para ranquear o ensemble · "
        "previsão = próximos H dias a partir da última data nos ficheiros."
        if full_period_train
        else "Split 80%/20%: métricas e gráficos no holdout final (20%); "
        "pipeline operacional reajustado em 100% do histórico · Optuna (TSS, trials adaptativos) + ensemble."
    )
    lbl_mae_grande = "MAE in-sample (100%)" if full_period_train else "MAE teste (20%)"
    lbl_r2_grande = "R² in-sample (100%)" if full_period_train else "R² teste (20%)"
    lbl_val_box = (
        "Seleção ensemble (~8% final do histórico)"
        if full_period_train
        else "Seleção ensemble (~8% do bloco de treino 80%)"
    )
    lbl_graf_series = (
        "Real vs modelo — treino (~92%) | validação (~8%) · linha vertical na separação"
        if full_period_train
        else "Real vs modelo — treino (80%) | teste (20%) · linha vertical na separação"
    )
    lbl_chart_treino = "Treino (~92%)" if full_period_train else "Treino (80%)"
    lbl_chart_hold = "Validação (~8%)" if full_period_train else "Teste (20%)"
    lbl_tr_js = json.dumps(lbl_chart_treino, ensure_ascii=False)
    lbl_te_js = json.dumps(lbl_chart_hold, ensure_ascii=False)
    lbl_scatter = "Dispersão VGV (in-sample)" if full_period_train else "Dispersão VGV (holdout 20%)"
    lbl_modelo_esc = (
        "Modelo escolhido (MAE na fatia final — seleção de ensemble)"
        if full_period_train
        else "Modelo escolhido (MAE na validação interna do treino 80%)"
    )

    tabs_html = ""
    for i, h in enumerate(horizontes):
        cls = "active" if i == 0 else "text-slate-600 hover:bg-white/50"
        tabs_html += (
            f'<button type="button" onclick="openTab(\'horizon_{h}\', this)" '
            f'class="tab-btn {cls} px-6 py-3 rounded-xl font-bold text-sm uppercase tracking-wide">'
            f"{h} dias</button>"
        )

    blocks = ""
    for i, h in enumerate(horizontes):
        ph = por_horizonte[h]
        q = ph["qtd"]
        v = ph["valor"]
        active = "active" if i == 0 else ""

        y_tq = _json_safe(
            [float(x) for x in (q.get("chart_y_real") or q["y_test"])]
        )
        y_pq = _json_safe(
            [float(x) for x in (q.get("chart_y_pred") or q["pred_test"])]
        )
        y_tv = _json_safe(
            [float(x) for x in (v.get("chart_y_real") or v["y_test"])]
        )
        y_pv = _json_safe(
            [float(x) for x in (v.get("chart_y_pred") or v["pred_test"])]
        )
        x_dates = _json_safe(q.get("chart_dates") or q["dates_test"])
        split_q = int(q.get("chart_split_index") or 0)
        split_v = int(v.get("chart_split_index") or 0)
        y_tv_hold = _json_safe([float(x) for x in v["y_test"]])
        y_pv_hold = _json_safe([float(x) for x in v["pred_test"]])
        imp_n = _json_safe(v["importance_names"])
        imp_v = _json_safe([float(x) for x in v["importance_vals"]])

        mv_t = q["metrics_test"]
        mv_v = q["metrics_val"]
        mv_t_v = v["metrics_test"]
        mv_v_v = v["metrics_val"]

        params_q = best_params_preview.get(h, {}).get("qtd", {})
        params_v = best_params_preview.get(h, {}).get("valor", {})
        params_block = (
            f"<p class='text-sm text-slate-600 mb-2'>{lbl_modelo_esc}: "
            f"<b class='text-slate-900'>{q['model_label']}</b> (qtd) · "
            f"<b class='text-slate-900'>{v['model_label']}</b> (VGV)</p>"
            f"<pre class='text-xs overflow-auto max-h-48 bg-slate-50 p-3 rounded-lg border'>"
            f"<b>Hiperparâmetros LGBM (Optuna) — volume</b>\n{json.dumps(params_q, indent=2, ensure_ascii=False)[:2200]}"
            f"\n\n<b>Hiperparâmetros LGBM (Optuna) — VGV</b>\n{json.dumps(params_v, indent=2, ensure_ascii=False)[:2200]}"
            f"</pre>"
        )

        pred_q = float(q["pred_ultimo_dia"])
        pred_vv = float(v["pred_ultimo_dia"])

        blocks += f"""
            <div id="horizon_{h}" class="tab-content {active}">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    <div class="bg-gradient-to-br from-blue-600 to-indigo-800 p-8 rounded-2xl shadow-xl text-white relative overflow-hidden">
                        <p class="text-blue-100 text-sm font-semibold mb-1">Previsão acumulada — quantidade (próx. {h} dias)</p>
                        <p class="text-5xl font-black">{pred_q:,.1f} <span class="text-xl font-medium text-blue-200">unid.</span></p>
                        <p class="text-blue-200/80 text-xs mt-3">Próximos {h} dias a partir da última data nos dados (não usa dados futuros)</p>
                    </div>
                    <div class="bg-gradient-to-br from-emerald-500 to-teal-800 p-8 rounded-2xl shadow-xl text-white relative overflow-hidden">
                        <p class="text-emerald-100 text-sm font-semibold mb-1">Previsão — Valor Real de Venda (próx. {h} dias)</p>
                        <p class="text-5xl font-black"><span class="text-2xl text-emerald-200">R$</span> {pred_vv/1e6:,.2f} <span class="text-xl text-emerald-200">mi</span></p>
                        <p class="text-emerald-200/80 text-xs mt-3">Próximos {h} dias a partir da última data nos dados (não usa dados futuros)</p>
                    </div>
                </div>

                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                    <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                        <div class="text-slate-400 text-xs font-bold uppercase">{lbl_mae_grande} — Qtd</div>
                        <div class="text-xl font-black text-slate-800">{mv_t['MAE']:.2f}</div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                        <div class="text-slate-400 text-xs font-bold uppercase">{lbl_r2_grande} — Qtd</div>
                        <div class="text-xl font-black text-indigo-600">{(mv_t['R2']*100):.1f}%</div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                        <div class="text-slate-400 text-xs font-bold uppercase">{lbl_mae_grande} — VGV</div>
                        <div class="text-xl font-black text-slate-800">{mv_t_v['MAE']/1e6:.2f} mi</div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                        <div class="text-slate-400 text-xs font-bold uppercase">{lbl_r2_grande} — VGV</div>
                        <div class="text-xl font-black text-emerald-600">{(mv_t_v['R2']*100):.1f}%</div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
                    <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                        <div class="text-slate-500 text-xs font-bold uppercase mb-1">{lbl_val_box} — Volume</div>
                        <div class="text-sm">MAE {mv_v['MAE']:.2f} · RMSE {mv_v['RMSE']:.2f} · sMAPE {mv_v['sMAPE']:.1f}% · MAPE (dias c/ y≥0,5) {mv_v['MAPE']:.1f}%</div>
                    </div>
                    <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                        <div class="text-slate-500 text-xs font-bold uppercase mb-1">{lbl_val_box} — VGV</div>
                        <div class="text-sm">MAE {mv_v_v['MAE']/1e6:.2f} mi · RMSE {mv_v_v['RMSE']/1e6:.2f} mi · sMAPE {mv_v_v['sMAPE']:.1f}% · MAPE (y altos) {mv_v_v['MAPE']:.1f}%</div>
                    </div>
                </div>

                <div class="glass-card p-6 mb-6">
                    <h4 class="text-lg font-bold text-slate-800 mb-3">Hiperparâmetros (LGBM — Optuna + ensemble)</h4>
                    {params_block}
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <div class="glass-card p-6">
                        <h4 class="text-lg font-bold text-slate-800 mb-4">{lbl_graf_series} — Volume</h4>
                        <div id="chart_q_{h}" style="height:380px;width:100%;"></div>
                    </div>
                    <div class="glass-card p-6">
                        <h4 class="text-lg font-bold text-slate-800 mb-4">{lbl_graf_series} — VGV</h4>
                        <div id="chart_v_{h}" style="height:380px;width:100%;"></div>
                    </div>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                    <div class="glass-card p-6 lg:col-span-2">
                        <h4 class="text-lg font-bold text-slate-800 mb-4">Importância (proxy LGBM — VGV)</h4>
                        <div id="chart_i_{h}" style="height:400px;width:100%;"></div>
                    </div>
                    <div class="glass-card p-6 bg-slate-900 text-white">
                        <h4 class="text-lg font-bold mb-4">{lbl_scatter}</h4>
                        <div id="chart_scatter_{h}" style="height:400px;width:100%;"></div>
                    </div>
                </div>
                <script>
                (function() {{
                    var layoutBase = {{
                        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {{family: 'Plus Jakarta Sans', color: '#64748b'}},
                        margin: {{t:52,l:50,r:20,b:60}},
                        xaxis: {{gridcolor: '#f1f5f9', zerolinecolor: '#e2e8f0'}},
                        yaxis: {{gridcolor: '#f1f5f9', zerolinecolor: '#e2e8f0'}},
                        legend: {{orientation: 'h', y: -0.2}}
                    }};
                    var xdt = {x_dates};
                    var splitIdxQ = {split_q};
                    var splitIdxV = {split_v};
                    var lblTr = {lbl_tr_js};
                    var lblTe = {lbl_te_js};
                    function layoutTrainTest(splitIdx) {{
                        var lo = Object.assign({{}}, layoutBase, {{
                            xaxis: Object.assign({{}}, layoutBase.xaxis, {{tickangle: -35}})
                        }});
                        if (splitIdx > 0 && splitIdx < xdt.length) {{
                            var sd = xdt[splitIdx];
                            lo.shapes = [{{
                                type: 'line', xref: 'x', yref: 'paper',
                                x0: sd, x1: sd, y0: 0, y1: 1,
                                line: {{ color: '#64748b', width: 2, dash: '4px,3px' }}
                            }}];
                            var imt = Math.max(0, Math.floor(splitIdx / 2));
                            var ime = Math.min(xdt.length - 1, splitIdx + Math.floor((xdt.length - splitIdx) / 2));
                            lo.annotations = [
                                {{ x: xdt[imt], y: 1.07, xref: 'x', yref: 'paper', text: lblTr, showarrow: false,
                                  font: {{size: 11, color: '#64748b'}}, xanchor: 'center' }},
                                {{ x: xdt[ime], y: 1.07, xref: 'x', yref: 'paper', text: lblTe, showarrow: false,
                                  font: {{size: 11, color: '#64748b'}}, xanchor: 'center' }}
                            ];
                        }}
                        return lo;
                    }}
                    Plotly.newPlot('chart_q_{h}', [
                        {{ x: xdt, y: {y_tq}, name: 'Real', mode: 'lines', line: {{color: '#94a3b8', width: 2}} }},
                        {{ x: xdt, y: {y_pq}, name: 'Modelo', mode: 'lines', line: {{color: '#6366f1', width: 2}} }}
                    ], Object.assign({{}}, layoutTrainTest(splitIdxQ)), {{responsive: true}});
                    Plotly.newPlot('chart_v_{h}', [
                        {{ x: xdt, y: {y_tv}, name: 'Real', mode: 'lines', line: {{color: '#94a3b8', width: 2}} }},
                        {{ x: xdt, y: {y_pv}, name: 'Modelo', mode: 'lines', line: {{color: '#10b981', width: 2}} }}
                    ], Object.assign({{}}, layoutTrainTest(splitIdxV)), {{responsive: true}});
                    Plotly.newPlot('chart_i_{h}', [
                        {{ x: {imp_v}, y: {imp_n}, type: 'bar', orientation: 'h', marker: {{color: '#8b5cf6'}} }}
                    ], Object.assign({{}}, layoutBase, {{yaxis: {{automargin: true, tickfont: {{size: 10}}}}, margin: {{l: 200, t: 10, b: 40}}}}), {{responsive: true}});
                    var ytv = {y_tv_hold}, ypv = {y_pv_hold};
                    var maxv = Math.max.apply(null, ytv.concat(ypv).concat([1]));
                    Plotly.newPlot('chart_scatter_{h}', [
                        {{ x: ytv, y: ypv, mode: 'markers', marker: {{color: '#38bdf8', size: 7, opacity: 0.65}}, name: 'Pontos' }},
                        {{ x: [0, maxv], y: [0, maxv], mode: 'lines', line: {{color: '#f43f5e', dash: 'dash'}}, name: 'y=x' }}
                    ], Object.assign({{}}, layoutBase, {{
                        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
                        font: {{family: 'Plus Jakarta Sans', color: '#94a3b8'}},
                        xaxis: {{title: 'Real', gridcolor: '#334155'}},
                        yaxis: {{title: 'Predito', gridcolor: '#334155'}}
                    }}), {{responsive: true}});
                }})();
                </script>
            </div>
        """

    appendix_html, appendix_scripts = _build_appendix_html(
        horizontes, por_horizonte, full_period_train
    )
    tech_blurb = (
        " · Features MinMax (0–1) · benchmark multi-modelo/ensembles (escolha por MAE na validação interna)"
    )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Previsão de vendas — ML</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    body {{ background: #e2e8f0; color: #0f172a; }}
    .glass-card {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 1rem;
      box-shadow: 0 4px 6px -1px rgba(15,23,42,0.08), 0 10px 24px -8px rgba(15,23,42,0.1);
    }}
    .tab-content {{ display: none; opacity: 0; transition: opacity 0.25s ease; }}
    .tab-content.active {{ display: block; opacity: 1; }}
    .tab-btn.active {{
      background: #2563eb; color: white;
      box-shadow: 0 4px 14px rgba(37,99,235,0.35);
    }}
    .section-pane {{ display: none; }}
    .section-pane.active {{ display: block; }}
    .sec-main-btn {{ background: #e2e8f0; color: #475569; transition: all 0.2s; }}
    .sec-main-btn.active {{
      background: #0f172a; color: white;
      box-shadow: 0 4px 14px rgba(15,23,42,0.2);
    }}
  </style>
</head>
<body class="p-4 md:p-8">
  <div class="max-w-7xl mx-auto">
    <header class="glass-card p-6 mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <h1 class="text-2xl md:text-3xl font-extrabold text-slate-900">Previsão de vendas</h1>
        <p class="text-slate-500 mt-1">{subtreino}{tech_blurb} · STL rolante · LGBM fair/quantil · blend médio+quantil alto · Ridge em TS · Optuna (TSS)</p>
      </div>
      <span class="text-sm font-semibold text-slate-600 bg-slate-100 px-4 py-2 rounded-full">{hoje}</span>
    </header>

    <div class="flex flex-wrap gap-2 mb-6 justify-center">
      <button type="button" class="sec-main-btn active px-5 py-2.5 rounded-xl font-bold text-sm uppercase tracking-wide"
        data-sec="previsoes" onclick="openSection('previsoes', this)">Previsões</button>
      <button type="button" class="sec-main-btn px-5 py-2.5 rounded-xl font-bold text-sm uppercase tracking-wide text-slate-600"
        data-sec="apendice" onclick="openSection('apendice', this)">Apêndice técnico</button>
    </div>

    <div id="sec-previsoes" class="section-pane active">
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <div class="glass-card p-5 border-t-4 border-slate-800">
        <p class="text-xs font-bold text-slate-400 uppercase">Leads</p>
        <p class="text-2xl font-black">{stats_base["leads"]:,.0f}</p>
      </div>
      <div class="glass-card p-5 border-t-4 border-blue-500">
        <p class="text-xs font-bold text-slate-400 uppercase">Conversão (vendas/leads)</p>
        <p class="text-2xl font-black">{conversao_funil:.2f}%</p>
      </div>
      <div class="glass-card p-5 border-t-4 border-emerald-500">
        <p class="text-xs font-bold text-slate-400 uppercase">VGV histórico</p>
        <p class="text-2xl font-black">R$ {stats_base["vgv"]/1e6:.1f}M</p>
      </div>
      <div class="glass-card p-5 border-t-4 border-amber-500">
        <p class="text-xs font-bold text-slate-400 uppercase">Ticket médio</p>
        <p class="text-2xl font-black">R$ {ticket_medio/1e3:,.0f}k</p>
      </div>
    </div>

    <div class="flex justify-center mb-8">
      <div class="inline-flex flex-wrap gap-2 bg-white border border-slate-200 p-2 rounded-2xl shadow-sm">
        <span class="self-center text-xs font-bold text-slate-500 px-2">Horizonte:</span>
        {tabs_html}
      </div>
    </div>

    <div class="glass-card p-6 mb-8">
      <h3 class="text-lg font-bold text-slate-800 mb-3">Funil agregado (volume de registros)</h3>
      <div id="global_funnel" style="height:320px;width:100%;"></div>
    </div>

    {blocks}
    </div>

    <div id="sec-apendice" class="section-pane">
    {appendix_html}
    </div>
  </div>

  <script>
    function openSection(secId, btn) {{
      document.querySelectorAll('.section-pane').forEach(function(p) {{
        p.classList.toggle('active', p.id === 'sec-' + secId);
      }});
      document.querySelectorAll('.sec-main-btn').forEach(function(b) {{
        var on = b.getAttribute('data-sec') === secId;
        b.classList.toggle('active', on);
        b.classList.toggle('text-slate-600', !on);
      }});
      window.dispatchEvent(new Event('resize'));
      if (secId === 'apendice' && typeof Plotly !== 'undefined') {{
        setTimeout(function() {{
          document.querySelectorAll('#sec-apendice .js-plotly-plot').forEach(function(gd) {{
            try {{ Plotly.Plots.resize(gd); }} catch (e) {{}}
          }});
        }}, 100);
      }}
    }}
    function openTab(tabId, el) {{
      document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
      document.querySelectorAll('.tab-btn').forEach(function(b) {{
        b.classList.remove('active');
        b.classList.add('text-slate-600', 'hover:bg-white/50');
      }});
      document.getElementById(tabId).classList.add('active');
      el.classList.add('active');
      el.classList.remove('text-slate-600', 'hover:bg-white/50');
      window.dispatchEvent(new Event('resize'));
    }}
    var dataFunnel = {funnel_json};
    Plotly.newPlot('global_funnel', [{{
      type: 'funnel', y: dataFunnel.map(function(d) {{ return d.etapa; }}), x: dataFunnel.map(function(d) {{ return d.valor; }}),
      textinfo: 'value+percent initial',
      marker: {{ color: ['#0f172a','#3b82f6','#6366f1','#8b5cf6','#10b981'] }}
    }}], {{
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {{ family: 'Plus Jakarta Sans', color: '#475569' }},
      margin: {{ l: 120, r: 20, t: 10, b: 10 }}
    }}, {{ responsive: true }});
    {appendix_scripts}
  </script>
</body>
</html>"""

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


_ROOT = Path(__file__).resolve().parent

import base64
import html
import streamlit as st

# --- Identidade visual (alinhado à ficha Vendas RJ) ---
COR_AZUL_ESC = "#04428f"
COR_VERMELHO = "#cb0935"
COR_VERMELHO_ESCURO = "#9e0828"
COR_BORDA = "#eef2f6"
COR_INPUT_BG = "#f0f2f6"
COR_TEXTO_MUTED = "#64748b"

LOGO_TOPO_ARQUIVO = "502.57_LOGO DIRECIONAL_V2F-01.png"
FAVICON_ARQUIVO = "502.57_LOGO D_COR_V3F.png"
URL_LOGO_DIRECIONAL_FALLBACK = (
    "https://logodownload.org/wp-content/uploads/2021/04/direcional-engenharia-logo.png"
)

BG_HERO_URL = (
    "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab"
    "?auto=format&fit=crop&w=1920&q=80"
)


def _hex_rgb_triplet(hex_color: str) -> str:
    x = (hex_color or "").strip().lstrip("#")
    if len(x) != 6:
        return "4, 66, 143"
    return f"{int(x[0:2], 16)}, {int(x[2:4], 16)}, {int(x[4:6], 16)}"


RGB_AZUL_CSS = _hex_rgb_triplet(COR_AZUL_ESC)
RGB_VERMELHO_CSS = _hex_rgb_triplet(COR_VERMELHO)

# --- Decisões automáticas (sem input do utilizador) ---
# Optuna: 0 = número de trials escalado ao tamanho da série em train_eval.
OPTUNA_TRIALS_AUTO = 0
# Top-K no super-ensemble / candidatos: 6 equilibra diversidade e custo (benchmark escolhe melhor combinação).
BLEND_TOP_K_FIXO = 6
RANDOM_SEED = 42
# Holdout 20% nas métricas do relatório (mais fiável que 100% in-sample).
FULL_PERIOD_TRAIN_FIXO = False
# tqdm no terminal pouco útil no Cloud — desativado.
SHOW_ML_PROGRESS = False

DEFAULT_SPREADSHEET_IDS: dict[str, str] = {
    "formulario_previsao": "1lBliB3AjR5vJyRy9SoDi6DQOA9x5LC5wYfNNo5cz0bE",
    "vendas": "1jb6bYBAlslele2V1CTUVHPNR55SJEiRojN1jGX3gJ4w",
    "pastas": "1GC6GQytaDhjVslJ7seNBRfuE9QKulOU8dCeZsDPy0_g",
    "leads": "1w-htBl8UxwqgFU1bspweBLW54-mzJwFBkcYC0Qib5Wk",
    "agendamentos": "1TE0J29jxASqd3MbgV6Frwn3kAzlCDf5gzX3I_tmVy1Y",
}

CSV_GID_HINTS: dict[str, list[str]] = {
    "formulario_previsao": ["155389951", "0"],
    "agendamentos": ["1490757093", "0"],
    "vendas": ["0"],
    "pastas": ["0"],
    "leads": ["0"],
}


def _resolver_png_raiz(nome: str) -> Path | None:
    for base in (_ROOT, _ROOT.parent):
        p = base / nome
        if p.is_file():
            return p
    for name in ("logo_direcional.png", "logo.png"):
        p = _ROOT / "assets" / name
        if p.is_file():
            return p
    return None


def _logo_url_secrets() -> str | None:
    try:
        b = st.secrets.get("branding")
        if isinstance(b, dict):
            u = (b.get("LOGO_URL") or "").strip()
            if u:
                return u
    except Exception:
        pass
    return None


def _exibir_logo_topo() -> None:
    path = _resolver_png_raiz(LOGO_TOPO_ARQUIVO)
    url = _logo_url_secrets()
    try:
        if path:
            ext = path.suffix.lower().lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            src = f"data:{mime};base64,{b64}"
        elif url:
            src = html.escape(url)
        else:
            src = html.escape(URL_LOGO_DIRECIONAL_FALLBACK)
        st.markdown(
            f'<div class="pv-logo-wrap"><img src="{src}" alt="Direcional Engenharia" /></div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            f'<div class="pv-logo-wrap"><img src="{html.escape(URL_LOGO_DIRECIONAL_FALLBACK)}" alt="Direcional" /></div>',
            unsafe_allow_html=True,
        )


def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
@keyframes pvFadeIn {{
  from {{ opacity: 0; transform: translateY(14px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pvShimmer {{
  0% {{ background-position: 0% 50%; }}
  100% {{ background-position: 200% 50%; }}
}}
html, body, [class*="css"] {{
  font-family: 'Inter', system-ui, sans-serif;
  color: #1e293b;
}}
.stApp, [data-testid="stApp"] {{
  background:
    linear-gradient(135deg, rgba({RGB_AZUL_CSS}, 0.85) 0%, rgba(30, 58, 95, 0.55) 42%, rgba({RGB_VERMELHO_CSS}, 0.18) 72%, rgba(15, 23, 42, 0.48) 100%),
    url("{BG_HERO_URL}") center / cover no-repeat !important;
}}
[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
/* Faixa central legível: fundo claro sólido sobre o gradiente/imagem */
[data-testid="stMain"] {{
  background: rgba(255, 255, 255, 0.96) !important;
  background-color: #f8fafc !important;
  border-radius: 20px !important;
  margin: 0.5rem 0.75rem 1.25rem 0.75rem !important;
  padding: 0.5rem 0 1rem 0 !important;
  box-shadow: 0 4px 24px rgba(15, 23, 42, 0.12) !important;
  box-sizing: border-box !important;
}}
[data-testid="stHeader"], [data-testid="stDecoration"] {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
[data-testid="stToolbar"] {{
  background: rgba(255, 255, 255, 0.92) !important;
  backdrop-filter: blur(8px);
  border-radius: 12px !important;
  color: #0f172a !important;
  padding: 4px 8px !important;
}}
[data-testid="stToolbar"] button {{ color: #0f172a !important; }}
.main .block-container {{
  max-width: 1040px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 1.5rem 1.75rem 2rem 1.75rem !important;
  background: #ffffff !important;
  backdrop-filter: none;
  border-radius: 18px !important;
  border: 1px solid #e2e8f0 !important;
  box-shadow:
    0 1px 3px rgba({RGB_AZUL_CSS}, 0.06),
    0 12px 32px -8px rgba(15, 23, 42, 0.08) !important;
  animation: pvFadeIn 0.65s ease both;
}}
/* Texto nativo Streamlit sempre escuro sobre o cartão branco */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {{
  color: #1e293b !important;
}}
[data-testid="stCaptionContainer"] {{
  color: #475569 !important;
}}
div[data-testid="stExpander"] {{
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
}}
div[data-testid="stExpander"] summary {{
  color: #0f172a !important;
}}
div[data-testid="stAlert"] {{
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  color: #0f172a !important;
}}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] div {{
  color: #1e293b !important;
}}
[data-testid="stDataFrame"], [data-testid="stDataFrame"] > div {{
  background: #ffffff !important;
}}
.pv-logo-wrap {{
  text-align: center;
  padding: 0.15rem 0 0.5rem 0;
}}
.pv-logo-wrap img {{
  max-height: 64px;
  width: auto;
  max-width: min(260px, 88vw);
  object-fit: contain;
}}
.pv-hero-title {{
  font-family: 'Montserrat', sans-serif;
  font-size: clamp(1.28rem, 3.2vw, 1.62rem);
  font-weight: 900;
  color: {COR_AZUL_ESC};
  text-align: center;
  margin: 0 0 0.4rem 0;
  letter-spacing: -0.02em;
  line-height: 1.2;
}}
.pv-hero-sub {{
  text-align: center;
  color: #475569;
  font-size: 0.94rem;
  line-height: 1.55;
  margin: 0 0 0.75rem 0;
}}
.pv-bar-wrap {{ margin: 0.75rem 0 1rem 0; }}
.pv-bar {{
  height: 4px;
  width: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, {COR_AZUL_ESC}, {COR_VERMELHO}, {COR_AZUL_ESC});
  background-size: 200% 100%;
  animation: pvShimmer 4s ease-in-out infinite alternate;
}}
.pv-status-pill {{
  display: inline-block;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-bottom: 0.85rem;
}}
.pv-status-ok {{ background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
.pv-status-warn {{ background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
.stButton > button[kind="primary"] {{
  background: linear-gradient(180deg, {COR_VERMELHO} 0%, {COR_VERMELHO_ESCURO} 100%) !important;
  color: #fff !important;
  border: none !important;
  font-weight: 700 !important;
  font-family: 'Montserrat', sans-serif !important;
  border-radius: 12px !important;
  padding: 0.65rem 2rem !important;
  min-height: 3rem !important;
  font-size: 1rem !important;
  letter-spacing: 0.02em;
}}
.stButton > button[kind="primary"]:hover {{
  box-shadow: 0 8px 22px -6px rgba({RGB_VERMELHO_CSS}, 0.45) !important;
}}
div.metric-card {{
  background: linear-gradient(180deg, #fff 0%, #fafbfc 100%);
  border-radius: 14px;
  padding: 0.95rem 1.1rem;
  border: 1px solid {COR_BORDA};
  box-shadow: 0 2px 10px rgba({RGB_AZUL_CSS}, 0.06);
  border-left: 3px solid {COR_AZUL_ESC};
  margin-bottom: 0.5rem;
}}
div.metric-card h4 {{
  color: {COR_TEXTO_MUTED};
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin: 0 0 0.3rem 0;
  font-weight: 700;
  font-family: 'Montserrat', sans-serif;
}}
div.metric-card .val {{
  color: #0f172a;
  font-size: 1.28rem;
  font-weight: 800;
  font-family: 'Montserrat', sans-serif;
}}
h2, h3 {{ font-family: 'Montserrat', sans-serif !important; color: {COR_AZUL_ESC} !important; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid {COR_BORDA}; }}
.stDownloadButton > button {{
  border-radius: 12px !important;
  border: 2px solid {COR_AZUL_ESC} !important;
  color: {COR_AZUL_ESC} !important;
  font-weight: 600 !important;
}}
.pv-foot {{
  text-align: center;
  font-size: 0.78rem;
  color: {COR_TEXTO_MUTED};
  margin-top: 1.25rem;
  padding-top: 0.75rem;
  border-top: 1px solid {COR_BORDA};
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _data_source_config() -> tuple[dict[str, str], dict[str, list[str]]]:
    ids = dict(DEFAULT_SPREADSHEET_IDS)
    hints = {k: list(v) for k, v in CSV_GID_HINTS.items()}
    try:
        if "spreadsheet_ids" in st.secrets:
            for k, v in dict(st.secrets["spreadsheet_ids"]).items():
                if k in ids:
                    ids[k] = str(v).strip()
        if "sheets" in st.secrets:
            for k, v in dict(st.secrets["sheets"]).items():
                if k not in ids:
                    continue
                if isinstance(v, (list, tuple)) and len(v) >= 1:
                    ids[k] = str(v[0]).strip()
                    if len(v) >= 2:
                        g = str(v[1]).strip()
                        cur = hints.get(k, [])
                        hints[k] = [g] + [x for x in cur if x != g]
        if "csv_gid_hints" in st.secrets:
            for k, v in dict(st.secrets["csv_gid_hints"]).items():
                if k in hints and isinstance(v, list):
                    hints[k] = [str(x) for x in v]
    except Exception:
        pass
    return ids, hints


@st.cache_data(ttl=180, show_spinner=False)
def _load_one_role_cached(
    use_sa: bool,
    spreadsheet_id: str,
    role_key: str,
    hints_tuple: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:

    gc = gspread_client_from_streamlit() if use_sa else None
    return load_role_dataframe(
        gc,
        spreadsheet_id,
        role_key,
        csv_gid_hints=list(hints_tuple) if hints_tuple else None,
    )


def build_data_bundle() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], bool]:

    ids, hints = _data_source_config()
    gc = gspread_client_from_streamlit()
    use_sa = gc is not None

    out: dict[str, pd.DataFrame] = {}
    metas: dict[str, dict[str, Any]] = {}
    mapping = [
        ("leads", "leads"),
        ("agendamentos", "agendamentos"),
        ("pastas", "pastas"),
        ("vendas", "vendas"),
        ("formulario_previsao", "formulario_previsao"),
    ]
    errors: list[str] = []
    for key, cfg_key in mapping:
        human = ROLE_LABELS.get(cfg_key, cfg_key)
        sid = ids[cfg_key]
        htuple = tuple(hints.get(cfg_key, []))
        try:
            df, meta = _load_one_role_cached(use_sa, sid, cfg_key, htuple)
            out[key] = df
            metas[cfg_key] = meta
        except Exception as e:
            errors.append(f"**{human}** (`{sid[:12]}…`): {e}")
    if errors:
        modo = (
            "**Conta de serviço:** confirme partilha com o e-mail `client_email` do JSON."
            if use_sa
            else "**Modo público:** partilhe as planilhas por ligação (Leitor) ou configure `GOOGLE_SERVICE_ACCOUNT_JSON`."
        )
        st.error("Não foi possível carregar todas as fontes.\n\n" + modo + "\n\n" + "\n\n".join(errors))
        st.stop()
    return out, metas, use_sa


def run_training_pipeline(
    dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], float, float, dict[int, dict[str, Any]], dict[int, dict[str, Any]], str]:
    df_master = build_daily_master(dfs, show_progress=SHOW_ML_PROGRESS)

    stats_base = {
        "leads": float(df_master["vol_leads"].sum()),
        "agend": float(df_master["vol_agend"].sum()),
        "visit": float(df_master["vol_visit"].sum()),
        "pastas": float(df_master["vol_pastas"].sum()),
        "vendas": float(df_master["target_qtd"].sum()),
        "vgv": float(df_master["target_valor"].sum()),
    }
    ticket = stats_base["vgv"] / stats_base["vendas"] if stats_base["vendas"] > 0 else 0.0
    conv = (stats_base["vendas"] / stats_base["leads"] * 100) if stats_base["leads"] > 0 else 0.0

    horizontes = [3, 7, 30]
    por_horizonte: dict[int, dict[str, Any]] = {}
    best_params_preview: dict[int, dict[str, dict[str, Any]]] = {}

    progress = st.progress(0, text="A preparar modelos…")
    total_steps = len(horizontes) * 2
    step = 0

    for h in horizontes:
        Xq, yq = build_xy_for_horizon(df_master, "target_qtd", h)
        Xv, yv = build_xy_for_horizon(df_master, "target_valor", h)

        progress.progress(
            (step + 0.5) / total_steps,
            text=f"Volume — {h} dias (Optuna + benchmark)…",
        )
        rq = train_one_target(
            Xq,
            yq,
            horizon=h,
            target_name="qtd",
            n_trials=OPTUNA_TRIALS_AUTO,
            random_state=RANDOM_SEED,
            optuna_seed=RANDOM_SEED,
            blend_top_k=BLEND_TOP_K_FIXO,
            show_progress=SHOW_ML_PROGRESS,
            full_period_train=FULL_PERIOD_TRAIN_FIXO,
        )
        step += 1
        progress.progress(step / total_steps, text=f"VGV — {h} dias…")

        rv = train_one_target(
            Xv,
            yv,
            horizon=h,
            target_name="valor",
            n_trials=OPTUNA_TRIALS_AUTO,
            random_state=RANDOM_SEED,
            optuna_seed=RANDOM_SEED + 1,
            blend_top_k=BLEND_TOP_K_FIXO,
            show_progress=SHOW_ML_PROGRESS,
            full_period_train=FULL_PERIOD_TRAIN_FIXO,
        )
        step += 1

        pred_q = predict_last_row(df_master, rq.pipeline, h)
        pred_v = predict_last_row(df_master, rv.pipeline, h)

        def pack(r: Any, pred_last: float) -> dict[str, Any]:
            return {
                "metrics_val": r.metrics_val,
                "metrics_test": r.metrics_test,
                "importance_names": r.importance.index.tolist() if len(r.importance) else [],
                "importance_vals": r.importance.values.tolist() if len(r.importance) else [],
                "y_test": r.y_test.tolist(),
                "pred_test": r.y_test_pred.tolist(),
                "dates_test": [d.strftime("%Y-%m-%d") for d in r.dates_test],
                "pred_ultimo_dia": pred_last,
                "model_label": r.model_label,
                "full_period_train": r.full_period_train,
                "chart_dates": r.chart_dates,
                "chart_y_real": r.chart_y_real,
                "chart_y_pred": r.chart_y_pred,
                "chart_split_index": r.chart_split_index,
                "benchmark_appendix": r.benchmark_appendix,
            }

        por_horizonte[h] = {"qtd": pack(rq, pred_q), "valor": pack(rv, pred_v)}
        best_params_preview[h] = {"qtd": rq.best_params, "valor": rv.best_params}

    progress.progress(1.0, text="A gerar relatório…")
    html_out = render_dashboard(
        stats_base,
        ticket,
        conv,
        horizontes,
        por_horizonte,
        best_params_preview,
        out_path=None,
        full_period_train=FULL_PERIOD_TRAIN_FIXO,
    )
    return stats_base, ticket, conv, por_horizonte, best_params_preview, html_out


def main() -> None:
    fav = _resolver_png_raiz(FAVICON_ARQUIVO)
    st.set_page_config(
        page_title="Previsão de vendas | Direcional",
        page_icon=str(fav) if fav else "📊",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    _exibir_logo_topo()

    st.markdown(
        f'<p class="pv-hero-title">Previsão de vendas</p>'
        f'<p class="pv-hero-sub">Consolidação analítica com validação temporal, otimização automática de hiperparâmetros '
        f'e seleção de modelo entre vários algoritmos. As planilhas são lidas na íntegra: <strong>todas as abas</strong> '
        f"são analisadas e a folha correta é escolhida por compatibilidade de colunas.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pv-bar-wrap"><div class="pv-bar"></div></div>', unsafe_allow_html=True)

    try:

        sa_ok = gspread_client_from_streamlit() is not None
    except Exception:
        sa_ok = False

    if sa_ok:
        st.markdown(
            '<div class="pv-status-pill pv-status-ok">Conta de serviço Google ativa · API com varredura de abas</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pv-status-pill pv-status-warn">Modo CSV público · Configure GOOGLE_SERVICE_ACCOUNT_JSON nas secrets '
            "para leitura privada via API</div>",
            unsafe_allow_html=True,
        )

    st.caption(
        f"Configuração automática: Optuna com trials adaptados ao histórico, ensemble até **{BLEND_TOP_K_FIXO}** candidatos, "
        f"holdout **20%** nas métricas, semente **{RANDOM_SEED}**."
    )

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        run_btn = st.button("Gerar previsões", type="primary", use_container_width=True, key="pv_gerar")

    with st.expander("Suporte e manutenção"):
        st.markdown(
            "• **Secrets:** `GOOGLE_SERVICE_ACCOUNT_JSON` (JSON completo) ou secção `google_service_account` · "
            "`spreadsheet_ids` / `sheets` / `csv_gid_hints` opcionais.\n\n"
            "• **Logo:** coloque `502.57_LOGO DIRECIONAL_V2F-01.png` na raiz do repositório ou defina `branding.LOGO_URL`.\n\n"
            "• **Favicon:** `502.57_LOGO D_COR_V3F.png` na raiz."
        )
        if st.button("Limpar cache das planilhas", key="pv_clear_cache"):
            _load_one_role_cached.clear()
            st.success("Cache limpo. Volte a clicar em **Gerar previsões**.")

    if "resultado" not in st.session_state:
        st.session_state.resultado = None

    if run_btn:
        with st.spinner("A carregar e analisar as planilhas…"):
            dfs, sheet_metas, used_sa = build_data_bundle()
        st.session_state["sheet_metas"] = sheet_metas
        st.session_state["used_service_account"] = used_sa
        try:
            with st.spinner(
                "A treinar modelos e gerar o relatório. Este passo pode demorar vários minutos — não feche a página."
            ):
                stats_base, ticket, conv, por_h, _bpp, html_out = run_training_pipeline(dfs)
            st.session_state.resultado = {
                "stats_base": stats_base,
                "ticket": ticket,
                "conv": conv,
                "por_h": por_h,
                "html": html_out,
                "full_train": FULL_PERIOD_TRAIN_FIXO,
                "sheet_metas": sheet_metas,
                "used_sa": used_sa,
            }
            st.success("Previsões geradas com sucesso.")
        except Exception as e:
            st.error(f"Erro na execução: {e}")
            st.exception(e)

    res = st.session_state.resultado
    if res is None:
        st.info(
            "Clique em **Gerar previsões** para sincronizar as bases Google, treinar os modelos e obter o relatório. "
            "Recomenda-se conta de serviço com acesso de leitura a todas as planilhas."
        )
        st.markdown(
            '<p class="pv-foot">Direcional Engenharia · ferramenta interna de apoio à decisão</p>',
            unsafe_allow_html=True,
        )
        return

    stats_base = res["stats_base"]
    ticket = res["ticket"]
    conv = res["conv"]
    por_h = res["por_h"]
    html_out = res["html"]
    full_train = res["full_train"]
    sheet_metas = res.get("sheet_metas") or st.session_state.get("sheet_metas")

    if sheet_metas:
        with st.expander("Folhas selecionadas automaticamente (auditoria)", expanded=False):
            rows_m = []
            for k, m in sheet_metas.items():
                rows_m.append(
                    {
                        "Base": k,
                        "Método": m.get("method", "—"),
                        "Livro": m.get("spreadsheet_title") or "—",
                        "Aba": m.get("worksheet_title") or "—",
                        "gid": m.get("gid", "—"),
                        "Cabeçalho (linha)": m.get("header_row_index", "—"),
                        "Pontuação": m.get("score", "—"),
                    }
                )
            st.dataframe(pd.DataFrame(rows_m), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h4>Vendas (histórico)</h4><div class="val">{stats_base["vendas"]:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h4>VGV acumulado</h4><div class="val">R$ {stats_base["vgv"]/1e6:.2f} M</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h4>Ticket médio</h4><div class="val">R$ {ticket/1e3:,.0f} k</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><h4>Conversão leads → vendas</h4><div class="val">{conv:.2f} %</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Previsões (último dia da série)")
    rows = []
    for h in (3, 7, 30):
        q = por_h[h]["qtd"]
        v = por_h[h]["valor"]
        lbl = "In-sample" if full_train else "Holdout 20%"
        rows.append(
            {
                "Horizonte (dias)": h,
                "Qtd prevista": f"{float(q['pred_ultimo_dia']):,.1f}",
                "VGV previsto (R$)": f"{float(v['pred_ultimo_dia']):,.0f}",
                f"MAE Qtd ({lbl})": f"{q['metrics_test']['MAE']:.2f}",
                f"MAE VGV ({lbl})": f"{v['metrics_test']['MAE']/1e6:.2f} mi",
                "Modelo Qtd": q["model_label"],
                "Modelo VGV": v["model_label"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.download_button(
        label="Descarregar relatório HTML (gráficos e apêndice técnico)",
        data=html_out.encode("utf-8"),
        file_name="dashboard_previsao_vendas.html",
        mime="text/html; charset=utf-8",
        use_container_width=True,
    )

    st.caption(
        "As métricas de erro referem-se ao holdout temporal definido no pipeline. O HTML inclui benchmark de modelos e curvas ROC (tarefa auxiliar)."
    )
    st.markdown(
        '<p class="pv-foot">Direcional Engenharia · Previsão de vendas</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
