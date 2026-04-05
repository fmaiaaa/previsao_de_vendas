# -*- coding: utf-8 -*-
"""
Previsão de vendas — Direcional (Streamlit).

Arquivo único para deploy (ex.: Streamlit Cloud): leitura Google Sheets/CSV, pipeline de ML,
relatório HTML e identidade visual alinhada à ficha Vendas RJ. Dependências: ver
`requirements.txt` / `requirements-previsao.txt`.
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


def _pv_tqdm(*args: Any, **kwargs: Any) -> Any:
    """tqdm com flush frequente e intervalo curto — no Streamlit a barra deixa de parecer congelada."""
    import sys

    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("mininterval", 0.12)
    kwargs.setdefault("miniters", 1)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


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
        it = _pv_tqdm(specs, desc="Benchmark modelos", leave=False, unit="m")

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
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

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


class PrevHtmlPolyEnricher(BaseEstimator, TransformerMixin):
    """
    Correlação absoluta com y → top 30 colunas + PolynomialFeatures(grau=2),
    como em prevhtml.py (sem bias polinomial).
    """

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("PrevHtmlPolyEnricher requer y no fit.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_s = pd.Series(np.asarray(y).ravel(), index=X_df.index).astype(float)
        X_num = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        corr = X_num.corrwith(y_s).abs().fillna(0.0)
        k = min(30, max(1, X_num.shape[1]))
        self.top_cols_: list[Any] = corr.nlargest(k).index.tolist()
        self.rest_cols_: list[Any] = [c for c in X_num.columns if c not in self.top_cols_]
        self._in_columns_: list[Any] = list(X_num.columns)
        self.poly_ = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        self.poly_.fit(X_num[self.top_cols_].astype(np.float32))
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_aligned = X_df.reindex(columns=self._in_columns_, fill_value=0.0)
        X_num = X_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        P = self.poly_.transform(X_num[self.top_cols_].astype(np.float32))
        names = [
            str(n).replace(" ", "_X_").replace(":", "")
            for n in self.poly_.get_feature_names_out(self.top_cols_)
        ]
        poly_df = pd.DataFrame(P, columns=names, index=X_num.index)
        return pd.concat([X_num[self.rest_cols_], poly_df], axis=1)


class _PrevHtmlQtdPredictor:
    """Regressor + enriquecimento prevhtml já ajustados (apenas .predict)."""

    def __init__(self, enricher: PrevHtmlPolyEnricher, regressor: Any) -> None:
        self._enricher = enricher
        self._regressor = regressor

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._enricher.transform(X)
        pr = np.asarray(self._regressor.predict(Xt), dtype=float).ravel()
        return np.maximum(0.0, np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0))


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


def _accuracy_vs_train_median(
    y_true: np.ndarray, y_pred: np.ndarray, y_train_reference: np.ndarray
) -> float:
    """
    Acurácia direcional auxiliar: concordância com a mediana do treino (acima/abaixo).
    Complementa MAE/R² na avaliação sem substituir métricas de regressão.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ref = np.asarray(y_train_reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(y_true) < 5 or len(ref) < 5:
        return float("nan")
    med = float(np.median(ref))
    if not np.isfinite(med):
        return float("nan")
    tb = (y_true > med).astype(int)
    pb = (y_pred > med).astype(int)
    return float((tb == pb).mean())


def temporal_train_test_indices(n: int, train_frac: float = 0.7) -> tuple[slice, slice]:
    """Primeiros train_frac para treino; resto para teste out-of-sample (ex.: 70%/30%)."""
    i_tr = max(1, int(n * train_frac))
    if i_tr >= n:
        raise ValueError(
            "Série muito curta para o split temporal (treino/teste). "
            "Aumente o histórico diário ou reduza o horizonte."
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


def _fit_lgbm_with_early_stopping(
    model: Any,
    X_tr,
    y_tr,
    X_va,
    y_va,
    sample_weight,
    *,
    stopping_rounds: int = 48,
) -> None:
    """Treino LGBM com validação no fold e *early stopping* nas árvores (métrica MAE)."""
    eval_set = [(X_va, np.asarray(y_va, dtype=float).ravel())]
    try:
        from lightgbm import early_stopping, log_evaluation

        cbs = [
            early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ]
        if sample_weight is not None:
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_metric="mae",
                callbacks=cbs,
            )
        else:
            model.fit(X_tr, y_tr, eval_set=eval_set, eval_metric="mae", callbacks=cbs)
    except (ImportError, TypeError, ValueError):
        if sample_weight is not None:
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_metric="mae",
                early_stopping_rounds=stopping_rounds,
                verbose=-1,
            )
        else:
            model.fit(
                X_tr,
                y_tr,
                eval_set=eval_set,
                eval_metric="mae",
                early_stopping_rounds=stopping_rounds,
                verbose=-1,
            )


def _optuna_patience_cfg(n_trials: int) -> tuple[int, int]:
    """(patience, min_trials) para parar Optuna quando o melhor valor estagna."""
    nt = max(8, int(n_trials))
    min_tr = max(14, min(52, (2 * nt) // 5))
    pat = max(10, min(40, nt // 5))
    if min_tr >= nt:
        min_tr = max(8, min(nt - 2, (2 * nt) // 5))
    return pat, max(6, min_tr)


def _make_optuna_patience_stopping_callback(*, patience: int, min_trials: int) -> Any:
    state: dict[str, float | int] = {"best": float("inf"), "no_improve": 0}

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value is None:
            return
        if not math.isfinite(float(trial.value)):
            return
        n_ok = 0
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
                continue
            if not math.isfinite(float(t.value)):
                continue
            n_ok += 1
        bv = float(study.best_value)
        if n_ok < min_trials:
            state["best"] = bv
            state["no_improve"] = 0
            return
        if bv < float(state["best"]) - 1e-9:
            state["best"] = bv
            state["no_improve"] = 0
        else:
            state["no_improve"] = int(state["no_improve"]) + 1
        if int(state["no_improve"]) >= patience:
            study.stop()

    return _cb


def _optuna_objective_lgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    random_state: int,
    target_name: str,
    horizon: int = 7,
) -> float:
    """
    Otimização orientada à generalização temporal:
    limites de complexidade escalonados com n, penalização de variância entre folds
    (sensível a overfitting) e amplitude fold-a-fold (instabilidade).
    """
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm não instalado")

    import sys

    sys.stderr.write(f"\n[Optuna] Trial {trial.number + 1} • {target_name} • H={horizon}\n")
    sys.stderr.flush()

    is_qtd = target_name == "qtd"
    long_h = int(horizon) >= 21
    n = max(int(len(X_train)), 35)

    leaves_cap = int(
        min(76 if long_h else 48, max(20, int(14 + 5.2 * float(np.sqrt(float(n))))))
    )
    depth_cap = int(
        min(9 if long_h else 7, max(3, 2 + int(np.log2(max(n // 42, 6)))))
    )
    depth_lo = 4 if long_h else 3
    depth_cap = max(depth_lo, depth_cap)

    ne_hi = int(min(495 if long_h else 360, max(125, min(430, n // 2 + 55))))
    ne_lo = int(max(88 if long_h else 78, ne_hi - (285 if long_h else 245)))
    ne_lo = min(max(ne_lo, 68), ne_hi - 22)

    mcs_lo = int(max(26 if is_qtd else 22, min(52, n // 24 + 16)))
    mcs_hi = int(max(mcs_lo + 14, min(112 if is_qtd else 86, n // 3 + 24)))

    lr_lo, lr_hi = ((0.028, 0.13) if long_h else (0.034, 0.11))
    subs_lo, subs_hi = ((0.52, 0.86) if long_h else (0.56, 0.86))
    cols_lo, cols_hi = ((0.52, 0.86) if long_h else (0.58, 0.86))
    ra_lo, ra_hi = ((0.035, 10.5) if long_h else (0.05, 14.0))
    rl_lo, rl_hi = ((0.12, 22.0) if long_h else (0.14, 20.0))
    mg_lo, mg_hi = ((0.002, 0.82) if long_h else (0.004, 0.95))

    param = {
        "n_estimators": trial.suggest_int("n_estimators", ne_lo, ne_hi),
        "max_depth": trial.suggest_int("max_depth", depth_lo, depth_cap),
        "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
        "subsample": trial.suggest_float("subsample", subs_lo, subs_hi),
        "colsample_bytree": trial.suggest_float("colsample_bytree", cols_lo, cols_hi),
        "reg_alpha": trial.suggest_float("reg_alpha", ra_lo, ra_hi, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", rl_lo, rl_hi, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 18, max(22, leaves_cap)),
        "min_child_samples": trial.suggest_int("min_child_samples", mcs_lo, mcs_hi),
        "min_split_gain": trial.suggest_float("min_split_gain", mg_lo, mg_hi, log=True),
        "random_state": random_state,
        "verbosity": -1,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    try:
        for step, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            sys.stderr.write(f"\r[Optuna] trial {trial.number + 1} • fold {step + 1}/{n_splits}…\033[K")
            sys.stderr.flush()
            model = LGBMRegressor(**param)
            y_tr = y_train.iloc[tr_idx]
            sw = sample_weights(y_tr, True, target_name, horizon)
            y_va_fold = y_train.iloc[va_idx]
            es_rounds = 56 if long_h else 44
            _fit_lgbm_with_early_stopping(
                model,
                X_train.iloc[tr_idx],
                y_tr,
                X_train.iloc[va_idx],
                y_va_fold,
                sw,
                stopping_rounds=es_rounds,
            )
            p = model.predict(X_train.iloc[va_idx])
            fold_mae = float(mean_absolute_error(y_train.iloc[va_idx], p))
            scores.append(fold_mae)
            trial.report(float(np.mean(scores)), step)
            if step >= 1 and trial.should_prune():
                raise optuna.TrialPruned()

        mean_mae = float(np.mean(scores))
        std_mae = float(np.std(scores)) if len(scores) > 1 else 0.0
        spread = float(max(scores) - min(scores)) if len(scores) > 1 else 0.0
        pen_std = 0.12 if long_h else 0.15
        pen_sp = 0.09 if long_h else 0.105
        return mean_mae + pen_std * std_mae + pen_sp * spread / (mean_mae + 1e-6)
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()


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
            ("model", Ridge(alpha=38.0, random_state=rs)),
        ]
    )


def build_elastic_net_pipe(rs: int, target_name: str) -> Pipeline:
    a = 0.18 if target_name == "qtd" else 88000.0
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
                    n_estimators=300 if long_h else 240,
                    max_depth=11 if long_h else 9,
                    min_samples_leaf=8,
                    min_samples_split=16,
                    max_features="sqrt",
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
            min_samples_leaf=44,
            learning_rate=0.055,
            l2_regularization=6.2,
            early_stopping=True,
            validation_fraction=0.14,
            n_iter_no_change=22,
            random_state=rs,
        )
    else:
        hgb = HistGradientBoostingRegressor(
            max_iter=560,
            max_depth=5,
            min_samples_leaf=30,
            learning_rate=0.038,
            l2_regularization=4.6,
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
        "reg_alpha": float(max(0.08, best_params.get("reg_alpha", 0.18) * 1.22)),
        "reg_lambda": float(max(0.65, best_params.get("reg_lambda", 1.4) * 1.52)),
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
        learning_rate=0.03,
        l2_leaf_reg=22.0,
        random_strength=0.52,
        bagging_temperature=0.32,
        rsm=0.78,
        border_count=254,
        random_seed=rs,
        verbose=False,
        loss_function="RMSE",
        allow_writing_files=False,
        thread_count=-1,
    )
    inner: Any = _wrap_log(cb) if use_log else cb
    return Pipeline([("scaler", _mm()), ("model", inner)])


def build_svm_pipe(rs: int, target_name: str) -> Pipeline:
    """SVR com regularização moderada (C não excessivo) para limitar overfitting."""
    c = 3.2 if target_name == "qtd" else 4.0
    eps = 0.15 if target_name == "qtd" else 95_000.0
    return Pipeline(
        [
            ("scaler", _mm()),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=c,
                    epsilon=eps,
                    gamma="scale",
                    max_iter=4000,
                ),
            ),
        ]
    )


def build_knn_pipe(rs: int, target_name: str) -> Pipeline:
    """k-NN com pesos por distância e k relativamente alto — suaviza ruído."""
    k = 13 if target_name == "qtd" else 11
    return Pipeline(
        [
            ("scaler", _mm()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=k,
                    weights="distance",
                    p=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_extratrees_pipe(
    best_params: dict[str, Any], use_log: bool, rs: int, horizon: int = 7
) -> Pipeline:
    long_h = int(horizon) >= 21
    n_cap = 380 if long_h else 320
    d_cap = 11 if long_h else 10
    n_est = min(n_cap, max(220 if long_h else 200, int(best_params.get("n_estimators", 320))))
    depth = min(d_cap, max(6 if long_h else 5, int(best_params.get("max_depth", 7))))
    et = ExtraTreesRegressor(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_leaf=10,
        min_samples_split=20,
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
        final_estimator=Ridge(alpha=44.0),
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
    """Nome, pipeline (template), usar sample_weight recency (alvo valor / VGV)."""
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

    out.append(("SVM (SVR)", build_svm_pipe(rs, target_name), False))
    out.append(("k-NN", build_knn_pipe(rs, target_name), False))

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
        it = _pv_tqdm(
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
    rel_min = 0.02 if long_h else 0.024
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
    """n_trials <= 0 → escala com log do tamanho da amostra (com poda, convém mais trials)."""
    if n_trials > 0:
        return n_trials
    base = int(max(62, min(165, int(28 * np.log1p(n_samples)))))
    if int(horizon) >= 21:
        base = min(178, base + 28)
    return base


def _prevhtml_model_templates(random_state: int) -> dict[str, Any]:
    """Mesmos estimadores e VotingRegressor que prevhtml.py."""
    bases: list[tuple[str, Any]] = []
    if XGBRegressor is not None:
        bases.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=random_state,
                ),
            )
        )
    bases.extend(
        [
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=100, max_depth=5, random_state=random_state
                ),
            ),
            (
                "Gradient Boosting",
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=random_state,
                ),
            ),
            ("Ridge Linear", Ridge(alpha=1.0)),
        ]
    )
    out = dict(bases)
    out["Ensemble (Voting)"] = VotingRegressor(
        estimators=[(n, clone(e)) for n, e in bases]
    )
    return out


def _train_one_target_prevhtml(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    random_state: int,
    show_progress: bool,
    full_period_train: bool,
    train_frac: float,
) -> TrainResult:
    """
    Estratégia prevhtml.py: correlação → top-30 → polinômio grau 2;
    split cronológico 70/30 dentro da zona de treino para competição por MAE;
    vencedor reajustado em toda a zona de treino; modelo final em toda a série (como o app).
    """
    hz = int(horizon)
    idx = X.index.intersection(y.index)
    Xa = X.loc[idx].copy()
    ya = pd.to_numeric(y.loc[idx], errors="coerce").astype(float)

    if not full_period_train:
        sl_tr, sl_te = temporal_train_test_indices(len(Xa), float(train_frac))
        X_fit_zone = Xa.iloc[sl_tr]
        y_fit_zone = ya.iloc[sl_tr]
        X_outer = Xa.iloc[sl_te]
        y_outer = ya.iloc[sl_te]
    else:
        X_fit_zone = Xa
        y_fit_zone = ya
        X_outer = Xa
        y_outer = ya

    enrich = PrevHtmlPolyEnricher()
    enrich.fit(X_fit_zone, y_fit_zone)
    Xef = enrich.transform(X_fit_zone)

    n_in = len(Xef)
    split_i = int(n_in * 0.7)
    if split_i < 1 or (n_in - split_i) < 1:
        raise ValueError(
            "Série curta demais para o split interno 70/30 (estratégia prevhtml, alvo quantidade)."
        )

    X_p_tr = Xef.iloc[:split_i]
    X_p_te = Xef.iloc[split_i:]
    y_p_tr = y_fit_zone.iloc[:split_i]
    y_p_te = y_fit_zone.iloc[split_i:]

    templates = _prevhtml_model_templates(random_state)
    maes: dict[str, float] = {}
    loop = list(templates.items())
    if show_progress:
        loop = list(tqdm(loop, desc="Prevhtml (qtd)", leave=False, unit="modelo"))
    for name, tpl in loop:
        m = clone(tpl)
        try:
            m.fit(X_p_tr, y_p_tr)
            pv = np.maximum(0.0, m.predict(X_p_te))
            maes[name] = float(mean_absolute_error(y_p_te, pv))
        except Exception:
            continue

    if not maes:
        dm = DummyRegressor(strategy="median")
        dm.fit(X_p_tr, y_p_tr)
        pv0 = np.maximum(0.0, dm.predict(X_p_te))
        win_name = "Baseline mediana (fallback)"
        templates = {win_name: dm}
        maes = {win_name: float(mean_absolute_error(y_p_te, pv0))}

    win_name = min(maes, key=lambda k: maes[k])
    win_tpl = templates[win_name]

    reg_fit_zone = clone(win_tpl)
    reg_fit_zone.fit(Xef, y_fit_zone)

    X_all_enr = enrich.transform(Xa)
    reg_prod = clone(win_tpl)
    reg_prod.fit(X_all_enr, ya)

    pipe = _PrevHtmlQtdPredictor(enrich, reg_prod)

    reg_eval = clone(win_tpl)
    reg_eval.fit(X_p_tr, y_p_tr)
    pred_inner_te = np.maximum(0.0, reg_eval.predict(X_p_te))
    y_max_tr = float(np.percentile(y_p_tr, 99.5)) if len(y_p_tr) else 0.0
    long_hz = hz >= 21
    cap_mult = 2.75 if long_hz else 2.35
    cap = max(y_max_tr * cap_mult, 1.0)
    pred_inner_te = np.clip(
        np.nan_to_num(pred_inner_te, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap
    )

    thr = float(np.median(y_p_tr))

    if full_period_train:
        pred_outer = pred_inner_te
        yo = y_p_te
        do = X_p_te.index
    else:
        X_outer_enr = enrich.transform(X_outer)
        pred_outer = np.maximum(0.0, reg_fit_zone.predict(X_outer_enr))
        pred_outer = np.clip(
            np.nan_to_num(pred_outer, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap
        )
        yo = y_outer
        do = X_outer.index

    metrics_val = _metrics_dict(y_p_te.values, pred_inner_te, "qtd")
    metrics_test = _metrics_dict(np.asarray(yo, dtype=float), np.asarray(pred_outer, dtype=float), "qtd")
    metrics_val["n_features"] = float(Xa.shape[1])
    metrics_test["n_features"] = float(Xa.shape[1])

    pred_chart = pipe.predict(Xa)
    pred_chart = np.clip(
        np.nan_to_num(pred_chart, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap
    )
    chart_dates = [d.strftime("%Y-%m-%d") for d in Xa.index]
    chart_y_real = ya.tolist()
    chart_y_pred = pred_chart.tolist()
    chart_split_index = int(len(X_fit_zone)) if not full_period_train else split_i

    y_tr_ref = np.asarray(y_p_tr, dtype=float)
    metrics_val["Acc_dir_mediana"] = _accuracy_vs_train_median(
        y_p_te.values, pred_inner_te, y_tr_ref
    )
    metrics_test["Acc_dir_mediana"] = _accuracy_vs_train_median(
        np.asarray(yo, dtype=float), pred_outer, y_tr_ref
    )

    bench_rows: list[dict[str, Any]] = []
    for name, tpl in templates.items():
        m = clone(tpl)
        try:
            m.fit(X_p_tr, y_p_tr)
            pv = np.maximum(0.0, m.predict(X_p_te))
            if full_period_train:
                pt = pv
                y_te_b = y_p_te
            else:
                pt = np.maximum(0.0, m.predict(enrich.transform(X_outer)))
                y_te_b = y_outer
            bench_rows.append(
                {
                    "name": name,
                    "mae_val": float(mean_absolute_error(y_p_te, pv)),
                    "reg_val": regression_metrics(y_p_te.values, pv),
                    "reg_test": regression_metrics(np.asarray(y_te_b, dtype=float), pt),
                    "bin_val": binary_from_regression(y_p_te.values, pv, thr),
                    "bin_test": binary_from_regression(
                        np.asarray(y_te_b, dtype=float), pt, thr
                    ),
                    "roc_test": roc_curve_data(
                        np.asarray(y_te_b, dtype=float), pt, thr
                    ),
                }
            )
        except Exception:
            continue
    bench_rows.sort(key=lambda r: r["mae_val"])
    benchmark_appendix = _sanitize_benchmark_appendix(
        {
            "rows": bench_rows,
            "winner_label": win_name,
            "winner_names": [win_name],
            "winner_weights": [1.0],
            "threshold_y": thr,
        }
    )

    best_params: dict[str, Any] = {
        "estrategia": "prevhtml_poly_correlacao_70_30",
        "modelo_vencedor": win_name,
        "mae_split_interno": maes,
        "horizonte_alvo": hz,
    }

    return TrainResult(
        pipeline=pipe,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        importance=pd.Series(dtype=float),
        y_val=y_p_te,
        y_val_pred=pred_inner_te,
        y_test=pd.Series(np.asarray(yo, dtype=float), index=do),
        y_test_pred=np.asarray(pred_outer, dtype=float),
        best_params=best_params,
        dates_val=X_p_te.index,
        dates_test=do,
        model_label=f"Prevhtml: {win_name}",
        full_period_train=full_period_train,
        chart_dates=chart_dates,
        chart_y_real=chart_y_real,
        chart_y_pred=chart_y_pred,
        chart_split_index=chart_split_index,
        benchmark_appendix=benchmark_appendix,
    )


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
    train_frac: float = 0.7,
) -> TrainResult:
    if target_name == "qtd":
        return _train_one_target_prevhtml(
            X,
            y,
            horizon=horizon,
            random_state=random_state,
            show_progress=show_progress,
            full_period_train=full_period_train,
            train_frac=train_frac,
        )

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
        sl_tr, sl_te = temporal_train_test_indices(n, float(train_frac))
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
                "n_estimators": 270,
                "max_depth": 6,
                "learning_rate": 0.065,
                "subsample": 0.72,
                "colsample_bytree": 0.72,
                "reg_alpha": 0.32,
                "reg_lambda": 3.0,
                "num_leaves": 38,
                "min_child_samples": 46,
                "min_split_gain": 0.032,
            }
        else:
            best_params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.068,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.42,
                "reg_lambda": 3.6,
                "num_leaves": 32,
                "min_child_samples": 50,
                "min_split_gain": 0.04,
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

        n_startup = max(12, min(32, nt // 3))
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=optuna_seed,
                multivariate=True,
                n_startup_trials=max(10, min(28, nt // 4)),
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=n_startup,
                interval_steps=1,
                n_warmup_steps=1,
            ),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        es_patience, es_min_trials = _optuna_patience_cfg(nt)
        study.optimize(
            objective,
            n_trials=nt,
            show_progress_bar=show_progress,
            callbacks=[
                _make_optuna_patience_stopping_callback(
                    patience=es_patience,
                    min_trials=es_min_trials,
                )
            ],
        )
        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if completed:
            best_params = study.best_params
        else:
            best_params = (
                {
                    "n_estimators": 280,
                    "max_depth": 6,
                    "learning_rate": 0.065,
                    "subsample": 0.72,
                    "colsample_bytree": 0.72,
                    "reg_alpha": 0.35,
                    "reg_lambda": 3.2,
                    "num_leaves": 36,
                    "min_child_samples": 48,
                    "min_split_gain": 0.035,
                }
                if hz >= 21
                else {
                    "n_estimators": 210,
                    "max_depth": 5,
                    "learning_rate": 0.07,
                    "subsample": 0.76,
                    "colsample_bytree": 0.76,
                    "reg_alpha": 0.45,
                    "reg_lambda": 3.8,
                    "num_leaves": 32,
                    "min_child_samples": 52,
                    "min_split_gain": 0.042,
                }
            )

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
    y_tr_ref = np.asarray(y_train, dtype=float)
    metrics_val["Acc_dir_mediana"] = _accuracy_vs_train_median(
        y_val.values, pred_val, y_tr_ref
    )
    metrics_test["Acc_dir_mediana"] = _accuracy_vs_train_median(
        np.asarray(y_test, dtype=float), pred_test, y_tr_ref
    )

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

import hashlib
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


def _strip_timezone_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx).normalize()


def _winsorize_positive_series(s: pd.Series, pct: float = 99.5, mult_vs_median: float = 80.0) -> pd.Series:
    """Limita cauda superior de valores positivos (erros de digitação em VGV) sem cortar dias normais."""
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)
    pos = v[v > 0]
    if len(pos) < 25:
        return v
    cap = float(np.nanpercentile(pos.to_numpy(dtype=float), pct))
    med = float(pos.median()) if len(pos) else 0.0
    hi = max(cap, med * mult_vs_median) if med > 0 else cap
    return v.clip(upper=max(hi, 1.0))


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
    out = out.loc[out["_d"].notna()].copy()
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
    out = out.loc[out["_dag"].notna()].copy()
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
    out = out.loc[out["_d"].notna()].copy()
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
            ["vgv", "real"],
            ["valor", "venda"],
            ["vgv"],
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
    out = out.loc[out["_d"].notna()].copy()
    out["_valor"] = _winsorize_positive_series(out[c_val])
    return out, c_data, c_val


def _vdim_slug(part: str, max_len: int = 36) -> str:
    t = normalize_header(str(part).strip())[:max_len]
    t = re.sub(r"[^a-z0-9_]+", "_", t).strip("_")
    return t or "na"


def build_vendas_wide_daily(
    df_v: pd.DataFrame,
    *,
    max_dim_combinations: int = 4000,
) -> pd.DataFrame:
    """
    Agrega a base bruta de vendas num painel diário largo: por dia, contagens e soma de VGV
    por combinação de dimensões (corretor, ranking, empreendimento, regional, canal, …).

    Colunas geradas: prefixo ``vdim_q__`` (quantidade) e ``vdim_v__`` (soma VGV).
    Não entram no treino ML (filtradas em ``build_feature_matrix`` / ``build_xy_for_horizon``).
    """
    if df_v is None or len(df_v) < 1:
        return pd.DataFrame()
    fn = normalize_dataframe_columns(df_v.copy())
    c_data = find_column_any(
        fn,
        [
            ["data", "venda"],
            ["data", "fechamento"],
        ],
    )
    c_val = find_column_any(
        fn,
        [
            ["valor", "real", "venda"],
            ["valor", "real"],
            ["vgv", "real"],
            ["valor", "venda"],
            ["vgv"],
        ],
    )
    if c_val is None:
        c_val = find_column(fn, ["valor"])
    if c_data is None or c_val is None:
        return pd.DataFrame()

    dim_candidates: list[tuple[str, str | None]] = [
        ("corretor", find_column_any(fn, [["corretor"], ["vendedor"], ["broker"]])),
        ("ranking", find_column_any(fn, [["ranking"], ["rank"]])),
        (
            "empreend",
            find_column_any(
                fn,
                [
                    ["empreendimento"],
                    ["empreend"],
                    ["produto"],
                ],
            ),
        ),
        ("regional", find_column_any(fn, [["regional", "imob"], ["regional"]])),
        ("canal", find_column(fn, ["canal"])),
    ]
    dims = [(tag, c) for tag, c in dim_candidates if c is not None and c in fn.columns]
    if not dims:
        return pd.DataFrame()

    work = fn.copy()
    work["_d"] = _to_day(work[c_data])
    work = work.loc[work["_d"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["_valor"] = pd.to_numeric(work[c_val], errors="coerce").fillna(0.0)

    parts: list[pd.Series] = []
    for _tag, col in dims:
        s = work[col].fillna("").astype(str).str.strip()
        s = s.replace("", "na").replace("nan", "na")
        parts.append(s.map(_vdim_slug))

    combo = parts[0].astype(str)
    for p in parts[1:]:
        combo = combo + "__" + p.astype(str)

    if combo.nunique() > max_dim_combinations:
        vc = combo.value_counts()
        top = set(vc.nlargest(max(1, max_dim_combinations - 1)).index)
        combo = combo.where(combo.isin(top), "_outros_agrupados")

    work["_dim"] = combo
    g = work.groupby(["_d", "_dim"], observed=True)
    cnt = g.size().unstack(fill_value=0).astype(np.float64)
    vgv = g["_valor"].sum().unstack(fill_value=0.0).astype(np.float64)

    def _col_name(prefix: str, raw: str) -> str:
        body = str(raw)
        if len(body) > 72:
            body = hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]
        else:
            body = re.sub(r"[^a-z0-9_]+", "_", body.lower()).strip("_") or "x"
        return f"{prefix}{body}"

    cnt.columns = [_col_name("vdim_q__", c) for c in cnt.columns]
    vgv.columns = [_col_name("vdim_v__", c) for c in vgv.columns]
    out = pd.concat([cnt, vgv], axis=1)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
    out = out.sort_index().fillna(0.0)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.replace([np.inf, -np.inf], 0.0)


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


def forward_sum_offsets(series: pd.Series, offsets: list[int]) -> pd.Series:
    """Soma das vendas nos dias t+off para cada off em `offsets` (off > 0)."""
    arr = series.fillna(0).to_numpy(dtype=float)
    n = len(arr)
    offs = sorted({int(o) for o in offsets if int(o) > 0})
    y = np.full(n, np.nan, dtype=float)
    for i in range(n):
        s = 0.0
        ok = True
        for o in offs:
            j = i + o
            if j >= n:
                ok = False
                break
            s += arr[j]
        if ok:
            y[i] = s
    return pd.Series(y, index=series.index)


def forward_calendar_features_offsets(
    index: pd.DatetimeIndex, offsets: list[int]
) -> pd.DataFrame:
    """Calendário e feriados BR apenas nos dias exatos t+off (ex.: combinação 4,11,12)."""
    from datetime import timedelta

    offs = sorted({int(o) for o in offsets if int(o) > 0})
    H = _br_holiday_dates_for_index(index)

    def _ponte(d) -> bool:
        if d not in H:
            return False
        return d.weekday() in (0, 4, 5, 6)

    n = len(index)
    fwd_wknd = np.zeros(n, dtype=float)
    fwd_bday = np.zeros(n, dtype=float)
    fwd_fer = np.zeros(n, dtype=float)
    fwd_ves = np.zeros(n, dtype=float)
    fwd_pos = np.zeros(n, dtype=float)
    fwd_pont = np.zeros(n, dtype=float)
    for i, ts in enumerate(index):
        for o in offs:
            dt = (ts + pd.Timedelta(days=int(o))).normalize()
            dow = int(dt.dayofweek)
            if dow >= 5:
                fwd_wknd[i] += 1.0
            else:
                fwd_bday[i] += 1.0
            d = dt.date()
            if H and d in H:
                fwd_fer[i] += 1.0
                if _ponte(d):
                    fwd_pont[i] += 1.0
            if H and (d + timedelta(days=1)) in H:
                fwd_ves[i] += 1.0
            if H and (d - timedelta(days=1)) in H:
                fwd_pos[i] += 1.0
    h = float(len(offs)) if offs else 1.0
    return pd.DataFrame(
        {
            "fwd_h_custom": h,
            "fwd_wknd_custom": fwd_wknd,
            "fwd_bday_custom": fwd_bday,
            "fwd_wknd_ratio_custom": fwd_wknd / (h + 1e-9),
            "fwd_feriados_custom": fwd_fer,
            "fwd_vesperas_custom": fwd_ves,
            "fwd_pos_feriado_custom": fwd_pos,
            "fwd_feriado_ponte_custom": fwd_pont,
        },
        index=index,
    )


def forward_sum_same_month_dom(series: pd.Series, doms: list[int]) -> pd.Series:
    """
    Soma vendas nos dias do mês (calendário) estritamente posteriores ao dia corrente.
    Ex.: no dia 3, com doms [7,14,15], soma vendas nos dias 7, 14 e 15 do mesmo mês.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(series.index).normalize())
    arr = series.fillna(0).to_numpy(dtype=float)
    n = len(arr)
    date_to_pos = {idx[i]: i for i in range(n)}
    y = np.full(n, np.nan, dtype=float)
    doms_u = sorted({int(d) for d in doms if 1 <= int(d) <= 31})

    for i in range(n):
        ts = idx[i]
        y0, m0, d0 = ts.year, ts.month, ts.day
        total = 0.0
        n_hit = 0
        valid = True
        for dom in doms_u:
            if dom <= d0:
                continue
            try:
                fut = pd.Timestamp(year=y0, month=m0, day=dom).normalize()
            except ValueError:
                valid = False
                break
            j = date_to_pos.get(fut)
            if j is None:
                valid = False
                break
            total += arr[j]
            n_hit += 1
        if not valid:
            y[i] = np.nan
        elif n_hit == 0:
            y[i] = 0.0
        else:
            y[i] = total
    return pd.Series(y, index=series.index)


def forward_calendar_features_dom(
    index: pd.DatetimeIndex, doms: list[int]
) -> pd.DataFrame:
    """Fins de semana e feriados nos dias-alvo (mesmo mês, dia do mês > dia de t)."""
    from datetime import timedelta

    doms_u = sorted({int(d) for d in doms if 1 <= int(d) <= 31})
    H = _br_holiday_dates_for_index(index)

    def _ponte(d) -> bool:
        if d not in H:
            return False
        return d.weekday() in (0, 4, 5, 6)

    n = len(index)
    fwd_wknd = np.zeros(n, dtype=float)
    fwd_bday = np.zeros(n, dtype=float)
    fwd_fer = np.zeros(n, dtype=float)
    fwd_ves = np.zeros(n, dtype=float)
    fwd_pos = np.zeros(n, dtype=float)
    fwd_pont = np.zeros(n, dtype=float)
    n_days = np.zeros(n, dtype=float)

    for i, ts in enumerate(index):
        y0, m0, d0 = ts.year, ts.month, ts.day
        for dom in doms_u:
            if dom <= d0:
                continue
            try:
                dti = pd.Timestamp(year=y0, month=m0, day=dom).normalize()
            except ValueError:
                continue
            n_days[i] += 1.0
            dow = int(dti.dayofweek)
            if dow >= 5:
                fwd_wknd[i] += 1.0
            else:
                fwd_bday[i] += 1.0
            d = dti.date()
            if H and d in H:
                fwd_fer[i] += 1.0
                if _ponte(d):
                    fwd_pont[i] += 1.0
            if H and (d + timedelta(days=1)) in H:
                fwd_ves[i] += 1.0
            if H and (d - timedelta(days=1)) in H:
                fwd_pos[i] += 1.0

    h = np.maximum(n_days, 1.0)
    return pd.DataFrame(
        {
            "fwd_h_dom": n_days,
            "fwd_wknd_dom": fwd_wknd,
            "fwd_bday_dom": fwd_bday,
            "fwd_wknd_ratio_dom": fwd_wknd / (h + 1e-9),
            "fwd_feriados_dom": fwd_fer,
            "fwd_vesperas_dom": fwd_ves,
            "fwd_pos_feriado_dom": fwd_pos,
            "fwd_feriado_ponte_dom": fwd_pont,
        },
        index=index,
    )


def forward_sum_calendar_date_range(
    series: pd.Series, d_start: pd.Timestamp, d_end: pd.Timestamp
) -> pd.Series:
    """
    Para cada dia t: soma do alvo nos dias d da série com t < d e d_start <= d <= d_end (inclusive).
    Linhas sem dias futuros no intervalo ficam com NaN (excluídas do treino).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(series.index).normalize())
    d_lo = pd.Timestamp(d_start).normalize()
    d_hi = pd.Timestamp(d_end).normalize()
    if d_hi < d_lo:
        d_lo, d_hi = d_hi, d_lo
    arr = series.fillna(0).to_numpy(dtype=float)
    n = len(arr)
    y = np.full(n, np.nan, dtype=float)
    for i in range(n):
        ts = idx[i]
        total = 0.0
        hits = 0
        for j in range(n):
            d_j = idx[j]
            if d_j <= ts or d_j < d_lo or d_j > d_hi:
                continue
            total += arr[j]
            hits += 1
        if hits == 0:
            y[i] = np.nan
        else:
            y[i] = total
    return pd.Series(y, index=series.index)


def forward_calendar_features_date_range(
    index: pd.DatetimeIndex, d_start: pd.Timestamp, d_end: pd.Timestamp
) -> pd.DataFrame:
    """Fins de semana e feriados BR nos dias da série que entram no intervalo (após t)."""
    from datetime import timedelta

    d_lo = pd.Timestamp(d_start).normalize()
    d_hi = pd.Timestamp(d_end).normalize()
    if d_hi < d_lo:
        d_lo, d_hi = d_hi, d_lo
    H = _br_holiday_dates_for_index(index)

    def _ponte(d) -> bool:
        if d not in H:
            return False
        return d.weekday() in (0, 4, 5, 6)

    n = len(index)
    idx_norm = pd.DatetimeIndex(pd.to_datetime(index).normalize())
    fwd_wknd = np.zeros(n, dtype=float)
    fwd_bday = np.zeros(n, dtype=float)
    fwd_fer = np.zeros(n, dtype=float)
    fwd_ves = np.zeros(n, dtype=float)
    fwd_pos = np.zeros(n, dtype=float)
    fwd_pont = np.zeros(n, dtype=float)
    n_days = np.zeros(n, dtype=float)

    for i in range(n):
        ts = idx_norm[i]
        for j in range(n):
            d_j = idx_norm[j]
            if d_j <= ts or d_j < d_lo or d_j > d_hi:
                continue
            n_days[i] += 1.0
            dow = int(d_j.dayofweek)
            if dow >= 5:
                fwd_wknd[i] += 1.0
            else:
                fwd_bday[i] += 1.0
            d = d_j.date()
            if H and d in H:
                fwd_fer[i] += 1.0
                if _ponte(d):
                    fwd_pont[i] += 1.0
            if H and (d + timedelta(days=1)) in H:
                fwd_ves[i] += 1.0
            if H and (d - timedelta(days=1)) in H:
                fwd_pos[i] += 1.0

    h = np.maximum(n_days, 1.0)
    return pd.DataFrame(
        {
            "fwd_h_range": n_days,
            "fwd_wknd_range": fwd_wknd,
            "fwd_bday_range": fwd_bday,
            "fwd_wknd_ratio_range": fwd_wknd / (h + 1e-9),
            "fwd_feriados_range": fwd_fer,
            "fwd_vesperas_range": fwd_ves,
            "fwd_pos_feriado_range": fwd_pos,
            "fwd_feriado_ponte_range": fwd_pont,
        },
        index=index,
    )


def build_xy_custom_date_range(
    df_master: pd.DataFrame,
    target_daily_col: str,
    d_start: pd.Timestamp,
    d_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    y = forward_sum_calendar_date_range(df_master[target_daily_col], d_start, d_end)
    feature_cols = [c for c in df_master.columns if c not in ("target_qtd", "target_valor")]
    X = df_master[feature_cols].copy()
    cal = forward_calendar_features_date_range(df_master.index, d_start, d_end)
    X = pd.concat([X, cal], axis=1)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0.0)
    return X, y


def build_xy_custom_offsets(
    df_master: pd.DataFrame,
    target_daily_col: str,
    offsets: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    y = forward_sum_offsets(df_master[target_daily_col], offsets)
    feature_cols = [c for c in df_master.columns if c not in ("target_qtd", "target_valor")]
    X = df_master[feature_cols].copy()
    cal = forward_calendar_features_offsets(df_master.index, offsets)
    X = pd.concat([X, cal], axis=1)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0.0)
    return X, y


def build_xy_custom_dom(
    df_master: pd.DataFrame,
    target_daily_col: str,
    doms: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    y = forward_sum_same_month_dom(df_master[target_daily_col], doms)
    feature_cols = [c for c in df_master.columns if c not in ("target_qtd", "target_valor")]
    X = df_master[feature_cols].copy()
    cal = forward_calendar_features_dom(df_master.index, doms)
    X = pd.concat([X, cal], axis=1)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0.0)
    return X, y


def predict_last_row_custom_offsets(
    df_master: pd.DataFrame, pipeline: Any, offsets: list[int]
) -> float:
    Xb = build_feature_matrix(df_master)
    cal = forward_calendar_features_offsets(df_master.index, offsets)
    X = pd.concat([Xb, cal], axis=1)
    X_last = X.iloc[[-1]]
    p = float(np.asarray(pipeline.predict(X_last)).ravel()[0])
    if not np.isfinite(p):
        p = 0.0
    return max(0.0, p)


def predict_last_row_custom_dom(
    df_master: pd.DataFrame, pipeline: Any, doms: list[int]
) -> float:
    Xb = build_feature_matrix(df_master)
    cal = forward_calendar_features_dom(df_master.index, doms)
    X = pd.concat([Xb, cal], axis=1)
    X_last = X.iloc[[-1]]
    p = float(np.asarray(pipeline.predict(X_last)).ravel()[0])
    if not np.isfinite(p):
        p = 0.0
    return max(0.0, p)


def predict_last_row_custom_date_range(
    df_master: pd.DataFrame,
    pipeline: Any,
    d_start: pd.Timestamp,
    d_end: pd.Timestamp,
) -> float:
    Xb = build_feature_matrix(df_master)
    cal = forward_calendar_features_date_range(df_master.index, d_start, d_end)
    X = pd.concat([Xb, cal], axis=1)
    X_last = X.iloc[[-1]]
    p = float(np.asarray(pipeline.predict(X_last)).ravel()[0])
    if not np.isfinite(p):
        p = 0.0
    return max(0.0, p)


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
            idx_iter = _pv_tqdm(
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


def _find_formulario_vgv_col(
    df: pd.DataFrame,
    alternatives: list[list[str]],
    *,
    used: set[str],
) -> str | None:
    """
    Coluna numérica de VGV no formulário (exclui colunas cujo nome sugere quantidade).
    Tenta várias convenções de cabeçalho: muitas folhas usam 'vgv' em vez de 'vendas'.
    """
    for parts in alternatives:
        for col in df.columns:
            if col in used:
                continue
            c = str(col)
            if "qtd" in c:
                continue
            if all(p in c for p in parts):
                used.add(col)
                return col
    return None


def _find_qtd_col(
    df: pd.DataFrame,
    parts: list[str],
    *,
    used: set[str] | None = None,
) -> str | None:
    for col in df.columns:
        if used is not None and col in used:
            continue
        c = str(col)
        if "qtd" not in c:
            continue
        if all(p in c for p in parts):
            if used is not None:
                used.add(col)
            return col
    return None


def _find_formulario_qtd_col(
    df: pd.DataFrame,
    alternatives: list[list[str]],
    *,
    used: set[str],
) -> str | None:
    for parts in alternatives:
        c = _find_qtd_col(df, parts, used=used)
        if c is not None:
            return c
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

    used_v: set[str] = set()
    val_prev_fac = _find_formulario_vgv_col(
        df,
        [
            ["vendas", "facilitadas", "previstas"],
            ["vgv", "facilitadas", "previstas"],
            ["valor", "facilitadas", "previstas"],
            ["facilitadas", "previstas", "vgv"],
            ["facilitadas", "previstas"],
        ],
        used=used_v,
    )
    val_prev_norm = _find_formulario_vgv_col(
        df,
        [
            ["vendas", "normais", "previstas"],
            ["vgv", "normais", "previstas"],
            ["valor", "normais", "previstas"],
            ["normais", "previstas", "vgv"],
            ["normais", "previstas"],
        ],
        used=used_v,
    )
    val_real_fac = _find_formulario_vgv_col(
        df,
        [
            ["vendas", "facilitadas", "reais"],
            ["vendas", "facilitadas", "real"],
            ["vgv", "facilitadas", "reais"],
            ["facilitadas", "reais", "vgv"],
            ["facilitadas", "reais"],
            ["facilitadas", "realizado"],
            ["facilitadas", "realizada"],
        ],
        used=used_v,
    )
    val_real_norm = _find_formulario_vgv_col(
        df,
        [
            ["vendas", "normais", "reais"],
            ["vendas", "normais", "real"],
            ["vgv", "normais", "reais"],
            ["normais", "reais", "vgv"],
            ["normais", "reais"],
            ["normais", "realizado"],
            ["normais", "realizada"],
        ],
        used=used_v,
    )
    used_q: set[str] = set()
    q_prev_fac = _find_formulario_qtd_col(
        df,
        [
            ["facilitadas", "previstas"],
            ["facilitada", "prevista"],
        ],
        used=used_q,
    )
    q_prev_norm = _find_formulario_qtd_col(
        df,
        [
            ["normais", "previstas"],
            ["normal", "prevista"],
        ],
        used=used_q,
    )
    q_real_fac = _find_formulario_qtd_col(
        df,
        [
            ["facilitadas", "reais"],
            ["facilitadas", "real"],
            ["facilitada", "real"],
        ],
        used=used_q,
    )
    q_real_norm = _find_formulario_qtd_col(
        df,
        [
            ["normais", "reais"],
            ["normais", "real"],
            ["normal", "real"],
        ],
        used=used_q,
    )

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
    if df_master.index.duplicated().any():
        df_master = df_master[~df_master.index.duplicated(keep="last")]

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

    try:
        vw = build_vendas_wide_daily(df_v)
        if vw is not None and not vw.empty:
            df_master = df_master.join(vw, how="left")
    except Exception as e:
        if show_progress:
            print(f"Aviso: painel desagregado de vendas (vdim_) omitido: {e}")

    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    bad_tgt = df_master["target_qtd"].isna() | df_master["target_valor"].isna()
    df_master = df_master.loc[~bad_tgt].copy()
    feat_only = [c for c in df_master.columns if c not in ("target_qtd", "target_valor")]
    df_master[feat_only] = df_master[feat_only].fillna(0.0)
    return df_master


def build_feature_matrix(df_master: pd.DataFrame) -> pd.DataFrame:
    """Matriz X alinhada ao índice diário (sem colunas de alvo bruto)."""
    feature_cols = [
        c
        for c in df_master.columns
        if c not in ("target_qtd", "target_valor") and not str(c).startswith("vdim_")
    ]
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
# Fallback quando o export oficial devolve 403/HTML a partir de datacenters (ex.: Streamlit Cloud).
GVIZ_CSV_URL = "https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&gid={gid}"
_CSV_FETCH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

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


_PEM_END_MARKERS = (
    "-----END PRIVATE KEY-----",
    "-----END RSA PRIVATE KEY-----",
    "-----END ENCRYPTED PRIVATE KEY-----",
)

_SERVICE_ACCOUNT_JSON_KEYS = frozenset(
    {
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
        "universe_domain",
    },
)


def _reparar_private_key_json_com_quebras_literais(s: str) -> str:
    """Se o PEM foi colado com quebras reais dentro da string JSON, reescape para json.loads."""
    k = s.find('"private_key"')
    if k == -1:
        k = s.find("'private_key'")
    if k == -1:
        return s
    colon = s.find(":", k)
    if colon == -1:
        return s
    q_open = s.find('"', colon)
    if q_open == -1:
        return s
    val_start = q_open + 1
    rest = s[val_start:]
    end_pem = -1
    for mark in _PEM_END_MARKERS:
        p = rest.find(mark)
        if p != -1:
            end_pem = p + len(mark)
            break
    if end_pem == -1:
        return s
    pem = rest[:end_pem]
    after = rest[end_pem:]
    i = 0
    while i < len(after) and after[i] in " \t\r\n":
        i += 1
    if i >= len(after) or after[i] != '"':
        return s
    tail = after[i + 1 :]
    inner_esc = json.dumps(pem)[1:-1]
    return s[:val_start] + inner_esc + '"' + tail


def _parse_service_account_json_string(raw: str) -> dict[str, Any] | None:
    """Parse do JSON da conta de serviço (string única ou multilinha TOML)."""
    s = (raw or "").strip().lstrip("\ufeff")
    if not s:
        return None
    candidates = (s, _reparar_private_key_json_com_quebras_literais(s))
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _service_account_credenciais_preenchidas(info: dict[str, Any]) -> bool:
    """Ignora template com campos vazios (ex.: secrets de exemplo)."""
    ce = str(info.get("client_email") or "").strip()
    pk = str(info.get("private_key") or "").strip()
    return bool(ce and pk and "BEGIN" in pk)


def _dict_google_sheets_sem_json_bruto(gs: dict[str, Any]) -> dict[str, Any]:
    """Só chaves do JSON Google; exclui strings JSON embutidas e ruído."""
    skip = frozenset(
        {
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            "google_service_account_json",
            "SERVICE_ACCOUNT_JSON",
            "service_account_json",
        }
    )
    return {
        str(k): gs[k]
        for k in gs
        if str(k) not in skip and k in _SERVICE_ACCOUNT_JSON_KEYS
    }


# Chaves em [google_sheets] que não são IDs de planilhas (credenciais, metadados, sub-tabelas).
_GOOGLE_SHEETS_RESERVED_LOWER: frozenset[str] = frozenset(
    x.lower()
    for x in (
        *_SERVICE_ACCOUNT_JSON_KEYS,
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "google_service_account_json",
        "SERVICE_ACCOUNT_JSON",
        "service_account_json",
        "SPREADSHEET_ID",
        "spreadsheet_id",
        "WORKSHEET_NAME",
        "worksheet_name",
        "GERENTES_WORKSHEET",
        "gerentes_worksheet",
        "NOME_CONTA_COLUMN",
        "nome_conta_column",
        "VALORES_FIXOS_WORKSHEET",
        "valores_fixos_worksheet",
        "spreadsheet_ids",
        "sheets",
        "csv_gid_hints",
    )
)


def service_account_from_google_sheets_section(st: Any) -> dict[str, Any] | None:
    """
    Leitura alinhada à **Ficha Vendas RJ**: credenciais na secção `[google_sheets]`.
    - `SERVICE_ACCOUNT_JSON` / `service_account_json` como **dict** (TOML) ou **string** JSON;
    - repara `private_key` com quebras literais (`_reparar_private_key_json_com_quebras_literais`);
    - ou as mesmas chaves do JSON em linhas TOML (`type`, `client_email`, `private_key`, …).
    """
    try:
        gs = st.secrets.get("google_sheets")
    except Exception:
        gs = None
    if not isinstance(gs, dict):
        return None
    for jkey in (
        "SERVICE_ACCOUNT_JSON",
        "service_account_json",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "google_service_account_json",
    ):
        raw = gs.get(jkey)
        if raw is None:
            continue
        if isinstance(raw, dict):
            if _service_account_credenciais_preenchidas(raw):
                return raw
            continue
        info = _parse_service_account_json_string(_secret_str(raw))
        if info and _service_account_credenciais_preenchidas(info):
            return info
    flat = _dict_google_sheets_sem_json_bruto(gs)
    if flat.get("type") == "service_account" or (
        str(flat.get("client_email") or "").strip()
        and str(flat.get("private_key") or "").strip()
    ):
        if _service_account_credenciais_preenchidas(flat):
            return flat
    return None


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

    for env_key in ("GOOGLE_SERVICE_ACCOUNT_JSON", "SERVICE_ACCOUNT_JSON"):
        raw = os.environ.get(env_key)
        if not raw or not str(raw).strip():
            continue
        info = _parse_service_account_json_string(str(raw))
        if info and _service_account_credenciais_preenchidas(info):
            return info
    return None


def _secret_str(val: Any) -> str:
    """Streamlit / TOML podem devolver tipos especiais; normaliza para string."""
    if val is None:
        return ""
    s = str(val).strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and "]" in s):
        return s
    return s


def _service_account_from_flat_streamlit_root(st: Any) -> dict[str, Any] | None:
    """Secrets com chaves do JSON Google na raiz do TOML (sem embutir JSON inteiro)."""
    try:
        if "client_email" not in st.secrets or "private_key" not in st.secrets:
            return None
        d: dict[str, Any] = {}
        for k in _SERVICE_ACCOUNT_JSON_KEYS:
            if k in st.secrets:
                v = st.secrets[k]
                if v is None:
                    continue
                if k == "private_key":
                    pk = str(v).replace("\r\n", "\n").strip()
                    if pk:
                        d[k] = pk
                else:
                    sv = str(v).strip()
                    if sv:
                        d[k] = sv
        if _service_account_credenciais_preenchidas(d):
            return d
    except Exception:
        pass
    return None


def service_account_info_from_streamlit_secrets() -> dict[str, Any] | None:
    """
    Ordem de leitura (compatível com **Ficha Vendas RJ** e deploys antigos):
    1. `[google_sheets]` — SERVICE_ACCOUNT_JSON (dict/str), GOOGLE_*, ou chaves espelhadas do JSON;
    2. Raiz: chaves planas `client_email` + `private_key`;
    3. Raiz: `GOOGLE_SERVICE_ACCOUNT_JSON` ou `SERVICE_ACCOUNT_JSON` (dict/str);
    4. Secções `google_service_account` / `gcp_service_account` / `service_account`.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        info_gs = service_account_from_google_sheets_section(st)
        if info_gs:
            return info_gs

        root_flat = _service_account_from_flat_streamlit_root(st)
        if root_flat:
            return root_flat

        if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
            raw = _secret_str(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
            if raw:
                info = _parse_service_account_json_string(raw)
                if info and _service_account_credenciais_preenchidas(info):
                    return info

        if "SERVICE_ACCOUNT_JSON" in st.secrets:
            raw_root = st.secrets["SERVICE_ACCOUNT_JSON"]
            if isinstance(raw_root, dict) and _service_account_credenciais_preenchidas(raw_root):
                return raw_root
            raw_s = _secret_str(raw_root)
            if raw_s:
                info = _parse_service_account_json_string(raw_s)
                if info and _service_account_credenciais_preenchidas(info):
                    return info

        for key in ("google_service_account", "gcp_service_account", "service_account"):
            if key in st.secrets:
                sec = st.secrets[key]
                if isinstance(sec, str) and sec.strip():
                    if sec.strip().startswith("{"):
                        info = _parse_service_account_json_string(sec)
                        if info and _service_account_credenciais_preenchidas(info):
                            return info
                if hasattr(sec, "keys"):
                    info = {str(k): sec[k] for k in sec.keys()}
                    if _service_account_credenciais_preenchidas(info):
                        return info
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


def _sa_fingerprint_for_cache() -> str:
    """Muda quando as credenciais mudam — evita cache CSV quando a API passa a estar disponível."""
    info = service_account_info_from_streamlit_secrets() or _service_account_info_from_env()
    if not info:
        return "no_sa"
    em = str(info.get("client_email") or "").strip()
    kid = str(info.get("private_key_id") or "").strip()
    return f"{em}|{kid}" if em else "sa_no_email"


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


def _fetch_csv_bytes_dual(sheet_id: str, gid: str) -> tuple[bytes | None, str]:
    """
    Tenta export oficial e, em seguida, endpoint gviz (útil quando o primeiro devolve 403/HTML).
    Devolve (bytes, "") ou (None, motivo) para diagnóstico.
    """
    headers = {
        "User-Agent": _CSV_FETCH_UA,
        "Accept": "text/csv,text/plain,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    last_err = ""
    for label, tmpl in (("export", EXPORT_URL), ("gviz", GVIZ_CSV_URL)):
        url = tmpl.format(sid=sheet_id, gid=gid)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                head = data[:1200].lower()
                if b"<html" in head or b"<!doctype html" in head:
                    last_err = f"{label}:HTML({resp.status})"
                    continue
                if b"sign in" in head or b"accounts.google" in head:
                    last_err = f"{label}:login({resp.status})"
                    continue
                if len(data) < 30:
                    last_err = f"{label}:curto({resp.status})"
                    continue
                return data, ""
        except urllib.error.HTTPError as e:
            last_err = f"{label}:HTTP{e.code}"
        except urllib.error.URLError as e:
            last_err = f"{label}:URL({e.reason!s})"
        except Exception as e:
            last_err = f"{label}:{type(e).__name__}"
    return None, last_err or "?"


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
        raw, why = _fetch_csv_bytes_dual(spreadsheet_id, gid)
        if raw is None:
            tried.append(f"{gid}:{why}")
            continue
        tried.append(f"{gid}:ok")
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
            f"(tentados: {', '.join(tried[:12])}…). "
            "Em muitos ambientes na nuvem o Google bloqueia export CSV anónimo: use **API com conta de serviço** "
            "(secrets `GOOGLE_SERVICE_ACCOUNT_JSON`) e partilhe cada livro com o e-mail `client_email` como **Leitor**, "
            "com APIs **Google Sheets** e **Google Drive** ativas no projeto GCP."
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

import numpy as np
import pandas as pd


def _json_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _numeric_series_for_corr_picker(df: pd.DataFrame, *, max_cols: int = 120) -> dict[str, list[float]]:
    """Colunas numéricas alinhadas ao índice diário (para dispersão e correlação entre qualquer par na UI)."""
    if df is None or len(df) == 0:
        return {}
    n = len(df)
    cols: list[Any] = []
    seen: set[str] = set()
    for c in df.select_dtypes(include=[np.number]).columns:
        key = str(c).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cols.append(c)
        if len(cols) >= max_cols:
            break
    out: dict[str, list[float]] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        vals: list[float] = []
        for v in s.to_numpy():
            try:
                if pd.isna(v):
                    vals.append(float("nan"))
                else:
                    fv = float(v)
                    vals.append(fv if math.isfinite(fv) else float("nan"))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        if len(vals) == n:
            out[str(c)] = vals
    return out


def _daily_pack_from_master(df: pd.DataFrame) -> dict[str, Any]:
    """Séries agregadas para gráficos (HTML + Streamlit); evita expor matriz completa de features."""
    if df is None or len(df) == 0:
        z = [0.0] * 7
        return {
            "dates": [],
            "vol_leads": [],
            "vol_agend": [],
            "vol_visit": [],
            "vol_pastas": [],
            "target_qtd": [],
            "target_valor": [],
            "dow_labels": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
            "dow_mean_qtd": z,
            "dow_mean_valor": z,
            "macro": {},
            "corr_labels": [],
            "corr_z": [],
            "n_rows": 0,
            "n_features": 0,
            "numeric_series_for_picker": {},
        }
    idx = pd.DatetimeIndex(df.index)
    dates = [d.strftime("%Y-%m-%d") for d in idx]

    def col(name: str) -> list[float]:
        if name not in df.columns:
            return [0.0] * len(df)
        s = pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return [float(x) if np.isfinite(x) else 0.0 for x in s]

    macro_cols = [c for c in df.columns if str(c).startswith("macro_")][:15]
    macro_data: dict[str, list[float]] = {c: col(c) for c in macro_cols}

    dff = df.copy()
    dff["_dow"] = idx.dayofweek
    dow_order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    gq = dff.groupby("_dow", sort=True)["target_qtd"].mean() if "target_qtd" in dff.columns else None
    gv = dff.groupby("_dow", sort=True)["target_valor"].mean() if "target_valor" in dff.columns else None
    dow_q = [float(gq.get(i, 0.0) or 0.0) for i in range(7)] if gq is not None else [0.0] * 7
    dow_v = [float(gv.get(i, 0.0) or 0.0) for i in range(7)] if gv is not None else [0.0] * 7

    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c
        in (
            "vol_leads",
            "vol_agend",
            "vol_visit",
            "vol_pastas",
            "target_qtd",
            "target_valor",
        )
        or str(c).startswith("macro_")
        or str(c).startswith("fb_vgv_")
        or str(c).startswith("fb_qtd_")
    ]
    num_cols = num_cols[:40]
    corr_labels: list[str] = []
    corr_z: list[list[float]] = []
    if len(num_cols) >= 2:
        sub = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if len(sub) > 1:
            cm = sub.corr().fillna(0.0)
            corr_labels = [str(c) for c in cm.columns]
            corr_z = [[float(cm.iloc[i, j]) for j in range(len(corr_labels))] for i in range(len(corr_labels))]

    return {
        "dates": dates,
        "vol_leads": col("vol_leads"),
        "vol_agend": col("vol_agend"),
        "vol_visit": col("vol_visit"),
        "vol_pastas": col("vol_pastas"),
        "target_qtd": col("target_qtd"),
        "target_valor": col("target_valor"),
        "dow_labels": dow_order,
        "dow_mean_qtd": dow_q,
        "dow_mean_valor": dow_v,
        "macro": macro_data,
        "corr_labels": corr_labels,
        "corr_z": corr_z,
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1]),
        "numeric_series_for_picker": _numeric_series_for_corr_picker(df),
    }


def _openai_key_and_model() -> tuple[str | None, str]:
    key: str | None = None
    model = "gpt-4o-mini"
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            raw = st.secrets.get("OPENAI_API_KEY")
            if raw:
                key = str(raw).strip() or None
            oa = st.secrets.get("openai")
            if isinstance(oa, dict):
                if not key and oa.get("api_key"):
                    key = str(oa["api_key"]).strip() or None
                if oa.get("model"):
                    model = str(oa["model"]).strip() or model
    except Exception:
        pass
    import os

    if not key:
        envk = os.environ.get("OPENAI_API_KEY", "").strip()
        key = envk or None
    env_m = os.environ.get("OPENAI_MODEL", "").strip()
    if env_m:
        model = env_m
    return key, model


def _openai_chat_completion(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    timeout: int = 75,
) -> str | None:
    import json
    import urllib.error
    import urllib.request

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.35,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception:
        return None


def _pearson_r_vectors(a: list[Any], b: list[Any]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _pearson_pairwise_complete(x: list[Any], y: list[Any]) -> tuple[float | None, int]:
    """Pearson em pares (xᵢ, yᵢ) com ambos finitos; ignora NaN. Devolve (r, n_válido)."""
    if len(x) != len(y):
        return None, 0
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(x, y):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fa) and math.isfinite(fb):
            xs.append(fa)
            ys.append(fb)
    n = len(xs)
    if n < 3:
        return None, n
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    if float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        return None, n
    return float(np.corrcoef(xa, ya)[0, 1]), n


def _corr_pairs_sorted(labels: list[str], z: list[list[float]]) -> list[tuple[str, str, float]]:
    n = len(labels)
    if n < 2 or len(z) != n or any(len(row) != n for row in z):
        return []
    out: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = z[i][j]
            if isinstance(r, (int, float)) and math.isfinite(float(r)):
                out.append((str(labels[i]), str(labels[j]), float(r)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out


def _interpret_text_funil_vendas(daily: dict[str, Any]) -> str:
    tq = daily.get("target_qtd") or []
    if len(tq) < 10:
        return ""
    n = len(tq)
    k = max(1, min(n // 5, 90))
    first = float(np.mean(np.asarray(tq[:k], dtype=float)))
    last = float(np.mean(np.asarray(tq[-k:], dtype=float)))
    if first < 1e-9:
        return ""
    chg = (last - first) / first * 100.0
    return (
        f"Em média, a quantidade vendida nos primeiros ~{k} dias da série foi {first:,.1f}; "
        f"já nos últimos ~{k} dias, fixou-se em {last:,.1f}, o que representa {chg:+.1f}% face ao início. "
        f"Assim, o gráfico permite confrontar, no tempo, os picos do funil com o comportamento das vendas e do VGV."
    )


def _interpret_text_rolagem_7d(daily: dict[str, Any]) -> str:
    tq = daily.get("target_qtd") or []
    if len(tq) < 14:
        return ""
    s = pd.to_numeric(pd.Series(tq), errors="coerce").fillna(0.0)
    roll = s.rolling(7, min_periods=1).mean()
    last = float(roll.iloc[-1])
    if len(roll) >= 30:
        prev = float(roll.iloc[-30])
        d = last - prev
        return (
            f"A média móvel de 7 dias no último dia regista {last:,.1f} unidades; por comparação, há cerca de "
            f"30 observações o valor situava-se em {prev:,.1f}, pelo que a variação é de {d:+,.1f}. "
            f"Deste modo, a curva suavizada atenua o ruído de curto prazo sem ocultar a tendência."
        )
    return (
        f"No último dia, a média móvel de 7 dias é de {last:,.1f} unidades; todavia, a série é curta demais "
        f"para uma comparação fiável com janelas mais longas."
    )


def _interpret_text_leads_vendas(daily: dict[str, Any]) -> str:
    x = daily.get("vol_leads") or []
    y = daily.get("target_qtd") or []
    r = _pearson_r_vectors(x, y)
    if r is None:
        return (
            "No cruzamento leads × vendas (mesmo dia), a amostra é insuficiente ou apresenta variância nula; "
            "por conseguinte, o coeficiente de Pearson não é estimável de forma robusta."
        )
    qual = "fraca"
    if abs(r) >= 0.5:
        qual = "moderada"
    if abs(r) >= 0.75:
        qual = "forte"
    sinal = "positiva" if r > 0 else "negativa"
    return (
        f"A correlação de Pearson, no mesmo dia, é r = {r:+.2f}, o que corresponde a uma associação linear {qual} e {sinal}. "
        f"Contudo, tal medida não implica relação causal: efeitos defasados, sazonalidade ou variáveis confundidoras "
        f"podem explicar parte do padrão observado."
    )


def _interpret_text_dow(daily: dict[str, Any]) -> str:
    dl = daily.get("dow_labels") or []
    dq = daily.get("dow_mean_qtd") or []
    dv = daily.get("dow_mean_valor") or []
    if len(dl) != len(dq) or not dq:
        return ""
    dq_f = [float(x) for x in dq]
    imax = int(np.argmax(dq_f))
    imin = int(np.argmin(dq_f))
    parts = [
        f"No perfil semanal, a maior média de quantidade verifica-se à {dl[imax]} ({dq_f[imax]:,.1f}), "
        f"ao passo que o menor nível médio ocorre à {dl[imin]} ({dq_f[imin]:,.1f})."
    ]
    if len(dv) == len(dl):
        dv_f = [float(v) for v in dv]
        jmax = int(np.argmax(dv_f))
        parts.append(
            f"Relativamente ao VGV, o maior valor médio ocorre à {dl[jmax]} (R$ {dv_f[jmax]/1e6:.2f} mi)."
        )
    return " ".join(parts)


def _interpret_text_correlation_matrix(labels: list[str], z: list[list[float]]) -> str:
    pairs = _corr_pairs_sorted(labels, z)
    if not pairs:
        return ""
    strong = [p for p in pairs if abs(p[2]) >= 0.7][:8]
    pos = sorted((p for p in pairs if p[2] > 0), key=lambda t: t[2], reverse=True)[:5]
    neg = sorted((p for p in pairs if p[2] < 0), key=lambda t: t[2])[:5]
    chunks: list[str] = []
    if strong:
        chunks.append(
            "Registam-se associações lineares fortes (|r| ≥ 0,70), nomeadamente: "
            + "; ".join(f"{a} ↔ {b} ({r:+.2f})" for a, b, r in strong)
            + "."
        )
    pos_f = [p for p in pos if p[2] >= 0.25]
    if pos_f:
        chunks.append(
            "Além disso, as correlações positivas mais marcadas são as seguintes: "
            + "; ".join(f"{a}–{b}: {r:+.2f}" for a, b, r in pos_f)
            + "."
        )
    neg_f = [p for p in neg if p[2] <= -0.25]
    if neg_f:
        chunks.append(
            "Por outro lado, destacam-se estas associações negativas: "
            + "; ".join(f"{a}–{b}: {r:+.2f}" for a, b, r in neg_f)
            + "."
        )
    near = [p for p in pairs if abs(p[2]) < 0.15][:3]
    if near and len(labels) > 4:
        chunks.append(
            "Há ainda pares com correlação quase nula (por exemplo, "
            + ", ".join(f"{a}–{b} ({r:+.2f})" for a, b, r in near)
            + "), o que sugere pouca ligação linear direta entre essas variáveis nesta amostra."
        )
    chunks.append(
        "Em suma, o coeficiente de Pearson quantifica apenas a tendência linear simultânea; "
        "não evidencia, por si só, relações causais nem efeitos com desfasamento temporal."
    )
    return " ".join(x for x in chunks if x)


def _interpret_text_macro(macro: dict[str, Any]) -> str:
    keys = [k for k in macro.keys() if isinstance(macro.get(k), list)][:12]
    if len(keys) < 2:
        return ""
    lengths = [len(macro[k]) for k in keys]
    m = min(lengths)
    if m < 6:
        return ""
    mat = np.array([[float(macro[k][i] or 0.0) for k in keys] for i in range(m)], dtype=float)
    cm = np.corrcoef(mat.T)
    pair_list: list[tuple[str, str, float]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = float(cm[i, j])
            if math.isfinite(r):
                pair_list.append((str(keys[i]), str(keys[j]), r))
    pair_list.sort(key=lambda t: abs(t[2]), reverse=True)
    top = pair_list[:5]
    if not top:
        return ""
    return (
        "Entre os indicadores macro, calculadas sobre o mesmo período e nos níveis originais, "
        "as correlações mais elevadas são: "
        + "; ".join(f"{a} ↔ {b}: {r:+.2f}" for a, b, r in top)
        + ". Observe que, no gráfico, as séries são normalizadas em z-score unicamente para facilitar "
        "a comparação da forma temporal, sem alterar a estrutura de correlação acima referida."
    )


def _interpret_text_vif_rows(vifs: list[Any]) -> str:
    if not vifs:
        return ""
    scored: list[tuple[str, float]] = []
    for r in vifs:
        if not isinstance(r, dict):
            continue
        name = str(r.get("variavel", "")) or "?"
        try:
            vf = float(r.get("vif"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf) and vf >= 5.0:
            scored.append((name, vf))
    scored.sort(key=lambda t: t[1], reverse=True)
    if not scored:
        return (
            "Neste subconjunto, nenhuma variável apresenta VIF ≥ 5; assim, há pouca evidência estatística "
            "de redundância linear acentuada entre as colunas consideradas."
        )
    top = scored[:8]
    return (
        "Identificam-se variáveis com VIF elevado (≥ 5), entre as quais: "
        + "; ".join(f"{n} (~{v:.1f})" for n, v in top)
        + ". Isto indica dependência linear relevante face a outras *features*; consequentemente, "
        "convém interpretar importâncias e coeficientes com prudência, sobretudo em modelos lineares."
    )


def _facts_corr_vif_for_llm(
    labels: list[str], z: list[list[float]], vifs: list[Any]
) -> str:
    chunks: list[str] = []
    if len(labels) >= 2 and z:
        t = _interpret_text_correlation_matrix(labels, z)
        if t:
            chunks.append(t)
    tv = _interpret_text_vif_rows(vifs)
    if tv:
        chunks.append(tv)
    return "\n".join(chunks)


def _facts_eda_compact(daily: dict[str, Any]) -> str:
    lines = [
        _interpret_text_funil_vendas(daily),
        _interpret_text_rolagem_7d(daily),
        _interpret_text_leads_vendas(daily),
        _interpret_text_dow(daily),
    ]
    cl = daily.get("corr_labels") or []
    cz = daily.get("corr_z") or []
    if len(cl) >= 2 and cz:
        lines.append(_interpret_text_correlation_matrix(cl, cz))
    lines.append(_interpret_text_macro(daily.get("macro") or {}))
    return "\n".join(x for x in lines if x)


def _openai_eda_synopsis(facts: str) -> str | None:
    key, model = _openai_key_and_model()
    if not key or not facts.strip():
        return None
    sys_m = (
        "Assume o papel de analista de dados sénior. Redige em português (Brasil), com 2 a 4 parágrafos curtos, "
        "tom profissional e conectivos adequados (por exemplo: contudo, além disso, neste contexto, em suma, assim). "
        "Utiliza exclusivamente os factos fornecidos; não inventes quantidades nem conclusões não suportadas. "
        "Referencia explicitamente que correlação não implica causalidade e que as vendas podem reagir com defasagem "
        "a variáveis explicativas."
    )
    return _openai_chat_completion(
        [
            {"role": "system", "content": sys_m},
            {
                "role": "user",
                "content": "Elabore uma síntese profissional para gestão, com base nos indicadores abaixo. "
                "Utilize conectivos adequados e estruture a resposta de forma lógica:\n\n" + facts.strip(),
            },
        ],
        api_key=key,
        model=model,
    )


def _html_eda_interpretacoes(daily: dict[str, Any]) -> str:
    blocks: list[tuple[str, str]] = [
        ("Funil e vendas", _interpret_text_funil_vendas(daily)),
        ("Suavização 7 dias", _interpret_text_rolagem_7d(daily)),
        ("Leads × vendas (dispersão)", _interpret_text_leads_vendas(daily)),
        ("Dia da semana", _interpret_text_dow(daily)),
    ]
    cl = daily.get("corr_labels") or []
    cz = daily.get("corr_z") or []
    if len(cl) >= 2 and cz:
        blocks.append(("Matriz de correlação", _interpret_text_correlation_matrix(cl, cz)))
    mx = _interpret_text_macro(daily.get("macro") or {})
    if mx:
        blocks.append(("Indicadores macro", mx))
    parts: list[str] = []
    for title, body in blocks:
        if not body:
            continue
        parts.append(
            f'<h5 class="font-semibold text-slate-800 text-sm mt-3 mb-1">{html_lib.escape(title)}</h5>'
            f'<p class="text-sm text-slate-600 text-justify leading-relaxed">{html_lib.escape(body)}</p>'
        )
    if not parts:
        return ""
    foot = html_lib.escape(
        "Nota metodológica: as interpretações são geradas automaticamente com base em regras e estatística descritiva; "
        "o relatório HTML permanece autónomo, isto é, não recorre a serviços de API externos."
    )
    return (
        '<div class="glass-card p-6 mb-6 border-l-4 border-sky-500">'
        '<h4 class="text-lg font-black text-slate-900 mb-1">Interpretações automáticas</h4>'
        + "".join(parts)
        + f'<p class="text-xs text-slate-400 mt-3">{foot}</p></div>'
    )


def _st_interpretacao_grafico(titulo: str, texto: str) -> None:
    if not texto:
        return
    st.markdown(
        f'<div class="pv-interpret-wrap"><p class="pv-interpret-title">{html_lib.escape(titulo)}</p>'
        f'<p class="pv-interpret-text">{html_lib.escape(texto)}</p></div>',
        unsafe_allow_html=True,
    )


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
    hold_lbl = "in-sample (100%)" if full_period_train else "holdout (30%)"
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
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
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
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{ family: 'Plus Jakarta Sans', size: 11 }},
        margin: {{ t: 28, l: 52, r: 12, b: 120 }},
        xaxis: {{ tickangle: -55, automargin: true }},
        yaxis: {{ title: ytitle, gridcolor: '#f1f5f9' }}
      }}, {{ responsive: true }});
    }})();
    """
    return html_frag, script


def _methodology_appendix_block(
    horizontes: list[int],
    por_horizonte: dict[int, dict[str, Any]],
    best_params_preview: dict[int, dict[str, dict[str, Any]]],
    full_period_train: bool,
    blend_top_k: int,
    random_seed: int,
    n_rows: int,
    n_features: int,
) -> str:
    hz_txt = ", ".join(str(x) for x in horizontes)
    modo = (
        "<b>100% da série</b> no modelo final; ~8% finais só para pesos do ensemble (sem holdout nas métricas principais)."
        if full_period_train
        else "<b>70% / 30%</b> cronológicos; ~8% no fim do treino para escolher o ensemble. "
        "MAE, RMSE, R² no <b>teste (30%)</b> salvo indicação em contrário."
    )
    params_blocks: list[str] = []
    for h in horizontes:
        pq = best_params_preview.get(h, {}).get("qtd", {})
        pv = best_params_preview.get(h, {}).get("valor", {})
        ql = html_lib.escape(str(por_horizonte[h]["qtd"].get("model_label", "—")))
        vl = html_lib.escape(str(por_horizonte[h]["valor"].get("model_label", "—")))
        pj = html_lib.escape(json.dumps(pq, indent=2, ensure_ascii=False)[:12000])
        pjv = html_lib.escape(json.dumps(pv, indent=2, ensure_ascii=False)[:12000])
        params_blocks.append(
            f'<div class="glass-card p-5 mb-5 border-t-4 border-slate-700">'
            f'<h4 class="font-bold text-slate-900 mb-2">Horizonte {h} dias — modelo operacional</h4>'
            f'<p class="text-sm text-slate-600 mb-2">Volume: <b>{ql}</b> · VGV: <b>{vl}</b></p>'
            f'<p class="text-xs font-bold text-slate-500 uppercase mb-1">Hiperparâmetros LightGBM (Optuna)</p>'
            f'<pre class="text-xs overflow-auto max-h-56 bg-slate-50 p-3 rounded-lg border mb-3">{pj}</pre>'
            f'<p class="text-xs font-bold text-slate-500 uppercase mb-1">Hiperparâmetros LightGBM — alvo VGV</p>'
            f'<pre class="text-xs overflow-auto max-h-56 bg-slate-50 p-3 rounded-lg border">{pjv}</pre></div>'
        )
    candidatos = (
        "Ridge, ElasticNet, RandomForest, LightGBM (regressão, fair, quantis, log1p para VGV), "
        "XGBoost e CatBoost quando disponíveis, HistGradientBoosting, ExtraTrees, NGBoost (VGV), "
        "regressão TS-linear, stacks leves e baseline mediana."
    )
    return f"""
    <div class="glass-card p-8 mb-8 border-l-4 border-blue-600">
      <h2 class="text-2xl font-black text-slate-900 mb-4">Metodologia</h2>
      <div class="text-sm text-slate-700 space-y-3 leading-relaxed text-justify">
        <p><b>Alvo.</b> Em cada <code>t</code>, soma de vendas (qtd ou VGV) nos próximos <code>H</code> dias, <code>H</code> ∈ {{{hz_txt}}}. <code>X</code> só com dados até <code>t</code> (lags, calendário, STL, feriados, macro BCB, formulário defasado, etc.).</p>
        <p><b>Dados.</b> <b>{n_rows:,}</b> dias; <b>{n_features}</b> colunas após engenharia.</p>
        <p><b>Validação.</b> {modo}</p>
        <p><b>Optuna / LightGBM.</b> TPE + <code>TimeSeriesSplit</code>, <code>MedianPruner</code>, MAE nos folds com penalização de variância entre folds; limites de complexidade ligados a <code>n</code>. Semente <b>{random_seed}</b>. Demais candidatos do benchmark com regularização mais conservadora.</p>
        <p><b>Benchmark e ensemble.</b> {candidatos} Ranking por MAE na validação interna. Os <b>{blend_top_k}</b> melhores entram num blend com pesos <code>∝ exp(−(MAE−MAE_min)/τ)</code> ou fica o melhor isolado se não houver ganho.</p>
        <p><b>Entradas.</b> <b>MinMaxScaler (0,1)</b> com <code>clip</code>.</p>
        <p><b>Métricas binárias (HTML).</b> Comparação à mediana de <code>y</code> no treino do benchmark.</p>
      </div>
    </div>
    <h3 class="text-lg font-black text-slate-800 mb-3">Parâmetros LightGBM finais (pós-Optuna) por horizonte</h3>
    {"".join(params_blocks)}
    """


def _build_analises_pane_html(daily: dict[str, Any]) -> tuple[str, str]:
    daily_html = {k: v for k, v in daily.items() if k != "numeric_series_for_picker"}
    dj = _json_safe(daily_html)
    interp_block = _html_eda_interpretacoes(daily)
    html_frag = """
    <div class="glass-card p-6 mb-6">
      <h3 class="text-xl font-black text-slate-900 mb-2">Séries diárias — funil e alvos</h3>
      <p class="text-sm text-slate-600 mb-4 text-justify">Contagens do funil (empilhadas), vendas diárias em quantidade e VGV.</p>
      <div id="an_combo" style="height:480px;width:100%;"></div>
    </div>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <div class="glass-card p-6">
        <h4 class="font-bold text-slate-800 mb-2">Média por dia da semana</h4>
        <div id="an_dow" style="height:380px;width:100%;"></div>
      </div>
      <div class="glass-card p-6">
        <h4 class="font-bold text-slate-800 mb-2">Correlação (variáveis-chave)</h4>
        <div id="an_corr" style="height:380px;width:100%;"></div>
      </div>
    </div>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6" id="an_scatter_row">
      <div class="glass-card p-6">
        <h4 class="font-bold text-slate-800 mb-2">Dispersão: leads × qtd (mesmo dia)</h4>
        <p class="text-sm text-slate-600 mb-3 text-justify">Complementa a matriz acima: cada ponto corresponde a um dia civil, com leads no eixo horizontal e vendas em quantidade no vertical.</p>
        <div id="an_scatter_q" style="height:380px;width:100%;"></div>
      </div>
      <div class="glass-card p-6">
        <h4 class="font-bold text-slate-800 mb-2">Dispersão: leads × VGV (mesmo dia)</h4>
        <p class="text-sm text-slate-600 mb-3 text-justify">Mesmo eixo horizontal (leads); o VGV diário é apresentado em milhões de reais, de modo a alinhar a escala à leitura do funil.</p>
        <div id="an_scatter_v" style="height:380px;width:100%;"></div>
      </div>
    </div>
    <div class="glass-card p-6 mb-6" id="an_macro_wrap">
      <h4 class="font-bold text-slate-800 mb-2">Indicadores macro (normalizados)</h4>
      <div id="an_macro" style="height:360px;width:100%;"></div>
    </div>
    """
    html_frag = html_frag + interp_block
    script = f"""
    (function() {{
      var D = {dj};
      var dates = D.dates || [];
      function zscore(arr) {{
        var m = arr.reduce(function(a,b){{return a+b;}},0) / (arr.length || 1);
        var v = arr.reduce(function(a,b){{return a+Math.pow(b-m,2);}},0) / (arr.length || 1);
        var s = Math.sqrt(v) || 1;
        return arr.map(function(x) {{ return (x - m) / s; }});
      }}
      function xAxisRangeFromDates(ds) {{
        if (!ds || ds.length < 2) return {{}};
        return {{ range: [ds[0], ds[ds.length - 1]] }};
      }}
      Plotly.newPlot('an_combo', [
        {{ x: dates, y: D.vol_leads, name: 'Leads', stackgroup: 'one', line: {{ width: 0 }}, fillcolor: 'rgba(15,23,42,0.35)', type: 'scatter', mode: 'lines' }},
        {{ x: dates, y: D.vol_agend, name: 'Agend.', stackgroup: 'one', line: {{ width: 0 }}, fillcolor: 'rgba(59,130,246,0.4)', type: 'scatter', mode: 'lines' }},
        {{ x: dates, y: D.vol_visit, name: 'Visitas', stackgroup: 'one', line: {{ width: 0 }}, fillcolor: 'rgba(99,102,241,0.4)', type: 'scatter', mode: 'lines' }},
        {{ x: dates, y: D.vol_pastas, name: 'Pastas', stackgroup: 'one', line: {{ width: 0 }}, fillcolor: 'rgba(139,92,246,0.45)', type: 'scatter', mode: 'lines' }},
        {{ x: dates, y: D.target_qtd, name: 'Vendas (linha)', yaxis: 'y2', line: {{ color: '#059669', width: 2.5 }}, type: 'scatter', mode: 'lines' }},
        {{ x: dates, y: D.target_valor.map(function(v){{return v/1e6;}}), name: 'VGV (linha)', yaxis: 'y3', line: {{ color: '#ea580c', width: 2, dash: 'dot' }}, type: 'scatter', mode: 'lines' }}
      ], {{
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{ family: 'Plus Jakarta Sans', size: 11 }},
        margin: {{ t: 96, l: 52, r: 56, b: 56 }},
        xaxis: Object.assign({{ title: 'Eixo X — data' }}, xAxisRangeFromDates(dates)),
        yaxis: {{ title: 'Eixo Y₁ — funil empilhado', gridcolor: '#e5e7eb' }},
        yaxis2: {{ overlaying: 'y', side: 'right', title: 'Eixo Y₂ — qtd (/ dia)', showgrid: false }},
        yaxis3: {{ anchor: 'free', overlaying: 'y', side: 'right', position: 0.98, title: 'Eixo Y₃ — valor (mi R$ / dia)', showgrid: false }},
        legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.06, x: 0.5, xanchor: 'center', bgcolor: 'rgba(248,250,252,0.94)', bordercolor: '#cbd5e1', borderwidth: 1, font: {{ size: 10 }}, title: {{ text: 'Séries', font: {{ size: 10, color: '#64748b' }} }} }}
      }}, {{ responsive: true }});

      var dl = D.dow_labels || [];
      Plotly.newPlot('an_dow', [
        {{ x: dl, y: D.dow_mean_qtd || [], name: 'Barras — qtd', type: 'bar', marker: {{ color: '#334155' }} }},
        {{ x: dl, y: (D.dow_mean_valor || []).map(function(v){{return v/1e6;}}), name: 'Barras — VGV', type: 'bar', marker: {{ color: '#0d9488' }}, yaxis: 'y2' }}
      ], {{
        barmode: 'group',
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {{ title: 'Eixo X — dia da semana' }},
        yaxis: {{ title: 'Eixo Y₁ — média qtd (/ dia)' }},
        yaxis2: {{ overlaying: 'y', side: 'right', title: 'Eixo Y₂ — média VGV (mi R$ / dia)' }},
        font: {{ family: 'Plus Jakarta Sans', size: 11 }},
        margin: {{ t: 56, l: 48, r: 48, b: 48 }},
        legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.04, x: 0.5, xanchor: 'center', bgcolor: 'rgba(248,250,252,0.94)', bordercolor: '#e2e8f0', borderwidth: 1, title: {{ text: 'Séries', font: {{ size: 10, color: '#64748b' }} }} }}
      }}, {{ responsive: true }});

      var labs = D.corr_labels || [];
      var z = D.corr_z || [];
      if (labs.length > 1 && z.length) {{
        Plotly.newPlot('an_corr', [{{
          z: z, x: labs, y: labs, type: 'heatmap', colorscale: 'RdBu', zmid: 0,
          hoverongaps: false
        }}], {{
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {{ t: 28, l: 120, r: 28, b: 120 }},
          font: {{ size: 10 }}, xaxis: {{ tickangle: -45 }}
        }}, {{ responsive: true }});
      }} else {{
        document.getElementById('an_corr').innerHTML = '<p class="text-sm text-slate-500 p-4">Correlação indisponível.</p>';
      }}

      var vl = D.vol_leads || [];
      var tqsc = D.target_qtd || [];
      var tvmi = (D.target_valor || []).map(function(v){{ return v / 1e6; }});
      var scatterOk = vl.length > 5 && vl.length === tqsc.length && vl.length === tvmi.length;
      if (scatterOk) {{
        Plotly.newPlot('an_scatter_q', [{{
          x: vl, y: tqsc, type: 'scatter', mode: 'markers',
          marker: {{ size: 7, opacity: 0.5, color: '#3b82f6', line: {{ width: 0.5, color: '#ffffff' }} }}
        }}], {{
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: {{ title: 'Eixo X — leads', gridcolor: '#e5e7eb' }}, yaxis: {{ title: 'Eixo Y — qtd vendida', gridcolor: '#e5e7eb' }},
          margin: {{ t: 20, l: 52, r: 28, b: 52 }},
          font: {{ family: 'Plus Jakarta Sans', size: 11 }}
        }}, {{ responsive: true }});
        Plotly.newPlot('an_scatter_v', [{{
          x: vl, y: tvmi, type: 'scatter', mode: 'markers',
          marker: {{ size: 7, opacity: 0.5, color: '#0d9488', line: {{ width: 0.5, color: '#ffffff' }} }}
        }}], {{
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: {{ title: 'Eixo X — leads', gridcolor: '#e5e7eb' }}, yaxis: {{ title: 'Eixo Y — VGV (mi R$)', gridcolor: '#e5e7eb' }},
          margin: {{ t: 20, l: 52, r: 28, b: 52 }},
          font: {{ family: 'Plus Jakarta Sans', size: 11 }}
        }}, {{ responsive: true }});
      }} else {{
        var srow = document.getElementById('an_scatter_row');
        if (srow) srow.style.display = 'none';
      }}

      var macro = D.macro || {{}};
      var mk = Object.keys(macro);
      if (mk.length) {{
        var traces = mk.slice(0, 5).map(function(k) {{
          return {{ x: dates, y: zscore(macro[k] || []), name: k, line: {{ width: 1.4 }} }};
        }});
        Plotly.newPlot('an_macro', traces, {{
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          title: 'Macro (z-score por série)',
          font: {{ family: 'Plus Jakarta Sans', size: 11 }},
          margin: {{ t: 72, l: 48, r: 28, b: 48 }},
          xaxis: Object.assign({{ title: 'Eixo X — data' }}, xAxisRangeFromDates(dates)),
          yaxis: {{ title: 'Eixo Y — z-score' }},
          legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.05, x: 0.5, xanchor: 'center', bgcolor: 'rgba(248,250,252,0.94)', bordercolor: '#cbd5e1', borderwidth: 1, title: {{ text: 'Séries', font: {{ size: 10, color: '#64748b' }} }} }}
        }}, {{ responsive: true }});
      }} else {{
        document.getElementById('an_macro_wrap').style.display = 'none';
      }}
    }})();
    """
    return html_frag, script


def _build_appendix_html(
    horizontes: list[int],
    por_horizonte: dict[int, dict[str, Any]],
    full_period_train: bool,
    best_params_preview: dict[int, dict[str, dict[str, Any]]],
    blend_top_k: int,
    random_seed: int,
    n_rows: int,
    n_features: int,
) -> tuple[str, str]:
    parts: list[str] = []
    scripts: list[str] = []
    intro_meth = _methodology_appendix_block(
        horizontes,
        por_horizonte,
        best_params_preview,
        full_period_train,
        blend_top_k,
        random_seed,
        n_rows,
        n_features,
    )
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
            f'<div class="mb-4"><h3 class="text-xl font-black text-slate-800 mb-3">Horizonte {h} dias — benchmark</h3>{hq}{hv}</div>'
        )
        if sq:
            scripts.append(sq)
        if sv:
            scripts.append(sv)
    bench_intro = """
    <div class="glass-card p-6 mb-8 border-l-4 border-indigo-500">
      <h2 class="text-xl font-black text-slate-900 mb-2">Benchmark</h2>
      <p class="text-sm text-slate-600">Regressão, tarefa binária auxiliar, ROC e MAE no teste. Scaler 0–1 com <code>clip</code>.</p>
    </div>
    """
    return intro_meth + bench_intro + "\n".join(parts), "\n".join(scripts)


def render_dashboard(
    stats_base: dict[str, float],
    ticket_medio: float,
    conversao_funil: float,
    horizontes: list[int],
    por_horizonte: dict[int, dict[str, Any]],
    best_params_preview: dict[int, dict[str, dict[str, Any]]],
    out_path: str | None = None,
    full_period_train: bool = False,
    daily_pack: dict[str, Any] | None = None,
    blend_top_k: int = 6,
    random_seed: int = 42,
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
        "Modelo final no histórico completo; ~8% finais para pesos do ensemble. Previsão: H dias após a última data."
        if full_period_train
        else "Métricas no bloco final (30%); modelo operacional reajustado em todo o histórico."
    )
    lbl_mae_grande = "MAE in-sample (100%)" if full_period_train else "MAE teste (30%)"
    lbl_r2_grande = "R² in-sample (100%)" if full_period_train else "R² teste (30%)"
    lbl_val_box = (
        "Seleção ensemble (~8% final do histórico)"
        if full_period_train
        else "Seleção ensemble (~8% do bloco de treino 70%)"
    )
    lbl_graf_series = (
        "Real vs modelo — treino (~92%) | validação (~8%) · linha vertical na separação"
        if full_period_train
        else "Real vs modelo — treino (70%) | teste (30%) · linha vertical na separação"
    )
    lbl_chart_treino = "Treino (~92%)" if full_period_train else "Treino (70%)"
    lbl_chart_hold = "Validação (~8%)" if full_period_train else "Teste (30%)"
    lbl_tr_js = json.dumps(lbl_chart_treino, ensure_ascii=False)
    lbl_te_js = json.dumps(lbl_chart_hold, ensure_ascii=False)
    lbl_scatter = "Dispersão VGV (in-sample)" if full_period_train else "Dispersão VGV (holdout 30%)"
    lbl_modelo_esc = (
        "Modelo escolhido (MAE na fatia final — seleção de ensemble)"
        if full_period_train
        else "Modelo escolhido (MAE na validação interna do treino 70%)"
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
                        if (xdt.length >= 2) {{
                            lo.xaxis = Object.assign({{}}, lo.xaxis, {{
                                range: [xdt[0], xdt[xdt.length - 1]]
                            }});
                        }}
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

    dp = daily_pack if daily_pack is not None else {}
    n_rows = int(dp.get("n_rows") or 0)
    n_features = int(dp.get("n_features") or 0)
    appendix_html, appendix_scripts = _build_appendix_html(
        horizontes,
        por_horizonte,
        full_period_train,
        best_params_preview,
        blend_top_k,
        random_seed,
        n_rows,
        n_features,
    )
    analises_html, analises_scripts = _build_analises_pane_html(dp)

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
<body class="p-3 md:p-5">
  <div class="max-w-7xl mx-auto px-1 sm:px-2">
    <header class="glass-card p-6 mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <h1 class="text-2xl md:text-3xl font-extrabold text-slate-900">Previsão de vendas</h1>
        <p class="text-slate-500 mt-1">{subtreino}</p>
      </div>
      <span class="text-sm font-semibold text-slate-600 bg-slate-100 px-4 py-2 rounded-full">{hoje}</span>
    </header>

    <div class="flex flex-wrap gap-2 mb-6 justify-center">
      <button type="button" class="sec-main-btn active px-5 py-2.5 rounded-xl font-bold text-sm uppercase tracking-wide"
        data-sec="analises" onclick="openSection('analises', this)">Análises</button>
      <button type="button" class="sec-main-btn px-5 py-2.5 rounded-xl font-bold text-sm uppercase tracking-wide text-slate-600"
        data-sec="previsoes" onclick="openSection('previsoes', this)">Previsão</button>
      <button type="button" class="sec-main-btn px-5 py-2.5 rounded-xl font-bold text-sm uppercase tracking-wide text-slate-600"
        data-sec="apendice" onclick="openSection('apendice', this)">Apêndice</button>
    </div>

    <div id="sec-analises" class="section-pane active">
    {analises_html}
    </div>

    <div id="sec-previsoes" class="section-pane">
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
      if (typeof Plotly !== 'undefined') {{
        setTimeout(function() {{
          var sel = secId === 'apendice' ? '#sec-apendice' : (secId === 'analises' ? '#sec-analises' : '#sec-previsoes');
          document.querySelectorAll(sel + ' .js-plotly-plot').forEach(function(gd) {{
            try {{ Plotly.Plots.resize(gd); }} catch (e) {{}}
          }});
        }}, 120);
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
    {analises_scripts}
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
import os
import streamlit as st

# --- Identidade visual (paridade com ficha de credenciamento Vendas RJ — aplicar_estilo) ---
COR_AZUL_ESC = "#04428f"
COR_VERMELHO = "#cb0935"
COR_BORDA = "#eef2f6"
COR_INPUT_BG = "#f0f2f6"
COR_TEXTO_MUTED = "#64748b"
COR_TEXTO_LABEL = "#1e293b"
COR_VERMELHO_ESCURO = "#9e0828"

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
    """Converte #RRGGBB em 'r, g, b' para uso em rgba(...)."""
    x = (hex_color or "").strip().lstrip("#")
    if len(x) != 6:
        return "0, 0, 0"
    return f"{int(x[0:2], 16)}, {int(x[2:4], 16)}, {int(x[4:6], 16)}"


RGB_AZUL_CSS = _hex_rgb_triplet(COR_AZUL_ESC)
RGB_VERMELHO_CSS = _hex_rgb_triplet(COR_VERMELHO)

# Cores secundárias para gráficos Plotly (identidade Direcional)
PLOT_AZUL = COR_AZUL_ESC
PLOT_VERMELHO = COR_VERMELHO
PLOT_ACCENT = "#0e7490"
PLOT_MUTED = "#64748b"
PLOT_GRID = "#e2e8f0"
# Fundo dos gráficos: transparente para fundir com o cartão (.block-container) no Streamlit e glass-card no HTML
PLOTLY_PAPER_BG = "rgba(0,0,0,0)"
PLOTLY_PLOT_BG = "rgba(0,0,0,0)"

# Legenda horizontal por baixo da área de plotagem, acima do rótulo/título do eixo X
def _plotly_legend_bottom() -> dict[str, Any]:
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.11,
        x=0.5,
        xanchor="center",
        font=dict(size=10),
        bgcolor="rgba(248,250,252,0.94)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        title=dict(text="Séries", font=dict(size=10, color="#64748b")),
    )


def _plotly_layout_direcional(
    title: str | None = None,
    height: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Layout base Plotly: paleta e tipografia alinhadas à marca."""
    lo: dict[str, Any] = {
        "template": "plotly_white",
        "paper_bgcolor": PLOTLY_PAPER_BG,
        "plot_bgcolor": PLOTLY_PLOT_BG,
        "font": dict(
            family="Montserrat, Inter, sans-serif",
            size=12,
            color="#1e293b",
        ),
        "xaxis": dict(
            gridcolor=PLOT_GRID,
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#cbd5e1",
            tickfont=dict(size=11),
        ),
        "yaxis": dict(
            gridcolor=PLOT_GRID,
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#cbd5e1",
            tickfont=dict(size=11),
        ),
        "hoverlabel": dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Montserrat, Inter, sans-serif",
            bordercolor=PLOT_GRID,
        ),
        "hovermode": "x unified",
        "legend": dict(
            orientation="h",
            yanchor="top",
            y=-0.11,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        "margin": dict(t=56, l=56, r=52, b=118),
    }
    if title:
        lo["title"] = dict(
            text=title,
            font=dict(size=17, color=PLOT_AZUL, family="Montserrat, sans-serif"),
            x=0.5,
            xanchor="center",
        )
    if height is not None:
        lo["height"] = height
    lo.update(extra)
    return lo


def _plotly_xaxis_range_from_dates(dates: Any) -> list[Any] | None:
    """Intervalo [mín, máx] das datas da série para o eixo X (sem folga além do necessário)."""
    if not dates:
        return None
    try:
        s = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
        if s.empty:
            return None
        t0, t1 = s.min(), s.max()
        if t0 == t1:
            t0 = t0 - pd.Timedelta(days=1)
            t1 = t1 + pd.Timedelta(days=1)
        return [t0, t1]
    except (TypeError, ValueError, OverflowError):
        return None


def _formulario_df_from_state() -> pd.DataFrame | None:
    """Prioridade: *snapshot* explícito (botão na aba Formulário) → resultado ML → análises exploratórias."""
    fs = st.session_state.get("formulario_snapshot")
    if isinstance(fs, pd.DataFrame) and not fs.empty:
        return fs.copy()
    r = st.session_state.get("resultado")
    if isinstance(r, dict):
        df = r.get("df_formulario")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    dex = st.session_state.get("dados_exploratorios")
    if isinstance(dex, dict):
        df = dex.get("df_formulario")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return None


def _formulario_nf_pair_cols(fn: pd.DataFrame) -> tuple[str | None, str | None]:
    hits = [
        c
        for c in fn.columns
        if "normal" in str(c).lower() and "facilitada" in str(c).lower()
    ]
    prev_c = next((c for c in hits if "real" not in str(c).lower()), hits[0] if hits else None)
    real_c = next((c for c in hits if "real" in str(c).lower()), None)
    return prev_c, real_c


def _formulario_map_columns(fn: pd.DataFrame) -> dict[str, Any]:
    m: dict[str, Any] = {}
    m["ref"] = find_column_any(
        fn,
        [
            ["data", "referencia", "sabado"],
            ["referencia", "sabado"],
            ["data", "referencia"],
        ],
    )
    m["empreendimento"] = find_column_any(
        fn,
        [["empreendimento", "previsto"], ["empreendimento"], ["previsto", "ter", "venda"]],
    )
    m["regional"] = find_column_any(fn, [["regional", "imob"], ["regional"]])
    m["canal"] = find_column(fn, ["canal"])
    m["regiao"] = find_column(fn, ["regiao"])
    m["imobiliaria"] = find_column(fn, ["imobiliaria"])
    m["gerente"] = find_column(fn, ["gerente"])
    m["erro"] = find_column(fn, ["erro", "previs"])
    m["vendas_prev"] = None
    for c in fn.columns:
        lc = str(c).lower()
        if (
            "vendas" in lc
            and ("previs" in lc or "previst" in lc)
            and "real" not in lc
            and "qtd" not in lc
            and "vgv" not in lc
        ):
            m["vendas_prev"] = c
            break
    m["vendas_real"] = None
    for c in fn.columns:
        lc = str(c).lower()
        if "vendas" in lc and "real" in lc and "qtd" not in lc and "vgv" not in lc:
            m["vendas_real"] = c
            break
    m["vgv_prev"] = None
    for c in fn.columns:
        lc = str(c).lower()
        if "vgv" in lc and ("previs" in lc or "previst" in lc) and "real" not in lc:
            m["vgv_prev"] = c
            break
    m["vgv_real"] = None
    for c in fn.columns:
        lc = str(c).lower()
        if "vgv" in lc and "real" in lc:
            m["vgv_real"] = c
            break
    used_q: set[str] = set()
    m["q_fac_p"] = _find_formulario_qtd_col(
        fn, [["facilitadas", "previstas"], ["facilitada", "prevista"]], used=used_q
    )
    m["q_fac_r"] = _find_formulario_qtd_col(
        fn,
        [["facilitadas", "reais"], ["facilitadas", "real"], ["facilitada", "real"]],
        used=used_q,
    )
    m["q_norm_p"] = _find_formulario_qtd_col(
        fn, [["normais", "previstas"], ["normal", "prevista"]], used=used_q
    )
    m["q_norm_r"] = _find_formulario_qtd_col(
        fn, [["normais", "reais"], ["normais", "real"], ["normal", "real"]], used=used_q
    )
    m["nf_prev"], m["nf_real"] = _formulario_nf_pair_cols(fn)
    m["visitas"] = find_column_any(fn, [["visitas", "totais", "esperadas"], ["visitas"]])
    return m


def _formulario_canon_canal(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "(vazio)"
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "(vazio)"
    if "imob" in s:
        return "IMOB"
    if "dv" in s:
        return "DV RJ" if "rj" in s else "DV"
    return str(x).strip()[:32]


def _formulario_num_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _sklearn_tree_split_thresholds_x0(tree_reg: Any) -> list[float]:
    """Limiares de divisão no eixo da *feature* 0 (ilustração 1D de árvore de decisão)."""
    try:
        tr = tree_reg.tree_
        out: list[float] = []
        stack = [0]
        while stack:
            node = int(stack.pop())
            left = int(tr.children_left[node])
            if left == -1:
                continue
            if int(tr.feature[node]) == 0:
                th = float(tr.threshold[node])
                if np.isfinite(th):
                    out.append(th)
            stack.append(int(tr.children_right[node]))
            stack.append(left)
        return sorted(set(out))
    except Exception:
        return []


# --- Decisões automáticas (sem input do utilizador) ---
# Optuna: 0 = número de trials escalado ao tamanho da série em train_eval.
OPTUNA_TRIALS_AUTO = 0
# Top-K no super-ensemble / candidatos: 6 equilibra diversidade e custo (benchmark escolhe melhor combinação).
BLEND_TOP_K_FIXO = 6
RANDOM_SEED = 42
# Split temporal 70% treino / 30% teste nas métricas reportadas (reduz ilusão de desempenho in-sample).
TRAIN_FRAC_FIXO = 0.7
# Holdout final nas métricas do relatório (mais fiável que 100% in-sample).
FULL_PERIOD_TRAIN_FIXO = False
# tqdm nas etapas longas (build master, STL, Optuna). Desative com PREVISAO_HIDE_TQDM=1.
SHOW_ML_PROGRESS = str(os.environ.get("PREVISAO_HIDE_TQDM", "0")).lower() not in (
    "1",
    "true",
    "yes",
)


def _configure_streamlit_progress() -> None:
    """Reduz bufferização da saída e força flush nas barras tqdm (Streamlit / terminal)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            try:
                stream.flush()
            except Exception:
                pass
    try:
        import tqdm.std as _tqdm_std

        if getattr(_tqdm_std.tqdm, "_pv_flush_patched", False):
            return
        _orig_update = _tqdm_std.tqdm.update

        def _update_flush(self: Any, n: int = 1) -> Any:
            r = _orig_update(self, n)
            try:
                fp = getattr(self, "fp", None)
                if fp is not None and hasattr(fp, "flush"):
                    fp.flush()
            except Exception:
                pass
            return r

        _tqdm_std.tqdm.update = _update_flush  # type: ignore[method-assign]
        _tqdm_std.tqdm._pv_flush_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass


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
    """CSS alinhado à ficha Vendas RJ (`aplicar_estilo`): gradiente global, cartão vidro, tipografia Montserrat+Inter."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
@keyframes fichaFadeIn {{
  from {{ opacity: 0; transform: translateY(18px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fichaShimmer {{
  0% {{ background-position: 0% 50%; }}
  100% {{ background-position: 200% 50%; }}
}}
:root {{
  --pv-content-max: min(1320px, 98vw);
}}
html, body {{
  font-family: 'Inter', sans-serif;
  color: {COR_TEXTO_LABEL};
  background: transparent !important;
  background-color: transparent !important;
}}
.stApp,
[data-testid="stApp"] {{
  background:
    linear-gradient(135deg, rgba({RGB_AZUL_CSS}, 0.82) 0%, rgba(30, 58, 95, 0.55) 38%, rgba({RGB_VERMELHO_CSS}, 0.22) 72%, rgba(15, 23, 42, 0.45) 100%),
    url("{BG_HERO_URL}") center / cover no-repeat !important;
  background-attachment: scroll !important;
  background-color: transparent !important;
}}
[data-testid="stAppViewContainer"] {{
  background: transparent !important;
  background-color: transparent !important;
}}
header[data-testid="stHeader"],
[data-testid="stHeader"] {{
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  border: none !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}}
[data-testid="stHeader"] > div,
[data-testid="stHeader"] header {{
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
}}
[data-testid="stDecoration"] {{
  background: transparent !important;
  background-color: transparent !important;
}}
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
[data-testid="stToolbar"] {{
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  color: rgba(255, 255, 255, 0.92) !important;
}}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a {{
  color: rgba(255, 255, 255, 0.92) !important;
  background: transparent !important;
  background-color: transparent !important;
}}
[data-testid="stHeader"] button {{
  background: transparent !important;
  background-color: transparent !important;
}}
[data-testid="stToolbar"] svg {{
  fill: currentColor !important;
  color: inherit !important;
}}
[data-testid="stToolbar"] svg path[stroke] {{
  stroke: currentColor !important;
}}
[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] a:hover,
[data-testid="stHeader"] button:hover {{
  background: rgba(255, 255, 255, 0.12) !important;
}}
/* Tabelas HTML (ex.: Introdução): ocupam toda a largura útil do cartão */
.pv-fullbleed-table-wrap {{
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 0 1rem 0 !important;
  box-sizing: border-box;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}
.pv-fullbleed-table-wrap table {{
  width: 100% !important;
  border-collapse: collapse;
  table-layout: auto;
}}
.pv-fullbleed-table-wrap th,
.pv-fullbleed-table-wrap td {{
  overflow-wrap: anywhere;
  word-break: break-word;
}}
/* Dataframes Streamlit: mesma largura útil (evita coluna estreita no cartão largo) */
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
div[data-testid="stDataFrameResizable"] {{
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}}
[data-testid="stMain"] {{
  padding-left: clamp(2px, 0.9vw, 14px) !important;
  padding-right: clamp(2px, 0.9vw, 14px) !important;
  padding-top: clamp(12px, 3.5vh, 40px) !important;
  padding-bottom: clamp(14px, 4vh, 44px) !important;
  box-sizing: border-box !important;
  background: transparent !important;
}}
section.main > div {{
  padding-top: 0.25rem !important;
  padding-bottom: 0.35rem !important;
}}
.block-container {{
  max-width: var(--pv-content-max) !important;
  width: 100% !important;
  margin-left: auto !important;
  margin-right: auto !important;
  margin-top: clamp(4px, 1vh, 14px) !important;
  margin-bottom: clamp(4px, 1vh, 14px) !important;
  padding: 1rem clamp(0.5rem, 1.6vw, 1.1rem) 1.1rem clamp(0.5rem, 1.6vw, 1.1rem) !important;
  background: rgba(255, 255, 255, 0.97) !important;
  backdrop-filter: blur(14px) saturate(1.2);
  -webkit-backdrop-filter: blur(14px) saturate(1.2);
  border-radius: 18px !important;
  border: 1px solid rgba(255, 255, 255, 0.85) !important;
  box-shadow:
    0 4px 6px -1px rgba({RGB_AZUL_CSS}, 0.08),
    0 24px 48px -12px rgba({RGB_AZUL_CSS}, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
  animation: fichaFadeIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
div[data-testid="stHorizontalBlock"] {{
  width: 100% !important;
}}
h1, h2, h3 {{
  font-family: 'Montserrat', sans-serif !important;
  color: {COR_AZUL_ESC} !important;
  text-align: center;
}}
h4, h5, h6 {{
  font-family: 'Montserrat', sans-serif !important;
  color: {COR_AZUL_ESC} !important;
  text-align: center !important;
}}
.katex-display {{
  margin-left: auto !important;
  margin-right: auto !important;
}}
.pv-section-title {{
  text-align: center !important;
  display: block;
  width: 100%;
  margin-left: auto;
  margin-right: auto;
  font-family: 'Montserrat', sans-serif;
  font-weight: 800;
  color: {COR_AZUL_ESC};
  font-size: 1.05rem;
  margin-top: 0.85rem;
  margin-bottom: 0.5rem;
  text-justify: auto !important;
  hyphens: none !important;
}}
.pv-interpret-wrap {{
  max-width: 100%;
  margin: 0.35rem auto 0.85rem auto;
  padding: 0 0.15rem;
  box-sizing: border-box;
  text-align: center;
}}
.pv-interpret-title {{
  font-family: 'Montserrat', sans-serif;
  font-weight: 700;
  font-size: 0.92rem;
  color: {COR_AZUL_ESC};
  margin: 0.4rem 0 0.25rem 0;
  text-align: center !important;
}}
.pv-interpret-text {{
  font-size: 0.86rem;
  color: #475569 !important;
  line-height: 1.5;
  margin: 0 0 0.5rem 0;
  text-align: center !important;
  text-justify: none !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
}}
[data-testid="stMarkdownContainer"] p.pv-interpret-title,
[data-testid="stMarkdownContainer"] p.pv-interpret-text {{
  text-align: center !important;
  text-justify: auto !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
}}
div[data-testid="stTabs"] {{
  margin-top: 0.4rem;
  margin-bottom: 0.65rem;
}}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 8px;
  background: transparent;
  border-bottom: 2px solid #e2e8f0;
  padding-bottom: 6px;
  justify-content: center;
  flex-wrap: wrap;
}}
div[data-testid="stTabs"] button[data-baseweb="tab"] {{
  border-radius: 12px 12px 0 0 !important;
  font-family: 'Montserrat', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  color: {COR_TEXTO_MUTED} !important;
  background: #f1f5f9 !important;
  border: 1px solid #e2e8f0 !important;
  padding: 0.45rem 0.85rem !important;
}}
div[data-testid="stTabs"] [aria-selected="true"] {{
  color: #ffffff !important;
  background: linear-gradient(180deg, {COR_AZUL_ESC} 0%, #063572 100%) !important;
  border-color: rgba({RGB_AZUL_CSS}, 0.45) !important;
}}
[data-testid="stDialog"] {{
  border-radius: 20px !important;
  border: 2px solid rgba({RGB_AZUL_CSS}, 0.2) !important;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
  color: #334155 !important;
  line-height: 1.55;
  text-align: justify;
  text-justify: inter-word;
  hyphens: auto;
  -webkit-hyphens: auto;
}}
[data-testid="stMarkdownContainer"] p.pv-section-title,
[data-testid="stMarkdownContainer"] p.pv-hero-title,
[data-testid="stMarkdownContainer"] p.pv-hero-sub,
[data-testid="stMarkdownContainer"] p.pv-foot,
[data-testid="stMarkdownContainer"] .pv-hero-block,
[data-testid="stMarkdownContainer"] .pv-foot-wrap {{
  text-align: center !important;
  text-justify: auto !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
}}
[data-testid="stMarkdownContainer"]:has(.pv-hero-block),
[data-testid="stMarkdownContainer"]:has(.pv-foot-wrap) {{
  text-align: center !important;
  width: 100% !important;
  max-width: 100% !important;
}}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {{
  text-align: center !important;
}}
[data-testid="stMetric"] {{
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  text-align: center !important;
}}
[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p {{
  text-align: center !important;
  text-justify: auto !important;
}}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
  text-align: center !important;
  justify-content: center !important;
}}
div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] p {{
  text-align: justify;
  text-justify: inter-word;
}}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
  color: #475569 !important;
}}
/* Legenda sob métricas (Análises): matriz diária — centrada */
[data-testid="stMarkdownContainer"] p.pv-caption-center {{
  text-align: center !important;
  text-justify: auto !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
  color: #475569 !important;
  font-size: 0.875rem !important;
  line-height: 1.5 !important;
  margin: 0.35rem auto 0.85rem auto !important;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}}
[data-testid="stMarkdownContainer"]:has(p.pv-caption-center) {{
  text-align: center !important;
  width: 100% !important;
  max-width: 100% !important;
}}
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {{
  color: {COR_TEXTO_LABEL} !important;
}}
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stCheckbox"] label {{
  color: {COR_TEXTO_LABEL} !important;
}}
div[data-testid="stExpander"] {{
  background: rgba(255, 255, 255, 0.98) !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 16px !important;
}}
div[data-testid="stExpander"] summary {{
  color: #0f172a !important;
}}
div[data-testid="stAlert"] {{
  border-radius: 14px !important;
  border: 2px solid {COR_AZUL_ESC} !important;
  background: #ffffff !important;
  box-shadow: 0 2px 12px rgba({RGB_AZUL_CSS}, 0.1) !important;
}}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"],
div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] * {{
  color: {COR_AZUL_ESC} !important;
}}
div[data-testid="stAlert"] svg {{
  fill: {COR_AZUL_ESC} !important;
  color: {COR_AZUL_ESC} !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 16px !important;
}}
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {{
  background: #ffffff !important;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid {COR_BORDA};
}}
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container {{
  background: transparent !important;
  background-color: transparent !important;
}}
div[data-baseweb="input"] {{
  border-radius: 10px !important;
  border: 1px solid #e2e8f0 !important;
  background-color: {COR_INPUT_BG} !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
div[data-baseweb="input"]:focus-within {{
  border-color: rgba({RGB_AZUL_CSS}, 0.35) !important;
  box-shadow: 0 0 0 3px rgba({RGB_AZUL_CSS}, 0.08) !important;
}}
.stButton > button,
div[data-testid="stButton"] > button {{
  border-radius: 12px !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}}
.stButton > button:hover,
div[data-testid="stButton"] > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 8px 20px -6px rgba({RGB_AZUL_CSS}, 0.25) !important;
}}
/* Botão primário (vermelho): texto e filhos em branco (Streamlit usa p/span e baseButton-primary). */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {{
  background: linear-gradient(180deg, {COR_VERMELHO} 0%, {COR_VERMELHO_ESCURO} 100%) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
  font-family: 'Montserrat', sans-serif !important;
  border-radius: 12px !important;
  padding: 0.65rem 2rem !important;
  min-height: 3rem !important;
  font-size: 1rem !important;
  letter-spacing: 0.02em;
}}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[data-testid="baseButton-primary"] span,
div[data-testid="stButton"] > button[kind="primary"] p,
div[data-testid="stButton"] > button[kind="primary"] span,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] p,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] span {{
  color: #ffffff !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:focus-visible,
.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:focus,
.stButton > button[data-testid="baseButton-primary"]:focus-visible,
.stButton > button[data-testid="baseButton-primary"]:active,
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:focus,
div[data-testid="stButton"] > button[kind="primary"]:focus-visible,
div[data-testid="stButton"] > button[kind="primary"]:active,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:focus,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:focus-visible,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:active {{
  color: #ffffff !important;
  box-shadow: 0 8px 22px -6px rgba({RGB_VERMELHO_CSS}, 0.45) !important;
}}
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span,
.stButton > button[data-testid="baseButton-primary"]:hover p,
.stButton > button[data-testid="baseButton-primary"]:hover span,
div[data-testid="stButton"] > button[kind="primary"]:hover p,
div[data-testid="stButton"] > button[kind="primary"]:hover span,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover p,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover span {{
  color: #ffffff !important;
}}
a[href*="whatsapp.com"],
a[href*="wa.me"] {{
  background-color: #25D366 !important;
  color: #ffffff !important;
  border: 1px solid #1ebe57 !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
}}
.ficha-alert {{
  border-radius: 14px;
  padding: 14px 16px;
  margin: 0 0 12px 0;
  font-size: 0.95rem;
  line-height: 1.55;
  box-sizing: border-box;
}}
.ficha-alert--azul {{
  border: 2px solid {COR_AZUL_ESC};
  background: #ffffff;
  color: {COR_AZUL_ESC};
  box-shadow: 0 2px 12px rgba({RGB_AZUL_CSS}, 0.1);
}}
.ficha-alert--vermelho {{
  border: 2px solid {COR_VERMELHO};
  background: #ffffff;
  color: {COR_AZUL_ESC};
  box-shadow: 0 2px 12px rgba({RGB_VERMELHO_CSS}, 0.12);
}}
/* Bloco do título principal: ocupa a largura e centra texto (evita herdar alinhamento do Streamlit) */
.pv-hero-block {{
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 auto !important;
  padding: 0 !important;
  box-sizing: border-box !important;
  text-align: center !important;
  display: block !important;
}}
.pv-hero-block p {{
  text-align: center !important;
  margin-left: auto !important;
  margin-right: auto !important;
}}
[data-testid="element-container"]:has(.pv-hero-block),
[data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"]:has(.pv-hero-block) {{
  width: 100% !important;
  max-width: 100% !important;
}}
.pv-logo-wrap, .ficha-logo-wrap {{
  text-align: center;
  padding: 0.1rem 0 0.45rem 0;
}}
.pv-logo-wrap img, .ficha-logo-wrap img {{
  max-height: 72px;
  width: auto;
  max-width: min(280px, 85vw);
  height: auto;
  object-fit: contain;
  display: inline-block;
  vertical-align: middle;
}}
.pv-hero-title, .ficha-title {{
  font-family: 'Montserrat', sans-serif;
  font-size: clamp(1.35rem, 3.5vw, 1.75rem);
  font-weight: 900;
  color: {COR_AZUL_ESC};
  text-align: center;
  margin: 0;
  line-height: 1.25;
  letter-spacing: -0.02em;
}}
.pv-hero-sub, .ficha-sub {{
  text-align: center;
  color: #475569;
  font-size: 0.95rem;
  margin: 0.45rem 0 0 0;
  line-height: 1.45;
}}
.pv-bar-wrap, .ficha-hero-bar-wrap {{
  width: 100%;
  max-width: 100%;
  margin: clamp(0.85rem, 2.4vw, 1.2rem) 0;
  padding: 0;
  box-sizing: border-box;
}}
.pv-bar, .ficha-hero-bar {{
  height: 4px;
  width: 100%;
  margin: 0;
  border-radius: 999px;
  background: linear-gradient(90deg, {COR_AZUL_ESC}, {COR_VERMELHO}, {COR_AZUL_ESC});
  background-size: 200% 100%;
  animation: fichaShimmer 4s ease-in-out infinite alternate;
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
div.metric-card {{
  border: 1px solid rgba(226, 232, 240, 0.95);
  background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
  border-radius: 16px;
  padding: 1.1rem 1.35rem 1rem 1.35rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba({RGB_AZUL_CSS}, 0.06);
  border-left: 3px solid {COR_AZUL_ESC};
  transition: box-shadow 0.35s ease, transform 0.35s ease;
  animation: fichaFadeIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
  text-align: center;
}}
div.metric-card:hover {{
  box-shadow: 0 8px 24px -6px rgba({RGB_AZUL_CSS}, 0.12);
  transform: translateY(-1px);
}}
div.metric-card h4 {{
  font-family: 'Montserrat', sans-serif;
  color: {COR_TEXTO_MUTED};
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 800;
  margin: 0 0 0.35rem 0;
  text-align: center;
}}
div.metric-card .val {{
  color: #0f172a;
  font-size: 1.28rem;
  font-weight: 800;
  font-family: 'Montserrat', sans-serif;
  text-align: center;
}}
.stDownloadButton > button {{
  border-radius: 12px !important;
  border: 2px solid {COR_AZUL_ESC} !important;
  color: {COR_AZUL_ESC} !important;
  font-weight: 600 !important;
}}
.pv-foot-wrap {{
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 auto !important;
  text-align: center !important;
  display: block !important;
  box-sizing: border-box !important;
}}
.pv-foot, .footer {{
  text-align: center !important;
  display: block;
  width: 100%;
  box-sizing: border-box;
  padding: 0.85rem 0 0.35rem 0;
  color: {COR_TEXTO_MUTED};
  font-size: 0.82rem;
  margin: 1rem auto 0 auto;
  border-top: 1px solid {COR_BORDA};
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _merge_spreadsheet_config_from_google_sheets(
    ids: dict[str, str], hints: dict[str, list[str]]
) -> None:
    """IDs e *hints* opcionais dentro de `[google_sheets]` (mesmo padrão da Ficha: uma secção só)."""
    try:
        gs = st.secrets.get("google_sheets")
    except Exception:
        return
    if not isinstance(gs, dict):
        return
    for k, v in gs.items():
        kl = str(k).strip().lower()
        if kl in _GOOGLE_SHEETS_RESERVED_LOWER:
            continue
        if k in ids and str(v).strip():
            ids[k] = str(v).strip()
    sid_map = gs.get("spreadsheet_ids")
    if isinstance(sid_map, dict):
        for k, v in dict(sid_map).items():
            if k in ids:
                ids[k] = str(v).strip()
    sh_map = gs.get("sheets")
    if isinstance(sh_map, dict):
        for k, v in dict(sh_map).items():
            if k not in ids:
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 1:
                ids[k] = str(v[0]).strip()
                if len(v) >= 2:
                    g = str(v[1]).strip()
                    cur = hints.get(k, [])
                    hints[k] = [g] + [x for x in cur if x != g]
    gh = gs.get("csv_gid_hints")
    if isinstance(gh, dict):
        for k, v in dict(gh).items():
            if k in hints and isinstance(v, list):
                hints[k] = [str(x) for x in v]


def _data_source_config() -> tuple[dict[str, str], dict[str, list[str]]]:
    ids = dict(DEFAULT_SPREADSHEET_IDS)
    hints = {k: list(v) for k, v in CSV_GID_HINTS.items()}
    try:
        _merge_spreadsheet_config_from_google_sheets(ids, hints)
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
    sa_fp: str,
    spreadsheet_id: str,
    role_key: str,
    hints_tuple: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Sempre tenta API se as credenciais existirem (não depender de use_sa calculado fora do cache).
    gc = gspread_client_from_streamlit()
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
    sa_fp = _sa_fingerprint_for_cache()
    json_ok = (
        service_account_info_from_streamlit_secrets() is not None
        or _service_account_info_from_env() is not None
    )

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
            df, meta = _load_one_role_cached(sa_fp, sid, cfg_key, htuple)
            out[key] = df
            metas[cfg_key] = meta
        except Exception as e:
            errors.append(f"**{human}** (`{sid[:12]}…`): {e}")
    if errors:
        if json_ok and not use_sa:
            modo = (
                "**Conta de serviço:** o JSON foi reconhecido nas secrets, mas o cliente **gspread não iniciou** "
                "(chave privada inválida, JSON truncado ou formato incorreto). Valide o ficheiro no Google Cloud Console.\n\n"
                "Se o cliente iniciar mas a leitura falhar: ative as APIs **Google Sheets** e **Google Drive** no projeto, "
                "e partilhe **cada** planilha com o e-mail `client_email` do JSON como **Leitor**."
            )
        elif use_sa:
            modo = (
                "**Conta de serviço ativa:** confirme partilha **Leitor** com o e-mail `client_email` em **todas** as planilhas "
                "e que o ID na configuração corresponde ao livro correto."
            )
        else:
            modo = (
                "**Sem API:** configure `GOOGLE_SERVICE_ACCOUNT_JSON` na raiz das secrets **ou** dentro de `[google_sheets]` "
                "(string com o JSON completo da conta de serviço). O export CSV anónimo costuma **falhar em servidores na nuvem** "
                "(respostas HTTP 403 ou HTML); a API é o método suportado em produção."
            )
        st.error("Não foi possível carregar todas as fontes.\n\n" + modo + "\n\n" + "\n\n".join(errors))
        st.stop()
    return out, metas, use_sa


def _build_ml_dossie_pack(df_master: pd.DataFrame) -> dict[str, Any]:
    """
    Agregados para o dossiê ML: descritivas, outliers (IQR), correlação, VIF, balanceamento.
    Evita guardar a matriz completa no session_state.
    """
    if df_master is None or len(df_master) < 5:
        return {"erro": "Série insuficiente para o dossiê estatístico."}
    dm = df_master
    idx = pd.DatetimeIndex(pd.to_datetime(dm.index).normalize())

    core = [
        c
        for c in (
            "vol_leads",
            "vol_agend",
            "vol_visit",
            "vol_pastas",
            "target_qtd",
            "target_valor",
        )
        if c in dm.columns
    ]
    num = dm.select_dtypes(include=[np.number])
    fb_cols = [
        c
        for c in num.columns
        if str(c).startswith("fb_vgv_") or str(c).startswith("fb_qtd_")
    ]
    seen = set(core) | set(fb_cols)
    cols_extra = [c for c in num.columns if c not in seen][:30]
    focus_cols = core + fb_cols + cols_extra

    rows_stats: list[dict[str, Any]] = []
    outliers_iqr: list[dict[str, Any]] = []
    for c in focus_cols:
        s = pd.to_numeric(dm[c], errors="coerce").dropna()
        if len(s) < 5:
            continue
        q1_f = float(s.quantile(0.25))
        q3_f = float(s.quantile(0.75))
        iqr = q3_f - q1_f
        lo, hi = q1_f - 1.5 * iqr, q3_f + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        mode_v = s.mode()
        mode_f = float(mode_v.iloc[0]) if len(mode_v) else float("nan")
        rows_stats.append(
            {
                "variavel": c,
                "n": int(len(s)),
                "media": float(s.mean()),
                "mediana": float(s.median()),
                "moda": mode_f,
                "desvio_padrao": float(s.std()),
                "variancia": float(s.var()),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "q1": q1_f,
                "q2": float(s.quantile(0.50)),
                "q3": q3_f,
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "assimetria": float(s.skew()),
                "curtose_excesso": float(s.kurtosis()),
            }
        )
        outliers_iqr.append(
            {
                "variavel": c,
                "n_outliers_iqr": n_out,
                "pct_outliers": float(100.0 * n_out / len(s)),
            }
        )

    sub = dm[focus_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr_labels: list[str] = []
    corr_z: list[list[float]] = []
    pares_mc: list[dict[str, Any]] = []
    if len(sub) > 2 and sub.shape[1] >= 2:
        cm = sub.corr().fillna(0.0)
        corr_labels = [str(x) for x in cm.columns]
        corr_z = [
            [float(cm.iloc[i, j]) for j in range(len(corr_labels))]
            for i in range(len(corr_labels))
        ]
        for i in range(len(cm.columns)):
            for j in range(i + 1, len(cm.columns)):
                v = float(cm.iloc[i, j])
                if abs(v) >= 0.85:
                    pares_mc.append(
                        {
                            "a": str(cm.columns[i]),
                            "b": str(cm.columns[j]),
                            "r_pearson": v,
                        }
                    )
        pares_mc.sort(key=lambda x: -abs(float(x["r_pearson"])))

    vif_rows: list[dict[str, Any]] = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        vif_cols = focus_cols[: min(16, len(focus_cols))]
        Xv = dm[vif_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xv = Xv.loc[:, Xv.std() > 1e-12]
        if Xv.shape[1] >= 2 and len(Xv) > Xv.shape[1] + 2:
            Xc = add_constant(Xv.values, has_constant="add")
            for i in range(1, Xc.shape[1]):
                try:
                    vi = float(variance_inflation_factor(Xc, i))
                except Exception:
                    vi = float("nan")
                vif_rows.append({"variavel": str(Xv.columns[i - 1]), "vif": vi})
    except Exception:
        pass

    balance: dict[str, Any] = {}
    hist_qtd: dict[str, Any] = {}
    hist_vgv: dict[str, Any] = {}
    if "target_qtd" in dm.columns:
        tq = pd.to_numeric(dm["target_qtd"], errors="coerce").fillna(0.0)
        med = float(tq.median())
        above = int((tq > med).sum())
        below = int((tq <= med).sum())
        balance = {
            "mediana_referencia": med,
            "dias_acima_mediana": above,
            "dias_abaixo_igual_mediana": below,
            "proporcao_classe_minoritaria": float(min(above, below) / max(len(tq), 1)),
        }
        counts, edges = np.histogram(tq.to_numpy(dtype=float), bins=min(32, max(8, len(tq) // 5)))
        hist_qtd = {
            "counts": [int(x) for x in counts],
            "edges": [float(x) for x in edges],
        }
    if "target_valor" in dm.columns:
        tv = pd.to_numeric(dm["target_valor"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(tv) > 5 and np.nanmax(tv) > 0:
            counts, edges = np.histogram(tv, bins=min(28, max(8, len(tv) // 6)))
            hist_vgv = {
                "counts": [int(x) for x in counts],
                "edges": [float(x) for x in edges],
            }

    return {
        "descritivas": rows_stats,
        "outliers_iqr": outliers_iqr,
        "correlation_labels": corr_labels,
        "correlation_matrix": corr_z,
        "pares_multicolinearidade": pares_mc[:45],
        "vif": vif_rows,
        "balanceamento_qtd": balance,
        "hist_target_qtd": hist_qtd,
        "hist_target_valor": hist_vgv,
        "n_linhas": int(len(dm)),
        "n_colunas": int(dm.shape[1]),
        "primeira_data": idx.min().strftime("%Y-%m-%d"),
        "ultima_data": idx.max().strftime("%Y-%m-%d"),
    }


def run_training_pipeline(
    dfs: dict[str, pd.DataFrame],
    custom_previsao: dict[str, Any] | None = None,
) -> tuple[
    dict[str, float],
    float,
    float,
    dict[Any, dict[str, Any]],
    dict[Any, dict[str, dict[str, Any]]],
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
]:
    df_master = build_daily_master(dfs, show_progress=SHOW_ML_PROGRESS)
    dossie_ml = _build_ml_dossie_pack(df_master)

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
    por_horizonte: dict[Any, dict[str, Any]] = {}
    best_params_preview: dict[Any, dict[str, dict[str, Any]]] = {}

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

    cvals: list[Any] = []
    if isinstance(custom_previsao, dict) and str(custom_previsao.get("mode")) != "range":
        cvals = list(custom_previsao.get("values") or [])
    range_mode = (
        isinstance(custom_previsao, dict) and str(custom_previsao.get("mode")) == "range"
    )
    ds_raw = (custom_previsao or {}).get("date_start") if range_mode else None
    de_raw = (custom_previsao or {}).get("date_end") if range_mode else None
    range_ok = range_mode and bool(ds_raw) and bool(de_raw)
    vals_ok = isinstance(custom_previsao, dict) and not range_mode and bool(cvals)
    have_custom = range_ok or vals_ok
    extra_custom = 2 if have_custom else 0
    progress = st.progress(0, text="A preparar modelos…")
    total_steps = len(horizontes) * 2 + extra_custom
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
            train_frac=TRAIN_FRAC_FIXO,
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
            train_frac=TRAIN_FRAC_FIXO,
        )
        step += 1

        pred_q = predict_last_row(df_master, rq.pipeline, h)
        pred_v = predict_last_row(df_master, rv.pipeline, h)

        por_horizonte[h] = {"qtd": pack(rq, pred_q), "valor": pack(rv, pred_v)}
        best_params_preview[h] = {"qtd": rq.best_params, "valor": rv.best_params}

    por_custom: dict[str, Any] | None = None
    if have_custom:
        if range_ok:
            d_lo = pd.Timestamp(ds_raw).normalize()
            d_hi = pd.Timestamp(de_raw).normalize()
            if d_hi < d_lo:
                d_lo, d_hi = d_hi, d_lo
            span = int((d_hi - d_lo).days) + 1
            hz_eff = max(14, min(120, span * 2))
            lbl = (
                f"Soma no intervalo [{d_lo.date()} → {d_hi.date()}] "
                "(dias da série estritamente após t)"
            )
            Xqc, yqc = build_xy_custom_date_range(df_master, "target_qtd", d_lo, d_hi)
            Xvc, yvc = build_xy_custom_date_range(df_master, "target_valor", d_lo, d_hi)
            mode = "range"
            vals: list[Any] = []
        else:
            mode = str((custom_previsao or {}).get("mode") or "offsets")
            vals = list(cvals)
            if mode == "offsets":
                hz_eff = max(vals)
                Xqc, yqc = build_xy_custom_offsets(df_master, "target_qtd", vals)
                Xvc, yvc = build_xy_custom_offsets(df_master, "target_valor", vals)
                lbl = "Soma nas defasagens t+k (dias): " + ", ".join(str(v) for v in vals)
            else:
                hz_eff = 18
                Xqc, yqc = build_xy_custom_dom(df_master, "target_qtd", vals)
                Xvc, yvc = build_xy_custom_dom(df_master, "target_valor", vals)
                lbl = "Soma nos dias do mês (estritamente após o dia corrente): " + ", ".join(
                    str(v) for v in vals
                )

        if len(Xqc) < 22 or len(Xvc) < 22:
            dossie_ml["aviso_previsao_custom"] = (
                "Cenário personalizado ignorado: menos de 22 observações válidas após alinhar o alvo."
            )
        else:
            progress.progress(
                max(0.01, (step + 0.5) / total_steps),
                text="Cenário personalizado — volume (Optuna + benchmark)…",
            )
            rqc = train_one_target(
                Xqc,
                yqc,
                horizon=hz_eff,
                target_name="qtd",
                n_trials=OPTUNA_TRIALS_AUTO,
                random_state=RANDOM_SEED,
                optuna_seed=RANDOM_SEED + 7,
                blend_top_k=BLEND_TOP_K_FIXO,
                show_progress=SHOW_ML_PROGRESS,
                full_period_train=FULL_PERIOD_TRAIN_FIXO,
                train_frac=TRAIN_FRAC_FIXO,
            )
            step += 1
            progress.progress(step / total_steps, text="Cenário personalizado — VGV…")
            rvc = train_one_target(
                Xvc,
                yvc,
                horizon=hz_eff,
                target_name="valor",
                n_trials=OPTUNA_TRIALS_AUTO,
                random_state=RANDOM_SEED,
                optuna_seed=RANDOM_SEED + 8,
                blend_top_k=BLEND_TOP_K_FIXO,
                show_progress=SHOW_ML_PROGRESS,
                full_period_train=FULL_PERIOD_TRAIN_FIXO,
                train_frac=TRAIN_FRAC_FIXO,
            )
            step += 1
            if mode == "range":
                pqc = predict_last_row_custom_date_range(
                    df_master, rqc.pipeline, d_lo, d_hi
                )
                pvc = predict_last_row_custom_date_range(
                    df_master, rvc.pipeline, d_lo, d_hi
                )
            elif mode == "offsets":
                pqc = predict_last_row_custom_offsets(df_master, rqc.pipeline, vals)
                pvc = predict_last_row_custom_offsets(df_master, rvc.pipeline, vals)
            else:
                pqc = predict_last_row_custom_dom(df_master, rqc.pipeline, vals)
                pvc = predict_last_row_custom_dom(df_master, rvc.pipeline, vals)
            por_custom = {
                "label": lbl,
                "mode": mode,
                "values": vals,
                "horizon_effective": hz_eff,
                "qtd": pack(rqc, pqc),
                "valor": pack(rvc, pvc),
            }
            if mode == "range":
                por_custom["date_start"] = str(d_lo.date())
                por_custom["date_end"] = str(d_hi.date())
            best_params_preview["custom"] = {
                "qtd": rqc.best_params,
                "valor": rvc.best_params,
            }

    daily_pack = _daily_pack_from_master(df_master)
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
        daily_pack=daily_pack,
        blend_top_k=BLEND_TOP_K_FIXO,
        random_seed=RANDOM_SEED,
    )
    return (
        stats_base,
        ticket,
        conv,
        por_horizonte,
        best_params_preview,
        html_out,
        daily_pack,
        dossie_ml,
        por_custom,
    )


def run_analytics_only_pipeline(
    dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], float, float, dict[str, Any], dict[str, Any]]:
    """Consolida matriz diária, *daily_pack* e dossiê EDA — sem treino de modelos preditivos."""
    df_master = build_daily_master(dfs, show_progress=SHOW_ML_PROGRESS)
    dossie_ml = _build_ml_dossie_pack(df_master)
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
    daily_pack = _daily_pack_from_master(df_master)
    return stats_base, ticket, conv, daily_pack, dossie_ml


def _streamlit_carregar_dados_exploratorios() -> None:
    """Lê planilhas, constrói matriz diária e preenche `dados_exploratorios` (sem treino ML)."""
    with st.spinner("A carregar e consolidar dados…"):
        dfs_e, sheet_e, used_e = build_data_bundle()
    st.session_state["sheet_metas"] = sheet_e
    st.session_state["used_service_account"] = used_e
    try:
        with st.spinner("A preparar indicadores e série diária…"):
            sb, tk, cv, dpk, dml = run_analytics_only_pipeline(dfs_e)
        st.session_state.dados_exploratorios = {
            "stats_base": sb,
            "ticket": tk,
            "conv": cv,
            "daily_pack": dpk,
            "dossie_ml": dml,
            "sheet_metas": sheet_e,
            "df_formulario": dfs_e.get("formulario_previsao"),
        }
        st.rerun()
    except Exception as e:
        st.error(f"Erro ao preparar análises: {e}")
        st.exception(e)


def _render_tab_formulario_previsao_humano() -> None:
    """Painéis a partir da planilha de respostas ao formulário (previsão vs real, filtros)."""
    import plotly.colors as plc
    import plotly.graph_objects as go

    st.markdown(
        '<p class="pv-section-title">Formulário — previsão humana</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:justify;color:#64748b;font-size:0.92rem;margin:0 auto 1rem auto;max-width:100%'>"
        "Respostas ao <strong>Esboço — formulário</strong>: quantidades e VGV previstos versus realizados, por "
        "<em>empreendimento</em>, <em>canal</em>, <em>regional</em> e <em>data de referência</em> (sábado). "
        "Utilize <strong>Análises → Carregar dados</strong> ou <strong>Previsões → Gerar previsões</strong>, "
        "ou o botão abaixo para sincronizar as planilhas.</div>",
        unsafe_allow_html=True,
    )
    if st.button(
        "Sincronizar planilhas (inclui formulário)",
        key="pv_form_reload",
        width="stretch",
    ):
        with st.spinner("A ler Google Sheets…"):
            dfs_f, meta_f, ua_f = build_data_bundle()
        st.session_state["sheet_metas"] = meta_f
        st.session_state["used_service_account"] = ua_f
        st.session_state["formulario_snapshot"] = dfs_f.get("formulario_previsao")
        st.success("Formulário atualizado a partir das fontes configuradas.")
        st.rerun()

    df_raw = _formulario_df_from_state()
    if df_raw is None or df_raw.empty:
        st.info(
            "Nenhuma base do formulário em memória. Carregue os dados na aba **Análises** ou **Previsões**, "
            "ou clique em **Sincronizar planilhas** acima."
        )
        return

    fn = normalize_dataframe_columns(df_raw.copy())
    mc = _formulario_map_columns(fn)

    def _ref_series() -> pd.Series:
        if not mc.get("ref") or mc["ref"] not in fn.columns:
            return pd.Series(pd.NaT, index=fn.index)
        return pd.to_datetime(fn[mc["ref"]], errors="coerce")

    rs = _ref_series()
    fn["_ref_d"] = rs.dt.normalize()

    emp_c = mc.get("empreendimento")
    reg_c = mc.get("regional")
    canal_c = mc.get("canal")
    regiao_c = mc.get("regiao")

    st.markdown("##### Filtros")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        emp_opts: list[str] = []
        if emp_c and emp_c in fn.columns:
            emp_opts = sorted({str(x).strip() for x in fn[emp_c].dropna().unique() if str(x).strip()})
        sel_emp = st.multiselect("Empreendimento", options=emp_opts, default=[], key="pv_ff_emp")
    with f2:
        dlist = sorted({_ for _ in fn["_ref_d"].dropna().unique()})
        d_labels = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in dlist]
        sel_d = st.multiselect("Data de referência (sábado)", options=d_labels, default=[], key="pv_ff_dt")
    with f3:
        canal_opts: list[str] = []
        if canal_c and canal_c in fn.columns:
            canal_opts = sorted({_formulario_canon_canal(x) for x in fn[canal_c].unique()})
        sel_canal = st.multiselect("Canal", options=canal_opts, default=[], key="pv_ff_canal")
    with f4:
        reg_opts: list[str] = []
        if reg_c and reg_c in fn.columns:
            reg_opts = sorted({str(x).strip() for x in fn[reg_c].dropna().unique() if str(x).strip()})
        sel_reg = st.multiselect("Regional ou IMOB", options=reg_opts, default=[], key="pv_ff_reg")

    dff = fn.copy()
    if sel_emp and emp_c and emp_c in dff.columns:
        dff = dff[dff[emp_c].astype(str).isin(sel_emp)]
    if sel_d:
        pick = {pd.Timestamp(s).normalize() for s in sel_d}
        dff = dff[dff["_ref_d"].isin(pick)]
    if sel_canal and canal_c and canal_c in dff.columns:
        dff = dff[dff[canal_c].map(_formulario_canon_canal).isin(sel_canal)]
    if sel_reg and reg_c and reg_c in dff.columns:
        dff = dff[dff[reg_c].astype(str).isin(sel_reg)]

    if dff.empty:
        st.warning("Sem linhas após aplicar os filtros.")
        return

    st.caption(f"**{len(dff):,}** linhas (após filtros) · **{dff.shape[1]}** colunas na base bruta.")

    qfp, qfr = mc.get("q_fac_p"), mc.get("q_fac_r")
    qnp, qnr = mc.get("q_norm_p"), mc.get("q_norm_r")
    vgp, vgr = mc.get("vgv_prev"), mc.get("vgv_real")
    vprev, vreal = mc.get("vendas_prev"), mc.get("vendas_real")
    nf_p = mc.get("nf_prev")

    s_qfp = _formulario_num_series(dff, qfp)
    s_qfr = _formulario_num_series(dff, qfr)
    s_qnp = _formulario_num_series(dff, qnp)
    s_qnr = _formulario_num_series(dff, qnr)
    s_vgp = _formulario_num_series(dff, vgp)
    s_vgr = _formulario_num_series(dff, vgr)

    tot_fac_p = float(s_qfp.sum())
    tot_fac_r = float(s_qfr.sum())
    tot_norm_p = float(s_qnp.sum())
    tot_norm_r = float(s_qnr.sum())
    tot_vgv_p = float(s_vgp.sum())
    tot_vgv_r = float(s_vgr.sum())

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("QTD facilitadas — previstas", f"{tot_fac_p:,.0f}")
    with m2:
        st.metric("QTD facilitadas — reais", f"{tot_fac_r:,.0f}")
    with m3:
        st.metric("QTD normais — previstas", f"{tot_norm_p:,.0f}")
    with m4:
        st.metric("QTD normais — reais", f"{tot_norm_r:,.0f}")
    with m5:
        st.metric("VGV previsto (soma)", f"R$ {tot_vgv_p/1e6:.2f} mi")
    with m6:
        st.metric("VGV real (soma)", f"R$ {tot_vgv_r/1e6:.2f} mi")

    if mc.get("erro") and mc["erro"] in dff.columns:
        err = pd.to_numeric(dff[mc["erro"]], errors="coerce").dropna()
        if len(err):
            e1, e2 = st.columns(2)
            with e1:
                st.metric("Erro de previsão — média |·|", f"{float(err.abs().mean()):,.2f}")
            with e2:
                st.metric("Erro de previsão — mediana", f"{float(err.median()):,.2f}")

    _ipc = {"displayModeBar": True, "displaylogo": False}
    pal = plc.qualitative.Bold + plc.qualitative.Pastel1 + plc.qualitative.Dark24

    def _stack_xy_color(
        title: str,
        xcol: str | None,
        ccol: str | None,
        yser: pd.Series,
        *,
        height: int = 400,
    ) -> go.Figure | None:
        if not xcol or xcol not in dff.columns or not ccol or ccol not in dff.columns:
            return None
        sub = pd.DataFrame({xcol: dff[xcol].astype(str), ccol: dff[ccol].astype(str), "_y": yser})
        sub = sub.groupby([xcol, ccol], as_index=False)["_y"].sum()
        sub = sub[sub["_y"] != 0]
        if sub.empty:
            return None
        xcats = sorted(sub[xcol].unique().tolist())
        colors = sorted(sub[ccol].unique().tolist())
        fig = go.Figure()
        for i, lab in enumerate(colors):
            chunk = sub[sub[ccol] == lab]
            yv = chunk.set_index(xcol)["_y"].reindex(xcats).fillna(0).values
            fig.add_trace(
                go.Bar(
                    name=str(lab)[:46],
                    x=xcats,
                    y=yv,
                    marker_color=pal[i % len(pal)],
                    text=[f"{v:.0f}" if v > 0 else "" for v in yv],
                    textposition="inside",
                    insidetextfont=dict(color="white", size=10),
                )
            )
        fig.update_layout(
            **_plotly_layout_direcional(
                title=title,
                height=height,
                barmode="stack",
                legend=_plotly_legend_bottom(),
                xaxis_tickangle=-34,
                margin=dict(t=100, l=52, r=36, b=max(128, min(240, 28 + 7 * max((len(str(x)) for x in xcats), default=0)))),
            )
        )
        return fig

    def _group_prev_real(
        title: str,
        xcol: str | None,
        s_pre: pd.Series,
        s_re: pd.Series,
        lab_p: str,
        lab_r: str,
        *,
        height: int = 380,
    ) -> go.Figure | None:
        if not xcol or xcol not in dff.columns:
            return None
        sub = pd.DataFrame({xcol: dff[xcol].astype(str), "_p": s_pre, "_r": s_re})
        g = sub.groupby(xcol, as_index=False).agg(_p=("_p", "sum"), _r=("_r", "sum"))
        if g.empty:
            return None
        g = g.sort_values("_p", ascending=False)
        xcats = g[xcol].tolist()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=lab_p,
                x=xcats,
                y=g["_p"],
                marker_color=PLOT_AZUL,
                text=[f"{v:.0f}" for v in g["_p"]],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                name=lab_r,
                x=xcats,
                y=g["_r"],
                marker_color=PLOT_ACCENT,
                text=[f"{v:.0f}" for v in g["_r"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            **_plotly_layout_direcional(
                title=title,
                height=height,
                barmode="group",
                legend=_plotly_legend_bottom(),
                xaxis_tickangle=-34,
                margin=dict(t=100, l=52, r=36, b=max(128, min(260, 28 + 7 * max((len(str(x)) for x in xcats), default=0)))),
            )
        )
        return fig

    st.markdown("##### Previsão por regional — empilhado (QTD)")
    st.caption("Cada cor é um **empreendimento**; o eixo X é **Regional ou IMOB**.")
    ra, rb = st.columns(2)
    with ra:
        fg = _stack_xy_color(
            "Previsão por produto — Vendas facilitadas (QTD previstas)",
            reg_c,
            emp_c,
            s_qfp,
        )
        if fg:
            st.plotly_chart(fg, width="stretch", config=_ipc)
        else:
            st.caption("—")
    with rb:
        fg = _stack_xy_color(
            "Previsão por produto — Vendas normais (QTD previstas)",
            reg_c,
            emp_c,
            s_qnp,
        )
        if fg:
            st.plotly_chart(fg, width="stretch", config=_ipc)
        else:
            st.caption("—")

    st.markdown("##### Previsão por produto — canal (QTD e VGV)")
    st.caption("Eixo X: **empreendimento**; empilhamento por **canal** (DV / IMOB).")
    if canal_c and emp_c:
        sc = dff[canal_c].map(_formulario_canon_canal)
        ca, cb = st.columns(2)
        with ca:
            sub = pd.DataFrame({emp_c: dff[emp_c].astype(str), "__canal__": sc, "_y": s_qnp})
            sub = sub.groupby([emp_c, "__canal__"], as_index=False)["_y"].sum()
            sub = sub[sub["_y"] != 0]
            if not sub.empty:
                xcats = sorted(sub[emp_c].unique().tolist())
                fig = go.Figure()
                for i, lab in enumerate(sorted(sub["__canal__"].unique().tolist())):
                    chunk = sub[sub["__canal__"] == lab]
                    yv = chunk.set_index(emp_c)["_y"].reindex(xcats).fillna(0).values
                    fig.add_trace(
                        go.Bar(
                            name=str(lab),
                            x=xcats,
                            y=yv,
                            marker_color=pal[i % len(pal)],
                            text=[f"{v:.0f}" if v > 0 else "" for v in yv],
                            textposition="inside",
                            insidetextfont=dict(color="white", size=10),
                        )
                    )
                fig.update_layout(
                    **_plotly_layout_direcional(
                        title="Previsão por produto — Normais (QTD)",
                        height=400,
                        barmode="stack",
                        legend=_plotly_legend_bottom(),
                        xaxis_tickangle=-34,
                        margin=dict(t=100, l=52, r=36, b=168),
                    )
                )
                st.plotly_chart(fig, width="stretch", config=_ipc)
            else:
                st.caption("—")
        with cb:
            sub = pd.DataFrame({emp_c: dff[emp_c].astype(str), "__canal__": sc, "_y": s_vgp})
            sub = sub.groupby([emp_c, "__canal__"], as_index=False)["_y"].sum()
            sub = sub[sub["_y"] != 0]
            if not sub.empty:
                xcats = sorted(sub[emp_c].unique().tolist())
                fig = go.Figure()
                for i, lab in enumerate(sorted(sub["__canal__"].unique().tolist())):
                    chunk = sub[sub["__canal__"] == lab]
                    yv = chunk.set_index(emp_c)["_y"].reindex(xcats).fillna(0).values
                    lbl_txt = [f"{float(v)/1e3:.1f} mil" if v > 0 else "" for v in yv]
                    fig.add_trace(
                        go.Bar(
                            name=str(lab),
                            x=xcats,
                            y=yv,
                            marker_color=pal[i % len(pal)],
                            text=lbl_txt,
                            textposition="inside",
                            insidetextfont=dict(color="white", size=9),
                        )
                    )
                fig.update_layout(
                    **_plotly_layout_direcional(
                        title="Previsão por produto — VGV",
                        height=400,
                        barmode="stack",
                        legend=_plotly_legend_bottom(),
                        xaxis_tickangle=-34,
                        yaxis_title="R$",
                        margin=dict(t=100, l=52, r=36, b=168),
                    )
                )
                st.plotly_chart(fig, width="stretch", config=_ipc)
            else:
                st.caption("—")
    else:
        st.caption("Colunas de empreendimento ou canal ausentes — gráficos por canal omitidos.")

    st.markdown("##### Previsão por produto — empilhado por regional (QTD)")
    st.caption("Eixo X: **empreendimento**; cores: **Regional ou IMOB**.")
    pc1, pc2 = st.columns(2)
    with pc1:
        fg = _stack_xy_color(
            "Previsão por produto — Facilitadas",
            emp_c,
            reg_c,
            s_qfp,
        )
        if fg:
            st.plotly_chart(fg, width="stretch", config=_ipc)
        else:
            st.caption("—")
    with pc2:
        fg = _stack_xy_color(
            "Previsão por produto — Normais",
            emp_c,
            reg_c,
            s_qnp,
        )
        if fg:
            st.plotly_chart(fg, width="stretch", config=_ipc)
        else:
            st.caption("—")

    if nf_p and nf_p in dff.columns and reg_c and reg_c in dff.columns:
        st.markdown("##### Previsão agregada por tipo (Normal / Facilitada / …)")
        vp = (
            _formulario_num_series(dff, vprev)
            if vprev
            else pd.Series(1.0, index=dff.index, dtype=float)
        )
        sub = pd.DataFrame(
            {
                reg_c: dff[reg_c].astype(str),
                nf_p: dff[nf_p].astype(str).replace("nan", "(vazio)"),
                "_y": vp,
            }
        )
        sub = sub.groupby([reg_c, nf_p], as_index=False)["_y"].sum()
        sub = sub[sub["_y"] != 0]
        if not sub.empty:
            xcats = sorted(sub[reg_c].unique().tolist())
            fig = go.Figure()
            for i, lab in enumerate(sorted(sub[nf_p].unique().tolist())):
                chunk = sub[sub[nf_p] == lab]
                yv = chunk.set_index(reg_c)["_y"].reindex(xcats).fillna(0).values
                fig.add_trace(
                    go.Bar(
                        name=str(lab)[:40],
                        x=xcats,
                        y=yv,
                        marker_color=pal[i % len(pal)],
                        text=[f"{v:.0f}" if v > 0 else "" for v in yv],
                        textposition="outside",
                    )
                )
            fig.update_layout(
                **_plotly_layout_direcional(
                    title="Previsão — soma de «Vendas Previsão» por regional e tipo",
                    height=400,
                    barmode="stack",
                    legend=_plotly_legend_bottom(),
                    xaxis_tickangle=-28,
                    margin=dict(t=100, l=52, r=36, b=138),
                )
            )
            st.plotly_chart(fig, width="stretch", config=_ipc)
        else:
            st.caption("Sem dados para o gráfico de tipos.")

    st.markdown("##### Previsão × Real — por **Regional ou IMOB**")
    gx1, gx2 = st.columns(2)
    with gx1:
        fg = _group_prev_real(
            "Vendas Previsão × Real — Facilitadas (QTD)",
            reg_c,
            s_qfp,
            s_qfr,
            "QTD facilitadas previstas",
            "QTD facilitadas reais",
        )
        if fg:
            st.plotly_chart(fg, width="stretch", config=_ipc)
        else:
            st.caption("—")
    with gx2:
        fg = _group_prev_real(
            "Vendas Previsão × Real — Normais (QTD)",
            reg_c,
            s_qnp,
            s_qnr,
            "QTD normais previstas",
            "QTD normais reais",
        )
        if fg:
            st.plotly_chart(fg, width="stretch", config=_ipc)
        else:
            st.caption("—")

    st.markdown("##### Previsão × Real — por **empreendimento** (top por volume previsto)")
    if emp_c and emp_c in dff.columns:
        sub = pd.DataFrame(
            {
                emp_c: dff[emp_c].astype(str),
                "_p": s_qnp + s_qfp,
                "_pf": s_qfp,
                "_pr": s_qfr,
                "_qn": s_qnp,
                "_qr": s_qnr,
            }
        )
        g = sub.groupby(emp_c, as_index=False).agg(
            _p=("_p", "sum"),
            _pf=("_pf", "sum"),
            _pr=("_pr", "sum"),
            _qn=("_qn", "sum"),
            _qr=("_qr", "sum"),
        )
        g = g.sort_values("_p", ascending=False).head(22)
        if not g.empty:
            xcats = g[emp_c].tolist()

            def _pair_fig(title: str, yp: str, yr: str, npref: str, nrref: str) -> go.Figure:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        name=npref,
                        x=xcats,
                        y=g[yp],
                        marker_color=PLOT_AZUL,
                        text=[f"{v:.0f}" for v in g[yp]],
                        textposition="outside",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name=nrref,
                        x=xcats,
                        y=g[yr],
                        marker_color=PLOT_ACCENT,
                        text=[f"{v:.0f}" for v in g[yr]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    **_plotly_layout_direcional(
                        title=title,
                        height=440,
                        barmode="group",
                        legend=_plotly_legend_bottom(),
                        xaxis_tickangle=-40,
                        margin=dict(t=100, l=52, r=36, b=188),
                    )
                )
                return fig

            gy1, gy2 = st.columns(2)
            with gy1:
                st.plotly_chart(
                    _pair_fig(
                        "Facilitadas — previstas × reais",
                        "_pf",
                        "_pr",
                        "QTD facilitadas previstas",
                        "QTD facilitadas reais",
                    ),
                    width="stretch",
                    config=_ipc,
                )
            with gy2:
                st.plotly_chart(
                    _pair_fig(
                        "Normais — previstas × reais",
                        "_qn",
                        "_qr",
                        "QTD normais previstas",
                        "QTD normais reais",
                    ),
                    width="stretch",
                    config=_ipc,
                )
    else:
        st.caption("—")

    st.markdown("##### Tabelas resumo — **Região** × canal (DV / IMOB)")
    if regiao_c and regiao_c in dff.columns and canal_c and canal_c in dff.columns:
        dpx = dff[[regiao_c, canal_c]].copy()
        dpx["_can"] = dpx[canal_c].map(_formulario_canon_canal)
        dpx["_q_prev"] = s_qnp + s_qfp
        dpx["_q_real"] = s_qnr + s_qfr
        pv = (
            dpx.pivot_table(
                index=regiao_c,
                columns="_can",
                values="_q_prev",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        pr = (
            dpx.pivot_table(
                index=regiao_c,
                columns="_can",
                values="_q_real",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        pv.columns = [str(c) if c != regiao_c else "Região" for c in pv.columns]
        pr.columns = [str(c) if c != regiao_c else "Região" for c in pr.columns]

        def _add_total_row(tab: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
            if tab.empty:
                return tab
            row = {tab.columns[0]: "Total geral"}
            for c in num_cols:
                if c in tab.columns:
                    row[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0).sum()
            return pd.concat([tab, pd.DataFrame([row])], ignore_index=True)

        num_p = [c for c in pv.columns if c != "Região"]
        num_r = [c for c in pr.columns if c != "Região"]
        st.caption("Totais de **QTD** (normais + facilitadas) por região e canal.")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Previsão (QTD)**")
            st.dataframe(_add_total_row(pv, num_p), width="stretch", hide_index=True)
        with t2:
            st.markdown("**Realizado (QTD)**")
            st.dataframe(_add_total_row(pr, num_r), width="stretch", hide_index=True)

        if vgp and vgr:
            dpx["_v_prev"] = s_vgp
            dpx["_v_real"] = s_vgr
            pv2 = (
                dpx.pivot_table(
                    index=regiao_c,
                    columns="_can",
                    values="_v_prev",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
            )
            pr2 = (
                dpx.pivot_table(
                    index=regiao_c,
                    columns="_can",
                    values="_v_real",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
            )
            pv2.columns = [str(c) if c != regiao_c else "Região" for c in pv2.columns]
            pr2.columns = [str(c) if c != regiao_c else "Região" for c in pr2.columns]
            st.markdown("**VGV por região e canal (R$)**")
            t3, t4 = st.columns(2)
            with t3:
                st.caption("Previsão")
                n3 = [c for c in pv2.columns if c != "Região"]
                st.dataframe(_add_total_row(pv2, n3), width="stretch", hide_index=True)
            with t4:
                st.caption("Realizado")
                n4 = [c for c in pr2.columns if c != "Região"]
                st.dataframe(_add_total_row(pr2, n4), width="stretch", hide_index=True)
    else:
        st.caption("Defina colunas **Região** e **Canal** na planilha para ver as tabelas cruzadas.")

    with st.expander("Pré-visualização dos dados filtrados (primeiras linhas)", expanded=False):
        show_cols = [
            c
            for c in [
                mc.get("ref"),
                emp_c,
                reg_c,
                canal_c,
                regiao_c,
                qfp,
                qfr,
                qnp,
                qnr,
                vgp,
                vgr,
                vprev,
                vreal,
                mc.get("erro"),
            ]
            if c and c in dff.columns
        ]
        st.dataframe(dff[show_cols].head(200), width="stretch", hide_index=True)


def _render_tab_introducao() -> None:
    """Secção estática: metodologia, métricas, modelos e exemplos (sem dados reais)."""
    import plotly.graph_objects as go

    def _ix(html: str) -> None:
        st.markdown(
            f'<div style="text-align:justify;text-justify:inter-word;hyphens:auto;-webkit-hyphens:auto;max-width:100%;margin:0 auto 1rem auto;color:#334155;line-height:1.65;font-size:0.95rem;box-sizing:border-box">{html}</div>',
            unsafe_allow_html=True,
        )

    def _lx(s: str) -> None:
        st.latex(s)

    _ipc = {"displayModeBar": True, "displaylogo": False}
    st.markdown(
        '<p class="pv-section-title">Introdução</p>',
        unsafe_allow_html=True,
    )
    _ix(
        "Esta aplicação estima <strong>quantidade</strong> e <strong>VGV</strong> a partir da matriz diária consolidada. "
        "Em cada instante <em>t</em>, o vetor <strong>X</strong> incorpora exclusivamente informação disponível até <em>t</em>, "
        "garantindo, deste modo, coerência temporal no treino e na previsão."
    )

    st.markdown("#### 1 · Problema e alvos")
    _ix(
        "Trata-se de um problema de regressão ao longo do tempo: o alvo <strong>Y</strong> é escalar e o preditor "
        "<strong>X</strong> é vetorial, ambos indexados por <em>t</em>. Ademais, cada componente de <strong>X</strong> "
        "deve depender apenas do passado observado até <em>t</em>, sob pena de enviesar a validação fora da amostra."
    )
    _ix(
        "<strong>Horizonte fixo H ∈ {3, 7, 30}.</strong> Para cada <em>t</em>, <strong>Y</strong> define-se como a soma de vendas "
        "(em quantidade ou valor) nos <strong>H</strong> primeiros dias da <em>série</em> imediatamente posteriores a <em>t</em>, "
        "considerando apenas os dias efetivamente presentes no índice temporal."
    )
    _ix(
        "<strong>Intervalo [d₁, d₂].</strong> Na aba <strong>Previsões</strong>, quando indicadas duas datas, "
        "define-se um alvo adicional correspondente à soma no calendário entre esses limites, complementando, portanto, "
        "os horizontes fixos H ∈ {3, 7, 30}."
    )
    _ix("Exemplo simbólico (H = 7, quantidade <em>q</em>):")
    _lx(r"Y_t^{(7)} = \sum_{d \in \mathcal{D}_{t,7}} q_d")
    _ix(
        "O conjunto 𝒟<sub>t,7</sub> reúne, por ordem temporal, até <strong>sete</strong> instantes da série "
        "<strong>estritamente posteriores</strong> a <em>t</em>, ou seja, o primeiro dia útil após <em>t</em> e os seis seguintes "
        "no índice observado."
    )

    st.markdown("#### 2 · Métricas de erro (regressão)")
    _ix(
        "No conjunto de <strong>teste</strong> — tipicamente os últimos ~30% da série, preservando a ordem cronológica — "
        "consideram-se <em>n</em> pares (<em>yᵢ</em>, <em>ŷᵢ</em>) para as definições seguintes:"
    )
    _lx(r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|")
    _ix("O MAE mede o erro médio absoluto e exprime-se na mesma unidade que o alvo, facilitando, assim, a leitura em escala de negócio.")
    _lx(r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
    _ix("O RMSE penaliza de forma mais acentuada os erros grandes; consequentemente, é mais sensível a outliers do que o MAE.")
    _lx(r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}")
    _ix(
        "O R² compara o modelo a uma referência ingênua que prediz sempre a média <em>ȳ</em> do bloco de teste; "
        "valores mais altos indicam melhor ajuste, embora devam ser interpretados em conjunto com MAE e RMSE."
    )

    st.markdown("#### 3 · Acurácia direcional (auxiliar)")
    _ix(
        "<strong>Acurácia direcional:</strong> calcula-se a mediana de <em>Y</em> no treino e, em teste, compara-se se "
        "<em>y</em> e <em>ŷ</em> ficam ambos acima ou ambos abaixo dessa mediana. Este indicador complementa, portanto, "
        "MAE, RMSE e R², focando na capacidade de antever a direção do desvio face ao patamar histórico."
    )
    _lx(
        r"\hat{b}_i = \mathbb{1}\left[ \hat{y}_i > \mathrm{mediana}_{\mathrm{train}}(Y) \right],\quad"
        r" b_i = \mathbb{1}\left[ y_i > \mathrm{mediana}_{\mathrm{train}}(Y) \right]"
    )
    _lx(r"\mathrm{Acc}_{\mathrm{dir}} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\hat{b}_i = b_i]")

    st.markdown("#### 4 · Pré-processamento e validação")
    _ix(
        "As entradas numéricas dos <em>pipelines</em> passam por <strong>MinMaxScaler (0, 1)</strong> com <em>clip</em>, "
        "uniformizando escalas entre algoritmos heterogêneos."
    )
    _ix(
        "A <strong>partição</strong> principal segue aproximadamente 70% treino e 30% teste, em ordem temporal; "
        "no bloco de treino, recorre-se a <strong>TimeSeriesSplit</strong> para manter a validação alinhada com a natureza da série."
    )
    _ix(
        "A <strong>Optuna</strong> (TPE e <em>MedianPruner</em>) minimiza o MAE médio nos <em>folds</em>, "
        "incorporando ainda penalização pela <strong>variância entre folds</strong>, de modo a privilegiar soluções estáveis no tempo."
    )
    _ix(
        "Sempre que o algoritmo o suporta, aplicam-se <strong>pesos amostrais</strong> com maior ênfase em observações recentes "
        "e em regiões de cauda de volume, refletindo, assim, a relevância operacional desses regimes."
    )

    st.markdown("#### 5 · Famílias de modelos candidatas")
    rows_mod = [
        {
            "Família": "Ridge / ElasticNet",
            "Papel": "Regularização L2 / L1+L2",
            "Notas": "Rápidos, baseline linear; escalados 0–1.",
        },
        {
            "Família": "SVR (RBF)",
            "Papel": "Kernel não linear",
            "Notas": "Regularização C, ε; sensível à escala (tratada pelo scaler).",
        },
        {
            "Família": "k-NN",
            "Papel": "Regressão por vizinhos",
            "Notas": "k elevado, pesos por distância; não extrapola muito além do treino.",
        },
        {
            "Família": "Random Forest / ExtraTrees",
            "Papel": "Árvores em *bagging*",
            "Notas": "Robustas a não linearidades; importância de *features*.",
        },
        {
            "Família": "LightGBM / XGBoost / CatBoost",
            "Papel": "Gradient boosting",
            "Notas": "Candidatos fortes em séries tabulares; muitos hiperparâmetros vão à Optuna.",
        },
        {
            "Família": "NGBoost",
            "Papel": "Boosting probabilístico",
            "Notas": "Usado sobretudo no alvo VGV quando estável.",
        },
        {
            "Família": "Baselines / *stacks* leves",
            "Papel": "Referência e combinação",
            "Notas": "Mediana, médias móveis, *stacks* simples para comparação.",
        },
    ]
    st.dataframe(pd.DataFrame(rows_mod), width="stretch", hide_index=True)
    _ix(
        f"Os <strong>{BLEND_TOP_K_FIXO}</strong> modelos com menor MAE na validação interna integram um <strong>ensemble</strong> "
        r"cujos pesos seguem ∝ exp(−(MAE−MAE<sub>min</sub>)/τ). Contudo, caso não exista ganho estatístico claro, "
        f"preserva-se unicamente o melhor modelo individual."
    )

    st.markdown("##### Ilustrações por tipo de modelo (dados sintéticos, 1D)")
    _ix(
        "Para cada família, mostra-se uma relação fictícia entre uma variável <em>x</em> e um alvo <em>y</em> ruidoso. "
        "Na prática, o vetor <strong>X</strong> tem muitas dimensões; estes painéis servem apenas para <strong>intuir a forma</strong> "
        "da função que cada método aprende (reta, degraus, curva suave, média de árvores, etc.)."
    )
    try:
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from plotly.subplots import make_subplots

        rng = np.random.default_rng(42)
        n_pt = 55
        xs = np.sort(rng.uniform(0, 10, n_pt))
        y_true = 2.0 + 0.82 * xs
        ys = y_true + rng.normal(0, 1.05, n_pt)
        x_col = xs.reshape(-1, 1)
        x_grid = np.linspace(0, 10, 200).reshape(-1, 1)

        def _lo_intro(
            title: str, *, yl: str = "y", xl: str = "x", height: int = 300
        ) -> dict[str, Any]:
            return _plotly_layout_direcional(
                title=title,
                height=height,
                xaxis_title=f"Eixo X — {xl}",
                yaxis_title=f"Eixo Y — {yl}",
                margin=dict(t=88, l=48, r=28, b=118),
                legend=_plotly_legend_bottom(),
            )

        sc_d = dict(size=7, color=PLOT_MUTED, opacity=0.72, line=dict(width=0.5, color="#fff"))

        lr = LinearRegression().fit(x_col, ys)
        y_lin = lr.predict(x_grid)
        y_on_pts = lr.predict(x_col)
        xx_res: list[Any] = []
        yy_res: list[Any] = []
        for i in range(n_pt):
            xx_res.extend([xs[i], xs[i], None])
            yy_res.extend([ys[i], float(y_on_pts[i]), None])
        fig_lin = go.Figure()
        fig_lin.add_trace(
            go.Scatter(
                x=xx_res,
                y=yy_res,
                mode="lines",
                line=dict(color="rgba(148,163,184,0.55)", width=1.2),
                name="Resíduo (vertical)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_lin.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="Observações", marker=sc_d)
        )
        fig_lin.add_trace(
            go.Scatter(
                x=x_grid.ravel(),
                y=y_lin,
                mode="lines",
                name="ŷ = β₀ + β₁x",
                line=dict(color=PLOT_AZUL, width=3),
            )
        )
        fig_lin.update_layout(**_lo_intro("Regressão linear (ideia Ridge / ElasticNet)"))

        knn = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(x_col, ys)
        y_knn = knn.predict(x_grid)
        _xq_demo = np.array([[5.0]], dtype=float)
        _, kn_ix = knn.kneighbors(_xq_demo)
        _x_band = xs[kn_ix[0]]
        b_lo, b_hi = float(_x_band.min()), float(_x_band.max())
        fig_knn = go.Figure()
        fig_knn.add_vrect(
            x0=b_lo,
            x1=b_hi,
            fillcolor="rgba(14, 116, 144, 0.14)",
            layer="below",
            line_width=0,
        )
        fig_knn.add_vline(
            x=5.0,
            line_width=1.5,
            line_dash="dash",
            line_color=PLOT_VERMELHO,
        )
        fig_knn.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="Observações", marker=sc_d)
        )
        fig_knn.add_trace(
            go.Scatter(
                x=x_grid.ravel(),
                y=y_knn,
                mode="lines",
                name="Média local (vizinhos)",
                line=dict(color=PLOT_VERMELHO, width=2.8),
            )
        )
        fig_knn.update_layout(**_lo_intro("k-NN — vizinhança em x (exemplo em x = 5)"))

        dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(x_col, ys)
        y_dt = dt.predict(x_grid)
        fig_dt = go.Figure()
        fig_dt.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="Observações", marker=sc_d)
        )
        fig_dt.add_trace(
            go.Scatter(
                x=x_grid.ravel(),
                y=y_dt,
                mode="lines",
                name="Predição em degraus",
                line=dict(color=PLOT_AZUL, width=2.5),
            )
        )
        for th in _sklearn_tree_split_thresholds_x0(dt):
            fig_dt.add_vline(
                x=th,
                line_width=1.2,
                line_dash="dot",
                line_color="rgba(100,116,139,0.72)",
            )
        fig_dt.update_layout(**_lo_intro("Árvore de decisão — cortes em x e patamares"))

        rf = RandomForestRegressor(
            n_estimators=60, max_depth=4, random_state=42, n_jobs=-1
        ).fit(x_col, ys)
        y_rf = rf.predict(x_grid)
        tricol = [PLOT_AZUL, PLOT_VERMELHO, PLOT_ACCENT]
        fig_rf = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.36, 0.64],
            vertical_spacing=0.13,
            subplot_titles=(
                "Três árvores de decisão do ensemble (cada curva = degraus)",
                "Random Forest: muitas árvores (cinza) + média agregada",
            ),
        )
        for j, est in enumerate(rf.estimators_[:3]):
            ytj = est.predict(x_grid)
            fig_rf.add_trace(
                go.Scatter(
                    x=x_grid.ravel(),
                    y=ytj,
                    mode="lines",
                    name=f"Árvore {j + 1}",
                    line=dict(width=2.6, color=tricol[j % len(tricol)]),
                ),
                row=1,
                col=1,
            )
        fig_rf.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name="Observações",
                marker=sc_d,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for est in rf.estimators_[:12]:
            yt = est.predict(x_grid)
            fig_rf.add_trace(
                go.Scatter(
                    x=x_grid.ravel(),
                    y=yt,
                    mode="lines",
                    line=dict(width=0.9, color="rgba(100,116,139,0.28)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
        fig_rf.add_trace(
            go.Scatter(
                x=x_grid.ravel(),
                y=y_rf,
                mode="lines",
                name="Média das árvores",
                line=dict(color=PLOT_AZUL, width=3),
            ),
            row=2,
            col=1,
        )
        fig_rf.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=sc_d,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig_rf.update_layout(
            **_plotly_layout_direcional(
                title="Random Forest — árvores de decisão e voto pela média",
                height=560,
                margin=dict(t=92, l=48, r=28, b=138),
                legend=_plotly_legend_bottom(),
            )
        )
        fig_rf.update_xaxes(title_text="Eixo X — x", row=1, col=1)
        fig_rf.update_xaxes(title_text="Eixo X — x", row=2, col=1)
        fig_rf.update_yaxes(title_text="Eixo Y — ŷ", row=1, col=1)
        fig_rf.update_yaxes(title_text="Eixo Y — ŷ", row=2, col=1)

        gbr = GradientBoostingRegressor(
            n_estimators=14,
            max_depth=2,
            learning_rate=0.45,
            subsample=0.9,
            random_state=42,
        ).fit(x_col, ys)
        staged = list(gbr.staged_predict(x_grid))
        fig_gb = go.Figure()
        fig_gb.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="Observações", marker=sc_d)
        )
        show_st = [0, 3, 7, len(staged) - 1]
        dash_c = ["dot", "dash", "longdash", "solid"]
        for i, si in enumerate(show_st):
            lab = f"Após árvore {si + 1}" if si < len(staged) - 1 else "Modelo completo"
            fig_gb.add_trace(
                go.Scatter(
                    x=x_grid.ravel(),
                    y=staged[si],
                    mode="lines",
                    name=lab,
                    line=dict(
                        width=2.4 if i == len(show_st) - 1 else 1.6,
                        color=PLOT_VERMELHO if i == len(show_st) - 1 else PLOT_ACCENT,
                        dash=dash_c[i],
                    ),
                )
            )
        fig_gb.update_layout(**_lo_intro("Gradient boosting — soma progressiva de árvores fracas"))

        svr = SVR(kernel="rbf", C=12.0, epsilon=0.2, gamma=0.28).fit(x_col, ys)
        y_svr = svr.predict(x_grid)
        eps_svr = float(getattr(svr, "epsilon", 0.2) or 0.2)
        xg1 = x_grid.ravel()
        y_up = y_svr.ravel() + eps_svr
        y_lo = y_svr.ravel() - eps_svr
        fig_svr = go.Figure()
        fig_svr.add_trace(
            go.Scatter(
                x=np.concatenate([xg1, xg1[::-1]]),
                y=np.concatenate([y_up, y_lo[::-1]]),
                fill="toself",
                fillcolor="rgba(14, 116, 144, 0.16)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"Tubo insensível ε = {eps_svr:g}",
                hoverinfo="skip",
            )
        )
        fig_svr.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="Observações", marker=sc_d)
        )
        fig_svr.add_trace(
            go.Scatter(
                x=xg1,
                y=y_svr,
                mode="lines",
                name="Kernel RBF (f)",
                line=dict(color=PLOT_AZUL, width=3),
            )
        )
        fig_svr.update_layout(**_lo_intro("SVR — tubo ε e curva suave (kernel RBF)"))

        r1a, r1b = st.columns(2)
        with r1a:
            st.plotly_chart(fig_lin, width="stretch", config=_ipc)
            st.caption(
                "**Lineares:** reta que minimiza o erro; os traços verticais cinzentos mostram o **resíduo** em cada ponto. "
                "Com Ridge/ElasticNet, os coeficientes encolhem (L2/L1)."
            )
        with r1b:
            st.plotly_chart(fig_knn, width="stretch", config=_ipc)
            st.caption(
                "**k-NN:** a **faixa esbatida** contém os *k*=5 pontos de treino mais próximos de *x*=5 (linha tracejada); "
                "a curva vermelha é a média **ponderada pela distância** ao longo de todo o eixo."
            )

        r2a, r2b = st.columns(2)
        with r2a:
            st.plotly_chart(fig_dt, width="stretch", config=_ipc)
            st.caption(
                "**Árvore única:** as **linhas tracejadas** marcam os cortes em *x*; entre dois cortes consecutivos a predição é **constante** (degraus)."
            )
        with r2b:
            st.plotly_chart(fig_rf, width="stretch", config=_ipc)
            st.caption(
                "**Random Forest:** em cima, **três árvores** reais do modelo (cada cor = uma função em degraus). "
                "Em baixo, dezenas de árvores (cinza) e a **média** (linha forte) — é assim que o *ensemble* reduz variância."
            )

        r3a, r3b = st.columns(2)
        with r3a:
            st.plotly_chart(fig_gb, width="stretch", config=_ipc)
            st.caption(
                "**Boosting (LightGBM / XGBoost / CatBoost):** cada curva tracejada acumula mais uma **árvore fraca** que mexe nos resíduos; "
                "a linha vermelha final é o modelo completo."
            )
        with r3b:
            st.plotly_chart(fig_svr, width="stretch", config=_ipc)
            st.caption(
                "**SVR:** a faixa semitransparente é o **tubo ε** (erros pequenos não penalizam); a linha azul é a função suave dada pelo **kernel RBF**."
            )

        med_y = float(np.median(ys))
        fig_base = go.Figure()
        fig_base.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", name="Observações", marker=sc_d)
        )
        fig_base.add_trace(
            go.Scatter(
                x=[0, 10],
                y=[med_y, med_y],
                mode="lines",
                name="Baseline mediana",
                line=dict(color=PLOT_VERMELHO, width=2.8, dash="dash"),
            )
        )
        fig_base.update_layout(**_lo_intro("Baseline — predição constante (mediana)"))
        st.plotly_chart(fig_base, width="stretch", config=_ipc)
        st.caption(
            "**Stack / ensemble:** combina predições de vários modelos (por média ponderada ou meta-modelo); "
            "a mediana é a referência ingénua mais simples."
        )
    except Exception as _e_intro_viz:
        st.caption(
            f"Não foi possível gerar as ilustrações automáticas (dependências ou ambiente: {_e_intro_viz!s}). "
            "Instale `scikit-learn` e confirme a versão compatível com o projeto."
        )

    st.markdown("#### 6 · LightGBM — parâmetros (referência)")
    _ix(
        "A tabela seguinte resume nomes usuais de hiperparâmetros; os valores efetivos resultam, contudo, "
        "da otimização via Optuna e podem variar entre execuções."
    )
    ex_hp = pd.DataFrame(
        [
            {"Parâmetro": "n_estimators", "Significado": "Número de árvores na sequência", "Exemplo": "200–2000"},
            {"Parâmetro": "learning_rate", "Significado": "Passo de cada árvore", "Exemplo": "0.01–0.2"},
            {"Parâmetro": "num_leaves", "Significado": "Complexidade folhas (2^profundidade efetiva)", "Exemplo": "16–256"},
            {"Parâmetro": "min_child_samples", "Significado": "Mínimo de amostras por folha", "Exemplo": "5–120"},
            {"Parâmetro": "subsample / colsample", "Significado": "Amostragem de linhas / colunas", "Exemplo": "0.6–1.0"},
            {"Parâmetro": "reg_alpha / reg_lambda", "Significado": "Regularização L1 / L2", "Exemplo": "1e-8–10"},
        ]
    )
    st.dataframe(ex_hp, width="stretch", hide_index=True)

    st.markdown("#### 7 · Dossiê (EDA)")
    _ix(
        "O <strong>Dossiê</strong> agrega, de forma estruturada, estatísticas descritivas, análise de outliers, "
        "matrizes de correlação, VIF, histogramas e o perfil semanal — permitindo, assim, uma leitura integrada da qualidade dos dados."
    )
    st.markdown(
        """
<div class="pv-fullbleed-table-wrap">
<table style="font-size:0.92rem;color:#334155">
<thead><tr style="border-bottom:2px solid #e2e8f0">
<th style="padding:10px 12px;text-align:left;width:22%">Bloco</th>
<th style="padding:10px 12px;text-align:left">Conteúdo típico</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 12px;text-align:left;vertical-align:top">Descritivas</td>
<td style="padding:10px 12px;text-align:left">Média, mediana, moda, DP, quartis, percentis, curtose</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 12px;text-align:left;vertical-align:top">Outliers IQR</td>
<td style="padding:10px 12px;text-align:left">Limites Q1−1,5·IQR e Q3+1,5·IQR</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 12px;text-align:left;vertical-align:top">Correlação</td>
<td style="padding:10px 12px;text-align:left">Pearson entre <em>features</em> e alvos amostrados</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 12px;text-align:left;vertical-align:top">VIF</td>
<td style="padding:10px 12px;text-align:left">Redundância linear (subconjunto de colunas)</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 12px;text-align:left;vertical-align:top">Distribuições</td>
<td style="padding:10px 12px;text-align:left">Histogramas de qtd e VGV diários</td></tr>
<tr><td style="padding:10px 12px;text-align:left;vertical-align:top">Perfil semanal</td>
<td style="padding:10px 12px;text-align:left">Médias por dia da semana</td></tr>
</tbody></table></div>
""",
        unsafe_allow_html=True,
    )
    _ix("<strong>Correlação de Pearson</strong> (amostral) entre variáveis <em>x</em> e <em>y</em>:")
    _lx(
        r"r_{xy} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}"
        r"{\sqrt{\sum_i (x_i-\bar{x})^2}\,\sqrt{\sum_i (y_i-\bar{y})^2}}"
    )
    _ix("<strong>VIF</strong> da variável <em>j</em> (via regressão OLS de <em>xⱼ</em> nas demais colunas do subconjunto):")
    _lx(r"\mathrm{VIF}_j = \frac{1}{1 - R^2_j}")
    _ix(
        "Aqui, R²ⱼ designa o coeficiente de determinação dessa regressão auxiliar; assim, valores de VIF elevados "
        "(&gt; 5–10) sugerem colinearidade potencialmente problemática."
    )
    _ix(
        "<strong>Outliers (regra IQR):</strong> define-se IQR = Q3 − Q1; os pontos fora de [L<sub>inf</sub>, L<sub>sup</sub>] "
        "são destacados para análise, todavia <strong>não</strong> são eliminados automaticamente do alvo de negócio."
    )
    _lx(
        r"L_{\mathrm{inf}} = Q_1 - 1.5 \cdot \mathrm{IQR},\quad "
        r"L_{\mathrm{sup}} = Q_3 + 1.5 \cdot \mathrm{IQR}"
    )

    st.markdown("#### 8 · Gráficos de exemplo")
    _ix(
        "Os gráficos abaixo replicam os tipos utilizados na aba <strong>Análises</strong>; porém, os dados são sintéticos "
        "e destinam-se apenas a ilustrar a leitura visual."
    )
    g1, g2 = st.columns(2)
    with g1:
        fig_ex = go.Figure(
            go.Bar(
                x=[0.28, 0.19, 0.14, 0.11, 0.09, 0.08, 0.06],
                y=[
                    "Lag vendas 1d",
                    "Leads méd. 7d",
                    "Dia da semana",
                    "Feriado (fwd)",
                    "Macro (z)",
                    "Ticket lag",
                    "Outras",
                ],
                orientation="h",
                marker_color=PLOT_AZUL,
            )
        )
        fig_ex.update_layout(
            **_plotly_layout_direcional(
                title="Importância relativa (exemplo)",
                height=360,
                showlegend=False,
                margin=dict(l=160, r=24, t=52, b=40),
                xaxis_title="Peso relativo",
            ),
        )
        st.plotly_chart(fig_ex, width="stretch", config=_ipc)
    with g2:
        lbl = ["A", "B", "C", "D"]
        z = [[1.0, 0.35, -0.12, 0.08], [0.35, 1.0, 0.22, -0.05], [-0.12, 0.22, 1.0, 0.41], [0.08, -0.05, 0.41, 1.0]]
        fig_c = go.Figure(
            go.Heatmap(
                z=z,
                x=lbl,
                y=lbl,
                colorscale=[[0, PLOT_VERMELHO], [0.5, "#f1f5f9"], [1, PLOT_AZUL]],
                zmin=-1,
                zmax=1,
                zmid=0,
            )
        )
        fig_c.update_layout(
            **_plotly_layout_direcional(
                title="Correlação (exemplo)",
                height=360,
                margin=dict(l=48, r=48, t=52, b=48),
            ),
        )
        st.plotly_chart(fig_c, width="stretch", config=_ipc)

    fig_ser = go.Figure()
    fig_ser.add_trace(
        go.Scatter(
            y=[12, 15, 11, 18, 22, 19, 24, 21, 26, 23],
            mode="lines+markers",
            name="Série A",
            line=dict(color=PLOT_AZUL, width=2.5),
        )
    )
    fig_ser.add_trace(
        go.Scatter(
            y=[13, 14, 13, 17, 21, 20, 23, 22, 25, 24],
            mode="lines",
            name="Série B",
            line=dict(color=PLOT_VERMELHO, width=2, dash="dot"),
        )
    )
    fig_ser.update_layout(
        **_plotly_layout_direcional(
            title="Duas séries (exemplo)",
            height=340,
            xaxis_title="Eixo X — ordem no tempo",
            yaxis_title="Eixo Y — valor",
            legend=_plotly_legend_bottom(),
            margin=dict(t=100, l=56, r=48, b=108),
        ),
    )
    st.plotly_chart(fig_ser, width="stretch", config=_ipc)

    st.markdown("#### 9 · Abas")
    _ix("Segue-se um resumo orientativo das secções da aplicação:")
    st.markdown(
        """
<div class="pv-fullbleed-table-wrap" style="margin-bottom:1.5rem !important">
<table style="font-size:0.92rem;color:#334155">
<thead><tr style="border-bottom:2px solid #e2e8f0">
<th style="padding:10px 14px;text-align:left;width:22%">Secção</th>
<th style="padding:10px 14px;text-align:left">Conteúdo</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 14px;text-align:left;vertical-align:top"><strong>Introdução</strong></td>
<td style="padding:10px 14px;text-align:left">Conceitos e notação</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 14px;text-align:left;vertical-align:top"><strong>Análises</strong></td>
<td style="padding:10px 14px;text-align:left">Indicadores e gráficos da série diária</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 14px;text-align:left;vertical-align:top"><strong>Formulário</strong></td>
<td style="padding:10px 14px;text-align:left">Previsões humanas do Esboço: gráficos previsão × real, filtros e tabelas por região/canal</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 14px;text-align:left;vertical-align:top"><strong>Previsões</strong></td>
<td style="padding:10px 14px;text-align:left">Treino, tabela e exportação HTML</td></tr>
<tr style="border-bottom:1px solid #f1f5f9"><td style="padding:10px 14px;text-align:left;vertical-align:top"><strong>Dossiê</strong></td>
<td style="padding:10px 14px;text-align:left">EDA; métricas e importância de <em>features</em> na sub-aba <strong>Modelos</strong> (com treino)</td></tr>
<tr><td style="padding:10px 14px;text-align:left;vertical-align:top"><strong>Apêndice</strong></td>
<td style="padding:10px 14px;text-align:left">Metodologia e hiperparâmetros</td></tr>
</tbody></table></div>
""",
        unsafe_allow_html=True,
    )
    _ix(
        "Repare-se que a opção <em>Carregar dados</em>, nas abas <strong>Análises</strong>, <strong>Dossiê</strong> e <strong>Apêndice</strong>, "
        "atualiza apenas a matriz e os resumos exploratórios, <strong>sem</strong> executar o treino preditivo."
    )


def _render_streamlit_ml_feature_importance(
    por_h: dict[Any, dict[str, Any]],
    plot_config: dict[str, Any],
) -> None:
    """Gráficos de importância de *features* por horizonte (dados do pipeline após treino)."""
    import plotly.graph_objects as go

    _tem_imp = any(
        (por_h.get(h) or {}).get("qtd", {}).get("importance_names")
        or (por_h.get(h) or {}).get("valor", {}).get("importance_names")
        for h in (3, 7, 30)
    )
    if _tem_imp:
        st.markdown(
            '<p class="pv-section-title">Importância de features (modelo operacional por horizonte)</p>',
            unsafe_allow_html=True,
        )
        for h in (3, 7, 30):
            sub = por_h.get(h) or {}
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"**H={h}d** — volume (quantidade)")
                names = sub.get("qtd", {}).get("importance_names") or []
                vals = sub.get("qtd", {}).get("importance_vals") or []
                if names and vals:
                    top_n = min(18, len(names))
                    fig_i = go.Figure(
                        go.Bar(
                            x=vals[:top_n][::-1],
                            y=names[:top_n][::-1],
                            orientation="h",
                            marker_color=PLOT_AZUL,
                            marker_line=dict(width=0),
                        )
                    )
                    fig_i.update_layout(
                        **_plotly_layout_direcional(
                            height=28 * top_n + 88,
                            margin=dict(l=220, r=40, t=40, b=48),
                            xaxis_title="Importância",
                            showlegend=False,
                        ),
                    )
                    st.plotly_chart(fig_i, width="stretch", config=plot_config)
                else:
                    st.caption("—")
            with c2:
                st.caption(f"**H={h}d** — VGV")
                names = sub.get("valor", {}).get("importance_names") or []
                vals = sub.get("valor", {}).get("importance_vals") or []
                if names and vals:
                    top_n = min(18, len(names))
                    fig_iv = go.Figure(
                        go.Bar(
                            x=vals[:top_n][::-1],
                            y=names[:top_n][::-1],
                            orientation="h",
                            marker_color=PLOT_VERMELHO,
                            marker_line=dict(width=0),
                        )
                    )
                    fig_iv.update_layout(
                        **_plotly_layout_direcional(
                            height=28 * top_n + 88,
                            margin=dict(l=220, r=40, t=40, b=48),
                            xaxis_title="Importância",
                            showlegend=False,
                        ),
                    )
                    st.plotly_chart(fig_iv, width="stretch", config=plot_config)
                else:
                    st.caption("—")
        st.divider()
    else:
        st.markdown(
            '<p style="text-align:center;color:#64748b;font-size:0.88rem;margin-top:0.5rem">'
            "Importância de <em>features</em> por horizonte: disponível após <strong>Gerar previsões</strong>.</p>",
            unsafe_allow_html=True,
        )
        st.divider()


def _render_streamlit_tab_analises(
    daily: dict[str, Any],
    stats_base: dict[str, float],
    ticket: float,
    conv: float,
) -> None:
    import plotly.graph_objects as go

    st.markdown(
        '<p class="pv-section-title">Indicadores consolidados (histórico diário)</p>',
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Leads", f"{stats_base['leads']:,.0f}")
    with k2:
        st.metric("Agendamentos", f"{stats_base['agend']:,.0f}")
    with k3:
        st.metric("Visitas", f"{stats_base['visit']:,.0f}")
    with k4:
        st.metric("Pastas", f"{stats_base['pastas']:,.0f}")
    with k5:
        st.metric("Vendas (qtd)", f"{stats_base['vendas']:,.0f}")

    k6, k7, k8 = st.columns(3)
    with k6:
        st.metric("VGV acumulado", f"R$ {stats_base['vgv']/1e6:.2f} M")
    with k7:
        st.metric("Ticket médio", f"R$ {ticket/1e3:,.0f} k")
    with k8:
        st.metric("Conversão leads → vendas", f"{conv:.2f} %")

    st.markdown(
        "<p class=\"pv-caption-center\">"
        f"A matriz diária subjacente contém {daily.get('n_rows', 0):,} dias e {daily.get('n_features', 0)} colunas "
        "após engenharia de atributos.</p>",
        unsafe_allow_html=True,
    )

    dates = daily.get("dates") or []
    if not dates:
        st.warning("Não há datas na série temporal; portanto, os gráficos não podem ser exibidos.")
        return

    _pc = {"displayModeBar": True, "displaylogo": False, "scrollZoom": True}
    fig_f = go.Figure()
    fig_f.add_trace(
        go.Scatter(
            x=dates,
            y=daily["vol_leads"],
            name="Leads",
            stackgroup="one",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(4, 66, 143, 0.42)",
            hovertemplate="<b>Leads</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=dates,
            y=daily["vol_agend"],
            name="Agend.",
            stackgroup="one",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(203, 9, 53, 0.32)",
            hovertemplate="<b>Agend.</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=dates,
            y=daily["vol_visit"],
            name="Visitas",
            stackgroup="one",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(14, 116, 144, 0.38)",
            hovertemplate="<b>Visitas</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=dates,
            y=daily["vol_pastas"],
            name="Pastas",
            stackgroup="one",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(4, 66, 143, 0.22)",
            hovertemplate="<b>Pastas</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=dates,
            y=daily["target_qtd"],
            name="Vendas (linha)",
            yaxis="y2",
            mode="lines",
            line=dict(color=PLOT_AZUL, width=2.8),
            hovertemplate="%{x}<br>Quantidade: %{y:,.0f}<extra></extra>",
        )
    )
    vgv_mi = [float(v) / 1e6 for v in daily["target_valor"]]
    fig_f.add_trace(
        go.Scatter(
            x=dates,
            y=vgv_mi,
            name="VGV (linha)",
            yaxis="y3",
            mode="lines",
            line=dict(color=PLOT_VERMELHO, width=2.2, dash="dot"),
            hovertemplate="%{x}<br>Valor: %{y:.3f} mi R$<extra></extra>",
        )
    )
    _lo_f = _plotly_layout_direcional(
        title="Funil diário (empilhado) e alvos",
        height=500,
        legend=_plotly_legend_bottom(),
        margin=dict(t=108, l=58, r=62, b=120),
    )
    _xr_f = _plotly_xaxis_range_from_dates(dates)
    if _xr_f:
        _lo_f = {**_lo_f, "xaxis": {**_lo_f["xaxis"], "range": _xr_f}}
    fig_f.update_layout(
        **_lo_f,
        xaxis_title="Eixo X — data",
        yaxis_title="Eixo Y₁ — contagens do funil (empilhadas, / dia)",
        yaxis2=dict(
            overlaying="y",
            side="right",
            title=dict(
                text="Eixo Y₂ — quantidade vendida (/ dia)",
                font=dict(size=12, color=PLOT_AZUL),
            ),
            showgrid=False,
            tickfont=dict(size=11),
        ),
        yaxis3=dict(
            anchor="free",
            overlaying="y",
            side="right",
            position=0.97,
            title=dict(
                text="Eixo Y₃ — valor vendido (mi R$ / dia)",
                font=dict(size=12, color=PLOT_VERMELHO),
            ),
            showgrid=False,
            tickfont=dict(size=11),
        ),
    )
    st.plotly_chart(fig_f, width="stretch", config=_pc)
    _st_interpretacao_grafico("Funil e alvos", _interpret_text_funil_vendas(daily))

    tq = pd.to_numeric(pd.Series(daily["target_qtd"]), errors="coerce").fillna(0.0)
    roll7 = tq.rolling(7, min_periods=1).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(
        go.Scatter(
            x=dates,
            y=tq.tolist(),
            name="Observado (dia a dia)",
            line=dict(color=PLOT_MUTED, width=1.2),
        )
    )
    fig_roll.add_trace(
        go.Scatter(
            x=dates,
            y=roll7.tolist(),
            name="Suavização (MM7)",
            line=dict(color=PLOT_AZUL, width=2.8),
        )
    )
    _lo_roll = _plotly_layout_direcional(
        title="Vendas diárias e suavização (7 dias)",
        height=380,
        xaxis_title="Eixo X — data",
        yaxis_title="Eixo Y — quantidade vendida (/ dia)",
        legend=_plotly_legend_bottom(),
        margin=dict(t=100, l=56, r=52, b=112),
    )
    _xr_roll = _plotly_xaxis_range_from_dates(dates)
    if _xr_roll:
        _lo_roll = {**_lo_roll, "xaxis": {**_lo_roll["xaxis"], "range": _xr_roll}}
    fig_roll.update_layout(**_lo_roll)
    st.plotly_chart(fig_roll, width="stretch", config=_pc)
    _st_interpretacao_grafico("Vendas e média móvel 7 dias", _interpret_text_rolagem_7d(daily))

    fig_sc = go.Figure()
    fig_sc.add_trace(
        go.Scatter(
            x=daily["vol_leads"],
            y=daily["target_qtd"],
            mode="markers",
            marker=dict(size=7, opacity=0.5, color=PLOT_AZUL, line=dict(width=0.5, color="#ffffff")),
            name="Leads × vendas (mesmo dia)",
        )
    )
    fig_sc.update_layout(
        **_plotly_layout_direcional(
            title="Dispersão: leads vs vendas (mesmo dia)",
            height=400,
            xaxis_title="Eixo X — leads (/ dia)",
            yaxis_title="Eixo Y — quantidade vendida (/ dia)",
        ),
    )
    st.plotly_chart(fig_sc, width="stretch", config=_pc)
    _st_interpretacao_grafico("Leads × vendas (mesmo dia)", _interpret_text_leads_vendas(daily))

    dl = daily.get("dow_labels") or []
    fig_d = go.Figure()
    fig_d.add_trace(
        go.Bar(
            x=dl,
            y=daily.get("dow_mean_qtd") or [],
            name="Barras — quantidade",
            marker_color=PLOT_AZUL,
            marker_line=dict(width=0),
        )
    )
    fig_d.add_trace(
        go.Bar(
            x=dl,
            y=[float(v) / 1e6 for v in (daily.get("dow_mean_valor") or [])],
            name="Barras — VGV",
            marker_color=PLOT_VERMELHO,
            marker_line=dict(width=0),
            yaxis="y2",
        )
    )
    fig_d.update_layout(
        **_plotly_layout_direcional(
            title="Perfil por dia da semana (média histórica)",
            height=420,
            barmode="group",
            xaxis_title="Eixo X — dia da semana",
            yaxis_title="Eixo Y₁ — média de quantidade (/ dia)",
            legend=_plotly_legend_bottom(),
            margin=dict(t=100, l=56, r=52, b=112),
            yaxis2=dict(
                overlaying="y",
                side="right",
                title="Eixo Y₂ — média de VGV (mi R$ / dia)",
                showgrid=False,
                tickfont=dict(size=11),
            ),
        ),
    )
    st.plotly_chart(fig_d, width="stretch", config=_pc)
    _st_interpretacao_grafico("Média por dia da semana", _interpret_text_dow(daily))

    cl = daily.get("corr_labels") or []
    cz = daily.get("corr_z") or []
    if len(cl) >= 2 and cz:
        fig_h = go.Figure(
            data=go.Heatmap(
                z=cz,
                x=cl,
                y=cl,
                colorscale=[
                    [0.0, PLOT_VERMELHO],
                    [0.5, "#f1f5f9"],
                    [1.0, PLOT_AZUL],
                ],
                zmid=0,
                zmin=-1,
                zmax=1,
                hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
            )
        )
        fig_h.update_layout(
            **_plotly_layout_direcional(
                title="Correlação (Pearson) — variáveis-chave",
                height=max(340, 30 * len(cl)),
                margin=dict(l=130, r=48, t=56, b=130),
            ),
        )
        st.plotly_chart(fig_h, width="stretch", config=_pc)
        _st_interpretacao_grafico("Matriz de correlação (Pearson)", _interpret_text_correlation_matrix(cl, cz))
        st.caption(
            "A matriz resume associações lineares entre pares de variáveis; para ver a nuvem de pontos entre **quaisquer duas** "
            "colunas numéricas (incluindo `target_qtd` e `target_valor`), utilize a secção **Dispersão — par de variáveis** abaixo — "
            "ou o gráfico fixo **leads × vendas** mais acima nesta aba."
        )
    else:
        st.markdown(
            '<p style="text-align:center;color:#94a3b8;font-size:0.88rem">A matriz de correlação não está disponível, '
            "seja por falta de colunas numéricas suficientes, seja por dados insuficientes.</p>",
            unsafe_allow_html=True,
        )

    ns_pick = daily.get("numeric_series_for_picker") or {}
    n_pick_cols = len(ns_pick)
    if n_pick_cols >= 2:
        ref_len = len(next(iter(ns_pick.values())))
        if any(len(v) != ref_len for v in ns_pick.values()):
            st.warning(
                "As séries numéricas do seletor têm comprimentos inconsistentes; volte a carregar ou a regenerar a matriz diária."
            )
        else:
            st.markdown(
                '<p class="pv-section-title">Dispersão — par de variáveis</p>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Escolha duas colunas numéricas da matriz diária: cada ponto corresponde ao mesmo dia civil (pares alinhados por data). "
                "O coeficiente r de Pearson resume a associação linear da nuvem. "
                "Para cruzar com vendas, utilize `target_qtd` ou `target_valor` em X ou em Y; para comparar métricas do funil, "
                "selecione, por exemplo, `vol_leads` e `vol_visit`."
            )
            opts_pick = sorted(ns_pick.keys())
            ix_x = opts_pick.index("vol_leads") if "vol_leads" in opts_pick else 0
            ix_y = opts_pick.index("target_qtd") if "target_qtd" in opts_pick else min(1, len(opts_pick) - 1)
            if ix_y == ix_x and len(opts_pick) > 1:
                ix_y = (ix_x + 1) % len(opts_pick)
            cxa, cxb = st.columns(2)
            with cxa:
                sel_x = st.selectbox(
                    "Variável — eixo horizontal (X)",
                    options=opts_pick,
                    index=ix_x,
                    key="pv_pair_var_x",
                )
            with cxb:
                sel_y = st.selectbox(
                    "Variável — eixo vertical (Y)",
                    options=opts_pick,
                    index=ix_y,
                    key="pv_pair_var_y",
                )
            if sel_x == sel_y:
                st.warning("Selecione duas variáveis distintas para exibir a dispersão e o coeficiente de correlação.")
            else:
                xv = ns_pick.get(sel_x) or []
                yv = ns_pick.get(sel_y) or []
                if len(xv) != len(yv):
                    st.warning(
                        "O comprimento das séries escolhidas não coincide; por favor, volte a carregar os dados."
                    )
                else:
                    r_xy, n_xy = _pearson_pairwise_complete(xv, yv)
                    mxa, mxb = st.columns(2)
                    with mxa:
                        st.metric("r de Pearson (X, Y)", f"{r_xy:+.3f}" if r_xy is not None else "—")
                    with mxb:
                        st.metric("Observações válidas (ambas finitas)", f"{n_xy:,}")
                    xlab = str(sel_x) if len(str(sel_x)) <= 48 else str(sel_x)[:45] + "…"
                    ylab = str(sel_y) if len(str(sel_y)) <= 48 else str(sel_y)[:45] + "…"
                    use_dates = len(dates) == len(xv) == len(yv)
                    htempl = (
                        f"Data=%{{customdata}}<br>{xlab}=%{{x:.4g}}<br>{ylab}=%{{y:.4g}}<extra></extra>"
                        if use_dates
                        else f"{xlab}=%{{x:.4g}}<br>{ylab}=%{{y:.4g}}<extra></extra>"
                    )
                    fig_pair = go.Figure(
                        data=go.Scatter(
                            x=xv,
                            y=yv,
                            mode="markers",
                            marker=dict(
                                size=8,
                                opacity=0.55,
                                color=PLOT_AZUL,
                                line=dict(width=0.45, color="#fff"),
                            ),
                            customdata=dates if use_dates else None,
                            hovertemplate=htempl,
                        )
                    )
                    fig_pair.update_layout(
                        **_plotly_layout_direcional(
                            title=f"Dispersão: {xlab} × {ylab}",
                            height=440,
                            xaxis_title=f"Eixo X — {xlab}",
                            yaxis_title=f"Eixo Y — {ylab}",
                            margin=dict(t=56, l=56, r=48, b=100),
                        ),
                    )
                    st.plotly_chart(fig_pair, width="stretch", config=_pc)
                    _txt_r = (
                        f"O coeficiente de Pearson estimado é r = {r_xy:+.3f}, com n = {n_xy} dias em que ambos os valores são finitos."
                        if r_xy is not None
                        else "O coeficiente de Pearson não é definido (poucos pontos ou variância nula num dos eixos)."
                    )
                    _st_interpretacao_grafico(
                        "Leitura",
                        f"«{sel_x}» (eixo X) e «{sel_y}» (eixo Y) estão emparelhados por dia civil. {_txt_r} "
                        "Trata-se de correlação linear contemporânea; assim, não implica causalidade nem substitui modelos com defasagem explícita.",
                    )
    elif n_pick_cols == 1:
        st.caption(
            "Para comparar duas variáveis na dispersão, a matriz diária precisa de pelo menos **duas** colunas numéricas além do índice."
        )
    elif not ns_pick:
        st.caption(
            "Para habilitar o seletor de pares numéricos, execute novamente **Carregar dados** ou **Gerar previsões**, "
            "pois resultados antigos em cache podem não incluir `numeric_series_for_picker`."
        )

    macro = daily.get("macro") or {}
    if macro:
        fig_m = go.Figure()
        for name, series in macro.items():
            s = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0.0)
            if s.std() and float(s.std()) > 0:
                z = ((s - s.mean()) / s.std()).tolist()
            else:
                z = [0.0] * len(s)
            fig_m.add_trace(
                go.Scatter(
                    x=dates,
                    y=z,
                    name=str(name),
                    mode="lines",
                    line=dict(width=2),
                )
            )
        _lo_m = _plotly_layout_direcional(
            title="Indicadores macro (z-score por série)",
            height=400,
            xaxis_title="Eixo X — data",
            yaxis_title="Eixo Y — desvio em σ (por série)",
            legend=_plotly_legend_bottom(),
            margin=dict(t=100, l=56, r=52, b=112),
        )
        _xr_m = _plotly_xaxis_range_from_dates(dates)
        if _xr_m:
            _lo_m = {**_lo_m, "xaxis": {**_lo_m["xaxis"], "range": _xr_m}}
        fig_m.update_layout(**_lo_m)
        st.plotly_chart(fig_m, width="stretch", config=_pc)
        _st_interpretacao_grafico("Indicadores macro", _interpret_text_macro(macro))

    st.divider()
    oa_key, oa_model = _openai_key_and_model()
    if oa_key:
        with st.expander("Síntese em prosa (OpenAI)", expanded=False):
            st.caption(
                f"Modelo configurado: `{oa_model}`. O texto gerado baseia-se exclusivamente nos resumos automáticos desta aba."
            )
            if st.button("Gerar síntese", key="pv_eda_openai_btn"):
                facts = _facts_eda_compact(daily)
                with st.spinner("A contactar a API…"):
                    syn = _openai_eda_synopsis(facts)
                if syn:
                    st.session_state["pv_eda_openai_syn"] = syn
                else:
                    st.error(
                        "Não foi obtida resposta válida da API; verifique a rede, quotas ou a chave de autenticação."
                    )
            if st.session_state.get("pv_eda_openai_syn"):
                st.markdown(st.session_state["pv_eda_openai_syn"])


def _render_streamlit_tab_apendice(
    por_h: dict[int, dict[str, Any]],
    best_params_preview: dict[Any, dict[str, dict[str, Any]]],
    full_period_train: bool,
    blend_top_k: int,
    random_seed: int,
    daily: dict[str, Any],
) -> None:
    hz_txt = ", ".join(str(x) for x in (3, 7, 30))
    modo = (
        "Neste modo, o modelo final é reajustado sobre **100%** da série histórica; todavia, uma fatia final (~8%) "
        "utiliza-se unicamente para ordenar os candidatos ao *ensemble*, sem constituir um holdout formal das métricas principais."
        if full_period_train
        else "Adota-se uma divisão **70% treino / 30% teste** segundo a ordem cronológica; ademais, nos 70% iniciais, "
        "reserva-se cerca de 8% no extremo temporal para validação interna na escolha do *ensemble*. "
        "Neste cenário, as métricas principais (MAE, RMSE, R², entre outras) reportam-se ao **bloco de teste (30%)**."
    )
    st.markdown(
        '<p class="pv-section-title">Apêndice</p>',
        unsafe_allow_html=True,
    )
    tab_met, tab3, tab7, tab30, tab_cust = st.tabs(
        ["Metodologia", "H = 3 dias", "H = 7 dias", "H = 30 dias", "Personalizado"]
    )
    with tab_met:
        st.markdown(
            f"""
#### Resumo

- **Alvo:** soma de vendas (qtd ou VGV) nos **H** dias imediatamente seguintes a *t*, com *H* ∈ {{{hz_txt}}}; o vetor *X* incorpora apenas informação até *t*. Além disso, na aba **Previsões**, um intervalo entre datas de início e fim define um alvo complementar.
- **Dados:** **{daily.get("n_rows", 0):,}** dias · **{daily.get("n_features", 0)}** colunas após engenharia.
- **Validação:** {modo}
- **Optuna:** algoritmo TPE, **TimeSeriesSplit** e **MedianPruner**; a função objetivo penaliza a variância entre *folds*, promovendo estabilidade. Semente: **{random_seed}**.
- **Modelos:** candidatos incluem Ridge, ElasticNet, SVR, k-NN, florestas, LightGBM/XGBoost/CatBoost, NGBoost (VGV), baselines e *stacks* leves; o ranking baseia-se no MAE na validação interna, sendo que os **{blend_top_k}** melhores podem integrar um *ensemble* ou, caso contrário, prevalece o modelo único vencedor. *Sample weights* aplicam-se quando o algoritmo o permite.
- **Entrada numérica:** **MinMaxScaler (0,1)** com *clip*.
- **Tabelas binárias no HTML:** comparam-se previsões à mediana de *y* no treino do benchmark, funcionando como métrica auxiliar.

O relatório HTML exportável na aba **Previsões** reúne, entre outros elementos, gráficos operacionais, curvas ROC e o *benchmark* de modelos.
"""
        )

    def _ap_horizon_block(h: int) -> None:
        sub = por_h.get(h) or {}
        qd = sub.get("qtd", {})
        vl = sub.get("valor", {})
        st.markdown(
            f'<p class="pv-section-title">Horizonte {h} dias</p>'
            f"<p style='text-align:center;color:#64748b;font-size:0.9rem;margin-top:-0.25rem'>"
            f"{qd.get('model_label', '—')} · {vl.get('model_label', '—')}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("**Volume (quantidade)** — pipeline vencedor")
        st.code(str(qd.get("model_label", "—")), language=None)
        st.markdown("**VGV** — pipeline vencedor")
        st.code(str(vl.get("model_label", "—")), language=None)
        bp = best_params_preview.get(h, {}) if isinstance(best_params_preview, dict) else {}
        st.markdown("**Hiperparâmetros LightGBM (volume)**")
        st.json(bp.get("qtd") or {})
        st.markdown("**Hiperparâmetros LightGBM (VGV)**")
        st.json(bp.get("valor") or {})

    with tab3:
        _ap_horizon_block(3)
    with tab7:
        _ap_horizon_block(7)
    with tab30:
        _ap_horizon_block(30)

    with tab_cust:
        bpc = best_params_preview.get("custom") if isinstance(best_params_preview, dict) else None
        if bpc:
            st.markdown(
                '<p class="pv-section-title">Cenário personalizado (referência Optuna)</p>',
                unsafe_allow_html=True,
            )
            st.json({"volume (qtd)": bpc.get("qtd") or {}, "vgv": bpc.get("valor") or {}})
        else:
            st.markdown(
                '<p style="text-align:center;color:#64748b;font-size:0.88rem">'
                "Sem cenário personalizado nesta execução.</p>",
                unsafe_allow_html=True,
            )


def _render_streamlit_dossie_ml(
    dossie: dict[str, Any],
    por_h: dict[Any, dict[str, Any]],
    daily_pack: dict[str, Any],
    por_custom: dict[str, Any] | None,
) -> None:
    import plotly.graph_objects as go

    _dpc = {"displayModeBar": True, "displaylogo": False, "scrollZoom": True}

    st.markdown(
        "<div style='text-align:justify;text-justify:inter-word;hyphens:auto;-webkit-hyphens:auto;max-width:100%;margin:0 auto 0.85rem auto;color:#475569;font-size:0.95rem;line-height:1.55'>"
        "O Dossiê consolida estatísticas descritivas, análise IQR, matrizes de correlação, VIF, histogramas e, "
        "quando aplicável, métricas de modelo — oferecendo, desta forma, uma visão integrada da qualidade dos dados e do ajuste.</div>",
        unsafe_allow_html=True,
    )

    if dossie.get("erro"):
        st.error(str(dossie["erro"]))
        return

    if dossie.get("aviso_previsao_custom"):
        st.warning(str(dossie["aviso_previsao_custom"]))

    td1, td2, td3, td4, td5 = st.tabs(
        [
            "Dados e tratamento",
            "Descritivas e outliers",
            "Correlação e VIF",
            "Distribuições e perfil",
            "Modelos e personalizado",
        ]
    )

    with td1:
        st.markdown(
            """
**Tratamento**
- O índice diário é deduplicado, conservando a última linha por data.
- Nos alvos, eliminam-se apenas as linhas sem `target_qtd` ou `target_valor`; valores não finitos são tratados antes da agregação.
- Nas *features*, `inf` é convertido para valor finito e a ausência de dados imputa-se a 0 após a engenharia (lags utilizam exclusivamente o passado).
- Nos modelos, aplica-se `MinMaxScaler(0,1)` com `clip`.
- Os outliers no alvo não são truncados; o *cap* aplicado às previsões deriva, portanto, do próprio treino.

**Validação / tuning**
- Utiliza-se *split* cronológico 70/30; em paralelo, a Optuna recorre a **TimeSeriesSplit** e **MedianPruner**, penalizando a variância entre *folds*.
- Os modelos SVR e k-NN adotam regularização e parâmetro *k* mais conservadores, de modo a reduzir sobreajuste.
"""
        )
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Observações na matriz diária", f"{dossie.get('n_linhas', 0):,}")
        with m2:
            st.metric("Colunas após engenharia", f"{dossie.get('n_colunas', 0):,}")
        st.caption(
            f"Período coberto pela matriz: de **{dossie.get('primeira_data', '—')}** a **{dossie.get('ultima_data', '—')}**."
        )

    desc = dossie.get("descritivas") or []
    oi = dossie.get("outliers_iqr") or []
    with td2:
        if desc:
            st.markdown("##### Estatísticas descritivas")
            st.caption(
                "A curtose reportada corresponde ao excesso (definição *pandas*); a moda corresponde ao primeiro valor modal encontrado."
            )
            st.dataframe(pd.DataFrame(desc), width="stretch", hide_index=True)
        else:
            st.caption("Não há tabela descritiva disponível para o conjunto atual.")
        st.divider()
        if oi:
            st.markdown("##### Outliers (regra IQR — 1,5×IQR)")
            st.dataframe(pd.DataFrame(oi), width="stretch", hide_index=True)
        else:
            st.caption("Não foi possível gerar o resumo de outliers para estes dados.")

    cl = dossie.get("correlation_labels") or []
    cz = dossie.get("correlation_matrix") or []
    pares = dossie.get("pares_multicolinearidade") or []
    vifs = dossie.get("vif") or []
    with td3:
        if len(cl) >= 2 and cz:
            st.markdown("##### Matriz de correlação (Pearson)")
            fig_h = go.Figure(
                data=go.Heatmap(
                    z=cz,
                    x=cl,
                    y=cl,
                    colorscale=[
                        [0.0, PLOT_VERMELHO],
                        [0.5, "#f1f5f9"],
                        [1.0, PLOT_AZUL],
                    ],
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
                )
            )
            fig_h.update_layout(
                **_plotly_layout_direcional(
                    title="Correlação entre variáveis-chave",
                    height=max(360, 28 * len(cl)),
                    margin=dict(l=130, r=48, t=56, b=130),
                ),
            )
            st.plotly_chart(fig_h, width="stretch", config=_dpc)
            _st_interpretacao_grafico(
                "Correlações (leitura automática)",
                _interpret_text_correlation_matrix(cl, cz),
            )
        else:
            st.markdown(
                '<p style="text-align:center;color:#94a3b8;font-size:0.88rem">A matriz de correlação não está disponível neste subconjunto (colunas ou dados insuficientes).</p>',
                unsafe_allow_html=True,
            )

        _vl3 = daily_pack.get("vol_leads") or []
        _tq3 = daily_pack.get("target_qtd") or []
        _tv3 = daily_pack.get("target_valor") or []
        if len(_vl3) == len(_tq3) == len(_tv3) and len(_vl3) > 5:
            st.markdown("##### Dispersão: leads × vendas (mesmo dia)")
            st.caption(
                "Além dos valores agregados na matriz de correlação, a dispersão posiciona cada dia no plano "
                "leads × volume, permitindo visualizar dispersão, aglomerados e observações atípicas que o coeficiente r resume de forma sintética."
            )
            from plotly.subplots import make_subplots

            _tv3_mi = [float(v) / 1e6 for v in _tv3]
            fig_td3 = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Quantidade diária", "VGV (mi R$ / dia)"),
                horizontal_spacing=0.12,
            )
            fig_td3.add_trace(
                go.Scatter(
                    x=_vl3,
                    y=_tq3,
                    mode="markers",
                    marker=dict(size=7, opacity=0.5, color=PLOT_AZUL, line=dict(width=0.5, color="#ffffff")),
                    name="qtd",
                    hovertemplate="Leads=%{x:.4g}<br>qtd=%{y:,.0f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig_td3.add_trace(
                go.Scatter(
                    x=_vl3,
                    y=_tv3_mi,
                    mode="markers",
                    marker=dict(size=7, opacity=0.5, color=PLOT_VERMELHO, line=dict(width=0.5, color="#ffffff")),
                    name="VGV",
                    hovertemplate="Leads=%{x:.4g}<br>VGV mi=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            base_td3 = _plotly_layout_direcional(height=420, showlegend=False)
            fig_td3.update_layout(
                **{
                    k: v
                    for k, v in base_td3.items()
                    if k not in ("xaxis", "yaxis", "margin", "title")
                },
                title=dict(
                    text="Dispersão (eixo horizontal: leads; mesmos dias da série diária)",
                    font=dict(size=15, color=PLOT_AZUL, family="Montserrat, sans-serif"),
                    x=0.5,
                    xanchor="center",
                ),
                margin=dict(t=56, l=56, r=48, b=56),
            )
            fig_td3.update_xaxes(title_text="Eixo X — leads (/ dia)", row=1, col=1)
            fig_td3.update_xaxes(title_text="Eixo X — leads (/ dia)", row=1, col=2)
            fig_td3.update_yaxes(title_text="Eixo Y₁ — qtd vendida (/ dia)", row=1, col=1)
            fig_td3.update_yaxes(title_text="Eixo Y₂ — VGV (mi R$ / dia)", row=1, col=2)
            st.plotly_chart(fig_td3, width="stretch", config=_dpc)

        st.divider()
        if pares:
            st.markdown("##### Pares com |r| ≥ 0,85")
            st.dataframe(pd.DataFrame(pares), width="stretch", hide_index=True)
        if vifs:
            st.markdown("##### VIF (subconjunto de *features*)")
            st.caption(
                "Valores de VIF superiores a 5–10 sugerem possível redundância linear entre *features*; interprete-as em conjunto com a matriz de correlação."
            )
            st.dataframe(pd.DataFrame(vifs), width="stretch", hide_index=True)
            _st_interpretacao_grafico("VIF (leitura automática)", _interpret_text_vif_rows(vifs))
        if not pares and not vifs:
            st.caption("Não há tabelas de multicolinearidade para apresentar nesta execução.")
        facts_cv = _facts_corr_vif_for_llm(cl, cz, vifs)
        oa_key_d, oa_model_d = _openai_key_and_model()
        if oa_key_d and facts_cv.strip():
            st.divider()
            with st.expander("Síntese: correlação e VIF (OpenAI)", expanded=False):
                st.caption(f"Modelo: `{oa_model_d}`.")
                if st.button("Gerar síntese", key="pv_dossie_openai_btn"):
                    with st.spinner("A contactar a API…"):
                        syn_d = _openai_eda_synopsis(facts_cv)
                    if syn_d:
                        st.session_state["pv_dossie_openai_syn"] = syn_d
                    else:
                        st.error("Sem resposta ou erro na API.")
                if st.session_state.get("pv_dossie_openai_syn"):
                    st.markdown(st.session_state["pv_dossie_openai_syn"])

    bal = dossie.get("balanceamento_qtd") or {}
    hq = dossie.get("hist_target_qtd") or {}
    hv = dossie.get("hist_target_valor") or {}
    dates = daily_pack.get("dates") or []
    vl = daily_pack.get("vol_leads") or []
    tq = daily_pack.get("target_qtd") or []
    dow_lbl = daily_pack.get("dow_labels") or []
    dow_q = daily_pack.get("dow_mean_qtd") or []
    with td4:
        if bal:
            st.markdown("##### Balanceamento do volume diário (vs mediana)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Dias acima da mediana", f"{bal.get('dias_acima_mediana', 0):,}")
            with c2:
                st.metric("Dias ≤ mediana", f"{bal.get('dias_abaixo_igual_mediana', 0):,}")
            with c3:
                st.metric(
                    "Proporção classe minoritária",
                    f"{bal.get('proporcao_classe_minoritaria', 0)*100:.1f}%",
                )
            fig_p = go.Figure(
                data=[
                    go.Pie(
                        labels=["Acima da mediana", "Abaixo ou igual"],
                        values=[
                            bal.get("dias_acima_mediana", 0),
                            bal.get("dias_abaixo_igual_mediana", 0),
                        ],
                        hole=0.38,
                        marker=dict(colors=[PLOT_AZUL, PLOT_MUTED], line=dict(color="#fff", width=2)),
                        textinfo="percent+label",
                    )
                ]
            )
            fig_p.update_layout(
                **_plotly_layout_direcional(
                    title="Partilha de dias por regime de volume", height=420, showlegend=False
                ),
            )
            st.plotly_chart(fig_p, width="stretch", config=_dpc)
        if hq.get("counts") and hq.get("edges"):
            st.markdown("##### Histograma — vendas diárias (qtd)")
            edges = hq["edges"]
            cnt = hq["counts"]
            xs = [(edges[i] + edges[i + 1]) / 2 for i in range(len(cnt))]
            fig = go.Figure(
                go.Bar(
                    x=xs,
                    y=cnt,
                    marker_color=PLOT_AZUL,
                    marker_line=dict(width=0),
                )
            )
            fig.update_layout(
                **_plotly_layout_direcional(
                    height=400,
                    xaxis_title="Volume (qtd)",
                    yaxis_title="Frequência",
                ),
            )
            st.plotly_chart(fig, width="stretch", config=_dpc)
        if hv.get("counts") and hv.get("edges"):
            st.markdown("##### Histograma — VGV diário")
            edges = hv["edges"]
            cnt = hv["counts"]
            xs = [((edges[i] + edges[i + 1]) / 2) / 1e6 for i in range(len(cnt))]
            figv = go.Figure(
                go.Bar(
                    x=xs,
                    y=cnt,
                    marker_color=PLOT_VERMELHO,
                    marker_line=dict(width=0),
                )
            )
            figv.update_layout(
                **_plotly_layout_direcional(
                    height=400,
                    xaxis_title="VGV (mi R$)",
                    yaxis_title="Frequência",
                ),
            )
            st.plotly_chart(figv, width="stretch", config=_dpc)
        if len(dates) == len(vl) == len(tq) and len(dates) > 5:
            st.markdown("##### Dispersão: leads × vendas (mesmo dia)")
            fig_s = go.Figure(
                go.Scatter(
                    x=vl,
                    y=tq,
                    mode="markers",
                    marker=dict(
                        size=7,
                        opacity=0.45,
                        color=PLOT_AZUL,
                        line=dict(width=0.5, color="#ffffff"),
                    ),
                )
            )
            fig_s.update_layout(
                **_plotly_layout_direcional(
                    height=420,
                    xaxis_title="Eixo X — leads (/ dia)",
                    yaxis_title="Eixo Y — quantidade vendida (/ dia)",
                ),
            )
            st.plotly_chart(fig_s, width="stretch", config=_dpc)
        if dow_lbl and dow_q:
            st.markdown("##### Média de volume por dia da semana")
            fig_b = go.Figure(
                go.Bar(
                    x=dow_lbl,
                    y=dow_q,
                    marker_color=PLOT_AZUL,
                    marker_line=dict(width=0),
                )
            )
            fig_b.update_layout(
                **_plotly_layout_direcional(
                    height=400,
                    yaxis_title="Qtd média",
                ),
            )
            st.plotly_chart(fig_b, width="stretch", config=_dpc)

    with td5:
        _render_streamlit_ml_feature_importance(por_h, _dpc)
        st.markdown("##### Métricas dos modelos (holdout) e acurácia direcional")
        st.markdown(
            "**Acurácia dir.:** no conjunto de teste, corresponde à percentagem de dias em que o valor real e a previsão "
            "se situam do mesmo lado da mediana de *Y* estimada no treino; assim, complementa MAE, RMSE e R² sem os substituir."
        )
        rows_m = []
        for h in (3, 7, 30):
            sub = por_h.get(h) or {}
            for alvo, nome in (("qtd", "Quantidade"), ("valor", "VGV")):
                m = (sub.get(alvo) or {}).get("metrics_test") or {}
                rows_m.append(
                    {
                        "Horizonte": f"{h}d",
                        "Alvo": nome,
                        "MAE": m.get("MAE"),
                        "RMSE": m.get("RMSE"),
                        "R²": m.get("R2"),
                        "Acurácia dir.": m.get("Acc_dir_mediana"),
                    }
                )
        if por_custom:
            for alvo, nome in (("qtd", "Quantidade"), ("valor", "VGV")):
                m = (por_custom.get(alvo) or {}).get("metrics_test") or {}
                rows_m.append(
                    {
                        "Horizonte": "Personalizado",
                        "Alvo": nome,
                        "MAE": m.get("MAE"),
                        "RMSE": m.get("RMSE"),
                        "R²": m.get("R2"),
                        "Acurácia dir.": m.get("Acc_dir_mediana"),
                    }
                )
        _hay_metricas = any(r.get("MAE") is not None for r in rows_m)
        if _hay_metricas:
            st.dataframe(pd.DataFrame(rows_m), width="stretch", hide_index=True)
        else:
            st.markdown(
                '<p style="text-align:center;color:#64748b;font-size:0.88rem;margin:0.5rem 0">'
                "As métricas detalhadas surgirão nesta secção após executar <strong>Gerar previsões</strong>.</p>",
                unsafe_allow_html=True,
            )
        low_acc: list[dict[str, Any]] = []
        for r in rows_m:
            ad = r.get("Acurácia dir.")
            if ad is None:
                continue
            try:
                adf = float(ad)
            except (TypeError, ValueError):
                continue
            if np.isfinite(adf) and adf < 0.8:
                low_acc.append(r)
        if _hay_metricas and low_acc:
            st.warning(
                "A acurácia direcional é inferior a 80% em, pelo menos, um horizonte ou alvo; analise o conjunto de métricas "
                "antes de utilizar as previsões como único critério de decisão."
            )
        st.divider()
        if por_custom:
            st.markdown("##### Intervalo personalizado")
            st.markdown(f"**{por_custom.get('label', '—')}**")
            qc = por_custom.get("qtd") or {}
            vc = por_custom.get("valor") or {}
            st.markdown(
                f"- **Qtd prevista (último dia):** {float(qc.get('pred_ultimo_dia', 0)):,.2f} · "
                f"MAE teste: {float((qc.get('metrics_test') or {}).get('MAE', 0)):.3f} · "
                f"Acurácia dir.: {float((qc.get('metrics_test') or {}).get('Acc_dir_mediana', 0))*100:.1f}%\n"
                f"- **VGV previsto:** R$ {float(vc.get('pred_ultimo_dia', 0)):,.0f} · "
                f"MAE teste: {float((vc.get('metrics_test') or {}).get('MAE', 0))/1e6:.3f} mi · "
                f"Acurácia dir.: {float((vc.get('metrics_test') or {}).get('Acc_dir_mediana', 0))*100:.1f}%"
            )
            st.caption(
                "As variáveis `fwd_*` incorporam fins de semana e feriados brasileiros dentro do horizonte da soma, quando aplicável."
            )
        else:
            st.caption("Não foi definido cenário personalizado por intervalo de datas nesta execução.")


def main() -> None:
    fav = _resolver_png_raiz(FAVICON_ARQUIVO)
    st.set_page_config(
        page_title="Previsão de vendas | Direcional",
        page_icon=str(fav) if fav else "📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _configure_streamlit_progress()
    inject_css()
    _exibir_logo_topo()

    st.markdown(
        '<div class="pv-hero-block">'
        '<p class="pv-hero-title">Direcional Engenharia · Previsão de vendas</p>'
        '<p class="pv-hero-sub">Consolidação da matriz diária, modelos nos horizontes 3, 7 e 30 dias e exportação de relatório HTML.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pv-bar-wrap"><div class="pv-bar"></div></div>', unsafe_allow_html=True)

    if "resultado" not in st.session_state:
        st.session_state.resultado = None
    if "dados_exploratorios" not in st.session_state:
        st.session_state.dados_exploratorios = None
    if "formulario_snapshot" not in st.session_state:
        st.session_state.formulario_snapshot = None

    tab_intro, tab_analises, tab_form, tab_previsoes, tab_dossie, tab_apendice = st.tabs(
        ["Introdução", "Análises", "Formulário", "Previsões", "Dossiê", "Apêndice"]
    )

    with tab_intro:
        _render_tab_introducao()

    with tab_form:
        _render_tab_formulario_previsao_humano()

    with tab_previsoes:
        st.markdown(
            '<p class="pv-section-title">Previsões e relatório</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:justify;text-justify:inter-word;hyphens:auto;-webkit-hyphens:auto;color:#64748b;font-size:0.92rem;width:100%;margin:0 auto 1rem auto'>"
            "Por padrão, estimam-se sempre os horizontes de <strong>3, 7 e 30</strong> dias. Adicionalmente, caso indique "
            "um intervalo opcional nas datas abaixo, treina-se um par de modelos cuja soma-alvo corresponde ao calendário entre essas datas.</div>",
            unsafe_allow_html=True,
        )
        ic1, ic2 = st.columns(2)
        with ic1:
            d_ini = st.date_input(
                "Dia inicial do intervalo (opcional)",
                value=None,
                key="pv_interval_start",
            )
        with ic2:
            d_fim = st.date_input(
                "Dia final do intervalo (opcional)",
                value=None,
                key="pv_interval_end",
            )
        if d_ini is not None and d_fim is not None:
            a, b = (d_ini, d_fim) if d_ini <= d_fim else (d_fim, d_ini)
            st.info(
                f"Foi definido um intervalo adicional de **{a.isoformat()}** a **{b.isoformat()}**, além dos horizontes fixos."
            )
        elif d_ini is None and d_fim is None:
            pass
        else:
            st.warning(
                "Indique **ambas** as datas (início e fim) ou, então, deixe **as duas** em branco, de forma consistente."
            )

        run_btn = st.button("Gerar previsões", type="primary", width="stretch", key="pv_gerar")

        if run_btn:
            if (d_ini is None) ^ (d_fim is None):
                st.error("Intervalo incompleto: são necessárias duas datas válidas ou nenhuma (deixe ambos os campos vazios).")
            else:
                with st.spinner("A carregar e analisar as planilhas…"):
                    dfs, sheet_metas, used_sa = build_data_bundle()
                st.session_state["sheet_metas"] = sheet_metas
                st.session_state["used_service_account"] = used_sa
                try:
                    custom_previsao = None
                    if d_ini is not None and d_fim is not None:
                        a, b = (d_ini, d_fim) if d_ini <= d_fim else (d_fim, d_ini)
                        custom_previsao = {
                            "mode": "range",
                            "date_start": a.isoformat(),
                            "date_end": b.isoformat(),
                        }
                    with st.spinner("A treinar modelos e gerar o relatório…"):
                        (
                            stats_base,
                            ticket,
                            conv,
                            por_h,
                            bpp,
                            html_out,
                            daily_pack,
                            dossie_ml,
                            por_custom,
                        ) = run_training_pipeline(dfs, custom_previsao=custom_previsao)
                    st.session_state.resultado = {
                        "stats_base": stats_base,
                        "ticket": ticket,
                        "conv": conv,
                        "por_h": por_h,
                        "best_params_preview": bpp,
                        "html": html_out,
                        "daily_pack": daily_pack,
                        "dossie_ml": dossie_ml,
                        "por_custom": por_custom,
                        "full_train": FULL_PERIOD_TRAIN_FIXO,
                        "sheet_metas": sheet_metas,
                        "used_sa": used_sa,
                        "df_formulario": dfs.get("formulario_previsao"),
                    }
                    st.success("Execução concluída com sucesso.")
                    st.session_state.dados_exploratorios = None
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a execução: {e}")
                    st.exception(e)

        res = st.session_state.resultado
        if res is None:
            pass
        else:
            stats_base = res["stats_base"]
            ticket = res["ticket"]
            conv = res["conv"]
            por_h = res["por_h"]
            html_out = res["html"]
            full_train = res["full_train"]
            sheet_metas = res.get("sheet_metas") or st.session_state.get("sheet_metas")

            if sheet_metas:
                st.markdown(
                    '<p class="pv-section-title">Fontes carregadas</p>',
                    unsafe_allow_html=True,
                )
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
                st.dataframe(pd.DataFrame(rows_m), width="stretch", hide_index=True)

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

            st.markdown(
                '<p class="pv-section-title">Previsões (último dia da série)</p>',
                unsafe_allow_html=True,
            )
            rows = []
            lbl = "In-sample" if full_train else "Holdout 30%"
            for h in (3, 7, 30):
                q = por_h[h]["qtd"]
                v = por_h[h]["valor"]
                acc_q = q.get("metrics_test", {}).get("Acc_dir_mediana")
                acc_v = v.get("metrics_test", {}).get("Acc_dir_mediana")
                rows.append(
                    {
                        "Horizonte (dias)": h,
                        "Qtd prevista": f"{float(q['pred_ultimo_dia']):,.1f}",
                        "VGV previsto (R$)": f"{float(v['pred_ultimo_dia']):,.0f}",
                        f"MAE Qtd ({lbl})": f"{q['metrics_test']['MAE']:.2f}",
                        f"MAE VGV ({lbl})": f"{v['metrics_test']['MAE']/1e6:.2f} mi",
                        "Acurácia dir. Qtd": f"{acc_q*100:.1f}%"
                        if acc_q is not None and np.isfinite(acc_q)
                        else "—",
                        "Acurácia dir. VGV": f"{acc_v*100:.1f}%"
                        if acc_v is not None and np.isfinite(acc_v)
                        else "—",
                        "Modelo Qtd": q["model_label"],
                        "Modelo VGV": v["model_label"],
                    }
                )
            pc = res.get("por_custom")
            if pc:
                qc = pc.get("qtd") or {}
                vc = pc.get("valor") or {}
                aq = (qc.get("metrics_test") or {}).get("Acc_dir_mediana")
                av = (vc.get("metrics_test") or {}).get("Acc_dir_mediana")
                rows.append(
                    {
                        "Horizonte (dias)": f"Personalizado: {(pc.get('label') or '')[:52]}",
                        "Qtd prevista": f"{float(qc.get('pred_ultimo_dia', 0)):,.1f}",
                        "VGV previsto (R$)": f"{float(vc.get('pred_ultimo_dia', 0)):,.0f}",
                        f"MAE Qtd ({lbl})": f"{float((qc.get('metrics_test') or {}).get('MAE', 0)):.2f}",
                        f"MAE VGV ({lbl})": f"{float((vc.get('metrics_test') or {}).get('MAE', 0))/1e6:.2f} mi",
                        "Acurácia dir. Qtd": f"{aq*100:.1f}%"
                        if aq is not None and np.isfinite(aq)
                        else "—",
                        "Acurácia dir. VGV": f"{av*100:.1f}%"
                        if av is not None and np.isfinite(av)
                        else "—",
                        "Modelo Qtd": qc.get("model_label", "—"),
                        "Modelo VGV": vc.get("model_label", "—"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            acc_warn = False
            for h in (3, 7, 30):
                for part in ("qtd", "valor"):
                    m = (por_h.get(h) or {}).get(part, {}).get("metrics_test") or {}
                    a = m.get("Acc_dir_mediana")
                    if a is not None and np.isfinite(float(a)) and float(a) < 0.8:
                        acc_warn = True
            if pc:
                for part in ("qtd", "valor"):
                    m = (pc.get(part) or {}).get("metrics_test") or {}
                    a = m.get("Acc_dir_mediana")
                    if a is not None and np.isfinite(float(a)) and float(a) < 0.8:
                        acc_warn = True
            if acc_warn:
                st.warning(
                    "A acurácia direcional situa-se abaixo de 80% em, pelo menos, um horizonte ou alvo; recomenda-se "
                    "cruzar este sinal com MAE, RMSE e estabilidade temporal antes de decisões operacionais."
                )

            st.download_button(
                label="Descarregar relatório HTML",
                data=html_out.encode("utf-8"),
                file_name="dashboard_previsao_vendas.html",
                mime="text/html; charset=utf-8",
                width="stretch",
            )

            st.caption(
                "O detalhe do esquema de métricas encontra-se no **Apêndice**; o ficheiro HTML inclui, adicionalmente, "
                "o *benchmark* de modelos e as curvas ROC."
            )

        st.divider()
        if st.button(
            "Limpar cenário opcional",
            width="stretch",
            key="pv_clear_cfg",
        ):
            for _k in ("pv_interval_start", "pv_interval_end"):
                st.session_state.pop(_k, None)
            st.session_state.pop("pv_custom_cfg", None)
            st.rerun()
        if st.button(
            "Limpar cache das planilhas",
            width="stretch",
            key="pv_clear_cache",
        ):
            _load_one_role_cached.clear()
            st.success("O cache das planilhas foi limpo; na próxima leitura, os ficheiros serão obtidos novamente.")

    with tab_analises:
        st.markdown(
            '<p class="pv-section-title">Análises</p>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Carregar dados (sem treino)",
            type="primary",
            width="stretch",
            key="pv_load_eda",
        ):
            _streamlit_carregar_dados_exploratorios()

        res_a = st.session_state.resultado
        dex_a = st.session_state.get("dados_exploratorios")
        fonte_a: dict[str, Any] | None = None
        if res_a and res_a.get("daily_pack"):
            fonte_a = res_a
        elif isinstance(dex_a, dict) and dex_a.get("daily_pack"):
            fonte_a = dex_a
        if fonte_a:
            _render_streamlit_tab_analises(
                fonte_a["daily_pack"],
                fonte_a["stats_base"],
                fonte_a["ticket"],
                fonte_a["conv"],
            )

    with tab_dossie:
        st.markdown('<p class="pv-section-title">Dossiê</p>', unsafe_allow_html=True)
        if st.button(
            "Carregar dados (sem treino)",
            type="primary",
            width="stretch",
            key="pv_load_eda_dossie",
        ):
            _streamlit_carregar_dados_exploratorios()

        res_d = st.session_state.resultado
        dex_d = st.session_state.get("dados_exploratorios")
        if res_d and res_d.get("dossie_ml") and res_d.get("daily_pack"):
            _render_streamlit_dossie_ml(
                res_d["dossie_ml"],
                res_d["por_h"],
                res_d["daily_pack"],
                res_d.get("por_custom"),
            )
        elif isinstance(dex_d, dict) and dex_d.get("dossie_ml") and dex_d.get("daily_pack"):
            _render_streamlit_dossie_ml(
                dex_d["dossie_ml"],
                {},
                dex_d["daily_pack"],
                None,
            )

    with tab_apendice:
        st.markdown('<p class="pv-section-title">Apêndice</p>', unsafe_allow_html=True)
        if st.button(
            "Carregar dados (sem treino)",
            type="primary",
            width="stretch",
            key="pv_load_eda_apendice",
        ):
            _streamlit_carregar_dados_exploratorios()

        res_ap = st.session_state.resultado
        dex_ap = st.session_state.get("dados_exploratorios")
        if res_ap and res_ap.get("daily_pack"):
            bpp = res_ap.get("best_params_preview") or {}
            _render_streamlit_tab_apendice(
                res_ap["por_h"],
                bpp,
                res_ap["full_train"],
                BLEND_TOP_K_FIXO,
                RANDOM_SEED,
                res_ap["daily_pack"],
            )
        elif isinstance(dex_ap, dict) and dex_ap.get("daily_pack"):
            _render_streamlit_tab_apendice(
                {},
                {},
                FULL_PERIOD_TRAIN_FIXO,
                BLEND_TOP_K_FIXO,
                RANDOM_SEED,
                dex_ap["daily_pack"],
            )

    st.markdown(
        '<div class="pv-foot-wrap">'
        '<p class="pv-foot">Direcional Engenharia · Previsão de vendas</p>'
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
