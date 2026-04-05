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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
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
    neg_f = [p for p in pairs if p[2] <= -0.25]
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
            f"<td class='p-2 border
