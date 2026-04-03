"""
Carregamento robusto de Google Sheets: conta de serviço (gspread) com varredura de
todas as abas, ou fallback por export CSV com vários gid.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

import pandas as pd

from .features import (
    _resolve_agendamentos,
    _resolve_leads,
    _resolve_pastas,
    _resolve_vendas,
)
from .load import find_column_any, normalize_dataframe_columns

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
