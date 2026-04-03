"""Leitura de Excel, normalização de cabeçalhos e EDA."""

from __future__ import annotations

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
