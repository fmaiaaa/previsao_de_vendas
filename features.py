"""Série diária agregada e engenharia de atributos (sem vazamento temporal)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .load import find_column_any

# Após o sábado de referência, os totais do formulário só entram como feature (evita vazamento).
FORMULARIO_FEATURE_LAG_DIAS = 7


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
    """
    fwd_wknd = np.zeros(len(index), dtype=float)
    fwd_bday = np.zeros(len(index), dtype=float)
    for i, ts in enumerate(index):
        dr = pd.date_range(ts + pd.Timedelta(days=1), periods=horizon, freq="D")
        dow = dr.dayofweek.to_numpy()
        fwd_wknd[i] = float((dow >= 5).sum())
        fwd_bday[i] = float((dow < 5).sum())
    h = float(horizon)
    return pd.DataFrame(
        {
            "fwd_h": h,
            "fwd_wknd_h": fwd_wknd,
            "fwd_bday_h": fwd_bday,
            "fwd_wknd_ratio": fwd_wknd / (h + 1e-9),
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
