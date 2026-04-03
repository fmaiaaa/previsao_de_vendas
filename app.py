"""
Previsão de vendas — Streamlit (Direcional).
Carrega planilhas Google (conta de serviço ou CSV público), motor em src/.
Um único botão de execução; hiperparâmetros fixos para máxima robustez sem decisão manual.
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import (  # noqa: E402
    build_daily_master,
    build_xy_for_horizon,
    predict_last_row,
)
from src.report_html import render_dashboard  # noqa: E402
from src.train_eval import train_one_target  # noqa: E402

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
    from src.gsheets_loader import gspread_client_from_streamlit, load_role_dataframe

    gc = gspread_client_from_streamlit() if use_sa else None
    return load_role_dataframe(
        gc,
        spreadsheet_id,
        role_key,
        csv_gid_hints=list(hints_tuple) if hints_tuple else None,
    )


def build_data_bundle() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], bool]:
    from src.gsheets_loader import ROLE_LABELS, gspread_client_from_streamlit

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
        from src.gsheets_loader import gspread_client_from_streamlit

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
