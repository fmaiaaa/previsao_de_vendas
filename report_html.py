"""Geração de HTML interativo (Plotly + Tailwind via CDN)."""

from __future__ import annotations

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
