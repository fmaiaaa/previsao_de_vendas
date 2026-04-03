"""
Pipeline local: upload (Tkinter) ou pasta com .xlsx, treino 80/20, Optuna + ensemble,
exportação do dashboard HTML. Previsão operacional reajusta em 100% do histórico.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garantir imports do pacote src
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk
from tkinter import filedialog

from tqdm.auto import tqdm

from src.features import (
    build_daily_master,
    build_xy_for_horizon,
    predict_last_row,
)
from src.load import classify_upload, load_four_files
from src.report_html import render_dashboard
from src.train_eval import train_one_target


def pick_files_dialog() -> dict[str, Path]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Selecione 4 ou 5 .xlsx: Leads, Agendamentos, Pastas, Vendas [+ Formulário previsão]",
        filetypes=[("Excel", "*.xlsx"), ("Todos", "*.*")],
    )
    root.destroy()
    if not paths or len(paths) not in (4, 5):
        raise SystemExit(
            "Selecione 4 bases (Leads, Agendamentos, Pastas, Vendas) ou 5 com o Excel do formulário de previsão."
        )
    mapping: dict[str, Path] = {}
    for p in paths:
        role = classify_upload(Path(p).name)
        if role in mapping:
            raise SystemExit(
                f"Dois ficheiros classificados como '{role}'. Renomeie ou selecione ficheiros distintos.\n"
                f"  {mapping[role].name}\n  {Path(p).name}"
            )
        mapping[role] = Path(p)
    required4 = {"leads", "agendamentos", "pastas", "vendas"}
    required5 = required4 | {"formulario_previsao"}
    got = set(mapping)
    if got != required4 and got != required5:
        raise SystemExit(
            f"Combinação de ficheiros inválida. Encontrados: {sorted(mapping)}. "
            f"Esperados: {sorted(required4)} ou {sorted(required5)}."
        )
    return mapping


def load_from_dir(data_dir: Path) -> dict[str, Path]:
    patterns = {
        "leads": "*Leads*.xlsx",
        "agendamentos": "*Agendamento*.xlsx",
        "pastas": "*Pastas*.xlsx",
        "vendas": "*Vendas*.xlsx",
    }
    out: dict[str, Path] = {}
    for role, pat in patterns.items():
        hits = [
            p
            for p in data_dir.glob(pat)
            if not p.name.startswith("~$")
        ]
        if not hits:
            raise FileNotFoundError(f"Em {data_dir}: nenhum ficheiro {pat} (excl. ~$)")
        hits.sort(key=lambda p: p.name.lower())
        out[role] = hits[0]
    form_hits = [
        p
        for p in data_dir.glob("*.xlsx")
        if not p.name.startswith("~$") and classify_upload(p.name) == "formulario_previsao"
    ]
    form_hits.sort(key=lambda p: p.name.lower())
    if form_hits:
        out["formulario_previsao"] = form_hits[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Previsão de vendas — ML local")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Pasta com os 4 .xlsx (sem popup). Ex.: previsão_de_vendas",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Caminho do HTML gerado (default: dashboard_previsao_vendas.html na pasta do script)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=0,
        help="Trials Optuna (LGBM). 0 = automático conforme tamanho da série (~55–140).",
    )
    parser.add_argument(
        "--blend-k",
        type=int,
        default=5,
        help="Até K modelos no super-blend ponderado (validação escolhe blend vs único)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Desativa barras de progresso (tqdm)",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Treina só em 100% do período (sem holdout 20%); métricas passam a ser in-sample.",
    )
    parser.add_argument(
        "--formulario",
        type=Path,
        default=None,
        help="Excel do formulário de previsão (ESBOÇO / respostas). Sobrepõe deteção automática na pasta.",
    )
    args = parser.parse_args()

    out_html = args.out or (_ROOT / "dashboard_previsao_vendas.html")

    if args.data_dir:
        files_map = load_from_dir(args.data_dir.resolve())
    else:
        files_map = pick_files_dialog()
    if args.formulario is not None:
        fp = args.formulario.resolve()
        if not fp.is_file():
            raise SystemExit(f"--formulario não é um ficheiro: {fp}")
        files_map["formulario_previsao"] = fp

    dfs = load_four_files({k: str(v) for k, v in files_map.items()})
    if not args.quiet:
        print("A construir série diária e features (STL pode demorar um pouco)...")
    df_master = build_daily_master(dfs, show_progress=not args.quiet)

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
    por_horizonte: dict = {}
    best_params_preview: dict = {}

    show_pb = not args.quiet
    full_train = args.full_train
    train_bar = tqdm(
        total=len(horizontes) * 2,
        desc="Treino (Optuna + ensemble)",
        unit="modelo",
        disable=not show_pb,
    )
    for h in horizontes:
        Xq, yq = build_xy_for_horizon(df_master, "target_qtd", h)
        Xv, yv = build_xy_for_horizon(df_master, "target_valor", h)

        train_bar.set_postfix_str(f"H={h}d qtd")
        rq = train_one_target(
            Xq,
            yq,
            horizon=h,
            target_name="qtd",
            n_trials=args.trials,
            random_state=args.seed,
            optuna_seed=args.seed,
            blend_top_k=args.blend_k,
            show_progress=show_pb,
            full_period_train=full_train,
        )
        train_bar.update(1)

        train_bar.set_postfix_str(f"H={h}d VGV")
        rv = train_one_target(
            Xv,
            yv,
            horizon=h,
            target_name="valor",
            n_trials=args.trials,
            random_state=args.seed,
            optuna_seed=args.seed + 1,
            blend_top_k=args.blend_k,
            show_progress=show_pb,
            full_period_train=full_train,
        )
        train_bar.update(1)

        pred_q = predict_last_row(df_master, rq.pipeline, h)
        pred_v = predict_last_row(df_master, rv.pipeline, h)

        def pack(r, pred_last):
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

    train_bar.close()

    if show_pb:
        print("A gerar dashboard HTML...")
    render_dashboard(
        stats_base,
        ticket,
        conv,
        horizontes,
        por_horizonte,
        best_params_preview,
        str(out_html),
        full_period_train=full_train,
    )
    print(f"Dashboard gravado em: {out_html.resolve()}")
    try:
        import webbrowser

        webbrowser.open(out_html.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()
