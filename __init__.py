"""Pipeline previsão de vendas — carregamento, features, treino e relatório HTML."""

from . import features, load, report_html, train_eval

__all__ = ["features", "load", "report_html", "train_eval"]