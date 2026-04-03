# Previsão de vendas (4 bases + formulário opcional → ensemble + relatório HTML)

## Como rodar

```bash
cd previsão_de_vendas
pip install -r requirements-previsao.txt
python run_previsao.py
```

### Aplicação web (Streamlit)

Interface única com carregamento direto das Google Sheets (export CSV público), CSS corporativo e descarga do relatório HTML:

```bash
cd previsão_de_vendas
pip install -r requirements.txt
streamlit run app.py
```

**Streamlit Community Cloud:** raiz do repositório = pasta `previsão_de_vendas` (ou defina *Main file* como `previsão_de_vendas/app.py` e *Requirements* como `previsão_de_vendas/requirements.txt`).

**Google Sheets**

- **Recomendado:** nas *Secrets*, coloque o **JSON completo** da conta de serviço em `GOOGLE_SERVICE_ACCOUNT_JSON` (ou a secção `[google_service_account]` com as mesmas chaves do ficheiro da Google Cloud). Partilhe **cada** planilha com o e-mail `client_email` da conta (Leitor).
- A aplicação **percorre todas as abas** de cada livro e escolhe a folha que melhor corresponde a Leads, Agendamentos, Pastas, Vendas ou Formulário (validação de colunas).
- **Sem credenciais:** modo exportação CSV **pública** + varredura de `gid` (sugestões em `CSV_GID_HINTS` / secrets `csv_gid_hints`).

Ver `secrets.toml.example` para exemplos TOML.

Abre-se uma janela para escolher **4 ou 5** ficheiros `.xlsx`: os quatro habituais (Leads, Agendamentos/Visitas, Pastas, Vendas) e, opcionalmente, o Excel do **formulário de previsão** (ex.: *ESBOÇO … respostas*, folha tipo “Respostas ao formulário”). Ao terminar, gera `dashboard_previsao_vendas.html` na mesma pasta e tenta abrir no navegador.

**Sem popup** (ficheiros já na pasta):

```bash
python run_previsao.py --data-dir "."
```

Procura automaticamente ficheiros cujo nome contenha `Leads`, `Agendamento`, `Pastas` e `Vendas`. Se existir um `.xlsx` classificado como formulário de previsão (*Esboço* / *respostas* + *formulário*), também é carregado.

- `--formulario caminho.xlsx` — força o Excel do formulário (sobrepõe a deteção na pasta).

Opções úteis:

- `--out caminho/relatorio.html` — destino do HTML
- `--trials 90` — trials Optuna (LGBM; padrão 75) · `--blend-k 6` — tamanho do super-blend
- `--quiet` — desativa barras **tqdm** (STL, Optuna, candidatos do ensemble, progresso global)

## Nomes esperados dos ficheiros (detecção automática)

- `Leads` no nome
- `Agendamento` ou `Visita` no nome (base de agendamentos e visitas)
- `Pastas` no nome
- `Vendas` no nome (base de vendas fechadas; não confundir com o ficheiro de *respostas* ao formulário)
- **Formulário (opcional):** nome com *Esboço* / *ESBOÇO* e previsão, ou *respostas* + *formulário*

### Integração do formulário (ESBOÇO 2)

Por cada **data de referência (sábado)**, somam-se em todas as linhas:

- `Vendas Facilitadas/Normais Previstas` e `… Reais` (valores monetários → features `fb_vgv_*`)
- `QTD Vendas Facilitadas/Normais Previstas` e `… Reais` (quantidades → features `fb_qtd_*`)

Cada dia da série diária usa o sábado da semana (**semana que termina ao sábado**, `W-SAT`). As features entram com **defasagem de 7 dias** após esse sábado, para reduzir risco de vazamento temporal relativamente ao preenchimento do formulário.

## Partição temporal

- **75%** treino (mais antigo) — `TimeSeriesSplit` + Optuna nos hiperparâmetros do LGBM
- **5%** validação — ajuste final do ensemble (treino+validação antes do teste)
- **20%** teste (mais recente) — métricas reportadas no HTML

Alvos: soma da **quantidade** de vendas e do **Valor Real de Venda** nos próximos **3, 7 e 30** dias (a partir de cada dia; features só com informação até esse dia).

## Estrutura do código

- `src/load.py` — leitura Excel e normalização de cabeçalhos
- `src/features.py` — série diária, calendário futuro (H dias), **STL rolante** (tendência/sazonalidade), picos (`rollmax`, quantis 85/90), eco semanal, burst vs média
- `src/ts_components.py` — **Ridge só em `ts_*`**, **LGBM médio + quantil alto** com γ calibrado no treino
- `src/weights.py` — pesos de amostra (recência + **magnitude do alvo** para não ignorar picos de VGV)
- `src/train_eval.py` — Optuna (LGBM, ~75 trials por defeito) + **zoo** (LGBM, **CatBoost**, XGB, HGB, **ExtraTrees**, **NGBoost** opcional, stack, baseline) + **super-blend**: combina os melhores K modelos com pesos `softmax(−MAE)` se bater o melhor individual na validação; peso temporal no LGBM/XGB/ET
- `src/report_html.py` — dashboard Plotly + Tailwind (CDN)

**Métricas:** o MAPE clássico explode quando há muitos zeros (volume em janelas curtas). O relatório mostra **sMAPE** e **MAPE restrito** (só onde o valor real é relevante).

Coloque os quatro `.xlsx` na pasta do projeto (ou noutra pasta e use `--data-dir`).
