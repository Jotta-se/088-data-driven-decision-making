# 000 · Math Models — Pipeline de Modelos Matemáticos Aplicados a Negócios

> Pipeline completo com **17 steps** (00–16): pré-pipeline de aquisição e auditoria de dados (steps 00–05) + 11 modelos matemáticos avançados aplicados a decisões de negócio, finanças e estratégia competitiva (steps 06–16). Os steps de modelagem são autocontidos, com aquisição automática de dados, cálculo analítico e geração de visualizações estáticas (PNG) e interativas (HTML/3D).

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Instalação do Ambiente com `uv`](#3-instalação-do-ambiente-com-uv)
4. [Estrutura do Projeto](#4-estrutura-do-projeto)
5. [Mapa de Dependências do Pipeline](#5-mapa-de-dependências-do-pipeline)
6. [Execução — Passo a Passo](#6-execução--passo-a-passo)
   - [Step 00 — Download Yahoo Finance](#step-00--download-yahoo-finance)
   - [Step 01 — Download World Bank API](#step-01--download-world-bank-api)
   - [Step 02 — Download UCI Online Retail II](#step-02--download-uci-online-retail-ii)
   - [Step 03 — Auditoria das Três Bases](#step-03--auditoria-das-três-bases)
   - [Step 04 — Tabelas Qualitativas e Quantitativas](#step-04--tabelas-qualitativas-e-quantitativas)
   - [Step 05 — Dicionário de Variáveis e Pré-processamento](#step-05--dicionário-de-variáveis-e-pré-processamento)
   - [Step 06 — Teoria dos Jogos e Equilíbrio de Nash](#step-06--teoria-dos-jogos-e-equilíbrio-de-nash)
   - [Step 07 — Equilíbrio de Stackelberg](#step-07--equilíbrio-de-stackelberg)
   - [Step 08 — Otimização de Pareto (Fronteira Eficiente)](#step-08--otimização-de-pareto-fronteira-eficiente)
   - [Step 09 — Análise Envoltória de Dados (DEA)](#step-09--análise-envoltória-de-dados-dea)
   - [Step 10 — Análise de Dados Topológica (TDA)](#step-10--análise-de-dados-topológica-tda)
   - [Step 11 — Teorema de Aproximação de Blackwell](#step-11--teorema-de-aproximação-de-blackwell)
   - [Step 12 — Teoria da Decisão Bayesiana (CLV)](#step-12--teoria-da-decisão-bayesiana-clv)
   - [Step 13 — Otimização Robusta](#step-13--otimização-robusta)
   - [Step 14 — Otimização Estocástica em Dois Estágios](#step-14--otimização-estocástica-em-dois-estágios)
   - [Step 15 — Transporte Ótimo (Wasserstein)](#step-15--transporte-ótimo-wasserstein)
   - [Step 16 — Equações de Bellman e Programação Dinâmica](#step-16--equações-de-bellman-e-programação-dinâmica)
7. [Referência de Outputs](#7-referência-de-outputs)
8. [Fontes de Dados](#8-fontes-de-dados)
9. [Dependências](#9-dependências)

---

## 1. Visão Geral

Este projeto implementa um pipeline progressivo de **17 steps (00–16)** organizados em dois blocos:

**Pré-pipeline de dados (steps 00–05)** — Download, auditoria e exploração das três fontes de dados:
- **Steps 00–02:** download e cache das bases (Yahoo Finance, World Bank API, UCI Online Retail II)
- **Step 03:** auditoria consolidada das três bases (nulos, duplicatas, tipos, aproveitamento)
- **Step 04:** tabelas de variáveis qualitativas e quantitativas das três bases
- **Step 05:** dicionário de variáveis consolidado + versões limpas das bases

**Pipeline de modelagem (steps 06–16)** — 11 modelos matemáticos avançados aplicados a finanças, estratégia competitiva e decisão. Cada step entrega:
- **1 arquivo CSV** com os resultados analíticos tabulados
- **3 gráficos estáticos PNG** (matplotlib / seaborn) de alta resolução
- **2 visualizações HTML interativas** (Plotly), sendo uma delas sempre um gráfico 3D

Os modelos avançam de **Teoria dos Jogos** (Nash, Stackelberg) até **Programação Dinâmica** e **Transporte Ótimo**, consumindo dados reais de fontes públicas.

---

## 2. Pré-requisitos

| Requisito | Versão mínima | Observação |
|-----------|--------------|------------|
| **Python** | 3.13 | Gerenciado automaticamente pelo `uv` |
| **uv** | 0.4+ | Gerenciador de pacotes e ambientes virtuais |
| **Git** | qualquer | Para clonar o repositório |
| Conexão à internet | — | Necessária para download dos dados na primeira execução |

### Instalar o `uv` (caso não tenha)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Após a instalação, confirme:
```bash
uv --version
# Exemplo: uv 0.4.x (...)
```

---

## 3. Instalação do Ambiente com `uv`

### Passo 1 — Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd "000 - Math_Models"
```

### Passo 2 — Criar o ambiente virtual

O `uv` lê o arquivo `.python-version` (que contém `3.13`) e o `pyproject.toml` para configurar tudo automaticamente.

```bash
uv venv
```

**O que acontece:** O `uv` cria a pasta `.venv/` dentro do projeto contendo o interpretador Python 3.13 e um ambiente isolado. Se o Python 3.13 não estiver instalado na máquina, o `uv` faz o download e instalação automaticamente.

**Output esperado:**
```
Using Python 3.13.x
Creating virtual environment at: .venv
Activate with: .venv\Scripts\activate  (Windows)
              source .venv/bin/activate (Linux/macOS)
```

### Passo 3 — Instalar as dependências

```bash
uv sync
```

**O que acontece:** O `uv` lê o `uv.lock` (arquivo de lock com versões exatas e hashes) e instala todas as dependências dentro do `.venv/`. Como o lock já existe, a instalação é determinística — todos na equipe obterão exatamente as mesmas versões.

**Output esperado:**
```
Resolved XX packages in Xs
Installed XX packages in Xs
 + matplotlib==3.x.x
 + numpy==2.x.x
 + pandas==3.x.x
 + plotly==6.x.x
 + ripser==0.x.x
 + yfinance==1.x.x
 ...
```

### Passo 4 — Ativar o ambiente (opcional, mas recomendado)

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Após ativação, o prompt exibirá `(000-math-models)` indicando que o ambiente está ativo.

> **Alternativa sem ativar:** Use `uv run python step-06.py` para executar qualquer script diretamente no ambiente virtual sem precisar ativá-lo.

### Passo 5 — Criar a pasta de output

```bash
mkdir output
```

Todos os scripts esperam que a pasta `output/` exista. Se não existir, o próprio script a cria via `os.makedirs`, mas é boa prática criá-la antes.

### Resumo rápido (setup completo em 4 comandos)

```bash
git clone <URL> && cd "000 - Math_Models"
uv venv
uv sync
mkdir output
```

---

## 4. Estrutura do Projeto

```
000 - Math_Models/
│
├── pyproject.toml              # Metadados do projeto e dependências
├── uv.lock                     # Lock file (versões exatas de todos os pacotes)
├── .python-version             # Versão do Python usada (3.13)
├── .gitignore                  # Ignora .venv, __pycache__, etc.
├── README.md                   # Este arquivo
│
├── main.py                     # Entry-point de demonstração (hello world)
│
├── step-00.py                  # PRÉ-PIPELINE: Download Yahoo Finance (cache CSV)
├── step-01.py                  # PRÉ-PIPELINE: Download World Bank API (cache CSV)
├── step-02.py                  # PRÉ-PIPELINE: Download UCI Online Retail II (cache CSV)
├── step-03.py                  # PRÉ-PIPELINE: Auditoria das três bases
├── step-04.py                  # PRÉ-PIPELINE: Tabelas qualitativas e quantitativas
├── step-05.py                  # PRÉ-PIPELINE: Dicionário de variáveis + pré-processamento
│
├── step-06.py                  # Teoria dos Jogos — Equilíbrio de Nash
├── step-07.py                  # Equilíbrio de Stackelberg (líder-seguidor)
├── step-08.py                  # Fronteira Eficiente de Pareto
├── step-09.py                  # Análise Envoltória de Dados (DEA)
├── step-10.py                  # Análise de Dados Topológica (TDA / ripser)
├── step-11.py                  # Blackwell Approachability
├── step-12.py                  # Inferência Bayesiana — CLV Beta-Binomial
├── step-13.py                  # Otimização Robusta (Min-Max / Minimax)
├── step-14.py                  # Otimização Estocástica em Dois Estágios
├── step-15.py                  # Transporte Ótimo — Distância de Wasserstein
├── step-16.py                  # Programação Dinâmica — Equação de Bellman
│
├── output/                     # Gerado automaticamente pelos scripts
│   ├── base_step_06_*.{csv,png}
│   ├── base_step06_*.html
│   └── ...                     # Todos os outputs dos steps 06–16
│
├── Source_KM/                  # Base de conhecimento teórico
│   ├── Modelos Matemáticos para Gestão (...).md
│   └── Modelos e Teoremas Matemáticos (...).md
│
└── .venv/                      # Ambiente virtual (gerenciado pelo uv)
```

---

## 5. Mapa de Dependências do Pipeline

```
PRÉ-PIPELINE (exploração de dados)
───────────────────────────────────────────────────────────────
step-00  ──► base_step_00_yahoo_finance.csv    (16 tickers, 2015-2023)
step-01  ──► base_step_01_world_bank.csv       (33 países, 3 indicadores, 2000-2022)
step-02  ──► base_step_02_online_retail.csv    (Online Retail II bruto)
   │
   ├── step-03  (Auditoria das 3 bases)        ──► base_step_03_auditoria.csv
   ├── step-04  (Tabelas quali/quanti)         ──► base_step_04_qualitativas.csv
   │                                           ──► base_step_04_quantitativas.csv
   └── step-05  (Dicionário + limpeza)         ──► base_step_05_dicionario.csv
                                               ──► base_step_05_*_limpa.csv (3 arquivos)

PIPELINE DE MODELAGEM (downloads independentes)
───────────────────────────────────────────────────────────────
Yahoo Finance ──────────────┐
(AAPL, MSFT, GOOGL, AMZN)  ├─► step-06  (Nash)
                            ├─► step-07  (Stackelberg)
(8 ativos — + JPM, JNJ...) └─► step-08  (Pareto)

World Bank API ─────────────┬─► step-09  (DEA)  ──── cache WB ──► step-13 (Robusta)
                            └──────────────────────────────────► step-14 (Estocástica)

UCI Online Retail II ───────────► step-10  (TDA) ── cache ORI ──► step-11 (Blackwell)
                                                 └── cache ORI ──► step-12 (Bayesiana)
                                                 └── cache ORI ──► step-16 (Bellman)

World Bank API (2000–2022) ──────► step-15 (Wasserstein)   [download direto]
```

> **Importante (pipeline de modelagem):** Os steps 11, 12 e 16 leem `output/base_step_10_online_retail_raw.csv` gerado pelo step 10. Os steps 13 e 14 leem `output/base_step_09_world_bank_raw.csv` do step 09. Execute na ordem numérica ou garanta que os caches existam.
>
> **Nota (pré-pipeline):** Os caches gerados pelos steps 00–02 são independentes dos caches usados no pipeline de modelagem (steps 09/10 fazem seus próprios downloads). Os steps 00–05 são para exploração, auditoria e documentação das bases.

---

## 6. Execução — Passo a Passo

Execute cada step a partir da raiz do projeto (pasta `000 - Math_Models`):

```bash
# Com ambiente ativado:
python step-00.py

# Ou sem ativar (usando uv run):
uv run python step-00.py
```

---

### PRÉ-PIPELINE — Steps 00 a 05

> Os steps 00–05 são **opcionais** para quem quer apenas rodar os modelos. Seu objetivo é explorar, auditar e documentar as três fontes de dados antes de qualquer modelagem.

---

### Step 00 — Download Yahoo Finance

**Objetivo:** Baixar e armazenar em cache os preços ajustados de **16 tickers** (universo completo usado nos steps 06–14) no formato CSV longo (`Data | Ticker | Close | Retorno_Diario`), cobrindo o período 2015–2023.

**Fonte de dados:** Yahoo Finance via `yfinance` (download automático).

**Tickers baixados:**

| Grupo | Tickers | Usado nos steps |
|-------|---------|-----------------|
| Big Tech | AAPL, MSFT, GOOGL, AMZN | 06, 07 |
| Diversificação | JPM, JNJ, XOM, BRK-B | 08 |
| ETFs globais | SPY, EEM, TLT, GLD, EWJ, EWZ, IEF, VEA | 13, 14 |

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_00_yahoo_finance.csv` | CSV | ~50 mil linhas — Data, Ticker, Close, Retorno_Diario |

**Comando:**
```bash
python step-00.py
```

**Output de console esperado:**
```
Baixando dados do Yahoo Finance (2015-01-01 a 2023-12-31)...
Tickers (16): AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, XOM, BRK-B, ...
Tickers validos apos download: [...]

Metrica                       Valor
Observacoes                   ~50.000
Tickers validos               16
Periodo inicio                2015-01-02
Periodo fim                   2023-12-29
Nulos Retorno_Diario          16 (1o registro de cada serie)

Arquivo salvo em: output/base_step_00_yahoo_finance.csv
```

---

### Step 01 — Download World Bank API

**Objetivo:** Baixar três indicadores macroeconômicos do Banco Mundial para **33 países** (união dos países usados nos steps 09 e 15), cobrindo os anos 2000–2022, e salvar em cache CSV.

**Fonte de dados:** World Bank REST API (download automático via `requests`).

**Indicadores e países:**

| Indicador | Código WB | Coluna no CSV |
|-----------|-----------|---------------|
| PIB per capita (USD) | `NY.GDP.PCAP.CD` | `PIB_pc_USD` |
| Formação bruta de capital (% PIB) | `NE.GFI.TOTL.ZS` | `Capital_pct_PIB` |
| Força de trabalho (total) | `SL.TLF.TOTL.IN` | `Forca_Trabalho` |

Regiões cobertas: América do Norte (3), América do Sul (5), Europa (11), Ásia (9), África (5).

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_01_world_bank.csv` | CSV | ~700 linhas — ISO, Pais, Ano, PIB_pc_USD, Capital_pct_PIB, Forca_Trabalho |

**Comando:**
```bash
python step-01.py
```

---

### Step 02 — Download UCI Online Retail II

**Objetivo:** Baixar o dataset **Online Retail II** do UCI Machine Learning Repository (~45 MB ZIP), extrair o sheet `Year 2010-2011` do Excel e salvar como cache CSV.

**Fonte de dados:** UCI Repository (download automático via `requests` + `zipfile` + `openpyxl`).

**Colunas da base:**

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Invoice` | Qualitativa | Código da fatura |
| `StockCode` | Qualitativa | Código do produto |
| `Description` | Qualitativa | Descrição do produto |
| `Quantity` | Quantitativa | Quantidade vendida |
| `InvoiceDate` | Qualitativa (data) | Data e hora da transação |
| `Price` | Quantitativa | Preço unitário (GBP) |
| `Customer ID` | Qualitativa (ID) | Identificador do cliente |
| `Country` | Qualitativa | País de origem |

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_02_online_retail.csv` | CSV | ~530 mil linhas — base completa bruta |

**Comando:**
```bash
python step-02.py
```

> **Nota:** A primeira execução faz download de ~45 MB. Pode levar 1–3 min dependendo da conexão.

---

### Step 03 — Auditoria das Três Bases

**Objetivo:** Realizar uma auditoria completa e consolidada das três bases do pipeline em um único script. Para cada base verifica: nulos por coluna (count e %), duplicatas, tipos de dados pandas, taxa de aproveitamento após `dropna()`. Produz um relatório tabular detalhado no console e um CSV de saída com os metadados de qualidade.

**Fonte de dados:** Caches gerados pelos steps 00, 01 e 02.

**Processo analítico:**
1. Lê os três CSVs de cache com `sep=';', decimal=','`.
2. Calcula `shape`, `df.duplicated().sum()`, `df.isna().sum()` por coluna.
3. Avalia o aproveitamento: `len(df.dropna()) / len(df) × 100`.
4. Imprime o detalhe por coluna para cada base e o resumo consolidado.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_03_auditoria.csv` | CSV | Por coluna e base: Dtype, Nulos, Nulos_%, Unicos, Exemplo |

**Comando:**
```bash
python step-03.py
```

**Output de console esperado (extrato):**
```
=================================================================
  BASE: Yahoo Finance
=================================================================
  Forma (linhas x colunas) : 50.320 x 4
  Duplicatas               : 0
  Celulas nulas            : 16  (0.0% das celulas)
  Registros apos dropna()  : 50.304  (99.97%)

  Auditoria por coluna:
 Coluna          Dtype    Nulos  Nulos_%  Unicos  Exemplo
 Data            object   0      0.0      2269    2015-01-02
 Ticker          object   0      0.0      16      AAPL
 Close           float64  0      0.0      49847   127.1399...
 Retorno_Diario  float64  16     0.0      49836   0.033...
```

---

### Step 04 — Tabelas Qualitativas e Quantitativas

**Objetivo:** Identificar e descrever em um único script todas as variáveis das três bases, separando-as em **qualitativas** (moda, número e lista de categorias, frequência da moda) e **quantitativas** (N observações, mínimo, máximo, média, quartis, desvio padrão). Gera dois CSVs consolidados: um para cada tipo de variável.

**Fonte de dados:** Caches gerados pelos steps 00, 01 e 02.

**Critério de classificação:**
- **Qualitativa:** nenhum valor da coluna converte para número via `pd.to_numeric(..., errors='coerce')`.
- **Quantitativa:** todos os valores não-nulos da coluna convertem para número.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_04_qualitativas.csv` | CSV | Base, Variável, N_Categorias, Moda, Freq_Moda_%, Categorias |
| `base_step_04_quantitativas.csv` | CSV | Base, Variável, N_obs, Min., Max., Média, Q1, Mediana, Q3, Desvio_Pad |

**Comando:**
```bash
python step-04.py
```

**Output de console esperado (extrato):**
```
=================================================================
  Yahoo Finance  —  VARIAVEIS QUALITATIVAS (2)
=================================================================
 Variavel  N_Categorias  Moda        Freq_Moda_%  Categorias
 Data      2269          2015-01-02  0.0          2015-01-02, 2015-01-05, ...
 Ticker    16            AAPL        6.2          AAPL, AMZN, BRK-B, EEM, ...

=================================================================
  Yahoo Finance  —  VARIAVEIS QUANTITATIVAS (2)
=================================================================
 Variavel        N_obs   Min.    Max.       Media    1o_Quartil  Mediana  3o_Quartil  Desvio_Pad
 Close           50304   8.4800  3617.0000  254.17   58.9400     135.40   287.0000    451.32
 Retorno_Diario  50304  -0.2099    0.1397    0.0005  -0.0068       0.00     0.0074      0.0185
```

---

### Step 05 — Dicionário de Variáveis e Pré-processamento

**Objetivo:** Gerar um **dicionário de variáveis consolidado** para todas as colunas das três bases (tipo, dtype pandas, % de nulos, valores únicos, exemplo) e produzir **versões limpas** de cada base, aplicando critérios específicos de limpeza por fonte:

| Base | Critério de limpeza |
|------|---------------------|
| Yahoo Finance | `dropna()` + `drop_duplicates()` |
| World Bank API | `dropna()` + `drop_duplicates()` (exige os 3 indicadores) |
| UCI Online Retail II | `Customer ID` não-nulo + `Quantity > 0` + `Price > 0` + `drop_duplicates()` |

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_05_dicionario.csv` | CSV | Por variável e base: Tipo, Dtype_pandas, Nulos_%, Unicos, Exemplo |
| `base_step_05_yahoo_limpa.csv` | CSV | Yahoo Finance limpa (sem NaNs de retorno) |
| `base_step_05_world_bank_limpa.csv` | CSV | World Bank com os 3 indicadores completos |
| `base_step_05_online_retail_limpa.csv` | CSV | UCI filtrada (~400 mil registros válidos) |

**Comando:**
```bash
python step-05.py
```

---

### PIPELINE DE MODELAGEM — Steps 06 a 16

> A partir daqui cada step é independente: faz seu próprio download dos dados necessários. Os caches dos steps 00–05 **não** são prerequisitos dos steps 06–16.

---

### Step 06 — Teoria dos Jogos e Equilíbrio de Nash

**Objetivo:** Modelar a interação estratégica entre quatro grandes empresas de tecnologia (AAPL, MSFT, GOOGL, AMZN) como um jogo de competição de mercado. Constrói a **matriz de payoff** com base nos retornos históricos reais e identifica os **Equilíbrios de Nash em estratégias puras**.

**Fonte de dados:** Yahoo Finance — download automático de preços ajustados (2021-01-01 a 2023-12-31).

**Processo analítico:**
1. Calcula retornos diários, retorno anual, volatilidade anual e Sharpe ratio para cada ativo.
2. Monta a matriz de payoff A e B onde `payoff_A[i,j] = retorno_i − correlação(i,j) × retorno_j` — capturando que a vantagem competitiva diminui quando os rivais estão correlacionados.
3. Varre todos os pares `(i, j)` procurando onde **ambos os jogadores estão na melhor resposta simultânea** (definição de Equilíbrio de Nash em estratégias puras).
4. Exporta estatísticas e equilíbrios encontrados.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_06_nash_equilibrio.csv` | CSV | Retorno anual, volatilidade, Sharpe e equilíbrios por empresa |
| `base_step_06_nash_payoff_heatmap.png` | PNG | Heatmap das matrizes de payoff do Jogador A e B |
| `base_step_06_nash_estrategias.png` | PNG | Espaço de estratégias: Retorno × Volatilidade (Nash destacado em dourado) |
| `base_step_06_nash_retornos_cumulados.png` | PNG | Retornos acumulados dos 4 concorrentes (2021–2023) |
| `base_step06_nash_payoff_interativo.html` | HTML | Heatmap interativo da matriz de payoff (Plotly) |
| `base_step06_nash_retornos_interativo.html` | HTML | Série temporal de retornos acumulados interativa |

**Comando:**
```bash
python step-06.py
```

**Output de console esperado:**
```
Baixando dados do Yahoo Finance...
Dados baixados: 754 observações, 4 ativos
Equilíbrios de Nash (estratégias puras) encontrados: N
  (AAPL, AAPL) → payoff_A=X.XXXX, payoff_B=X.XXXX
Arquivo salvo: output/base_step_06_nash_equilibrio.csv
```

---

### Step 07 — Equilíbrio de Stackelberg

**Objetivo:** Modelar a **competição sequencial líder-seguidor** (modelo de Stackelberg) no mercado de tecnologia. A Apple (AAPL) age como líder — anuncia sua quantidade de produção primeiro — e Microsoft, Google e Amazon (seguidores) respondem de forma ótima. Compara o resultado com o equilíbrio simultâneo de **Nash-Cournot**.

**Fonte de dados:** Yahoo Finance — mesmo período do step 06.

**Processo analítico:**
1. Estima os **custos marginais** de cada empresa com base no índice de Sharpe normalizado: empresas mais eficientes (maior Sharpe) têm custos menores.
2. Resolve o **Nash-Cournot simultâneo** por iteração de ponto fixo (1000 iterações, tolerância 1e-9).
3. Resolve o **Stackelberg sequencial** via `scipy.optimize.minimize_scalar`: o líder maximiza seu lucro antecipando a resposta ótima dos seguidores.
4. Calcula a **vantagem do primeiro mover** (% de aumento de lucro do líder no Stackelberg vs Cournot).

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_07_stackelberg_resultado.csv` | CSV | Papel, custo marginal, quantidade e lucro por modelo (Cournot/Stackelberg) |
| `base_step_07_stackelberg_equilibrio.png` | PNG | Quantidades de equilíbrio: Cournot vs Stackelberg por empresa |
| `base_step_07_stackelberg_comparativo.png` | PNG | Lucros comparativos com líder destacado |
| `base_step_07_stackelberg_lucros.png` | PNG | Vantagem do Stackelberg vs Cournot (%) por empresa |
| `base_step07_stackelberg_interativo.html` | HTML | Gráfico interativo de lucros por modelo |
| `base_step07_stackelberg_3d.html` | HTML | Superfície 3D do lucro do líder em função de (q_Líder, q_Seguidores) |

**Comando:**
```bash
python step-07.py
```

---

### Step 08 — Otimização de Pareto (Fronteira Eficiente)

**Objetivo:** Construir a **fronteira eficiente de Pareto** para um portfólio de 8 ativos via simulação de Monte Carlo. Identifica carteiras ótimas que não podem melhorar retorno sem aumentar risco (e vice-versa). Compara o portfólio de **Sharpe máximo** com o de **mínima variância**.

**Fonte de dados:** Yahoo Finance — 8 ativos (AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, XOM, BRK-B), período 2019–2023.

**Processo analítico:**
1. Simula **8.000 portfólios aleatórios** (pesos amostrados de uma distribuição Dirichlet para garantir soma = 1 e pesos ≥ 0).
2. Para cada portfólio calcula retorno anualizado, volatilidade e Sharpe ratio.
3. Identifica a **fronteira de Pareto**: portfólio P é eficiente se não existe Q com retorno ≥ e volatilidade ≤ (e ao menos uma diferença estrita).
4. Otimiza o portfólio de **mínima variância** via `scipy.optimize.minimize` com SLSQP.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_08_pareto_portfolios.csv` | CSV | Retorno, volatilidade, Sharpe, flag Pareto e pesos para todos os 8.000 portfólios |
| `base_step_08_pareto_fronteira.png` | PNG | Scatter de risco×retorno com fronteira de Pareto, Sharpe máx (★) e mínima variância (◆) |
| `base_step_08_pareto_pesos_otimos.png` | PNG | Barras horizontais dos pesos nos dois portfólios ótimos |
| `base_step_08_pareto_retornos_hist.png` | PNG | Distribuição de retornos: todos vs Pareto eficientes |
| `base_step08_pareto_fronteira_interativa.html` | HTML | Fronteira de Pareto interativa (hover com métricas) |
| `base_step08_pareto_3d.html` | HTML | Nuvem 3D Risco × Retorno × Sharpe colorida por índice de Sharpe |

**Comando:**
```bash
python step-08.py
```

---

### Step 09 — Análise Envoltória de Dados (DEA)

**Objetivo:** Medir a **eficiência relativa de 30 países** (DMUs — Decision Making Units) usando o modelo **DEA-CRS** (Constant Returns to Scale / modelo CCR). Inputs: formação bruta de capital (% PIB) e força de trabalho. Output: PIB per capita. Identifica quais países estão na **fronteira de eficiência** (θ = 1) e quais têm espaço de melhoria.

**Fonte de dados:** World Bank API — download automático com cache em CSV. Na primeira execução consulta a API; nas seguintes lê o cache local.

**Processo analítico:**
1. Baixa 3 indicadores via API REST do Banco Mundial para 30 países (2021).
2. Normaliza inputs/output com Min-Max Scaling.
3. Para cada DMU k resolve um **problema de programação linear** (via `scipy.linprog` com método HiGHS): maximiza o score de eficiência θ_k sujeito às restrições de output e input.
4. Classifica países como eficientes (θ ≥ 0.999) ou ineficientes.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_09_world_bank_raw.csv` | CSV | **Cache** dos dados brutos do World Bank (usado pelos steps 13 e 14) |
| `base_step_09_dea_eficiencia.csv` | CSV | Score DEA, rank e indicadores originais por país |
| `base_step_09_dea_scores.png` | PNG | Barras horizontais dos scores DEA (top 25 países) |
| `base_step_09_dea_inputs_outputs.png` | PNG | Scatter: Capital vs PIB per capita e Trabalho (log) vs PIB, coloridos por eficiência |
| `base_step_09_dea_benchmarks.png` | PNG | Histograma da distribuição dos scores DEA |
| `base_step09_dea_interativo.html` | HTML | Gráfico de barras interativo dos scores DEA |
| `base_step09_dea_3d.html` | HTML | 3D: Capital × Trabalho × PIB per capita colorido por eficiência |

**Comando:**
```bash
python step-09.py
```

> **Cache:** Se `output/base_step_09_world_bank_raw.csv` já existir, o step não consulta a API. Para forçar atualização, delete o cache antes de executar.

---

### Step 10 — Análise de Dados Topológica (TDA)

**Objetivo:** Extrair **estruturas topológicas latentes** nos dados de transações de e-commerce usando **Homologia Persistente**. Computa features RFM (Recência, Frequência, Monetário) por cliente, aplica `ripser` para calcular diagramas de persistência (H0 e H1) e estima os **Números de Betti** para segmentação topológica de clientes.

**Fonte de dados:** Online Retail II (UCI Repository) — download automático do arquivo ZIP (~45 MB). O sheet `Year 2010-2011` é usado. Cache salvo em CSV para steps posteriores.

**Processo analítico:**
1. Baixa e descomprime o dataset UCI, salva cache CSV.
2. Limpa os dados (remove `Customer ID` nulos, `Quantity ≤ 0`, `Price ≤ 0`).
3. Calcula **features RFM** por cliente: recência (dias desde última compra), frequência (nº de pedidos únicos), monetário (receita total).
4. Padroniza com `StandardScaler` e amostra 500 clientes.
5. Calcula **Homologia Persistente** (H0 = componentes conectados, H1 = loops/ciclos) via `ripser` — com fallback gracioso se `ripser` não estiver disponível.
6. Segmenta clientes em **Bronze / Prata / Ouro** via RFM Score.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_10_online_retail_raw.csv` | CSV | **Cache** do dataset Online Retail II (usado pelos steps 11, 12 e 16) |
| `base_step_10_tda_features_rfm.csv` | CSV | Features RFM, segmento, cluster e Números de Betti por cliente |
| `base_step_10_tda_persistencia.png` | PNG | Diagramas de persistência H0 (componentes) e H1 (loops) |
| `base_step_10_tda_betti.png` | PNG | Lollipop chart dos Números de Betti por dimensão homológica |
| `base_step_10_tda_rfm_distribuicao.png` | PNG | Distribuição de Recência, Frequência e Monetário por cluster |
| `base_step10_tda_scatter_interativo.html` | HTML | Scatter interativo Recência × Monetário colorido por cluster |
| `base_step10_tda_rfm_3d.html` | HTML | Espaço RFM 3D com segmentação topológica |

**Comando:**
```bash
python step-10.py
```

> **Nota:** A primeira execução faz download de ~45 MB do UCI. Pode levar 1–3 min dependendo da conexão. Execuções subsequentes usam o cache.

---

### Step 11 — Teorema de Aproximação de Blackwell

**Objetivo:** Demonstrar o **Teorema de Aproximação de Blackwell** (Blackwell Approachability) em um jogo sequencial de recomendação de produtos. Um agente escolhe categorias de produto para recomendar ao longo de **T = 1.000 rodadas**; o ambiente responde. O algoritmo garante que o **payoff médio converge ao conjunto-alvo C**, com arrependimento externo decaindo em O(1/√T).

**Fonte de dados:** `output/base_step_10_online_retail_raw.csv` (gerado pelo step 10).

**Processo analítico:**
1. Extrai as top 6 categorias por receita (primeiros 2 chars do StockCode como proxy de categoria).
2. Estima payoffs de receita e aceitação normalizados por categoria.
3. Executa o **algoritmo de Blackwell**: a cada rodada, escolhe ação segundo distribuição mista; recebe payoff com ruído; atualiza a distribuição na direção que minimiza a distância ao conjunto-alvo C.
4. Mede **arrependimento externo** (distância para a melhor ação fixa) e **distância ao conjunto C** ao longo das rodadas.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_11_blackwell_convergencia.csv` | CSV | Por rodada: ação, payoffs, médias acumuladas, distância a C e regret |
| `base_step_11_blackwell_regret.png` | PNG | Arrependimento externo real vs limite teórico O(1/√T) |
| `base_step_11_blackwell_trajetoria.png` | PNG | Trajetória do payoff médio convergindo ao conjunto-alvo C |
| `base_step_11_blackwell_payoffs.png` | PNG | Frequência de ações + evolução da distância ao conjunto C |
| `base_step11_blackwell_convergencia_interativa.html` | HTML | Convergência interativa (regret, limite teórico e distância a C) |
| `base_step11_blackwell_regret_interativo.html` | HTML | Trajetória do payoff médio no espaço 2D (Receita × Aceitação) |

**Comando:**
```bash
python step-11.py
```

> **Dependência:** Requer `output/base_step_10_online_retail_raw.csv`. Execute o step 10 antes.

---

### Step 12 — Teoria da Decisão Bayesiana (CLV)

**Objetivo:** Estimar o **Customer Lifetime Value (CLV) Bayesiano** dos clientes do Online Retail II usando um modelo **Beta-Binomial conjugado**. A prior Beta(α₀=2, β₀=5) codifica a crença inicial sobre a probabilidade de recompra; os dados observados atualizam a distribuição posterior por cliente. Segmenta clientes em quatro níveis de valor (Bronze, Prata, Ouro, Diamante).

**Fonte de dados:** `output/base_step_10_online_retail_raw.csv`.

**Processo analítico:**
1. Divide os dados em janela de calibração (70% do período por data) e holdout (30%).
2. Para cada cliente: calcula n_pedidos, recência, ticket médio e flag de recompra no holdout.
3. Atualiza posterior: `Beta(α₀ + s, β₀ + (1-s))` onde `s = 1` se recomprou.
4. Calcula `CLV_Bayesiano = p_recompra × ticket_médio × n_pedidos × horizonte`.
5. Segmenta por quartis de CLV e gera regras de priorização.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_12_bayesiana_clv.csv` | CSV | Por cliente: α_post, β_post, p_recompra, CLV, incerteza, segmento, rank |
| `base_step_12_bayesiana_posterior.png` | PNG | Prior → Posterior Beta-Binomial por segmento (2×2 subplots) |
| `base_step_12_bayesiana_clv_dist.png` | PNG | KDE do CLV Bayesiano por segmento |
| `base_step_12_bayesiana_atualizacao.png` | PNG | Convergência da posterior com acúmulo de evidências (4 taxas de recompra) |
| `base_step12_bayesiana_interativo.html` | HTML | Scatter interativo CLV × P(Recompra) × ticket médio |
| `base_step12_bayesiana_clv_3d.html` | HTML | 3D: Recência × Frequência × CLV colorido por segmento |

**Comando:**
```bash
python step-12.py
```

---

### Step 13 — Otimização Robusta

**Objetivo:** Construir um portfólio de 8 ETFs globais que seja **robustamente ótimo para o pior cenário** dentro de um conjunto de incerteza elipsoidal nos retornos. Resolve o problema Minimax: `max_w min_{r̃ ∈ Γ} wᵀr̃`, cuja solução analítica é `max_w [wᵀμ − κ·√(wᵀΣw)]`. Compara 5 níveis de conservadorismo (κ = 0, 0.5, 1.0, 1.5, 2.0).

**Fonte de dados:** `output/base_step_09_world_bank_raw.csv` (cache do step 09) + Yahoo Finance (ETFs: EEM, VEA, SPY, EWJ, EWZ, GLD, TLT, IEF), período 2018–2023.

**Processo analítico:**
1. Calcula retornos e covariância anualizados para os 8 ETFs.
2. Para cada κ: resolve o problema robusto via SLSQP; calcula retorno médio, volatilidade e **retorno garantido** (pior caso) = retorno médio − κ × volatilidade.
3. Gera 2.000 cenários Monte Carlo via Cholesky para avaliar a distribuição de retornos de cada solução.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_13_robusta_resultado.csv` | CSV | Por κ: retorno médio, volatilidade, retorno garantido, Sharpe e pesos |
| `base_step_13_robusta_portfolios.png` | PNG | Alocação dos portfólios por nível de conservadorismo (barras agrupadas) |
| `base_step_13_robusta_comparativo.png` | PNG | Retorno médio vs retorno garantido por κ (com faixa de incerteza) |
| `base_step_13_robusta_incerteza.png` | PNG | Violin plot da distribuição de retornos nos cenários Monte Carlo por κ |
| `base_step13_robusta_fronteira_interativa.html` | HTML | Fronteira robusta interativa: Risco × Retorno Médio e Garantido |
| `base_step13_robusta_cenarios.html` | HTML | Violin interativo da distribuição de retornos por nível de robustez |

**Comando:**
```bash
python step-13.py
```

---

### Step 14 — Otimização Estocástica em Dois Estágios

**Objetivo:** Resolver um problema de **programação estocástica em dois estágios** para alocação de portfólio. Compara três soluções: **EEV** (solução determinística com valor esperado dos retornos), **RP** (Recourse Problem — solução estocástica com penalidade de concentração), e **WS** (Wait-and-See — solução perfeita para cada cenário). Calcula o **VSS** (Value of Stochastic Solution = RP − EEV) e o **EVPI** (Expected Value of Perfect Information = WS − RP).

**Fonte de dados:** `output/base_step_09_world_bank_raw.csv` + Yahoo Finance (SPY, EEM, TLT, GLD, EWJ), período 2015–2023.

**Processo analítico:**
1. Gera 500 cenários Monte Carlo multivariados via decomposição de Cholesky.
2. Resolve **EEV**: otimização determinística sobre o retorno esperado.
3. Resolve **RP**: maximiza Sharpe com penalidade entrópica de concentração.
4. Resolve **WS**: para cada cenário, encontra o portfólio ótimo para aquele cenário específico.
5. Calcula VaR 5% para cada solução.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_14_estocastica_cenarios.csv` | CSV | Por cenário: retornos EEV, RP, WS e retornos individuais dos ativos |
| `base_step_14_estocastica_distribuicao.png` | PNG | KDE dos retornos EEV, RP e WS com VaR 5% marcado |
| `base_step_14_estocastica_ev_vs_ws.png` | PNG | Barras comparativas EEV × RP × WS + barras VSS e EVPI |
| `base_step_14_estocastica_solucoes.png` | PNG | Composição dos portfólios EEV vs RP |
| `base_step14_estocastica_cenarios_interativo.html` | HTML | Histograma interativo de retornos por solução |
| `base_step14_estocastica_3d.html` | HTML | 3D: Cenários SPY × EEM × Retorno RP colorido por retorno |

**Comando:**
```bash
python step-14.py
```

---

### Step 15 — Transporte Ótimo (Wasserstein)

**Objetivo:** Usar a teoria do **Transporte Ótimo** para comparar distribuições de renda (PIB per capita) entre 5 regiões geográficas (Américas do Norte e Sul, Europa, Ásia, África) ao longo de 22 anos (2000–2022). Calcula a **Distância de Wasserstein** (Earth Mover's Distance) entre regiões e resolve o **plano de transporte ótimo** entre Europa e África.

**Fonte de dados:** World Bank API — download direto (sem cache), indicador NY.GDP.PCAP.CD para 29 países de 2000 a 2022.

**Processo analítico:**
1. Baixa PIB per capita para 29 países em 5 regiões, 2000–2022.
2. Para cada par de regiões e cada ano de análise (2000, 2005, 2010, 2015, 2020, 2022): calcula `wasserstein_distance` via `scipy.stats`.
3. Resolve o **plano de transporte ótimo** Europa → África em 2022 via programação linear (linprog com método HiGHS), minimizando o custo de transporte `|u_i − v_j|`.
4. Calcula a evolução temporal da distância Wasserstein Europa × África (2000–2022).

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_15_transporte_plano.csv` | CSV | Matriz do plano de transporte ótimo Europa → África (2022) |
| `base_step_15_transporte_heatmap.png` | PNG | Heatmap da matriz de distâncias Wasserstein entre regiões (2022) |
| `base_step_15_transporte_distribuicoes.png` | PNG | Violin plot do log(PIB per capita) por região (2022) |
| `base_step_15_transporte_wasserstein.png` | PNG | Série temporal da distância Wasserstein Europa × África |
| `base_step15_transporte_interativo.html` | HTML | Heatmap interativo do plano de transporte ótimo |
| `base_step15_transporte_3d.html` | HTML | 3D: Distâncias Wasserstein por par de regiões ao longo do tempo |

**Comando:**
```bash
python step-15.py
```

---

### Step 16 — Equações de Bellman e Programação Dinâmica

**Objetivo:** Resolver um problema de **gestão de estoque via Programação Dinâmica** (indução retroativa). Usa os dados de quantidade do Online Retail II para estimar a distribuição de demanda diária do produto mais vendido. Define a **equação de Bellman** `V(t,s) = min_a [c(s,a) + γ·E_d[V(t+1,s')]]` e encontra a **política ótima de reposição** para um horizonte de 30 dias.

**Fonte de dados:** `output/base_step_10_online_retail_raw.csv`.

**Processo analítico:**
1. Identifica o produto mais transacionado e estima a distribuição de demanda diária (truncada no percentil 95).
2. Define o MDP: estados = estoque s ∈ {0…100}, ações = pedido a ∈ {0…50}, custos de pedido fixo (R$20), estoque (R$0,50/un) e falta (R$5/un), fator de desconto γ = 0,95.
3. Executa **indução retroativa** (backward induction) por T = 30 períodos: para cada par (s, a) avalia o custo esperado incluindo o custo de falta e o valor futuro descontado.
4. Extrai a política ótima `a*(s)` — quantidade a pedir para cada nível de estoque — e o **ponto de reposição** s*.

**Outputs gerados:**

| Arquivo | Tipo | Conteúdo |
|---------|------|----------|
| `base_step_16_bellman_politica.csv` | CSV | Por nível de estoque: ação ótima, função valor V*(s) e tipo de ação |
| `base_step_16_bellman_valor.png` | PNG | Função valor V*(s) com ponto de reposição marcado |
| `base_step_16_bellman_convergencia.png` | PNG | Convergência ΔV por iteração retroativa (escala log) |
| `base_step_16_bellman_politica_mapa.png` | PNG | Subplots: política ótima a*(s) e função valor V*(s) |
| `base_step16_bellman_interativo.html` | HTML | Política e função valor interativas (subplots Plotly) |
| `base_step16_bellman_3d.html` | HTML | Superfície 3D da evolução de V(s) ao longo das iterações retroativas |

**Comando:**
```bash
python step-16.py
```

---

## 7. Referência de Outputs

A tabela abaixo lista todos os arquivos gerados na pasta `output/`:

```
output/
│
│  PRÉ-PIPELINE (steps 00–05)
├── base_step_00_yahoo_finance.csv           ← Cache Yahoo Finance (longo)
├── base_step_01_world_bank.csv              ← Cache World Bank API (wide)
├── base_step_02_online_retail.csv           ← Cache UCI Online Retail II
├── base_step_03_auditoria.csv               ← Auditoria por coluna/base
├── base_step_04_qualitativas.csv            ← Tabela variáveis qualitativas
├── base_step_04_quantitativas.csv           ← Tabela variáveis quantitativas
├── base_step_05_dicionario.csv              ← Dicionário consolidado
├── base_step_05_yahoo_limpa.csv             ← Yahoo Finance limpa
├── base_step_05_world_bank_limpa.csv        ← World Bank limpa
├── base_step_05_online_retail_limpa.csv     ← UCI Online Retail limpa
│
│  PIPELINE DE MODELAGEM (steps 06–16)
├── base_step_06_nash_equilibrio.csv
├── base_step_06_nash_payoff_heatmap.png
├── base_step_06_nash_estrategias.png
├── base_step_06_nash_retornos_cumulados.png
├── base_step06_nash_payoff_interativo.html
├── base_step06_nash_retornos_interativo.html
│
├── base_step_07_stackelberg_resultado.csv
├── base_step_07_stackelberg_equilibrio.png
├── base_step_07_stackelberg_comparativo.png
├── base_step_07_stackelberg_lucros.png
├── base_step07_stackelberg_interativo.html
├── base_step07_stackelberg_3d.html
│
├── base_step_08_pareto_portfolios.csv
├── base_step_08_pareto_fronteira.png
├── base_step_08_pareto_pesos_otimos.png
├── base_step_08_pareto_retornos_hist.png
├── base_step08_pareto_fronteira_interativa.html
├── base_step08_pareto_3d.html
│
├── base_step_09_world_bank_raw.csv          ← CACHE (usado por steps 13 e 14)
├── base_step_09_dea_eficiencia.csv
├── base_step_09_dea_scores.png
├── base_step_09_dea_inputs_outputs.png
├── base_step_09_dea_benchmarks.png
├── base_step09_dea_interativo.html
├── base_step09_dea_3d.html
│
├── base_step_10_online_retail_raw.csv       ← CACHE (usado por steps 11, 12 e 16)
├── base_step_10_tda_features_rfm.csv
├── base_step_10_tda_persistencia.png
├── base_step_10_tda_betti.png
├── base_step_10_tda_rfm_distribuicao.png
├── base_step10_tda_scatter_interativo.html
├── base_step10_tda_rfm_3d.html
│
├── base_step_11_blackwell_convergencia.csv
├── base_step_11_blackwell_regret.png
├── base_step_11_blackwell_trajetoria.png
├── base_step_11_blackwell_payoffs.png
├── base_step11_blackwell_convergencia_interativa.html
├── base_step11_blackwell_regret_interativo.html
│
├── base_step_12_bayesiana_clv.csv
├── base_step_12_bayesiana_posterior.png
├── base_step_12_bayesiana_clv_dist.png
├── base_step_12_bayesiana_atualizacao.png
├── base_step12_bayesiana_interativo.html
├── base_step12_bayesiana_clv_3d.html
│
├── base_step_13_robusta_resultado.csv
├── base_step_13_robusta_portfolios.png
├── base_step_13_robusta_comparativo.png
├── base_step_13_robusta_incerteza.png
├── base_step13_robusta_fronteira_interativa.html
├── base_step13_robusta_cenarios.html
│
├── base_step_14_estocastica_cenarios.csv
├── base_step_14_estocastica_distribuicao.png
├── base_step_14_estocastica_ev_vs_ws.png
├── base_step_14_estocastica_solucoes.png
├── base_step14_estocastica_cenarios_interativo.html
├── base_step14_estocastica_3d.html
│
├── base_step_15_transporte_plano.csv
├── base_step_15_transporte_heatmap.png
├── base_step_15_transporte_distribuicoes.png
├── base_step_15_transporte_wasserstein.png
├── base_step15_transporte_interativo.html
├── base_step15_transporte_3d.html
│
├── base_step_16_bellman_politica.csv
├── base_step_16_bellman_valor.png
├── base_step_16_bellman_convergencia.png
├── base_step_16_bellman_politica_mapa.png
├── base_step16_bellman_interativo.html
└── base_step16_bellman_3d.html
```

**Pré-pipeline: 10 CSVs**
**Pipeline de modelagem: 11 CSVs + 2 caches + 33 PNGs + 22 HTMLs = 68 arquivos**
**Total geral: 78 arquivos**

---

## 8. Fontes de Dados

| Fonte | Steps | Dados | Protocolo |
|-------|-------|-------|-----------|
| **Yahoo Finance** (`yfinance`) | **00**, 06, 07, 08, 13, 14 | Preços ajustados de ações e ETFs (16 tickers, 2015–2023) | Step 00 gera cache exploratório; steps 06–14 fazem download independente |
| **World Bank API** | **01**, 09, 15 | PIB per capita, formação de capital, força de trabalho (33 países, 2000–2022) | Step 01 gera cache exploratório; step 09 usa cache próprio; step 15 faz download direto |
| **UCI Online Retail II** | **02**, 10, 11, 12, 16 | Transações de e-commerce (2010–2011, ~530 mil linhas) | Step 02 gera cache exploratório; step 10 gera cache para modelagem |

---

## 9. Dependências

Todas as dependências são declaradas no `pyproject.toml` e instaladas automaticamente pelo `uv sync`:

| Pacote | Versão mínima | Uso principal |
|--------|--------------|---------------|
| `matplotlib` | ≥ 3.10.9 | Gráficos estáticos (PNG) |
| `seaborn` | ≥ 0.13.2 | Heatmaps, violin plots, histogramas |
| `plotly` | ≥ 6.8.0 | Gráficos interativos HTML e 3D |
| `numpy` | ≥ 2.4.6 | Computação numérica vetorizada |
| `pandas` | ≥ 3.0.3 | Manipulação de DataFrames e I/O CSV |
| `scipy` | ≥ 1.17.1 | Otimização (SLSQP, linprog HiGHS), distribuições estatísticas, Wasserstein |
| `yfinance` | ≥ 1.4.1 | Download de dados de mercado (Yahoo Finance) |
| `requests` | ≥ 2.34.2 | Requisições HTTP (World Bank API, UCI) |
| `openpyxl` | ≥ 3.1.5 | Leitura do arquivo `.xlsx` do UCI |
| `ripser` | ≥ 0.6.15 | Homologia Persistente (step 10) |
| `persim` | ≥ 0.3.8 | Diagramas de persistência (step 10) |
| `nbformat` | ≥ 5.10.4 | Suporte a notebooks Jupyter |

### Notas sobre compatibilidade

- **`ripser`** requer um compilador C++ na máquina. No Windows, instale o [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Se `ripser` não puder ser importado, o step 10 usa um fallback automático (estimativa de Betti via distâncias) sem erros fatais.
- **`scipy.linprog` com método `highs`** requer scipy ≥ 1.7.0 (já garantido pela versão mínima declarada).
- **Python 3.13** é requerido. O `uv` gerencia isso automaticamente.

---

*Projeto mantido por Jotta — Série de Modelos Matemáticos para Gestão, Inteligência Competitiva e Data-Driven Decision Making.*
