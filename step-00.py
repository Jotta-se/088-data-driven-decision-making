# =============================================================
# STEP 00 - Download e cache da base Yahoo Finance
# -------------------------------------------------------------
# Objetivo : Baixar precos ajustados de fechamento para todos
#            os tickers usados nos steps 06-14 e salvar como
#            cache CSV (formato longo) para auditoria e
#            exploracao nos steps seguintes.
# Entrada  : Yahoo Finance (download automatico via yfinance)
# Saida    : output/base_step_00_yahoo_finance.csv
# Depende  : -- (primeiro step do pre-pipeline, requer internet)
# =============================================================
import pandas as pd
import numpy as np
import yfinance as yf
import os

file_output = 'output/base_step_00_yahoo_finance.csv'
os.makedirs('output', exist_ok=True)

# Universo completo de tickers usados nos steps 06-14
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN',   # steps 06, 07 (Nash, Stackelberg)
    'JPM',  'JNJ',  'XOM',  'BRK-B',   # step 08 (Pareto / fronteira eficiente)
    'SPY',  'EEM',  'TLT',  'GLD',      # steps 13, 14 (Robusta, Estocastica)
    'EWJ',  'EWZ',  'IEF',  'VEA',      # steps 13, 14
]
START = '2015-01-01'
END   = '2023-12-31'

print(f"Baixando dados do Yahoo Finance ({START} a {END})...")
print(f"Tickers ({len(TICKERS)}): {', '.join(TICKERS)}\n")

raw    = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)
precos = raw['Close'].dropna(axis=1, how='all')
tickers_validos = list(precos.columns)
print(f"Tickers validos apos download: {tickers_validos}\n")

# Converter para formato longo (tidy): Data | Ticker | Close | Retorno_Diario
# Formato longo facilita auditoria e tabelas descritivas.
frames = []
for ticker in tickers_validos:
    serie = precos[ticker].dropna().reset_index()
    serie.columns = ['Data', 'Close']
    serie['Ticker']          = ticker
    serie['Retorno_Diario']  = serie['Close'].pct_change()
    frames.append(serie[['Data', 'Ticker', 'Close', 'Retorno_Diario']])

df = pd.concat(frames, ignore_index=True)
df['Data'] = df['Data'].astype(str)

total_obs       = len(df)
total_tickers   = df['Ticker'].nunique()
periodo_inicio  = df['Data'].min()
periodo_fim     = df['Data'].max()
nulos_retorno   = df['Retorno_Diario'].isna().sum()

resultados_df = pd.DataFrame({
    'Metrica':  ['Observacoes', 'Tickers validos', 'Periodo inicio', 'Periodo fim',
                 'Nulos Retorno_Diario'],
    'Valor':    [f'{total_obs:,}', total_tickers, periodo_inicio, periodo_fim,
                 f'{nulos_retorno} (1o registro de cada serie)'],
})
print(resultados_df.to_string(index=False))

df.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo em: {file_output}")