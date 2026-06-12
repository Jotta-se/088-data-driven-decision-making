# =============================================================
# STEP 05 - Dicionario de variaveis e pre-processamento
# -------------------------------------------------------------
# Objetivo : Gerar um dicionario de variaveis consolidado para
#            as tres bases (tipo, nulos%, unicos, exemplo) e
#            produzir versoes limpas (sem nulos criticos, sem
#            duplicatas) prontas para consumo pelos steps 06-16.
# Entrada  : output/base_step_00_yahoo_finance.csv
#            output/base_step_01_world_bank.csv
#            output/base_step_02_online_retail.csv
# Saida    : output/base_step_05_dicionario.csv
#            output/base_step_05_yahoo_limpa.csv
#            output/base_step_05_world_bank_limpa.csv
#            output/base_step_05_online_retail_limpa.csv
# Depende  : STEPS 00, 01, 02
# =============================================================
import pandas as pd
import numpy as np
import os

file_dicionario  = 'output/base_step_05_dicionario.csv'
file_yf_limpa    = 'output/base_step_05_yahoo_limpa.csv'
file_wb_limpa    = 'output/base_step_05_world_bank_limpa.csv'
file_ori_limpa   = 'output/base_step_05_online_retail_limpa.csv'
os.makedirs('output', exist_ok=True)


def como_numero(serie):
    return pd.to_numeric(
        serie.dropna().astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    )


def classificar_tipo(serie_bruta, serie_limpa):
    """Classifica a variavel como Qualitativa ou Quantitativa."""
    num = como_numero(serie_limpa)
    return 'Quantitativa' if num.notna().all() else 'Qualitativa'


def print_tabela_bonita(df):
    cols = list(df.columns)
    df_print = df.copy()
    for c in cols:
        df_print[c] = df_print[c].astype(str).str.replace('\n', ' ')
    col_widths = {c: max(df_print[c].str.len().max(), len(c)) for c in cols}
    header = ' ' + '  '.join(c.ljust(col_widths[c]) for c in cols) + ' '
    print(header)
    for _, row in df_print.iterrows():
        linha = ' ' + '  '.join(str(row[c]).ljust(col_widths[c]) for c in cols) + ' '
        print(linha)


# -------------------------------------------------------
dicionario = []


# --- Yahoo Finance ---
print("=" * 65)
print("  Pre-processamento: Yahoo Finance")
print("=" * 65)
df_yf = pd.read_csv('output/base_step_00_yahoo_finance.csv',
                    sep=';', decimal=',', encoding='utf-8-sig')

# Limpar: remover nulos (primeiro retorno de cada serie) e duplicatas
df_yf_limpa = df_yf.dropna().drop_duplicates()

n_bruto = len(df_yf)
n_limpo = len(df_yf_limpa)
print(f"  Bruto   : {n_bruto:,} registros")
print(f"  Limpo   : {n_limpo:,} registros  "
      f"(removidos: {n_bruto - n_limpo:,} — {(n_bruto - n_limpo) / n_bruto * 100:.1f}%)")

for col in df_yf.columns:
    s_bruta = df_yf[col]
    s_limpa = df_yf_limpa[col] if col in df_yf_limpa.columns else s_bruta
    exemplo = str(s_bruta.dropna().iloc[0]) if s_bruta.dropna().shape[0] > 0 else 'N/A'
    if len(exemplo) > 25:
        exemplo = exemplo[:22] + '...'
    dicionario.append({
        'Base':        'Yahoo Finance',
        'Variavel':    col,
        'Tipo':        classificar_tipo(s_bruta, s_limpa),
        'Dtype_pandas': str(s_bruta.dtype),
        'Nulos_pct':   round(s_bruta.isna().mean() * 100, 1),
        'Unicos':      s_bruta.nunique(),
        'Exemplo':     exemplo,
    })


# --- World Bank API ---
print("\n" + "=" * 65)
print("  Pre-processamento: World Bank API")
print("=" * 65)
df_wb = pd.read_csv('output/base_step_01_world_bank.csv',
                    sep=';', decimal=',', encoding='utf-8-sig')

# Limpar: manter apenas registros com os tres indicadores completos
df_wb_limpa = df_wb.dropna().drop_duplicates()

n_bruto = len(df_wb)
n_limpo = len(df_wb_limpa)
print(f"  Bruto   : {n_bruto:,} registros (pais x ano)")
print(f"  Limpo   : {n_limpo:,} registros  "
      f"(removidos: {n_bruto - n_limpo:,} — {(n_bruto - n_limpo) / n_bruto * 100:.1f}%)")

for col in df_wb.columns:
    s_bruta = df_wb[col]
    s_limpa = df_wb_limpa[col] if col in df_wb_limpa.columns else s_bruta
    exemplo = str(s_bruta.dropna().iloc[0]) if s_bruta.dropna().shape[0] > 0 else 'N/A'
    if len(exemplo) > 25:
        exemplo = exemplo[:22] + '...'
    dicionario.append({
        'Base':        'World Bank API',
        'Variavel':    col,
        'Tipo':        classificar_tipo(s_bruta, s_limpa),
        'Dtype_pandas': str(s_bruta.dtype),
        'Nulos_pct':   round(s_bruta.isna().mean() * 100, 1),
        'Unicos':      s_bruta.nunique(),
        'Exemplo':     exemplo,
    })


# --- UCI Online Retail II ---
print("\n" + "=" * 65)
print("  Pre-processamento: UCI Online Retail II")
print("=" * 65)
df_ori = pd.read_csv('output/base_step_02_online_retail.csv',
                     sep=';', decimal=',', encoding='utf-8-sig',
                     low_memory=False)

# Limpar: remover Customer ID nulo, Quantity <= 0, Price <= 0, duplicatas
# Converter Quantity e Price para numerico antes de filtrar
df_ori['Quantity'] = pd.to_numeric(
    df_ori['Quantity'].astype(str).str.replace(',', '.'), errors='coerce'
)
df_ori['Price'] = pd.to_numeric(
    df_ori['Price'].astype(str).str.replace(',', '.'), errors='coerce'
)
df_ori_limpa = (
    df_ori[df_ori['Customer ID'].notna()]
    .loc[lambda d: d['Quantity'] > 0]
    .loc[lambda d: d['Price'] > 0]
    .drop_duplicates()
    .reset_index(drop=True)
)

n_bruto = len(df_ori)
n_limpo = len(df_ori_limpa)
print(f"  Bruto   : {n_bruto:,} registros")
print(f"  Limpo   : {n_limpo:,} registros  "
      f"(removidos: {n_bruto - n_limpo:,} — {(n_bruto - n_limpo) / n_bruto * 100:.1f}%)")
print(f"  Criterios de limpeza: Customer ID nao-nulo | Quantity > 0 | Price > 0")

for col in df_ori.columns:
    s_bruta = df_ori[col]
    s_limpa = df_ori_limpa[col] if col in df_ori_limpa.columns else s_bruta
    exemplo = str(s_bruta.dropna().iloc[0]) if s_bruta.dropna().shape[0] > 0 else 'N/A'
    if len(exemplo) > 25:
        exemplo = exemplo[:22] + '...'
    dicionario.append({
        'Base':        'UCI Online Retail',
        'Variavel':    col,
        'Tipo':        classificar_tipo(s_bruta, s_limpa),
        'Dtype_pandas': str(s_bruta.dtype),
        'Nulos_pct':   round(s_bruta.isna().mean() * 100, 1),
        'Unicos':      s_bruta.nunique(),
        'Exemplo':     exemplo,
    })


# -------------------------------------------------------
# Imprimir dicionario consolidado
# -------------------------------------------------------
print("\n\n" + "=" * 65)
print("  DICIONARIO DE VARIAVEIS CONSOLIDADO")
print("=" * 65)
df_dic = pd.DataFrame(dicionario)
print_tabela_bonita(df_dic)


# -------------------------------------------------------
# Salvar todos os arquivos
# -------------------------------------------------------
df_dic.to_csv(
    file_dicionario, sep=';', index=False, encoding='utf-8-sig', decimal=','
)
df_yf_limpa.to_csv(
    file_yf_limpa, sep=';', index=False, encoding='utf-8-sig', decimal=','
)
df_wb_limpa.to_csv(
    file_wb_limpa, sep=';', index=False, encoding='utf-8-sig', decimal=','
)
df_ori_limpa.to_csv(
    file_ori_limpa, sep=';', index=False, encoding='utf-8-sig', decimal=','
)

print(f"\nArquivos salvos em:")
print(f"  {file_dicionario}")
print(f"  {file_yf_limpa}")
print(f"  {file_wb_limpa}")
print(f"  {file_ori_limpa}")