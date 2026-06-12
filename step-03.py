# =============================================================
# STEP 03 - Auditoria das tres bases de dados
# -------------------------------------------------------------
# Objetivo : Auditar as tres bases do pipeline (Yahoo Finance,
#            World Bank API e UCI Online Retail II), verificando
#            nulos por coluna, duplicatas, tipos de dado e taxa
#            de aproveitamento apos dropna(). Produz um relatorio
#            consolidado unico cobrindo as tres fontes.
# Entrada  : output/base_step_00_yahoo_finance.csv
#            output/base_step_01_world_bank.csv
#            output/base_step_02_online_retail.csv
# Saida    : output/base_step_03_auditoria.csv
# Depende  : STEPS 00, 01, 02
# =============================================================
import pandas as pd
import numpy as np
import os

file_output = 'output/base_step_03_auditoria.csv'
os.makedirs('output', exist_ok=True)

BASES = {
    'Yahoo Finance':     'output/base_step_00_yahoo_finance.csv',
    'World Bank API':    'output/base_step_01_world_bank.csv',
    'UCI Online Retail': 'output/base_step_02_online_retail.csv',
}


def print_separador(titulo):
    linha = '=' * 65
    print(f"\n{linha}")
    print(f"  {titulo}")
    print(linha)


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
resumo_geral  = []
linhas_colunas = []

for nome_base, caminho in BASES.items():
    print_separador(f'BASE: {nome_base}')
    print(f"  Arquivo: {caminho}")

    df = pd.read_csv(caminho, sep=';', decimal=',',
                     encoding='utf-8-sig', low_memory=False)

    n_linhas, n_colunas = df.shape
    n_duplicatas        = df.duplicated().sum()
    n_nulos_total       = df.isna().sum().sum()
    pct_nulos_celulas   = (n_nulos_total / (n_linhas * n_colunas)) * 100
    df_sem_nulos        = df.dropna()
    n_aproveitados      = len(df_sem_nulos)
    pct_aproveitamento  = (n_aproveitados / n_linhas) * 100

    print(f"\n  Forma (linhas x colunas) : {n_linhas:,} x {n_colunas}")
    print(f"  Duplicatas               : {n_duplicatas:,}")
    print(f"  Celulas nulas            : {n_nulos_total:,}  "
          f"({pct_nulos_celulas:.1f}% das {n_linhas * n_colunas:,} celulas)")
    print(f"  Registros apos dropna()  : {n_aproveitados:,}  "
          f"({pct_aproveitamento:.1f}% do total)")

    # Detalhe por coluna
    print(f"\n  Auditoria por coluna:")
    tbl_cols = []
    for col in df.columns:
        n_nulos  = int(df[col].isna().sum())
        pct_col  = round((n_nulos / n_linhas) * 100, 1)
        n_unicos = df[col].nunique()
        dtype    = str(df[col].dtype)
        exemplo  = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else 'N/A'
        if len(exemplo) > 30:
            exemplo = exemplo[:27] + '...'
        row_col = {
            'Base':      nome_base,
            'Coluna':    col,
            'Dtype':     dtype,
            'Nulos':     n_nulos,
            'Nulos_%':   pct_col,
            'Unicos':    n_unicos,
            'Exemplo':   exemplo,
        }
        tbl_cols.append(row_col)
        linhas_colunas.append(row_col)

    print_tabela_bonita(
        pd.DataFrame(tbl_cols)[['Coluna', 'Dtype', 'Nulos', 'Nulos_%', 'Unicos', 'Exemplo']]
    )

    resumo_geral.append({
        'Base':               nome_base,
        'Linhas':             n_linhas,
        'Colunas':            n_colunas,
        'Duplicatas':         n_duplicatas,
        'Nulos_total':        n_nulos_total,
        'Nulos_pct_celulas':  round(pct_nulos_celulas, 1),
        'Aproveitados':       n_aproveitados,
        'Aproveitamento_pct': round(pct_aproveitamento, 1),
    })


# Resumo consolidado das tres bases
print_separador('RESUMO CONSOLIDADO DAS TRES BASES')
df_resumo = pd.DataFrame(resumo_geral)
print_tabela_bonita(df_resumo)

# Salvar detalhes por coluna (todas as bases em um unico CSV)
df_saida = pd.DataFrame(linhas_colunas)
df_saida.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo em: {file_output}")