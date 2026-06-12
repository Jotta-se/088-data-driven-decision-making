# =============================================================
# STEP 04 - Tabelas de variaveis qualitativas e quantitativas
# -------------------------------------------------------------
# Objetivo : Identificar e descrever as variaveis qualitativas
#            (moda, numero de categorias) e quantitativas
#            (min, max, media, quartis, mediana) das tres bases
#            do pipeline em um unico step consolidado.
# Entrada  : output/base_step_00_yahoo_finance.csv
#            output/base_step_01_world_bank.csv
#            output/base_step_02_online_retail.csv
# Saida    : output/base_step_04_qualitativas.csv
#            output/base_step_04_quantitativas.csv
# Depende  : STEPS 00, 01, 02
# =============================================================
import pandas as pd
import numpy as np
import os

file_quali  = 'output/base_step_04_qualitativas.csv'
file_quanti = 'output/base_step_04_quantitativas.csv'
os.makedirs('output', exist_ok=True)

BASES = {
    'Yahoo Finance':     'output/base_step_00_yahoo_finance.csv',
    'World Bank API':    'output/base_step_01_world_bank.csv',
    'UCI Online Retail': 'output/base_step_02_online_retail.csv',
}

MAX_CATS_EXIBIDAS = 8   # maximo de categorias distintas exibidas na tabela


def como_numero(serie):
    """Converte serie para numerico, tratando virgula decimal."""
    return pd.to_numeric(
        serie.dropna().astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    )


def eh_qualitativa(serie):
    """Retorna True se NENHUM valor da serie converte para numero."""
    num = como_numero(serie)
    return num.notna().sum() == 0


def eh_quantitativa(serie):
    """Retorna True se TODOS os valores nao-nulos da serie sao numericos."""
    if serie.dropna().empty:
        return False
    return como_numero(serie).notna().sum() == len(serie.dropna())


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
todas_quali  = []
todas_quanti = []

for nome_base, caminho in BASES.items():
    df = pd.read_csv(caminho, sep=';', decimal=',',
                     encoding='utf-8-sig', low_memory=False)

    qualitativas  = [c for c in df.columns if eh_qualitativa(df[c])]
    quantitativas = [c for c in df.columns if eh_quantitativa(df[c])]

    # --------------------------------------------------
    # TABELA DE VARIAVEIS QUALITATIVAS
    # --------------------------------------------------
    print_separador(f'{nome_base}  —  VARIAVEIS QUALITATIVAS ({len(qualitativas)})')

    if qualitativas:
        linhas_q = []
        for col in qualitativas:
            serie        = df[col].dropna()
            n_cats       = serie.nunique()
            moda         = serie.mode().iloc[0] if not serie.mode().empty else 'N/A'
            freq_moda    = int((serie == moda).sum())
            pct_moda     = round(freq_moda / len(serie) * 100, 1)
            cats_lista   = sorted(serie.unique().astype(str))
            cats_exib    = ', '.join(cats_lista[:MAX_CATS_EXIBIDAS])
            if n_cats > MAX_CATS_EXIBIDAS:
                cats_exib += f'  ... (+{n_cats - MAX_CATS_EXIBIDAS})'
            linhas_q.append({
                'Base':        nome_base,
                'Variavel':    col,
                'N_Categorias': n_cats,
                'Moda':        str(moda),
                'Freq_Moda_%': pct_moda,
                'Categorias':  cats_exib,
            })
            todas_quali.append(linhas_q[-1])

        print_tabela_bonita(
            pd.DataFrame(linhas_q)[['Variavel', 'N_Categorias', 'Moda',
                                     'Freq_Moda_%', 'Categorias']]
        )
    else:
        print("  (nenhuma variavel qualitativa detectada nesta base)")

    # --------------------------------------------------
    # TABELA DE VARIAVEIS QUANTITATIVAS
    # --------------------------------------------------
    print_separador(f'{nome_base}  —  VARIAVEIS QUANTITATIVAS ({len(quantitativas)})')

    if quantitativas:
        linhas_n = []
        for col in quantitativas:
            s = como_numero(df[col])
            linhas_n.append({
                'Base':       nome_base,
                'Variavel':   col,
                'N_obs':      int(s.count()),
                'Min.':       round(s.min(), 4),
                'Max.':       round(s.max(), 4),
                'Media':      round(s.mean(), 4),
                '1o_Quartil': round(s.quantile(0.25), 4),
                'Mediana':    round(s.median(), 4),
                '3o_Quartil': round(s.quantile(0.75), 4),
                'Desvio_Pad': round(s.std(), 4),
            })
            todas_quanti.append(linhas_n[-1])

        pd.options.display.float_format = lambda x: f'{x:,.4f}'
        print_tabela_bonita(
            pd.DataFrame(linhas_n)[['Variavel', 'N_obs', 'Min.', 'Max.',
                                     'Media', '1o_Quartil', 'Mediana',
                                     '3o_Quartil', 'Desvio_Pad']]
        )
    else:
        print("  (nenhuma variavel quantitativa detectada nesta base)")


# -------------------------------------------------------
# Salvar CSVs consolidados
# -------------------------------------------------------
pd.DataFrame(todas_quali).to_csv(
    file_quali, sep=';', index=False, encoding='utf-8-sig', decimal=','
)
pd.DataFrame(todas_quanti).to_csv(
    file_quanti, sep=';', index=False, encoding='utf-8-sig', decimal=','
)

print(f"\nArquivos salvos em:")
print(f"  {file_quali}")
print(f"  {file_quanti}")