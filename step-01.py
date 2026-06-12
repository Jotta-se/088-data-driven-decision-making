# =============================================================
# STEP 01 - Download e cache da base World Bank API
# -------------------------------------------------------------
# Objetivo : Baixar tres indicadores macroeconomicos do Banco
#            Mundial para o conjunto de paises usados nos
#            steps 09 (DEA) e 15 (Wasserstein) e salvar como
#            cache CSV para auditoria e exploracao.
# Entrada  : World Bank API (download automatico via requests)
# Saida    : output/base_step_01_world_bank.csv
# Depende  : -- (requer internet)
# =============================================================
import pandas as pd
import requests
import os

file_output = 'output/base_step_01_world_bank.csv'
os.makedirs('output', exist_ok=True)

# Uniao dos paises usados nos steps 09 e 15
# Step 09 (DEA)        : 30 paises, 2021
# Step 15 (Wasserstein): 29 paises, 2000-2022
PAISES = [
    'USA', 'CAN', 'MEX',                              # America do Norte (3)
    'BRA', 'ARG', 'CHL', 'COL', 'PER',               # America do Sul (5)
    'DEU', 'FRA', 'GBR', 'ITA', 'ESP',
    'NLD', 'SWE', 'NOR', 'POL', 'CHE', 'BEL',        # Europa (11)
    'CHN', 'JPN', 'KOR', 'IND', 'IDN',
    'PHL', 'THA', 'SGP', 'MYS',                       # Asia (9)
    'NGA', 'ZAF', 'EGY', 'ETH', 'GHA',               # Africa (5)
]
INDICADORES = {
    'NY.GDP.PCAP.CD': 'PIB_pc_USD',
    'NE.GFI.TOTL.ZS': 'Capital_pct_PIB',
    'SL.TLF.TOTL.IN': 'Forca_Trabalho',
}
ANO_INICIO = 2000
ANO_FIM    = 2022

paises_str = ';'.join(PAISES)
frames = []

for codigo, nome in INDICADORES.items():
    print(f"Baixando: {codigo} ({nome})...")
    url = (
        f'https://api.worldbank.org/v2/country/{paises_str}'
        f'/indicator/{codigo}?format=json'
        f'&date={ANO_INICIO}:{ANO_FIM}&per_page=3000'
    )
    r = requests.get(url, timeout=90)
    if r.status_code != 200:
        print(f"  ERRO: status HTTP {r.status_code}")
        continue
    dados_raw = r.json()
    if len(dados_raw) < 2:
        print("  ERRO: resposta vazia da API")
        continue
    n_entradas = 0
    for entry in dados_raw[1]:
        if entry.get('value') is not None:
            frames.append({
                'ISO':       entry['countryiso3code'],
                'Pais':      entry['country']['value'],
                'Ano':       int(entry['date']),
                'Indicador': nome,
                'Valor':     float(entry['value']),
            })
            n_entradas += 1
    print(f"  OK — {n_entradas:,} observacoes")

df_long = pd.DataFrame(frames)

# Pivotar para formato wide: ISO | Pais | Ano | PIB_pc_USD | Capital_pct_PIB | Forca_Trabalho
df = df_long.pivot_table(
    index=['ISO', 'Pais', 'Ano'],
    columns='Indicador',
    values='Valor'
).reset_index()
df.columns.name = None

total_obs    = len(df)
total_paises = df['ISO'].nunique()
total_anos   = df['Ano'].nunique()
total_nulos  = df.isna().sum().sum()

resultados_df = pd.DataFrame({
    'Metrica':  ['Observacoes (pais x ano)', 'Paises unicos', 'Anos cobertos',
                 'Ano inicio', 'Ano fim', 'Nulos total'],
    'Valor':    [f'{total_obs:,}', total_paises,
                 total_anos, df['Ano'].min(), df['Ano'].max(),
                 f'{total_nulos:,}'],
})
print(f"\n{resultados_df.to_string(index=False)}")

df.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo em: {file_output}")