# =============================================================
# STEP 02 - Download e cache da base UCI Online Retail II
# -------------------------------------------------------------
# Objetivo : Baixar o dataset Online Retail II do UCI Machine
#            Learning Repository, extrair o sheet 'Year 2010-2011'
#            do arquivo Excel e salvar como cache CSV para
#            auditoria e exploracao nos steps seguintes.
# Entrada  : UCI Repository (download automatico ~45 MB ZIP)
# Saida    : output/base_step_02_online_retail.csv
# Depende  : -- (requer internet; pode demorar 1-3 min)
# =============================================================
import pandas as pd
import requests
import zipfile
import io
import os

file_output = 'output/base_step_02_online_retail.csv'
os.makedirs('output', exist_ok=True)

URL_ZIP = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'
SHEET   = 'Year 2010-2011'

print("Baixando Online Retail II do UCI Repository (~45 MB)...")
print(f"URL: {URL_ZIP}\n")

r = requests.get(URL_ZIP, timeout=300)
tamanho_mb = len(r.content) / 1_048_576
print(f"Download concluido: status {r.status_code} | {tamanho_mb:.1f} MB")

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    arquivos_zip = z.namelist()
    print(f"Arquivos no ZIP: {arquivos_zip}")
    xlsx_nome = next(f for f in arquivos_zip if f.endswith('.xlsx'))
    print(f"Lendo: {xlsx_nome} | Sheet: '{SHEET}'...")
    with z.open(xlsx_nome) as f:
        df = pd.read_excel(f, sheet_name=SHEET, engine='openpyxl')

# Normalizar nome de colunas (remover espacos extras)
df.columns = [c.strip() for c in df.columns]

# Converter InvoiceDate para string para compatibilidade CSV
df['InvoiceDate'] = df['InvoiceDate'].astype(str)

total_original   = len(df)
total_colunas    = len(df.columns)
total_nulos      = df.isna().sum().sum()
clientes_unicos  = df['Customer ID'].nunique()
produtos_unicos  = df['StockCode'].nunique()
paises_unicos    = df['Country'].nunique()
data_inicio      = df['InvoiceDate'].min()
data_fim         = df['InvoiceDate'].max()

resultados_df = pd.DataFrame({
    'Metrica':  ['Registros brutos', 'Colunas', 'Nulos total',
                 'Clientes unicos', 'Produtos unicos', 'Paises',
                 'Data inicio', 'Data fim'],
    'Valor':    [f'{total_original:,}', total_colunas, f'{total_nulos:,}',
                 f'{clientes_unicos:,}', f'{produtos_unicos:,}', paises_unicos,
                 data_inicio, data_fim],
})
print(f"\n{resultados_df.to_string(index=False)}")

df.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo em: {file_output}")