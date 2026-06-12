# =============================================================
# STEP 09 - Análise Envoltória de Dados (DEA)
# -------------------------------------------------------------
# Objetivo: Medir a eficiência relativa de países (DMUs) usando
#           DEA com modelo CRS (Constant Returns to Scale).
#           Inputs: capital (formação bruta de capital / PIB) e
#           trabalho (força de trabalho). Output: PIB per capita.
#           Identifica fronteira de eficiência e benchmarks.
# Entrada  : output/base_step_01_world_bank.csv  (gerado pelo step-01)
# Saídas   : output/base_step_09_dea_eficiencia.csv
#            output/base_step_09_dea_scores.png
#            output/base_step_09_dea_inputs_outputs.png
#            output/base_step_09_dea_benchmarks.png
#            output/base_step09_dea_interativo.html
#            output/base_step09_dea_3d.html
# Depende  : STEP 01
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.optimize import linprog

file_output          = 'output/base_step_09_dea_eficiencia.csv'
file_scores          = 'output/base_step_09_dea_scores.png'
file_inputs_outputs  = 'output/base_step_09_dea_inputs_outputs.png'
file_benchmarks      = 'output/base_step_09_dea_benchmarks.png'
file_interativo      = 'output/base_step09_dea_interativo.html'
file_3d              = 'output/base_step09_dea_3d.html'
file_cache           = 'output/base_step_09_world_bank_raw.csv'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Aquisição de dados (base auditada step-01) ----------
# Países do DEA (subconjunto dos 33 países do step-01)
paises_dea = [
    'USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'BRA', 'IND', 'CAN', 'KOR',
    'AUS', 'ESP', 'MEX', 'IDN', 'NLD', 'CHE', 'SAU', 'TUR', 'ARG', 'ZAF',
    'NGA', 'EGY', 'POL', 'SWE', 'NOR', 'PRT', 'COL', 'CHL', 'PER', 'PHL',
]
print("Carregando base auditada do step-01 (World Bank)...")
df_wb_raw = pd.read_csv('output/base_step_01_world_bank.csv',
                        sep=';', decimal=',', encoding='utf-8-sig')
# Renomear coluna para compatibilidade com o modelo DEA deste step
df_wb_raw = df_wb_raw.rename(columns={'Capital_pct_PIB': 'Formacao_Capital_pct_PIB'})
# Filtrar países DEA e ano 2021
df_wb = (df_wb_raw[df_wb_raw['ISO'].isin(paises_dea) & (df_wb_raw['Ano'] == 2021)]
         .dropna(subset=['PIB_pc_USD', 'Formacao_Capital_pct_PIB', 'Forca_Trabalho'])
         .reset_index(drop=True))
print(f"DMUs disponíveis: {len(df_wb)} países (ano 2021)")

# Salvar cache local para dependência dos steps 13 e 14
os.makedirs('output', exist_ok=True)
df_wb.to_csv(file_cache,
             sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"Cache salvo: {file_cache}")

df = df_wb.copy()
# Ajuste de nomes para o restante do código
df = df.rename(columns={'PIB_pc_USD': 'PIB_per_capita_USD'})
for col in ['PIB_per_capita_USD', 'Formacao_Capital_pct_PIB', 'Forca_Trabalho']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

# ---------- 2. DEA — Modelo CRS (CCR) via Programação Linear ----------
# Para cada DMU k: max θ_k s.t.
#   Σ λ_j * y_j >= θ_k * y_k     (output constraint)
#   Σ λ_j * x_ij <= x_ik         (input constraints)
#   λ_j >= 0                      (intensidade)
# Reformulado como min -θ:
#   c = [-1, 0, 0, ..., 0]  (apenas θ na obj)
#   variáveis: [θ, λ_1, ..., λ_n]

# Normalização Min-Max para DEA
def minmax(s): return (s - s.min()) / (s.max() - s.min() + 1e-12) + 0.01

Y = minmax(df['PIB_per_capita_USD'].values)          # output (1D)
X1 = minmax(df['Formacao_Capital_pct_PIB'].values)   # input 1
X2 = minmax(df['Forca_Trabalho'].values)             # input 2
X  = np.vstack([X1, X2])                             # 2 x n_dmu

n_dmu = len(df)
eficiencias = np.zeros(n_dmu)

for k in range(n_dmu):
    # min c @ [θ, λ_1..λ_n]  =>  c = [-1, 0..0]
    c_obj = np.zeros(1 + n_dmu)
    c_obj[0] = -1.0   # maximizar θ

    # Restrição output: -Y @ λ + θ * Y[k] <= 0  => -Y[j]*λ[j] + θ*Y[k] <= 0
    # Ub format: A_ub @ x <= b_ub
    A_out = np.zeros((1, 1 + n_dmu))
    A_out[0, 0] = Y[k]          # coef de θ
    A_out[0, 1:] = -Y            # coef de λ
    b_out = np.array([0.0])

    # Restrição inputs: X[m,:] @ λ <= X[m,k]  para m=0,1
    A_inp = np.zeros((2, 1 + n_dmu))
    b_inp = np.zeros(2)
    for m in range(2):
        A_inp[m, 1:] = X[m, :]
        b_inp[m]     = X[m, k]

    A_ub = np.vstack([A_out, A_inp])
    b_ub = np.hstack([b_out, b_inp])

    bounds_lp = [(0, None)] * (1 + n_dmu)   # θ >= 0, λ >= 0

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    eficiencias[k] = res.x[0] if res.success else np.nan

df['Eficiencia_DEA'] = np.round(eficiencias, 6)
df['Rank']           = df['Eficiencia_DEA'].rank(ascending=False).astype(int)
df_sorted            = df.sort_values('Eficiencia_DEA', ascending=False).reset_index(drop=True)

print(f"\nTop 5 mais eficientes:")
print(df_sorted[['Pais', 'Eficiencia_DEA']].head())
print(f"\nFronteira eficiente (θ=1.0): {(df['Eficiencia_DEA'] >= 0.999).sum()} DMUs")

# ---------- 3. Salvar resultados ----------
df_sorted.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Scores de eficiência DEA por país
top_n = min(25, len(df_sorted))
df_plot = df_sorted.head(top_n)
cores_bar = ['#C62828' if e >= 0.999 else '#1565C0' for e in df_plot['Eficiencia_DEA']]
plt.figure(figsize=(12, 6))
plt.barh(df_plot['Pais'][::-1], df_plot['Eficiencia_DEA'][::-1], color=cores_bar[::-1])
plt.axvline(1.0, color='gold', linewidth=1.5, linestyle='--', label='Fronteira (θ=1)')
plt.xlabel('Score de Eficiência DEA (CRS)', fontsize=11)
plt.title(f'Scores de Eficiência DEA — Top {top_n} países (2021)', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, axis='x', alpha=0.35)
plt.tight_layout()
plt.savefig(file_scores, dpi=150)
plt.show()
plt.close()

# 4.2 Inputs × Output por país
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(df['Formacao_Capital_pct_PIB'], df['PIB_per_capita_USD'],
                     c=df['Eficiencia_DEA'], cmap='coolwarm', s=60, edgecolors='grey', linewidths=0.4)
plt.colorbar(sc, ax=axes[0], label='Eficiência DEA')
axes[0].set_xlabel('Formação de Capital (% PIB)')
axes[0].set_ylabel('PIB per capita (USD)')
axes[0].set_title('Capital vs PIB per capita')
axes[0].grid(True, alpha=0.3)

sc2 = axes[1].scatter(np.log1p(df['Forca_Trabalho']), df['PIB_per_capita_USD'],
                      c=df['Eficiencia_DEA'], cmap='coolwarm', s=60, edgecolors='grey', linewidths=0.4)
plt.colorbar(sc2, ax=axes[1], label='Eficiência DEA')
axes[1].set_xlabel('log(Força de Trabalho)')
axes[1].set_ylabel('PIB per capita (USD)')
axes[1].set_title('Trabalho (log) vs PIB per capita')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Inputs × Output — Mapa de Eficiência DEA', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(file_inputs_outputs, dpi=150)
plt.show()
plt.close()

# 4.3 Distribuição dos scores de eficiência
plt.figure(figsize=(9, 5))
sns.histplot(df['Eficiencia_DEA'], bins=20, kde=True, color='#1565C0')
plt.axvline(df['Eficiencia_DEA'].mean(), color='red', linestyle='--',
            label=f'Média: {df["Eficiencia_DEA"].mean():.3f}')
plt.axvline(1.0, color='gold', linestyle='--', linewidth=1.5, label='Fronteira (θ=1)')
plt.xlabel('Score de Eficiência DEA')
plt.ylabel('Frequência')
plt.title('Distribuição dos Scores de Eficiência DEA (CRS)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_benchmarks, dpi=150)
plt.show()
plt.close()

# 4.4 Gráfico interativo: eficiência por país (Plotly)
fig_inter = px.bar(df_sorted.head(top_n), x='Eficiencia_DEA', y='Pais', orientation='h',
                   color='Eficiencia_DEA', color_continuous_scale='RdBu',
                   title=f'Scores de Eficiência DEA — Top {top_n} Países',
                   labels={'Eficiencia_DEA': 'Eficiência DEA', 'Pais': 'País'})
fig_inter.add_vline(x=1.0, line_dash='dash', line_color='gold', annotation_text='Fronteira')
fig_inter.write_html(file_interativo)
fig_inter.show()
print(f"Gráfico interativo salvo: {file_interativo}")

# 4.5 Gráfico 3D: Capital × Trabalho × PIB, colorido por eficiência
fig_3d = px.scatter_3d(
    df, x='Formacao_Capital_pct_PIB', y=np.log1p(df['Forca_Trabalho']),
    z='PIB_per_capita_USD', color='Eficiencia_DEA',
    color_continuous_scale='RdBu', hover_name='Pais',
    title='DEA 3D: Capital × Trabalho × PIB per capita',
    labels={'x': 'Capital (% PIB)', 'y': 'log(Trabalho)', 'z': 'PIB per capita'},
    opacity=0.85
)
fig_3d.write_html(file_3d)
fig_3d.show()
print(f"Gráfico 3D salvo: {file_3d}")

print("\nArquivos salvos:")
print(f"  {file_cache}")
print(f"  {file_output}")
print(f"  {file_scores}")
print(f"  {file_inputs_outputs}")
print(f"  {file_benchmarks}")
print(f"  {file_interativo}")
print(f"  {file_3d}")