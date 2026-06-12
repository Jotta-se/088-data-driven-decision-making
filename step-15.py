# =============================================================
# STEP 15 - Transporte Ótimo
# -------------------------------------------------------------
# Objetivo: Aplicar a teoria do Transporte Ótimo (Wasserstein)
#           para comparar distribuições de renda (PIB per capita)
#           entre grupos de países. Calcula a distância de
#           Wasserstein (Earth Mover's Distance) entre regiões
#           geográficas e visualiza o plano de transporte ótimo
#           e a evolução temporal das distribuições.
# Entrada  : output/base_step_01_world_bank.csv  (gerado pelo step-01)
# Saídas   : output/base_step_15_transporte_plano.csv
#            output/base_step_15_transporte_heatmap.png
#            output/base_step_15_transporte_distribuicoes.png
#            output/base_step_15_transporte_wasserstein.png
#            output/base_step15_transporte_interativo.html
#            output/base_step15_transporte_3d.html
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
from scipy.stats import wasserstein_distance
from scipy.optimize import linprog

file_output          = 'output/base_step_15_transporte_plano.csv'
file_heatmap         = 'output/base_step_15_transporte_heatmap.png'
file_distribuicoes   = 'output/base_step_15_transporte_distribuicoes.png'
file_wasserstein     = 'output/base_step_15_transporte_wasserstein.png'
file_interativo      = 'output/base_step15_transporte_interativo.html'
file_3d              = 'output/base_step15_transporte_3d.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Aquisição de dados (base auditada step-01) ----------
print("Carregando base auditada do step-01 (World Bank)...")
paises_por_regiao = {
    'América do Norte':  ['USA', 'CAN', 'MEX'],
    'América do Sul':    ['BRA', 'ARG', 'CHL', 'COL', 'PER'],
    'Europa':            ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'SWE', 'NOR', 'POL'],
    'Ásia':              ['CHN', 'JPN', 'KOR', 'IND', 'IDN', 'PHL', 'THA'],
    'África':            ['NGA', 'ZAF', 'EGY', 'ETH', 'GHA'],
}
todos_paises = [p for ps in paises_por_regiao.values() for p in ps]

df_wb = pd.read_csv('output/base_step_01_world_bank.csv',
                    sep=';', decimal=',', encoding='utf-8-sig')
# Selecionar colunas relevantes e filtrar países do mapa regional
df_wb = df_wb[df_wb['ISO'].isin(todos_paises)][['ISO', 'Pais', 'Ano', 'PIB_pc_USD']].copy()
df_wb = df_wb.rename(columns={'PIB_pc_USD': 'PIB_pc'})
# Adicionar coluna Regiao mapeando pelo dict
iso_to_regiao = {iso: reg for reg, isos in paises_por_regiao.items() for iso in isos}
df_wb['Regiao'] = df_wb['ISO'].map(iso_to_regiao)
df = df_wb.dropna(subset=['PIB_pc']).sort_values(['ISO', 'Ano']).reset_index(drop=True)
print(f"Dados carregados: {len(df)} observações, {df['ISO'].nunique()} países, {df['Ano'].nunique()} anos")

# ---------- 2. Distâncias de Wasserstein entre regiões ----------
anos_analise = [2000, 2005, 2010, 2015, 2020, 2022]
regioes      = list(paises_por_regiao.keys())
n_reg        = len(regioes)

# Calcular Wasserstein entre pares de regiões para cada ano
registros_wasserstein = []
for ano in anos_analise:
    df_ano = df[df['Ano'] == ano].dropna(subset=['PIB_pc'])
    matriz_W = np.zeros((n_reg, n_reg))
    for i, r1 in enumerate(regioes):
        for j, r2 in enumerate(regioes):
            if i != j:
                u = df_ano[df_ano['Regiao'] == r1]['PIB_pc'].values
                v = df_ano[df_ano['Regiao'] == r2]['PIB_pc'].values
                if len(u) > 0 and len(v) > 0:
                    w_dist = wasserstein_distance(u, v)
                    matriz_W[i, j] = w_dist
            registros_wasserstein.append({
                'Ano':     ano,
                'Regiao_A': r1,
                'Regiao_B': r2,
                'Wasserstein': round(matriz_W[i, j], 2),
            })

df_wasser = pd.DataFrame(registros_wasserstein)

# Plano de transporte ótimo entre duas regiões (2022)
df_2022   = df[df['Ano'] == 2022].dropna(subset=['PIB_pc'])
reg_A, reg_B = 'Europa', 'África'
df_2022_A = df_2022[df_2022['Regiao'] == reg_A].sort_values('PIB_pc')
df_2022_B = df_2022[df_2022['Regiao'] == reg_B].sort_values('PIB_pc')
u_vals = df_2022_A['PIB_pc'].values
v_vals = df_2022_B['PIB_pc'].values
n_u, n_v = len(u_vals), len(v_vals)

# Plano de transporte: min cᵀγ  s.t. γ_ij ≥ 0, rowsum = 1/n_u, colsum = 1/n_v
c_cost = np.abs(u_vals[:, None] - v_vals[None, :]).flatten()
A_eq_rows = np.zeros((n_u, n_u * n_v))
for i in range(n_u):
    A_eq_rows[i, i * n_v:(i + 1) * n_v] = 1
A_eq_cols = np.zeros((n_v, n_u * n_v))
for j in range(n_v):
    A_eq_cols[j, j::n_v] = 1
A_eq = np.vstack([A_eq_rows, A_eq_cols])
b_eq = np.concatenate([np.ones(n_u) / n_u, np.ones(n_v) / n_v])
res_ot = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n_u * n_v), method='highs')
gamma  = res_ot.x.reshape(n_u, n_v) if res_ot.success else np.zeros((n_u, n_v))

wdist_2022_AB = wasserstein_distance(u_vals, v_vals)
print(f"\nWasserstein {reg_A} × {reg_B} (2022): {wdist_2022_AB:.0f} USD")

# Evolução temporal de Wasserstein entre Europa e África
ev_anos  = sorted(df['Ano'].unique())
ev_wdist = []
for ano in ev_anos:
    df_a = df[df['Ano'] == ano].dropna(subset=['PIB_pc'])
    u = df_a[df_a['Regiao'] == reg_A]['PIB_pc'].values
    v = df_a[df_a['Regiao'] == reg_B]['PIB_pc'].values
    ev_wdist.append(wasserstein_distance(u, v) if len(u) > 0 and len(v) > 0 else np.nan)

# ---------- 3. Salvar resultados ----------
df_plano = pd.DataFrame(gamma, index=[f'EU_{p}' for p in df_2022_A['Pais'].values],
                         columns=[f'AF_{p}' for p in df_2022_B['Pais'].values])
df_plano.to_csv(file_output, sep=';', encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Matriz de Wasserstein entre regiões (2022)
df_2022_w = df_wasser[df_wasser['Ano'] == 2022].pivot(
    index='Regiao_A', columns='Regiao_B', values='Wasserstein').reindex(
    index=regioes, columns=regioes).fillna(0)
plt.figure(figsize=(9, 7))
sns.heatmap(df_2022_w, annot=True, fmt='.0f', cmap='YlOrRd',
            cbar_kws={'label': 'Distância de Wasserstein (USD PIB pc)'})
plt.title('Matriz de Distâncias de Wasserstein entre Regiões (2022)', fontsize=12)
plt.tight_layout()
plt.savefig(file_heatmap, dpi=150)
plt.show()
plt.close()

# 4.2 Distribuições de PIB per capita por região (2022)
# SKILL: >3 grupos em KDE = spaghetti proibido → violin plot (5 regiões)
cores_reg = ['#1565C0', '#C62828', '#2E7D32', '#FF9800', '#7B1FA2']
df_violin = pd.DataFrame({
    'Regiao':   [reg for reg in regioes
                 for _ in df_2022[df_2022['Regiao'] == reg]['PIB_pc'].dropna()],
    'log_PIB':  [np.log1p(v) for reg in regioes
                 for v in df_2022[df_2022['Regiao'] == reg]['PIB_pc'].dropna().values],
})
palette_reg = dict(zip(regioes, cores_reg))
fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(data=df_violin, x='Regiao', y='log_PIB',
               palette=palette_reg, ax=ax, inner='box', linewidth=1.2)
ax.set_xlabel('Região', fontsize=11)
ax.set_ylabel('log(1 + PIB per capita USD) — 2022')
ax.set_title('Distribuições de PIB per capita por Região (escala log)', fontsize=12)
ax.tick_params(axis='x', rotation=15)
ax.grid(True, axis='y', alpha=0.35)
plt.tight_layout()
plt.savefig(file_distribuicoes, dpi=150)
plt.show()
plt.close()

# 4.3 Evolução temporal da distância Wasserstein Europa × África
plt.figure(figsize=(10, 5))
plt.plot(ev_anos, [w / 1000 for w in ev_wdist], color='#C62828', linewidth=2.5, marker='o', markersize=5)
plt.fill_between(ev_anos, [w / 1000 for w in ev_wdist], alpha=0.15, color='#C62828')
plt.xlabel('Ano', fontsize=11)
plt.ylabel('Distância de Wasserstein (× 1.000 USD)')
plt.title(f'Evolução da Distância de Wasserstein: {reg_A} × {reg_B}', fontsize=12)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_wasserstein, dpi=150)
plt.show()
plt.close()

# 4.4 Heatmap interativo do plano de transporte ótimo
fig_inter = go.Figure(data=go.Heatmap(
    z=gamma * 1000,
    x=[f'AF_{p}' for p in df_2022_B['Pais'].values],
    y=[f'EU_{p}' for p in df_2022_A['Pais'].values],
    colorscale='Blues',
    texttemplate='%{z:.2f}',
))
fig_inter.update_layout(
    title=f'Plano de Transporte Ótimo: {reg_A} → {reg_B} (2022, ×10⁻³)',
    xaxis_title='Destino (África)', yaxis_title='Origem (Europa)',
    height=400
)
fig_inter.write_html(file_interativo)
fig_inter.show()
print(f"Plano de transporte interativo salvo: {file_interativo}")

# 4.5 3D: Wasserstein por par de regiões ao longo do tempo
df_w_pivot = df_wasser[df_wasser['Regiao_A'] != df_wasser['Regiao_B']].copy()
df_w_pivot['Par'] = df_w_pivot['Regiao_A'].str[:3] + '×' + df_w_pivot['Regiao_B'].str[:3]
fig_3d = px.scatter_3d(df_w_pivot, x='Ano', y='Par', z='Wasserstein',
                        color='Wasserstein', color_continuous_scale='YlOrRd',
                        opacity=0.8, size_max=8,
                        title='Distâncias de Wasserstein por Par de Regiões e Ano',
                        labels={'Wasserstein': 'W (USD)', 'Par': 'Par de Regiões'})
fig_3d.write_html(file_3d)
fig_3d.show()
print(f"Gráfico 3D salvo: {file_3d}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_heatmap}")
print(f"  {file_distribuicoes}")
print(f"  {file_wasserstein}")
print(f"  {file_interativo}")
print(f"  {file_3d}")