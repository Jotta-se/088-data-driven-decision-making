# =============================================================
# STEP 08 - Otimização de Pareto (Fronteira Eficiente)
# -------------------------------------------------------------
# Objetivo: Construir a fronteira eficiente de Pareto para um
#           portfólio de ações via simulação de Monte Carlo.
#           Identifica carteiras ótimas no sentido de Pareto
#           (maior retorno dado o risco, ou menor risco dado
#           o retorno). Compara portfólio Sharpe máximo com
#           portfólio de mínima variância.
# Entrada  : output/base_step_00_yahoo_finance.csv  (gerado pelo step-00)
# Saídas   : output/base_step_08_pareto_portfolios.csv
#            output/base_step_08_pareto_fronteira.png
#            output/base_step_08_pareto_pesos_otimos.png
#            output/base_step_08_pareto_retornos_hist.png
#            output/base_step08_pareto_fronteira_interativa.html
#            output/base_step08_pareto_3d.html
# Depende  : STEP 00
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.optimize import minimize

file_output          = 'output/base_step_08_pareto_portfolios.csv'
file_fronteira       = 'output/base_step_08_pareto_fronteira.png'
file_pesos           = 'output/base_step_08_pareto_pesos_otimos.png'
file_retornos_hist   = 'output/base_step_08_pareto_retornos_hist.png'
file_fronteira_inter = 'output/base_step08_pareto_fronteira_interativa.html'
file_3d              = 'output/base_step08_pareto_3d.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Aquisição de dados (base auditada step-00) ----------
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'BRK-B']
print("Carregando base auditada do step-00 (Yahoo Finance)...")
df_cache = pd.read_csv('output/base_step_00_yahoo_finance.csv',
                       sep=';', decimal=',', encoding='utf-8-sig')
df_cache['Data'] = pd.to_datetime(df_cache['Data'])
mask = (df_cache['Ticker'].isin(tickers) &
        (df_cache['Data'] >= '2019-01-01') &
        (df_cache['Data'] <= '2023-12-31'))
dados = df_cache[mask].pivot(index='Data', columns='Ticker', values='Close')
dados = dados.dropna(axis=1, how='all')
dados.columns = [c.replace('-', '_') for c in dados.columns]
tickers_validos = list(dados.columns)
print(f"Dados carregados: {dados.shape[0]} observações, {len(tickers_validos)} ativos válidos")

# ---------- 2. Otimização de Pareto via Monte Carlo ----------
retornos    = dados.pct_change().dropna()
ret_medios  = retornos.mean() * 252
cov_matrix  = retornos.cov() * 252
n_ativos    = len(tickers_validos)
N_SIM       = 8000
np.random.seed(42)

port_ret  = np.zeros(N_SIM)
port_vol  = np.zeros(N_SIM)
port_shrp = np.zeros(N_SIM)
port_w    = np.zeros((N_SIM, n_ativos))

for i in range(N_SIM):
    w = np.random.dirichlet(np.ones(n_ativos))
    r = np.dot(w, ret_medios)
    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    port_ret[i]  = r
    port_vol[i]  = v
    port_shrp[i] = r / v
    port_w[i]    = w

# Portfólio Sharpe Máximo
idx_sharpe = np.argmax(port_shrp)
w_sharpe   = port_w[idx_sharpe]

# Portfólio de Mínima Variância (via scipy)
def vol_portfolio(w):
    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds      = tuple((0, 1) for _ in range(n_ativos))
res_mv      = minimize(vol_portfolio, x0=np.ones(n_ativos) / n_ativos,
                       method='SLSQP', bounds=bounds, constraints=constraints)
w_minvar    = res_mv.x
ret_minvar  = np.dot(w_minvar, ret_medios)
vol_minvar  = vol_portfolio(w_minvar)

# Identificar fronteira de Pareto (eficiente):
# Um portfólio p é Pareto-dominado se existe q com ret_q>=ret_p e vol_q<=vol_p (e ao menos uma desigualdade estrita)
is_pareto   = np.ones(N_SIM, dtype=bool)
for i in range(N_SIM):
    dominated = ((port_ret >= port_ret[i]) & (port_vol <= port_vol[i]) &
                 ((port_ret > port_ret[i]) | (port_vol < port_vol[i])))
    if dominated.any():
        is_pareto[i] = False

n_pareto = is_pareto.sum()
print(f"\nPortfólios simulados : {N_SIM}")
print(f"Portfólios Pareto    : {n_pareto}")
print(f"Sharpe máx           : {port_shrp[idx_sharpe]:.4f}  "
      f"(ret={port_ret[idx_sharpe]:.4f}, vol={port_vol[idx_sharpe]:.4f})")
print(f"Mínima variância     : ret={ret_minvar:.4f}, vol={vol_minvar:.4f}")

# ---------- 3. Salvar resultados ----------
df_portfolios = pd.DataFrame({
    'Retorno_Anual':  np.round(port_ret, 6),
    'Volatilidade':   np.round(port_vol, 6),
    'Sharpe':         np.round(port_shrp, 6),
    'Pareto_Eficiente': is_pareto.astype(int),
})
df_pesos = pd.DataFrame(port_w, columns=[f'Peso_{t}' for t in tickers_validos])
df_resultado = pd.concat([df_portfolios, df_pesos], axis=1)
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Fronteira Eficiente de Pareto
# 4.1 Fronteira Eficiente de Pareto
plt.figure(figsize=(10, 6))
plt.scatter(port_vol[~is_pareto] * 100, port_ret[~is_pareto] * 100,
            c=port_shrp[~is_pareto], cmap='Blues', alpha=0.3, s=8, label='Portfólios simulados')

sc = plt.scatter(port_vol[is_pareto] * 100, port_ret[is_pareto] * 100,  # ← capturar retorno
            c=port_shrp[is_pareto], cmap='YlOrRd', alpha=0.8, s=18, label='Fronteira de Pareto')

plt.scatter(port_vol[idx_sharpe] * 100, port_ret[idx_sharpe] * 100,
            marker='*', s=400, color='gold', edgecolors='black', linewidths=1, zorder=10,
            label=f'Sharpe Máx ({port_shrp[idx_sharpe]:.2f})')
plt.scatter(vol_minvar * 100, ret_minvar * 100,
            marker='D', s=200, color='cyan', edgecolors='navy', linewidths=1.5, zorder=10,
            label='Mínima Variância')

plt.colorbar(sc, label='Índice de Sharpe')  # ← usar sc em vez de ScalarMappable manual
plt.xlabel('Volatilidade Anual (%)', fontsize=11)
plt.ylabel('Retorno Anual (%)', fontsize=11)
plt.title('Fronteira Eficiente de Pareto — Otimização de Portfólio', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(file_fronteira, dpi=150)
plt.show()
plt.close()

# 4.2 Pesos dos portfólios ótimos
# Substituir pie chart por barplot horizontal ordenado (SKILL: pie EVITAR com >5 categorias;
# barplot horizontal é superior para comparar proporções exatas entre muitos grupos)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cores_bar = sns.color_palette('tab10', n_ativos)

# Portfólio Sharpe Máximo
df_sh = pd.DataFrame({'Ativo': tickers_validos, 'Peso': w_sharpe * 100})
df_sh = df_sh[df_sh['Peso'] > 0.5].sort_values('Peso')   # exibir apenas pesos relevantes, ordenados
axes[0].barh(df_sh['Ativo'], df_sh['Peso'],
             color=[cores_bar[tickers_validos.index(t)] for t in df_sh['Ativo']])
axes[0].set_xlabel('Peso no Portfólio (%)')
axes[0].set_title('Portfólio — Sharpe Máximo', fontsize=11)
axes[0].grid(True, axis='x', alpha=0.4)
for i, (_, row) in enumerate(df_sh.iterrows()):
    axes[0].text(row['Peso'] + 0.3, i, f"{row['Peso']:.1f}%", va='center', fontsize=9)

# Portfólio Mínima Variância
df_mv = pd.DataFrame({'Ativo': tickers_validos, 'Peso': w_minvar * 100})
df_mv = df_mv[df_mv['Peso'] > 0.5].sort_values('Peso')
axes[1].barh(df_mv['Ativo'], df_mv['Peso'],
             color=[cores_bar[tickers_validos.index(t)] for t in df_mv['Ativo']])
axes[1].set_xlabel('Peso no Portfólio (%)')
axes[1].set_title('Portfólio — Mínima Variância', fontsize=11)
axes[1].grid(True, axis='x', alpha=0.4)
for i, (_, row) in enumerate(df_mv.iterrows()):
    axes[1].text(row['Peso'] + 0.3, i, f"{row['Peso']:.1f}%", va='center', fontsize=9)

plt.suptitle('Composição dos Portfólios Ótimos de Pareto', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(file_pesos, dpi=150)
plt.show()
plt.close()

# 4.3 Distribuição de retornos dos portfólios
plt.figure(figsize=(10, 5))
sns.histplot(port_ret * 100, bins=60, kde=True, color='#1565C0', label='Todos portfólios')
sns.histplot(port_ret[is_pareto] * 100, bins=40, kde=True, color='#C62828', label='Pareto Eficientes')
plt.axvline(port_ret[idx_sharpe] * 100, color='gold', linewidth=2, linestyle='--', label='Sharpe Máx')
plt.axvline(ret_minvar * 100, color='cyan', linewidth=2, linestyle='--', label='Mín Variância')
plt.xlabel('Retorno Anual (%)')
plt.ylabel('Frequência')
plt.title('Distribuição de Retornos — Portfólios Simulados vs Pareto', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_retornos_hist, dpi=150)
plt.show()
plt.close()

# 4.4 Fronteira interativa (Plotly)
tipo = np.where(is_pareto, 'Pareto Eficiente', 'Dominado')
fig_inter = px.scatter(
    x=port_vol * 100, y=port_ret * 100, color=tipo,
    color_discrete_map={'Pareto Eficiente': '#E53935', 'Dominado': '#90CAF9'},
    opacity=0.5, size_max=6,
    title='Fronteira Eficiente de Pareto — Interativo',
    labels={'x': 'Volatilidade (%)', 'y': 'Retorno Anual (%)'},
)
fig_inter.add_scatter(x=[port_vol[idx_sharpe] * 100], y=[port_ret[idx_sharpe] * 100],
                      mode='markers', marker=dict(symbol='star', size=18, color='gold',
                                                   line=dict(color='black', width=1)),
                      name='Sharpe Máximo')
fig_inter.add_scatter(x=[vol_minvar * 100], y=[ret_minvar * 100],
                      mode='markers', marker=dict(symbol='diamond', size=14, color='cyan',
                                                   line=dict(color='navy', width=1.5)),
                      name='Mínima Variância')
fig_inter.write_html(file_fronteira_inter)
fig_inter.show()
print(f"Fronteira interativa salva: {file_fronteira_inter}")

# 4.5 Superfície 3D: Retorno × Volatilidade × Sharpe
fig_3d = px.scatter_3d(
    x=port_vol * 100, y=port_ret * 100, z=port_shrp,
    color=port_shrp, color_continuous_scale='Viridis',
    opacity=0.4, size_max=4,
    title='Espaço Risco-Retorno-Sharpe (3D)',
    labels={'x': 'Volatilidade (%)', 'y': 'Retorno (%)', 'z': 'Sharpe'},
)
fig_3d.write_html(file_3d)
fig_3d.show()
print(f"Gráfico 3D salvo: {file_3d}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_fronteira}")
print(f"  {file_pesos}")
print(f"  {file_retornos_hist}")
print(f"  {file_fronteira_inter}")
print(f"  {file_3d}")