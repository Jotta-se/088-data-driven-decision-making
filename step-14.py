# =============================================================
# STEP 14 - Otimização Estocástica
# -------------------------------------------------------------
# Objetivo: Resolver um problema de programação estocástica em
#           dois estágios (alocação de ativos) com geração de
#           cenários via simulação de Monte Carlo. Compara:
#           EEV (Expected Value of Expected Value), RP (Recourse
#           Problem) e WS (Wait-and-See). Calcula VSS (Value of
#           Stochastic Solution) e EVPI (Expected Value of
#           Perfect Information).
# Entrada  : output/base_step_09_world_bank_raw.csv  (gerado pelo step-09)
#            output/base_step_00_yahoo_finance.csv   (gerado pelo step-00)
# Saídas   : output/base_step_14_estocastica_cenarios.csv
#            output/base_step_14_estocastica_distribuicao.png
#            output/base_step_14_estocastica_ev_vs_ws.png
#            output/base_step_14_estocastica_solucoes.png
#            output/base_step14_estocastica_cenarios_interativo.html
#            output/base_step14_estocastica_3d.html
# Depende  : STEP 00, STEP 09
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.optimize import minimize, linprog

file_input              = 'output/base_step_09_world_bank_raw.csv'
file_output             = 'output/base_step_14_estocastica_cenarios.csv'
file_distribuicao       = 'output/base_step_14_estocastica_distribuicao.png'
file_ev_vs_ws           = 'output/base_step_14_estocastica_ev_vs_ws.png'
file_solucoes           = 'output/base_step_14_estocastica_solucoes.png'
file_cenarios_inter     = 'output/base_step14_estocastica_cenarios_interativo.html'
file_3d                 = 'output/base_step14_estocastica_3d.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Carregar dados e preparar ativos ----------
print(f"Carregando cache World Bank: {file_input}")
df_wb = pd.read_csv(file_input, sep=';', encoding='utf-8-sig')
print(f"Dados WB carregados: {len(df_wb)} países")

tickers = ['SPY', 'EEM', 'TLT', 'GLD', 'EWJ']
nomes   = ['EUA (SPY)', 'Emerg. (EEM)', 'T-Bond (TLT)', 'Ouro (GLD)', 'Japão (EWJ)']
print("\nCarregando dados de retornos da base auditada step-00...")
df_cache = pd.read_csv('output/base_step_00_yahoo_finance.csv',
                       sep=';', decimal=',', encoding='utf-8-sig')
df_cache['Data'] = pd.to_datetime(df_cache['Data'])
mask = (df_cache['Ticker'].isin(tickers) &
        (df_cache['Data'] >= '2015-01-01') &
        (df_cache['Data'] <= '2023-12-31'))
dados = df_cache[mask].pivot(index='Data', columns='Ticker', values='Close')
dados = dados.dropna(axis=1, how='all')
tickers_val = list(dados.columns)
nomes_val   = [nomes[tickers.index(t)] for t in tickers_val if t in tickers]
n           = len(tickers_val)

ret_diarios = dados.pct_change().dropna()
mu          = ret_diarios.mean().values * 252
Sigma       = ret_diarios.cov().values  * 252
print(f"Ativos válidos: {tickers_val}")

# ---------- 2. Programação Estocástica em Dois Estágios ----------
# Estágio 1: Decidir alocação inicial w (antes de observar o cenário)
# Estágio 2: Rebalancear y (após observar o cenário) com custo de rebalanceamento
# Objetivo: max  E_ξ [ wᵀr̃ - c * ||y||₁ ]
# s.t.       1ᵀw = 1, w ≥ 0, 1ᵀ(w+y) = 1, (w+y) ≥ 0

N_CEN       = 500
CUSTO_REBALAN = 0.002   # 0.2% de custo de transação
np.random.seed(42)

# Gerar cenários por simulação Monte Carlo multivariada
L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))
cenarios = (mu + (L @ np.random.randn(n, N_CEN)).T)  # N_CEN × n

# --- EEV: Solução baseada no valor esperado dos retornos ---
def neg_retorno(w):
    return -np.dot(w, mu)

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds      = tuple((0, 1) for _ in range(n))
res_eev     = minimize(neg_retorno, np.ones(n)/n, method='SLSQP',
                       bounds=bounds, constraints=constraints)
w_eev       = res_eev.x
ret_eev_cen = cenarios @ w_eev   # retorno do portfólio EEV em cada cenário

# --- RP (Recourse Problem): solução estocástica ---
# Simplificação: busca o w que maximiza retorno médio menos custo esperado de rebalanceamento
# Aqui, usamos uma heurística: max Sharpe com penalidade de concentração
def obj_rp(w):
    ret  = np.dot(w, mu)
    risco = np.sqrt(np.dot(w, np.dot(Sigma, w)))
    entropia = -np.sum(w * np.log(w + 1e-12))   # penalidade de concentração
    return -(ret / (risco + 1e-9) + 0.1 * entropia)

res_rp  = minimize(obj_rp, np.ones(n)/n, method='SLSQP',
                   bounds=bounds, constraints=constraints)
w_rp    = res_rp.x
ret_rp_cen = cenarios @ w_rp

# --- WS (Wait-and-See): solução ótima para cada cenário individualmente ---
ret_ws_cen = np.zeros(N_CEN)
w_ws_all   = np.zeros((N_CEN, n))
for i, r_cen in enumerate(cenarios):
    res_i = minimize(lambda w: -np.dot(w, r_cen), np.ones(n)/n,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    w_ws_all[i]  = res_i.x
    ret_ws_cen[i] = np.dot(res_i.x, r_cen)

# Métricas de comparação
EEV   = ret_eev_cen.mean()
RP    = ret_rp_cen.mean()
WS    = ret_ws_cen.mean()
VSS   = RP - EEV    # Value of Stochastic Solution
EVPI  = WS - RP     # Expected Value of Perfect Information

print(f"\n{'='*40}")
print(f"EEV  (det. simples)  : {EEV:.4f}  ({EEV*100:.2f}%)")
print(f"RP   (estocástico)   : {RP:.4f}  ({RP*100:.2f}%)")
print(f"WS   (perfeito)      : {WS:.4f}  ({WS*100:.2f}%)")
print(f"VSS  (RP - EEV)      : {VSS:.4f}  ({VSS*100:.2f}%)")
print(f"EVPI (WS - RP)       : {EVPI:.4f}  ({EVPI*100:.2f}%)")
print(f"{'='*40}")

# Risco (VaR 5%) de cada solução
var95_eev = np.percentile(ret_eev_cen, 5)
var95_rp  = np.percentile(ret_rp_cen, 5)
var95_ws  = np.percentile(ret_ws_cen, 5)

# ---------- 3. Salvar resultados ----------
df_resultado = pd.DataFrame({
    'Cenario':      range(1, N_CEN + 1),
    'Ret_EEV':      np.round(ret_eev_cen, 6),
    'Ret_RP':       np.round(ret_rp_cen, 6),
    'Ret_WS':       np.round(ret_ws_cen, 6),
    **{f'r_{t}': np.round(cenarios[:, i], 6) for i, t in enumerate(tickers_val)},
})
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Distribuição de retornos nos cenários por solução
plt.figure(figsize=(11, 5))
sns.kdeplot(ret_eev_cen * 100, color='#1565C0', linewidth=2, label=f'EEV (média={EEV*100:.2f}%)')
sns.kdeplot(ret_rp_cen  * 100, color='#2E7D32', linewidth=2, label=f'RP  (média={RP*100:.2f}%)')
sns.kdeplot(ret_ws_cen  * 100, color='#C62828', linewidth=2, label=f'WS  (média={WS*100:.2f}%)')
plt.axvline(var95_eev * 100, color='#1565C0', linewidth=1, linestyle=':', label=f'VaR5% EEV')
plt.axvline(var95_rp  * 100, color='#2E7D32', linewidth=1, linestyle=':', label=f'VaR5% RP')
plt.axvline(var95_ws  * 100, color='#C62828', linewidth=1, linestyle=':', label=f'VaR5% WS')
plt.xlabel('Retorno Anual (%) — Cenários Monte Carlo')
plt.ylabel('Densidade')
plt.title('Distribuição de Retornos: EEV × RP (Estocástico) × WS (Informação Perfeita)', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_distribuicao, dpi=150)
plt.show()
plt.close()

# 4.2 VSS e EVPI — comparativo de valor
metricas   = ['EEV', 'RP\n(Estocástico)', 'WS\n(Perfeito)']
valores    = [EEV * 100, RP * 100, WS * 100]
cores_met  = ['#1565C0', '#2E7D32', '#C62828']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(metricas, valores, color=cores_met, edgecolor='white', linewidth=1.2)
axes[0].set_ylabel('Retorno Médio Esperado (%)')
axes[0].set_title('Comparativo EEV × RP × WS')
axes[0].grid(True, axis='y', alpha=0.4)
for i, v in enumerate(valores):
    axes[0].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)

gains     = ['VSS\n(RP - EEV)', 'EVPI\n(WS - RP)']
vals_gain = [VSS * 100, EVPI * 100]
cores_g   = ['#2E7D32', '#FF9800']
axes[1].bar(gains, vals_gain, color=cores_g, edgecolor='white', linewidth=1.2)
axes[1].set_ylabel('Ganho em Retorno (%)')
axes[1].set_title('VSS (Valor da Informação Estocástica) e EVPI')
axes[1].grid(True, axis='y', alpha=0.4)
for i, v in enumerate(vals_gain):
    axes[1].text(i, v + 0.005, f'{v:.3f}%', ha='center', fontweight='bold', fontsize=11)
plt.suptitle('Programação Estocástica em Dois Estágios', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(file_ev_vs_ws, dpi=150)
plt.show()
plt.close()

# 4.3 Composição dos portfólios EEV vs RP
x   = np.arange(n)
w_b = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - w_b/2, w_eev * 100, width=w_b, label='EEV (Determinístico)', color='#1565C0')
plt.bar(x + w_b/2, w_rp  * 100, width=w_b, label='RP  (Estocástico)',    color='#2E7D32')
plt.xticks(x, nomes_val, rotation=15, ha='right', fontsize=9)
plt.ylabel('Alocação (%)')
plt.title('Comparação de Portfólios: EEV vs RP (Estocástico)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(file_solucoes, dpi=150)
plt.show()
plt.close()

# 4.4 Cenários interativos (Plotly)
df_cen_melt = pd.DataFrame({
    'Cenario': list(range(1, N_CEN+1)) * 3,
    'Retorno': np.concatenate([ret_eev_cen * 100, ret_rp_cen * 100, ret_ws_cen * 100]),
    'Solucao': ['EEV'] * N_CEN + ['RP'] * N_CEN + ['WS'] * N_CEN,
})
fig_inter = px.histogram(df_cen_melt, x='Retorno', color='Solucao', barmode='overlay',
                         nbins=60, opacity=0.6,
                         color_discrete_map={'EEV': '#1565C0', 'RP': '#2E7D32', 'WS': '#C62828'},
                         title='Distribuição de Retornos por Solução — Interativo',
                         labels={'Retorno': 'Retorno Anual (%)'})
fig_inter.write_html(file_cenarios_inter)
fig_inter.show()
print(f"Cenários interativos salvos: {file_cenarios_inter}")

# 4.5 Gráfico 3D: cenários (SPY × EEM × Retorno RP)
fig_3d = px.scatter_3d(
    x=cenarios[:, 0] * 100, y=cenarios[:, 1] * 100, z=ret_rp_cen * 100,
    color=ret_rp_cen * 100, color_continuous_scale='RdBu',
    opacity=0.6,
    title='Cenários Estocásticos 3D: SPY × EEM × Retorno RP',
    labels={'x': 'Retorno SPY (%)', 'y': 'Retorno EEM (%)', 'z': 'Retorno RP (%)'},
)
fig_3d.write_html(file_3d)
fig_3d.show()
print(f"Gráfico 3D salvo: {file_3d}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_distribuicao}")
print(f"  {file_ev_vs_ws}")
print(f"  {file_solucoes}")
print(f"  {file_cenarios_inter}")
print(f"  {file_3d}")
