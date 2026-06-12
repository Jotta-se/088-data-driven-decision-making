# =============================================================
# STEP 13 - Otimização Robusta
# -------------------------------------------------------------
# Objetivo: Construir portfólio de ativos robustamente ótimo
#           para o pior cenário dentro de um conjunto de
#           incerteza elipsoidal nos retornos. Compara a
#           solução robusta (minimax) com a determinística
#           (média-variância) em termos de retorno garantido
#           e comportamento nos cenários adversos.
# Entrada  : output/base_step_09_world_bank_raw.csv  (gerado pelo step-09)
#            output/base_step_00_yahoo_finance.csv   (gerado pelo step-00)
# Saídas   : output/base_step_13_robusta_resultado.csv
#            output/base_step_13_robusta_portfolios.png
#            output/base_step_13_robusta_comparativo.png
#            output/base_step_13_robusta_incerteza.png
#            output/base_step13_robusta_fronteira_interativa.html
#            output/base_step13_robusta_cenarios.html
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
from scipy.optimize import minimize

file_input            = 'output/base_step_09_world_bank_raw.csv'
file_output           = 'output/base_step_13_robusta_resultado.csv'
file_portfolios       = 'output/base_step_13_robusta_portfolios.png'
file_comparativo      = 'output/base_step_13_robusta_comparativo.png'
file_incerteza        = 'output/base_step_13_robusta_incerteza.png'
file_fronteira_inter  = 'output/base_step13_robusta_fronteira_interativa.html'
file_cenarios         = 'output/base_step13_robusta_cenarios.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Carregar dados e preparar ativos ----------
print(f"Carregando cache World Bank: {file_input}")
df_wb = pd.read_csv(file_input, sep=';', encoding='utf-8-sig')
print(f"Dados WB carregados: {len(df_wb)} países")

# Usar dados de mercado da base auditada step-00 para construir portfólio robusto
# (ativos representando economias das regiões do World Bank)
tickers = ['EEM', 'VEA', 'SPY', 'EWJ', 'EWZ', 'GLD', 'TLT', 'IEF']
nomes   = ['Emerg. Markets', 'Europa Dev.', 'EUA (S&P500)', 'Japão',
           'Brasil', 'Ouro', 'T-Bond Longo', 'T-Bond Médio']
print(f"\nCarregando dados de mercado da base auditada step-00...")
df_cache = pd.read_csv('output/base_step_00_yahoo_finance.csv',
                       sep=';', decimal=',', encoding='utf-8-sig')
df_cache['Data'] = pd.to_datetime(df_cache['Data'])
mask = (df_cache['Ticker'].isin(tickers) &
        (df_cache['Data'] >= '2018-01-01') &
        (df_cache['Data'] <= '2023-12-31'))
dados = df_cache[mask].pivot(index='Data', columns='Ticker', values='Close')
dados = dados.dropna(axis=1, how='all')
tickers_val = list(dados.columns)
nomes_val   = [nomes[tickers.index(t)] for t in tickers_val]
print(f"Ativos válidos: {tickers_val}")

ret_diarios = dados.pct_change().dropna()
mu          = ret_diarios.mean().values * 252        # retornos anualizados
Sigma       = ret_diarios.cov().values  * 252        # covariância anualizada
n           = len(tickers_val)

# ---------- 2. Otimização Robusta (Min-Max) ----------
# Conjunto de incerteza elipsoidal: Γ = {r̃ : (r̃ - μ)ᵀ Σ⁻¹ (r̃ - μ) ≤ κ²}
# Problema robusto: max_w  min_{r̃ ∈ Γ}  wᵀr̃
# Solução: max_w  wᵀμ - κ * √(wᵀΣw)   s.t. 1ᵀw=1, w≥0

kappas = [0.0, 0.5, 1.0, 1.5, 2.0]   # níveis de conservadorismo

def obj_robusta(w, kappa):
    ret  = np.dot(w, mu)
    risco = np.sqrt(np.dot(w, np.dot(Sigma, w)))
    return -(ret - kappa * risco)

constraints = ({'type': 'eq',   'fun': lambda w: np.sum(w) - 1})
bounds      = tuple((0, 1) for _ in range(n))
w0          = np.ones(n) / n

resultados = []
for kappa in kappas:
    res = minimize(obj_robusta, w0, args=(kappa,), method='SLSQP',
                   bounds=bounds, constraints=constraints, options={'ftol': 1e-9})
    w_rob   = res.x
    ret_rob = np.dot(w_rob, mu)
    vol_rob = np.sqrt(np.dot(w_rob, np.dot(Sigma, w_rob)))
    ret_gar = ret_rob - kappa * vol_rob   # retorno garantido (pior caso)
    resultados.append({
        'Kappa':          kappa,
        'Retorno_Medio':  round(ret_rob, 6),
        'Volatilidade':   round(vol_rob, 6),
        'Retorno_Garantido': round(ret_gar, 6),
        'Sharpe':         round(ret_rob / (vol_rob + 1e-9), 6),
        **{f'w_{t}': round(w_rob[i], 6) for i, t in enumerate(tickers_val)},
    })
    print(f"κ={kappa:.1f} | ret={ret_rob:.4f} | vol={vol_rob:.4f} | garantido={ret_gar:.4f}")

df_resultado = pd.DataFrame(resultados)

# Simulação de Monte Carlo de cenários para cada solução
N_CEN = 2000
np.random.seed(42)
L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))
cenarios = mu + (L @ np.random.randn(n, N_CEN)).T   # N_CEN x n

retornos_por_kappa = {}
for row in df_resultado.itertuples():
    w_k = np.array([getattr(row, f'w_{t}') for t in tickers_val])
    retornos_por_kappa[row.Kappa] = cenarios @ w_k

# ---------- 3. Salvar resultados ----------
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Pesos dos portfólios por κ
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(tickers_val))
w = 0.15
cores_k = ['#1565C0', '#1976D2', '#42A5F5', '#90CAF9', '#E3F2FD']
for j, row in df_resultado.iterrows():
    pesos = [getattr(row, f'w_{t}') for t in tickers_val]
    ax.bar(x + (j - 2) * w, pesos, width=w, label=f'κ={row.Kappa}', color=cores_k[j])
ax.set_xticks(x)
ax.set_xticklabels(nomes_val, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Peso no Portfólio')
ax.set_title('Alocação dos Portfólios Robustos por Nível de Conservadorismo (κ)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(file_portfolios, dpi=150)
plt.show()
plt.close()

# 4.2 Retorno médio vs retorno garantido (pior caso) por κ
plt.figure(figsize=(9, 5))
plt.plot(df_resultado['Kappa'], df_resultado['Retorno_Medio'] * 100,
         'o-', color='#1565C0', linewidth=2, label='Retorno Médio Esperado', markersize=7)
plt.plot(df_resultado['Kappa'], df_resultado['Retorno_Garantido'] * 100,
         's--', color='#C62828', linewidth=2, label='Retorno Garantido (pior caso)', markersize=7)
plt.fill_between(df_resultado['Kappa'],
                 df_resultado['Retorno_Garantido'] * 100,
                 df_resultado['Retorno_Medio'] * 100,
                 alpha=0.12, color='grey', label='Intervalo de incerteza')
plt.xlabel('Parâmetro de Robustez (κ)', fontsize=11)
plt.ylabel('Retorno Anual (%)')
plt.title('Retorno Esperado vs Retorno Garantido por Nível de Robustez', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_comparativo, dpi=150)
plt.show()
plt.close()

# 4.3 Distribuição dos retornos por κ nos cenários Monte Carlo
# SKILL: >3 grupos em KDE = spaghetti proibido → violin plot (5 κ valores)
df_violin = pd.DataFrame({
    'Retorno (%)': np.concatenate([retornos_por_kappa[k] * 100 for k in kappas]),
    'κ': np.concatenate([[f'κ={k}'] * len(retornos_por_kappa[k]) for k in kappas])
})
fig, ax = plt.subplots(figsize=(11, 5))
sns.violinplot(data=df_violin, x='κ', y='Retorno (%)',
               palette='Blues_r', ax=ax, inner='box', linewidth=1.2)
ax.axhline(0, color='black', linewidth=0.9, linestyle='--', label='Retorno zero')
ax.set_xlabel('Nível de Conservadorismo (κ)', fontsize=11)
ax.set_ylabel('Retorno Anual (%) — Cenários Monte Carlo')
ax.set_title('Distribuição de Retornos nos Cenários de Incerteza por Nível de Robustez', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.35)
plt.tight_layout()
plt.savefig(file_incerteza, dpi=150)
plt.show()
plt.close()

# 4.4 Fronteira Robusta interativa (Plotly)
fig_inter = go.Figure()
fig_inter.add_trace(go.Scatter(
    x=df_resultado['Volatilidade'] * 100,
    y=df_resultado['Retorno_Medio'] * 100,
    mode='lines+markers+text',
    text=[f'κ={k}' for k in df_resultado['Kappa']],
    textposition='top center',
    marker=dict(size=12, color=df_resultado['Kappa'], colorscale='Blues', showscale=True,
                colorbar=dict(title='κ')),
    line=dict(color='#1565C0', width=2),
    name='Retorno Médio',
))
fig_inter.add_trace(go.Scatter(
    x=df_resultado['Volatilidade'] * 100,
    y=df_resultado['Retorno_Garantido'] * 100,
    mode='lines+markers',
    marker=dict(size=10, color='#C62828', symbol='square'),
    line=dict(color='#C62828', width=2, dash='dash'),
    name='Retorno Garantido',
))
fig_inter.update_layout(
    title='Fronteira Robusta: Retorno Médio vs Garantido por κ',
    xaxis_title='Volatilidade (%)', yaxis_title='Retorno (%)'
)
fig_inter.write_html(file_fronteira_inter)
fig_inter.show()
print(f"Fronteira robusta interativa salva: {file_fronteira_inter}")

# 4.5 Distribuições de cenários (interativo)
fig_cen = go.Figure()
for j, kappa in enumerate(kappas):
    fig_cen.add_trace(go.Violin(
        y=retornos_por_kappa[kappa] * 100,
        name=f'κ={kappa}',
        box_visible=True, meanline_visible=True,
    ))
fig_cen.update_layout(
    title='Distribuição de Retornos por Nível de Robustez (Violin Plot)',
    yaxis_title='Retorno Anual (%) — Monte Carlo', violinmode='group'
)
fig_cen.write_html(file_cenarios)
fig_cen.show()
print(f"Cenários interativos salvos: {file_cenarios}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_portfolios}")
print(f"  {file_comparativo}")
print(f"  {file_incerteza}")
print(f"  {file_fronteira_inter}")
print(f"  {file_cenarios}")