# =============================================================
# STEP 06 - Teoria dos Jogos e Equilíbrio de Nash
# -------------------------------------------------------------
# Objetivo: Modelar interações estratégicas entre empresas
#           concorrentes via Equilíbrio de Nash. Constrói a
#           matriz de payoff a partir de retornos históricos
#           e identifica estratégias de equilíbrio (puras e
#           mistas) em um jogo de competição de mercado.
# Entrada  : output/base_step_00_yahoo_finance.csv  (gerado pelo step-00)
# Saídas   : output/base_step_06_nash_equilibrio.csv
#            output/base_step_06_nash_payoff_heatmap.png
#            output/base_step_06_nash_estrategias.png
#            output/base_step_06_nash_retornos_cumulados.png
#            output/base_step06_nash_payoff_interativo.html
#            output/base_step06_nash_retornos_interativo.html
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

file_output          = 'output/base_step_06_nash_equilibrio.csv'
file_payoff_heatmap  = 'output/base_step_06_nash_payoff_heatmap.png'
file_estrategias     = 'output/base_step_06_nash_estrategias.png'
file_retornos_cumul  = 'output/base_step_06_nash_retornos_cumulados.png'
file_payoff_inter    = 'output/base_step06_nash_payoff_interativo.html'
file_retornos_inter  = 'output/base_step06_nash_retornos_interativo.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Aquisição de dados (base auditada step-00) ----------
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
print("Carregando base auditada do step-00 (Yahoo Finance)...")
df_cache = pd.read_csv('output/base_step_00_yahoo_finance.csv',
                       sep=';', decimal=',', encoding='utf-8-sig')
df_cache['Data'] = pd.to_datetime(df_cache['Data'])
mask = (df_cache['Ticker'].isin(tickers) &
        (df_cache['Data'] >= '2021-01-01') &
        (df_cache['Data'] <= '2023-12-31'))
dados = df_cache[mask].pivot(index='Data', columns='Ticker', values='Close')[tickers]
print(f"Dados carregados: {dados.shape[0]} observações, {len(tickers)} ativos")

# ---------- 2. Construção da Matriz de Payoff e Equilíbrio de Nash ----------
retornos      = dados.pct_change().dropna()
ret_anual     = retornos.mean() * 252
vol_anual     = retornos.std() * np.sqrt(252)
sharpe        = ret_anual / vol_anual
ret_acum      = (1 + retornos).cumprod()

n = len(tickers)

# Payoff do jogador A (linha) ao adotar estratégia i contra estratégia j do rival:
# payoff_A[i,j] = retorno_i - correlação(i,j) * retorno_j
# (quanto menor a correlação, menos o rival captura do ganho de i)
payoff_A = np.zeros((n, n))
payoff_B = np.zeros((n, n))
for i, t1 in enumerate(tickers):
    for j, t2 in enumerate(tickers):
        corr = retornos[t1].corr(retornos[t2])
        payoff_A[i, j] = ret_anual[t1] - corr * ret_anual[t2]
        payoff_B[i, j] = ret_anual[t2] - corr * ret_anual[t1]

# Equilíbrios de Nash em estratégias puras: (i*, j*) é Nash se
#   payoff_A[i*, j*] >= payoff_A[i, j*]  para todo i  (melhor resposta de A)
#   payoff_B[i*, j*] >= payoff_B[i*, j]  para todo j  (melhor resposta de B)
nash_equilibria = []
for i in range(n):
    for j in range(n):
        mr_A = (payoff_A[i, j] == payoff_A[:, j].max())
        mr_B = (payoff_B[i, j] == payoff_B[i, :].max())
        if mr_A and mr_B:
            nash_equilibria.append({
                'Empresa_A': tickers[i],
                'Empresa_B': tickers[j],
                'Payoff_A':  round(payoff_A[i, j], 6),
                'Payoff_B':  round(payoff_B[i, j], 6),
            })

print(f"\nEquilíbrios de Nash (estratégias puras) encontrados: {len(nash_equilibria)}")
for ne in nash_equilibria:
    print(f"  ({ne['Empresa_A']}, {ne['Empresa_B']}) → "
          f"payoff_A={ne['Payoff_A']:.4f}, payoff_B={ne['Payoff_B']:.4f}")

# Estatísticas dos ativos
df_stats = pd.DataFrame({
    'Ticker':          tickers,
    'Retorno_Anual':   [round(ret_anual[t], 6) for t in tickers],
    'Volatilidade':    [round(vol_anual[t], 6) for t in tickers],
    'Sharpe':          [round(sharpe[t], 6) for t in tickers],
    'Nash_Equilibrio': ['Sim' if any(ne['Empresa_A'] == t for ne in nash_equilibria) else 'Não'
                        for t in tickers],
})

df_nash = pd.DataFrame(nash_equilibria)
df_resultado = pd.concat([df_stats, df_nash.rename(columns={
    'Empresa_A': 'Nash_Empresa_A', 'Empresa_B': 'Nash_Empresa_B',
    'Payoff_A': 'Nash_Payoff_A',  'Payoff_B': 'Nash_Payoff_B'
})], axis=1)

# ---------- 3. Salvar resultados ----------
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Heatmap da Matriz de Payoff (Jogador A)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(payoff_A, annot=True, fmt='.3f', xticklabels=tickers, yticklabels=tickers,
            cmap='coolwarm', center=0, ax=axes[0])
axes[0].set_title('Matriz de Payoff — Jogador A')
axes[0].set_xlabel('Estratégia do Rival (Coluna)')
axes[0].set_ylabel('Estratégia Própria (Linha)')

sns.heatmap(payoff_B, annot=True, fmt='.3f', xticklabels=tickers, yticklabels=tickers,
            cmap='coolwarm', center=0, ax=axes[1])
axes[1].set_title('Matriz de Payoff — Jogador B')
axes[1].set_xlabel('Estratégia Própria (Coluna)')
axes[1].set_ylabel('Estratégia do Rival (Linha)')

plt.suptitle('Matrizes de Payoff — Equilíbrio de Nash', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(file_payoff_heatmap, dpi=150)
plt.show()
plt.close()

# 4.2 Espaço de Estratégias: Retorno vs Volatilidade
cores = ['#1565C0', '#C62828', '#2E7D32', '#E65100']
plt.figure(figsize=(9, 6))
for i, t in enumerate(tickers):
    plt.scatter(vol_anual[t] * 100, ret_anual[t] * 100,
                s=250, color=cores[i], label=t, zorder=5)
    plt.annotate(t, (vol_anual[t] * 100, ret_anual[t] * 100),
                 textcoords='offset points', xytext=(9, 4), fontsize=11, fontweight='bold')
for ne in nash_equilibria:
    idx = tickers.index(ne['Empresa_A'])
    plt.scatter(vol_anual[tickers[idx]] * 100, ret_anual[tickers[idx]] * 100,
                s=600, edgecolors='gold', facecolors='none', linewidths=2.5, zorder=6)
plt.xlabel('Volatilidade Anual (%)', fontsize=11)
plt.ylabel('Retorno Anual (%)', fontsize=11)
plt.title('Espaço de Estratégias: Retorno × Volatilidade\n(Nash em equilíbrio destacado em dourado)',
          fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_estrategias, dpi=150)
plt.show()
plt.close()

# 4.3 Retornos Acumulados
plt.figure(figsize=(11, 5))
for i, t in enumerate(tickers):
    plt.plot(ret_acum.index, ret_acum[t], label=t, linewidth=2, color=cores[i])
plt.title('Retorno Acumulado dos Concorrentes (2021–2023)', fontsize=12)
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado (base 1)')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_retornos_cumul, dpi=150)
plt.show()
plt.close()

# 4.4 Heatmap interativo da Matriz de Payoff (Plotly)
texto = [[f'{payoff_A[i,j]:.4f}' for j in range(n)] for i in range(n)]
fig_heat = go.Figure(data=go.Heatmap(
    z=payoff_A, x=tickers, y=tickers,
    colorscale='RdBu', zmid=0,
    text=texto, texttemplate='%{text}', showscale=True
))
fig_heat.update_layout(
    title='Matriz de Payoff (Jogador A) — Interativo',
    xaxis_title='Estratégia do Rival', yaxis_title='Estratégia Própria'
)
fig_heat.write_html(file_payoff_inter)
fig_heat.show()
print(f"Heatmap interativo salvo: {file_payoff_inter}")

# 4.5 Retornos acumulados interativo (Plotly)
df_ret_plot = ret_acum.reset_index().melt(id_vars='Data', var_name='Empresa', value_name='Retorno_Acumulado')
fig_ret = px.line(df_ret_plot, x='Data', y='Retorno_Acumulado', color='Empresa',
                  title='Retorno Acumulado dos Concorrentes (Equilíbrio de Nash)',
                  labels={'Data': 'Data', 'Retorno_Acumulado': 'Retorno Acumulado'})
fig_ret.write_html(file_retornos_inter)
fig_ret.show()
print(f"Retornos interativos salvos: {file_retornos_inter}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_payoff_heatmap}")
print(f"  {file_estrategias}")
print(f"  {file_retornos_cumul}")
print(f"  {file_payoff_inter}")
print(f"  {file_retornos_inter}")
