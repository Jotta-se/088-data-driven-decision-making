# =============================================================
# STEP 07 - Equilíbrio de Stackelberg
# -------------------------------------------------------------
# Objetivo: Modelar competição líder-seguidor (Stackelberg) no
#           mercado de tecnologia. O líder anuncia quantidade
#           primeiro; os seguidores respondem de forma ótima.
#           Compara o equilíbrio de Stackelberg com o de Nash-
#           Cournot para mensurar a vantagem do primeiro mover.
# Entrada  : output/base_step_00_yahoo_finance.csv  (gerado pelo step-00)
# Saídas   : output/base_step_07_stackelberg_resultado.csv
#            output/base_step_07_stackelberg_equilibrio.png
#            output/base_step_07_stackelberg_comparativo.png
#            output/base_step_07_stackelberg_lucros.png
#            output/base_step07_stackelberg_interativo.html
#            output/base_step07_stackelberg_3d.html
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
from scipy.optimize import minimize_scalar

file_output          = 'output/base_step_07_stackelberg_resultado.csv'
file_equilibrio      = 'output/base_step_07_stackelberg_equilibrio.png'
file_comparativo     = 'output/base_step_07_stackelberg_comparativo.png'
file_lucros          = 'output/base_step_07_stackelberg_lucros.png'
file_interativo      = 'output/base_step07_stackelberg_interativo.html'
file_3d              = 'output/base_step07_stackelberg_3d.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Aquisição de dados (base auditada step-00) ----------
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
lider   = 'AAPL'
seguidores = ['MSFT', 'GOOGL', 'AMZN']
print("Carregando base auditada do step-00 (Yahoo Finance)...")
df_cache = pd.read_csv('output/base_step_00_yahoo_finance.csv',
                       sep=';', decimal=',', encoding='utf-8-sig')
df_cache['Data'] = pd.to_datetime(df_cache['Data'])
mask = (df_cache['Ticker'].isin(tickers) &
        (df_cache['Data'] >= '2021-01-01') &
        (df_cache['Data'] <= '2023-12-31'))
dados = df_cache[mask].pivot(index='Data', columns='Ticker', values='Close')[tickers]
print(f"Dados carregados: {dados.shape[0]} observações, {len(tickers)} ativos")

retornos  = dados.pct_change().dropna()
ret_anual = retornos.mean() * 252
vol_anual = retornos.std() * np.sqrt(252)

# ---------- 2. Modelo de Stackelberg ----------
# Premissa Cournot linear: demanda inversa P = a - b*(Q_total)
# Custo marginal c_i estimado como (1 - Sharpe_normalizado) * a
# Lucro_i = (P - c_i) * q_i

a = 100.0   # preço máximo (normalizado)
b = 1.0     # inclinação da demanda

sharpe     = ret_anual / vol_anual
sharpe_min = sharpe.min()
sharpe_max = sharpe.max()
# Custo marginal: empresas mais eficientes (maior Sharpe) têm custo menor
custo = {t: a * 0.2 + 0.6 * a * (1 - (sharpe[t] - sharpe_min) / (sharpe_max - sharpe_min + 1e-9))
         for t in tickers}

# --- Nash-Cournot (simultâneo) ---
# q_i* = (a - c_i - b * soma(q_j, j!=i)) / (2*b)  → iteração de ponto fixo
n = len(tickers)
q_cournot = np.ones(n) * (a / (b * (n + 1)))
for _ in range(1000):
    q_novo = np.array([
        max(0, (a - list(custo.values())[i] - b * (q_cournot.sum() - q_cournot[i])) / (2 * b))
        for i in range(n)
    ])
    if np.max(np.abs(q_novo - q_cournot)) < 1e-9:
        break
    q_cournot = q_novo.copy()

P_cournot = a - b * q_cournot.sum()
lucro_cournot = {tickers[i]: (P_cournot - list(custo.values())[i]) * q_cournot[i] for i in range(n)}

# --- Stackelberg (sequencial) ---
# Melhor resposta dos seguidores para quantidade do líder q_L:
# q_seg_i*(q_L) = (a - c_i - b*q_L - b*soma(q_outros_segs)) / (2*b)
# Em equilíbrio simétrico dos seguidores:
# q_seg_i* = (a - c_i - b*q_L) / (b*(n_segs+1))   [simplificado para segs homogêneos]
n_segs = len(seguidores)

def lucro_lider(q_L):
    soma_segs = sum(
        max(0, (a - custo[s] - b * q_L) / (b * (n_segs + 1)))
        for s in seguidores
    )
    P = a - b * (q_L + soma_segs)
    return -(P - custo[lider]) * q_L   # negativo para minimizar

res     = minimize_scalar(lucro_lider, bounds=(0, a / b), method='bounded')
q_L_st  = res.x

q_segs_st = {s: max(0, (a - custo[s] - b * q_L_st) / (b * (n_segs + 1)))
             for s in seguidores}
Q_st    = q_L_st + sum(q_segs_st.values())
P_st    = a - b * Q_st

lucro_st = {lider: (P_st - custo[lider]) * q_L_st}
lucro_st.update({s: (P_st - custo[s]) * q_segs_st[s] for s in seguidores})

print(f"\nCournot  — Preço: {P_cournot:.2f}  |  Q total: {q_cournot.sum():.2f}")
print(f"Stackelberg — Preço: {P_st:.2f}  |  Q total: {Q_st:.2f}")
print(f"\nLucro Líder ({lider})  Cournot: {lucro_cournot[lider]:.2f}  |  Stackelberg: {lucro_st[lider]:.2f}")

# ---------- 3. Salvar resultados ----------
registros = []
for t in tickers:
    idx = tickers.index(t)
    registros.append({
        'Empresa':           t,
        'Papel':             'Líder' if t == lider else 'Seguidor',
        'Custo_Marginal':    round(custo[t], 4),
        'Q_Cournot':         round(q_cournot[idx], 4),
        'Lucro_Cournot':     round(lucro_cournot[t], 4),
        'Q_Stackelberg':     round(q_L_st if t == lider else q_segs_st.get(t, 0), 4),
        'Lucro_Stackelberg': round(lucro_st[t], 4),
        'Vantagem_Lider_%':  round((lucro_st[t] / (lucro_cournot[t] + 1e-9) - 1) * 100, 2),
    })

df_resultado = pd.DataFrame(registros)
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Comparativo de quantidades: Cournot vs Stackelberg
x   = np.arange(len(tickers))
w   = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, [round(q_cournot[i], 2) for i in range(n)],   width=w, label='Cournot (Nash)', color='#1565C0')
ax.bar(x + w/2, [df_resultado.loc[df_resultado.Empresa == t, 'Q_Stackelberg'].values[0]
                 for t in tickers], width=w, label='Stackelberg', color='#C62828')
ax.set_xticks(x)
ax.set_xticklabels(tickers, fontsize=11)
ax.set_ylabel('Quantidade de Equilíbrio')
ax.set_title('Quantidades de Equilíbrio: Cournot vs Stackelberg', fontsize=12)
ax.legend(fontsize=10)
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(file_equilibrio, dpi=150)
plt.show()
plt.close()

# 4.2 Lucros comparativos
lucros_cournot_list    = [lucro_cournot[t] for t in tickers]
lucros_stackelberg_list = [lucro_st[t] for t in tickers]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, lucros_cournot_list,    width=w, label='Cournot', color='#1565C0')
ax.bar(x + w/2, lucros_stackelberg_list, width=w, label='Stackelberg', color='#C62828')
ax.set_xticks(x)
ax.set_xticklabels(tickers, fontsize=11)
ax.set_ylabel('Lucro')
ax.set_title('Lucros de Equilíbrio: Cournot vs Stackelberg\n(Líder destacado)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.4)
# Destacar líder
ax.axvspan(-0.5, 0.5, alpha=0.08, color='gold')
ax.annotate('Líder', xy=(0, max(lucros_stackelberg_list) * 0.95),
            ha='center', fontsize=9, color='#8B6914')
plt.tight_layout()
plt.savefig(file_comparativo, dpi=150)
plt.show()
plt.close()

# 4.3 Vantagem do primeiro mover (%)
# Ordenar por valor (SKILL: ordenar barras se não há ordem natural entre categorias)
df_vantagem = df_resultado[['Empresa', 'Vantagem_Lider_%']].sort_values('Vantagem_Lider_%')
cores_bar = ['#C62828' if v > 0 else '#1565C0' for v in df_vantagem['Vantagem_Lider_%']]
plt.figure(figsize=(8, 5))
plt.bar(df_vantagem['Empresa'], df_vantagem['Vantagem_Lider_%'], color=cores_bar)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Vantagem Stackelberg vs Cournot (%)\n(positivo = Stackelberg melhor)', fontsize=12)
plt.ylabel('Variação do Lucro (%)')
plt.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(file_lucros, dpi=150)
plt.show()
plt.close()

# 4.4 Gráfico interativo: Cournot vs Stackelberg (Plotly)
df_melt = df_resultado[['Empresa', 'Lucro_Cournot', 'Lucro_Stackelberg']].melt(
    id_vars='Empresa', var_name='Modelo', value_name='Lucro')
fig_inter = px.bar(df_melt, x='Empresa', y='Lucro', color='Modelo', barmode='group',
                   title='Lucros de Equilíbrio: Cournot vs Stackelberg',
                   labels={'Lucro': 'Lucro', 'Empresa': 'Empresa'},
                   color_discrete_map={'Lucro_Cournot': '#1565C0', 'Lucro_Stackelberg': '#C62828'})
fig_inter.write_html(file_interativo)
fig_inter.show()
print(f"Gráfico interativo salvo: {file_interativo}")

# 4.5 Superfície 3D: lucro do líder em função de (q_L, q_seguidor)
q_l_range  = np.linspace(0, 50, 60)
q_s_range  = np.linspace(0, 50, 60)
QL, QS     = np.meshgrid(q_l_range, q_s_range)
P_surf     = np.maximum(0, a - b * (QL + QS))
Lucro_surf = (P_surf - custo[lider]) * QL

fig_3d = go.Figure(data=[go.Surface(z=Lucro_surf, x=q_l_range, y=q_s_range,
                                    colorscale='Viridis', opacity=0.85)])
fig_3d.add_trace(go.Scatter3d(x=[q_L_st], y=[sum(q_segs_st.values())], z=[-lucro_lider(q_L_st)],
                              mode='markers',
                              marker=dict(size=8, color='red', symbol='diamond'),
                              name='Ótimo Stackelberg'))
fig_3d.update_layout(
    title='Superfície de Lucro do Líder (Stackelberg)',
    scene=dict(xaxis_title='q_Líder', yaxis_title='q_Seguidores', zaxis_title='Lucro_Líder')
)
fig_3d.write_html(file_3d)
fig_3d.show()
print(f"Gráfico 3D salvo: {file_3d}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_equilibrio}")
print(f"  {file_comparativo}")
print(f"  {file_lucros}")
print(f"  {file_interativo}")
print(f"  {file_3d}")