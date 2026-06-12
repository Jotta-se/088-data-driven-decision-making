# =============================================================
# STEP 12 - Teoria da Decisão Bayesiana
# -------------------------------------------------------------
# Objetivo: Aplicar inferência bayesiana para estimar o Customer
#           Lifetime Value (CLV) de clientes do Online Retail II.
#           Usa modelo Beta-Binomial para a probabilidade de
#           recompra (prior conjugado) e atualiza posteriors
#           com dados observados. Gera regras de decisão ótimas
#           para priorização de clientes.
# Entrada  : output/base_step_10_online_retail_raw.csv
# Saídas   : output/base_step_12_bayesiana_clv.csv
#            output/base_step_12_bayesiana_posterior.png
#            output/base_step_12_bayesiana_clv_dist.png
#            output/base_step_12_bayesiana_atualizacao.png
#            output/base_step12_bayesiana_interativo.html
#            output/base_step12_bayesiana_clv_3d.html
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.stats import beta as beta_dist, gamma as gamma_dist

file_input          = 'output/base_step_02_online_retail.csv'
file_output         = 'output/base_step_12_bayesiana_clv.csv'
file_posterior      = 'output/base_step_12_bayesiana_posterior.png'
file_clv_dist       = 'output/base_step_12_bayesiana_clv_dist.png'
file_atualizacao    = 'output/base_step_12_bayesiana_atualizacao.png'
file_interativo     = 'output/base_step12_bayesiana_interativo.html'
file_clv_3d         = 'output/base_step12_bayesiana_clv_3d.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Carregar dados ----------
df_raw = pd.read_csv(file_input, sep=';', encoding='utf-8-sig', decimal=',')
print(f"Linhas originais: {len(df_raw)}")

df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]
# Normalização defensiva: Online Retail II pode salvar 'InvoiceDate' ou 'Invoice Date'
if 'InvoiceDate' in df.columns and 'Invoice Date' not in df.columns:
    df = df.rename(columns={'InvoiceDate': 'Invoice Date'})
df = df[df['Customer ID'].notna()].copy()
df = df[df['Quantity'].astype(float) > 0].copy()
df = df[df['Price'].astype(float) > 0].copy()
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
df['Revenue']      = df['Quantity'].astype(float) * df['Price'].astype(float)
df['Customer ID']  = df['Customer ID'].astype(int)
df = df.sort_values('Invoice Date')

# ---------- 2. Inferência Bayesiana — Modelo Beta-Binomial para CLV ----------
# Dividir em janela de calibração (70%) e holdout (30%) por data
data_split = df['Invoice Date'].quantile(0.70)
df_cal     = df[df['Invoice Date'] <= data_split]
df_hold    = df[df['Invoice Date'] > data_split]

data_ref = df['Invoice Date'].max()

# Features por cliente na calibração
def calcular_rfm(df_subset, data_referencia):
    return df_subset.groupby('Customer ID').agg(
        n_pedidos    = ('Invoice', 'nunique'),
        recencia     = ('Invoice Date', lambda x: (data_referencia - x.max()).days),
        ticket_medio = ('Revenue', lambda x: x.sum() / df_subset.loc[x.index, 'Invoice'].nunique()),
        receita_tot  = ('Revenue', 'sum'),
    ).reset_index()

rfm_cal  = calcular_rfm(df_cal, data_split)
rfm_hold = calcular_rfm(df_hold, data_ref)
rfm_hold = rfm_hold.rename(columns={'n_pedidos': 'n_pedidos_hold', 'receita_tot': 'receita_hold'})

# Clientes presentes em ambas as janelas
clientes_comuns = set(rfm_cal['Customer ID']) & set(rfm_hold['Customer ID'])
rfm_cal['Recomprou'] = rfm_cal['Customer ID'].isin(clientes_comuns).astype(int)
print(f"\nClientes na calibração : {len(rfm_cal)}")
print(f"Taxa de recompra real  : {rfm_cal['Recomprou'].mean():.3f}")

# Prior Beta(α₀, β₀): crença inicial sobre probabilidade de recompra
# α₀ = 2, β₀ = 5 → prior fraco (média = 0.29, equivalente a 7 observações)
alpha0, beta0 = 2.0, 5.0

# Posterior por cliente: Beta(α₀ + s, β₀ + (n - s))
# onde s = recomprou (1 ou 0), n = 1 (observação binária por cliente)
rfm_cal['alpha_post'] = alpha0 + rfm_cal['Recomprou']
rfm_cal['beta_post']  = beta0  + (1 - rfm_cal['Recomprou'])
rfm_cal['p_recompra'] = rfm_cal['alpha_post'] / (rfm_cal['alpha_post'] + rfm_cal['beta_post'])
rfm_cal['p_recompra_std'] = np.sqrt(
    rfm_cal['alpha_post'] * rfm_cal['beta_post'] /
    ((rfm_cal['alpha_post'] + rfm_cal['beta_post'])**2 *
     (rfm_cal['alpha_post'] + rfm_cal['beta_post'] + 1))
)

# CLV Bayesiano: E[CLV] = p_recompra * ticket_medio * horizonte_esperado
# horizonte_esperado ~ Gamma prior (forma de vida do cliente)
HORIZONTE_MESES = 12
rfm_cal['CLV_Bayesiano']   = rfm_cal['p_recompra'] * rfm_cal['ticket_medio'] * rfm_cal['n_pedidos'] * (HORIZONTE_MESES / 12)
rfm_cal['CLV_Incerteza']   = rfm_cal['p_recompra_std'] * rfm_cal['ticket_medio'] * rfm_cal['n_pedidos']

# Segmentação por CLV
rfm_cal['Segmento_CLV'] = pd.qcut(rfm_cal['CLV_Bayesiano'],
                                   q=4, labels=['Bronze', 'Prata', 'Ouro', 'Diamante'])

# Prioridade de ação: taxa Bayesiana * receita esperada
rfm_cal['Prioridade'] = rfm_cal['p_recompra'] * rfm_cal['CLV_Bayesiano']
rfm_cal['Rank_CLV']   = rfm_cal['CLV_Bayesiano'].rank(ascending=False).astype(int)

print(f"\nCLV Bayesiano médio   : R$ {rfm_cal['CLV_Bayesiano'].mean():.2f}")
print(f"CLV Bayesiano mediano : R$ {rfm_cal['CLV_Bayesiano'].median():.2f}")
print(f"Top 10% CLV médio     : R$ {rfm_cal['CLV_Bayesiano'].quantile(0.9):.2f}")

# ---------- 3. Salvar resultados ----------
rfm_cal.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Distribuições Prior e Posterior por segmento
p_vals = np.linspace(0, 1, 200)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
segmentos = ['Bronze', 'Prata', 'Ouro', 'Diamante']
cores_seg  = ['#CD7F32', '#A8A9AD', '#FFD700', '#0D47A1']
for ax, seg, cor in zip(axes.flat, segmentos, cores_seg):
    sub = rfm_cal[rfm_cal['Segmento_CLV'] == seg]
    a_med, b_med = sub['alpha_post'].mean(), sub['beta_post'].mean()
    prior_vals   = beta_dist.pdf(p_vals, alpha0, beta0)
    post_vals    = beta_dist.pdf(p_vals, a_med, b_med)
    ax.plot(p_vals, prior_vals, color='grey', linestyle='--', linewidth=1.5, label='Prior')
    ax.plot(p_vals, post_vals, color=cor, linewidth=2, label=f'Posterior {seg}')
    ax.axvline(a_med / (a_med + b_med), color=cor, linestyle=':', linewidth=1.2)
    ax.set_title(f'Segmento {seg} (n={len(sub)})', fontsize=10)
    ax.set_xlabel('p (prob. recompra)')
    ax.set_ylabel('Densidade')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Prior → Posterior Beta-Binomial por Segmento de CLV', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(file_posterior, dpi=150)
plt.show()
plt.close()

# 4.2 Distribuição do CLV Bayesiano por segmento
plt.figure(figsize=(11, 5))
for seg, cor in zip(segmentos, cores_seg):
    sub = rfm_cal[rfm_cal['Segmento_CLV'] == seg]
    sns.kdeplot(sub['CLV_Bayesiano'], label=f'{seg} (n={len(sub)})', color=cor, linewidth=2)
plt.axvline(rfm_cal['CLV_Bayesiano'].mean(), color='black', linestyle='--',
            linewidth=1.5, label=f'Média geral: R${rfm_cal["CLV_Bayesiano"].mean():.0f}')
plt.xlabel('CLV Bayesiano (R$)', fontsize=11)
plt.ylabel('Densidade')
plt.title('Distribuição do CLV Bayesiano por Segmento', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_clv_dist, dpi=150)
plt.show()
plt.close()

# 4.3 Atualização Bayesiana: evidência acumulada
n_obs_range = np.arange(0, 21)
# Reduzido de 5 para 4 séries — SKILL: >4 linhas = spaghetti; usar ≤4 séries representativas
# Remove taxa=0.5 (central, redundante) para preservar legibilidade sem perder a mensagem analítica
recompra_rates = [0, 0.33, 0.67, 1.0]   # 4 taxas que cobrem todo o espaço de 0% a 100%
cores_rates = ['#C62828', '#FF9800', '#2E7D32', '#1565C0']
fig, ax = plt.subplots(figsize=(10, 5))
for taxa, cor in zip(recompra_rates, cores_rates):
    p_posts = []
    for n in n_obs_range:
        s   = int(round(n * taxa))
        a_n = alpha0 + s
        b_n = beta0 + (n - s)
        p_posts.append(a_n / (a_n + b_n))
    ax.plot(n_obs_range, p_posts, marker='o', markersize=4, color=cor,
            label=f'Taxa obs. = {taxa:.0%}')
ax.axhline(alpha0 / (alpha0 + beta0), color='black', linestyle='--',
           linewidth=1.5, label=f'Prior médio = {alpha0/(alpha0+beta0):.2f}')
ax.set_xlabel('Número de Observações', fontsize=11)
ax.set_ylabel('P(recompra) Posterior')
ax.set_title('Atualização Bayesiana: Convergência da Posterior', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_atualizacao, dpi=150)
plt.show()
plt.close()

# 4.4 Scatter interativo CLV × p_recompra × ticket_medio
rfm_plot = rfm_cal.sample(min(3000, len(rfm_cal)), random_state=42)
fig_inter = px.scatter(rfm_plot, x='p_recompra', y='CLV_Bayesiano',
                       color='Segmento_CLV', size=np.log1p(rfm_plot['ticket_medio']) * 2 + 2,
                       hover_data=['Customer ID', 'n_pedidos', 'ticket_medio', 'Rank_CLV'],
                       color_discrete_map={'Bronze': '#CD7F32', 'Prata': '#A8A9AD',
                                           'Ouro': '#FFD700', 'Diamante': '#0D47A1'},
                       title='CLV Bayesiano × Probabilidade de Recompra',
                       labels={'p_recompra': 'P(Recompra) Posterior', 'CLV_Bayesiano': 'CLV (R$)'})
fig_inter.write_html(file_interativo)
fig_inter.show()
print(f"Scatter interativo salvo: {file_interativo}")

# 4.5 3D: Recência × Frequência × CLV Bayesiano
fig_3d = px.scatter_3d(rfm_plot, x='recencia', y='n_pedidos', z='CLV_Bayesiano',
                        color='Segmento_CLV', opacity=0.75, hover_name='Customer ID',
                        color_discrete_map={'Bronze': '#CD7F32', 'Prata': '#A8A9AD',
                                            'Ouro': '#FFD700', 'Diamante': '#0D47A1'},
                        title='3D: Recência × Frequência × CLV Bayesiano',
                        labels={'recencia': 'Recência (dias)', 'n_pedidos': 'Frequência',
                                'CLV_Bayesiano': 'CLV (R$)'})
fig_3d.write_html(file_clv_3d)
fig_3d.show()
print(f"CLV 3D salvo: {file_clv_3d}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_posterior}")
print(f"  {file_clv_dist}")
print(f"  {file_atualizacao}")
print(f"  {file_interativo}")
print(f"  {file_clv_3d}")