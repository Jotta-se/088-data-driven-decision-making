# =============================================================
# STEP 11 - Teorema de Aproximação de Blackwell
# -------------------------------------------------------------
# Objetivo: Demonstrar o Teorema de Aproximação de Blackwell em
#           um jogo sequencial de recomendação de produtos.
#           O agente escolhe ações (categorias de produto) e
#           recebe respostas adversariais (rejeição/aceitação).
#           O algoritmo garante que o payoff médio converge ao
#           conjunto-alvo C. Mede o arrependimento externo e
#           interno ao longo das rodadas.
# Entrada  : output/base_step_10_online_retail_raw.csv
# Saídas   : output/base_step_11_blackwell_convergencia.csv
#            output/base_step_11_blackwell_regret.png
#            output/base_step_11_blackwell_trajetoria.png
#            output/base_step_11_blackwell_payoffs.png
#            output/base_step11_blackwell_convergencia_interativa.html
#            output/base_step11_blackwell_regret_interativo.html
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

file_input              = 'output/base_step_02_online_retail.csv'
file_output             = 'output/base_step_11_blackwell_convergencia.csv'
file_regret             = 'output/base_step_11_blackwell_regret.png'
file_trajetoria         = 'output/base_step_11_blackwell_trajetoria.png'
file_payoffs            = 'output/base_step_11_blackwell_payoffs.png'
file_convergencia_inter = 'output/base_step11_blackwell_convergencia_interativa.html'
file_regret_inter       = 'output/base_step11_blackwell_regret_interativo.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Carregar dados ----------
df_raw = pd.read_csv(file_input, sep=';', encoding='utf-8-sig', decimal=',')
print(f"Linhas originais: {len(df_raw)}")

df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]
df = df[df['Quantity'].astype(float) > 0].copy()
df = df[df['Price'].astype(float) > 0].copy()
df['Revenue'] = df['Quantity'].astype(float) * df['Price'].astype(float)

# Extrair top categorias (primeiros 4 chars do StockCode como proxy de categoria)
df['Categoria'] = df['StockCode'].astype(str).str[:2]
top_cats = df.groupby('Categoria')['Revenue'].sum().nlargest(6).index.tolist()
df_cats  = df[df['Categoria'].isin(top_cats)].copy()
print(f"Top categorias usadas: {top_cats}")
print(f"Transações nas categorias: {len(df_cats)}")

# ---------- 2. Blackwell Approachability ----------
# Configuração do jogo:
#   Ações do agente (recomendador):   K categorias de produto
#   Respostas do ambiente (adversário): aceitar (1) ou rejeitar (0)
#   Payoff: vetor 2D (receita normalizada, taxa de aceitação)
#   Conjunto-alvo C: {(r, a) | r >= r_min, a >= a_min}  (cone positivo)

np.random.seed(2024)
K = len(top_cats)   # número de ações
T = 1000            # número de rodadas

# Estimar payoff médio e variação por categoria dos dados reais
stats_cat = df_cats.groupby('Categoria').agg(
    rev_med = ('Revenue', 'mean'),
    qtd_med = ('Quantity', 'mean')
).reindex(top_cats).fillna(0)

rev_norm = (stats_cat['rev_med'] / stats_cat['rev_med'].max()).values  # payoff dim 1
acc_norm = (stats_cat['qtd_med'] / stats_cat['qtd_med'].max()).values  # payoff dim 2

# Conjunto-alvo C: ponto de referência (média das melhores ações)
c_ref = np.array([rev_norm.mean(), acc_norm.mean()])

# Algoritmo de Blackwell:
# Distribuição mista p_t sobre K ações
# Atualização: p_{t+1} proporcional à projeção do payoff médio em C
payoff_medio  = np.zeros(2)
distancia_C   = np.zeros(T)
regret_externo = np.zeros(T)
payoffs_hist  = np.zeros((T, 2))
acoes_hist    = np.zeros(T, dtype=int)
p             = np.ones(K) / K   # uniforme inicial

for t in range(T):
    # Agente escolhe ação segundo distribuição mista p
    acao = np.random.choice(K, p=p)
    acoes_hist[t] = acao

    # Adversário escolhe resposta (ruído realístico baseado em dados)
    ruido = np.random.normal(0, 0.05, 2)
    payoff_t = np.array([rev_norm[acao], acc_norm[acao]]) + ruido
    payoff_t = np.clip(payoff_t, 0, 1)
    payoffs_hist[t] = payoff_t

    # Atualizar payoff médio
    payoff_medio = (payoff_medio * t + payoff_t) / (t + 1)

    # Distância ao conjunto-alvo C (projeção no ponto mais próximo acima de c_ref)
    proj = np.maximum(payoff_medio, c_ref)
    distancia_C[t] = np.linalg.norm(payoff_medio - proj)

    # Arrependimento externo: diferença para a melhor ação fixa
    melhor_acao_payoff = max(rev_norm[k] + acc_norm[k] for k in range(K)) / 2
    regret_externo[t]  = melhor_acao_payoff - (payoff_medio[0] + payoff_medio[1]) / 2

    # Atualizar distribuição: direção de Blackwell
    direcao = c_ref - payoff_medio
    if np.linalg.norm(direcao) > 1e-9:
        # Escolha a ação cujo payoff maximiza o produto interno com a direção
        scores = np.array([np.dot(direcao, [rev_norm[k], acc_norm[k]]) for k in range(K)])
        scores = scores - scores.min()
        s_sum = scores.sum()
        p = scores / s_sum if s_sum > 1e-12 else np.ones(K) / K
    else:
        p = np.ones(K) / K
    # Garantir soma exatamente 1.0 (precisão float)
    p = p / p.sum()

print(f"\nDistância final ao conjunto C: {distancia_C[-1]:.6f}")
print(f"Arrependimento externo final:   {regret_externo[-1]:.6f}")
print(f"Payoff médio convergido:        R={payoff_medio[0]:.4f}, A={payoff_medio[1]:.4f}")
print(f"Ponto de referência C:          R={c_ref[0]:.4f}, A={c_ref[1]:.4f}")

# Frequência de ações
freq_acoes = pd.Series(acoes_hist).value_counts().sort_index()
print(f"\nFrequência de ações (categoria):")
for i, cat in enumerate(top_cats):
    print(f"  {cat}: {freq_acoes.get(i, 0)} vezes ({freq_acoes.get(i,0)/T*100:.1f}%)")

# ---------- 3. Salvar resultados ----------
df_resultado = pd.DataFrame({
    'Rodada':           range(1, T + 1),
    'Acao':             [top_cats[a] for a in acoes_hist],
    'Payoff_Receita':   np.round(payoffs_hist[:, 0], 6),
    'Payoff_Aceitacao': np.round(payoffs_hist[:, 1], 6),
    'Payoff_Medio_R':   np.round(np.cumsum(payoffs_hist[:, 0]) / (np.arange(T) + 1), 6),
    'Payoff_Medio_A':   np.round(np.cumsum(payoffs_hist[:, 1]) / (np.arange(T) + 1), 6),
    'Distancia_C':      np.round(distancia_C, 6),
    'Regret_Externo':   np.round(regret_externo, 6),
})
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Arrependimento externo ao longo das rodadas
rodadas = np.arange(1, T + 1)
bound_teorico = 1 / np.sqrt(rodadas)   # O(1/√T)

plt.figure(figsize=(11, 5))
plt.plot(rodadas, regret_externo, color='#1565C0', linewidth=1.2, label='Regret Externo (real)', alpha=0.8)
plt.plot(rodadas, bound_teorico, color='#C62828', linewidth=1.5, linestyle='--',
         label='Limite teórico O(1/√T)')
plt.axhline(0, color='black', linewidth=0.7)
plt.xlabel('Rodada (t)', fontsize=11)
plt.ylabel('Arrependimento Externo')
plt.title('Convergência do Arrependimento — Blackwell Approachability', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_regret, dpi=150)
plt.show()
plt.close()

# 4.2 Trajetória do payoff médio convergindo ao conjunto C
plt.figure(figsize=(8, 7))
pm_R = df_resultado['Payoff_Medio_R'].values
pm_A = df_resultado['Payoff_Medio_A'].values
plt.plot(pm_R, pm_A, color='#1565C0', linewidth=1, alpha=0.7, label='Trajetória payoff médio')
plt.scatter(pm_R[0], pm_A[0], s=150, color='#2E7D32', zorder=5, label='Início')
plt.scatter(pm_R[-1], pm_A[-1], s=150, color='#C62828', marker='*', zorder=6, label='Final')
plt.scatter(c_ref[0], c_ref[1], s=200, color='gold', marker='D', edgecolors='black',
            linewidths=1.2, zorder=7, label='Ponto alvo C')
plt.axhline(c_ref[1], color='grey', linewidth=0.8, linestyle=':', alpha=0.6)
plt.axvline(c_ref[0], color='grey', linewidth=0.8, linestyle=':', alpha=0.6)
plt.fill_between([c_ref[0], 1.1], c_ref[1], 1.1, alpha=0.08, color='green', label='Conjunto C')
plt.xlabel('Payoff Médio — Receita', fontsize=11)
plt.ylabel('Payoff Médio — Aceitação', fontsize=11)
plt.title('Trajetória de Convergência ao Conjunto-Alvo C', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.35)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(file_trajetoria, dpi=150)
plt.show()
plt.close()

# 4.3 Frequência de ações e distância ao conjunto C
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(top_cats, [freq_acoes.get(i, 0) / T * 100 for i in range(K)], color='#1565C0')
axes[0].set_xlabel('Categoria de Produto')
axes[0].set_ylabel('Frequência (%)')
axes[0].set_title('Distribuição de Ações — Blackwell')
axes[0].grid(True, axis='y', alpha=0.4)

axes[1].plot(rodadas, distancia_C, color='#C62828', linewidth=1.2)
axes[1].plot(rodadas, bound_teorico * 0.5, color='#FF9800', linewidth=1.5,
             linestyle='--', label='O(1/√T) / 2')
axes[1].set_xlabel('Rodada (t)')
axes[1].set_ylabel('Distância ao Conjunto C')
axes[1].set_title('Distância ao Conjunto-Alvo C ao Longo do Tempo')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.35)
plt.suptitle('Blackwell Approachability — Diagnóstico', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(file_payoffs, dpi=150)
plt.show()
plt.close()

# 4.4 Convergência interativa (Plotly)
fig_inter = go.Figure()
fig_inter.add_trace(go.Scatter(x=rodadas.tolist(), y=regret_externo.tolist(),
                               mode='lines', name='Regret Externo',
                               line=dict(color='#1565C0', width=1.5)))
fig_inter.add_trace(go.Scatter(x=rodadas.tolist(), y=bound_teorico.tolist(),
                               mode='lines', name='Limite O(1/√T)',
                               line=dict(color='#C62828', width=2, dash='dash')))
fig_inter.add_trace(go.Scatter(x=rodadas.tolist(), y=distancia_C.tolist(),
                               mode='lines', name='Distância a C',
                               line=dict(color='#2E7D32', width=1.5)))
fig_inter.update_layout(title='Blackwell Approachability — Convergência Interativa',
                        xaxis_title='Rodada', yaxis_title='Valor')
fig_inter.write_html(file_convergencia_inter)
fig_inter.show()
print(f"Convergência interativa salva: {file_convergencia_inter}")

# 4.5 Trajetória interativa no espaço de payoff
fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=pm_R.tolist(), y=pm_A.tolist(), mode='lines+markers',
                             marker=dict(size=3, color=np.arange(T), colorscale='Blues'),
                             name='Trajetória payoff médio',
                             line=dict(color='#1565C0', width=1)))
fig_reg.add_trace(go.Scatter(x=[c_ref[0]], y=[c_ref[1]], mode='markers',
                             marker=dict(size=15, symbol='diamond', color='gold',
                                         line=dict(color='black', width=2)),
                             name='Conjunto-alvo C'))
fig_reg.update_layout(title='Trajetória de Convergência ao Conjunto C — Interativo',
                      xaxis_title='Payoff Médio (Receita)', yaxis_title='Payoff Médio (Aceitação)')
fig_reg.write_html(file_regret_inter)
fig_reg.show()
print(f"Trajetória interativa salva: {file_regret_inter}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_regret}")
print(f"  {file_trajetoria}")
print(f"  {file_payoffs}")
print(f"  {file_convergencia_inter}")
print(f"  {file_regret_inter}")