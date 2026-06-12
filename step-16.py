# =============================================================
# STEP 16 - Equações de Bellman e Programação Dinâmica
# -------------------------------------------------------------
# Objetivo: Resolver um problema de gestão de estoque via
#           Programação Dinâmica (indução retroativa). Usa dados
#           históricos de quantidade do Online Retail II para
#           estimar a distribuição de demanda. Define a equação
#           de Bellman V(t,s) = min_{a} [c(s,a) + E[V(t+1,s')]]
#           e encontra a política ótima de reposição.
# Entrada  : output/base_step_10_online_retail_raw.csv
# Saídas   : output/base_step_16_bellman_politica.csv
#            output/base_step_16_bellman_valor.png
#            output/base_step_16_bellman_convergencia.png
#            output/base_step_16_bellman_politica_mapa.png
#            output/base_step16_bellman_interativo.html
#            output/base_step16_bellman_3d.html
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

file_input         = 'output/base_step_02_online_retail.csv'
file_output        = 'output/base_step_16_bellman_politica.csv'
file_valor         = 'output/base_step_16_bellman_valor.png'
file_convergencia  = 'output/base_step_16_bellman_convergencia.png'
file_politica_mapa = 'output/base_step_16_bellman_politica_mapa.png'
file_interativo    = 'output/base_step16_bellman_interativo.html'
file_3d            = 'output/base_step16_bellman_3d.html'

os.makedirs(os.path.dirname(file_output), exist_ok=True)

# ---------- 1. Carregar dados e estimar distribuição de demanda ----------
df_raw = pd.read_csv(file_input, sep=';', encoding='utf-8-sig', decimal=',')
print(f"Linhas originais: {len(df_raw)}")

df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]
# Normalização defensiva: Online Retail II pode salvar 'InvoiceDate' ou 'Invoice Date'
if 'InvoiceDate' in df.columns and 'Invoice Date' not in df.columns:
    df = df.rename(columns={'InvoiceDate': 'Invoice Date'})
df = df[df['Quantity'].astype(float) > 0].copy()
df['Quantity'] = df['Quantity'].astype(float).astype(int)

# Estimar distribuição de demanda diária por produto
# Usar o produto mais transacionado como DMU representativo
top_produto = df.groupby('StockCode')['Quantity'].sum().idxmax()
df_prod     = df[df['StockCode'] == top_produto].copy()
df_prod['Invoice Date'] = pd.to_datetime(df_prod['Invoice Date'])
df_prod['Data']         = df_prod['Invoice Date'].dt.date
demanda_diaria = df_prod.groupby('Data')['Quantity'].sum()

print(f"\nProduto selecionado : {top_produto}")
print(f"Dias com venda       : {len(demanda_diaria)}")
print(f"Demanda média/dia    : {demanda_diaria.mean():.1f}")
print(f"Demanda max/dia      : {demanda_diaria.max()}")

# Truncar demanda para estados manejáveis
D_MAX     = int(min(demanda_diaria.quantile(0.95), 50))
demanda_v = np.clip(demanda_diaria.values, 0, D_MAX).astype(int)
probs_d   = np.bincount(demanda_v, minlength=D_MAX + 1) / len(demanda_v)
print(f"D_MAX (95p)          : {D_MAX}")

# ---------- 2. Programação Dinâmica — Equação de Bellman ----------
# Estados:  s ∈ {0, 1, ..., S_MAX}  (nível de estoque)
# Ações:    a ∈ {0, 1, ..., A_MAX}  (quantidade a pedir)
# Transição: s' = max(0, s - d) + a    (demanda d ~ probs_d)
# Custo imediato: c(s, a) = c_pedido * I(a>0) + c_estoque * s + c_falta * max(0, d-s)
# Equação de Bellman: V*(s) = min_a { c(s,a) + γ * E_d[V*(s')] }

S_MAX   = 100      # estoque máximo
A_MAX   = 50       # pedido máximo por período
T       = 30       # horizonte de planejamento (dias)
GAMMA   = 0.95     # fator de desconto
C_PED   = 20.0     # custo fixo de pedido
C_EST   = 0.5      # custo de manutenção de estoque por unidade/dia
C_FALT  = 5.0      # custo de falta por unidade
C_UNIT  = 2.0      # custo unitário de compra

estados = np.arange(S_MAX + 1)   # 0 a S_MAX
acoes   = np.arange(A_MAX + 1)   # 0 a A_MAX
demandas = np.arange(D_MAX + 1)  # 0 a D_MAX

# Indução retroativa: V_{T} = 0 (sem custo no final)
V = np.zeros(S_MAX + 1)          # função valor
politica = np.zeros(S_MAX + 1, dtype=int)  # política ótima
hist_V   = [V.copy()]
hist_pol = [politica.copy()]
del_V    = []

for t in range(T - 1, -1, -1):
    V_novo  = np.full(S_MAX + 1, np.inf)
    pol_novo = np.zeros(S_MAX + 1, dtype=int)

    for s in estados:
        melhor_custo = np.inf
        melhor_a     = 0
        for a in acoes:
            s_apos = min(s + a, S_MAX)   # estoque após receber pedido

            # Custo imediato
            custo_imediato = C_UNIT * a + (C_PED if a > 0 else 0) + C_EST * s_apos

            # Custo esperado do próximo período
            custo_futuro = 0.0
            for d, p_d in zip(demandas, probs_d):
                if p_d < 1e-12:
                    continue
                s_prox      = max(0, s_apos - d)
                custo_falta = C_FALT * max(0, d - s_apos)
                custo_futuro += p_d * (custo_falta + GAMMA * V[s_prox])

            custo_total = custo_imediato + custo_futuro
            if custo_total < melhor_custo:
                melhor_custo = custo_total
                melhor_a     = a

        V_novo[s]  = melhor_custo
        pol_novo[s] = melhor_a

    delta = np.max(np.abs(V_novo - V))
    del_V.append(delta)
    V        = V_novo
    politica = pol_novo
    hist_V.append(V.copy())
    hist_pol.append(politica.copy())

    if t % 5 == 0:
        print(f"t={t:2d} | ΔV={delta:.4f} | V(0)={V[0]:.2f} | V(S_MAX)={V[S_MAX]:.2f}")

print(f"\nPolítica ótima (amostra):")
for s in [0, 10, 20, 30, 50, 80, 100]:
    print(f"  Estoque={s:3d} → Pedir={politica[s]:3d} unidades")

# Ponto de reposição (s de pedido) e quantidade ótima
s_reposicao = next((s for s in estados if politica[s] > 0), 0)
print(f"\nPonto de reposição: s*={s_reposicao}, a*={politica[s_reposicao]}")

# ---------- 3. Salvar resultados ----------
df_resultado = pd.DataFrame({
    'Estoque_s':   estados,
    'Acao_otima':  politica,
    'Valor_V':     np.round(V, 4),
    'Tipo_Acao':   ['Repor' if politica[s] > 0 else 'Manter' for s in estados],
})
df_resultado.to_csv(file_output, sep=';', index=False, encoding='utf-8-sig', decimal=',')
print(f"\nArquivo salvo: {file_output}")

# ---------- 4. Visualizações ----------

# 4.1 Função Valor V*(s)
plt.figure(figsize=(10, 5))
plt.plot(estados, V, color='#1565C0', linewidth=2, label='V*(s) — Valor ótimo')
plt.axvline(s_reposicao, color='#C62828', linewidth=1.5, linestyle='--',
            label=f'Ponto de reposição s*={s_reposicao}')
plt.xlabel('Nível de Estoque (s)', fontsize=11)
plt.ylabel('Custo Esperado Descontado V*(s)')
plt.title('Função Valor Ótima — Programação Dinâmica (Bellman)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_valor, dpi=150)
plt.show()
plt.close()

# 4.2 Convergência da função valor ao longo das iterações (retroativas)
plt.figure(figsize=(10, 5))
plt.plot(range(len(del_V)), del_V, color='#C62828', linewidth=2, marker='o', markersize=3)
plt.axhline(0.01, color='grey', linewidth=1, linestyle='--', label='Tolerância 0.01')
plt.yscale('log')
plt.xlabel('Iteração (retroativa)', fontsize=11)
plt.ylabel('ΔV = max|V_novo - V_ant| (log)')
plt.title('Convergência da Indução Retroativa — Equação de Bellman', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(file_convergencia, dpi=150)
plt.show()
plt.close()

# 4.3 Política ótima e função valor — subplots separados
# SKILL: twinx (eixo duplo) PROIBIDO → dividir em 2 subplots independentes
cores_pol = ['#C62828' if politica[s] > 0 else '#1565C0' for s in estados]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.bar(estados, politica, color=cores_pol, alpha=0.7, label='Ação ótima a*(s)')
ax1.axvline(s_reposicao, color='gold', linewidth=1.5, linestyle='--', label=f's*={s_reposicao}')
ax1.set_ylabel('Quantidade a Pedir a*(s)', fontsize=10)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, axis='y', alpha=0.35)

ax2.plot(estados, V, color='#1565C0', linewidth=2, label='V*(s) — Custo esperado descontado')
ax2.axvline(s_reposicao, color='gold', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Nível de Estoque (s)', fontsize=11)
ax2.set_ylabel('Custo Esperado V*(s)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.35)

fig.suptitle('Política Ótima a*(s) e Função Valor V*(s) — Bellman', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(file_politica_mapa, dpi=150)
plt.show()
plt.close()

# 4.4 Interativo: função valor e política ótima — subplots separados (Plotly)
# SKILL: secondary_y (eixo duplo) PROIBIDO → make_subplots rows=2
fig_inter = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=['Ação ótima a*(s)', 'Função Valor V*(s)'],
                          vertical_spacing=0.12)
fig_inter.add_trace(go.Bar(x=estados.tolist(), y=politica.tolist(), name='Ação ótima a*(s)',
                           marker_color=['#C62828' if politica[s] > 0 else '#90CAF9' for s in estados],
                           opacity=0.7), row=1, col=1)
fig_inter.add_trace(go.Scatter(x=estados.tolist(), y=V.tolist(), name='V*(s)',
                               line=dict(color='#1565C0', width=2.5),
                               mode='lines'), row=2, col=1)
fig_inter.update_xaxes(title_text='Nível de Estoque (s)', row=2, col=1)
fig_inter.update_yaxes(title_text='Quantidade a Pedir', row=1, col=1)
fig_inter.update_yaxes(title_text='Custo Esperado V*(s)', row=2, col=1)
fig_inter.update_layout(title='Política Ótima a*(s) e Função Valor V*(s) — Interativo',
                        height=600)
fig_inter.write_html(file_interativo)
fig_inter.show()
print(f"Interativo salvo: {file_interativo}")

# 4.5 3D: Evolução de V(s) ao longo das iterações retroativas
iters_plot = np.arange(0, len(hist_V), max(1, len(hist_V) // 10))
Z_surf = np.array([hist_V[i] for i in iters_plot])
fig_3d = go.Figure(data=[go.Surface(
    z=Z_surf, x=estados.tolist(), y=iters_plot.tolist(),
    colorscale='Viridis', opacity=0.85,
)])
fig_3d.update_layout(
    title='Evolução da Função Valor V(s) — Indução Retroativa (3D)',
    scene=dict(xaxis_title='Estoque (s)', yaxis_title='Iteração', zaxis_title='V(s)')
)
fig_3d.write_html(file_3d)
fig_3d.show()
print(f"Gráfico 3D salvo: {file_3d}")

print("\nArquivos salvos:")
print(f"  {file_output}")
print(f"  {file_valor}")
print(f"  {file_convergencia}")
print(f"  {file_politica_mapa}")
print(f"  {file_interativo}")
print(f"  {file_3d}")