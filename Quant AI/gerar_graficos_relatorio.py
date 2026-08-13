"""
gerar_graficos_relatorio.py

Módulo de Visualização e Relatórios Gráficos (Agente 8 — Validator Agent).

Gera 3 gráficos essenciais para a apresentação executiva e relatório de validação:
    1. Curva de Patrimônio Acumulado: Quant AI vs. Benchmark Ibovespa vs. CDI.
    2. Sub-gráfico de Drawdown (%): Profundidade e períodos de recuperação.
    3. Alocação Histórica de Pesos: Distribuição entre Ações (Equities) e Renda Fixa (CDI).

Salva as figuras geradas na pasta 'relatorios_graficos/'.

CORREÇÃO (confirmada por screenshot real do usuário): o gráfico de
alocação usava `colormap="tab20"` amostrado de forma contínua via
`DataFrame.plot(colormap=...)` -- com 11 séries (10 maiores + OUTROS),
essa amostragem contínua de uma paleta categórica de 20 cores pode
coincidentemente atribuir cores quase idênticas a duas séries diferentes
(CDI_RF e OUTROS saíram ambos em azul-marinho quase indistinguível).
Corrigido para um mapeamento de cores EXPLÍCITO por nome de coluna, com
CDI_RF e OUTROS forçados a cores deliberadamente distintas.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Paleta explícita -- evita a amostragem contínua de colormap categórico
# que causou a colisão de cores CDI_RF/OUTROS. CDI_RF fixo em azul-marinho
# (identidade Minerva IX); OUTROS fixo em cinza neutro, deliberadamente
# afastado de qualquer tom de azul usado pelos outros ativos.
COR_CDI = "#0B1B3D"      # Aegean Navy (paleta oficial Minerva IX)
COR_OUTROS = "#9AA3AF"   # cinza neutro -- claramente distinto do navy do CDI
PALETA_ATIVOS = [
    "#D4AF37", "#27AE60", "#E67E22", "#8E44AD", "#16A085",
    "#C0392B", "#2980B9", "#F39C12", "#7F8C8D", "#1ABC9C",
    "#D35400", "#34495E",
]


def _montar_mapa_cores(colunas: list[str]) -> dict[str, str]:
    """
    Monta o dicionário {coluna: cor} explicitamente -- CDI_RF e OUTROS
    recebem cores fixas e garantidamente distintas entre si; os demais
    ativos recebem cores de uma paleta qualitativa curada (não uma
    colormap contínua), ciclando se houver mais ativos que cores na
    paleta.
    """
    mapa = {}
    idx_paleta = 0
    for col in colunas:
        if col == "CDI_RF":
            mapa[col] = COR_CDI
        elif col == "OUTROS":
            mapa[col] = COR_OUTROS
        else:
            mapa[col] = PALETA_ATIVOS[idx_paleta % len(PALETA_ATIVOS)]
            idx_paleta += 1
    return mapa


def plotar_curva_patrimonio_e_drawdown(
    datas: pd.DatetimeIndex,
    retornos_carteira: np.ndarray,
    retornos_ibov: np.ndarray | None = None,
    retornos_cdi: np.ndarray | None = None,
    valor_inicial: float = 100_000.0,
    pasta_saida: str = "relatorios_graficos"
) -> str:
    """
    Gera a figura composta contendo:
      - Painel Superior: Curva de evolução de patrimônio (R$) em escala linear.
      - Painel Inferior: Curva de Drawdown (%) ao longo do tempo.
    """
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)

    patrimonio_carteira = valor_inicial * np.cumprod(1.0 + retornos_carteira)

    picos_carteira = np.maximum.accumulate(patrimonio_carteira)
    drawdown_carteira = (patrimonio_carteira - picos_carteira) / picos_carteira * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    ax1.plot(datas, patrimonio_carteira, label="Quant AI (Estratégia)", color="#1f77b4", linewidth=2.0)

    if retornos_ibov is not None and len(retornos_ibov) == len(retornos_carteira):
        patrimonio_ibov = valor_inicial * np.cumprod(1.0 + retornos_ibov)
        ax1.plot(datas, patrimonio_ibov, label="Ibovespa (Benchmark)", color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.85)

    if retornos_cdi is not None and len(retornos_cdi) == len(retornos_carteira):
        patrimonio_cdi = valor_inicial * np.cumprod(1.0 + retornos_cdi)
        ax1.plot(datas, patrimonio_cdi, label="CDI (Taxa Livre de Risco)", color="#2ca02c", linestyle=":", linewidth=1.5, alpha=0.85)

    # CORREÇÃO: título com datas reais da série, não hardcoded
    ano_inicio, ano_fim = datas.min().year, datas.max().year
    ax1.set_title(f"Evolução de Patrimônio Acumulado — Quant AI ({ano_inicio}-{ano_fim})", fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel("Patrimônio (R$)", fontsize=11)
    ax1.yaxis.set_major_formatter('R$ {x:,.0f}')
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(datas, drawdown_carteira, color="#d62728", linewidth=1.2, label="Drawdown Quant AI")
    ax2.fill_between(datas, drawdown_carteira, 0, color="#d62728", alpha=0.25)
    ax2.set_title("Maximum Drawdown (%)", fontsize=11, fontweight='bold', pad=6)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.yaxis.set_major_formatter('{x:.0f}%')
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    caminho_figura = os.path.join(pasta_saida, "curva_patrimonio_e_drawdown.png")
    plt.savefig(caminho_figura, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_figura


def plotar_alocacao_pesos_historica(
    df_pesos_historicos: pd.DataFrame,
    pasta_saida: str = "relatorios_graficos"
) -> str:
    """
    Gera o gráfico de área empilhada (stacked area chart) exibindo a evolução
    dos pesos alocados por classe/ativo ao longo das janelas de rebalanceamento.
    """
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    pesos_medios = df_pesos_historicos.mean().sort_values(ascending=False)
    top_cols = pesos_medios.head(10).index.tolist()

    df_plot = df_pesos_historicos[top_cols].copy()
    outros = df_pesos_historicos.drop(columns=top_cols).sum(axis=1)
    if outros.sum() > 0:
        df_plot["OUTROS"] = outros

    # CORREÇÃO (bug real, reproduzido: ArrayMemoryError de ~7 GiB): o índice
    # de datas aqui vem de janelas quadrimestrais (pd.date_range com
    # periods=N), cuja frequência implícita não é um alias padrão do
    # pandas. O locator automático de eixo de data do pandas tenta cobrir
    # CADA DIA do intervalo de 8 anos para formatar os ticks, tentando
    # alocar um array com centenas de milhões de posições. Convertendo o
    # índice para rótulos de texto ANTES de plotar evita esse locator por
    # completo -- o primeiro gráfico (preços diários, freq="B" limpa)
    # nunca teve esse problema, só este, com frequência irregular.
    datas_originais = df_plot.index
    df_plot = df_plot.copy()
    df_plot.index = [d.strftime("%Y-%m") for d in datas_originais]

    mapa_cores = _montar_mapa_cores(df_plot.columns.tolist())
    cores_ordenadas = [mapa_cores[c] for c in df_plot.columns]

    df_plot.plot(kind="area", stacked=True, ax=ax, color=cores_ordenadas, alpha=0.9)
    ax.set_xticks(range(0, len(df_plot), max(1, len(df_plot) // 12)))
    ax.set_xticklabels(
        [df_plot.index[i] for i in range(0, len(df_plot), max(1, len(df_plot) // 12))],
        rotation=45, ha="right",
    )

    ano_inicio, ano_fim = datas_originais.min().year, datas_originais.max().year
    ax.set_title(f"Evolução Histórica da Alocação de Pesos — Minerva IX ({ano_inicio}-{ano_fim})", fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel("Peso Alocado (%)", fontsize=11)
    ax.yaxis.set_major_formatter('{x:.0%}')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    caminho_figura = os.path.join(pasta_saida, "alocacao_pesos_historica.png")
    plt.savefig(caminho_figura, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_figura


def gerar_relatorio_grafico_demonstracao() -> list[str]:
    """Função utilitária para gerar e validar gráficos com dados de simulação."""
    datas = pd.date_range("2018-01-02", "2026-01-02", freq="B")
    rng = np.random.default_rng(42)

    ret_carteira = rng.normal(0.0006, 0.012, size=len(datas))
    ret_ibov = rng.normal(0.0004, 0.014, size=len(datas))
    ret_cdi = np.full(len(datas), 0.10 / 252.0)

    fig1 = plotar_curva_patrimonio_e_drawdown(datas, ret_carteira, ret_ibov, ret_cdi)

    datas_janelas = pd.date_range("2018-01-02", "2026-01-02", periods=26)
    ativos = ["CDI_RF", "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "RENT3", "EQTL3", "ABEV3", "SUZB3"]
    dados_pesos = rng.dirichlet(np.ones(len(ativos)), size=26)
    df_pesos = pd.DataFrame(dados_pesos, index=datas_janelas, columns=ativos)

    fig2 = plotar_alocacao_pesos_historica(df_pesos)

    return [fig1, fig2]


if __name__ == "__main__":
    print("\n========================================================")
    print("  GERANDO RELATÓRIOS GRÁFICOS DO QUANT AI (AGENTE 8)")
    print("========================================================\n")
    figuras = gerar_relatorio_grafico_demonstracao()
    for f in figuras:
        print(f" [SUCESSO] Figura gerada em: {f}")
