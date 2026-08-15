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

# CORREÇÃO: a paleta anterior tinha dois pares de cores próximas demais
# (UGPA3/BBDC4, os dois em laranja/âmbar; WEGE3/OUTROS, os dois em cinza)
# -- confirmado visualmente pelo usuário num gráfico real com 10 ativos.
# Esta paleta foi selecionada COMPUTACIONALMENTE (não à mão): todo par
# de cores aqui -- incluindo contra COR_CDI e COR_OUTROS -- tem distância
# euclidiana em RGB >= 70 (ver _checar_paleta_distinguivel), o que já
# pegou colisões que uma escolha manual anterior não tinha percebido
# (ex: ouro vs. laranja, azul vs. índigo).
PALETA_ATIVOS = [
    "#D4AF37",  # ouro (identidade Minerva IX)
    "#4E79A7",  # azul
    "#59A14F",  # verde
    "#E15759",  # vermelho
    "#FF9DA7",  # rosa
    "#9C755F",  # marrom
    "#C77DFF",  # violeta
    "#6B4226",  # marrom escuro
    "#06D6A0",  # verde-menta
    "#FFD166",  # amarelo
    "#7209B7",  # roxo escuro
]


def _distancia_rgb(cor_hex_a: str, cor_hex_b: str) -> float:
    """Distância euclidiana simples no espaço RGB -- suficiente para
    detectar cores 'perigosamente parecidas', não precisa da precisão
    perceptual de Lab/Delta-E para esse propósito."""
    a = tuple(int(cor_hex_a.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    b = tuple(int(cor_hex_b.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _checar_paleta_distinguivel(cores_em_uso: list[str], limite_minimo: float = 70.0) -> None:
    """
    Roda ANTES de qualquer gráfico usar as cores -- avisa se duas cores
    atribuídas na mesma legenda ficaram próximas demais (o mesmo tipo de
    bug já confirmado duas vezes nesta base: CDI_RF/OUTROS, depois
    UGPA3/BBDC4 e WEGE3/OUTROS). limite_minimo=60 é conservador o
    suficiente para pegar tons muito próximos sem disparar por cores
    que só compartilham a mesma família de matiz mas são distinguíveis.
    """
    for i in range(len(cores_em_uso)):
        for j in range(i + 1, len(cores_em_uso)):
            dist = _distancia_rgb(cores_em_uso[i], cores_em_uso[j])
            if dist < limite_minimo:
                print(f"[AVISO PALETA] Cores {cores_em_uso[i]} e {cores_em_uso[j]} estão "
                      f"visualmente próximas (distância RGB={dist:.1f}, limite={limite_minimo}) "
                      f"-- risco de confusão na legenda.")


# CORREÇÃO: antes, a cor de cada ativo vinha da posição dele no ranking
# de peso médio DAQUELA rodada específica -- o mesmo ativo (ex: WEGE3)
# podia sair rosa numa figura e dourado em outra, só porque o ranking
# de topo mudou entre execuções. Isso atrapalha comparar gráficos lado
# a lado no relatório. Agora, os tickers mais frequentes do projeto têm
# cor FIXA, atribuída uma vez, nunca dependente do ranking da rodada.
CORES_TICKERS_CONHECIDOS: dict[str, str] = {
    "PETR4": "#D4AF37", "PETR3": "#4E79A7", "VALE3": "#59A14F", "ITUB4": "#E15759",
    "BBDC4": "#FF9DA7", "BBAS3": "#9C755F", "WEGE3": "#C77DFF", "ABEV3": "#6B4226",
    "B3SA3": "#06D6A0", "ITSA4": "#FFD166", "SUZB3": "#7209B7", "RENT3": "#B5179E",
    "EQTL3": "#4361EE", "GGBR4": "#4CC9F0", "RAIL3": "#F72585", "PRIO3": "#780000",
    "ELET3": "#C1121F", "MRVE3": "#FDF0D5",
}


def _cor_hash_estavel(ticker: str, cores_ja_usadas: set[str]) -> str:
    """
    Fallback para ticker fora de CORES_TICKERS_CONHECIDOS: escolhe uma
    cor da paleta de forma determinística (hash estável via hashlib, não
    o hash() nativo do Python -- esse é randomizado por processo, não
    serviria para manter a MESMA cor entre execuções diferentes). Evita
    colisão com cores já em uso nesta mesma figura via linear probing.
    """
    import hashlib
    indice_base = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % len(PALETA_ATIVOS)
    for offset in range(len(PALETA_ATIVOS)):
        candidata = PALETA_ATIVOS[(indice_base + offset) % len(PALETA_ATIVOS)]
        if candidata not in cores_ja_usadas:
            return candidata
    return PALETA_ATIVOS[indice_base]  # paleta esgotada -- aceita repetição como último recurso


def _montar_mapa_cores(colunas: list[str]) -> dict[str, str]:
    """
    Monta o dicionário {coluna: cor}. CDI_RF e OUTROS têm cor fixa e
    distinta entre si. Tickers conhecidos (CORES_TICKERS_CONHECIDOS) têm
    cor fixa, estável entre QUALQUER gráfico e QUALQUER rodada -- não
    depende mais do ranking de peso da execução atual. Tickers não
    catalogados caem no hash estável determinístico.
    """
    mapa = {}
    cores_usadas: set[str] = {COR_CDI, COR_OUTROS}

    for col in colunas:
        if col == "CDI_RF":
            mapa[col] = COR_CDI
        elif col == "OUTROS":
            mapa[col] = COR_OUTROS
        elif col in CORES_TICKERS_CONHECIDOS:
            mapa[col] = CORES_TICKERS_CONHECIDOS[col]
            cores_usadas.add(mapa[col])
        else:
            mapa[col] = _cor_hash_estavel(col, cores_usadas)
            cores_usadas.add(mapa[col])

    _checar_paleta_distinguivel(list(mapa.values()))
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
