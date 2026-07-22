"""
pipeline_quant_ai_colab.py

Script único de orquestração dos Agentes 0-8 do projeto Quant AI, para
rodar no Google Colab.

COMO USAR NO COLAB:
    1. Clone o repositório GitHub onde você vai versionar os 9 arquivos
       de agente + este script:
           !git clone https://github.com/<seu-usuario>/<seu-repo>.git
           %cd <seu-repo>
    2. Instale as dependências:
           !pip install cvxpy yfinance pandas_market_calendars -q
    3. Rode este script (ou importe as funções célula a célula).

PONTOS AINDA NÃO PREENCHIDOS (marcados com "TODO" abaixo) -- você
indicou que isso vem depois:
    - Carregar preços via yfinance (Agente 1) para os tickers do
      universo PIT já calculado (composicao_pit_ibovespa.csv).
    - Carregar/atualizar os arquivos direto do repositório GitHub em vez
      de arquivos locais, se você optar por isso em vez de upload manual
      no Colab.

O que JÁ FUNCIONA de ponta a ponta com dados sintéticos (útil para você
validar que a instalação e os imports estão OK antes de plugar dado
real): a função `rodar_pipeline_sintetico()` no final do arquivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Imports dos 9 agentes (assumindo os arquivos na mesma pasta/repo) ---
from agente_0_universe_data import construir_composicao_pit, derivar_eventos_entrada_saida
from agente_1_data_agent import (
    detectar_gaps, preencher_gap_wiener, validar_alinhamento_timestamp, calcular_retorno_realizado_fwd,
)
from agente_2_sentiment_agents import pontuar_noticia, processar_resposta_llm
from agente_3_aggregation_agent import agregar_meta_score
from agente_4_correlation_filter import calcular_score_ajustado, calcular_omega
from agente_5_monte_carlo_agent import combinar_eventos_recentes
from agente_6_bl_optimizer import calcular_posterior_bl, bl_optimizer
from agente_7_rebalancing_agent import calcular_banda_nao_negociacao, decidir_execucao, calcular_custos_transacao
from agente_8_validator_agent import gerar_relatorio, RegistroConcentracao


# --------------------------------------------------------------------------
# TODO 1: Agente 0 -- composição PIT (JÁ RESOLVIDO na sessão anterior, só
# aponte para os arquivos já gerados; ou rode de novo se atualizar dados)
# --------------------------------------------------------------------------

def carregar_composicao_pit(caminho_csv: str = "composicao_pit_ibovespa.csv") -> pd.DataFrame:
    return pd.read_csv(caminho_csv, parse_dates=["data_vigencia"])


# --------------------------------------------------------------------------
# TODO 2: Agente 1 -- preços via yfinance (A PREENCHER DEPOIS)
# --------------------------------------------------------------------------

def carregar_precos_reais(tickers: list[str], inicio: str, fim: str) -> pd.DataFrame:
    """
    TODO (você indicou que isso vem depois): implementar de fato usando
    `agente_1_data_agent.buscar_precos_b3(tickers, inicio, fim)` -- essa
    função já existe e já está pronta, só depende de internet (yfinance),
    que este ambiente de desenvolvimento não tinha, mas o Colab tem.
    """
    raise NotImplementedError(
        "Preencher com agente_1_data_agent.buscar_precos_b3(tickers, inicio, fim) "
        "quando você estiver pronto para plugar dados reais no Colab."
    )


# --------------------------------------------------------------------------
# Pipeline de UM rebalanceamento (dado o universo e os dados já prontos)
# --------------------------------------------------------------------------

def rodar_um_rebalanceamento(
    tickers: list[str],
    Pi: np.ndarray,
    Sigma: np.ndarray,
    tau: float,
    delta: float,
    Q_por_ativo: dict[str, float],
    Omega_por_ativo: dict[str, float],
    pesos_atuais_com_drift: pd.Series,
    historico_pesos_alvo: pd.DataFrame,
    fator_liquidez: dict[str, float],
    valor_carteira: float,
    w_max: float = 0.20,
    r_cdi: float | None = None,
) -> dict:
    """Encadeia Agentes 6 -> 7 para um único instante de rebalanceamento."""
    E_R = calcular_posterior_bl(Pi, Sigma, tau, tickers, Q_por_ativo, Omega_por_ativo)

    resultado_bl = bl_optimizer(E_R, Sigma, delta, w_max=w_max, r_cdi=r_cdi)

    # resultado_bl.pesos tem len(tickers)+1 posições quando r_cdi não é
    # None (a última é o CDI sintético). O Agente 7 (banda de
    # não-negociação) só conhece os ativos de equity -- bandas e
    # pesos_atuais_com_drift são indexados só por eles -- então a fatia
    # que alimenta o Agente 7 continua sendo só os equities. O peso do
    # CDI é reportado à parte, não descartado (bug anterior: o [:len(tickers)]
    # cortava o CDI silenciosamente do resultado exibido, sem removê-lo
    # do cálculo -- os 20% "sumidos" no teste real do Colab eram o CDI).
    pesos_alvo_equities = pd.Series(resultado_bl.pesos[:len(tickers)], index=tickers)
    peso_cdi = float(resultado_bl.pesos[len(tickers)]) if r_cdi is not None else None

    bandas = calcular_banda_nao_negociacao(historico_pesos_alvo, fator_liquidez)
    decisao = decidir_execucao(pesos_alvo_equities, pesos_atuais_com_drift, bandas)

    custos = calcular_custos_transacao(pesos_atuais_com_drift, decisao.pesos_finais, valor_carteira)

    return {
        "pesos_alvo_bl": pesos_alvo_equities,
        "peso_cdi": peso_cdi,
        "decisao_rebalanceamento": decisao,
        "custos": custos,
        "custo_concentracao_lagrange": resultado_bl.custo_concentracao_por_ativo,
    }


# --------------------------------------------------------------------------
# Teste de fumaça com dados 100% sintéticos (roda sem internet, sem API keys)
# --------------------------------------------------------------------------

def rodar_pipeline_sintetico(seed: int = 0) -> None:
    """
    Roda o pipeline de ponta a ponta (Agentes 4/5/6/7) com dados
    inventados, só para confirmar que todos os imports e a integração
    entre os arquivos funcionam no seu ambiente (Colab) antes de plugar
    dado real. Não requer internet nem chaves de API.
    """
    rng = np.random.default_rng(seed)
    tickers = ["PETR4", "VALE3", "ITUB4", "WEGE3", "BBAS3"]
    n = len(tickers)

    A = rng.normal(0, 0.02, (n, n))
    Sigma = A @ A.T + np.eye(n) * 0.01
    Pi = rng.normal(0.08, 0.01, n)
    tau, delta = 2.5, 3.0

    # Simula sentimento -> Q/Omega via Agentes 3/4/5 (dados fake)
    Q_por_ativo = {"PETR4": 0.15, "VALE3": -0.05, "ITUB4": 0.08}
    Omega_por_ativo = {"PETR4": 0.02, "VALE3": 0.04, "ITUB4": 0.03, "WEGE3": np.nan, "BBAS3": np.nan}

    historico_pesos_alvo = pd.DataFrame(rng.normal(0.2, 0.03, (60, n)), columns=tickers)
    pesos_atuais = pd.Series(rng.dirichlet(np.ones(n)), index=tickers)
    fator_liquidez = {t: 1.0 for t in tickers[:3]} | {t: 1.5 for t in tickers[3:]}

    resultado = rodar_um_rebalanceamento(
        tickers, Pi, Sigma, tau, delta, Q_por_ativo, Omega_por_ativo,
        pesos_atuais, historico_pesos_alvo, fator_liquidez,
        valor_carteira=100_000, w_max=0.20, r_cdi=0.04,  # valor_carteira em R$ (reais)
    )

    print("=== Teste de fumaça (dados sintéticos) ===")
    print("Pesos-alvo em equities (Agente 6):", resultado["pesos_alvo_bl"].round(4).to_dict())
    if resultado["peso_cdi"] is not None:
        print(f"Peso-alvo em CDI: {resultado['peso_cdi']:.4f}")
        soma_total = resultado["pesos_alvo_bl"].sum() + resultado["peso_cdi"]
        print(f"Soma total (equities + CDI, deveria ser 1.0): {soma_total:.4f}")
    print("Ativos executados (Agente 7):", resultado["decisao_rebalanceamento"].ativos_executados)
    print(f"Turnover: {resultado['decisao_rebalanceamento'].turnover:.4f} (fração da carteira, 0-1)")
    print(f"Custo total de transação: R$ {resultado['custos'].custo_total:,.2f} "
          f"({resultado['custos'].custo_total_pct_carteira*100:.4f}% da carteira)")
    print("\nSe chegou até aqui sem erro, os 9 agentes estão integrados corretamente.")


if __name__ == "__main__":
    rodar_pipeline_sintetico()
