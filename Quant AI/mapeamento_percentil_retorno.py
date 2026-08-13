"""
mapeamento_percentil_retorno.py

Substitui a escala linear fixa (Q = score * 0.15, igual para todos os
ativos) por um mapeamento baseado na distribuição EMPÍRICA de retornos
de cada ativo: o score de sentimento [-1, 1] é traduzido para um
percentil [0, 100] da distribuição histórica de retornos DAQUELE ativo,
e o Q usado na posterior do BL é o valor de retorno observado
historicamente nesse percentil.

Por que isso é melhor que a escala fixa: um ativo com volatilidade 3x
maior que outro tem uma distribuição de retornos 3x mais espalhada --
usar a mesma constante (0.15) para os dois ignora essa diferença. Usar
o percentil da PRÓPRIA distribuição do ativo faz o "tamanho" da view
respeitar a escala natural de movimento de cada papel.

Fallback: se o ativo não tiver histórico suficiente (ex: IPO recente,
< min_obs_por_ativo observações), usa a distribuição do índice
(Ibovespa) como aproximação documentada, não uma distribuição vazia.

DISCIPLINA WALK-FORWARD (obrigatória, mesma dos outros parâmetros do
projeto): a tabela deve ser construída SÓ com preços conhecidos até
t-1 no momento do rebalanceamento -- nunca com a série de preços
completa 2018-2026 de uma vez, o que vazaria retorno futuro para
dentro da escala usada em decisões passadas. A recalibração periódica
fica a cargo de quem orquestra o walk-forward (pipeline_quant_ai_colab.py),
na mesma cadência do epsilon_min.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TabelaPercentis:
    tickers_com_tabela_propria: list[str]
    tickers_usando_fallback_indice: list[str]
    breakpoints_por_ticker: dict[str, np.ndarray]


def construir_tabela_percentis_retorno(
    df_precos: pd.DataFrame,
    tickers: list[str],
    ticker_indice_fallback: str = "IBOV",
    n_percentis: int = 20,
    horizonte_dias: int = 5,
    min_obs_por_ativo: int = 60,
) -> TabelaPercentis:
    """
    df_precos: DataFrame wide (index=data, colunas=tickers + índice se
    disponível), contendo APENAS preços até a data de corte desejada
    (walk-forward -- filtrar antes de chamar esta função).

    n_percentis: número de buckets (20 = a cada 5 percentis).
    horizonte_dias: horizonte do retorno usado para construir a
        distribuição (mesma convenção de 5 dias já usada em outros
        pontos do pipeline), depois anualizado para bater com a escala
        anual de Pi/Sigma.
    """
    # CORREÇÃO: anualização por RAIZ do tempo (convenção padrão para
    # dispersão/volatilidade), não multiplicação linear (252/horizonte).
    # A multiplicação linear de uma observação de retorno de curtíssimo
    # prazo, COM todo o ruído dela, extrapola esse ruído como se fosse uma
    # taxa persistente o ano inteiro -- produzia magnitudes irreais (Q de
    # 100%+ para score extremo), testado e confirmado antes desta correção.
    fator_anualizacao = np.sqrt(252 / horizonte_dias)

    def _distribuicao_anualizada(serie_precos: pd.Series) -> np.ndarray | None:
        precos = serie_precos.dropna()
        if len(precos) < min_obs_por_ativo + horizonte_dias:
            return None
        retornos = np.log(precos.shift(-horizonte_dias) / precos).dropna() * fator_anualizacao
        if len(retornos) < min_obs_por_ativo:
            return None
        return retornos.values

    dist_indice = None
    if ticker_indice_fallback in df_precos.columns:
        dist_indice = _distribuicao_anualizada(df_precos[ticker_indice_fallback])

    breakpoints_por_ticker: dict[str, np.ndarray] = {}
    com_tabela_propria: list[str] = []
    usando_fallback: list[str] = []

    for tk in tickers:
        dist = _distribuicao_anualizada(df_precos[tk]) if tk in df_precos.columns else None

        if dist is not None:
            com_tabela_propria.append(tk)
        elif dist_indice is not None:
            dist = dist_indice
            usando_fallback.append(tk)
        else:
            # Nem o ativo nem o índice têm dado suficiente -- fallback
            # final, documentado como o pior caso (equivalente a um
            # "não sei", simétrico e conservador).
            dist = np.array([-0.15, 0.0, 0.15])
            usando_fallback.append(tk)

        breakpoints_por_ticker[tk] = np.percentile(dist, np.linspace(0, 100, n_percentis + 1))

    return TabelaPercentis(
        tickers_com_tabela_propria=com_tabela_propria,
        tickers_usando_fallback_indice=usando_fallback,
        breakpoints_por_ticker=breakpoints_por_ticker,
    )


def score_para_retorno_percentil(score: float, breakpoints: np.ndarray) -> float:
    """
    Mapeia score de sentimento [-1,1] linearmente para percentil [0,100]
    (score=-1 -> percentil 0, score=0 -> percentil 50 [mediana histórica],
    score=+1 -> percentil 100), e interpola o valor de retorno correspondente
    na tabela de breakpoints empíricos.
    """
    score_clip = max(-1.0, min(1.0, score))
    percentil = (score_clip + 1.0) / 2.0 * 100.0

    n_breakpoints = len(breakpoints)
    posicoes = np.linspace(0, 100, n_breakpoints)
    return float(np.interp(percentil, posicoes, breakpoints))


def score_para_q(score: float, tabela: TabelaPercentis, ticker: str) -> float:
    """Ponto de entrada único usado pelo pipeline -- resolve o ticker na tabela e aplica o mapeamento."""
    breakpoints = tabela.breakpoints_por_ticker.get(ticker)
    if breakpoints is None:
        raise KeyError(f"Ticker {ticker} não está na tabela de percentis construída para esta janela.")
    return score_para_retorno_percentil(score, breakpoints)
