"""
agente_4_correlation_filter.py

Agente 4 do pipeline Quant AI — Correlation Filter.

Duas responsabilidades (Fase 0):
  (a) Purificar o Meta-Score de sentimento via correlação histórica móvel
      com o índice: Score_adj = Score_meta * |corr(ativo, index)|.
  (b) Derivar Omega (incerteza da view) a partir da divergência entre os
      3 LLMs e do volume de notícias -- ver calibracao_epsilon_min.py.

CORREÇÃO DE PROJETO (nesta etapa): o filtro de correlação puro tem um
problema real em quebras de regime (ex: pandemia). Setores estruturalmente
defensivos (farmácia, hospitais, saneamento) têm BAIXA correlação
histórica com o Ibovespa por natureza -- não por ruído. Multiplicar o
Meta-Score por |corr| SUPRIME sinal genuíno desses setores exatamente
quando ele seria mais valioso (ex: sentimento forte e positivo em
Hypera/Fleury durante a pandemia, amortecido pelo filtro por causa de
uma correlação historicamente baixa e estruturalmente esperada).

Correção: piso mínimo de correlação (corr_min), no mesmo espírito do
epsilon_min já aplicado a Omega -- acima do piso, nada muda (ativos de
alta correlação continuam tratados normalmente); abaixo do piso, a
supressão do sinal é limitada, nunca zerando o Meta-Score por completo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Purificação do sinal via correlação (com piso mínimo)
# --------------------------------------------------------------------------

def calcular_correlacao_movel(
    retornos_ativo: pd.Series,
    retornos_index: pd.Series,
    janela_dias: int = 90,
) -> pd.Series:
    """Correlação histórica móvel (rolling) entre o ativo e o Ibovespa."""
    return retornos_ativo.rolling(janela_dias).corr(retornos_index)


def calcular_score_ajustado(
    score_meta: float,
    correlacao: float,
    corr_min: float = 0.25,
) -> float:
    """
    Score_adj = Score_meta * max(|corr(ativo, index)|, corr_min)

    corr_min: piso mínimo de correlação absoluta. Acima do piso, o
    comportamento é idêntico à fórmula original (nada muda para ativos
    com correlação naturalmente alta ou moderada). Abaixo do piso -- o
    caso de setores estruturalmente defensivos -- a supressão do sinal
    fica limitada a, no máximo, (1 - corr_min) do Meta-Score original,
    em vez de poder ir a zero.

    Mesma lógica de design do epsilon_min em Omega: o piso existe para
    um caso de borda estrutural (aqui, correlação genuinamente baixa),
    não para "amaciar" o filtro de forma geral -- por isso corr_min
    deveria ser calibrado empiricamente (ex: quantil baixo da distribuição
    histórica de |corr| do próprio universo), não escolhido no chute.
    """
    if pd.isna(correlacao):
        correlacao = 0.0  # sem histórico suficiente ainda -- tratado como pior caso, piso se aplica
    fator = max(abs(correlacao), corr_min)
    return score_meta * fator


# --------------------------------------------------------------------------
# 2. Derivação de Omega (consolidado aqui — mesma fórmula já validada)
# --------------------------------------------------------------------------

def calcular_omega(
    scores_llm: np.ndarray,
    n_noticias: np.ndarray,
    epsilon_min: float,
    penalidade_modelo_unico: float = 0.15,
) -> np.ndarray:
    """
    Versão ROBUSTA a falhas parciais do Agente 2 (decisão de projeto:
    aproveitar dado disponível, não descartar a notícia inteira quando
    1 ou 2 dos 3 LLMs falham).

    scores_llm: array (N, 3), podendo conter np.nan nas posições em que
    um provider não respondeu (falha de API, timeout, parsing).

    Regras:
      - 3 ou 2 scores válidos: variância calculada normalmente sobre os
        valores disponíveis (Var de 2 pontos ainda é informativa).
      - 1 score válido: não há como medir concordância entre modelos --
        aplica-se uma PENALIDADE FIXA de incerteza (penalidade_modelo_unico,
        maior que epsilon_min por padrão), não um piso "otimista".
      - 0 scores válidos: retorna NaN -- não deveria virar view no Agente 6
        (ausência de sinal, não sinal neutro).
    """
    scores_llm = np.asarray(scores_llm, dtype=float)
    n_obs = scores_llm.shape[0]
    var_llm = np.full(n_obs, np.nan)

    for i in range(n_obs):
        validos = scores_llm[i][~np.isnan(scores_llm[i])]
        if len(validos) >= 2:
            var_llm[i] = np.var(validos, ddof=0)
        elif len(validos) == 1:
            var_llm[i] = penalidade_modelo_unico
        # len(validos) == 0 -> permanece NaN (sem view possível)

    var_llm = np.where(np.isnan(var_llm), np.nan, np.maximum(var_llm, epsilon_min))
    n_seguro = np.maximum(n_noticias, 1)
    return var_llm / np.sqrt(n_seguro)
