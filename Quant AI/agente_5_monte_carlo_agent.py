"""
agente_5_monte_carlo_agent.py

Agente 5 do pipeline Quant AI — Monte Carlo Agent.

Responsabilidade (Fase 0): aproximar via simulação de Monte Carlo o
decaimento temporal do impacto de uma notícia sobre o preço esperado,
produzindo o vetor de retornos esperados (Q) e o nível de confiança
associado, para os Agentes 4/6.

Problema estrutural que este agente resolve: o Agente 3 agrega
sentimento POR DIA -- se um ativo teve uma notícia forte há 3 dias e
nenhuma notícia hoje, o Agente 3 simplesmente não gera nada para hoje
(sem notícia = sem linha no groupby). Sem o Agente 5, o sinal "some"
abruptamente no dia seguinte à notícia, o que não reflete como mercados
realmente absorvem informação (o impacto decai gradualmente, não cai a
zero de uma vez). O Agente 5 mantém viva uma versão decaída do sinal
por uma janela de memória, até ele se tornar despreztível.

Modelo: processo tipo Ornstein-Uhlenbeck discreto (reversão à média
zero) simulado via Monte Carlo -- a média das trajetórias simuladas
converge ao decaimento exponencial fechado exp(-lambda*t); a variância
das trajetórias cresce com o tempo decorrido, capturando a incerteza
crescente sobre o quanto do impacto original ainda está "vivo" no preço.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ImpactoDecaido:
    ticker: str
    q_estimado: float           # média das trajetórias simuladas (decaimento esperado)
    variancia_temporal: float   # variância das trajetórias (incerteza por decaimento)
    dias_desde_noticia: int


# --------------------------------------------------------------------------
# 1. Núcleo: simulação de Monte Carlo do decaimento (1 evento de notícia)
# --------------------------------------------------------------------------

def simular_decaimento_wiener(
    score_inicial: float,
    dias_decorridos: int,
    lambda_decaimento: float = 0.15,
    sigma_decaimento: float = 0.05,
    n_simulacoes: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula n_simulacoes trajetórias diárias de um processo tipo
    Ornstein-Uhlenbeck discreto:
        impacto_t = impacto_{t-1} * (1 - lambda_decaimento) + ruido_t
        ruido_t ~ N(0, sigma_decaimento)

    Retorna o vetor de valores finais (após dias_decorridos passos) das
    n_simulacoes trajetórias -- a média converge ao decaimento
    exponencial fechado; a dispersão cresce com dias_decorridos.

    dias_decorridos=0 retorna um array constante = score_inicial (sem
    decaimento, sem incerteza -- a notícia acabou de sair).
    """
    if dias_decorridos == 0:
        return np.full(n_simulacoes, score_inicial)

    rng = np.random.default_rng(seed)
    trajetorias = np.full(n_simulacoes, score_inicial, dtype=float)

    for _ in range(dias_decorridos):
        ruido = rng.normal(0.0, sigma_decaimento, size=n_simulacoes)
        trajetorias = trajetorias * (1.0 - lambda_decaimento) + ruido

    return trajetorias


def estimar_impacto_decaido(
    ticker: str,
    score_inicial: float,
    dias_decorridos: int,
    lambda_decaimento: float = 0.15,
    sigma_decaimento: float = 0.05,
    n_simulacoes: int = 2000,
    seed: int | None = None,
) -> ImpactoDecaido:
    trajetorias = simular_decaimento_wiener(
        score_inicial, dias_decorridos, lambda_decaimento, sigma_decaimento, n_simulacoes, seed
    )
    return ImpactoDecaido(
        ticker=ticker,
        q_estimado=float(np.mean(trajetorias)),
        variancia_temporal=float(np.var(trajetorias)),
        dias_desde_noticia=dias_decorridos,
    )


# --------------------------------------------------------------------------
# 2. Combinação de múltiplos eventos recentes -> Q(t) e confiança do dia
# --------------------------------------------------------------------------

def combinar_eventos_recentes(
    eventos_ticker: pd.DataFrame,  # colunas: data_noticia, score_meta (saída do Agente 3)
    data_alvo: pd.Timestamp,
    janela_memoria_dias: int = 15,
    lambda_decaimento: float = 0.15,
    sigma_decaimento: float = 0.05,
    n_simulacoes: int = 2000,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Para um ticker, combina TODAS as notícias dentro da janela de memória
    (janela_memoria_dias) anteriores a data_alvo, cada uma decaída pelo
    tempo decorrido desde sua publicação, numa média ponderada pela
    confiança de cada uma (1/variancia_temporal -- eventos mais recentes,
    com menos incerteza acumulada, pesam mais).

    Retorna (Q_combinado, variancia_temporal_combinada) -- este último
    deve ser SOMADO à variância de divergência entre LLMs já calculada
    no Agente 4 (Omega), não substituí-la: são duas fontes de incerteza
    diferentes (desacordo entre modelos vs. desatualização temporal).
    """
    relevantes = eventos_ticker[
        (eventos_ticker["data_noticia"] <= data_alvo) &
        (eventos_ticker["data_noticia"] >= data_alvo - pd.Timedelta(days=janela_memoria_dias))
    ]

    if relevantes.empty:
        return 0.0, np.nan  # sem notícia recente na janela -- sem view (Agente 6 já sabe tratar NaN)

    impactos = []
    for _, evento in relevantes.iterrows():
        dias = (data_alvo - evento["data_noticia"]).days
        impacto = estimar_impacto_decaido(
            ticker="", score_inicial=evento["score_meta"], dias_decorridos=dias,
            lambda_decaimento=lambda_decaimento, sigma_decaimento=sigma_decaimento,
            n_simulacoes=n_simulacoes, seed=seed,
        )
        impactos.append(impacto)

    pesos = np.array([1.0 / max(imp.variancia_temporal, 1e-6) for imp in impactos])
    q_valores = np.array([imp.q_estimado for imp in impactos])
    pesos_norm = pesos / pesos.sum()

    q_combinado = float(np.sum(pesos_norm * q_valores))
    variancia_combinada = float(1.0 / pesos.sum())  # variância da média ponderada por precisão

    return q_combinado, variancia_combinada
