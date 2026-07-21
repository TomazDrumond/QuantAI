"""
agente_3_aggregation_agent.py

Agente 3 do pipeline Quant AI — Aggregation Agent.

Responsabilidade: colapsar N notícias por (ativo, data) -- cada uma já
pontuada pelos 3 LLMs independentes do Agente 2 -- num único registro
diário no formato que os Agentes 4/6 e os scripts de calibração já
esperam: data, ticker, s1, s2, s3, n_noticias, score_meta.

Decisão de projeto: agregação por dia usa apenas dados do próprio dia
(sem ponderar por recência dentro do dia) -- o decaimento temporal já é
responsabilidade explícita do Agente 5 (Monte Carlo/Wiener), então o
Agente 3 não deveria antecipar esse papel. Isso mantém a separação de
responsabilidades definida na Fase 0.

MÉTODO DE AGREGAÇÃO ENTRE NOTÍCIAS DO MESMO DIA: MEDIANA (padrão fixo,
decisão fechada nesta etapa) -- não média. Justificativa: uma alternativa
de dar peso extra a sentimento pessimista (assimetria documentada na
literatura para o mercado brasileiro) foi cogitada e DELIBERADAMENTE
DESCARTADA por risco de overfitting -- calibrar um terceiro parâmetro via
grid search sobre a mesma amostra de backtest que já calibra epsilon_min
e corr_min consome graus de liberdade adicionais contra a mesma janela
histórica, inflando artificialmente o Sharpe reportado (o mesmo tipo de
fragilidade criticada no paper-base, Sharpe 3.02). Mediana foi escolhida
por ser a opção mais robusta a notícias isoladas sensacionalistas SEM
introduzir nenhum parâmetro novo a calibrar.

Entrada esperada (uma linha por notícia, já processada pelo Agente 2):
    data, ticker, noticia_id, s1, s2, s3

Saída (uma linha por ativo/dia, pronta para Agente 4 e calibração):
    data, ticker, s1, s2, s3, n_noticias, score_meta
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def agregar_meta_score(
    df_bruto: pd.DataFrame,
    metodo_agregacao: Literal["media", "mediana"] = "mediana",
) -> pd.DataFrame:
    """
    df_bruto: uma linha por notícia, colunas obrigatórias:
        data, ticker, noticia_id, s1, s2, s3
    (s1/s2/s3 são os scores dos 3 LLMs para AQUELA notícia específica,
    já no intervalo [-1, 1] -- validação de range é responsabilidade do
    Agente 2, não repetida aqui).

    Retorna: uma linha por (data, ticker), pronta para os Agentes 4/6 e
    calibracao_epsilon_min.py / calibracao_corr_min.py.
    """
    obrigatorias = {"data", "ticker", "noticia_id", "s1", "s2", "s3"}
    faltando = obrigatorias - set(df_bruto.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")

    agregador = "median" if metodo_agregacao == "mediana" else "mean"

    agrupado = (
        df_bruto
        .groupby(["data", "ticker"])
        .agg(
            s1=("s1", agregador),
            s2=("s2", agregador),
            s3=("s3", agregador),
            n_noticias=("noticia_id", "nunique"),
        )
        .reset_index()
    )

    agrupado["score_meta"] = agrupado[["s1", "s2", "s3"]].mean(axis=1)
    return agrupado


def diagnostico_divergencia_diaria(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnóstico auxiliar (não obrigatório no pipeline principal): para
    dias com múltiplas notícias do mesmo ativo, mede o quanto os scores
    das notícias individuais divergem entre si (não confundir com a
    divergência ENTRE OS 3 LLMS na mesma notícia, que já é o que Omega
    mede no Agente 4). Útil para decidir entre 'media' e 'mediana' com
    base no comportamento real do seu corpus de notícias em português.
    """
    scores_noticia = df_bruto.assign(score_noticia=df_bruto[["s1", "s2", "s3"]].mean(axis=1))
    return (
        scores_noticia
        .groupby(["data", "ticker"])["score_noticia"]
        .agg(n_noticias="count", dispersao_entre_noticias="std")
        .reset_index()
        .query("n_noticias > 1")
    )
