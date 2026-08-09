"""
agente_1_data_agent.py

Agente 1 do pipeline Quant AI — Data Agent.

Responsabilidade única (por decisão de projeto, Fase 0): buscar preços
diários e alinhar timestamps de notícias, DADO um universo já definido
pelo Agente 0 (composição PIT). Este agente NÃO decide quais ativos
existem — só busca dado para os ativos que o Agente 0 já validou.

Duas garantias não-negociáveis, fechadas na Fase 0:
  1. NUNCA look-ahead bias: timestamp(notícia) < timestamp(decisão)
     <= timestamp(preço de referência do retorno).
  2. NUNCA interpolação bidirecional de preços ausentes — só simulação
     forward-only (Wiener, usando apenas volatilidade histórica
     conhecida até t-1). Interpolação linear clássica usa P(t+1), que
     é o mesmo pecado de look-ahead que o item 1 proíbe.

Dependências externas (instalar no ambiente real, ex: Antigravity):
    pip install yfinance pandas_market_calendars

NOTA DE AMBIENTE: este script foi desenvolvido num sandbox sem acesso à
internet. As funções que dependem de rede (`buscar_precos_b3`,
`obter_calendario_pregao`) não puderam ser executadas de fato aqui — a
lógica pura (detecção de gaps, preenchimento forward-only, validação de
timestamp) foi testada com dados sintéticos e está validada.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YFINANCE_DISPONIVEL = True
except ImportError:
    _YFINANCE_DISPONIVEL = False

try:
    import pandas_market_calendars as mcal
    _MCAL_DISPONIVEL = True
except ImportError:
    _MCAL_DISPONIVEL = False


# --------------------------------------------------------------------------
# 1. Busca de preços (B3 via Yahoo Finance — evita o captcha da B3 direta)
# --------------------------------------------------------------------------

def buscar_precos_b3(tickers: list[str], inicio: str, fim: str) -> pd.DataFrame:
    """
    tickers: lista SEM sufixo (ex: ["PETR4", "VALE3"]) -- a função adiciona
    ".SA" automaticamente (convenção do Yahoo Finance para ativos B3).

    Retorna DataFrame long-format: colunas [data, ticker, close].
    """
    if not _YFINANCE_DISPONIVEL:
        raise ImportError("yfinance não instalado. Rode `pip install yfinance` no ambiente real (Antigravity).")

    tickers_yf = [f"{t}.SA" for t in tickers]
    bruto = yf.download(tickers_yf, start=inicio, end=fim, progress=False)["Close"]

    bruto = bruto.rename(columns={f"{t}.SA": t for t in tickers})
    longo = bruto.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
    longo = longo.rename(columns={"Date": "data"}).dropna(subset=["close"])
    return longo.sort_values(["ticker", "data"]).reset_index(drop=True)


def obter_calendario_pregao(inicio: str, fim: str) -> pd.DatetimeIndex:
    """Calendário oficial de pregão B3 (BVMF) -- sem depender de scraping ao vivo."""
    if not _MCAL_DISPONIVEL:
        raise ImportError("pandas_market_calendars não instalado. Rode `pip install pandas_market_calendars`.")

    calendario = mcal.get_calendar("BVMF")
    return calendario.schedule(start_date=inicio, end_date=fim).index


# --------------------------------------------------------------------------
# 2. Detecção e preenchimento de gaps — FORWARD-ONLY (sem look-ahead)
# --------------------------------------------------------------------------

def detectar_gaps(precos_ticker: pd.Series, dias_pregao: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    precos_ticker: Series indexada por data, valores = preço de fechamento.
    dias_pregao: calendário oficial de pregão no período.

    Retorna as datas em que HOUVE pregão mas o ativo não tem preço --
    esses são os gaps que precisam de preenchimento forward-only.
    """
    return dias_pregao.difference(precos_ticker.dropna().index)


def identificar_ativos_com_gap_prolongado(
    precos_ticker: pd.Series,
    dias_pregao: pd.DatetimeIndex,
    limite_dias_uteis: int = 5,
) -> pd.DatetimeIndex:
    """
    Fecha pendência do Documento Consolidado Fase 1, Seção 6: decisão já
    fechada com o usuário era um teto de 5 dias úteis consecutivos de
    gap antes de EXCLUIR o ativo temporariamente do rebalanceamento
    daquela janela, em vez de continuar simulando via Wiener
    indefinidamente (risco: gaps longos costumam ser eventos
    informativos -- fraude, suspensão regulatória -- que o processo de
    Wiener, calibrado com volatilidade histórica pré-evento, não tem
    capacidade de representar).

    Retorna as datas em que o ativo deveria ficar EXCLUÍDO do universo
    elegível (gap consecutivo > limite_dias_uteis), para o Agente 0/6
    tratarem como ausência temporária, não como preço a simular.
    """
    gaps = detectar_gaps(precos_ticker, dias_pregao)
    if len(gaps) == 0:
        return pd.DatetimeIndex([])

    gaps_ordenados = gaps.sort_values()
    datas_excluidas = []
    contagem = 1

    for i in range(1, len(gaps_ordenados)):
        posicao_anterior = dias_pregao.get_loc(gaps_ordenados[i - 1])
        posicao_atual = dias_pregao.get_loc(gaps_ordenados[i])
        consecutivo = (posicao_atual - posicao_anterior) == 1

        if consecutivo:
            contagem += 1
        else:
            if contagem > limite_dias_uteis:
                datas_excluidas.extend(gaps_ordenados[i - contagem:i])
            contagem = 1

    if contagem > limite_dias_uteis:
        datas_excluidas.extend(gaps_ordenados[len(gaps_ordenados) - contagem:])

    return pd.DatetimeIndex(datas_excluidas)




def preencher_gap_wiener(
    precos_ticker: pd.Series,
    dias_pregao: pd.DatetimeIndex,
    janela_vol_dias: int = 60,
    seed: int | None = None,
) -> pd.Series:
    """
    Preenche gaps EXCLUSIVAMENTE com informação disponível até t-1 --
    nunca usa preço em t+1 (o que a interpolação linear clássica faria).

    Para cada gap em t: simula um passo de processo de Wiener a partir
    do ÚLTIMO PREÇO VÁLIDO conhecido, usando a volatilidade histórica
    calculada com dados estritamente anteriores a t (janela_vol_dias).
    Se o gap persistir por múltiplos dias, cada passo usa o preço
    simulado do passo anterior como base -- nunca olha para frente.
    """
    rng = np.random.default_rng(seed)
    serie = precos_ticker.reindex(dias_pregao).copy()

    for i, data in enumerate(dias_pregao):
        if pd.notna(serie.loc[data]):
            continue  # sem gap aqui

        historico_ate_aqui = serie.iloc[:i].dropna()
        if historico_ate_aqui.empty:
            continue  # sem preço válido anterior -- não há como simular forward

        ultimo_preco_valido = historico_ate_aqui.iloc[-1]
        retornos_log = np.log(historico_ate_aqui / historico_ate_aqui.shift(1)).dropna()
        vol_historica = retornos_log.tail(janela_vol_dias).std()
        vol_historica = vol_historica if pd.notna(vol_historica) and vol_historica > 0 else 0.01

        choque = rng.normal(loc=0.0, scale=vol_historica)
        serie.loc[data] = ultimo_preco_valido * np.exp(choque)

    return serie


# --------------------------------------------------------------------------
# 3. Validação de integridade de timestamp — anti look-ahead bias
# --------------------------------------------------------------------------

@dataclass
class ViolacaoTimestamp:
    ticker: str
    timestamp_noticia: datetime
    timestamp_decisao: datetime
    timestamp_preco: datetime
    motivo: str


def validar_alinhamento_timestamp(
    eventos_noticia: pd.DataFrame,  # colunas: ticker, timestamp_noticia, timestamp_decisao, timestamp_preco
) -> list[ViolacaoTimestamp]:
    """
    Regra obrigatória (Fase 0):
        timestamp(notícia) < timestamp(decisão) <= timestamp(preço)

    Retorna a lista de violações encontradas -- lista vazia significa
    que o dataset está íntegro. Isso deve ser rodado ANTES de qualquer
    dado ser passado ao Agente 2/3 (sentimento) ou ao script de
    calibração de epsilon_min.
    """
    violacoes: list[ViolacaoTimestamp] = []

    for _, ev in eventos_noticia.iterrows():
        if not (ev["timestamp_noticia"] < ev["timestamp_decisao"]):
            violacoes.append(ViolacaoTimestamp(
                ticker=ev["ticker"], timestamp_noticia=ev["timestamp_noticia"],
                timestamp_decisao=ev["timestamp_decisao"], timestamp_preco=ev["timestamp_preco"],
                motivo="notícia não precede estritamente a decisão",
            ))
        elif not (ev["timestamp_decisao"] <= ev["timestamp_preco"]):
            violacoes.append(ViolacaoTimestamp(
                ticker=ev["ticker"], timestamp_noticia=ev["timestamp_noticia"],
                timestamp_decisao=ev["timestamp_decisao"], timestamp_preco=ev["timestamp_preco"],
                motivo="decisão ocorre depois do preço de referência",
            ))

    return violacoes


# --------------------------------------------------------------------------
# 4. Retorno realizado forward — interface de saída para o Agente 3
# --------------------------------------------------------------------------

def calcular_retorno_realizado_fwd(precos_ticker: pd.Series, horizonte_dias: int = 1) -> pd.Series:
    """
    Retorno log entre t e t+horizonte_dias -- é o que alimenta a coluna
    `retorno_realizado_fwd` esperada por calibracao_epsilon_min.py.
    NUNCA usado para decidir nada em t -- só existe depois que t já
    passou, para fins de avaliação/calibração walk-forward.
    """
    return np.log(precos_ticker.shift(-horizonte_dias) / precos_ticker)
