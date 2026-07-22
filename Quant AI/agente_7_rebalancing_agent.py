"""
agente_7_rebalancing_agent.py

Agente 7 do pipeline Quant AI — Rebalancing Agent.

Responsabilidade (Fase 0): aplicar a banda de não-negociação (desacopla
recálculo diário do sinal da execução real), e estimar custos de
transação B3 (emolumentos, corretagem, slippage) sobre o que de fato
for executado.

Duas entradas de peso são conceitualmente diferentes e não devem ser
confundidas:
  - pesos_atuais_com_drift: o peso REAL de cada ativo na carteira HOJE,
    já refletindo a variação natural de preço desde o último rebalance
    (não é "o último peso-alvo" -- ativos que não foram negociados
    continuam mudando de peso conforme o preço se move).
  - pesos_alvo: a saída do Agente 6 (BL Optimizer) para o rebalanceamento
    de hoje, SE fosse executado sem restrição de banda.

Decisão de projeto (fechada na Fase 0): banda = desvio-padrão histórico
do peso-alvo do próprio ativo (não da volatilidade do retorno -- unidade
teria que ser a mesma da grandeza comparada, que é peso, não retorno),
escalado por um fator de liquidez (1,0x líquidos, 1,5x menos líquidos).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Banda de não-negociação
# --------------------------------------------------------------------------

def calcular_banda_nao_negociacao(
    historico_pesos_alvo: pd.DataFrame,  # index=data, colunas=tickers, valores=peso-alvo do Agente 6
    fator_liquidez: dict[str, float],     # {ticker: 1.0 ou 1.5}
    janela_dias: int = 60,
) -> pd.Series:
    """
    banda_i = desvio_padrao(peso-alvo histórico do ativo i, janela móvel) * fator_liquidez_i

    Retorna a banda mais recente (última linha da janela) por ticker.
    """
    sigma_sinal = historico_pesos_alvo.tail(janela_dias).std()
    fatores = pd.Series(fator_liquidez)
    return sigma_sinal * fatores.reindex(sigma_sinal.index).fillna(1.0)


# --------------------------------------------------------------------------
# 2. Decisão de execução (per-ativo, respeitando a banda)
# --------------------------------------------------------------------------

@dataclass
class DecisaoRebalanceamento:
    pesos_finais: pd.Series
    ativos_executados: list[str]
    turnover: float


def decidir_execucao(
    pesos_alvo: pd.Series,
    pesos_atuais_com_drift: pd.Series,
    bandas: pd.Series,
) -> DecisaoRebalanceamento:
    """
    Para cada ativo: executa (adota peso_alvo) SE |peso_alvo - peso_atual|
    > banda; senão mantém o peso atual (com drift, não o alvo antigo).

    Como só um subconjunto dos ativos pode ser executado, a soma dos
    pesos finais pode não fechar em 1 (ex: ativos executados somam menos
    do que os que ficaram de fora deixaram de ocupar) -- renormaliza no
    final para manter a restrição de orçamento (sum(w)==1) do Agente 6.
    """
    tickers = pesos_alvo.index
    desvio = (pesos_alvo - pesos_atuais_com_drift).abs()
    executar = desvio > bandas.reindex(tickers).fillna(0.0)

    pesos_finais = pesos_atuais_com_drift.copy()
    pesos_finais[executar] = pesos_alvo[executar]

    soma = pesos_finais.sum()
    if soma > 0:
        pesos_finais = pesos_finais / soma

    turnover = 0.5 * (pesos_finais - pesos_atuais_com_drift).abs().sum()

    return DecisaoRebalanceamento(
        pesos_finais=pesos_finais,
        ativos_executados=list(tickers[executar]),
        turnover=float(turnover),
    )


# --------------------------------------------------------------------------
# 3. Custos de transação B3
# --------------------------------------------------------------------------

@dataclass
class CustosTransacao:
    """
    Todos os valores monetários (custo_emolumentos, custo_corretagem,
    custo_slippage_estimado, custo_total) estão na MESMA unidade
    monetária de valor_carteira -- este projeto assume R$ (reais) em
    todo o pipeline (não há conversão de moeda em nenhum agente).
    custo_total_pct_carteira é adimensional (fração, ex: 0.0002 = 0,02%).
    """
    custo_emolumentos: float
    custo_corretagem: float
    custo_slippage_estimado: float
    custo_total: float
    custo_total_pct_carteira: float


def calcular_custos_transacao(
    pesos_antes: pd.Series,
    pesos_depois: pd.Series,
    valor_carteira: float,
    taxa_emolumentos: float = 0.0325 / 100,   # aproximação B3, ajustável
    corretagem_por_operacao: float = 0.0,      # muitas corretoras oferecem 0 para ações -- ajustável
    slippage_bps_por_ativo: dict[str, float] | None = None,  # custo de impacto, em bps, por ticker
) -> CustosTransacao:
    """
    UNIDADE: valor_carteira deve estar em R$ (reais) -- o pipeline
    inteiro (Agente 6, Agente 8) assume reais em todos os valores
    monetários, sem nenhuma conversão de moeda implementada em nenhum
    agente. O retorno (custo_emolumentos, custo_corretagem,
    custo_slippage_estimado, custo_total) sai na MESMA unidade que
    valor_carteira -- se você passar 100_000 pensando em R$ 100 mil,
    o custo_total sai em reais; se passar pensando em outra coisa
    (milhares de reais, dólares), o número sai "certo"
    matematicamente mas errado na interpretação, sem nenhum aviso do
    código. custo_total_pct_carteira é a única saída adimensional
    (fração do valor da carteira, independente de moeda).

    Estima custos SOMENTE sobre o volume de fato negociado (ativos com
    variação de peso entre antes/depois), não sobre a carteira inteira.

    NOTA DE LIMITAÇÃO (documentada, não escondida): não inclui Imposto de
    Renda sobre ganho de capital -- isso exige rastreamento de custo
    médio de aquisição por lote (FIFO), que é uma extensão futura fora
    do escopo desta versão. Ver Fase0_Premissas_QuantAI.docx, Seção 10
    (Limitações Declaradas).
    """
    variacao = (pesos_depois - pesos_antes).abs()
    ativos_negociados = variacao[variacao > 1e-9]

    volume_negociado = float((ativos_negociados * valor_carteira).sum())
    custo_emolumentos = volume_negociado * taxa_emolumentos
    custo_corretagem = len(ativos_negociados) * corretagem_por_operacao

    if slippage_bps_por_ativo:
        custo_slippage = sum(
            ativos_negociados.get(t, 0.0) * valor_carteira * (bps / 10000)
            for t, bps in slippage_bps_por_ativo.items()
        )
    else:
        custo_slippage = 0.0

    custo_total = custo_emolumentos + custo_corretagem + custo_slippage
    return CustosTransacao(
        custo_emolumentos=custo_emolumentos,
        custo_corretagem=custo_corretagem,
        custo_slippage_estimado=custo_slippage,
        custo_total=custo_total,
        custo_total_pct_carteira=custo_total / valor_carteira if valor_carteira > 0 else 0.0,
    )
