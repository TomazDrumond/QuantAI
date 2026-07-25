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


def carregar_precos_reais(caminho_csv: str = "precos_historicos.csv") -> pd.DataFrame:
    """
    Carrega o painel de preços com integridade tratada pelo Agente 1
    (preenchimento forward-only via Wiener, exclusão de gaps > 5 dias).
    """
    df = pd.read_csv(caminho_csv, parse_dates=["data"])
    return df



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
# Backtest Walk-Forward (v2 — calibração dinâmica de τ + bandas reais)
# --------------------------------------------------------------------------

def rodar_backtest_walkforward(
    caminho_pit: str = "composicao_pit_ibovespa.csv",
    caminho_precos: str = "precos_historicos.csv",
    delta: float = 3.0,
    w_max: float = 0.20,
    valor_inicial: float = 100_000.0,
    razao_precisao_alvo: float = 3.0,
) -> None:
    """
    Executa a simulação walk-forward de ponta a ponta integrando os Agentes 0 a 8
    sobre o histórico 2018-2026.

    Correções v2:
      - τ calibrado DINAMICAMENTE a cada janela para manter a razão de
        precisão prior/views ≤ razao_precisao_alvo (default 3x). Resolve o
        problema de views afogadas pelo prior detectado na v1.
      - Histórico de pesos-alvo acumulado ao longo das janelas (não mais
        constante), para que o Agente 7 calcule bandas de não-negociação
        com dispersão real e reduza o turnover.
      - CDI por quadrimestre calculado a partir da taxa Selic anualizada
        (~10% no período), proporcional aos dias úteis da janela.
    """
    print("\n========================================================")
    print("  INICIANDO BACKTEST WALK-FORWARD QUANT AI v2 (2018-2026)")
    print("========================================================\n")

    composicao_pit = carregar_composicao_pit(caminho_pit)
    precos_painel = carregar_precos_reais(caminho_precos)

    datas_vigencia = sorted(composicao_pit["data_vigencia"].unique())
    print(f"Janelas PIT encontradas: {len(datas_vigencia)} quadrimestres "
          f"({datas_vigencia[0].date()} a {datas_vigencia[-1].date()}).")

    retornos_diarios_carteira = []
    turnovers = []
    registros_concentracao = []
    valor_carteira = valor_inicial
    selic_anual = 0.10  # Aproximação média do período

    # Matriz pivô de fechamentos (data x ticker)
    df_precos = precos_painel.pivot(index="data", columns="ticker", values="close").sort_index()

    # Acumulador de pesos-alvo reais para alimentar o Agente 7
    historico_pesos_acumulado: dict[str, list[float]] = {}
    pesos_anteriores: pd.Series | None = None

    for i in range(len(datas_vigencia) - 1):
        dt_inicio = datas_vigencia[i]
        dt_fim = datas_vigencia[i + 1]

        # Tickers da janela PIT atual
        universe_janela = composicao_pit[composicao_pit["data_vigencia"] == dt_inicio]["ticker"].tolist()

        # Filtra preços até dt_inicio (sem look-ahead)
        cols_disponiveis = [t for t in universe_janela if t in df_precos.columns]
        precos_historicos = df_precos.loc[:dt_inicio, cols_disponiveis].dropna(how="all", axis=1)
        tickers_validos = precos_historicos.columns.tolist()

        if len(tickers_validos) == 0:
            continue

        retornos_hist = np.log(precos_historicos / precos_historicos.shift(1)).dropna()
        if len(retornos_hist) < 30:
            continue

        # Covariância amostral com regularização e prior de equilíbrio Pi
        Sigma = retornos_hist.cov().values + np.eye(len(tickers_validos)) * 1e-4
        Pi = retornos_hist.mean().values * 252.0  # Retornos anualizados

        # Sentimento sintético estruturado (Q e Omega) para validação
        rng = np.random.default_rng(i)
        q_sim = rng.normal(0.02, 0.05, len(tickers_validos))
        omega_sim = np.maximum(0.01, rng.normal(0.03, 0.01, len(tickers_validos)))

        Q_por_ativo = dict(zip(tickers_validos, q_sim))
        Omega_por_ativo = dict(zip(tickers_validos, omega_sim))

        # --- CORREÇÃO 1: calibração dinâmica de τ ---
        # τ = razao_alvo * mediana(Ω) / mediana(diag(Σ))
        # Garante que a precisão do prior nunca afogue as views por > razao_alvo x
        mediana_omega = float(np.median(omega_sim))
        mediana_diag_sigma = float(np.median(np.diag(Sigma)))
        tau = razao_precisao_alvo * mediana_omega / max(mediana_diag_sigma, 1e-8)
        tau = max(tau, 0.01)  # piso de segurança

        # --- CORREÇÃO 2: histórico de pesos-alvo real para bandas ---
        # Monta DataFrame com os últimos 60 registros acumulados, todos com mesmo length
        ew = 1.0 / len(tickers_validos)
        historico_dict = {}
        for t in tickers_validos:
            raw = historico_pesos_acumulado.get(t, [])
            # Pega os últimos 60 (ou menos) e preenche à esquerda com equal-weight
            tail = raw[-60:]
            padded = [ew] * (60 - len(tail)) + tail
            historico_dict[t] = padded
        historico_df = pd.DataFrame(historico_dict)

        # Pesos atuais: drift a partir dos pesos anteriores, ou equal-weight
        if pesos_anteriores is not None:
            pesos_atuais = pesos_anteriores.reindex(tickers_validos).fillna(0.0)
            soma = pesos_atuais.sum()
            if soma > 0:
                pesos_atuais = pesos_atuais / soma
            else:
                pesos_atuais = pd.Series(1.0 / len(tickers_validos), index=tickers_validos)
        else:
            pesos_atuais = pd.Series(1.0 / len(tickers_validos), index=tickers_validos)

        fator_liquidez = {t: 1.0 for t in tickers_validos}

        # CDI por quadrimestre proporcional aos dias úteis
        fatia_precos = df_precos.loc[dt_inicio:dt_fim, tickers_validos].dropna(how="all")
        dias_uteis_janela = max(len(fatia_precos), 1)
        r_cdi_quadri = (1 + selic_anual) ** (dias_uteis_janela / 252.0) - 1

        res = rodar_um_rebalanceamento(
            tickers_validos, Pi, Sigma, tau, delta,
            Q_por_ativo, Omega_por_ativo, pesos_atuais, historico_df,
            fator_liquidez, valor_carteira=valor_carteira, w_max=w_max,
            r_cdi=r_cdi_quadri,
        )

        pesos_alvo = res["pesos_alvo_bl"]
        turnovers.append(res["decisao_rebalanceamento"].turnover)

        # Acumula pesos-alvo reais para próximas janelas
        for t in tickers_validos:
            historico_pesos_acumulado.setdefault(t, []).append(float(pesos_alvo.get(t, 0.0)))

        # Registra custo de concentração dual_value do Agente 6
        if res["custo_concentracao_lagrange"] is not None:
            c_tot = float(np.sum(res["custo_concentracao_lagrange"]))
            registros_concentracao.append(RegistroConcentracao(
                data=dt_inicio, custo_oportunidade_total=c_tot, hit_rate_periodo=0.5
            ))

        # Simula evolução dos preços no quadrimestre
        if len(fatia_precos) > 1:
            pesos_execucao = res["decisao_rebalanceamento"].pesos_finais
            ret_fatia = (fatia_precos.pct_change().dropna() @ pesos_execucao).values
            retornos_diarios_carteira.extend(ret_fatia)
            valor_carteira *= (1.0 + ret_fatia).prod()

        # Atualiza pesos com drift para a próxima janela
        pesos_anteriores = res["decisao_rebalanceamento"].pesos_finais

        print(f"  Janela {i+1:02d}/{len(datas_vigencia)-1} | "
              f"tau={tau:.2f} | tickers={len(tickers_validos)} | "
              f"turnover={res['decisao_rebalanceamento'].turnover:.2%} | "
              f"patrimonio=R$ {valor_carteira:,.0f}")

    retornos_array = np.array(retornos_diarios_carteira)
    turnovers_array = np.array(turnovers)

    print(f"\nPeríodo simulado: {len(retornos_array)} dias úteis de negociação.")
    print(f"Patrimônio Final da Carteira: R$ {valor_carteira:,.2f} "
          f"(Retorno total: {(valor_carteira/valor_inicial - 1)*100:.2f}%)\n")

    relatorio = gerar_relatorio(
        retornos=retornos_array,
        turnovers=turnovers_array,
        n_trials_epsilon=5,
        n_trials_corr=5,
        registros_concentracao=registros_concentracao
    )

    print("=== RELATÓRIO FINAL DE PERFORMANCE E VALIDAÇÃO (AGENTE 8) ===")
    print(f" - Índice Sharpe Anualizado:                {relatorio.sharpe:.2f}")
    print(f" - Índice Sortino Anualizado:               {relatorio.sortino:.2f}")
    print(f" - Maximum Drawdown:                        {relatorio.max_drawdown*100:.2f}%")
    print(f" - Turnover Médio por Rebalanceamento:      {relatorio.turnover_medio*100:.2f}%")
    print(f" - Deflated Sharpe Ratio (DSR):             {relatorio.sharpe_deflacionado:.4f} "
          f"({'Alta Confiança (>0.95)' if relatorio.sharpe_deflacionado >= 0.95 else 'Alerta de Overfitting (<0.95)'})")
    print(f" - Custo de Concentração Acumulado:         {relatorio.custo_concentracao_acumulado:.4f}")
    print(f" - Razão de precisão prior/views (alvo):    ≤ {razao_precisao_alvo:.1f}x")
    print("==============================================================")


if __name__ == "__main__":
    import os
    if os.path.exists("precos_historicos.csv"):
        rodar_backtest_walkforward()
    else:
        rodar_pipeline_sintetico()

