"""
pipeline_quant_ai_colab.py

Script de orquestração dos Agentes 0-8 do projeto Quant AI para o Google Colab.
Integra rigorosamente todas as decisões de projeto da Fase 0 e do Edital:
  - Anti-look-ahead bias via alinhamento estrito de timestamps (Agente 1).
  - Encolhimento de covariância via Ledoit-Wolf (Agente 6).
  - Prior de equilíbrio por otimização reversa (Π = δΣw_mkt) (Agente 6).
  - Purificação de sinal via correlação histórica móvel real (Agente 4).
  - Tratamento correto de liquidez e CDI na simulação de retornos.
  - Recalibração past-only real de epsilon_min e corr_min (Agentes 4 e 8).
  - Métricas de validação completas sem valores hardcoded (Agente 8).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# --- Adiciona o diretório atual e a pasta do Colab ao sys.path ---
diretorio_atual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if diretorio_atual not in sys.path:
    sys.path.insert(0, diretorio_atual)

caminho_colab = '/content/QuantAI/Quant AI'
if os.path.exists(caminho_colab) and caminho_colab not in sys.path:
    sys.path.insert(0, caminho_colab)

try:
    from sklearn.covariance import LedoitWolf
    _LEDOIT_WOLF_DISPONIVEL = True
except ImportError:
    _LEDOIT_WOLF_DISPONIVEL = False

# --- Imports dos 9 agentes ---
from agente_0_universe_data import construir_composicao_pit, derivar_eventos_entrada_saida, selecionar_top_liquidos_por_janela
from agente_1_data_agent import (
    detectar_gaps, preencher_gap_wiener, validar_alinhamento_timestamp, calcular_retorno_realizado_fwd,
)
from agente_2_sentiment_agents import pontuar_noticia, processar_resposta_llm
from agente_3_aggregation_agent import agregar_meta_score
from agente_4_correlation_filter import (
    calcular_correlacao_movel, calcular_score_ajustado, calcular_omega, combinar_omega_final
)
from agente_5_monte_carlo_agent import combinar_eventos_recentes
from agente_6_bl_optimizer import calcular_posterior_bl, bl_optimizer
from agente_7_rebalancing_agent import calcular_banda_nao_negociacao, decidir_execucao, calcular_custos_transacao
from agente_8_validator_agent import (
    gerar_relatorio, RegistroConcentracao, EstadoCalibracaoAgente8, passo_walk_forward,
    calcular_metricas_benchmark,
)
from gerar_graficos_relatorio import plotar_curva_patrimonio_e_drawdown, plotar_alocacao_pesos_historica
from calibracao_epsilon_min import rodar_grid_calibracao as grid_epsilon, resumir_grid as resumo_epsilon
from calibracao_corr_min import rodar_grid_calibracao as grid_corr, resumir_grid as resumo_corr
from mapeamento_percentil_retorno import construir_tabela_percentis_retorno, score_para_q
from processar_noticias_reais import _pasta_persistente  # mesma fonte de verdade do caminho (Drive se montado, local senão)


# --------------------------------------------------------------------------
# Tabela histórica de Selic anual por ano (fallback para CDI diário real)
# --------------------------------------------------------------------------
SELIC_HISTORICA_ANUAL = {
    2018: 0.0650,
    2019: 0.0590,
    2020: 0.0275,
    2021: 0.0425,
    2022: 0.1235,
    2023: 0.1300,
    2024: 0.1075,
    2025: 0.1200,
    2026: 0.1200,
}

def obter_taxa_selic_ano(ano: int) -> float:
    return SELIC_HISTORICA_ANUAL.get(ano, 0.10)


def carregar_composicao_pit(caminho_csv: str = "composicao_pit_ibovespa.csv") -> pd.DataFrame:
    return pd.read_csv(caminho_csv, parse_dates=["data_vigencia"])


def carregar_precos_reais(caminho_csv: str = "precos_historicos.csv") -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, parse_dates=["data"])
    return df


# --------------------------------------------------------------------------
# Pipeline de UM rebalanceamento
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
    retorno_janela_anterior: float = 0.0,
) -> dict:
    """Encadeia Agentes 6 -> 7 para um único instante de rebalanceamento."""
    E_R = calcular_posterior_bl(Pi, Sigma, tau, tickers, Q_por_ativo, Omega_por_ativo)
    resultado_bl = bl_optimizer(E_R, Sigma, delta, w_max=w_max, r_cdi=r_cdi)

    pesos_alvo_equities = pd.Series(resultado_bl.pesos[:len(tickers)], index=tickers)
    # Mesmo ruído numérico de ponto flutuante que já tratamos nos pesos de
    # equity (Agente 7) também aparece aqui -- o cvxpy pode devolver algo
    # como -1e-12 em vez de exatamente 0.0 para uma restrição w>=0 ativa
    # no limite. Sem o clip, isso quebra o gráfico de área empilhada
    # (exige coluna estritamente >=0 ou <=0 em TODAS as linhas -- um
    # único valor residual negativo já derruba o gráfico inteiro).
    peso_cdi = max(0.0, float(resultado_bl.pesos[len(tickers)])) if r_cdi is not None else 0.0

    bandas = calcular_banda_nao_negociacao(historico_pesos_alvo, fator_liquidez)
    decisao = decidir_execucao(pesos_alvo_equities, pesos_atuais_com_drift, bandas)

    # IR incide sobre GANHO REALIZADO, não sobre volume negociado.
    # Sem rastreamento FIFO completo (limitação já declarada no projeto),
    # aproxima-se o ganho por: turnover desta janela (fração da carteira
    # de fato negociada) x retorno acumulado na janela ANTERIOR (proxy de
    # "quanto valorizou" a posição que está sendo parcialmente vendida
    # agora). Calculado aqui, não antes, porque decisao.turnover só
    # existe depois de decidir_execucao rodar.
    ganho_realizado_estimado = 2.0 * decisao.turnover * valor_carteira * retorno_janela_anterior
    custos = calcular_custos_transacao(
        pesos_atuais_com_drift, decisao.pesos_finais, valor_carteira,
        ganho_realizado=ganho_realizado_estimado,
    )

    return {
        "pesos_alvo_bl": pesos_alvo_equities,
        "peso_cdi": peso_cdi,
        "decisao_rebalanceamento": decisao,
        "custos": custos,
        "custo_concentracao_lagrange": resultado_bl.custo_concentracao_por_ativo,
    }


# --------------------------------------------------------------------------
# Backtest Walk-Forward Completo
# --------------------------------------------------------------------------

def rodar_backtest_walkforward(
    caminho_pit: str = "composicao_pit_ibovespa.csv",
    caminho_precos: str = "precos_historicos.csv",
    delta: float = 3.0,
    w_max: float = 0.20,
    valor_inicial: float = 100_000.0,
    razao_precisao_alvo: float = 3.0,
    escala_q: float = 0.05,
    n_ativos_liquidos: int | None = 15,
) -> None:
    print("\n========================================================")
    print("  INICIANDO BACKTEST WALK-FORWARD QUANT AI (2018-2026)")
    print("========================================================\n")

    composicao_pit = carregar_composicao_pit(caminho_pit)

    # Corte para os N ativos mais líquidos de cada janela (decisão de
    # projeto da Fase 0, nunca antes integrada ao pipeline -- confirmado
    # rodando com 56-80 ativos em vez dos 10-15 documentados). Usa
    # peso_pct (já disponível no CSV oficial B3, via Agente 0) como proxy
    # de liquidez -- corte é point-in-time, refeito dentro de cada janela.
    if n_ativos_liquidos is not None:
        n_antes = composicao_pit.groupby("data_vigencia")["ticker"].nunique().mean()
        composicao_pit = selecionar_top_liquidos_por_janela(composicao_pit, n_ativos=n_ativos_liquidos)
        n_depois = composicao_pit.groupby("data_vigencia")["ticker"].nunique().mean()
        print(f"Universo cortado por liquidez: média de {n_antes:.0f} -> {n_depois:.0f} ativos por janela "
              f"(teto={n_ativos_liquidos}).")

    precos_painel = carregar_precos_reais(caminho_precos)

    datas_vigencia = sorted(composicao_pit["data_vigencia"].unique())
    print(f"Janelas PIT encontradas: {len(datas_vigencia)} quadrimestres "
          f"({datas_vigencia[0].date()} a {datas_vigencia[-1].date()}).")

    retornos_diarios_carteira = []
    datas_diarias_carteira = []
    turnovers = []
    registros_concentracao = []
    valor_carteira = valor_inicial
    # Rastreia o retorno da carteira na janela ANTERIOR -- usado só para
    # a estimativa de ganho realizado do IR (ver comentário em
    # rodar_um_rebalanceamento). Começa em 0.0: a primeira janela não
    # tem "período anterior" para estimar ganho, então o IR dela é 0
    # por construção (não há venda de posição ainda aberta há mais tempo).
    retorno_janela_anterior = 0.0
    custo_total_acumulado = 0.0
    historico_pesos_janelas: list[dict] = []
    datas_janelas_rebalanceamento = []

    df_precos = precos_painel.pivot(index="data", columns="ticker", values="close").sort_index()

    historico_pesos_acumulado: dict[str, list[float]] = {}
    pesos_anteriores: pd.Series | None = None
    estado_calib = EstadoCalibracaoAgente8()

    # Define grades de candidatos para derivação dinâmica de n_trials
    candidatos_eps = np.concatenate([[0.0], np.logspace(-4, -0.3, 12)])
    candidatos_corr = np.linspace(0.0, 0.60, 13)

    for i in range(len(datas_vigencia) - 1):
        dt_inicio = datas_vigencia[i]
        dt_fim = datas_vigencia[i + 1]
        selic_anual_janela = obter_taxa_selic_ano(dt_inicio.year)

        universe_janela = composicao_pit[composicao_pit["data_vigencia"] == dt_inicio]["ticker"].tolist()
        cols_disponiveis = [t for t in universe_janela if t in df_precos.columns]
        precos_historicos = df_precos.loc[:dt_inicio, cols_disponiveis].dropna(how="all", axis=1)

        # Tabela de percentis de retorno (substitui a escala fixa Q=score*0.15).
        # Reconstruída a cada janela usando SÓ precos_historicos (já cortado até
        # dt_inicio, sem look-ahead). Índice sintético (média equal-weight dos
        # ativos disponíveis) usado como fallback para ativos sem histórico
        # próprio suficiente (ex: IPO recente dentro da janela).
        precos_para_tabela = precos_historicos.copy()
        if not precos_para_tabela.empty:
            precos_para_tabela["_INDICE_SINTETICO"] = precos_para_tabela.mean(axis=1, skipna=True)
            tabela_percentis = construir_tabela_percentis_retorno(
                precos_para_tabela, tickers=cols_disponiveis,
                ticker_indice_fallback="_INDICE_SINTETICO",
            )
        else:
            tabela_percentis = None
        tickers_validos = precos_historicos.columns.tolist()

        if len(tickers_validos) == 0:
            continue

        retornos_hist = np.log(precos_historicos / precos_historicos.shift(1)).dropna()
        if len(retornos_hist) < 30:
            continue

        # Item E: Covariância encolhida via Ledoit-Wolf (Fase 0) — Anualizada para 252 dias
        if globals().get("_LEDOIT_WOLF_DISPONIVEL", False):
            try:
                Sigma_diaria = LedoitWolf().fit(retornos_hist.values).covariance_
            except Exception:
                Sigma_diaria = retornos_hist.cov().values + np.eye(len(tickers_validos)) * 1e-4
        else:
            Sigma_diaria = retornos_hist.cov().values + np.eye(len(tickers_validos)) * 1e-4

        Sigma = Sigma_diaria * 252.0  # Covariância Anualizada

        # Item D: Prior de equilíbrio Pi por Otimização Reversa (Π = δΣw_mkt) em escala Anual
        w_mkt = np.ones(len(tickers_validos)) / len(tickers_validos)  # Proxy equal-weighted da janela
        Pi = delta * (Sigma @ w_mkt)

        # Calculo de retorno proxy do índice para o filtro de correlação (Item C)
        retornos_mkt = retornos_hist.mean(axis=1)

        # CORREÇÃO (bug real, confirmado por gráfico de alocação sem
        # nenhum sentimento aplicado): este caminho era fixo e local --
        # deixou de encontrar o arquivo depois que processar_noticias_reais.py
        # passou a salvar no Drive (para persistir entre sessões). Sem o
        # arquivo, o backtest inteiro rodava com Q_por_ativo/Omega_por_ativo
        # sempre vazios, silenciosamente -- BL caindo no prior puro.
        caminho_sentimento = os.path.join(_pasta_persistente(), "sentimento_agregado_diario.csv")
        if i == 0 and not os.path.exists(caminho_sentimento):
            print(
                f"\n[ALERTA CRÍTICO] '{caminho_sentimento}' não encontrado -- o backtest vai "
                f"rodar SEM NENHUM sinal de sentimento (Black-Litterman cai no prior puro em "
                f"toda janela). Rode processar_noticias_reais.py antes desta célula.\n"
            )
        Q_por_ativo = {}
        Omega_por_ativo = {}

        # Item F: Recalibração de epsilon_min e corr_min com histórico past-only real
        dt_atual_date = dt_inicio.date()
        
        def _recalib_epsilon(hist_df):
            if hist_df is not None and not hist_df.empty and len(hist_df) >= 40:
                try:
                    res_grid = grid_epsilon(hist_df, candidatos_eps, n_folds=3, janela_min_treino_dias=30)
                    res_resumo = resumo_epsilon(res_grid)
                    if not res_resumo.empty:
                        return float(res_resumo.iloc[0]["epsilon_min"])
                except Exception as e:
                    print(f"[AVISO] Recalibração de epsilon_min falhou em {dt_atual_date}: {e} -- usando default 0.01.")
            return 0.01

        def _recalib_corr(hist_df):
            if hist_df is not None and not hist_df.empty and len(hist_df) >= 40:
                try:
                    res_grid = grid_corr(hist_df, candidatos_corr, n_folds=3, janela_min_treino_dias=30)
                    res_resumo = resumo_corr(res_grid)
                    if not res_resumo.empty:
                        return float(res_resumo.iloc[0]["corr_min"])
                except Exception as e:
                    print(f"[AVISO] Recalibração de corr_min falhou em {dt_atual_date}: {e} -- usando default 0.25.")
            return 0.25

        # Carrega histórico de notícias acumulado até t-1 se disponível
        hist_sent_acumulado = None
        if os.path.exists(caminho_sentimento):
            df_s_full = pd.read_csv(caminho_sentimento)
            if "data" in df_s_full.columns:
                df_s_full["data"] = pd.to_datetime(df_s_full["data"])
                hist_sent_acumulado = df_s_full[df_s_full["data"] < dt_inicio].copy()
                if not hist_sent_acumulado.empty and "retorno_realizado_fwd" not in hist_sent_acumulado.columns:
                    rets_fwd = []
                    for idx_s, r_s in hist_sent_acumulado.iterrows():
                        tk_tmp = r_s["ticker"]
                        dt_tmp = r_s["data"]
                        if tk_tmp in df_precos.columns and dt_tmp in df_precos.index:
                            sub_p = df_precos.loc[dt_tmp:, tk_tmp].dropna()
                            if len(sub_p) > 5:
                                ret_5d = np.log(sub_p.iloc[5] / sub_p.iloc[0])
                                rets_fwd.append(ret_5d)
                            else:
                                rets_fwd.append(0.0)
                        else:
                            rets_fwd.append(0.0)
                    hist_sent_acumulado["retorno_realizado_fwd"] = rets_fwd

                # CORREÇÃO (auditoria externa, confirmada): calibracao_corr_min.py
                # exige a coluna 'correlacao', que nunca era construída aqui -- a
                # recalibração de corr_min lançava ValueError sempre, capturada
                # silenciosamente pelo except acima, caindo sempre no default 0.25
                # (nunca recalibrando de verdade). Calcula a correlação móvel real
                # (mesma função usada no fluxo diário, Agente 4) para cada linha,
                # usando apenas dado até a própria data da linha (sem look-ahead --
                # correlação móvel é causal por construção, mas o .loc[:dt_tmp]
                # abaixo garante isso explicitamente mesmo assim).
                if not hist_sent_acumulado.empty and "correlacao" not in hist_sent_acumulado.columns:
                    try:
                        retornos_mkt_completo = np.log(df_precos / df_precos.shift(1)).mean(axis=1)
                    except Exception:
                        retornos_mkt_completo = None

                    cache_corr_series: dict[str, pd.Series] = {}
                    correlacoes = []
                    for idx_s, r_s in hist_sent_acumulado.iterrows():
                        tk_tmp = r_s["ticker"]
                        dt_tmp = r_s["data"]
                        corr_val = 0.50
                        if retornos_mkt_completo is not None and tk_tmp in df_precos.columns:
                            if tk_tmp not in cache_corr_series:
                                ret_tk = np.log(df_precos[tk_tmp] / df_precos[tk_tmp].shift(1))
                                cache_corr_series[tk_tmp] = calcular_correlacao_movel(
                                    ret_tk, retornos_mkt_completo, janela_dias=60
                                )
                            serie_ate_dt = cache_corr_series[tk_tmp].loc[:dt_tmp].dropna()
                            if not serie_ate_dt.empty:
                                corr_val = float(serie_ate_dt.iloc[-1])
                        correlacoes.append(corr_val)
                    hist_sent_acumulado["correlacao"] = correlacoes

        estado_calib = passo_walk_forward(
            estado_calib, dt_atual_date,
            _recalib_epsilon, _recalib_corr,
            historico_ate_t_menos_1=hist_sent_acumulado
        )

        if os.path.exists(caminho_sentimento):
            df_sent = pd.read_csv(caminho_sentimento)
            if "data" in df_sent.columns and not pd.api.types.is_datetime64_any_dtype(df_sent["data"]):
                df_sent["data"] = pd.to_datetime(df_sent["data"])

            fatia_sent = df_sent[(df_sent["data"] <= dt_inicio) & (df_sent["ticker"].isin(tickers_validos))].copy()
            
            if not fatia_sent.empty:
                col_ts = "timestamp_noticia" if "timestamp_noticia" in fatia_sent.columns else "data"
                df_ev_valida = pd.DataFrame({
                    "ticker": fatia_sent["ticker"],
                    "timestamp_noticia": pd.to_datetime(fatia_sent[col_ts]),
                    "timestamp_decisao": pd.to_datetime(dt_inicio),
                    "timestamp_preco": pd.to_datetime(dt_inicio)
                })
                violacoes = validar_alinhamento_timestamp(df_ev_valida)
                if violacoes:
                    print(f"[AVISO LOG] {len(violacoes)} notícias violaram a regra anti-look-ahead em {dt_inicio.date()} e foram descartadas.")
                    indices_validos = [i for i, ev in df_ev_valida.iterrows() if ev["timestamp_noticia"] < ev["timestamp_decisao"]]
                    fatia_sent = fatia_sent.loc[indices_validos]

                eventos_sent = fatia_sent.rename(columns={"data": "data_noticia"})
                omega_llm_map = {}
                omega_temp_map = {}

                for tk_s in tickers_validos:
                    df_tk = eventos_sent[eventos_sent["ticker"] == tk_s]
                    if not df_tk.empty:
                        q_dec, var_temp = combinar_eventos_recentes(
                            eventos_ticker=df_tk,
                            data_alvo=dt_inicio,
                            janela_memoria_dias=15,
                            lambda_decaimento=0.15,
                            n_simulacoes=1000
                        )
                        
                        if not np.isnan(q_dec) and q_dec != 0.0:
                            serie_tk = retornos_hist[tk_s]
                            corr_serie = calcular_correlacao_movel(serie_tk, retornos_mkt, janela_dias=60)
                            corr_real = float(corr_serie.iloc[-1]) if not corr_serie.dropna().empty else 0.50

                            score_purificado = calcular_score_ajustado(
                                float(q_dec), corr_real, corr_min=estado_calib.corr_min
                            )

                            # CORREÇÃO: escala fixa (score * 0.15, igual para todo
                            # ativo) substituída por mapeamento via percentil
                            # empírico de retorno, específico por ativo (com
                            # fallback ao índice sintético para ativos sem
                            # histórico suficiente). Ver mapeamento_percentil_retorno.py.
                            if tabela_percentis is not None and tk_s in tabela_percentis.breakpoints_por_ticker:
                                Q_por_ativo[tk_s] = score_para_q(score_purificado, tabela_percentis, tk_s)
                            else:
                                # Fallback final -- só se a tabela não pôde ser
                                # construída nesta janela (ex: sem preço nenhum
                                # disponível ainda). Documentado, não silencioso.
                                Q_por_ativo[tk_s] = float(score_purificado) * 0.15
                            
                            s_ult = df_tk.sort_values("data_noticia").iloc[-1]
                            scores_arr = np.array([[s_ult["s1"], s_ult["s2"], s_ult["s3"]]])
                            n_not_arr = np.array([s_ult["n_noticias"]])
                            om_llm = float(calcular_omega(scores_arr, n_not_arr, epsilon_min=estado_calib.epsilon_min)[0])
                            
                            omega_llm_map[tk_s] = max(om_llm, estado_calib.epsilon_min)
                            omega_temp_map[tk_s] = var_temp if not np.isnan(var_temp) else 0.0

                omega_combinado = combinar_omega_final(omega_llm_map, omega_temp_map)
                for tk_s, om_final in omega_combinado.items():
                    if not np.isnan(om_final):
                        Omega_por_ativo[tk_s] = max(om_final, estado_calib.epsilon_min)

        # CORREÇÃO P0.5 (auditoria externa, confirmada): NÃO preencher Q=0.0/
        # Omega=0.03 para ativos sem notícia -- isso cria uma VIEW VÁLIDA de
        # retorno zero (que puxa a posterior para baixo, favorecendo CDI),
        # quando o correto é AUSÊNCIA de view. O mecanismo de montar_views_validas
        # (agente_6_bl_optimizer.py), já testado, trata corretamente um ticker
        # ausente de Q_por_ativo/Omega_por_ativo como "sem view" -- a posterior
        # dele colapsa exatamente ao prior Pi. Não precisa (e não deve) inventar
        # uma entrada artificial aqui.
        tickers_com_view_real = list(Q_por_ativo.keys())

        if tickers_com_view_real:
            q_sim = np.array([Q_por_ativo[t] for t in tickers_com_view_real])
            omega_sim = np.array([Omega_por_ativo[t] for t in tickers_com_view_real])
            mediana_omega = float(np.median(omega_sim))
        else:
            # Nenhum ativo com sinal real nesta janela -- usa piso padrão só
            # para a heurística de calibração de tau abaixo, sem inventar
            # nenhuma view no Q_por_ativo/Omega_por_ativo reais.
            mediana_omega = 0.03

        mediana_diag_sigma = float(np.median(np.diag(Sigma)))
        tau = razao_precisao_alvo * mediana_omega / max(mediana_diag_sigma, 1e-8)
        tau = max(tau, 0.01)

        ew = 1.0 / len(tickers_validos)
        historico_dict = {}
        for t in tickers_validos:
            raw = historico_pesos_acumulado.get(t, [])
            tail = raw[-60:]
            padded = [ew] * (60 - len(tail)) + tail
            historico_dict[t] = padded
        historico_df = pd.DataFrame(historico_dict)

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

        fatia_precos = df_precos.loc[dt_inicio:dt_fim, tickers_validos].dropna(how="all")
        
        # CORREÇÃO P0.1 (auditoria externa, confirmada): Pi = delta*Sigma@w_mkt e Q
        # (escala "alpha") estão em RETORNO EXCEDENTE (prêmio de risco), não retorno
        # total. Passar a taxa Selic total (~10-13% a.a.) como r_cdi comparava
        # diretamente retorno excedente (equities) contra retorno total (CDI) na
        # mesma função-objetivo -- isso empurrava a solução para CDI de forma
        # estruturalmente enviesada, mesmo com a otimização sendo conjunta/correta.
        # Como o CDI É o ativo livre de risco, seu retorno EXCEDENTE sobre si mesmo
        # é 0 por definição -- não a taxa Selic observada. selic_anual_janela
        # continua usada abaixo para simular o retorno TOTAL realizado (equities
        # em retorno total de preço + CDI em retorno total da Selic) -- a mistura
        # excesso/total só era um problema DENTRO da função-objetivo do otimizador,
        # não na simulação de patrimônio.
        r_cdi_anual = 0.0

        res = rodar_um_rebalanceamento(
            tickers_validos, Pi, Sigma, tau, delta,
            Q_por_ativo, Omega_por_ativo, pesos_atuais, historico_df,
            fator_liquidez, valor_carteira=valor_carteira, w_max=w_max,
            r_cdi=r_cdi_anual, retorno_janela_anterior=retorno_janela_anterior,
        )

        pesos_alvo = res["pesos_alvo_bl"]
        turnovers.append(res["decisao_rebalanceamento"].turnover)

        for t in tickers_validos:
            historico_pesos_acumulado.setdefault(t, []).append(float(pesos_alvo.get(t, 0.0)))

        # Item I: Hit rate real calculado sobre ativos que atingiram o teto de concentração
        hit_rate_real = 0.50
        if res["custo_concentracao_lagrange"] is not None:
            c_tot = float(np.sum(res["custo_concentracao_lagrange"]))
            # Identifica quais ativos bateram no teto w_max
            ativos_teto = [tk for tk, w in pesos_alvo.items() if w >= (w_max - 0.001)]
            if ativos_teto and len(fatia_precos) > 1:
                rets_futuros = (fatia_precos.iloc[-1] / fatia_precos.iloc[0] - 1.0)
                acertos = [1 for tk in ativos_teto if rets_futuros.get(tk, 0) > 0]
                hit_rate_real = float(len(acertos) / len(ativos_teto))

            registros_concentracao.append(RegistroConcentracao(
                data=dt_inicio, custo_oportunidade_total=c_tot, hit_rate_periodo=hit_rate_real
            ))

        # Item B: Simulação rigorosa do retorno incluindo o peso do CDI
        if len(fatia_precos) > 1:
            pesos_execucao = res["decisao_rebalanceamento"].pesos_finais  # equities (soma 1.0)
            peso_cdi = res["peso_cdi"] if res["peso_cdi"] is not None else 0.0
            peso_equities_efetivo = max(0.0, 1.0 - peso_cdi)

            ret_equities = (fatia_precos.pct_change().dropna() @ (pesos_execucao * peso_equities_efetivo)).values
            selic_diaria_j = (1.0 + selic_anual_janela) ** (1.0 / 252.0) - 1.0
            ret_cdi_diario = peso_cdi * selic_diaria_j

            ret_fatia = ret_equities + ret_cdi_diario
            datas_ret = fatia_precos.index[1:]
            retornos_diarios_carteira.extend(ret_fatia)
            datas_diarias_carteira.extend(datas_ret.tolist())

            # CORREÇÃO (bug confirmado: custos calculados desde o início do
            # projeto, mas NUNCA descontados do patrimônio -- todo Sharpe,
            # retorno total e drawdown reportado até agora era BRUTO, não
            # líquido). O custo é descontado UMA VEZ no início da janela
            # (é quando o rebalanceamento acontece), antes do retorno de
            # mercado do período ser aplicado.
            valor_carteira -= res["custos"].custo_total
            custo_total_acumulado += res["custos"].custo_total
            valor_carteira = max(0.0, valor_carteira)  # nunca negativo por custo isolado
            valor_carteira *= (1.0 + ret_fatia).prod()

            retorno_janela_anterior = float((1.0 + ret_fatia).prod() - 1.0)

        # Acumula pesos por janela para o gráfico de alocação histórica (soma exata = 100%)
        peso_cdi_val = res["peso_cdi"] if res["peso_cdi"] is not None else 0.0
        fator_equities = max(0.0, 1.0 - peso_cdi_val)
        pesos_execucao_series = res["decisao_rebalanceamento"].pesos_finais * fator_equities
        
        linha_pesos = pesos_execucao_series.to_dict()
        linha_pesos["CDI_RF"] = peso_cdi_val
        historico_pesos_janelas.append(linha_pesos)
        datas_janelas_rebalanceamento.append(dt_inicio)

        pesos_anteriores = res["decisao_rebalanceamento"].pesos_finais

        print(f"  Janela {i+1:02d}/{len(datas_vigencia)-1} | "
              f"tau={tau:.2f} | tickers={len(tickers_validos)} | "
              f"turnover={res['decisao_rebalanceamento'].turnover:.2%} | "
              f"patrimonio=R$ {valor_carteira:,.0f}")

    retornos_array = np.array(retornos_diarios_carteira)
    turnovers_array = np.array(turnovers)

    print(f"\nPeríodo simulado: {len(retornos_array)} dias úteis de negociação.")
    print(f"Patrimônio Final da Carteira: R$ {valor_carteira:,.2f} "
          f"(Retorno total LÍQUIDO de custos: {(valor_carteira/valor_inicial - 1)*100:.2f}%)")
    print(f"Custo total acumulado (emolumentos + IR): R$ {custo_total_acumulado:,.2f} "
          f"({custo_total_acumulado/valor_inicial*100:.2f}% do capital inicial)\n")

    # Item G: n_trials derivados do tamanho real dos candidatos dos grids (13 cada)
    relatorio = gerar_relatorio(
        retornos=retornos_array,
        turnovers=turnovers_array,
        n_trials_epsilon=len(candidatos_eps),
        n_trials_corr=len(candidatos_corr),
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
    print(f" - Razão de precisão prior/views (alvo):    <= {razao_precisao_alvo:.1f}x")
    print("==============================================================")

    try:
        import matplotlib
        matplotlib.use("Agg")

        datas_idx = pd.DatetimeIndex(datas_diarias_carteira)
        # CORREÇÃO (mesma inconsistência que a auditoria externa apontou):
        # a linha do CDI no gráfico usava 10% fixo, divergindo da tabela
        # SELIC_HISTORICA_ANUAL real já usada na simulação de retorno da
        # carteira -- o gráfico mentia sobre o próprio benchmark que mostra.
        selic_diaria_por_data = np.array([
            (1.0 + obter_taxa_selic_ano(d.year)) ** (1.0 / 252.0) - 1.0
            for d in datas_idx
        ])
        retornos_cdi_diarios = selic_diaria_por_data

        retornos_ibov_real = None
        try:
            import yfinance as yf
            df_bvsp = yf.download("^BVSP", start=datas_idx[0], end=datas_idx[-1], progress=False)
            if not df_bvsp.empty:
                serie_fechamento = df_bvsp["Close"]
                if isinstance(serie_fechamento, pd.DataFrame):
                    serie_fechamento = serie_fechamento.iloc[:, 0]
                ret_bvsp = serie_fechamento.pct_change().reindex(datas_idx).fillna(0.0).values
                if len(ret_bvsp) == len(retornos_array):
                    retornos_ibov_real = ret_bvsp
        except Exception:
            pass

        # Métricas de Ibovespa e CDI para a tabela executiva do relatório --
        # MESMA fórmula usada na estratégia (calcular_metricas_benchmark
        # reusa calcular_sharpe/sortino/max_drawdown), para a comparação
        # não favorecer nenhum lado com metodologia diferente.
        #
        # CORREÇÃO (achado real: Sharpe do CDI saiu 36,3 numa rodada de
        # relatório): Sharpe = média(excesso) / desvio-padrão(excesso).
        # O CDI, sendo o próprio ativo livre de risco, tem uma curva de
        # juros compostos quase perfeitamente lisa -- desvio-padrão diário
        # próximo de zero. Dividir uma média positiva por um desvio quase
        # nulo produz um valor matematicamente "correto" mas sem
        # significado de comparação: Sharpe mede retorno por unidade de
        # risco ACIMA do ativo livre de risco, e o CDI não tem prêmio de
        # risco sobre si mesmo, por definição. Reportado como não
        # aplicável, não como um número que parece (e não é) um resultado
        # bom.
        print("\n=== MÉTRICAS DE BENCHMARK (para a tabela executiva do relatório) ===")
        m_cdi = calcular_metricas_benchmark(retornos_cdi_diarios)
        print(f" CDI        | Retorno Total: {m_cdi.retorno_total*100:6.2f}% | "
              f"Sharpe: N/A (ativo livre de risco, sem prêmio sobre si mesmo) | "
              f"Sortino: N/A | MDD: {m_cdi.max_drawdown*100:6.2f}%")
        if retornos_ibov_real is not None:
            m_ibov = calcular_metricas_benchmark(retornos_ibov_real)
            print(f" Ibovespa   | Retorno Total: {m_ibov.retorno_total*100:6.2f}% | "
                  f"Sharpe: {m_ibov.sharpe:5.2f} | Sortino: {m_ibov.sortino:5.2f} | "
                  f"MDD: {m_ibov.max_drawdown*100:6.2f}%")
        else:
            print(" Ibovespa   | indisponível nesta rodada (falha ao baixar ^BVSP via yfinance)")
        print(" (Turnover e Sharpe Deflacionado não se aplicam a um índice passivo -- use '-' na tabela)")
        print(" (Sharpe/Sortino do CDI não se aplicam -- use '-' na tabela, ver comentário no código)")
        print("==============================================================")


        fig1 = plotar_curva_patrimonio_e_drawdown(
            datas=datas_idx,
            retornos_carteira=retornos_array,
            retornos_ibov=retornos_ibov_real,
            retornos_cdi=retornos_cdi_diarios,
            valor_inicial=valor_inicial,
        )
        print(f"\n[GRAFICO] Curva de patrimônio e drawdown salva em: {fig1}")

        if historico_pesos_janelas:
            df_pesos_hist = pd.DataFrame(
                historico_pesos_janelas,
                index=pd.DatetimeIndex(datas_janelas_rebalanceamento)
            ).fillna(0.0)
            fig2 = plotar_alocacao_pesos_historica(df_pesos_hist)
            print(f"[GRAFICO] Alocação histórica de pesos salva em: {fig2}")

    except Exception as e_graf:
        print(f"[AVISO] Geração de gráficos falhou: {e_graf}")


if __name__ == "__main__":
    import os
    if os.path.exists("precos_historicos.csv"):
        rodar_backtest_walkforward()
