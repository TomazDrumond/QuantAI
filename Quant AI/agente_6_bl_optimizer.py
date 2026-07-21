"""
agente_6_bl_optimizer.py

Agente 6 do pipeline Quant AI — BL Optimizer.

Recebe E[R] (retorno esperado posterior do Black-Litterman) e Sigma
(covariância com shrinkage Ledoit-Wolf, Fase 0) e resolve a alocação de
pesos via otimização média-variância convexa.

Decisões de projeto já fechadas e implementadas aqui:
  - LONG-ONLY: w_i >= 0 para todo ativo (sem venda a descoberto — evita
    custo de aluguel de ações / BTC na B3, ainda não modelado em nenhum
    agente do pipeline).
  - TETO FIXO DE CONCENTRAÇÃO: w_i <= 0.20 para todo ativo, com universo
    de até 15 ativos. Testado numericamente e confirmado factível (soma
    máxima possível = 15 x 20% = 300% >> 100% exigido), ao contrário da
    tentativa de aplicar a regra 5/10/40 (UCITS) literalmente, que se
    mostrou matematicamente inviável para um universo de 10-15 ativos
    (mínimo necessário para satisfazer 5/10/40 com soma=1 é 16 ativos).
  - Sem quantização em múltiplos de 5% — decisão explícita de não pagar
    o custo computacional de um MIQP (branch-and-bound), mantendo o
    problema 100% convexo contínuo.
  - Resolução via cvxpy: garante matematicamente convergência ao ótimo
    global (o problema é convexo), diferente de solvers genéricos como
    scipy.optimize.SLSQP, que não garantem isso formalmente.

Custo de oportunidade da concentração:
  O multiplicador de Lagrange (dual_value) da restrição w <= w_max mede
  quanto retorno esperado marginal foi abdicado, por ativo, só para
  respeitar o teto de segurança. Isso deve ser acumulado pelo Agente 8
  ao longo do backtest walk-forward e reportado como métrica formal
  (decisão já aprovada nesta etapa do projeto).

NOTA DE AMBIENTE: este script depende de cvxpy, que precisa ser
instalado no ambiente de execução real (ex: Antigravity, com internet):
    pip install cvxpy
A lógica de otimização foi validada nesta sessão com um solver
equivalente (scipy.optimize/SLSQP) sobre os mesmos dados de teste, com
resultado consistente (restrição de 20% ativa corretamente no ativo de
maior sinal, soma dos pesos = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

try:
    import cvxpy as cp
    _CVXPY_DISPONIVEL = True
except ImportError:
    _CVXPY_DISPONIVEL = False


# --------------------------------------------------------------------------
# 1. Otimizador principal (Agente 6)
# --------------------------------------------------------------------------

@dataclass
class ResultadoOtimizacao:
    pesos: np.ndarray
    custo_concentracao_por_ativo: np.ndarray  # dual_value da restrição w <= w_max
    status: str


def montar_views_validas(
    tickers: list[str],
    Q_por_ativo: dict[str, float],
    Omega_por_ativo: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Filtra os ativos com view válida (Omega não-NaN, vindo do Agente 4 --
    ver calcular_omega em agente_4_correlation_filter.py, que retorna NaN
    quando 0 LLMs responderam para aquele ativo/dia).

    Monta a matriz de seleção de views P (K x N, K = nº de ativos com
    view válida, N = nº total de ativos no universo) no formato exigido
    pela equação posterior do BL -- cada linha de P é um one-hot
    indicando qual ativo aquela view absoluta se refere.

    Retorna: (P, Q_validos, Omega_diag_validos, tickers_com_view)
    Se K=0 (nenhum ativo com view válida no dia), retorna P vazio -- a
    posterior cai no prior puro (Pi), sem quebrar a equação.
    """
    tickers_com_view = [
        t for t in tickers
        if t in Q_por_ativo and t in Omega_por_ativo and not np.isnan(Omega_por_ativo[t])
    ]

    N = len(tickers)
    K = len(tickers_com_view)
    P = np.zeros((K, N))
    for k, ticker in enumerate(tickers_com_view):
        P[k, tickers.index(ticker)] = 1.0

    Q_validos = np.array([Q_por_ativo[t] for t in tickers_com_view])
    Omega_diag = np.array([Omega_por_ativo[t] for t in tickers_com_view])

    return P, Q_validos, Omega_diag, tickers_com_view


def calcular_posterior_bl(
    Pi: np.ndarray,
    Sigma: np.ndarray,
    tau: float,
    tickers: list[str],
    Q_por_ativo: dict[str, float],
    Omega_por_ativo: dict[str, float],
) -> np.ndarray:
    """
    Equação posterior do Black-Litterman, já filtrando ativos sem view
    válida (Omega=NaN) via montar_views_validas:

        E[R] = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1
               [(tau*Sigma)^-1 Pi + P^T Omega^-1 Q]

    Ativos sem view válida (K exclui esse ativo de P) simplesmente não
    recebem nenhum "puxão" de Q -- a posterior deles fica igual ao prior
    de equilíbrio Pi, propriamente, sem NaN vazando para o solver do
    Agente 6.
    """
    N = len(tickers)
    P, Q_validos, Omega_diag, tickers_com_view = montar_views_validas(tickers, Q_por_ativo, Omega_por_ativo)

    tau_sigma_inv = np.linalg.inv(tau * Sigma)

    if len(tickers_com_view) == 0:
        # Nenhuma view válida no dia -- posterior colapsa exatamente ao prior
        return Pi.copy()

    omega_inv = np.diag(1.0 / Omega_diag)
    termo_precisao = tau_sigma_inv + P.T @ omega_inv @ P
    termo_media = tau_sigma_inv @ Pi + P.T @ omega_inv @ Q_validos

    E_R = np.linalg.solve(termo_precisao, termo_media)
    return E_R


def bl_optimizer(
    E_R: np.ndarray,
    Sigma: np.ndarray,
    delta: float,
    w_max: float = 0.20,
    r_cdi: float | None = None,
) -> ResultadoOtimizacao:
    """
    Resolve: max_w  E_R^T w - (delta/2) w^T Sigma w
             s.a.    sum(w) == 1
                     w >= 0                    (long-only)
                     w_equities <= w_max        (teto de concentração, 20%, só para equities)

    Se `r_cdi` for informado (retorno do CDI no período, ex: 0.045 para
    4,5% no período de rebalanceamento), o CDI entra como um ativo
    sintético adicional: variância zero, covariância zero com todos os
    outros ativos (é o ativo livre de risco, por definição). Isso
    implementa a separação de Tobin (two-fund separation) -- quando o
    retorno esperado ajustado por sentimento dos equities não compensa
    o risco extra sobre o CDI, o otimizador aloca capital para CDI de
    forma orgânica, via a própria matemática de delta, sem regra ad-hoc.

    O teto de concentração de 20% NÃO se aplica ao CDI -- ele é o "porto
    seguro" da carteira, não uma posição de risco de concentração no
    mesmo sentido que uma ação individual.
    """
    if not _CVXPY_DISPONIVEL:
        raise ImportError(
            "cvxpy não está instalado neste ambiente. Rode `pip install cvxpy` "
            "no ambiente de execução real (Antigravity) antes de chamar bl_optimizer."
        )

    n_equities = len(E_R)
    if Sigma.shape != (n_equities, n_equities):
        raise ValueError(f"Sigma precisa ser {n_equities}x{n_equities}, recebido {Sigma.shape}")

    if r_cdi is None:
        n = n_equities
        E_R_ext = E_R
        Sigma_ext = Sigma
        w_max_vetor = np.full(n, w_max)
    else:
        n = n_equities + 1  # +1 = CDI sintético, último índice
        E_R_ext = np.concatenate([E_R, [r_cdi]])
        Sigma_ext = np.zeros((n, n))
        Sigma_ext[:n_equities, :n_equities] = Sigma  # covariância com CDI = 0 (ativo livre de risco)
        w_max_vetor = np.concatenate([np.full(n_equities, w_max), [1.0]])  # CDI sem teto de concentração

    w = cp.Variable(n)
    risco = cp.quad_form(w, Sigma_ext)
    objetivo = cp.Maximize(E_R_ext @ w - (delta / 2) * risco)

    restricao_orcamento = cp.sum(w) == 1
    restricao_long_only = w >= 0
    restricao_concentracao = w <= w_max_vetor

    problema = cp.Problem(objetivo, [restricao_orcamento, restricao_long_only, restricao_concentracao])
    problema.solve()

    if problema.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Otimização não convergiu: status={problema.status}")

    return ResultadoOtimizacao(
        pesos=w.value,
        custo_concentracao_por_ativo=np.asarray(restricao_concentracao.dual_value),
        status=problema.status,
    )


# --------------------------------------------------------------------------
# 2. Estado de calibração persistente (parâmetros adaptativos do pipeline)
# --------------------------------------------------------------------------

@dataclass
class EstadoCalibracao:
    """
    Estado persistente entre execuções do walk-forward. Schema versionado,
    consumido pelos Agentes 4 e 6, atualizado pelo Agente 8.

    epsilon_min: piso de incerteza de Omega (Agente 4) — recalibração
        trimestral (corpus de sentimento é relativamente estável).
    w_max: teto de concentração (Agente 6) — atualmente FIXO em 0.20 por
        decisão de projeto. O mecanismo de recalibração adaptativa
        (mensal, via custo de oportunidade acumulado x hit rate) foi
        desenhado numa etapa anterior mas ainda não confirmado como
        ativo após a decisão de fixar w_max em 20% — ver pergunta ao
        final da resposta que acompanha este arquivo.
    """
    epsilon_min: float = 0.01
    w_max: float = 0.20
    ultima_atualizacao_epsilon: date | None = None
    ultima_atualizacao_wmax: date | None = None


JANELA_EPSILON_DIAS = 90  # trimestral — alinhado ao PIT do Agente 0
JANELA_WMAX_DIAS = 21     # mensal — só relevante se w_max voltar a ser adaptativo


def deve_recalibrar(ultima_data: date | None, data_atual: date, janela_dias: int) -> bool:
    if ultima_data is None:
        return True
    return (data_atual - ultima_data).days >= janela_dias
