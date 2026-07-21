"""
agente_8_validator_agent.py

Agente 8 do pipeline Quant AI — Validator Agent.

Responsabilidade (Fase 0): rodar o backtest walk-forward out-of-sample e
reportar métricas -- Sharpe, Sortino, Max Drawdown, Turnover -- além de
três pendências acumuladas ao longo da Fase 1 que ficaram registradas
para serem fechadas aqui:

  1. Sharpe DEFLACIONADO pelo número de parâmetros calibrados via grid
     search sobre a mesma amostra (epsilon_min, corr_min) -- discussão
     sobre overfitting de múltiplos parâmetros, quando o usuário
     recusou adicionar um terceiro parâmetro (assimetria pessimista)
     justamente por esse motivo.
  2. Custo de concentração acumulado (multiplicadores de Lagrange do
     Agente 6, teto de 20%) reportado como métrica formal ao longo do
     backtest.
  3. Agendador de recalibração walk-forward para epsilon_min (Agente 4)
     e corr_min (Agente 4), com cadências INDEPENDENTES -- discutido
     como analogia a "duas velocidades de decisão de um gestor": o
     corpus de sentimento (epsilon_min) é mais estável, recalibra numa
     janela mais longa; a estrutura de correlação (corr_min) pode reagir
     a quebras de regime, recalibra numa janela mais curta. NOTA: a
     cadência de corr_min não foi explicitamente fixada pelo usuário até
     este ponto -- usei o mesmo raciocínio de w_max original (mensal,
     mais reativo que epsilon_min) como valor default, mas isso é uma
     ASSUNÇÃO que deveria ser confirmada, não uma decisão já fechada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Métricas padrão de backtest
# --------------------------------------------------------------------------

def calcular_sharpe(retornos: np.ndarray, retornos_risco_livre: np.ndarray | float = 0.0, periodos_ano: int = 252) -> float:
    excesso = retornos - retornos_risco_livre
    if np.std(excesso) == 0:
        return float("nan")
    return float(np.mean(excesso) / np.std(excesso) * np.sqrt(periodos_ano))


def calcular_sortino(retornos: np.ndarray, retornos_risco_livre: np.ndarray | float = 0.0, periodos_ano: int = 252) -> float:
    excesso = retornos - retornos_risco_livre
    downside = excesso[excesso < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float("nan")
    desvio_downside = np.sqrt(np.mean(downside ** 2))  # semi-desvio, só a cauda negativa
    return float(np.mean(excesso) / desvio_downside * np.sqrt(periodos_ano))


def calcular_max_drawdown(retornos: np.ndarray) -> float:
    riqueza = np.cumprod(1 + retornos)
    pico = np.maximum.accumulate(riqueza)
    drawdown = (riqueza - pico) / pico
    return float(drawdown.min())  # valor negativo, ex: -0.23 = -23%


# --------------------------------------------------------------------------
# 2. Sharpe deflacionado (Bailey & López de Prado, 2014 -- aproximação)
# --------------------------------------------------------------------------

_GAMMA_EULER_MASCHERONI = 0.5772156649

def _cdf_normal_padrao(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _inversa_normal_padrao(p: float) -> float:
    # Aproximação de Beasley-Springer-Moro -- suficiente para este uso
    # (não crítico de precisão de cauda extrema).
    if p <= 0 or p >= 1:
        raise ValueError("p precisa estar em (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def calcular_sharpe_esperado_sob_nulo(variancia_sharpe_trials: float, n_trials: int) -> float:
    """
    E[max SR] esperado sob a hipótese nula de que todos os N trials
    (candidatos testados no grid search) são igualmente aleatórios --
    aproximação de Bailey & López de Prado (2014). Quanto maior N
    (mais candidatos testados via grid), maior o Sharpe máximo esperado
    só por acaso -- é isso que o Sharpe reportado precisa superar.
    """
    if n_trials <= 1:
        return 0.0
    termo1 = (1 - _GAMMA_EULER_MASCHERONI) * _inversa_normal_padrao(1 - 1 / n_trials)
    termo2 = _GAMMA_EULER_MASCHERONI * _inversa_normal_padrao(1 - 1 / (n_trials * math.e))
    return math.sqrt(variancia_sharpe_trials) * (termo1 + termo2)


def calcular_deflated_sharpe_ratio(
    sharpe_observado: float,
    n_observacoes: int,
    n_trials_calibrados: int,
    variancia_sharpe_trials: float = 1.0,
    skew_retornos: float = 0.0,
    kurtosis_retornos: float = 3.0,
) -> float:
    """
    DSR = P(SR_verdadeiro > SR_0 | SR_observado), corrigido por:
      - número de parâmetros/trials calibrados via grid na mesma amostra
        (epsilon_min, corr_min -- n_trials_calibrados = soma dos tamanhos
        de grade testados, não só "2 parâmetros").
      - assimetria e curtose dos retornos (Sharpe assume normalidade;
        retornos financeiros reais tipicamente não são normais).

    Retorna um valor em [0,1] -- probabilidade de que o Sharpe observado
    reflita habilidade real, não sorte de múltiplas tentativas. DSR < 0.95
    é um sinal de alerta razoável (referência comum na literatura, não
    uma lei).
    """
    sr0 = calcular_sharpe_esperado_sob_nulo(variancia_sharpe_trials, n_trials_calibrados)

    numerador = (sharpe_observado - sr0) * math.sqrt(n_observacoes - 1)
    denominador = math.sqrt(
        1 - skew_retornos * sharpe_observado + ((kurtosis_retornos - 1) / 4) * sharpe_observado ** 2
    )
    if denominador <= 0:
        return float("nan")

    return _cdf_normal_padrao(numerador / denominador)


# --------------------------------------------------------------------------
# 3. Custo de concentração acumulado (Agente 6, multiplicadores de Lagrange)
# --------------------------------------------------------------------------

@dataclass
class RegistroConcentracao:
    data: pd.Timestamp
    custo_oportunidade_total: float   # soma(dual_value * desvio) daquele rebalanceamento
    hit_rate_periodo: float           # acerto de sinal dos ativos que bateram no teto


def acumular_custo_concentracao(registros: list[RegistroConcentracao]) -> dict:
    if not registros:
        return {"custo_acumulado": 0.0, "hit_rate_medio_em_periodos_restritos": float("nan")}
    return {
        "custo_acumulado": sum(r.custo_oportunidade_total for r in registros),
        "hit_rate_medio_em_periodos_restritos": float(np.mean([r.hit_rate_periodo for r in registros])),
    }


# --------------------------------------------------------------------------
# 4. Agendador de recalibração walk-forward (epsilon_min, corr_min)
# --------------------------------------------------------------------------

@dataclass
class EstadoCalibracaoAgente8:
    epsilon_min: float = 0.01
    corr_min: float = 0.25
    ultima_atualizacao_epsilon: date | None = None
    ultima_atualizacao_corr: date | None = None


JANELA_EPSILON_DIAS = 90  # trimestral -- corpus de sentimento, mais estável (decisão fechada)
JANELA_CORR_MIN_DIAS = 30  # ASSUNÇÃO NÃO CONFIRMADA -- ver docstring do módulo


def deve_recalibrar(ultima_data: date | None, data_atual: date, janela_dias: int) -> bool:
    if ultima_data is None:
        return True
    return (data_atual - ultima_data).days >= janela_dias


def passo_walk_forward(
    estado: EstadoCalibracaoAgente8,
    data_atual: date,
    funcao_recalibrar_epsilon,  # callable(historico_ate_t_menos_1) -> float
    funcao_recalibrar_corr,     # callable(historico_ate_t_menos_1) -> float
    historico_ate_t_menos_1,
) -> EstadoCalibracaoAgente8:
    if deve_recalibrar(estado.ultima_atualizacao_epsilon, data_atual, JANELA_EPSILON_DIAS):
        estado.epsilon_min = funcao_recalibrar_epsilon(historico_ate_t_menos_1)
        estado.ultima_atualizacao_epsilon = data_atual

    if deve_recalibrar(estado.ultima_atualizacao_corr, data_atual, JANELA_CORR_MIN_DIAS):
        estado.corr_min = funcao_recalibrar_corr(historico_ate_t_menos_1)
        estado.ultima_atualizacao_corr = data_atual

    return estado


# --------------------------------------------------------------------------
# 5. Relatório consolidado do backtest
# --------------------------------------------------------------------------

@dataclass
class RelatorioBacktest:
    sharpe: float
    sortino: float
    max_drawdown: float
    turnover_medio: float
    sharpe_deflacionado: float
    n_trials_calibrados: int
    custo_concentracao_acumulado: float


def gerar_relatorio(
    retornos: np.ndarray,
    turnovers: np.ndarray,
    n_trials_epsilon: int,
    n_trials_corr: int,
    registros_concentracao: list[RegistroConcentracao],
    skew_retornos: float = 0.0,
    kurtosis_retornos: float = 3.0,
) -> RelatorioBacktest:
    sharpe = calcular_sharpe(retornos)
    n_trials_total = n_trials_epsilon + n_trials_corr

    sharpe_defl = calcular_deflated_sharpe_ratio(
        sharpe_observado=sharpe, n_observacoes=len(retornos),
        n_trials_calibrados=n_trials_total,
        skew_retornos=skew_retornos, kurtosis_retornos=kurtosis_retornos,
    )
    concentracao = acumular_custo_concentracao(registros_concentracao)

    return RelatorioBacktest(
        sharpe=sharpe,
        sortino=calcular_sortino(retornos),
        max_drawdown=calcular_max_drawdown(retornos),
        turnover_medio=float(np.mean(turnovers)),
        sharpe_deflacionado=sharpe_defl,
        n_trials_calibrados=n_trials_total,
        custo_concentracao_acumulado=concentracao["custo_acumulado"],
    )
