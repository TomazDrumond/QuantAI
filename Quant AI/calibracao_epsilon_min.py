"""
calibracao_epsilon_min.py

Grid de calibração para o piso mínimo de incerteza (epsilon_min) usado na
derivação de Omega (Agente 4 - Correlation Filter / Agente 6 - BL Optimizer)
do pipeline Quant AI (Black-Litterman + Sentimento Multi-LLM).

Fórmula calibrada (definida na Fase 0):

    omega_i = max( Var(s1_i, s2_i, s3_i), epsilon_min ) / sqrt(n_i)

onde s1,s2,s3 são os scores dos 3 LLMs e n_i é o número de notícias que
geraram o sinal agregado do ativo i naquela data.

Este script NÃO escolhe epsilon_min sozinho — ele produz e avalia uma
grade de candidatos via walk-forward (expanding window), respeitando a
defasagem t-1 estabelecida na Fase 0 para evitar circularidade entre o
Agente 4 (consome epsilon_min) e o Agente 8 (calibra epsilon_min a partir
do erro histórico do próprio Meta-Score).

Uso:
    python calibracao_epsilon_min.py --demo
        Roda com dados sintéticos, só para validar que o script funciona
        antes de existirem dados reais coletados.

    python calibracao_epsilon_min.py --data caminho.csv --out resultados/
        Roda com dados reais. O CSV precisa ter as colunas:
        data, ticker, s1, s2, s3, n_noticias, retorno_realizado_fwd

IMPORTANTE: retorno_realizado_fwd precisa já vir alinhado sem look-ahead
(retorno do período SEGUINTE à data do sinal). Essa checagem de timestamp
é responsabilidade do Agente 1 (Data Agent), não deste script — o grid
assume que o dado de entrada já é íntegro.

DECISÃO DE PROJETO (fechada nesta etapa): a carteira é LONG-ONLY. Ativos
com sinal de sentimento negativo recebem peso zero no proxy de Sharpe
usado para calibração — nunca posição vendida a descoberto, para evitar
o custo de aluguel de ações (BTC) na B3, que ainda não está modelado em
nenhum agente. Essa mesma restrição precisa ser replicada no Agente 6
(BL Optimizer) na hora de resolver a otimização de fato, não só aqui.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Núcleo matemático: derivação de Q e Omega a partir do sentimento
# --------------------------------------------------------------------------

def calcular_omega(scores_llm: np.ndarray, n_noticias: np.ndarray, epsilon_min: float) -> np.ndarray:
    """
    scores_llm: array (N, 3) com os 3 scores de LLM por observação
    n_noticias: array (N,) com o número de notícias que geraram aquele score
    epsilon_min: piso mínimo de variância (candidato sendo testado)
    """
    var_llm = np.maximum(np.var(scores_llm, axis=1, ddof=0), epsilon_min)
    n_seguro = np.maximum(n_noticias, 1)  # nunca divide por zero / n < 1
    return var_llm / np.sqrt(n_seguro)


def calcular_q(meta_score: np.ndarray, escala: float = 0.05) -> np.ndarray:
    """
    Q = view de retorno absoluto derivada do Meta-Score.
    escala: fator que converte score (-1,1) em retorno esperado plausível.
    Mantido fixo aqui — não é o parâmetro sob teste neste grid.
    """
    return meta_score * escala


# --------------------------------------------------------------------------
# 2. Métricas de avaliação de um candidato epsilon_min
# --------------------------------------------------------------------------

@dataclass
class MetricasFold:
    epsilon_min: float
    fold: int
    erro_calibracao: float   # corr(confiança, erro de previsão) — queremos bem negativo
    hit_rate: float          # % de acerto de sinal (view vs. retorno realizado)
    sharpe_proxy: float      # Sharpe de uma estratégia simples ponderada por 1/omega
    n_obs: int


def _erro_calibracao(omega: np.ndarray, erro_previsao: np.ndarray) -> float:
    """
    Confiança (1/omega) deveria se correlacionar NEGATIVAMENTE com o erro
    de previsão |q_i - retorno_realizado_i|: mais confiança, menor erro
    esperado. Um epsilon_min bem calibrado produz correlação bem negativa.
    """
    confianca = 1.0 / np.maximum(omega, 1e-8)
    if np.std(confianca) == 0 or np.std(erro_previsao) == 0:
        return float("nan")
    return float(np.corrcoef(confianca, erro_previsao)[0, 1])


def _hit_rate(q: np.ndarray, retorno_realizado: np.ndarray) -> float:
    return float(np.mean(np.sign(q) == np.sign(retorno_realizado)))


def _sharpe_proxy(q: np.ndarray, omega: np.ndarray, retorno_realizado: np.ndarray) -> float:
    """
    Estratégia simplificada LONG-ONLY: aloca proporcionalmente à confiança
    (1/omega) apenas nos ativos com sinal positivo (q > 0). Ativos com
    sentimento negativo recebem peso zero (simplesmente não são comprados,
    nunca vendidos a descoberto) -- decisão fechada no projeto para evitar
    o custo de aluguel de ações (BTC/securities lending) na B3, ainda não
    modelado em nenhum agente.

    NÃO é o BL completo (isso roda no Agente 6, que também vai precisar
    impor a restrição long-only na otimização, não só neste proxy) -- é
    um proxy rápido para comparar candidatos de epsilon_min sem rodar a
    otimização completa em cada ponto da grade.
    """
    peso = np.maximum(q, 0.0) / np.maximum(omega, 1e-8)  # q <= 0 -> peso 0 (nunca vende a descoberto)
    soma_abs = np.sum(peso)
    if soma_abs == 0:
        return float("nan")  # nenhum sinal positivo no bloco -- estratégia fica 100% fora do mercado
    peso = peso / soma_abs
    retorno_estrategia = peso * retorno_realizado
    if np.std(retorno_estrategia) == 0:
        return float("nan")
    return float(np.mean(retorno_estrategia) / np.std(retorno_estrategia) * np.sqrt(252))


# --------------------------------------------------------------------------
# 3. Walk-forward (expanding window) — sem vazamento de dados
# --------------------------------------------------------------------------

def gerar_folds_walk_forward(datas_unicas: np.ndarray, n_folds: int, janela_min_treino: int):
    """
    Expanding window: cada fold testa um bloco de datas posterior a toda
    a janela mínima de "aquecimento". epsilon_min é um candidato global
    (mesmo valor testado em todos os folds) — a walk-forward aqui serve
    para medir ESTABILIDADE do candidato ao longo do tempo, não para
    ajustar o parâmetro por fold. Nenhum fold usa dados de folds
    posteriores para decidir nada.
    """
    n = len(datas_unicas)
    tamanho_teste = max((n - janela_min_treino) // n_folds, 1)
    for k in range(n_folds):
        fim_treino = janela_min_treino + k * tamanho_teste
        fim_teste = min(fim_treino + tamanho_teste, n)
        if fim_treino >= n or fim_treino >= fim_teste:
            break
        yield k, datas_unicas[fim_treino:fim_teste]


# --------------------------------------------------------------------------
# 4. Grid de calibração
# --------------------------------------------------------------------------

def rodar_grid_calibracao(
    df: pd.DataFrame,
    candidatos_epsilon: np.ndarray,
    n_folds: int = 5,
    janela_min_treino_dias: int = 60,
    escala_q: float = 0.05,
) -> pd.DataFrame:
    obrigatorias = {"data", "ticker", "s1", "s2", "s3", "n_noticias", "retorno_realizado_fwd"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes no dataframe: {faltando}")

    df = df.sort_values("data").reset_index(drop=True)
    datas_unicas = np.sort(df["data"].unique())

    resultados: list[MetricasFold] = []

    for epsilon in candidatos_epsilon:
        for fold_idx, datas_teste in gerar_folds_walk_forward(datas_unicas, n_folds, janela_min_treino_dias):
            bloco = df[df["data"].isin(datas_teste)]
            if bloco.empty:
                continue

            scores_llm = bloco[["s1", "s2", "s3"]].to_numpy()
            n_noticias = bloco["n_noticias"].to_numpy()
            retorno_real = bloco["retorno_realizado_fwd"].to_numpy()

            meta_score = scores_llm.mean(axis=1)
            omega = calcular_omega(scores_llm, n_noticias, epsilon)
            q = calcular_q(meta_score, escala=escala_q)
            erro_previsao = np.abs(q - retorno_real)

            resultados.append(MetricasFold(
                epsilon_min=epsilon,
                fold=fold_idx,
                erro_calibracao=_erro_calibracao(omega, erro_previsao),
                hit_rate=_hit_rate(q, retorno_real),
                sharpe_proxy=_sharpe_proxy(q, omega, retorno_real),
                n_obs=len(bloco),
            ))

    return pd.DataFrame([r.__dict__ for r in resultados])


def resumir_grid(tabela: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por epsilon_min: média e desvio-padrão entre folds. Um bom
    candidato tem Sharpe médio alto E desvio-padrão baixo entre folds —
    Sharpe médio alto com alta variância entre folds sugere overfitting
    a um período específico, não um epsilon_min robusto.
    """
    resumo = tabela.groupby("epsilon_min").agg(
        erro_calibracao_medio=("erro_calibracao", "mean"),
        hit_rate_medio=("hit_rate", "mean"),
        sharpe_proxy_medio=("sharpe_proxy", "mean"),
        sharpe_proxy_desvio=("sharpe_proxy", "std"),
        n_folds_validos=("fold", "nunique"),
    ).reset_index()

    resumo["score_robustez"] = (
        resumo["sharpe_proxy_medio"] / resumo["sharpe_proxy_desvio"].replace(0, np.nan)
    )
    return resumo.sort_values("score_robustez", ascending=False)


# --------------------------------------------------------------------------
# 5. Dados sintéticos para teste imediato (--demo), antes dos dados reais
# --------------------------------------------------------------------------

def gerar_dados_sinteticos(n_ativos: int = 12, n_dias: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2023-01-02", periods=n_dias, freq="B")
    tickers = [f"ATIVO{i}" for i in range(n_ativos)]

    linhas = []
    for data in datas:
        for ticker in tickers:
            if rng.random() < 0.35:  # nem todo ativo tem notícia todo dia
                continue
            sentimento_real = rng.normal(0, 0.4)
            ruido_llm = rng.normal(0, 0.15, size=3)
            s = np.clip(sentimento_real + ruido_llm, -1, 1)
            n_noticias = int(rng.poisson(2) + 1)
            retorno_fwd = sentimento_real * 0.03 + rng.normal(0, 0.02)
            linhas.append({
                "data": data, "ticker": ticker,
                "s1": s[0], "s2": s[1], "s3": s[2],
                "n_noticias": n_noticias,
                "retorno_realizado_fwd": retorno_fwd,
            })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid de calibração de epsilon_min (Agente 4/8).")
    parser.add_argument("--data", type=str, default=None,
                         help="CSV com colunas: data,ticker,s1,s2,s3,n_noticias,retorno_realizado_fwd")
    parser.add_argument("--demo", action="store_true", help="Roda com dados sintéticos para validar o script.")
    parser.add_argument("--out", type=str, default="resultados_calibracao", help="Pasta de saída.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--janela-min-treino", type=int, default=60,
                         help="Nº mínimo de datas antes do primeiro fold de teste.")
    args = parser.parse_args()

    if args.demo:
        df = gerar_dados_sinteticos()
        print(f"[demo] Dados sintéticos gerados: {len(df)} observações, {df['ticker'].nunique()} ativos.")
    elif args.data:
        df = pd.read_csv(args.data, parse_dates=["data"])
    else:
        raise SystemExit("Forneça --data <arquivo.csv> ou --demo para teste com dados sintéticos.")

    candidatos = np.concatenate([
        np.array([0.0]),             # sem piso — referência do problema numérico original (Omega -> 0)
        np.logspace(-4, -0.3, 12),   # ~0.0001 a ~0.5, escala log
    ])

    tabela = rodar_grid_calibracao(df, candidatos, n_folds=args.n_folds, janela_min_treino_dias=args.janela_min_treino)
    resumo = resumir_grid(tabela)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(out_dir / "grid_detalhado_por_fold.csv", index=False)
    resumo.to_csv(out_dir / "grid_resumo_por_epsilon.csv", index=False)

    print("\n=== Top 5 candidatos por robustez (Sharpe médio / desvio entre folds) ===")
    print(resumo.head(5).to_string(index=False))
    print(f"\nResultados completos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
