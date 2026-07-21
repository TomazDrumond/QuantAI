"""
calibracao_corr_min.py

Grid de calibração para o piso mínimo de correlação (corr_min) usado na
purificação do sinal do Agente 4 (Correlation Filter):

    score_adj_i = score_meta_i * max(|corr(ativo_i, index)|, corr_min)

Mesmo padrão metodológico do calibracao_epsilon_min.py: walk-forward
(expanding window), sem vazamento de dados, candidato global testado em
múltiplos blocos temporais sucessivos para medir estabilidade, não só
performance média.

Motivação (Fase 1): sem piso, ativos de setores estruturalmente
defensivos (baixa correlação histórica com o índice por natureza, não
por ruído) têm seu sinal de sentimento amortecido de forma desigual em
relação a ativos cíclicos -- especialmente severo em quebras de regime
(ex: pandemia, rotação para farmácia/hospitais). O corr_min certo é o
que recupera esse sinal sem "amaciar" o filtro de forma geral para
ativos que já têm correlação alta legítima.

Uso:
    python calibracao_corr_min.py --demo
    python calibracao_corr_min.py --data caminho.csv --out resultados/

CSV de entrada esperado (mesma disciplina anti-look-ahead do Agente 1):
    data, ticker, score_meta, correlacao, retorno_realizado_fwd
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Núcleo: score ajustado por correlação, com piso
# --------------------------------------------------------------------------

def calcular_score_ajustado(score_meta: np.ndarray, correlacao: np.ndarray, corr_min: float) -> np.ndarray:
    correlacao_segura = np.nan_to_num(correlacao, nan=0.0)  # sem histórico -- pior caso, piso se aplica
    fator = np.maximum(np.abs(correlacao_segura), corr_min)
    return score_meta * fator


# --------------------------------------------------------------------------
# 2. Métricas por candidato (long-only, mesmo proxy do Agente 6)
# --------------------------------------------------------------------------

@dataclass
class MetricasFold:
    corr_min: float
    fold: int
    hit_rate: float
    sharpe_proxy: float
    score_medio_absoluto: float  # diagnóstico: quanto sinal sobra, em média, após o filtro
    n_obs: int


def _hit_rate(score_adj: np.ndarray, retorno_realizado: np.ndarray) -> float:
    return float(np.mean(np.sign(score_adj) == np.sign(retorno_realizado)))


def _sharpe_proxy_long_only(score_adj: np.ndarray, retorno_realizado: np.ndarray) -> float:
    """
    Mesma disciplina do Agente 6: LONG-ONLY. Ativos com score ajustado
    negativo recebem peso zero -- nunca posição vendida a descoberto.
    """
    peso = np.maximum(score_adj, 0.0)
    soma = np.sum(peso)
    if soma == 0:
        return float("nan")
    peso = peso / soma
    retorno_estrategia = peso * retorno_realizado
    if np.std(retorno_estrategia) == 0:
        return float("nan")
    return float(np.mean(retorno_estrategia) / np.std(retorno_estrategia) * np.sqrt(252))


# --------------------------------------------------------------------------
# 3. Walk-forward (idêntico em espírito ao calibracao_epsilon_min.py)
# --------------------------------------------------------------------------

def gerar_folds_walk_forward(datas_unicas: np.ndarray, n_folds: int, janela_min_treino: int):
    n = len(datas_unicas)
    tamanho_teste = max((n - janela_min_treino) // n_folds, 1)
    for k in range(n_folds):
        fim_treino = janela_min_treino + k * tamanho_teste
        fim_teste = min(fim_treino + tamanho_teste, n)
        if fim_treino >= n or fim_treino >= fim_teste:
            break
        yield k, datas_unicas[fim_treino:fim_teste]


def rodar_grid_calibracao(
    df: pd.DataFrame,
    candidatos_corr_min: np.ndarray,
    n_folds: int = 5,
    janela_min_treino_dias: int = 60,
) -> pd.DataFrame:
    obrigatorias = {"data", "ticker", "score_meta", "correlacao", "retorno_realizado_fwd"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")

    df = df.sort_values("data").reset_index(drop=True)
    datas_unicas = np.sort(df["data"].unique())
    resultados: list[MetricasFold] = []

    for corr_min in candidatos_corr_min:
        for fold_idx, datas_teste in gerar_folds_walk_forward(datas_unicas, n_folds, janela_min_treino_dias):
            bloco = df[df["data"].isin(datas_teste)]
            if bloco.empty:
                continue

            score_adj = calcular_score_ajustado(
                bloco["score_meta"].to_numpy(), bloco["correlacao"].to_numpy(), corr_min
            )
            retorno_real = bloco["retorno_realizado_fwd"].to_numpy()

            resultados.append(MetricasFold(
                corr_min=corr_min,
                fold=fold_idx,
                hit_rate=_hit_rate(score_adj, retorno_real),
                sharpe_proxy=_sharpe_proxy_long_only(score_adj, retorno_real),
                score_medio_absoluto=float(np.mean(np.abs(score_adj))),
                n_obs=len(bloco),
            ))

    return pd.DataFrame([r.__dict__ for r in resultados])


def resumir_grid(tabela: pd.DataFrame) -> pd.DataFrame:
    resumo = tabela.groupby("corr_min").agg(
        hit_rate_medio=("hit_rate", "mean"),
        sharpe_proxy_medio=("sharpe_proxy", "mean"),
        sharpe_proxy_desvio=("sharpe_proxy", "std"),
        score_medio_absoluto=("score_medio_absoluto", "mean"),
        n_folds_validos=("fold", "nunique"),
    ).reset_index()
    resumo["score_robustez"] = (
        resumo["sharpe_proxy_medio"] / resumo["sharpe_proxy_desvio"].replace(0, np.nan)
    )
    return resumo.sort_values("score_robustez", ascending=False)


# --------------------------------------------------------------------------
# 4. Dados sintéticos: metade cíclicos (alta corr), metade defensivos
#    (baixa corr, mas sinal igualmente informativo) -- para o teste ter
#    o mesmo tipo de tensão que motivou esta calibração.
# --------------------------------------------------------------------------

def gerar_dados_sinteticos(n_dias: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2023-01-02", periods=n_dias, freq="B")

    ciclicos = [f"CICLICO{i}" for i in range(6)]
    defensivos = [f"DEFENSIVO{i}" for i in range(6)]

    linhas = []
    for data in datas:
        for ticker in ciclicos + defensivos:
            if rng.random() < 0.35:
                continue
            sentimento_real = rng.normal(0, 0.4)  # mesmo poder preditivo para os dois grupos
            score_meta = np.clip(sentimento_real + rng.normal(0, 0.1), -1, 1)
            if ticker in ciclicos:
                correlacao = np.clip(rng.normal(0.8, 0.08), 0.3, 0.98)
            else:
                correlacao = np.clip(rng.normal(0.10, 0.05), -0.3, 0.3)
            retorno_fwd = sentimento_real * 0.03 + rng.normal(0, 0.02)
            linhas.append({
                "data": data, "ticker": ticker, "score_meta": score_meta,
                "correlacao": correlacao, "retorno_realizado_fwd": retorno_fwd,
            })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid de calibração de corr_min (Agente 4).")
    parser.add_argument("--data", type=str, default=None,
                         help="CSV com colunas: data,ticker,score_meta,correlacao,retorno_realizado_fwd")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=str, default="resultados_calibracao_corr")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--janela-min-treino", type=int, default=60)
    args = parser.parse_args()

    if args.demo:
        df = gerar_dados_sinteticos()
        print(f"[demo] {len(df)} observações, {df['ticker'].nunique()} ativos (metade cíclicos, metade defensivos).")
    elif args.data:
        df = pd.read_csv(args.data, parse_dates=["data"])
    else:
        raise SystemExit("Forneça --data <arquivo.csv> ou --demo.")

    candidatos = np.linspace(0.0, 0.6, 13)  # 0.0 (sem piso) a 0.6, passo 0.05

    tabela = rodar_grid_calibracao(df, candidatos, n_folds=args.n_folds, janela_min_treino_dias=args.janela_min_treino)
    resumo = resumir_grid(tabela)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(out_dir / "grid_detalhado_por_fold.csv", index=False)
    resumo.to_csv(out_dir / "grid_resumo_por_corr_min.csv", index=False)

    print("\n=== Top 5 candidatos por robustez ===")
    print(resumo.head(5).to_string(index=False))
    print(f"\nResultados completos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
