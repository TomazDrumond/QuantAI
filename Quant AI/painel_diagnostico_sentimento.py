"""
painel_diagnostico_sentimento.py

Painel de diagnóstico do Agente 2/3 -- amostragem ALEATÓRIA de notícias
processadas, em vez de um ranking "top N mais fortes".

Motivo (bug real encontrado nesta sessão): um painel de "top 10 sinais
mais fortes" ordenando um campo constante (score_meta = 0.0 para tudo)
ainda mostra 10 linhas com números -- parece estar funcionando mesmo
quando não há sinal nenhum. Amostragem aleatória, combinada com
estatísticas de dispersão, expõe isso de imediato.

Uso:
    python painel_diagnostico_sentimento.py --csv sentimento_agregado_diario.csv --n 15
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def amostra_aleatoria_diagnostico(
    df_bruto: pd.DataFrame,   # saída do Agente 2, uma linha por notícia: ticker,data,noticia_id,texto_noticia,s1,s2,s3
    n: int = 15,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Amostra aleatória (não ordenada por magnitude) de notícias já
    pontuadas, com o texto e os 3 scores lado a lado -- para inspeção
    visual rápida de "isso faz sentido?".
    """
    colunas = [c for c in ["data", "ticker", "texto_noticia", "s1", "s2", "s3"] if c in df_bruto.columns]
    if df_bruto.empty:
        return pd.DataFrame(columns=colunas)
    n_efetivo = min(n, len(df_bruto))
    return df_bruto.sample(n=n_efetivo, random_state=seed)[colunas].reset_index(drop=True)


def estatisticas_de_sanidade(df_bruto: pd.DataFrame) -> dict:
    """
    Estatísticas que matam a dúvida "o Agente 2 está produzindo sinal de
    verdade?" sem precisar olhar linha por linha.
    """
    resultado = {}
    for col in ["s1", "s2", "s3"]:
        if col not in df_bruto.columns or df_bruto.empty:
            resultado[col] = None
            continue
        serie = df_bruto[col]
        resultado[col] = {
            "n": len(serie),
            "std": float(serie.std()),
            "pct_exatamente_zero": float((serie == 0.0).mean() * 100),
            "min": float(serie.min()),
            "max": float(serie.max()),
            "valores_distintos": int(serie.nunique()),
        }
    return resultado


def imprimir_painel(df_bruto: pd.DataFrame, n_amostra: int = 15, seed: int | None = None) -> bool:
    """
    Imprime o painel completo. Retorna False se detectar sinal de
    falha (std ~0 em algum provider), True se parecer saudável -- útil
    para travar um pipeline automaticamente antes de rodar 8 anos de
    backtest sobre dado quebrado.
    """
    print("\n" + "=" * 78)
    print("  PAINEL DE DIAGNÓSTICO -- AMOSTRA ALEATÓRIA (não ordenada por força)")
    print("=" * 78)

    if df_bruto.empty:
        print("  [VAZIO] Nenhuma notícia processada ainda.")
        return False

    amostra = amostra_aleatoria_diagnostico(df_bruto, n=n_amostra, seed=seed)
    with pd.option_context("display.max_colwidth", 60, "display.width", 140):
        print(amostra.to_string(index=False))

    print("\n" + "-" * 78)
    print("  ESTATÍSTICAS DE SANIDADE POR PROVIDER")
    print("-" * 78)

    stats = estatisticas_de_sanidade(df_bruto)
    saudavel = True
    for provider, s in stats.items():
        if s is None:
            continue
        print(f"  {provider}: n={s['n']:6d}  std={s['std']:.5f}  "
              f"zeros={s['pct_exatamente_zero']:5.1f}%  "
              f"min={s['min']:+.3f}  max={s['max']:+.3f}  "
              f"distintos={s['valores_distintos']}")
        if s["std"] < 1e-6:
            print(f"    [ALERTA CRÍTICO] {provider} tem desvio-padrão ~0 -- "
                  f"TODOS OS SCORES SÃO CONSTANTES. Esse provider não está "
                  f"produzindo sinal algum. Verifique diagnostico_agente_2.py.")
            saudavel = False
        elif s["pct_exatamente_zero"] > 80:
            print(f"    [AVISO] {provider} tem {s['pct_exatamente_zero']:.0f}% dos scores "
                  f"exatamente em 0.0 -- suspeito, mesmo sem ser 100%.")

    print("=" * 78)
    if saudavel:
        print("  RESULTADO: nenhum provider com variância nula detectado.")
    else:
        print("  RESULTADO: FALHA DETECTADA -- não prossiga para o backtest completo "
              "sem investigar o(s) provider(s) sinalizado(s) acima.")
    print("=" * 78 + "\n")

    return saudavel


def main():
    parser = argparse.ArgumentParser(description="Painel de diagnóstico de sentimento com amostragem aleatória.")
    parser.add_argument("--csv", default="sentimento_bruto_por_noticia.csv",
                         help="CSV com uma linha por notícia (colunas: ticker,data,noticia_id,texto_noticia,s1,s2,s3)")
    parser.add_argument("--n", type=int, default=15, help="Tamanho da amostra aleatória exibida.")
    parser.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade da amostra (opcional).")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{args.csv}' não encontrado. Rode processar_noticias_reais.py primeiro "
              f"(ele precisa salvar o CSV bruto por notícia, não só o agregado diário).")
        return

    saudavel = imprimir_painel(df, n_amostra=args.n, seed=args.seed)
    if not saudavel:
        raise SystemExit(
            "Painel detectou provider(s) sem variância -- interrompendo para evitar "
            "rodar o backtest completo sobre sentimento quebrado."
        )


if __name__ == "__main__":
    main()
