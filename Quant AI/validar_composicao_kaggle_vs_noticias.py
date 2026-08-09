"""
validar_composicao_kaggle_vs_noticias.py

Agente 0 (parcial) — validação de qualidade da composição histórica do
Ibovespa antes de usá-la como universo point-in-time (PIT) no backtest.

Contexto (decisão desta etapa do projeto): a B3 não expõe um endpoint
único com composição histórica completa do índice. A estratégia adotada
é usar um dataset comunitário (ex: Kaggle) como fonte de VOLUME, cruzado
com uma reconstrução manual a partir de notícias de imprensa financeira
(Mais Retorno, Suno, Investidor10 etc.) como fonte de VERDADE pontual —
cada eventos de troca de carteira é público e datado precisamente.

Este script NÃO reconstrói a composição inteira a partir de notícias
(isso exigiria achar, para cada quadrimestre, um artigo com a LISTA
COMPLETA de ativos, o que nem sempre existe) — em vez disso, usa os
eventos de ENTRADA/SAÍDA noticiados (que são amplamente divulgados a
cada troca) como pontos de verificação (checkpoints) contra o dataset
de volume (Kaggle), que assumimos ter a composição completa mas cuja
proveniência não é auditada.

Uso:
    python validar_composicao_kaggle_vs_noticias.py \
        --eventos eventos_composicao_ibovespa_seed.csv \
        --kaggle caminho_para_dataset_kaggle.csv \
        --coluna-data nome_da_coluna_data \
        --coluna-ticker nome_da_coluna_ticker

O arquivo de eventos (formato já fixado, ver eventos_composicao_ibovespa_seed.csv):
    data_inicio_vigencia, data_fim_vigencia, tickers_entrada, tickers_saida,
    n_ativos, n_empresas, fonte_url

O dataset Kaggle é assumido em formato "longo": uma linha por (data, ticker)
indicando que aquele ticker fazia parte do índice naquela data. Datasets
reais variam de schema — ajuste --coluna-data / --coluna-ticker conforme
necessário, ou adapte `carregar_kaggle()` se o formato for muito diferente
(ex: uma coluna por ticker, formato "largo").
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Carregamento
# --------------------------------------------------------------------------

def carregar_eventos(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["data_inicio_vigencia", "data_fim_vigencia"])
    df["tickers_entrada"] = df["tickers_entrada"].fillna("").apply(
        lambda s: [t.strip() for t in s.split(";") if t.strip()]
    )
    df["tickers_saida"] = df["tickers_saida"].fillna("").apply(
        lambda s: [t.strip() for t in s.split(";") if t.strip()]
    )
    return df


def carregar_kaggle(caminho: str, coluna_data: str, coluna_ticker: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=[coluna_data])
    df = df.rename(columns={coluna_data: "data", coluna_ticker: "ticker"})
    return df[["data", "ticker"]]


# --------------------------------------------------------------------------
# 2. Checagens de consistência (checkpoints, não reconstrução completa)
# --------------------------------------------------------------------------

@dataclass
class Discrepancia:
    evento_data: pd.Timestamp
    ticker: str
    tipo_evento: str        # "entrada" ou "saida"
    esperado: str
    encontrado_no_kaggle: str
    fonte_url: str


def _presente_no_kaggle(kaggle_df: pd.DataFrame, ticker: str, data_alvo: pd.Timestamp,
                         tolerancia_dias: int = 5) -> bool | None:
    """
    Verifica se o ticker aparece no kaggle_df numa janela A PARTIR da data
    alvo (forward-only: [data_alvo, data_alvo + tolerancia_dias]). A janela
    é deliberadamente unilateral -- olhar para trás da data de transição
    sempre encontraria o estado ANTERIOR (ticker ainda presente/ausente
    do lado errado da troca), gerando falso positivo sistemático em toda
    checagem de saída.

    Retorna None se não há nenhuma observação do kaggle_df (de nenhum
    ticker) nessa janela -- dado insuficiente para concluir, não é uma
    discrepância, é uma lacuna de cobertura.
    """
    janela_geral = kaggle_df[
        (kaggle_df["data"] >= data_alvo) &
        (kaggle_df["data"] <= data_alvo + pd.Timedelta(days=tolerancia_dias))
    ]
    if janela_geral.empty:
        return None
    return (janela_geral["ticker"] == ticker).any()


def validar(eventos_df: pd.DataFrame, kaggle_df: pd.DataFrame) -> tuple[list[Discrepancia], list[str]]:
    discrepancias: list[Discrepancia] = []
    lacunas_cobertura: list[str] = []

    for _, evento in eventos_df.iterrows():
        data_evento = evento["data_inicio_vigencia"]

        for ticker in evento["tickers_entrada"]:
            # Esperado: ticker DEVE aparecer no kaggle a partir dessa data
            presente = _presente_no_kaggle(kaggle_df, ticker, data_evento)
            if presente is None:
                lacunas_cobertura.append(f"{ticker} em {data_evento.date()} (evento de entrada) -- sem dado kaggle na janela")
            elif not presente:
                discrepancias.append(Discrepancia(
                    evento_data=data_evento, ticker=ticker, tipo_evento="entrada",
                    esperado="presente no índice a partir desta data",
                    encontrado_no_kaggle="ausente",
                    fonte_url=evento["fonte_url"],
                ))

        for ticker in evento["tickers_saida"]:
            # Esperado: ticker NÃO deve mais aparecer no kaggle a partir dessa data
            presente = _presente_no_kaggle(kaggle_df, ticker, data_evento)
            if presente is None:
                lacunas_cobertura.append(f"{ticker} em {data_evento.date()} (evento de saída) -- sem dado kaggle na janela")
            elif presente:
                discrepancias.append(Discrepancia(
                    evento_data=data_evento, ticker=ticker, tipo_evento="saida",
                    esperado="ausente do índice a partir desta data",
                    encontrado_no_kaggle="presente",
                    fonte_url=evento["fonte_url"],
                ))

    return discrepancias, lacunas_cobertura


# --------------------------------------------------------------------------
# 3. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Valida dataset Kaggle contra eventos de composição reconstruídos via notícias.")
    parser.add_argument("--eventos", required=True, help="CSV de eventos (ver eventos_composicao_ibovespa_seed.csv)")
    parser.add_argument("--kaggle", required=True, help="CSV do dataset comunitário a validar")
    parser.add_argument("--coluna-data", required=True, help="Nome da coluna de data no CSV do kaggle")
    parser.add_argument("--coluna-ticker", required=True, help="Nome da coluna de ticker no CSV do kaggle")
    parser.add_argument("--tolerancia-dias", type=int, default=5)
    args = parser.parse_args()

    eventos_df = carregar_eventos(args.eventos)
    kaggle_df = carregar_kaggle(args.kaggle, args.coluna_data, args.coluna_ticker)

    discrepancias, lacunas = validar(eventos_df, kaggle_df)

    print(f"=== {len(discrepancias)} discrepância(s) encontrada(s) ===")
    for d in discrepancias:
        print(f"  [{d.evento_data.date()}] {d.ticker} ({d.tipo_evento}): "
              f"esperado='{d.esperado}', kaggle='{d.encontrado_no_kaggle}' -- fonte: {d.fonte_url}")

    print(f"\n=== {len(lacunas)} lacuna(s) de cobertura (sem dado suficiente p/ concluir) ===")
    for l in lacunas:
        print(f"  {l}")

    if not discrepancias and not lacunas:
        print("Nenhuma discrepância nem lacuna -- checkpoints todos consistentes com o dataset kaggle.")


if __name__ == "__main__":
    main()
