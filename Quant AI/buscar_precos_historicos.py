"""
buscar_precos_historicos.py

Item 1 do plano de trabalho: buscar preços reais (yfinance) para todo o
universo de tickers que já apareceu em composicao_pit_ibovespa.csv, e
aplicar a lógica de integridade do Agente 1 (detecção de gap, exclusão
de gap prolongado > 5 dias úteis, preenchimento forward-only via
Wiener para gaps curtos).

RODAR NO COLAB (precisa de internet) -- não pôde ser testado no
ambiente de desenvolvimento original.

Uso:
    python buscar_precos_historicos.py \
        --composicao composicao_pit_ibovespa.csv \
        --inicio 2018-01-01 --fim 2026-01-01 \
        --out precos_historicos.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agente_1_data_agent import (
    buscar_precos_b3,
    obter_calendario_pregao,
    detectar_gaps,
    preencher_gap_wiener,
    identificar_ativos_com_gap_prolongado,
)


def extrair_universo_completo(composicao_pit: pd.DataFrame) -> list[str]:
    """
    União de TODOS os tickers que já apareceram em qualquer quadrimestre
    -- a elegibilidade point-in-time (qual ativo vale em qual data) é
    aplicada DEPOIS, na hora de montar cada rebalanceamento, não aqui.
    Aqui só precisamos garantir que temos preço disponível para todo
    ativo que algum dia esteve no índice.
    """
    return sorted(composicao_pit["ticker"].unique())


def buscar_em_lotes(tickers: list[str], inicio: str, fim: str, tamanho_lote: int = 20) -> pd.DataFrame:
    """
    yfinance pode falhar ou ficar instável com lotes muito grandes de
    tickers numa única chamada -- busca em lotes menores e concatena.
    Tickers que falharem (ex: delistados, renomeados, sem correspondente
    exato no Yahoo Finance) são reportados, não travam o lote inteiro.
    """
    blocos = []
    falhas = []

    for i in range(0, len(tickers), tamanho_lote):
        lote = tickers[i:i + tamanho_lote]
        try:
            bloco = buscar_precos_b3(lote, inicio, fim)
            blocos.append(bloco)
        except Exception as e:
            falhas.extend(lote)
            print(f"[AVISO] Lote {lote} falhou: {e}")

    if falhas:
        print(f"\n[AVISO] {len(falhas)} ticker(s) não retornaram preço: {falhas}")
        print("Motivos comuns: delistagem, mudança de código/razão social, "
              "ticker específico não mapeado 1:1 no Yahoo Finance (ex: units, BDRs).")

    if not blocos:
        raise RuntimeError("Nenhum lote retornou dado -- verifique conexão/tickers.")

    return pd.concat(blocos, ignore_index=True)


def aplicar_integridade_por_ticker(
    precos_long: pd.DataFrame,
    dias_pregao: pd.DatetimeIndex,
    limite_gap_dias: int = 5,
) -> pd.DataFrame:
    """
    Para cada ticker: detecta gaps, marca datas de exclusão temporária
    (gap > limite_gap_dias), preenche os gaps curtos via Wiener
    forward-only. Retorna painel completo (todo ticker x todo dia de
    pregão no período), com uma coluna booleana `ativo_valido` marcando
    False nos dias em que o ativo deveria ficar fora do universo elegível.
    """
    resultado = []

    for ticker in precos_long["ticker"].unique():
        serie = precos_long[precos_long["ticker"] == ticker].set_index("data")["close"]
        serie = serie.reindex(dias_pregao)

        datas_excluidas = identificar_ativos_com_gap_prolongado(serie, dias_pregao, limite_gap_dias)
        serie_preenchida = preencher_gap_wiener(serie, dias_pregao, seed=hash(ticker) % (2**32))

        painel_ticker = pd.DataFrame({
            "data": dias_pregao,
            "ticker": ticker,
            "close": serie_preenchida.values,
            "ativo_valido": ~dias_pregao.isin(datas_excluidas),
        })
        resultado.append(painel_ticker)

    return pd.concat(resultado, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Busca preços reais e aplica integridade do Agente 1.")
    parser.add_argument("--composicao", required=True)
    parser.add_argument("--inicio", required=True)
    parser.add_argument("--fim", required=True)
    parser.add_argument("--out", default="precos_historicos.csv")
    parser.add_argument("--limite-gap-dias", type=int, default=5)
    args = parser.parse_args()

    composicao = pd.read_csv(args.composicao, parse_dates=["data_vigencia"])
    universo = extrair_universo_completo(composicao)
    print(f"Universo completo: {len(universo)} tickers únicos ao longo de todo o histórico.")

    dias_pregao = obter_calendario_pregao(args.inicio, args.fim)
    print(f"Calendário de pregão B3: {len(dias_pregao)} dias úteis entre {args.inicio} e {args.fim}.")

    precos_brutos = buscar_em_lotes(universo, args.inicio, args.fim)
    print(f"Preços brutos obtidos: {len(precos_brutos)} observações.")

    painel_final = aplicar_integridade_por_ticker(precos_brutos, dias_pregao, args.limite_gap_dias)

    pct_invalido = (~painel_final["ativo_valido"]).mean() * 100
    print(f"{pct_invalido:.2f}% das observações marcadas como inválidas (gap > {args.limite_gap_dias} dias úteis).")

    out_path = Path(args.out)
    painel_final.to_csv(out_path, index=False)
    print(f"Salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
