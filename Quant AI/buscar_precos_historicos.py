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
import time
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


DEPARA_TICKERS_B3 = {
    'VVAR3': 'BHIA3',
    'VIIA3': 'BHIA3',
    'VIVT4': 'VIVT3',
    'KROT3': 'COGN3',
    'ESTC3': 'YDUQ3',
    'BRDT3': 'VBBR3',
    'BTOW3': 'AMER3',
    'LAME4': 'AMER3',
    'ENBR3': 'ENGI11',
    'BRML3': 'ALOS3',
    'ALSO3': 'ALOS3',
    'SOMA3': 'AZZA3',
    'ARZZ3': 'AZZA3',
    'GNDI3': 'HAPV3',
    'LCAM3': 'RENT3',
    'SMLS3': 'GOLL4',
    'SULA11': 'RDOR3',
    'TIMP3': 'TIMS3',
    'IGTA3': 'IGTI11',
    'BVMF3': 'B3SA3',
    'AXIA6': 'B3SA3',
    'RRRP3': 'BRAV3',
    'PCAR4': 'PCAR3',
    'CYRE4': 'CYRE3',
    'ELET6': 'ELET3',
    'CPLE6': 'CPLE3',
    'EMBR3': 'EMBJ3',
    'MRFG3': 'MBRF3',
    'PETZ3': 'AUAU3',
    'HGTX3': 'AZZA3',
    'BIDI4': 'INBR31',
    'BIDI11': 'INBR31',
}


def buscar_em_lotes(
    tickers: list[str], inicio: str, fim: str, tamanho_lote: int = 20,
    pausa_entre_lotes_seg: float = 2.0, pausa_entre_individuais_seg: float = 1.0,
) -> tuple[pd.DataFrame, list[str]]:
    blocos = []
    falhas_em_lote = []

    for i in range(0, len(tickers), tamanho_lote):
        lote = tickers[i:i + tamanho_lote]
        try:
            bloco = buscar_precos_b3(lote, inicio, fim)
        except Exception as e:
            falhas_em_lote.extend(lote)
            print(f"[AVISO] Lote inteiro falhou (exceção): {lote}: {e}")
            time.sleep(pausa_entre_lotes_seg)
            continue

        obtidos = set(bloco["ticker"].unique()) if not bloco.empty else set()
        faltando_no_lote = [t for t in lote if t not in obtidos]
        falhas_em_lote.extend(faltando_no_lote)
        if not bloco.empty:
            blocos.append(bloco)
        time.sleep(pausa_entre_lotes_seg)

    falhas_finais = []
    if falhas_em_lote:
        print(f"\n[RETENTATIVA] {len(falhas_em_lote)} ticker(s) falharam em lote -- "
              f"tentando individualmente e via tabela De-Para de fusões/sucessores B3...")
        recuperados = []
        for ticker in falhas_em_lote:
            sucessor = DEPARA_TICKERS_B3.get(ticker)
            sucesso = False
            
            # 1. Tenta ticker original
            try:
                bloco_ind = buscar_precos_b3([ticker], inicio, fim)
                if not bloco_ind.empty:
                    blocos.append(bloco_ind)
                    recuperados.append(ticker)
                    sucesso = True
            except Exception:
                pass
            
            # 2. Se falhou e existe sucessor, tenta ticker sucessor e rotula com o ticker original
            if not sucesso and sucessor:
                try:
                    bloco_suc = buscar_precos_b3([sucessor], inicio, fim)
                    if not bloco_suc.empty:
                        bloco_suc["ticker"] = ticker  # Mapeia para o ticker original PIT
                        blocos.append(bloco_suc)
                        recuperados.append(f"{ticker}->{sucessor}")
                        sucesso = True
                except Exception:
                    pass
            
            if not sucesso:
                falhas_finais.append(ticker)
                
            time.sleep(pausa_entre_individuais_seg)

        if recuperados:
            print(f"[RECUPERADOS via retentativa/De-Para]: {recuperados}")
        if falhas_finais:
            print(f"[FALHA CONFIRMADA APÓS DE-PARA]: {falhas_finais}")

    if not blocos:
        raise RuntimeError("Nenhum lote/ticker retornou dado -- verifique conexão.")

    return pd.concat(blocos, ignore_index=True), falhas_finais



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
    import yfinance
    if tuple(int(x) for x in yfinance.__version__.split(".")[:3]) < (0, 2, 55):
        print(
            f"[ALERTA] yfinance {yfinance.__version__} detectado -- versões < 0.2.55 têm bug "
            f"conhecido que causa falso 'possibly delisted; no timezone found' até para ativos "
            f"blue-chip. Rode `pip install --upgrade yfinance` antes de continuar."
        )

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

    precos_brutos, tickers_com_falha_confirmada = buscar_em_lotes(universo, args.inicio, args.fim)
    print(f"\nPreços brutos obtidos: {len(precos_brutos)} observações "
          f"({precos_brutos['ticker'].nunique()} de {len(universo)} tickers).")

    if tickers_com_falha_confirmada:
        falhas_path = Path(args.out).with_name("tickers_sem_preco.txt")
        falhas_path.write_text("\n".join(sorted(tickers_com_falha_confirmada)))
        print(f"\n{len(tickers_com_falha_confirmada)} ticker(s) sem preço mesmo após retentativa individual "
              f"-- lista salva em {falhas_path.resolve()} para revisão manual "
              f"(provável mudança de código/fusão/delistagem real -- ex: VIVT4->VIVT3, "
              f"BTOW3/LAME4 fundidos em AMER3 em 2021).")

    painel_final = aplicar_integridade_por_ticker(precos_brutos, dias_pregao, args.limite_gap_dias)

    pct_invalido = (~painel_final["ativo_valido"]).mean() * 100
    print(f"{pct_invalido:.2f}% das observações marcadas como inválidas (gap > {args.limite_gap_dias} dias úteis).")

    out_path = Path(args.out)
    painel_final.to_csv(out_path, index=False)
    print(f"Salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
