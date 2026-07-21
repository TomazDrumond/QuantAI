"""
agente_0_universe_data.py

Agente 0 do pipeline Quant AI — Universe & Reference Data (parte 1:
composição point-in-time do Ibovespa).

Esta versão substitui a estratégia anterior (reconstrução via notícias +
validação cruzada contra dataset Kaggle não auditado) por uma fonte
PRIMÁRIA E OFICIAL: os arquivos "ViradaFinal" da própria B3, que contêm
a composição teórica completa do Ibovespa (ticker + peso %) a cada
virada quadrimestral, com a data de vigência exata no rodapé de cada
arquivo ("Pregão Base").

Isso resolve de forma definitiva o problema identificado na Fase 1
(survivorship bias / universo point-in-time) -- não é mais necessário
validar contra um dataset comunitário de proveniência incerta, porque a
fonte agora É a fonte primária.

Estrutura de pastas esperada (a que você enviou):
    Carteira IBOV/
        2018/ViradaFinal1Q18.xlsx, ViradaFinal2Q18.XLSX, ...
        2019/...
        ...

Duas variações de formato de arquivo foram detectadas e são tratadas
automaticamente pelo parser:
  - Anos mais antigos: cabeçalho "COD./ACAO/TIPO/QTDE. TEORICA/PART. %",
    rodapé "PREGÃO BASE: DD/MM/AAAA" (data embutida na mesma célula, texto).
  - Anos mais recentes: cabeçalho "CÓDIGO/AÇÃO/TIPO/QTDE. TEÓRICA/PART. %",
    rodapé "Pregão Base:" com a data em datetime na célula ao lado.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def _normalizar(texto) -> str:
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return texto.strip().upper()


# --------------------------------------------------------------------------
# 1. Parser de um único arquivo ViradaFinal
# --------------------------------------------------------------------------

def parsear_arquivo_virada(caminho: Path, sheet: str = "IBOV") -> tuple[pd.Timestamp, pd.DataFrame]:
    df = pd.read_excel(caminho, sheet_name=sheet, header=None)

    # 1. Localizar linha de cabeçalho (COD./CÓDIGO na primeira coluna)
    linha_header = None
    for i in range(min(5, len(df))):
        if _normalizar(df.iloc[i, 0]) in ("COD.", "CODIGO"):
            linha_header = i
            break
    if linha_header is None:
        raise ValueError(f"Cabeçalho não encontrado em {caminho}")

    # 2. Localizar linha de rodapé ("QUANTIDADE TEORICA TOTAL", em col 0 ou col 1)
    linha_fim = None
    for i in range(linha_header + 1, len(df)):
        if "QUANTIDADE TEORICA TOTAL" in (_normalizar(df.iloc[i, 0]) + _normalizar(df.iloc[i, 1])):
            linha_fim = i
            break
    if linha_fim is None:
        raise ValueError(f"Rodapé (QUANTIDADE TEORICA TOTAL) não encontrado em {caminho}")

    dados = df.iloc[linha_header + 1: linha_fim].copy()
    dados.columns = ["ticker", "nome", "tipo", "qtde_teorica", "peso_pct"]
    dados = dados.dropna(subset=["ticker"]).reset_index(drop=True)
    dados["ticker"] = dados["ticker"].astype(str).str.strip()
    dados["peso_pct"] = pd.to_numeric(dados["peso_pct"], errors="coerce")

    # 3. Localizar data de vigência ("PREGÃO BASE" / "PREGAO BASE", em qualquer linha do rodapé)
    data_vigencia = None
    for i in range(linha_fim, len(df)):
        celula0 = _normalizar(df.iloc[i, 0])
        if "PREGAO BASE" in celula0:
            match = re.search(r"(\d{2}/\d{2}/\d{4})", str(df.iloc[i, 0]))
            if match:
                data_vigencia = pd.to_datetime(match.group(1), dayfirst=True)
            elif pd.notna(df.iloc[i, 1]):
                data_vigencia = pd.to_datetime(df.iloc[i, 1])
            break
    if data_vigencia is None:
        raise ValueError(f"Data de vigência (Pregão Base) não encontrada em {caminho}")

    return data_vigencia, dados[["ticker", "nome", "tipo", "peso_pct"]]


# --------------------------------------------------------------------------
# 2. Varredura da pasta inteira -> composição PIT consolidada
# --------------------------------------------------------------------------

def construir_composicao_pit(pasta_raiz: str) -> pd.DataFrame:
    """
    Varre recursivamente todos os arquivos "ViradaFinal*.xlsx" (case-
    insensitive na extensão) dentro de pasta_raiz, e retorna um único
    DataFrame long-format:
        data_vigencia, ticker, nome, tipo, peso_pct, arquivo_fonte
    """
    caminhos = sorted(Path(pasta_raiz).rglob("ViradaFinal*"))
    caminhos = [c for c in caminhos if c.suffix.lower() == ".xlsx"]

    blocos = []
    erros = []
    for caminho in caminhos:
        try:
            data_vigencia, dados = parsear_arquivo_virada(caminho)
            dados = dados.copy()
            dados["data_vigencia"] = data_vigencia
            dados["arquivo_fonte"] = str(caminho)
            blocos.append(dados)
        except Exception as e:
            erros.append((str(caminho), str(e)))

    if erros:
        print(f"[AVISO] {len(erros)} arquivo(s) não puderam ser processados:")
        for caminho, erro in erros:
            print(f"  {caminho}: {erro}")

    composicao = pd.concat(blocos, ignore_index=True)
    return composicao.sort_values(["data_vigencia", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Derivar eventos de entrada/saída a partir de snapshots consecutivos
# --------------------------------------------------------------------------

def derivar_eventos_entrada_saida(composicao_pit: pd.DataFrame) -> pd.DataFrame:
    """
    Compara cada snapshot de vigência com o snapshot IMEDIATAMENTE
    ANTERIOR (ordenado por data) para derivar entradas/saídas -- mesmo
    schema do eventos_composicao_ibovespa_seed.csv anterior, agora
    derivado de dado primário, não de notícia.
    """
    datas_vigencia = sorted(composicao_pit["data_vigencia"].unique())
    eventos = []

    for i in range(1, len(datas_vigencia)):
        data_atual = datas_vigencia[i]
        data_anterior = datas_vigencia[i - 1]

        tickers_atual = set(composicao_pit.loc[composicao_pit["data_vigencia"] == data_atual, "ticker"])
        tickers_anterior = set(composicao_pit.loc[composicao_pit["data_vigencia"] == data_anterior, "ticker"])

        entradas = sorted(tickers_atual - tickers_anterior)
        saidas = sorted(tickers_anterior - tickers_atual)

        eventos.append({
            "data_inicio_vigencia": data_atual,
            "data_fim_vigencia": pd.NaT,  # preenchido no passo seguinte
            "tickers_entrada": ";".join(entradas),
            "tickers_saida": ";".join(saidas),
            "n_ativos": len(tickers_atual),
            "fonte_url": "arquivo oficial B3 (ViradaFinal) -- ver arquivo_fonte em composicao_pit.csv",
        })

    eventos_df = pd.DataFrame(eventos)
    # data_fim_vigencia de cada linha = data_inicio_vigencia da linha seguinte
    eventos_df["data_fim_vigencia"] = eventos_df["data_inicio_vigencia"].shift(-1)
    return eventos_df


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ETL da composição PIT do Ibovespa a partir dos arquivos oficiais B3.")
    parser.add_argument("--pasta", required=True, help="Pasta raiz contendo as subpastas por ano (ex: 'Carteira IBOV/')")
    parser.add_argument("--out", default="resultados_agente0")
    args = parser.parse_args()

    composicao = construir_composicao_pit(args.pasta)
    eventos = derivar_eventos_entrada_saida(composicao)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    composicao.to_csv(out_dir / "composicao_pit_ibovespa.csv", index=False)
    eventos.to_csv(out_dir / "eventos_composicao_ibovespa.csv", index=False)

    print(f"{composicao['data_vigencia'].nunique()} quadrimestres processados "
          f"({composicao['data_vigencia'].min().date()} a {composicao['data_vigencia'].max().date()})")
    print(f"{len(eventos)} eventos de virada derivados")
    print(f"Salvo em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
