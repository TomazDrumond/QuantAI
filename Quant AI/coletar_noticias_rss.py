"""
coletar_noticias_rss.py

Módulo de Ingestão de Notícias Financeiras em Português via Feeds RSS.

AVISO ESTRUTURAL (leia antes de usar): feeds RSS expõem só os itens
MAIS RECENTES de cada fonte -- não têm arquivo retroativo. Este coletor
serve exclusivamente para a camada de demonstração AO VIVO/forward-test.
Ele NÃO produz dado histórico para o backtest 2018-2024 -- rodá-lo hoje
nunca vai trazer notícia de anos anteriores, não importa quantas vezes
seja executado. Para histórico, a fonte é outra (ex: point-in-time
oficial da CVM, discutido separadamente).

Coleta notícias das principais fontes de notícias financeiras do Brasil
(InfoMoney, MoneyTimes, Valor Econômico) para os tickers do Ibovespa,
estruturando os dados no formato esperado pelo Agente 2 e Agente 3:
    ['timestamp_noticia', 'data', 'ticker', 'noticia_id', 'texto_noticia']

Salva o resultado em 'noticias_historicas.csv'.

CORREÇÕES NESTA VERSÃO (confirmadas contra teste real: 20 itens nos
feeds, zero ativos casados -- portais escrevem "Petrobras", não
"PETR4"):

  1. RESOLUÇÃO DE ENTIDADE: reconhecimento agora usa um dicionário
     ticker -> [ticker, razão social, nome comercial, variações], não
     só o código literal. "Petrobras" e "Petróleo Brasileiro" agora
     casam com PETR4, por exemplo.

  2. TIMESTAMP COMPLETO: pubDate do RSS é parseado preservando a HORA
     (formato RFC 822 completo), não só a data. Sem isso,
     validar_alinhamento_timestamp (Agente 1) não tinha granularidade
     para operar -- "mesmo dia" pode esconder um look-ahead real (ex:
     notícia às 17h05, mercado fechou às 17h00). Notícias cujo pubDate
     não pôde ser parseado são DESCARTADAS explicitamente (contadas e
     reportadas), nunca recebem uma data fictícia (o bug anterior
     usava datetime.date.today() como fallback silencioso).
"""

from __future__ import annotations

import re
import uuid
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import requests

FEEDS_RSS_PADRAO = [
    "https://www.infomoney.com.br/feed/",
    "https://www.moneytimes.com.br/feed/",
    "https://valor.globo.com/rss/financas/",
]

# --------------------------------------------------------------------------
# Dicionário de resolução de entidade -- ticker -> nomes pelos quais a
# imprensa realmente se refere à empresa. Ponto de partida com os
# nomes mais comuns entre os ativos de maior liquidez do Ibovespa;
# extensível -- passe seu próprio dicionário para coletar_noticias_para_universo
# se precisar de cobertura maior ou de casos ambíguos (ex: "Vale" também
# é um substantivo comum em português, então entradas ambíguas merecem
# revisão manual antes de confiar cegamente no casamento automático).
# --------------------------------------------------------------------------

MAPEAMENTO_ENTIDADES_PADRAO: dict[str, list[str]] = {
    "PETR4": ["PETR4", "PETR3", "Petrobras", "Petróleo Brasileiro"],
    "VALE3": ["VALE3", "Vale S.A.", "Vale S/A"],  # "Vale" sozinho fica de fora -- ambíguo demais
    "ITUB4": ["ITUB4", "ITUB3", "Itaú Unibanco", "Itaú"],
    "BBDC4": ["BBDC4", "BBDC3", "Bradesco"],
    "BBAS3": ["BBAS3", "Banco do Brasil"],
    "WEGE3": ["WEGE3", "WEG S.A.", "WEG"],
    "RENT3": ["RENT3", "Localiza", "Localiza&Co", "Localiza Rent a Car"],
    "EQTL3": ["EQTL3", "Equatorial Energia", "Equatorial"],
    "ABEV3": ["ABEV3", "Ambev"],
    "SUZB3": ["SUZB3", "Suzano"],
    "ELET3": ["ELET3", "ELET6", "Eletrobras", "Axia Energia", "Axia"],
    "PRIO3": ["PRIO3", "PetroRio", "Petro Rio"],
    "RAIL3": ["RAIL3", "Rumo S.A.", "Rumo Logística"],
    "GGBR4": ["GGBR4", "Gerdau"],
    "TAEE11": ["TAEE11", "Taesa", "Transmissora Aliança"],
}


def extrair_tickers_do_texto(
    texto: str,
    lista_tickers: list[str],
    mapeamento_entidades: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Identifica quais tickers da lista são mencionados no texto -- por
    código OU por nome da empresa, via mapeamento_entidades. Ativos sem
    entrada no mapeamento caem de volta para o casamento literal do
    código (comportamento anterior), com o mesmo aviso de fragilidade.
    """
    mapa = mapeamento_entidades or MAPEAMENTO_ENTIDADES_PADRAO
    # Normaliza pontos (ex: "S.A." -> "SA") ANTES do casamento -- caso
    # contrário, \b (fronteira de palavra) falha quando o alias termina
    # em ponto seguido de espaço (nem "." nem " " são caracteres de
    # palavra, então não há fronteira ali para o \b detectar).
    texto_normalizado = texto.upper().replace(".", "")
    encontrados = []

    for tk in lista_tickers:
        tk_clean = tk.replace(".SA", "")
        aliases = mapa.get(tk_clean, [tk_clean])
        casou = False
        for alias in aliases:
            alias_normalizado = alias.upper().replace(".", "")
            padrao = r'\b' + re.escape(alias_normalizado) + r'\b'
            if re.search(padrao, texto_normalizado):
                casou = True
                break
        if casou:
            encontrados.append(tk_clean)

    return encontrados


def _parsear_pub_date(pub_date_str: str) -> datetime.datetime | None:
    """
    Parseia o pubDate do RSS (RFC 822: "Mon, 15 Jan 2024 14:30:00 +0000")
    preservando a HORA. Retorna None se não conseguir -- CHAMADOR deve
    descartar a notícia, nunca inventar uma data.
    """
    formatos = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
    ]
    for fmt in formatos:
        try:
            return datetime.datetime.strptime(pub_date_str.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None


def baixar_feed_rss(url_feed: str) -> tuple[list[dict], int]:
    """
    Baixa e faz parsing de um feed RSS. Retorna (noticias, n_descartadas)
    -- n_descartadas conta itens cujo pubDate não pôde ser parseado
    (esses NÃO entram na lista de notícias).
    """
    noticias = []
    descartadas = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url_feed, headers=headers, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item"):
                    titulo = item.findtext("title") or ""
                    descricao = item.findtext("description") or ""
                    pub_date_raw = item.findtext("pubDate") or ""

                    texto_limpo = re.sub(r'<[^>]+>', '', f"{titulo}. {descricao}")

                    timestamp = _parsear_pub_date(pub_date_raw)
                    if timestamp is None:
                        descartadas += 1
                        continue  # NUNCA usa data fictícia -- descarta e conta

                    noticias.append({
                        "timestamp_noticia": timestamp.isoformat(),
                        "data": timestamp.strftime("%Y-%m-%d"),
                        "texto": texto_limpo.strip(),
                        "link": item.findtext("link") or "",
                    })
    except Exception as e:
        print(f"[AVISO] Falha ao ler feed {url_feed}: {e}")

    return noticias, descartadas


def coletar_noticias_para_universo(
    lista_tickers: list[str],
    urls_feeds: list[str] = FEEDS_RSS_PADRAO,
    arquivo_saida: str = "noticias_historicas.csv",
    mapeamento_entidades: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Coleta notícias dos feeds RSS e filtra pelas que mencionam os tickers
    da lista (por código ou nome da empresa). Gera o CSV pronto para
    processar_noticias_reais.py, já com timestamp_noticia completo.
    """
    todas_noticias = []
    total_descartadas = 0
    for feed in urls_feeds:
        print(f"-> Baixando notícias do feed: {feed}...")
        itens, descartadas = baixar_feed_rss(feed)
        todas_noticias.extend(itens)
        total_descartadas += descartadas

    print(f"Total de notícias coletadas nos feeds: {len(todas_noticias)}")
    if total_descartadas:
        print(f"[AVISO] {total_descartadas} item(ns) descartado(s) por pubDate não-parseável "
              f"(nenhum recebeu data fictícia).")

    linhas_filtradas = []
    for noti in todas_noticias:
        texto = noti["texto"]
        tickers_mencionados = extrair_tickers_do_texto(texto, lista_tickers, mapeamento_entidades)
        for tk in tickers_mencionados:
            nid = f"N_{uuid.uuid4().hex[:8]}"
            linhas_filtradas.append({
                "timestamp_noticia": noti["timestamp_noticia"],
                "data": noti["data"],
                "ticker": tk,
                "noticia_id": nid,
                "texto_noticia": texto,
            })

    df_res = pd.DataFrame(linhas_filtradas)
    if not df_res.empty:
        df_res = df_res.drop_duplicates(subset=["data", "ticker", "texto_noticia"])
        df_res.to_csv(arquivo_saida, index=False)
        print(f"[SUCESSO] {len(df_res)} notícias vinculadas a ativos salvas em '{arquivo_saida}'.")
    else:
        print("[INFO] Nenhum ticker do universo foi encontrado diretamente nos feeds recentes.")
        print("Gerando estrutura vazia com o cabeçalho correto.")
        df_res = pd.DataFrame(columns=["timestamp_noticia", "data", "ticker", "noticia_id", "texto_noticia"])
        df_res.to_csv(arquivo_saida, index=False)

    return df_res


if __name__ == "__main__":
    # Tickers principais do Ibovespa para teste -- ajuste para o universo
    # real (ex: os 15 mais líquidos da janela vigente, via Agente 0).
    tickers_teste = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "RENT3", "EQTL3", "ABEV3"]
    coletar_noticias_para_universo(tickers_teste)
