"""
coletar_noticias_rss.py

==============================================================================
AVISO CRÍTICO DE USO DO COLETOR RSS
==============================================================================
Este coletor serve apenas para a camada de demonstração ao vivo / forward-testing.
NÃO produz dado histórico para o backtest 2018-2024 — feeds RSS expõem apenas
as notícias recentes e não possuem arquivo retroativo.
==============================================================================

Módulo de Ingestão de Notícias Financeiras ao Vivo via Feeds RSS.
Coleta notícias das principais fontes de notícias financeiras do Brasil
(InfoMoney, MoneyTimes, Valor Econômico) para os tickers do Ibovespa,
estruturando os dados com timestamp completo (data + hora).
"""

from __future__ import annotations

import os
import re
import uuid
import datetime
import email.utils
import xml.etree.ElementTree as ET
import pandas as pd
import requests

FEEDS_RSS_PADRAO = [
    "https://www.infomoney.com.br/feed/",
    "https://www.moneytimes.com.br/feed/",
    "https://valor.globo.com/rss/financas/",
]

def parsear_pubdate_rss(pub_date: str) -> datetime.datetime | None:
    """
    Parseia a string pubDate em formato RFC 822 / RSS (ex: Mon, 15 Jan 2024 14:30:00 +0000).
    Retorna objeto datetime completo ou None se o parse falhar.
    NUNCA retorna fallback silencioso de data atual.
    """
    if not pub_date or not pub_date.strip():
        return None

    # Tenta parser oficial RFC 822 / RFC 2822 da biblioteca padrão
    try:
        dt_tuple = email.utils.parsedate_tz(pub_date)
        if dt_tuple is not None:
            stamp = email.utils.mktime_tz(dt_tuple)
            return datetime.datetime.fromtimestamp(stamp, tz=datetime.timezone.utc)
    except Exception:
        pass

    # Formatos manuais de fallback comuns em feeds RSS brasileiros
    formatos = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    pub_date_clean = pub_date.strip()
    for fmt in formatos:
        try:
            return datetime.datetime.strptime(pub_date_clean, fmt)
        except ValueError:
            continue

    return None


def extrair_tickers_do_texto(texto: str, lista_tickers: list[str]) -> list[str]:
    """Identifica quais tickers da lista aparecem no título/corpo da notícia."""
    encontrados = []
    texto_upper = texto.upper()
    for tk in lista_tickers:
        tk_clean = tk.replace(".SA", "")
        padrao = r'\b' + re.escape(tk_clean) + r'\b'
        if re.search(padrao, texto_upper):
            encontrados.append(tk_clean)
    return encontrados


def baixar_feed_rss(url_feed: str) -> list[dict]:
    """
    Baixa e realiza parsing de um feed RSS em formato XML.
    Descarta rigorosamente qualquer notícia cujo timestamp (pubDate) não puder ser parseado.
    """
    noticias = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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

                    dt_obj = parsear_pubdate_rss(pub_date_raw)
                    if dt_obj is None:
                        print(f"[AVISO] Descartando notícia — não foi possível parsear pubDate: '{pub_date_raw}' ({titulo[:40]}...)")
                        continue  # DESCARTA rigorosamente — sem data fictícia!

                    texto_limpo = re.sub(r'<[^>]+>', '', f"{titulo}. {descricao}")
                    timestamp_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

                    noticias.append({
                        "timestamp_noticia": timestamp_str,
                        "texto": texto_limpo.strip(),
                        "link": item.findtext("link") or ""
                    })
    except Exception as e:
        print(f"[AVISO] Falha ao ler feed {url_feed}: {e}")
    return noticias


def coletar_noticias_para_universo(
    lista_tickers: list[str],
    urls_feeds: list[str] = FEEDS_RSS_PADRAO,
    arquivo_saida: str = "noticias_historicas.csv"
) -> pd.DataFrame:
    """
    Coleta notícias dos feeds RSS e filtra pelas que mencionam os tickers da lista.
    Estrutura a saída com 'timestamp_noticia' preservando hora completa.
    """
    todas_noticias = []
    for feed in urls_feeds:
        print(f"-> Baixando notícias do feed: {feed}...")
        itens = baixar_feed_rss(feed)
        todas_noticias.extend(itens)

    print(f"Total de notícias com timestamp válido coletadas: {len(todas_noticias)}")

    linhas_filtradas = []
    for noti in todas_noticias:
        texto = noti["texto"]
        tickers_mencionados = extrair_tickers_do_texto(texto, lista_tickers)
        for tk in tickers_mencionados:
            nid = f"N_{uuid.uuid4().hex[:8]}"
            linhas_filtradas.append({
                "timestamp_noticia": noti["timestamp_noticia"],
                "ticker": tk,
                "noticia_id": nid,
                "texto_noticia": texto,
            })

    df_res = pd.DataFrame(linhas_filtradas)
    if not df_res.empty:
        df_res = df_res.drop_duplicates(subset=["timestamp_noticia", "ticker", "texto_noticia"])
        df_res.to_csv(arquivo_saida, index=False)
        print(f"[SUCESSO] {len(df_res)} notícias salvas em '{arquivo_saida}' com timestamp completo.")
    else:
        print("[INFO] Nenhum ticker do universo foi encontrado nos feeds ao vivo.")
        df_res = pd.DataFrame(columns=["timestamp_noticia", "ticker", "noticia_id", "texto_noticia"])
        df_res.to_csv(arquivo_saida, index=False)

    return df_res


if __name__ == "__main__":
    tickers_teste = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "RENT3", "EQTL3", "ABEV3"]
    coletar_noticias_para_universo(tickers_teste)
