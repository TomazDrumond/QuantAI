"""
coletar_noticias_newsapi.py

Coletor de notícias financeiras reais via NewsAPI.org para alimentar os
Agentes 2 e 3 do pipeline Minerva IX.

==============================================================================
ESCOPO: Notícias dos ÚLTIMOS 30 DIAS (limitação do plano gratuito NewsAPI).
Finalidade: Demonstração ao vivo do pipeline LLM completo e validação
            funcional do sinal de sentimento com textos reais em PT-BR.
O backtest histórico 2018-2026 permanece com calibração neutra (Q=0).
==============================================================================

Pré-requisito:
    - Obtenha uma chave gratuita em https://newsapi.org/register
    - Configure no Colab Secrets: NEWS_API_KEY = "sua_chave_aqui"
    - Instale: pip install requests
"""

from __future__ import annotations

import os
import uuid
import datetime
import time
import requests
import pandas as pd

# =============================================================================
# Configuração
# =============================================================================

BASE_URL = "https://newsapi.org/v2/everything"

# Empresas do Ibovespa com ticker + nomes em PT-BR para busca
EMPRESAS_UNIVERSO = {
    "PETR4": ["Petrobras", "PETR4"],
    "VALE3": ["Vale", "VALE3"],
    "ITUB4": ["Itaú", "Itau Unibanco", "ITUB4"],
    "BBDC4": ["Bradesco", "BBDC4"],
    "BBAS3": ["Banco do Brasil", "BBAS3"],
    "WEGE3": ["WEG", "WEGE3"],
    "RENT3": ["Localiza", "RENT3"],
    "EQTL3": ["Equatorial", "EQTL3"],
    "ABEV3": ["Ambev", "ABEV3"],
    "SUZB3": ["Suzano", "SUZB3"],
    "ELET3": ["Eletrobras", "ELET3"],
    "RAIL3": ["Rumo", "RAIL3"],
    "GGBR4": ["Gerdau", "GGBR4"],
    "PRIO3": ["PetroRio", "PRIO3"],
    "VIVT3": ["Claro", "Vivo", "Telefonica", "VIVT3"],
    "LREN3": ["Lojas Renner", "LREN3"],
    "MGLU3": ["Magazine Luiza", "Magalu", "MGLU3"],
    "JBSS3": ["JBS", "JBSS3"],
    "BPAC11": ["BTG Pactual", "BPAC11"],
    "RDOR3": ["Rede D'Or", "RDOR3"],
}


def obter_chave_newsapi() -> str:
    """Recupera a chave da NewsAPI dos Secrets do Colab ou variável de ambiente."""
    chave = os.environ.get("NEWS_API_KEY", "")
    if not chave:
        try:
            from google.colab import userdata
            chave = userdata.get("NEWS_API_KEY")
        except Exception:
            pass
    if not chave:
        raise ValueError(
            "Chave NEWS_API_KEY não encontrada.\n"
            "1. Acesse https://newsapi.org/register e obtenha uma chave gratuita.\n"
            "2. No Colab, vá em Secrets (ícone 🔑 no menu lateral) e adicione NEWS_API_KEY."
        )
    return chave


def buscar_noticias_empresa(
    ticker: str,
    termos: list[str],
    api_key: str,
    dias_atras: int = 28,
    max_artigos: int = 20,
) -> list[dict]:
    """
    Busca notícias para uma empresa nos últimos `dias_atras` dias via NewsAPI.
    Retorna lista de dicts com timestamp_noticia, ticker, noticia_id, texto_noticia.
    """
    data_inicio = (datetime.date.today() - datetime.timedelta(days=dias_atras)).isoformat()
    query = " OR ".join(f'"{t}"' for t in termos)

    params = {
        "q": query,
        "language": "pt",
        "from": data_inicio,
        "sortBy": "publishedAt",
        "pageSize": min(max_artigos, 100),
        "apiKey": api_key,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        if resp.status_code == 429:
            print(f"  [AVISO] Rate limit atingido. Aguardando 10s...")
            time.sleep(10)
            resp = requests.get(BASE_URL, params=params, timeout=15)

        if resp.status_code != 200:
            print(f"  [ERRO] Status {resp.status_code} para {ticker}: {resp.json().get('message', '')}")
            return []

        dados = resp.json()
        artigos = dados.get("articles", [])

    except Exception as e:
        print(f"  [ERRO] Falha de conexão para {ticker}: {e}")
        return []

    noticias = []
    for art in artigos:
        published_at = art.get("publishedAt", "")
        if not published_at:
            continue

        # Converte ISO 8601 para string local
        try:
            dt = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        titulo = art.get("title") or ""
        descricao = art.get("description") or ""
        conteudo = art.get("content") or ""
        texto = f"{titulo}. {descricao}. {conteudo}".strip()

        # Remove truncamentos da NewsAPI ("[+X chars]")
        import re
        texto = re.sub(r'\[\+\d+ chars\]', '', texto).strip()

        if len(texto) < 30:
            continue

        noticias.append({
            "timestamp_noticia": ts_str,
            "ticker": ticker,
            "noticia_id": f"N_{uuid.uuid4().hex[:8]}",
            "texto_noticia": texto[:1000],  # Limita para economia de tokens LLM
        })

    return noticias


def coletar_todas_as_noticias(
    empresas: dict[str, list[str]] = EMPRESAS_UNIVERSO,
    dias_atras: int = 28,
    arquivo_saida: str = "noticias_historicas.csv",
    max_por_empresa: int = 20,
) -> pd.DataFrame:
    """
    Coleta notícias dos últimos 28 dias para todas as empresas do universo.
    Salva em 'noticias_historicas.csv' para ser consumido por processar_noticias_reais.py.
    """
    api_key = obter_chave_newsapi()

    print(f"\n{'='*60}")
    print(f"  COLETANDO NOTÍCIAS VIA NEWSAPI ({len(empresas)} empresas)")
    print(f"  Período: últimos {dias_atras} dias")
    print(f"{'='*60}\n")

    todas = []
    for i, (ticker, termos) in enumerate(empresas.items(), 1):
        print(f"  [{i:02d}/{len(empresas)}] {ticker:8s} | buscando por: {', '.join(termos[:2])}...")
        noticias = buscar_noticias_empresa(ticker, termos, api_key, dias_atras, max_por_empresa)
        todas.extend(noticias)
        print(f"            → {len(noticias)} notícias coletadas.")
        time.sleep(0.5)  # Respeita rate limit do plano gratuito (500 req/dia)

    if not todas:
        print("\n[AVISO] Nenhuma notícia coletada. Verifique a chave ou conexão.")
        return pd.DataFrame(columns=["timestamp_noticia", "ticker", "noticia_id", "texto_noticia"])

    df = pd.DataFrame(todas)
    df = df.drop_duplicates(subset=["ticker", "texto_noticia"])
    df = df.sort_values("timestamp_noticia").reset_index(drop=True)

    # Adiciona coluna 'data' obrigatória para compatibilidade com processar_noticias_reais.py
    df["data"] = pd.to_datetime(df["timestamp_noticia"]).dt.strftime("%Y-%m-%d")

    df.to_csv(arquivo_saida, index=False)

    print(f"\n{'='*60}")
    print(f"  [SUCESSO] {len(df)} notícias salvas em '{arquivo_saida}'.")
    print(f"  Período: {df['data'].min()} → {df['data'].max()}")
    print(f"  Tickers cobertos: {sorted(df['ticker'].unique().tolist())}")
    print(f"{'='*60}\n")

    return df


def resumir_coleta(df: pd.DataFrame) -> None:
    """Imprime resumo da cobertura de notícias coletadas."""
    if df.empty:
        print("Nenhum dado para resumir.")
        return
    print("\n=== RESUMO DA COBERTURA DE NOTÍCIAS ===")
    resumo = df.groupby("ticker").agg(
        total_noticias=("noticia_id", "count"),
        data_mais_antiga=("data", "min"),
        data_mais_recente=("data", "max"),
    ).reset_index()
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    df_noticias = coletar_todas_as_noticias()
    resumir_coleta(df_noticias)
    print("\nPróximo passo: execute 'processar_noticias_reais.py' para pontuar com os 3 LLMs.")
