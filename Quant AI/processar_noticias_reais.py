"""
processar_noticias_reais.py

Orquestrador dos Agentes 2 (Sentiment Agents) e 3 (Aggregation Agent).

Fluxo:
    1. Ingestão de notícias financeiras (CSV/JSON contendo: timestamp_noticia/data, ticker, noticia_id, texto_noticia).
    2. Processamento via Agente 2: pontua cada notícia com 3 LLMs independentes (Claude, GPT, Gemini).
       - Checa chaves POR PROVIDER INDIVIDUAL: se Anthropic tem chave, usa API real; se OpenAI não tem chave, usa fallback simulado apenas para OpenAI.
    3. Agregação via Agente 3: colapsa múltiplas notícias do mesmo (ativo, data) calculando a MEDIANA
       dos sentimentos para gerar (s1, s2, s3, n_noticias, score_meta).
    4. Exportação do resultado consolidado em 'sentimento_agregado_diario.csv'.
"""

from __future__ import annotations

import os
import re
import pandas as pd
import numpy as np

from agente_2_sentiment_agents import pontuar_noticia, ScoreSentimento, processar_resposta_llm


# --------------------------------------------------------------------------
# Fallback Deterministico para Simulação Semântica (por provider sem chave)
# --------------------------------------------------------------------------

PALAVRAS_POSITIVAS = [
    "lucro", "recorde", "alta", "crescimento", "superou", "expansão", "aprovado",
    "dividendo", "recompras", "elevação", "forte", "positivo", "alta de"
]
PALAVRAS_NEGATIVAS = [
    "prejuízo", "queda", "recuo", "multa", "investigação", "corte", "rebaixado",
    "fraco", "calote", "crise", "risco", "processo", "queda de"
]

def _simular_resposta_llm_fallback(ticker: str, texto_noticia: str, provider: str, seed: int = 42) -> str:
    """
    Simula uma resposta JSON estruturada de LLM com pequeno ruído estocástico
    quando a chave do provider específico não estiver configurada.
    """
    texto_lc = texto_noticia.lower()
    score_base = 0.0

    for p in PALAVRAS_POSITIVAS:
        if p in texto_lc:
            score_base += 0.35
    for n in PALAVRAS_NEGATIVAS:
        if n in texto_lc:
            score_base -= 0.35

    # Ruído stocástico leve para evitar variância artificial zero constante
    rng = np.random.default_rng(seed + abs(hash(provider)) % 10000)
    prov_offset = {"anthropic": 0.05, "openai": -0.05, "gemini": 0.0}.get(provider, 0.0)
    ruido = rng.normal(loc=0.0, scale=0.04)

    score_final = max(-1.0, min(1.0, round(score_base + prov_offset + ruido, 2)))
    justificativa = "Sentimento positivo (simulado)" if score_final > 0 else ("Sentimento negativo (simulado)" if score_final < 0 else "Sentimento neutro (simulado)")
    return f'{{"score": {score_final}, "justificativa": "{justificativa} para {ticker}"}}'


def obter_chaves_api() -> dict[str, str]:
    """Recupera chaves de API do ambiente do sistema."""
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
    }


def processar_base_noticias(
    df_noticias: pd.DataFrame,
    usar_fallback_se_sem_chave: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recebe um DataFrame com colunas: ['timestamp_noticia' (ou 'data'), 'ticker', 'noticia_id', 'texto_noticia']
    Retorna:
        - df_bruto_pontuado: uma linha por notícia com s1, s2, s3.
        - df_agregado_diario: uma linha por (data, ticker) agregada via Agente 3 por mediana.
    """
    # Aceita tanto 'timestamp_noticia' quanto 'data'
    if "timestamp_noticia" in df_noticias.columns and "data" not in df_noticias.columns:
        df_noticias = df_noticias.copy()
        df_noticias["data"] = pd.to_datetime(df_noticias["timestamp_noticia"]).dt.strftime("%Y-%m-%d")

    obrigadas = {"data", "ticker", "noticia_id", "texto_noticia"}
    if not obrigadas.issubset(df_noticias.columns):
        raise ValueError(f"O DataFrame de entrada deve conter as colunas: {obrigadas}")

    api_keys = obter_chaves_api()
    providers_presentes = [p for p, k in api_keys.items() if bool(k)]
    providers_faltantes = [p for p, k in api_keys.items() if not bool(k)]

    if providers_presentes:
        print(f"[INFO] Chaves reais encontradas para: {providers_presentes}. Usando API real para estes modelos.")
    if providers_faltantes:
        print(f"[AVISO] Chaves ausentes para: {providers_faltantes}. Modo fallback ativado APENAS para estes providers.")
        print("[AVISO CRÍTICO] NÃO use a saída do modo fallback para calibrar epsilon_min/corr_min, a variância entre 'modelos' é artificial.")

    linhas_pontuadas = []

    for idx, row in df_noticias.iterrows():
        dt = row["data"]
        tk = row["ticker"]
        nid = str(row["noticia_id"])
        txt = str(row["texto_noticia"])
        ts_noticia = row.get("timestamp_noticia", f"{dt} 08:00:00")

        # Processamento POR PROVIDER (não descarta providers com chave válida!)
        prov_map = {}
        for prov in ["anthropic", "openai", "gemini"]:
            if api_keys.get(prov):
                # Executa API real do provider individual
                try:
                    score_obj = pontuar_noticia(tk, nid, txt, {prov: api_keys[prov]})
                    if score_obj:
                        prov_map[prov] = score_obj[0].score
                    else:
                        prov_map[prov] = 0.0
                except Exception as e:
                    print(f"[AVISO] Erro na API real do provider '{prov}': {e}. Usando fallback para esta notícia.")
                    resp_mock = _simular_resposta_llm_fallback(tk, txt, prov, seed=idx)
                    score_obj = processar_resposta_llm(resp_mock, tk, nid, prov)
                    prov_map[prov] = score_obj.score
            else:
                # Fallback estocástico apenas para o provider sem chave
                resp_mock = _simular_resposta_llm_fallback(tk, txt, prov, seed=idx)
                score_obj = processar_resposta_llm(resp_mock, tk, nid, prov)
                prov_map[prov] = score_obj.score

        linhas_pontuadas.append({
            "timestamp_noticia": ts_noticia,
            "data": dt,
            "ticker": tk,
            "noticia_id": nid,
            "s1": prov_map.get("anthropic", 0.0),
            "s2": prov_map.get("openai", 0.0),
            "s3": prov_map.get("gemini", 0.0),
        })

    df_bruto = pd.DataFrame(linhas_pontuadas)
    
    # Importa Agente 3 para agregar
    from agente_3_aggregation_agent import agregar_meta_score
    df_agregado = agregar_meta_score(df_bruto, metodo_agregacao="mediana")

    return df_bruto, df_agregado


def gerar_amostra_noticias_demonstracao() -> pd.DataFrame:
    """Gera um conjunto de dados de notícias financeiras para demonstração e testes."""
    dados = [
        {"timestamp_noticia": "2024-01-15 08:30:00", "data": "2024-01-15", "ticker": "PETR4", "noticia_id": "N001", "texto_noticia": "Petrobras anuncia lucro recorde no trimestre e pagamento de dividendos extraordinários aos acionistas."},
        {"timestamp_noticia": "2024-01-15 09:15:00", "data": "2024-01-15", "ticker": "PETR4", "noticia_id": "N002", "texto_noticia": "Preço do petróleo cai no mercado internacional após aumento de estoques nos EUA."},
        {"timestamp_noticia": "2024-01-15 14:00:00", "data": "2024-01-15", "ticker": "VALE3", "noticia_id": "N003", "texto_noticia": "Vale assina acordo de expansão em mina e eleva projeções de produção de minério de ferro."},
        {"timestamp_noticia": "2024-01-16 08:45:00", "data": "2024-01-16", "ticker": "VALE3", "noticia_id": "N004", "texto_noticia": "Justiça aplica multa em processo sobre barragem da Vale em Minas Gerais."},
        {"timestamp_noticia": "2024-01-16 10:00:00", "data": "2024-01-16", "ticker": "ITUB4", "noticia_id": "N005", "texto_noticia": "Itaú Unibanco apresenta crescimento na carteira de crédito e inadimplência sob controle no período."},
        {"timestamp_noticia": "2024-01-17 09:00:00", "data": "2024-01-17", "ticker": "BBAS3", "noticia_id": "N006", "texto_noticia": "Banco do Brasil atinge marca histórica de rentabilidade com forte avanço no crédito agronegócio."},
    ]
    return pd.DataFrame(dados)


if __name__ == "__main__":
    print("\n========================================================")
    print("  TESTANDO PIPELINE DE SENTIMENTO REAL (AGENTES 2 & 3)")
    print("========================================================\n")

    df_noticias = gerar_amostra_noticias_demonstracao()
    print(f"Carregadas {len(df_noticias)} notícias de teste.")

    df_bruto, df_agregado = processar_base_noticias(df_noticias)

    print("\n--- 1. Scores Brutos por Notícia (Agente 2 - 3 LLMs) ---")
    print(df_bruto.to_string(index=False))

    print("\n--- 2. Agregação Diária por Ativo (Agente 3 - Mediana) ---")
    print(df_agregado.to_string(index=False))

    caminho_saida = "sentimento_agregado_diario.csv"
    df_agregado.to_csv(caminho_saida, index=False)
    print(f"\n[SUCESSO] Base de sentimento agregada salva com sucesso em '{caminho_saida}'.")
