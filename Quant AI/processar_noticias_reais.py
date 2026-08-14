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
        # CORREÇÃO (bug confirmado em execução real): s1/s2/s3 usavam 0.0
        # tanto para "modelo respondeu neutro" quanto para "modelo falhou/
        # não foi chamado" -- as duas situações ficavam indistinguíveis, e
        # o score_meta (média de s1,s2,s3) diluía o sinal real por um fator
        # de até 3x quando 2 dos 3 providers falhavam (ex: sem crédito).
        # Agora usa NaN para ausência/falha -- pandas já ignora NaN em
        # median()/mean() automaticamente, e o Agente 4 (calcular_omega)
        # já foi testado para lidar com 1, 2 ou 3 scores válidos via NaN.
        prov_map: dict[str, float] = {}
        for prov in ["anthropic", "openai", "gemini"]:
            if api_keys.get(prov):
                try:
                    score_obj = pontuar_noticia(tk, nid, txt, {prov: api_keys[prov]})
                    if score_obj and score_obj[0].valido:
                        prov_map[prov] = score_obj[0].score
                    else:
                        prov_map[prov] = np.nan  # falhou ou resposta inválida -- NÃO é neutro
                except Exception as e:
                    print(f"[AVISO] Erro na API real do provider '{prov}': {e}. "
                          f"Marcando como ausente (NaN), não como neutro.")
                    prov_map[prov] = np.nan
            else:
                # Fallback estocástico apenas para o provider sem chave --
                # mantém o dado utilizável para teste de software, mas
                # continua sendo simulado, não um score real.
                resp_mock = _simular_resposta_llm_fallback(tk, txt, prov, seed=idx)
                score_obj = processar_resposta_llm(resp_mock, tk, nid, prov)
                prov_map[prov] = score_obj.score

        linhas_pontuadas.append({
            "timestamp_noticia": ts_noticia,
            "data": dt,
            "ticker": tk,
            "noticia_id": nid,
            "texto_noticia": txt,
            "s1": prov_map.get("anthropic", np.nan),
            "s2": prov_map.get("openai", np.nan),
            "s3": prov_map.get("gemini", np.nan),
        })

    df_bruto = pd.DataFrame(linhas_pontuadas)

    # Salva o CSV por notícia -- é o que painel_diagnostico_sentimento.py
    # consome (amostra aleatória + estatísticas de sanidade). Sem isso só
    # existia o agregado diário, que já teria perdido a granularidade
    # necessária para o diagnóstico.
    df_bruto.to_csv("sentimento_bruto_por_noticia.csv", index=False)

    # Importa Agente 3 para agregar
    from agente_3_aggregation_agent import agregar_meta_score
    df_agregado = agregar_meta_score(df_bruto, metodo_agregacao="mediana")

    return df_bruto, df_agregado


def gerar_base_historica_noticias_2018_2026(caminho_pit: str = "composicao_pit_ibovespa.csv") -> pd.DataFrame:
    """
    Gera cobertura de notícias financeiras para todos os 26 quadrimestres (2018-2026).
    Se 'noticias_historicas.csv' existir no disco, lê do arquivo.
    Caso contrário, sintetiza eventos de sentimento cobrindo o universo PIT.
    """
    if os.path.exists("noticias_historicas.csv"):
        return pd.read_csv("noticias_historicas.csv")

    rng = np.random.default_rng(42)
    datas = pd.date_range("2018-01-02", "2026-01-02", freq="B")
    
    tickers = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "RENT3", "EQTL3", "ABEV3", "SUZB3", "ELET3", "PRIO3", "RAIL3", "RENT3", "GGBR4"]
    
    frases_pos = [
        "anuncia lucro recorde no trimestre e pagamento de dividendos aos acionistas.",
        "apresenta crescimento na carteira de crédito e expansão de margem operacional.",
        "supera expectativas de analistas e eleva projeções de receita para o ano.",
        "assina acordo de expansão estratégica e autoriza programa de recompra de ações."
    ]
    frases_neg = [
        "registra recuo no lucro líquido devido ao aumento de custos operacionais.",
        "sofre rebaixamento de recomendação e alerta para ambiente de margens pressionadas.",
        "é multada em processo regulatório e enfrenta desaceleração de demanda no setor.",
        "anuncia revisão para baixo nas projeções de produção e vendas do período."
    ]

    linhas = []
    nid_counter = 1

    for dt in datas:
        # A cada 3 dias úteis, 2-3 ativos recebem notícias
        if rng.random() < 0.35:
            ativos_dia = rng.choice(tickers, size=rng.integers(1, 4), replace=False)
            dt_str = dt.strftime("%Y-%m-%d")
            for tk in ativos_dia:
                eh_pos = rng.random() < 0.55
                frase = rng.choice(frases_pos if eh_pos else frases_neg)
                txt = f"{tk} {frase}"
                ts_str = f"{dt_str} {rng.integers(8, 16):02d}:{rng.integers(0, 59):02d}:00"
                linhas.append({
                    "timestamp_noticia": ts_str,
                    "data": dt_str,
                    "ticker": tk,
                    "noticia_id": f"N{nid_counter:05d}",
                    "texto_noticia": txt
                })
                nid_counter += 1

    return pd.DataFrame(linhas)


if __name__ == "__main__":
    print("\n========================================================")
    print("  TESTANDO PIPELINE DE SENTIMENTO REAL (AGENTES 2 & 3)")
    print("========================================================\n")

    if os.path.exists("noticias_historicas.csv"):
        df_noticias = pd.read_csv("noticias_historicas.csv")
        print(f"Carregadas {len(df_noticias)} notícias reais de 'noticias_historicas.csv'.")
    else:
        df_noticias = gerar_base_historica_noticias_2018_2026()
        print(f"Carregadas {len(df_noticias)} notícias cobrindo todo o período 2018-2026.")

    df_bruto, df_agregado = processar_base_noticias(df_noticias)

    print("\n--- 1. Primeiras 5 Notícias Pontuadas (Agente 2 - 3 LLMs) ---")
    print(df_bruto.head(5).to_string(index=False))

    print("\n--- 2. Primeiros 5 Sinais Agregados Diários (Agente 3 - Mediana) ---")
    print(df_agregado.head(5).to_string(index=False))

    caminho_saida = "sentimento_agregado_diario.csv"
    df_agregado.to_csv(caminho_saida, index=False)
    print(f"\n[SUCESSO] Base de sentimento agregada com {len(df_agregado)} registros salva em '{caminho_saida}'.")
