"""
cache_sentimento.py

Cache persistente (SQLite) dos scores de sentimento já processados pelo
Agente 2, para que um backtest repetido não pague de novo o custo de
tempo/API de pontuar notícias que já foram pontuadas.

DECISÃO DE CHAVE (crítica): a chave do cache inclui o PROVIDER e o NOME
DO MODELO, não só o texto da notícia. Sem isso, trocar de modelo (ex:
gemini-2.0-flash -> gemini-3.5-flash, ou GPT -> Llama) reutilizaria
silenciosamente os scores do modelo antigo -- você acharia que testou o
modelo novo e teria lido cache do velho.

DECISÃO SOBRE FALHAS: chamadas que falharam NÃO são gravadas. O cache
guarda exclusivamente resultados válidos. Consequência deliberada: uma
notícia que falhou será re-chamada em toda execução até dar certo -- é
mais lento, e é esse o preço aceito para garantir que nenhum score
inválido possa jamais entrar no backtest disfarçado de sinal real.
Como o banco nunca contém falhas, não existe caminho possível para um
zero antigo ser reutilizado depois que a causa da falha for corrigida.

Contexto real que motivou essa decisão: em diagnóstico de fev/2026,
Anthropic e OpenAI retornaram erro de crédito e o Gemini retornou 404
(modelo descontinuado) -- as ~1400 notícias do backtest ficaram todas
com score 0.0. Se aquelas falhas tivessem sido cacheadas, o dataset
ficaria zerado mesmo depois de corrigidos os problemas de origem.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agente_2_sentiment_agents import ScoreSentimento

CAMINHO_PADRAO = "cache_sentimento.db"


# --------------------------------------------------------------------------
# 1. Estrutura do banco
# --------------------------------------------------------------------------

def inicializar_cache(caminho: str = CAMINHO_PADRAO) -> sqlite3.Connection:
    """
    Não há coluna `valido` no schema: por política do módulo, só
    resultados válidos entram no banco, então uma coluna que seria
    sempre 1 só criaria a ilusão de que falhas poderiam estar ali.
    """
    conn = sqlite3.connect(caminho)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores_llm (
            hash_chave          TEXT PRIMARY KEY,
            ticker              TEXT NOT NULL,
            noticia_id          TEXT,
            provider            TEXT NOT NULL,
            modelo              TEXT NOT NULL,
            score               REAL NOT NULL,
            justificativa       TEXT,
            data_processamento  TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON scores_llm(provider, modelo)")
    conn.commit()
    return conn


def calcular_hash_chave(texto_noticia: str, ticker: str, provider: str, modelo: str) -> str:
    """
    Chave = f(texto, ticker, provider, modelo). O texto entra normalizado
    (strip + lower) para que diferenças triviais de espaçamento não
    gerem entradas duplicadas do mesmo conteúdo.
    """
    base = f"{texto_noticia.strip().lower()}||{ticker}||{provider}||{modelo}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 2. Leitura e escrita
# --------------------------------------------------------------------------

def buscar_no_cache(
    conn: sqlite3.Connection,
    texto_noticia: str,
    ticker: str,
    provider: str,
    modelo: str,
) -> ScoreSentimento | None:
    """
    Retorna o ScoreSentimento cacheado, ou None se não houver -- e None
    significa sempre "precisa chamar a API".

    Como falhas nunca são gravadas (ver política no topo do módulo), todo
    resultado que sai daqui é necessariamente válido: não existe o caso
    de "achei no cache, mas é lixo". Notícias que falharam simplesmente
    não estão no banco, e por isso são re-tentadas automaticamente na
    execução seguinte, sem precisar de nenhuma flag ou limpeza manual.
    """
    chave = calcular_hash_chave(texto_noticia, ticker, provider, modelo)
    cur = conn.execute(
        "SELECT ticker, noticia_id, provider, score, justificativa "
        "FROM scores_llm WHERE hash_chave = ?", (chave,)
    )
    linha = cur.fetchone()
    if linha is None:
        return None

    tk, nid, prov, score, justificativa = linha
    return ScoreSentimento(
        ticker=tk, noticia_id=nid or "", provider=prov, score=score,
        justificativa=justificativa or "", resposta_bruta="[do cache]", valido=True,
    )


def salvar_no_cache(
    conn: sqlite3.Connection,
    texto_noticia: str,
    modelo: str,
    resultado: ScoreSentimento,
) -> bool:
    """
    Grava o resultado -- SOMENTE se for válido. Retorna True se gravou,
    False se ignorou por ser uma falha.

    Ignorar falhas é a política deliberada do módulo: garante que o banco
    contenha apenas sinal real, ao custo de re-chamar notícias que
    falharam em toda execução até que funcionem.
    """
    if not resultado.valido:
        return False

    chave = calcular_hash_chave(texto_noticia, resultado.ticker, resultado.provider, modelo)
    conn.execute(
        "INSERT OR REPLACE INTO scores_llm "
        "(hash_chave, ticker, noticia_id, provider, modelo, score, justificativa, data_processamento) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chave, resultado.ticker, resultado.noticia_id, resultado.provider, modelo,
            float(resultado.score), resultado.justificativa[:500],
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    return True


# --------------------------------------------------------------------------
# 3. Manutenção e diagnóstico
# --------------------------------------------------------------------------

def estatisticas_cache(conn: sqlite3.Connection) -> dict:
    cur = conn.execute(
        "SELECT provider, modelo, COUNT(*) AS n, AVG(score) AS score_medio, "
        "       MIN(score) AS score_min, MAX(score) AS score_max, "
        "       COUNT(DISTINCT score) AS scores_distintos "
        "FROM scores_llm GROUP BY provider, modelo"
    )
    por_provider = [
        {
            "provider": r[0], "modelo": r[1], "n": r[2], "score_medio": r[3],
            "score_min": r[4], "score_max": r[5], "scores_distintos": r[6],
        }
        for r in cur.fetchall()
    ]
    total = conn.execute("SELECT COUNT(*) FROM scores_llm").fetchone()[0]
    return {"total_entradas": total, "por_provider": por_provider}


def imprimir_estatisticas(conn: sqlite3.Connection) -> None:
    stats = estatisticas_cache(conn)
    print(f"\n=== CACHE DE SENTIMENTO ({stats['total_entradas']} entradas válidas) ===")
    if not stats["por_provider"]:
        print("  (vazio -- nenhuma chamada bem-sucedida ainda)")
        return
    for p in stats["por_provider"]:
        print(f"  {p['provider']:10s} [{p['modelo']:20s}] n={p['n']:6d}  "
              f"média={p['score_medio']:+.4f}  min={p['score_min']:+.2f}  "
              f"max={p['score_max']:+.2f}  distintos={p['scores_distintos']}")
        # Alerta de sanidade: muitos scores, poucos valores distintos =
        # o provider provavelmente não está lendo a notícia de verdade.
        if p["n"] > 10 and p["scores_distintos"] <= 1:
            print(f"    [ALERTA] {p['provider']} tem {p['n']} scores mas apenas "
                  f"{p['scores_distintos']} valor(es) distinto(s) -- provável falha silenciosa.")


def limpar_cache_do_modelo(conn: sqlite3.Connection, provider: str, modelo: str) -> int:
    """
    Remove todas as entradas de um provider/modelo específico -- útil se
    você suspeitar que os scores daquele modelo estão contaminados (ex:
    prompt alterado, ou o modelo estava respondendo mal sem erro
    explícito). Não existe função para "limpar falhas" porque falhas
    nunca chegam a ser gravadas.
    """
    cur = conn.execute("DELETE FROM scores_llm WHERE provider=? AND modelo=?", (provider, modelo))
    conn.commit()
    return cur.rowcount
