"""
agente_2_sentiment_agents.py

Agente 2 do pipeline Quant AI — Sentiment Agents (x3 LLMs independentes).

Responsabilidade única: dado o texto de uma notícia (em português) e o
ativo a que ela se refere, extrair um score de polaridade em [-1, 1] via
3 modelos de LLM INDEPENDENTES entre si -- não 3 chamadas ao mesmo
modelo, que geraria pseudo-independência (mesma arquitetura, mesmo
corpus de treino, mesmos pontos cegos -- discutido na Fase 0 como risco
de viés correlacionado disfarçado de consenso).

Providers escolhidos (3 arquiteturas/organizações diferentes, para
reduzir correlação de viés entre os modelos):
    1. Anthropic (Claude)
    2. OpenAI (GPT)
    3. Google (Gemini)

Cada chamada usa o MESMO prompt estruturado (abaixo), variando só o
provider -- isso isola a fonte de divergência entre os 3 scores como
sendo o modelo em si, não uma diferença de instrução.

NOTA DE AMBIENTE: este script foi desenvolvido num sandbox sem acesso à
internet e sem chaves de API configuradas. As três funções de chamada
(`_chamar_anthropic`, `_chamar_openai`, `_chamar_gemini`) não puderam
ser executadas de fato aqui. A lógica de CONSTRUÇÃO DO PROMPT e de
PARSING/VALIDAÇÃO DA RESPOSTA -- que é onde a maioria dos bugs sutis
costuma aparecer (modelo retorna texto fora do formato esperado, score
fora do intervalo, JSON malformado) -- foi testada com respostas mock,
incluindo casos malformados de propósito.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


PROMPT_TEMPLATE = """Você é um analista financeiro. Leia a notícia abaixo e avalie o \
sentimento dela em relação especificamente à empresa/ativo indicado, \
numa escala de -1 (extremamente negativo para o valor do ativo) a \
+1 (extremamente positivo para o valor do ativo). 0 significa neutro \
ou sem relação direta com o valor do ativo.

Ativo: {ticker}
Notícia: "{texto_noticia}"

Responda APENAS com um JSON no formato exato:
{{"score": <número entre -1.0 e 1.0>, "justificativa": "<até 15 palavras>"}}
"""


@dataclass
class ScoreSentimento:
    ticker: str
    noticia_id: str
    provider: str
    score: float
    justificativa: str
    resposta_bruta: str
    valido: bool


# --------------------------------------------------------------------------
# 1. Parsing e validação da resposta do LLM (testável sem rede)
# --------------------------------------------------------------------------

def _extrair_json_da_resposta(resposta_texto: str) -> dict | None:
    """
    LLMs frequentemente envolvem o JSON pedido em texto extra (crases de
    markdown, frases de preâmbulo) mesmo quando instruídos a responder
    "apenas" com JSON. Tenta o parse direto primeiro; se falhar, tenta
    extrair o primeiro bloco {...} da resposta via regex antes de desistir.
    """
    try:
        return json.loads(resposta_texto.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", resposta_texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def processar_resposta_llm(
    resposta_texto: str,
    ticker: str,
    noticia_id: str,
    provider: str,
) -> ScoreSentimento:
    """
    Converte a resposta bruta de um LLM em um ScoreSentimento validado.
    NUNCA lança exceção para resposta malformada -- marca valido=False
    e score=0.0 (neutro, o valor mais seguro na ausência de sinal
    confiável), para que uma falha de parsing em 1 notícia não derrube
    o pipeline inteiro do Agente 3 rio abaixo.
    """
    dado = _extrair_json_da_resposta(resposta_texto)

    if dado is None or "score" not in dado:
        return ScoreSentimento(
            ticker=ticker, noticia_id=noticia_id, provider=provider,
            score=0.0, justificativa="[parsing falhou -- resposta não continha JSON válido]",
            resposta_bruta=resposta_texto, valido=False,
        )

    try:
        score_bruto = float(dado["score"])
    except (TypeError, ValueError):
        return ScoreSentimento(
            ticker=ticker, noticia_id=noticia_id, provider=provider,
            score=0.0, justificativa="[score não numérico]",
            resposta_bruta=resposta_texto, valido=False,
        )

    score_clipado = max(-1.0, min(1.0, score_bruto))
    fora_do_range = score_clipado != score_bruto

    return ScoreSentimento(
        ticker=ticker, noticia_id=noticia_id, provider=provider,
        score=score_clipado,
        justificativa=str(dado.get("justificativa", ""))[:200],
        resposta_bruta=resposta_texto,
        valido=not fora_do_range,  # ainda usável (score clipado), mas sinaliza anomalia do modelo
    )


# --------------------------------------------------------------------------
# 2. Chamadas reais aos 3 providers (dependem de rede + chaves de API)
# --------------------------------------------------------------------------

def _chamar_anthropic(prompt: str, api_key: str, modelo: str = "claude-sonnet-4-6") -> str:
    import anthropic
    cliente = anthropic.Anthropic(api_key=api_key)
    resposta = cliente.messages.create(
        model=modelo, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resposta.content[0].text


def _chamar_openai(prompt: str, api_key: str, modelo: str = "gpt-4o") -> str:
    from openai import OpenAI
    cliente = OpenAI(api_key=api_key)
    resposta = cliente.chat.completions.create(
        model=modelo, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resposta.choices[0].message.content


def _chamar_gemini(prompt: str, api_key: str, modelo: str = "gemini-2.0-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    cliente = genai.GenerativeModel(modelo)
    resposta = cliente.generate_content(prompt)
    return resposta.text


# --------------------------------------------------------------------------
# 3. Orquestração: 1 notícia -> 3 scores independentes
# --------------------------------------------------------------------------

def pontuar_noticia(
    ticker: str,
    noticia_id: str,
    texto_noticia: str,
    api_keys: dict[str, str],  # {"anthropic": "...", "openai": "...", "gemini": "..."}
) -> list[ScoreSentimento]:
    """
    Chama os 3 providers com o MESMO prompt e retorna os 3 scores
    (já validados/parseados). Falha de UM provider não derruba os
    outros dois -- cada chamada é isolada em try/except.
    """
    prompt = PROMPT_TEMPLATE.format(ticker=ticker, texto_noticia=texto_noticia)
    chamadas = {
        "anthropic": lambda: _chamar_anthropic(prompt, api_keys["anthropic"]),
        "openai": lambda: _chamar_openai(prompt, api_keys["openai"]),
        "gemini": lambda: _chamar_gemini(prompt, api_keys["gemini"]),
    }

    resultados = []
    for provider, chamada in chamadas.items():
        try:
            resposta_bruta = chamada()
        except Exception as e:
            resultados.append(ScoreSentimento(
                ticker=ticker, noticia_id=noticia_id, provider=provider,
                score=0.0, justificativa=f"[chamada ao provider falhou: {e}]",
                resposta_bruta="", valido=False,
            ))
            continue
        resultados.append(processar_resposta_llm(resposta_bruta, ticker, noticia_id, provider))

    return resultados
