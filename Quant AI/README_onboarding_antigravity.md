# Contexto do Projeto — Quant AI (Itaú Quantamental Challenge 2026)

Você vai me ajudar a continuar um projeto de otimização de portfólio que
integra sentimento multi-LLM ao framework Black-Litterman, para o
universo Ibovespa/B3. As decisões teóricas e de arquitetura já foram
fechadas numa sessão anterior (com Claude, via chat). Este README
resume o que já está decidido — trate como premissas fixas, não como
sugestões a revisitar sem necessidade.

## Decisões já fechadas (não reabrir sem justificativa forte)

- **Universo**: 10-15 ativos de maior liquidez do Ibovespa, selecionados
  dentro de cada janela point-in-time (PIT) — nunca lista fixa atual
  aplicada retroativamente (survivorship bias).
- **Fonte de notícia**: apenas português (InfoMoney, Money Times, B3
  fatos relevantes/CVM — priorizar RSS/APIs oficiais sobre scraping de
  HTML). X/Twitter descartado como fonte de backtest histórico (full-
  archive search custa Enterprise, ~US$42k/mês) — só cogitável para uma
  camada de demonstração ao vivo, fora do escopo do backtest reportado.
- **Carteira**: LONG-ONLY (sem venda a descoberto), teto fixo de 20% por
  ativo (ver `agente_6_bl_optimizer.py` — testei e confirmei que isso é
  factível com até 15 ativos; a regra 5/10/40 da UCITS foi tentada e
  descartada por ser matematicamente inviável para um universo tão
  pequeno).
- **Omega (incerteza da view)**: `omega_i = max(Var(s1,s2,s3), epsilon_min) / sqrt(n_noticias)`
  — ver `calibracao_epsilon_min.py` para o grid de calibração walk-forward
  desse parâmetro.
- **Composição histórica do Ibovespa**: RESOLVIDO com dado primário oficial
  da B3 (arquivos "ViradaFinal", fornecidos pelo usuário) -- ver
  `agente_0_universe_data.py`, que já rodou e gerou
  `composicao_pit_ibovespa.csv` (26 quadrimestres, dez/2017 a jan/2026)
  e `eventos_composicao_ibovespa.csv`. O dataset Kaggle e a reconstrução
  via notícia (eventos_composicao_ibovespa_seed.csv) ficam como validação
  cruzada opcional, não mais como fonte primária -- todos os 6 eventos
  reconstruídos via notícia na sessão anterior foram conferidos contra
  este dado oficial e bateram exatamente.

## Arquitetura de 9 agentes (0 a 8)

0. Universe & Reference Data — composição PIT, delistagens, calendário B3
1. Data Agent — preços + alinhamento de timestamps (SEM look-ahead bias:
   `timestamp(notícia) < timestamp(decisão) <= timestamp(preço)`)
2. Sentiment Agents (x3 LLMs independentes, só português)
3. Aggregation Agent — Meta-Score de consenso
4. Correlation Filter — deriva Omega, calibrado com defasagem t-1 do Agente 8
5. Monte Carlo Agent — decaimento temporal via processo de Wiener (forward-only,
   nunca interpolação bidirecional — vazaria dado futuro)
6. BL Optimizer — ver `agente_6_bl_optimizer.py` (cvxpy, long-only, teto 20%)
7. Rebalancing Agent — banda de não-negociação proporcional à liquidez
   (1,0 desvio-padrão do sinal para ativos líquidos, 1,5 para menos líquidos)
8. Validator Agent — backtest walk-forward, retroalimenta epsilon_min (Agente 4)

## Bootstrap no Colab (SEMPRE use este trecho, não a versão antiga)

A versão antiga ("clona só se a pasta não existir") deixa o Colab preso
em versões desatualizadas do repositório depois da primeira execução --
use este reset completo em vez disso, toda vez que tiver atualizado
algo no GitHub:

```python
import os

# garante que o processo NUNCA está dentro da pasta que vamos apagar
%cd /content

if os.path.exists('/content/QuantAI'):
    !rm -rf /content/QuantAI

!git clone https://github.com/TomazDrumond/QuantAI.git /content/QuantAI
%cd "/content/QuantAI/Quant AI"

import sys
caminho_modulos = '/content/QuantAI/Quant AI'
if caminho_modulos not in sys.path:
    sys.path.insert(0, caminho_modulos)

!pip install cvxpy yfinance pandas_market_calendars -q
```

**Ordem obrigatória, sem pular etapa**: (1) reinicie a sessão primeiro
(`Ambiente de execução → Reiniciar sessão`), (2) só depois rode esta
célula. Rodar esta célula SEM reiniciar antes pode apagar a pasta
enquanto o processo ainda está posicionado dentro dela (erro típico:
`shell-init: error retrieving current directory`) -- o `%cd /content`
no início mitiga isso, mas reiniciar continua sendo necessário para
limpar o cache de import do Python (ver abaixo).

**Depois de rodar isso, SEMPRE reinicie a sessão** (`Ambiente de
execução → Reiniciar sessão`) antes de rodar as células de import --
o Python mantém módulos já importados em cache (`sys.modules`) mesmo
que o arquivo tenha mudado no disco; sem reiniciar, o import não
relê o arquivo atualizado. Depois de reiniciar, rode as células na
ordem, de cima para baixo.

## Achado crítico de calibração (out/2026 -- ver agente_6_bl_optimizer.py)

Teste de fumaça real no Colab revelou que `tau` e `Omega` podem estar
em escalas incompatíveis: com valores arbitrários (tau=0.05, Omega na
faixa 0.02-0.04), a precisão do prior ficou 40-75x maior que a precisão
das views -- toda view era praticamente ignorada, e o BL Optimizer
devolvia pesos quase idênticos entre ativos, independente do sentimento
real. Um `verificar_calibracao_tau_omega()` foi adicionado ao Agente 6
e roda automaticamente dentro de `calcular_posterior_bl`, imprimindo um
aviso quando essa razão passa de 5x -- mas ele só AVISA, não corrige
sozinho. `tau` ainda precisa ser calibrado de verdade (valor de
referência testado nesta sessão: tau=2.5 fez a view responder de forma
sensata; tau=0.05, o do teste de fumaça original, não).



1. Rode `python calibracao_epsilon_min.py --demo`,
   `python calibracao_corr_min.py --demo` e confirme que os testes do
   Agente 1 (ver docstring/testes já embutidos) rodam sem erro.
2. Instale `cvxpy`, `yfinance`, `pandas_market_calendars`
   (`pip install cvxpy yfinance pandas_market_calendars`) e valide que
   `agente_6_bl_optimizer.py` e `agente_1_data_agent.py` importam sem erro.
3. A composição PIT do Ibovespa (Agente 0) JÁ ESTÁ PRONTA --
   `composicao_pit_ibovespa.csv` e `eventos_composicao_ibovespa.csv` já
   foram gerados a partir dos arquivos oficiais da B3. Não precisa
   reprocessar, a menos que novos quadrimestres (2026 em diante) precisem
   ser adicionados -- nesse caso, rode
   `python agente_0_universe_data.py --pasta "Carteira IBOV/" --out resultados_agente0`
   de novo com os novos arquivos na pasta.
4. Próximo gargalo real: buscar preços diários (Agente 1, via `yfinance`,
   tickers com sufixo `.SA`) para os ativos que aparecem em
   `composicao_pit_ibovespa.csv`, e coletar notícias em português
   (InfoMoney/Money Times RSS) para alimentar o Agente 2.

## Regras gerais de trabalho

- Priorize RSS/APIs oficiais sobre scraping agressivo de HTML.
- Toda decisão de parâmetro (ex: epsilon_min, w_max, banda de
  rebalanceamento) precisa ser calibrada com dados até t-1, nunca
  contemporâneos à decisão que está sendo tomada (walk-forward estrito).
- Documente qualquer desvio das premissas acima como limitação explícita,
  não como mudança silenciosa.

## Status de Implementação (atualizado ago/2026)

| Agente / Módulo | Arquivo | Status |
|---|---|---|
| Agente 0 — Universe & PIT | `agente_0_universe_data.py` | ✅ Completo |
| Agente 1 — Data Agent | `agente_1_data_agent.py` + `buscar_precos_historicos.py` | ✅ Completo |
| Agente 2 — Sentiment (3 LLMs) | `agente_2_sentiment_agents.py` | ✅ Completo |
| Agente 3 — Aggregation (Mediana) | `agente_3_aggregation_agent.py` | ✅ Completo |
| Agente 4 — Correlation Filter | `agente_4_correlation_filter.py` | ✅ Completo |
| Agente 5 — Monte Carlo Decay | `agente_5_monte_carlo_agent.py` | ✅ Completo |
| Agente 6 — BL Optimizer | `agente_6_bl_optimizer.py` | ✅ Completo |
| Agente 7 — Rebalancing | `agente_7_rebalancing_agent.py` | ✅ Completo |
| Agente 8 — Validator | `agente_8_validator_agent.py` | ✅ Completo |
| Orquestrador Sentimento | `processar_noticias_reais.py` | ✅ Completo |
| Pipeline Walk-Forward | `pipeline_quant_ai_colab.py` | ✅ Completo |
| Gráficos de Relatório | `gerar_graficos_relatorio.py` | ✅ Completo |
| Calibração ε_min | `calibracao_epsilon_min.py` | ✅ Completo |
| Calibração corr_min | `calibracao_corr_min.py` | ✅ Completo |
| Deploy Google Colab | `QUANT_AI_COLAB_DEPLOY.py` | ✅ Completo |

## Mapa de Arquivos do Projeto

```
Quant AI/
├── agente_0_universe_data.py          # PIT Ibovespa
├── agente_1_data_agent.py             # Preços + gaps Wiener
├── agente_2_sentiment_agents.py       # Claude + GPT + Gemini
├── agente_3_aggregation_agent.py      # Mediana entre LLMs e notícias
├── agente_4_correlation_filter.py     # Purificação + Omega
├── agente_5_monte_carlo_agent.py      # Decaimento OU via MC
├── agente_6_bl_optimizer.py           # Black-Litterman + CVXPY
├── agente_7_rebalancing_agent.py      # Bandas de não-negociação
├── agente_8_validator_agent.py        # DSR, Sharpe, Sortino, Drawdown
├── buscar_precos_historicos.py        # yfinance + De-Para B3
├── calibracao_epsilon_min.py          # Grid walk-forward para ε_min
├── calibracao_corr_min.py             # Grid walk-forward para corr_min
├── processar_noticias_reais.py        # Orquestrador Agentes 2+3
├── gerar_graficos_relatorio.py        # Matplotlib: patrimônio + pesos
├── pipeline_quant_ai_colab.py         # Orquestrador principal (Agentes 0-8)
├── QUANT_AI_COLAB_DEPLOY.py           # Guia de deploy Colab (6 células)
├── requirements.txt                   # Dependências pip
├── composicao_pit_ibovespa.csv        # PIT oficial B3 (26 quadrimestres)
├── eventos_composicao_ibovespa.csv    # Eventos de entrada/saída PIT
└── precos_historicos.csv              # Preços diários (gerado pelo Agente 1)
```

## Pendências Remanescentes

1. **Corpus de notícias reais (PT-BR)**: alimentar `processar_noticias_reais.py`
   com um CSV real de `(data, ticker, noticia_id, texto_noticia)` coletado
   via InfoMoney/Money Times RSS. Enquanto isso, o fallback semântico interno
   garante execução ponta-a-ponta do pipeline.

2. **Chaves de API LLMs**: cadastrar `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` e
   `GEMINI_API_KEY` nos Secrets do Colab (ícone 🔑) para ativar os 3 modelos
   reais de sentimento no Agente 2.

3. **Calibração empírica de ε_min e corr_min com dados reais**: após coletar
   o corpus de notícias, rodar:
   ```bash
   python calibracao_epsilon_min.py --data sentimento_agregado_diario.csv
   python calibracao_corr_min.py --data sentimento_com_correlacao.csv
   ```

4. **Benchmark do Ibovespa no gráfico**: plugar retornos diários reais do BOVA11
   ou `^BVSP` via `yfinance` em `plotar_curva_patrimonio_e_drawdown()` para
   comparação visual real vs. sintética.
