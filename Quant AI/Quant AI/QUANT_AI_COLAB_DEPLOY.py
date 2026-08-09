"""
QUANT_AI_COLAB_DEPLOY.py
========================
Arquivo único de deploy do pipeline Quant AI para o Google Colab.

INSTRUÇÕES DE USO NO COLAB
---------------------------
1. Faça upload de TODOS os arquivos do projeto para o seu repositório GitHub
   (ou Drive) e ajuste o REPO_URL abaixo.
2. Cole este arquivo inteiro numa célula do Colab e execute.
3. As variáveis de ambiente das chaves de API devem ser setadas via
   Colab Secrets (ícone de chave 🔑 na barra lateral) antes de rodar.

ESTRUTURA DE CÉLULAS (separe por `# %%` no VS Code ou use o Colab UI)
----------------------------------------------------------------------
"""

# ============================================================
# CÉLULA 1 — Clona repositório e instala dependências
# ============================================================
# Cole e execute esta célula primeiro. Sempre reinicie a sessão
# (Ambiente de execução → Reiniciar sessão) após rodar esta célula,
# antes de prosseguir para as próximas.

REPO_URL = "https://github.com/TomazDrumond/QuantAI.git"
REPO_DIR = "/content/QuantAI"
PROJ_DIR = '/content/QuantAI/Quant AI'

CELULA_1 = f"""
import os, sys

# Garante que o processo nunca está dentro da pasta que vamos apagar
os.chdir("/content")

if os.path.exists("{REPO_DIR}"):
    import shutil
    shutil.rmtree("{REPO_DIR}")

os.system("git clone {REPO_URL} {REPO_DIR}")
os.chdir("{PROJ_DIR}")

if "{PROJ_DIR}" not in sys.path:
    sys.path.insert(0, "{PROJ_DIR}")

os.system("pip install -q cvxpy yfinance pandas_market_calendars "
          "anthropic openai google-generativeai matplotlib requests openpyxl")

print("[OK] Repositório clonado e dependências instaladas.")
print("[ATENÇÃO] Reinicie a sessão agora antes de prosseguir!")
"""

# ============================================================
# CÉLULA 2 — Configuração de chaves de API (via Colab Secrets)
# ============================================================
CELULA_2 = """
import os
from google.colab import userdata

# Lê chaves cadastradas nos Secrets do Colab (ícone 🔑 na barra lateral)
# Se uma chave não estiver configurada, o pipeline usa o fallback semântico
for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    try:
        os.environ[var] = userdata.get(var)
        print(f"[OK] {var} carregada.")
    except Exception:
        print(f"[AVISO] {var} não encontrada nos Secrets — fallback semântico ativo.")
"""

# ============================================================
# CÉLULA 3 — Coleta de preços históricos (Agentes 0 e 1)
# ============================================================
CELULA_3 = """
import subprocess, sys

# Agente 0: composição PIT já gerada — composicao_pit_ibovespa.csv
# Agente 1: busca preços para todos os tickers PIT (2018–2026)
result = subprocess.run(
    [sys.executable, "buscar_precos_historicos.py",
     "--composicao", "composicao_pit_ibovespa.csv",
     "--inicio", "2018-01-01",
     "--fim",    "2026-01-01",
     "--out",    "precos_historicos.csv"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])  # últimas 3000 chars do log
if result.returncode != 0:
    print("[ERRO]", result.stderr[-1000:])
"""

# ============================================================
# CÉLULA 4 — Sentimento real / demo (Agentes 2 e 3)
# ============================================================
CELULA_4 = """
# Modo demo: usa o fallback semântico interno para gerar
# sentimento_agregado_diario.csv sem precisar de APIs.
# Substitua por notícias reais quando disponíveis.
import subprocess, sys
result = subprocess.run(
    [sys.executable, "processar_noticias_reais.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("[ERRO]", result.stderr[-1000:])
"""

# ============================================================
# CÉLULA 5 — Backtest Walk-Forward completo (Agentes 4–8)
# ============================================================
CELULA_5 = """
from pipeline_quant_ai_colab import rodar_backtest_walkforward

rodar_backtest_walkforward(
    caminho_pit    = "composicao_pit_ibovespa.csv",
    caminho_precos = "precos_historicos.csv",
    delta          = 3.0,
    w_max          = 0.20,
    valor_inicial  = 100_000.0,
    razao_precisao_alvo = 3.0,
)
"""

# ============================================================
# CÉLULA 6 — Exibe gráficos no Colab
# ============================================================
CELULA_6 = """
from IPython.display import Image, display
import os

pasta = "relatorios_graficos"
for nome in ["curva_patrimonio_e_drawdown.png", "alocacao_pesos_historica.png"]:
    caminho = os.path.join(pasta, nome)
    if os.path.exists(caminho):
        print(f"--- {nome} ---")
        display(Image(caminho))
    else:
        print(f"[AVISO] {caminho} não encontrado — verifique se o backtest gerou os gráficos.")
"""

# ============================================================
# Ponto de entrada local (fora do Colab)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  QUANT AI — GUIA DE DEPLOY PARA O GOOGLE COLAB")
    print("=" * 60)
    print("\nCopie e execute as células abaixo, NESTA ORDEM, no Colab:\n")

    for idx, celula in enumerate([CELULA_1, CELULA_2, CELULA_3, CELULA_4, CELULA_5, CELULA_6], 1):
        print(f"\n{'='*55}")
        print(f"  CÉLULA {idx}")
        print(f"{'='*55}")
        print(celula.strip())

    print("\n[PRONTO] Cole as células acima no Google Colab e execute em ordem.")
    print("Lembre-se: reinicie a sessão após a Célula 1 e novamente após a Célula 2.")
