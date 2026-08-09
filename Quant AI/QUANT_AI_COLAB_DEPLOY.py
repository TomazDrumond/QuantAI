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
"""

REPO_URL = "https://github.com/TomazDrumond/QuantAI.git"
REPO_DIR = "/content/QuantAI"

CELULA_1 = f"""
import os, sys

os.chdir("/content")

if os.path.exists("{REPO_DIR}"):
    import shutil
    shutil.rmtree("{REPO_DIR}")

os.system("git clone {REPO_URL} {REPO_DIR}")

# Busca dinamicamente a pasta onde os arquivos .py dos agentes estão
pasta_agentes = None
for root, dirs, files in os.walk("{REPO_DIR}"):
    if "agente_0_universe_data.py" in files:
        pasta_agentes = root
        break

if pasta_agentes:
    os.chdir(pasta_agentes)
    if pasta_agentes not in sys.path:
        sys.path.insert(0, pasta_agentes)
    print(f"[OK] Pasta dos agentes localizada em: '{{pasta_agentes}}'")
else:
    print("[ERRO] agente_0_universe_data.py não foi encontrado após git clone.")

os.system("pip install -q cvxpy yfinance pandas_market_calendars "
          "anthropic openai google-generativeai matplotlib requests openpyxl scikit-learn")

print("[OK] Repositório clonado e dependências instaladas com sucesso.")
print("[ATENÇÃO] Reinicie a sessão agora (Ambiente de execução → Reiniciar sessão) antes de prosseguir!")
"""

CELULA_2 = """
import os
from google.colab import userdata

for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    try:
        os.environ[var] = userdata.get(var)
        print(f"[OK] {var} carregada.")
    except Exception:
        print(f"[AVISO] {var} não encontrada nos Secrets — fallback semântico ativo.")
"""

CELULA_3 = """
import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "buscar_precos_historicos.py",
     "--composicao", "composicao_pit_ibovespa.csv",
     "--inicio", "2018-01-01",
     "--fim",    "2026-01-01",
     "--out",    "precos_historicos.csv"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("[ERRO]", result.stderr[-1000:])
"""

CELULA_4 = """
import subprocess, sys
result = subprocess.run(
    [sys.executable, "processar_noticias_reais.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("[ERRO]", result.stderr[-1000:])
"""

CELULA_5 = """
import os, sys

# Garante que a pasta dos agentes está no sys.path e os.chdir
if not os.path.exists("agente_0_universe_data.py"):
    for root, dirs, files in os.walk("/content"):
        if "agente_0_universe_data.py" in files:
            os.chdir(root)
            if root not in sys.path:
                sys.path.insert(0, root)
            break

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
