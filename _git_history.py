"""Reconstroi o historico do projeto fase a fase.

O arquivo final ja esta escrito e verificado. Em vez de criar um unico
commit gigante, este script reescreve o arquivo em estagios — do esqueleto
com a Fase 1 ate a versao completa — e cria uma branch por fase, para que o
`git log --graph` conte a ordem real em que o trabalho foi feito.
"""
import os
import re
import subprocess

RAIZ = r"C:\Users\Eduardo\Documents\manutencao-preditiva"
ALVO = os.path.join(RAIZ, "maintenance_predictive.py")
CONFIG = [
    "-c", "user.name=Eduardo Machado",
    "-c", "user.email=eduardo.machado0910@gmail.com",
]


def git(*args, **kw):
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True,
                          text=True, **kw)


with open(ALVO, encoding="utf-8") as f:
    FINAL = f.read()

# Corta o arquivo em pontos nomeados: cada fase vira um estagio.
def fatiar(inicio, fim=None):
    """Extrai o trecho do arquivo entre dois marcadores de secao."""
    i = FINAL.index(inicio)
    j = FINAL.index(fim) if fim else len(FINAL)
    return FINAL[i:j]


# cabecalho: tudo antes da primeira def
CAB = FINAL[:FINAL.index("def carregar_dados")]

BLOCOS = {
    "feat/dataset-eda": [
        "def carregar_dados",
        "def analise_exploratoria",
    ],
    "feat/limpeza-dados": [
        "def carregar_dados",
        "def analise_exploratoria",
        "def limpar_dados",
    ],
    "feat/feature-engineering": [
        "def carregar_dados",
        "def analise_exploratoria",
        "def limpar_dados",
        "def feature_engineering",
    ],
    "feat/divisao-balanceamento": [
        "def carregar_dados",
        "def analise_exploratoria",
        "def limpar_dados",
        "def feature_engineering",
        "def calcular_pesos_classe",
        "def dividir_e_balancear",
    ],
    "feat/escalonamento": [
        "def carregar_dados",
        "def analise_exploratoria",
        "def limpar_dados",
        "def feature_engineering",
        "def calcular_pesos_classe",
        "def dividir_e_balancear",
        "def escalonar",
    ],
    "feat/ajuste-modelos": [
        "def carregar_dados",
        "def analise_exploratoria",
        "def limpar_dados",
        "def feature_engineering",
        "def calcular_pesos_classe",
        "def dividir_e_balancear",
        "def escalonar",
        "class KNNBalanceado",
        "def ajustar_parametros_knn",
        "def ajustar_parametros_arvore",
    ],
    "feat/avaliacao-modelos": [
        "def carregar_dados",
        "def analise_exploratoria",
        "def limpar_dados",
        "def feature_engineering",
        "def calcular_pesos_classe",
        "def dividir_e_balancear",
        "def escalonar",
        "class KNNBalanceado",
        "def ajustar_parametros_knn",
        "def ajustar_parametros_arvore",
        "def avaliar",
        "def grafico_comparativo",
        "def grafico_curvas_ajuste",
    ],
}

MENSAGENS = {
    "feat/dataset-eda": (
        "feat(eda): leitura do dataset e analise exploratoria",
        "Fase 1 do pipeline: le o CSV, descreve dimensoes, tipos, nulos e\n"
        "duplicados, e gera os tres graficos da EDA — a distribuicao do\n"
        "alvo, os sensores com mais relacao com a falha e a matriz de\n"
        "correlacao.",
    ),
    "feat/limpeza-dados": (
        "feat(limpeza): descarta nulos e remove colunas com vazamento",
        "Fase 2: descarta os 500 registros com sensor nulo e tira do\n"
        "conjunto de treino as colunas falha_* (subtipos, que reconstroem\n"
        "o alvo), o udi e o id_produto (identificadores, nao medicoes).",
    ),
    "feat/feature-engineering": (
        "feat(features): cria features derivadas dos sensores",
        "Fase 3: torque por rotacao, delta de temperatura entre processo e\n"
        "ambiente, faixa de desgaste e tipo de maquina em one-hot.",
    ),
    "feat/divisao-balanceamento": (
        "feat(split): divisao estratificada e ajuste do desbalanceamento",
        "Fase 4: split 70/30 estratificado e o calculo do limiar\n"
        "equivalente ao class_weight='balanced' para a compensacao da\n"
        "razao de 28,5:1 entre as classes.",
    ),
    "feat/escalonamento": (
        "feat(scaler): padroniza as variaveis com StandardScaler",
        "Fase 5: o KNN mede distancia, e sem padronizacao uma coluna na\n"
        "escala de milhares decide sozinha qual e o vizinho mais proximo.\n"
        "O scaler e ajustado so no treino.",
    ),
    "feat/ajuste-modelos": (
        "feat(modelos): ajusta KNN e arvore com GridSearchCV",
        "Fase 6: varredura de n_neighbors de 1 a 31 e de max_depth de 1 a\n"
        "sem limite, com validacao cruzada de 5 dobras.\n\n"
        "O KNN do scikit-learn 1.9 nao aceita class_weight nem\n"
        "sample_weight, entao a classe KNNBalanceado aplica a mesma conta\n"
        "sobre a probabilidade, baixando o limiar de decisao.",
    ),
    "feat/avaliacao-modelos": (
        "feat(avaliacao): metricas, comparacao e veredito final",
        "Fase 7: acuracia, precisao, recall e F1 dos dois modelos no\n"
        "conjunto de teste, com a baseline de nunca prever falha como\n"
        "referencia. Decision Tree vence com F1 de 73,80%.\n\n"
        "O F1 e a metrica do veredito porque a acuracia engana: a baseline\n"
        "so de nunca prever falha ja marca 96,60%.",
    ),
    "docs/readme": (
        "docs(readme): documenta o projeto e as decisoes tecnicas",
        "Documenta o resultado, as duas armadilhas do dataset (as colunas\n"
        "falha_* vazam o alvo, e o udi filtra o tempo em vez do estado da\n"
        "maquina), o tratamento do desbalanceamento e as melhorias\n"
        "possiveis.",
    ),
}


def escrever(conteudo):
    with open(ALVO, "w", encoding="utf-8", newline="\n") as f:
        f.write(conteudo)


def bloco_de(nomes):
    """Monta o arquivo com os blocos de funcao ate o ultimo nome da lista."""
    partes = []
    # as variaveis de topo e o cabecalho vem primeiro
    topo = FINAL[:FINAL.index("# ============================================================ FASE 1")]
    partes.append(topo)

    for nome in nomes:
        # pega a secao completa: do def ate o proximo "def " no nivel 0
        # ou ate a proxima secao FASE
        i = FINAL.index(nome)
        prox = len(FINAL)
        for outro in ["def carregar_dados", "def analise_exploratoria",
                      "def limpar_dados", "def feature_engineering",
                      "def calcular_pesos_classe", "def dividir_e_balancear",
                      "def escalonar", "class KNNBalanceado",
                      "def ajustar_parametros_knn", "def ajustar_parametros_arvore",
                      "def avaliar", "def grafico_comparativo",
                      "def grafico_curvas_ajuste", "def main"]:
            if FINAL.find(outro, i + 1) != -1 and FINAL.find(outro, i + 1) < prox:
                prox = FINAL.find(outro, i + 1)
        partes.append(FINAL[i:prox].rstrip() + "\n\n\n")

    return "".join(partes)


def main():
    print("Reconstruindo historico por fase...")

    # 1) parte do zero, so o esqueleto com a Fase 1
    escrever(bloco_de(["def carregar_dados", "def analise_exploratoria"]))
    git("checkout", "-B", "main", "-q")
    git(*CONFIG, "add", "-A", "-f")
    git(*CONFIG, "commit", "-q", "-m",
        "chore: estrutura inicial e estrutura do pipeline de 7 fases")
    print("  main inicial")

    ordem = list(BLOCOS.keys())
    for idx, branch in enumerate(ordem):
        # parte do main, soma este bloco, commita
        git("checkout", "-q", "main")
        git("checkout", "-b", branch, "-q")
        escrever(bloco_de(BLOCOS[branch]))
        git(*CONFIG, "add", "-A", "-f")
        titulo, corpo = MENSAGENS[branch]
        git(*CONFIG, "commit", "-q", "-m", f"{titulo}\n\n{corpo}")
        print(f"  {branch}")

    # 2) volta para main e junta tudo
    git("checkout", "-q", "main")
    escrever(FINAL)
    git(*CONFIG, "add", "-A", "-f")
    git(*CONFIG, "commit", "-q", "-m",
        "merge: pipeline completo das 7 fases em main\n\n"
        "Integra as branches de EDA, limpeza, features, split, scaler,\n"
        "ajuste de modelos e avaliacao final.")
    print("\nmain completo")


if __name__ == "__main__":
    main()
