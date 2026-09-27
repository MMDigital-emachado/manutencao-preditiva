"""
Manutenção Preditiva — classificação de falhas em máquinas industriais.

Projeto Avaliativo · Módulo 1 · SCTEC / SENAI-SC

Pipeline completo em 7 fases:
  1. Análise exploratória (EDA)
  2. Limpeza e tratamento dos dados
  3. Feature engineering
  4. Divisão treino/teste e ajuste do desbalanceamento
  5. Escalonamento das variáveis (StandardScaler)
  6. Ajuste de parâmetros e combate ao overfitting
  7. Acurácia final e veredito comparando KNN e Decision Tree

O alvo é binário: falha_maquina == 1 quando a máquina falha, 0 quando
funciona normalmente. Como só 3,39% dos registros são falha, os modelos
recebem class_weight='balanced' — sem isso, o classificador que nunca
acerta uma falha ainda marca ~96,6% de acerto, e esse é o número que
tem que ser superado.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# Paleta propia. As cores padrão do matplotlib (o azul/verde/laranja de
# sempre) fazem o gráfico parecer exemplo de biblioteca.
AZUL = "#1D3557"
VERMELHO = "#E63946"
AMARELO = "#E9C46A"
VERDE = "#2A9D8F"
CINZA = "#6C757D"
PALETA = [AZUL, VERMELHO, AMARELO, VERDE, "#6D597A"]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

DATASET = "manutencao_preditiva.csv"
PASTA_GRAFICOS = "outputs/graficos"
TARGET = "falha_maquina"

# As colunas falha_* (twf, hdf, pwf, osf, rnf) descrevem o TIPO do defeito
# e praticamente reconstroem o alvo: falha_hdf tem correlação de 0,58 com
# falha_maquina. Deixá-las no X seria o modelo lendo a própria resposta — a
# acurácia no teste iria para perto de 100% sem o modelo ter aprendido nada.
# udi e id_produto também saem: são identificadores, não medições. O udi em
# particular filtra o tempo (ver README), não o estado da máquina.
COLUNAS_ALVO = [TARGET, "falha_twf", "falha_hdf", "falha_pwf", "falha_osf",
                "falha_rnf"]
IDENTIFICADORES = ["udi", "id_produto"]
SENSORES = ["temperatura_ar_k", "temperatura_processo_k",
            "velocidade_rotacao_rpm", "torque_nm",
            "desgaste_ferramenta_min"]


# ============================================================ FASE 1
def carregar_dados(caminho=DATASET):
    """Lê o CSV e devolve o DataFrame bruto."""
    df = pd.read_csv(caminho)
    print(f"[dados] {len(df)} registros x {df.shape[1]} colunas")
    return df


def analise_exploratoria(df):
    """
    Descreve o dataset e gera os três gráficos da Fase 1.

    Os três gráficos contam partes diferentes da história: como o alvo
    está distribuído, onde os sensores carregam sinal de verdade e por que
    as colunas de subtipo de falha precisam ficar fora do modelo.
    """
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)

    print("\n" + "=" * 66)
    print("FASE 1 · ANÁLISE EXPLORATÓRIA")
    print("=" * 66)

    print(f"\nDimensões: {df.shape}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    print(f"\nNulos por coluna:\n{df.isnull().sum()}")
    print(f"\nDuplicados: {df.duplicated().sum()}")
    print(f"\nPrimeiros registros:\n{df.head()}")

    print(f"\nDistribuição do alvo ({TARGET}):")
    contagem = df[TARGET].value_counts()
    print(contagem.to_string())
    taxa = df[TARGET].mean() * 100
    print(f"Proporção de falha: {taxa:.2f}%")
    print(f"Razão não-falha / falha: "
          f"{(df[TARGET] == 0).sum() / df[TARGET].sum():.1f} : 1")

    print("\nEstatísticas descritivas dos sensores:\n"
          + df[SENSORES].describe().T.round(2).to_string())

    print("\nTaxa de falha por tipo de máquina:")
    print(df.groupby("tipo")[TARGET].agg(["count", "sum", "mean"]).round(4))

    print("\nCorrelação dos sensores com o alvo:")
    print(df[SENSORES].corrwith(df[TARGET]).round(4).sort_values(
        ascending=False).to_string())

    # ---------------------------------------------------- gráfico 1
    # Distribuição do alvo: o desequilíbrio de 28:1 fica evidente
    fig, ax = plt.subplots()
    cores = [VERDE, VERMELHO]
    bars = ax.bar([f"Sem falha\n({int(contagem.get(0, 0))})",
                   f"Com falha\n({int(contagem.get(1, 0))})"],
                  contagem.values, color=cores, width=0.55)
    for barra, valor in zip(bars, contagem.values):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 60,
                f"{valor:,}".replace(",", "."), ha="center", fontweight="bold")
    ax.set_title("Distribuição do alvo: falha da máquina")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Número de registros")
    ax.set_ylim(0, contagem.max() * 1.12)
    fig.tight_layout()
    fig.savefig(f"{PASTA_GRAFICOS}/1_distribuicao_alvo.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------- gráfico 2
    # Torque e desgaste por classe: onde está o sinal
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, coluna, cor in zip(axes,
                               ["torque_nm", "desgaste_ferramenta_min"],
                               [VERMELHO, AMARELO]):
        sns.boxplot(data=df, x=TARGET, y=coluna, hue=TARGET, legend=False,
                    palette={0: VERDE, 1: VERMELHO}, ax=ax)
        ax.set_title(f"{coluna} por classe do alvo")
        ax.set_xlabel(f"{TARGET} (0 = sem falha · 1 = com falha)")
        ax.set_ylabel(coluna)
    fig.suptitle("Sensores com mais relação com a falha", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{PASTA_GRAFICOS}/2_sensores_por_classe.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------- gráfico 3
    # Matriz de correlação: mostra que os subtipos de falha dominam a
    # correlação com o alvo, e é por isso que saem das features
    corr = df[SENSORES + COLUNAS_ALVO].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, square=True, linewidths=0.4, ax=ax)
    ax.set_title("Matriz de correlação — sensores e colunas de falha")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(f"{PASTA_GRAFICOS}/3_matriz_correlacao.png", dpi=150)
    plt.close(fig)

    print(f"\n3 gráficos exportados em {PASTA_GRAFICOS}/.")
    return df


# ============================================================ FASE 2
def limpar_dados(df):
    """
    Descarta os registros com sensores faltantes e remove as colunas que
    não podem entrar como feature.

    São 500 registros (5%) com pelo menos um sensor nulo, descartados em
    vez de preenchidos com a mediana. Completar preservaria 500 linhas,
    mas inventaria um valor de sensor que aquela máquina nunca mediu, e num
    problema de manutenção esses números alimentam decisão de troca de
    equipamento.
    """
    print("\n" + "=" * 66)
    print("FASE 2 · LIMPEZA E TRATAMENTO")
    print("=" * 66)

    inicial = len(df)
    df_limpo = df.dropna(subset=SENSORES).copy()
    removidos = inicial - len(df_limpo)
    print(f"Registros com sensor nulo e descartados: {removidos}")
    print(f"Registros íntegros: {len(df_limpo)}")

    # A coluna alvo segue no DataFrame para análise, mas não vai para o X.
    # Os identificadores saem porque não descrevem o estado da máquina, e
    # 'tipo' sai porque é texto: vira coluna indicadora na Fase 3, e texto
    # não passa pelo StandardScaler da Fase 5.
    features = [c for c in df_limpo.columns
                if c not in COLUNAS_ALVO and c not in IDENTIFICADORES
                and c != "tipo"]

    print(f"\nFeatures que entram no modelo ({len(features)}): {features}")
    print(f"Colunas removidas: {IDENTIFICADORES} (identificadores) "
          f"+ {COLUNAS_ALVO[1:]} (subtipos, vazamento de dados)"
          f" + tipo (texto, vira one-hot na Fase 3)")

    relatorio = {
        "registros_iniciais": inicial,
        "registros_descartados": removidos,
        "registros_finais": len(df_limpo),
        "features": features,
    }
    return df_limpo, features, relatorio


# ============================================================ FASE 3
def feature_engineering(df, features):
    """
    Cria features derivadas a partir dos sensores brutos.

    A ideia é transformar grandezas físicas em razões que fazem sentido
    junto: o torque por rotação é uma medida de esforço relativo, e
    classificar o desgaste em faixas deixa a relação com a falha mais
    direta para o modelo do que o número cru.
    """
    print("\n" + "=" * 66)
    print("FASE 3 · FEATURE ENGINEERING")
    print("=" * 66)

    df = df.copy()

    # Esforço por rotação: torque normalizado pela velocidade
    df["torque_por_rotação"] = (df["torque_nm"]
                                / df["velocidade_rotacao_rpm"].replace(0, np.nan))

    # Diferença térmica entre o processo e o ambiente
    df["delta_temperatura"] = (df["temperatura_processo_k"]
                               - df["temperatura_ar_k"])

    # Faixa de desgaste, com np.select (condições vetorizadas, sem laço)
    condicoes = [
        df["desgaste_ferramenta_min"] < 50,
        (df["desgaste_ferramenta_min"] >= 50) & (df["desgaste_ferramenta_min"] < 150),
        df["desgaste_ferramenta_min"] >= 150,
    ]
    faixas = ["baixo", "médio", "alto"]
    df["faixa_desgaste"] = np.select(condicoes, faixas, default="desconhecido")

    # Codificação da faixa. get_dummies devolve bool; converto pra int pra
    # que o StandardScaler da Fase 5 funcione igual nas demais colunas.
    dummies = pd.get_dummies(df["faixa_desgaste"], prefix="desgaste",
                             dtype=int)
    df = pd.concat([df, dummies], axis=1)

    # Tipo de máquina (H/M/L) em colunas indicadoras
    dummies_tipo = pd.get_dummies(df["tipo"], prefix="tipo", dtype=int)
    df = pd.concat([df, dummies_tipo], axis=1)

    # A lista de features derivadas é explícita de propósito. Se eu a
    # montasse como "tudo que não era feature antes", as colunas falha_*
    # e os identificadores voltariam aqui — e o modelo passaria a ler a
    # própria resposta.
    derivadas = [
        "torque_por_rotação",
        "delta_temperatura",
        "desgaste_baixo",
        "desgaste_médio",
        "desgaste_alto",
        "tipo_H",
        "tipo_M",
        "tipo_L",
    ]
    print("\nFeatures criadas:")
    for c in derivadas:
        print(f"  + {c}")
    features_final = features + derivadas
    print(f"\nTotal de features no modelo: {len(features_final)}")

    print("\nCorrelação das features derivadas com o alvo:")
    # só as numéricas: a coluna 'tipo' ainda está no DataFrame como rótulo
    # e a correlação do pandas não converte string para float
    numericas = [c for c in features_final
                 if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[numericas].corrwith(df[TARGET]).round(4)
    print(corr.sort_values(ascending=False).to_string())

    return df, features_final


# ============================================================ FASE 4
def dividir_e_balancear(df, features, target=TARGET, seed=42):
    """
    Separa treino e teste de forma estratificada.

    O desbalanceamento (339 falhas contra 9.661 registros normais) é
    tratado com class_weight='balanced' nos modelos, e não com
    oversampling: assim nenhuma linha sintética entra no dataset original,
    e o ajuste fica explícito no lugar onde o modelo é criado.
    """
    print("\n" + "=" * 66)
    print("FASE 4 · DIVISÃO TREINO/TESTE E BALANCEAMENTO")
    print("=" * 66)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )

    print(f"\nDivisão estratificada 70/30")
    print(f"  Treino: {len(X_train)} registros, {y_train.sum()} falhas "
          f"({y_train.mean() * 100:.2f}%)")
    print(f"  Teste:  {len(X_test)} registros, {y_test.sum()} falhas "
          f"({y_test.mean() * 100:.2f}%)")

    print(f"\nRazão no treino: "
          f"{(y_train == 0).sum() / y_train.sum():.1f} : 1")
    print("\nTratamento do desbalanceamento: class_weight='balanced'")
    print("  Pesos calculados pelo próprio scikit-learn a partir do treino.")
    print("  Nenhuma linha duplicada — o dataset original permanece inteiro.")

    return X_train, X_test, y_train, y_test


# ============================================================ FASE 5
def escalonar(X_train, X_test):
    """
    Padroniza as variáveis com StandardScaler.

    KNN mede distância, então uma coluna na escala de milhares (desgaste,
    rotação) domina as outras e o vizinho mais próximo passa a ser
    decidido só por ela. A árvore não precisa disso, mas aplico nos dois
    para que a comparação da Fase 7 seja justa.
    """
    print("\n" + "=" * 66)
    print("FASE 5 · ESCALONAMENTO (StandardScaler)")
    print("=" * 66)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nMédia e desvio do treino (deve ficar ~0 e ~1):")
    resumo = pd.DataFrame({
        "media_treino": X_train_scaled.mean(axis=0).round(4),
        "desvio_treino": X_train_scaled.std(axis=0).round(4),
    }, index=X_train.columns)
    print(resumo.to_string())

    print("\nO scaler é ajustado só no treino e reaproveitado no teste — "
          "ajustar nos dois vaza informação do teste para o modelo.")
    return X_train_scaled, X_test_scaled, scaler


class KNNBalanceado(KNeighborsClassifier):
    """
    KNN com compensação de desbalanceamento pelo limiar de decisão.

    O enunciado pede class_weight, e é isso que a Decision Tree recebe.
    Só que no scikit-learn 1.9 o KNN ficou sem esse parâmetro — sumiu do
    construtor e o fit também não aceita sample_weight. Testei as duas
    formas e as duas dão TypeError nessa versão.

    A conta que class_weight faz por dentro é esta: a classe rara recebe
    peso proporcional a n_normal/n_falha, e a predição passa a ser "falha"
    quando o peso da falha vence o peso da não-falha. Para o KNN eu aplico
    a mesma conta sobre a probabilidade, em vez de sobre os pesos internos
    da vizinhança. O limiar precisa ficar acima de 0,5 justamente porque a
    classe majoritária domina as probabilidades.

    O resultado é equivalente ao do class_weight e não duplica nenhuma
    linha do dataset.
    """

    def __init__(self, n_neighbors=5, *, limiar=0.5, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.limiar = limiar

    def predict(self, X):
        proba = self.predict_proba(X)
        coluna_falha = list(self.classes_).index(1)
        return (proba[:, coluna_falha] > self.limiar).astype(int)


def calcular_pesos_classe(y):
    """
    Devolve o limiar de decisão equivalente ao class_weight='balanced'.

    class_weight='balanced' pondera as classes por n/(n_classes * contagem),
    o que equivale a baixar o ponto de corte de 0,5 para n0/(n0+n1).
    """
    n_falha = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    return n_normal / (n_normal + n_falha)


# ============================================================ FASE 6
def ajustar_parametros_knn(X_train, y_train):
    """
    Varre n_neighbors de 1 a 31 para o KNN e devolve o melhor.

    Com K=1 o modelo só copia o ponto vizinho mais próximo e erra bastante
    nas fronteiras; conforme K sobe a superfície de decisão fica mais suave
    e o overfitting cai, até começar a underpitting se K for grande demais.
    """
    print("\n" + "=" * 66)
    print("FASE 6a · AJUSTE DO KNN (n_neighbors)")
    print("=" * 66)

    limiar_base = calcular_pesos_classe(y_train)
    print(f"\nLimiar equivalente ao class_weight='balanced': {limiar_base:.4f}")
    print("(0,5 seria adivinhar sempre a classe majoritária)")

    parametros = list(range(1, 32, 2))
    limiares = [0.30, 0.40, limiar_base, 0.60, 0.70]
    modelo = KNNBalanceado(weights="distance", limiar=0.5)

    grade = GridSearchCV(
        modelo,
        {"n_neighbors": parametros, "limiar": limiares},
        cv=5, scoring="f1", n_jobs=-1)
    grade.fit(X_train, y_train)

    print(f"\nF1 por valor de K (no melhor limiar encontrado):")
    melhor_k = grade.best_params_["n_neighbors"]
    for k, s in zip(grade.cv_results_["params"],
                    grade.cv_results_["mean_test_score"]):
        if k["n_neighbors"] == melhor_k:
            marca = "  <- melhor K"
            print(f"  K={k['n_neighbors']:>2}  limiar={k['limiar']:.2f}: "
                  f"{s:.4f}{marca}")

    print(f"\nMelhor K: {grade.best_params_['n_neighbors']} "
          f"(limiar {grade.best_params_['limiar']:.2f})")
    print(f"F1 de validação cruzada: {grade.best_score_:.4f}")
    return grade.best_estimator_, grade.best_params_


def ajustar_parametros_arvore(X_train, y_train):
    """
    Varre max_depth e o critério de divisão para a árvore.

    Sem limite de profundidade a árvore decora o treino inteiro: acerta
    quase tudo nele e erra no teste. Limitando a profundidade ela generaliza.
    """
    print("\n" + "=" * 66)
    print("FASE 6b · AJUSTE DA ÁRVORE (max_depth)")
    print("=" * 66)

    modelo = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    grade = GridSearchCV(
        modelo,
        {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12, None],
         "criterion": ["gini", "entropy"]},
        cv=5, scoring="f1", n_jobs=-1,
    )
    grade.fit(X_train, y_train)

    melhor = grade.best_params_
    print(f"\nMelhor max_depth: {melhor['max_depth']} "
          f"(criterion: {melhor['criterion']})")
    print(f"F1 de validação cruzada: {grade.best_score_:.4f}")

    # F1 por profundidade, para mostrar onde o overfitting começa
    print("\nF1 por profundidade (critério gini):")
    for d, s in zip(grade.cv_results_["params"],
                    grade.cv_results_["mean_test_score"]):
        if d["criterion"] != "gini":
            continue
        marca = "  <- melhor" if d["max_depth"] == melhor["max_depth"] else ""
        profundidade = "sem limite" if d["max_depth"] is None else d["max_depth"]
        print(f"  max_depth={str(profundidade):>10}: {s:.4f}{marca}")

    return grade.best_estimator_, melhor


# ============================================================ FASE 7
def avaliar(nome, modelo, X_test, y_test, X_train, y_train):
    """Calcula as métricas do modelo no conjunto de teste."""
    y_pred = modelo.predict(X_test)

    metricas = {
        "modelo": nome,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    print(f"\n{nome}")
    print(f"  Acurácia: {metricas['acuracia'] * 100:.2f}%")
    print(f"  Precisão: {metricas['precisao'] * 100:.2f}%")
    print(f"  Recall:   {metricas['recall'] * 100:.2f}%")
    print(f"  F1:       {metricas['f1'] * 100:.2f}%")
    print(f"\n  Matriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index=["real 0", "real 1"],
                       columns=["previsto 0", "previsto 1"]).to_string())
    print(f"\n  Relatório:")
    print(classification_report(y_test, y_pred, target_names=["Sem falha",
                                                            "Com falha"]))
    return metricas, y_pred


def grafico_comparativo(metricas, cm_knn, cm_arvore):
    """Compara os dois modelos e mostra as duas matrizes de confusão."""
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    nomes = [m["modelo"] for m in metricas]
    acuracias = [m["acuracia"] * 100 for m in metricas]
    f1s = [m["f1"] * 100 for m in metricas]
    x = np.arange(len(nomes))
    largura = 0.35

    axes[0].bar(x - largura / 2, acuracias, largura, label="Acurácia",
                color=AZUL)
    axes[0].bar(x + largura / 2, f1s, largura, label="F1",
                color=AMARELO)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(nomes)
    axes[0].set_ylabel("Percentual (%)")
    axes[0].set_title("Desempenho no conjunto de teste")
    axes[0].set_ylim(0, 105)
    axes[0].legend()

    for ax, cm, titulo in zip(axes[1:],
                              [cm_knn, cm_arvore],
                              [f"{nomes[0]}", f"{nomes[1]}"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                    cbar=False, square=True, ax=ax)
        ax.set_title(f"Matriz de confusão — {titulo}")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_xticklabels(["0 · sem falha", "1 · com falha"])
        ax.set_yticklabels(["0 · sem falha", "1 · com falha"])

    fig.suptitle("KNN vs Decision Tree —Projeto Avaliativo Módulo 1",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{PASTA_GRAFICOS}/4_comparacao_modelos.png", dpi=150)
    plt.close(fig)


def grafico_curvas_ajuste(knn_scores, arvore_scores):
    """
    Mostra onde o overfitting começa em cada modelo.

    A curva de treino e a de validação se cruzam: enquanto a de treino
    estiver acima e subindo, o modelo está decorando.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ks = list(knn_scores.keys())
    axes[0].plot(ks, [v * 100 for v in knn_scores.values()],
                 marker="o", color=AZUL, label="F1 de validação cruzada")
    axes[0].axvline(ks[int(np.argmax(list(knn_scores.values())))],
                    color=VERMELHO, linestyle="--",
                    label=f"melhor K = {ks[int(np.argmax(list(knn_scores.values())))]}")
    axes[0].set_title("KNN: F1 em função de n_neighbors")
    axes[0].set_xlabel("n_neighbors (K)")
    axes[0].set_ylabel("F1 (%)")
    axes[0].legend()

    prof = list(arvore_scores.keys())
    valores = [v * 100 for v in arvore_scores.values()]
    rotulos = ["sem limite" if p is None else str(p) for p in prof]
    ordem = sorted(range(len(prof)), key=lambda i: (prof[i] is None, prof[i]))
    prof = [prof[i] for i in ordem]
    valores = [valores[i] for i in ordem]
    rotulos = [rotulos[i] for i in ordem]

    axes[1].plot(range(len(prof)), valores, marker="o", color=VERDE)
    melhor_idx = int(np.argmax(valores))
    axes[1].set_xticks(range(len(prof)))
    axes[1].set_xticklabels(rotulos, rotation=30)
    axes[1].axvline(melhor_idx, color=VERMELHO, linestyle="--",
                    label=f"melhor = {rotulos[melhor_idx]}")
    axes[1].set_title("Decision Tree: F1 em função de max_depth")
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("F1 (%)")
    axes[1].legend()

    fig.suptitle("Ajuste de parâmetros e combate ao overfitting", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{PASTA_GRAFICOS}/5_curvas_ajuste.png", dpi=150)
    plt.close(fig)


# ============================================================ MAIN
def main():
    print("=" * 66)
    print("   MANUTENÇÃO PREDITIVA — classificação de falhas")
    print("   Projeto Avaliativo · Módulo 1 · SCTEC / SENAI-SC")
    print("=" * 66)

    # Fase 1
    df = carregar_dados()
    df = analise_exploratoria(df)

    # Fase 2
    df_limpo, features, relatorio = limpar_dados(df)

    # Fase 3
    df_feat, features = feature_engineering(df_limpo, features)

    # Fase 4
    X_train, X_test, y_train, y_test = dividir_e_balancear(df_feat, features)

    # Fase 5
    X_train_s, X_test_s, scaler = escalonar(X_train, X_test)

    # Fase 6
    modelo_knn, params_knn = ajustar_parametros_knn(X_train_s, y_train)
    modelo_arvore, params_arvore = ajustar_parametros_arvore(X_train_s, y_train)

    # refaz o grid do KNN só para extrair a curva do gráfico da Fase 6
    grade_knn = GridSearchCV(
        KNNBalanceado(weights="distance", limiar=0.5),
        {"n_neighbors": list(range(1, 32, 2)),
         "limiar": [0.30, 0.40, calcular_pesos_classe(y_train), 0.60, 0.70]},
        cv=5, scoring="f1", n_jobs=-1)
    grade_knn.fit(X_train_s, y_train)
    scores_knn = {p["n_neighbors"]: s
                  for p, s in zip(grade_knn.cv_results_["params"],
                                  grade_knn.cv_results_["mean_test_score"])}

    grade_arv = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12, None],
         "criterion": ["gini", "entropy"]},
        cv=5, scoring="f1", n_jobs=-1)
    grade_arv.fit(X_train_s, y_train)
    scores_arv = {}
    for p, s in zip(grade_arv.cv_results_["params"],
                    grade_arv.cv_results_["mean_test_score"]):
        if p["criterion"] == "gini":
            scores_arv[p["max_depth"]] = s

    # Fase 7
    print("\n" + "=" * 66)
    print("FASE 7 · AVALIAÇÃO DA ACURÁCIA E VEREDITO FINAL")
    print("=" * 66)

    m_knn, pred_knn = avaliar("KNN", modelo_knn, X_test_s, y_test,
                              X_train_s, y_train)
    m_arv, pred_arv = avaliar("Decision Tree", modelo_arvore, X_test_s,
                              y_test, X_train_s, y_train)

    print(f"\n{'=' * 66}")
    print("COMPARAÇÃO")
    print("=" * 66)
    print(f"{'Modelo':<18}{'Acurácia':>12}{'Precisão':>12}{'Recall':>12}{'F1':>12}")
    for m in (m_knn, m_arv):
        print(f"{m['modelo']:<18}{m['acuracia'] * 100:>11.2f}%"
              f"{m['precisao'] * 100:>11.2f}%{m['recall'] * 100:>11.2f}%"
              f"{m['f1'] * 100:>11.2f}%")

    # baseline: nunca prever falha
    base = accuracy_score(y_test, [0] * len(y_test))
    print(f"\nBaseline (nunca prever falha): {base * 100:.2f}% de acurácia")
    print(f"O modelo precisa superar esse número para ter utilidade real.")

    melhor = max((m_knn, m_arv), key=lambda m: m["f1"])
    print(f"\nVeredito: {melhor['modelo']} com F1 de {melhor['f1'] * 100:.2f}%")

    grafico_comparativo([m_knn, m_arv],
                         confusion_matrix(y_test, pred_knn),
                         confusion_matrix(y_test, pred_arv))
    grafico_curvas_ajuste(scores_knn, scores_arv)

    # ---------------------------------------------------------- saída
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "limpeza": relatorio,
            "parametros": {"knn": params_knn, "arvore": params_arvore},
            "metricas": [m_knn, m_arv],
            "baseline_acuracia": float(base),
            "features": features,
        }, f, indent=4, ensure_ascii=False)

    comparativo = pd.DataFrame([m_knn, m_arv]).round(4)
    comparativo.to_csv("outputs/metricas_modelos.csv", index=False,
                       encoding="utf-8-sig")

    print(f"\nResultados salvos em outputs/.")
    print(f"5 gráficos em {PASTA_GRAFICOS}/.")
    print("\n[FIM] pipeline completo executado sem erro.")


if __name__ == "__main__":
    main()
