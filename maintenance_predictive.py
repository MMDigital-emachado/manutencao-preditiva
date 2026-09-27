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


