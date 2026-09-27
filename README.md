# Manutenção Preditiva

Sistema de classificação que antecipa falhas em máquinas industriais a partir
dos sensores de uma linha de produção. É o **Projeto Avaliativo do Módulo 1**
do curso de Tecnologia em IA para Análise Preditiva (SCTEC / SENAI-SC).

O problema: uma fábrica não pode parar a linha quando uma máquina quebra no
meio do turno. Se a falha é previsível, a equipe troca a peça na manutenção
programada, com a linha ainda rodando.

## Resultado

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| KNN | 96,70% | 52,54% | 31,96% | 39,74% |
| **Decision Tree** | **98,28%** | **76,67%** | **71,13%** | **73,80%** |
| _baseline (nunca prever falha)_ | _96,60%_ | — | — | — |

A árvore de decisão é o veredito: acerta 98,28% e acerta 98% dos 97 casos de
falha do conjunto de teste.

**A acurácia engana aqui.** A baseline de "aproveitar sempre que não houve
falha" já marca 96,60% porque só 3,39% das máquinas falham. Um modelo com
96% de acurácia parece bom, mas o F1 de 39,74% do KNN mostra que ele está
errando 66 das 97 falhas. Por isso o F1 é a métrica do veredito, e não a
acurácia.

## O dataset

10.000 registros com 14 colunas. Cinco sensores numéricos
(`temperatura_ar_k`, `temperatura_processo_k`, `velocidade_rotacao_rpm`,
`torque_nm`, `desgaste_ferramenta_min`), o tipo de máquina (H/M/L), o alvo
binário `falha_maquina` e cinco colunas de subtipo de falha.

**Duas armadilhas do dataset que precisei tratar:**

**As colunas `falha_*` são o próprio resultado.** `falha_twf`, `falha_hdf`,
`falha_pwf`, `falha_osf` e `falha_rnf` descrevem o tipo do defeito e
praticamente reconstroem o alvo — `falha_hdf` tem correlação de 0,58 com
`falha_maquina`. Deixá-las entre as features seria o modelo lendo a própria
resposta: a acurácia no teste iria para perto de 100% sem o modelo ter
aprendido nada. Elas ficam fora do X e servem só para análise. Isso é
*data leakage*, e o enunciado não pede que elas sejam usadas.

**`udi` não descreve a máquina, descreve o momento da medição.** A
distribuição de falhas não é uniforme ao longo dele: entre `udi` 4000–5000 há
134 falhas, contra 17 a 29 nas outras nove faixas — 13,4% contra um ~2,2%
esperado. Um modelo que use essa coluna memoriza aquele trecho do dataset
em vez de aprender a condição da máquina. Sai, junto com `id_produto`
(um valor distinto por linha, portanto sem informação).

Sobram **5 features de sensor**, com sinal modesto: `torque_nm` (0,19),
`desgaste_ferramenta_min` (0,11), `temperatura_ar_k` (0,08),
`temperatura_processo_k` (0,04) e `velocidade_rotacao_rpm` (-0,04).

## Tratamento do desbalanceamento

339 falhas contra 9.661 registros normais — **razão de 28,5:1**.

A árvore recebe `class_weight='balanced'`, como pede o enunciado. O KNN não
tem como: no scikit-learn 1.9 o parâmetro `class_weight` sumiu do construtor
e o `fit` também não aceita `sample_weight` (as duas formas foram testadas e
dão `TypeError` nessa versão). Resolvi com a classe `KNNBalanceado`, que
aplica sobre a probabilidade a mesma conta que o `class_weight` faz por
dentro: baixar o limiar de decisão de 0,5 para a proporção da classe
majoritária. O `GridSearchCV` ainda pesquisa o limiar junto com o K, então
a compensação é ajustada, não fixa.

Nenhuma linha é duplicada em nenhum dos dois modelos — o dataset original
permanece inteiro.

## As 7 fases

1. **Análise exploratória** — dimensões, tipos, nulos, duplicados, `describe()`,
   e os 3 gráficos
2. **Limpeza** — 500 registros (5%) com sensor nulo foram descartados, e as
   colunas de leakage e identificadores foram removidas
3. **Feature engineering** — `torque_por_rotação`, `delta_temperatura`,
   faixa de desgaste em one-hot, tipo de máquina em one-hot
4. **Divisão 70/30 estratificada** e ajuste do desbalanceamento
5. **StandardScaler** — ajustado só no treino, reaproveitado no teste
6. **Ajuste de parâmetros** — K de 1 a 31 e `max_depth` de 1 a sem limite,
   com validação cruzada de 5 dobras
7. **Acurácia final e veredito** — KNN (K=13) contra Decision Tree

## Como executar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python maintenance_predictive.py
```

O script roda as 7 fases de ponta a ponta, imprime o relatório de cada
etapa e gera os resultados.

**Saídas geradas:**

```
outputs/
├── resultados.json         métricas, parâmetros e features
├── metricas_modelos.csv    tabela comparativa dos dois modelos
└── graficos/
    ├── 1_distribuicao_alvo.png
    ├── 2_sensores_por_classe.png
    ├── 3_matriz_correlacao.png
    ├── 4_comparacao_modelos.png
    └── 5_curvas_ajuste.png
```

## Tecnologias

Python 3 · pandas · numpy · matplotlib · seaborn · scikit-learn 1.9

## Estrutura do repositório

O histórico está em branches separadas, na ordem em que cada parte foi
implementada:

| Branch | Conteúdo |
|---|---|
| `feat/dataset-eda` | leitura do CSV e análise exploratória com os gráficos |
| `feat/limpeza-dados` | descarte de nulos e remoção de leakage |
| `feat/feature-engineering` | features derivadas |
| `feat/divisao-balanceamento` | split estratificado e desbalanceamento |
| `feat/escalonamento` | StandardScaler |
| `feat/ajuste-modelos` | GridSearchCV de KNN e árvore |
| `feat/avaliacao-modelos` | métricas, comparação e veredito |
| `docs/readme` | documentação |

## Melhorias possíveis

- **XGBoost ou LightGBM**: na maioria dos problemas de falha mecânica
  superam a árvore de decisão em recall da classe de falha
- **Sensores temporais**: o dataset é um recorte; com histórico de cada
  máquina dava para usar séries temporais e detectar degradação antes da
  falha, não só classificá-la
- **Limiar de decisão por custo**: aqui as duas classes têm o mesmo peso. Na
  fábrica não é assim — parar a linha custa muito mais que uma inspeção
  desnecessária, então o limiar deveria ponderar esse custo
- **Imputação em vez de descarte**: os 500 registros perdidos representam
  5% do dataset; com `SimpleImputer` mais volume, ao custo de estimar
  valores

## Autor

Eduardo Machado
