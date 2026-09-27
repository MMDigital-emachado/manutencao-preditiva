# Kanban — Projeto Avaliativo Módulo 1

## Backlog

- Testar com XGBoost / LightGBM para comparar com a árvore de decisão
- Usar séries temporais dos sensores em vez de um recorte isolado
- Ajustar o limiar de decisão pelo custo real de parar a linha
- Validar o resultado com a engenharia de manutenção da fábrica

## A Fazer

- Gravar o vídeo de até 7 minutos para o Google Drive
- Enviar os links do repositório e do vídeo na tarefa do AVA

## Em Andamento

- Escrever o README final com os resultados e as decisões técnicas

## Concluído

- **Fase 1 — Análise exploratória:** dimensões (10.000 × 14), tipos, nulos,
  duplicados, `describe()` e os 3 gráficos da EDA
- **Fase 2 — Limpeza:** 500 registros com sensor nulo descartados; removidas
  as colunas `falha_*` (vazamento de dados) e os identificadores `udi` e
  `id_produto`
- **Fase 3 — Feature engineering:** `torque_por_rotação`, `delta_temperatura`,
  faixa de desgaste e tipo de máquina em one-hot
- **Fase 4 — Divisão e desbalanceamento:** split 70/30 estratificado;
  `class_weight='balanced'` na árvore e limiar ajustado no KNN
- **Fase 5 — Escalonamento:** `StandardScaler` ajustado só no treino
- **Fase 6 — Ajuste de parâmetros:** K de 1 a 31 e `max_depth` de 1 a
  sem limite, com validação cruzada de 5 dobras
- **Fase 7 — Veredito:** Decision Tree com 98,28% de acurácia e F1 de
  73,80%, contra 96,70% e 39,74% do KNN
- Documentação dos gráficos de comparação e das curvas de ajuste
