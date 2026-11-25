# Evidências e revisão da grade PNS 2019

Conferência da grade do projeto CIN0208, semestre 2026.2. O pacote do grupo reúne 150 receitas documentadas com dez avaliações por receita. As 110 novas incluem frequências de votos para a conferência independente da AUC. As 40 históricas foram rastreadas aos CSVs de origem e aos scripts de preparação, ajuste e avaliação.

| Arquivo | Conteúdo |
|---|---|
| `ranking_auc.csv` | 150 receitas ordenadas por AUC-ROC média |
| `ranking_f1.csv` | 150 receitas ordenadas por F1 médio |
| `auc_f1_grade.png` | Comparação das métricas médias das 150 receitas |
| `roc_receitas_novas.png` | Curvas ROC OOF de G098 e G320 reconstruídas pelas frequências de votos |
| `Revisao_relatorio.md` | Texto, tabelas e indicações de revisão para o relatório principal |
| `Revisao_apresentacao.md` | Conteúdo por slide e notas de fala mantendo a divisão do grupo |
| `auditar_grade.py` | Verificações reproduzíveis no ZIP de evidências recebido do Pedro |
| `gerar_resultados.py` | Exportação dos dois rankings e dos dois gráficos |

## Conferências

- 150 IDs únicos, 150 receitas documentadas, 1.500 combinações únicas de ID e fold, dez folds completos por ID.
- 110 receitas novas distintas pelas chaves do manifesto, com exclusão de geometrias equivalentes em amostra do treino do primeiro fold. Nenhuma dupla de receitas produziu a mesma sequência de métricas e matrizes nos dez folds.
- Tripla concordância nas receitas novas entre CSV consolidado, CSV da execução e frequências de votos. Acurácia e F1 foram recalculados nas 1.500 linhas. A AUC foi recalculada nas 1.100 linhas novas e cruzada com o CSV de origem nas 400 históricas.
- Médias, desvios-padrão amostrais e células médias conferidos nas 150 linhas agregadas. Os rankings trazem as quatro células médias, tempos e dimensões quando registrados na rodada adicional.
- As 40 receitas históricas usam somente a regra de três votos em cinco e, quando há repetições, somente a primeira. Pesos de atributos multiplicam colunas antes da classificação. O script de treino histórico e o módulo correspondente instanciam o kNN padrão. Nos scripts com reamostragem, as operações de ajuste e a reamostragem ocorrem nas partições de treino.
- Nove CSVs históricos coincidem em conteúdo com o commit `1cf3b8abf4fede42085567801225b85494ce8c8c` da branch `feature/treinamento-pedro`. O CSV das três escalas consta no pacote enviado. Os dez hashes constantes do inventário histórico e o hash do CSV da grade adicional conferem. As assinaturas dos dois arquivos de código também conferem com o manifesto.

## Limites da conferência

As contagens de classe por fold são idênticas entre as configurações e os scripts registram a mesma estratificação e semente de divisão. O ZIP não traz o Parquet ou os índices por observação, de modo que a identidade das partições no nível das linhas e a assinatura da base não puderam ser recalculadas. O desenho dos scripts preserva o teste reservado na seleção dos modelos. As frequências de votos da rodada adicional permitem recalcular a AUC sem dados individuais. Para a histórica, a AUC depende das saídas de origem.

Os rankings comparam as receitas no desenvolvimento. O filtro de geometrias foi calculado no treino do primeiro fold, que contém observações usadas na validação de outros folds. O filtro usou a geometria, sem as métricas de avaliação, mas o conjunto de receitas ainda depende do desenvolvimento. A confirmação de desempenho em dados separados exige avaliação posterior com escolhas congeladas.

O documento da branch de Pedro apresenta referências diferentes das do PDF principal. As revisões são trechos para aplicar em cópias das fontes correspondentes. As referências bibliográficas e as citações originais devem ser conservadas integralmente.

## Reprodução da auditoria

```powershell
python auditar_grade.py PNS_2019_evidencias_150.zip --json auditoria.json
python gerar_resultados.py auditoria.json PNS_2019_evidencias_150.zip resultados
```

O primeiro comando precisa apenas da biblioteca padrão do Python. O segundo usa Matplotlib. Para versionar os materiais, adicionar os dois rankings, as duas figuras, as revisões e os scripts em uma pasta de evidências do repositório. Manter o ZIP original disponível para a reprodução da auditoria.
