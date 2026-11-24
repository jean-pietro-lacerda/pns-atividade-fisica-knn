# Revisão do relatório da PNS 2019

Aplicar os trechos abaixo em uma cópia editável de `projetoCD20262.pdf`. A numeração final de figuras, tabelas e páginas deve ser gerada após a diagramação. A seção de referências e as citações existentes devem ser mantidas no documento principal.

## Resumo

Este trabalho investiga a classificação da prática suficiente de atividade física no lazer a partir de dez preditores sociodemográficos da Pesquisa Nacional de Saúde de 2019. A base analítica contém 88.456 adultos, dos quais 28,3% pertencem à classe positiva. Foram consolidadas 150 combinações distintas de pré-processamento avaliadas com kNN padrão em validação cruzada estratificada de dez folds no conjunto de desenvolvimento, preservando-se 20% da amostra. Na referência com unidade da Federação e RobustScaler, a AUC-ROC média foi de 0,6281 e o F1 médio foi de 0,3590. A configuração de cinco regiões com Tomek Links atingiu a maior AUC-ROC média (0,6380), enquanto SMOTENC seguido de Tomek Links atingiu o maior F1 médio (0,4514), com AUC-ROC de 0,6228 e mais falsos positivos. Entre as 110 novas receitas, a maior AUC-ROC média foi de 0,6349. Os resultados mostram que as escolhas de pré-processamento alteraram de formas distintas a ordenação dos escores e o desempenho no ponto de decisão padrão. As estimativas referem-se ao conjunto de desenvolvimento e a hipótese de sobreposição entre classes requer investigação específica.

## Abstract

This study examines the classification of sufficient leisure-time physical activity using ten sociodemographic predictors from the 2019 Brazilian National Health Survey. The analytical dataset comprised 88,456 adults, with 28.3% in the positive class. We consolidated 150 distinct preprocessing combinations evaluated with default k-nearest neighbors using stratified ten-fold cross-validation on the development set, with 20% of the analytical sample reserved. The benchmark using states and RobustScaler achieved a mean ROC AUC of 0.6281 and a mean F1 of 0.3590. The five-region representation with Tomek Links achieved the highest mean ROC AUC (0.6380). SMOTENC followed by Tomek Links achieved the highest mean F1 (0.4514), with a mean ROC AUC of 0.6228 and more false positives. The highest mean ROC AUC among the 110 new configurations was 0.6349. Preprocessing choices affected score ranking and performance at the default decision point differently. The estimates describe the development set, and the class overlap hypothesis requires direct investigation.

## 3.1 Base de dados e análise exploratória

Complemento a inserir após a caracterização da base e os filtros. Manter as citações e os parágrafos originais.

A renda domiciliar per capita apresentou forte assimetria à direita. O diagnóstico pelo intervalo interquartil sinalizou 8.397 registros acima de R$ 3.419, preservados por representarem rendas plausíveis. As ausências condicionais das perguntas sobre exercício foram interpretadas conforme o fluxo do questionário. A preparação removeu 22 registros sem renda e dez sem categoria válida de raça/cor. As análises de redundância registraram correlação de Spearman de aproximadamente −0,36 entre idade e número de moradores e V de Cramér corrigido de 0,316 entre escolaridade e área de residência. Esses resultados embasaram a manutenção dos dez preditores na referência, seguida de ablações e seleção de colunas codificadas durante os experimentos. As proporções descrevem a amostra tratada e são não ponderadas.

## 3.2 Protocolo experimental e grade

Preservar os parágrafos que descrevem a divisão da base e os protocolos de SMOTENC e TSTR. Substituir a passagem que trata a execução da grade como pendente pelos parágrafos seguintes.

A grade consolidada reúne 40 receitas históricas e 110 receitas adicionais. Cada receita corresponde a uma combinação de pré-processamento com o classificador padrão e a uma validação cruzada estratificada de dez folds. Para as receitas históricas com múltiplos limiares, foi utilizada exclusivamente a decisão por maioria de três votos em cinco. Para as três escalas com validação repetida, utilizou-se apenas a primeira repetição. Peso por atributo designa multiplicação de colunas codificadas, com votação uniforme do kNN. Votação por distância, alterações de limiar, TSTR e outros classificadores permanecem na análise complementar.

As receitas adicionais variaram entre escalonamento com tratamento de extremos e discretização dos atributos numéricos por quantis. A seleção supervisionada reteve percentuais das colunas codificadas, com base no teste F ajustado no treino, e o PCA foi configurado por fração de variância ou desativado. Foram escolhidas 110 receitas dentre as candidatas com corte por IQR, logaritmo da renda ou discretização. A escolha usou uma amostra de 96 observações do treino do primeiro fold e comparou vetores de distâncias normalizados. Esse procedimento filtrou geometrias equivalentes na amostra e antecedeu a avaliação de desempenho. Em cada fold, os transformadores e o classificador foram ajustados no treino, e a validação recebeu as transformações correspondentes. Como o filtro usou dados de treino de um fold do próprio conjunto de desenvolvimento, a avaliação serve à comparação interna das receitas. Uma confirmação fora do desenvolvimento dependeria de avaliação posterior com as escolhas congeladas.

| Bloco de receitas | Configurações | Avaliações por fold |
|---|---:|---:|
| Histórico, com pré-processamentos de referência, ablações, reamostragem, PCA, Kernel PCA e ponderação de atributos | 40 | 400 |
| Novas, com corte de extremos por IQR | 34 | 340 |
| Novas, com transformação logarítmica da renda | 48 | 480 |
| Novas, com discretização em três faixas por quantis | 13 | 130 |
| Novas, com discretização em cinco faixas por quantis | 15 | 150 |
| Total | 150 | 1.500 |

## 3.3 Métricas e ambiente computacional

Acrescentar ao trecho sobre as métricas e manter as versões já documentadas para os diagnósticos.

A auditoria recalculou o F1 e a acurácia a partir das matrizes de confusão de todos os 1.500 resultados. Também recalculou as médias e os desvios-padrão amostrais das métricas por receita. Para as 110 receitas adicionais, a AUC-ROC foi recalculada a partir das frequências de zero a cinco votos positivos por classe real em cada fold, atribuindo-se meio crédito aos empates. Para as 40 receitas históricas, os valores de AUC foram conferidos contra os CSVs de origem. O manifesto da execução adicional registra scikit-learn 1.9.1, as duas sementes e as assinaturas SHA-256 da base analítica, do runner e do pré-processador.

## 4.4 Resultados da grade consolidada

A AUC-ROC média variou de 0,6042 a 0,6380 e o F1 médio, de 0,3403 a 0,4514. O maior valor de acurácia média foi de 70,02%, enquanto a regra que prevê a classe majoritária acerta 71,66% das observações de desenvolvimento e tem F1 igual a zero. Essa regra evidencia a limitação da acurácia isolada para a classe de interesse.

| Receita | AUC-ROC média ± DP | F1 médio ± DP | Acurácia média ± DP |
|---|---:|---:|---:|
| H03 · UF, RobustScaler e one-hot, referência original | 0,6281 ± 0,0061 | 0,3590 ± 0,0071 | 0,6956 ± 0,0039 |
| H15 · Cinco regiões, referência regional | 0,6304 ± 0,0071 | 0,3642 ± 0,0104 | 0,6965 ± 0,0032 |
| H17 · Cinco regiões e Tomek Links | 0,6380 ± 0,0074 | 0,4191 ± 0,0113 | 0,6797 ± 0,0050 |
| H06 · Sem número de moradores e raça/cor | 0,6359 ± 0,0085 | 0,3647 ± 0,0118 | 0,6974 ± 0,0040 |
| H19 · Cinco regiões, SMOTENC e Tomek Links | 0,6228 ± 0,0089 | 0,4514 ± 0,0076 | 0,5913 ± 0,0058 |
| G098 · Z-score, corte por IQR, seleção de 90% e PCA de 90% | 0,6349 ± 0,0077 | 0,3716 ± 0,0092 | 0,6981 ± 0,0056 |
| G320 · Três faixas, seleção de 25% e PCA de 99% | 0,6260 ± 0,0118 | 0,4044 ± 0,0203 | 0,6862 ± 0,0112 |

| Receita | VN médio | FP médio | FN médio | VP médio |
|---|---:|---:|---:|---:|
| H03 | 4.319,0 | 752,2 | 1.401,9 | 603,3 |
| H15 | 4.313,0 | 758,2 | 1.389,8 | 615,4 |
| H17 | 3.992,3 | 1.078,9 | 1.187,5 | 817,7 |
| H19 | 2.994,7 | 2.076,5 | 815,3 | 1.189,9 |
| G098 | 4.308,3 | 762,9 | 1.373,5 | 631,7 |
| G320 | 4.100,7 | 970,5 | 1.249,9 | 755,3 |

H17 superou a referência regional H15 em 0,0076 de AUC-ROC e 0,0549 de F1. Em média, classificou corretamente mais 202,3 positivos por fold, com mais 320,7 falsos positivos. H19 atingiu o maior F1, mas teve AUC-ROC inferior à de H15 e mais 1.318,3 falsos positivos por fold. Entre as novas receitas, G098 atingiu a maior AUC-ROC. G320 atingiu o maior F1, com desvio-padrão de 0,0203 nessa métrica, ante 0,0092 em G098. Os desvios-padrão descrevem a variação entre partições, cujos treinos se sobrepõem.

Em 12 pares de receitas adicionais que diferiam somente pela escala, RobustScaler superou MinMax em AUC nos 12 pares, com ganho mediano de 0,0054, e em F1 em 11 pares, com ganho mediano de 0,0076. Em 19 pares que diferiam somente entre corte de extremos por IQR e transformação logarítmica da renda, a diferença mediana de AUC foi −0,0001 ao passar de log para IQR. Em cinco pares, a passagem de três para cinco faixas elevou a AUC em quatro, com diferença mediana de 0,0044. Essas comparações caracterizam as combinações disponíveis. A grade filtrada tem cobertura desigual de níveis e interações.

H17 preserva a leitura direta das colunas codificadas e modifica a vizinhança pela remoção de exemplos. G098 e G320 tomam decisões em componentes que combinam colunas originais, o que dificulta atribuir influência a um atributo individual. Os tempos médios por receita na rodada adicional variaram de 0,31 a 3,30 segundos por fold, com mediana de 1,99 segundo. A medição inclui preparação quando indisponível em cache e depende da ordem e do paralelismo. As medições das baterias históricas abrangem etapas diferentes e não permitem ordenar o custo integral das abordagens.

Inserir a figura `auc_f1_grade.png` com legenda indicando um ponto por receita e médias dos dez folds. A figura `roc_receitas_novas.png` pode acompanhar a comparação entre as duas líderes novas. Sua legenda deve indicar que as AUCs mostradas são calculadas sobre escores reunidos fora do treino. Os dois rankings completos e as matrizes médias por receita estão nos CSVs anexos.

## 5 Conclusões

Manter a síntese dos diagnósticos e substituir os parágrafos que deixam a escolha global condicionada à execução da grade.

Na validação cruzada das 150 receitas, a representação por cinco regiões com Tomek Links apresentou a maior AUC-ROC média, de 0,6380, e alcançou F1 médio de 0,4191. Essa configuração é a candidata quando se prioriza a ordenação dos escores. A combinação SMOTENC com Tomek Links obteve o maior F1 médio, de 0,4514, com AUC inferior e mais falsos positivos. Entre as novas receitas, G098 apresentou AUC média de 0,6349 e manteve a acurácia próxima à referência com UF, com menor F1 que H17. A comparação entre candidatas depende da importância atribuída a falsos positivos, à estabilidade e à interpretabilidade.

Os ganhos foram modestos em AUC na amostra estudada. As conclusões decorrem da validação interna e não estabelecem um limite de informação dos atributos. A reserva de 17.692 observações permite examinar desempenho fora do desenvolvimento em uma avaliação futura, após congelar as escolhas. A análise também não permite estimar prevalências populacionais, pois o desenho amostral da PNS não foi incorporado.

## Apêndice de reprodutibilidade

O pacote de evidências contém o inventário das receitas, os 1.500 registros por fold, o resumo por receita, as frequências de votos da rodada adicional, os arquivos históricos de origem e o código com manifesto. As matrizes por fold mostram de 7.076 a 7.077 observações e somam 50.712 negativos e 20.052 positivos por receita. Os hashes dos CSVs de origem, do runner e do módulo de pré-processamento foram conferidos. Nove arquivos históricos também correspondem aos publicados na branch de Pedro após normalização de codificação e quebras de linha. O CSV das três escalas integra a evidência local entregue pelo grupo. Para H04 a H08, as células das matrizes foram conferidas a partir dos totais por classe e das taxas de acerto de cada classe.

O protocolo e as contagens por fold são consistentes entre as receitas. A identidade observação a observação das partições e a assinatura do Parquet registrado no manifesto dependem do acesso à base exata ou aos índices da execução. A auditoria da rodada adicional pôde recuperar a AUC por fold com as frequências dos votos. Nos casos históricos, a conferência de AUC usou os CSVs de origem. A busca por duplicidade cobriu as receitas documentadas e a igualdade dos resultados por fold, complementada pela filtragem geométrica amostral do runner para as receitas novas.
