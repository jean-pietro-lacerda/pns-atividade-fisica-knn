# Predição de Atividade Física no Lazer — PNS 2019

Projeto de Ciência de Dados que investiga a associação entre características socioeconômicas e demográficas e a prática suficiente de atividade física no lazer, utilizando microdados da **Pesquisa Nacional de Saúde (PNS 2019)**, realizada pelo IBGE.

O projeto é desenvolvido no contexto da disciplina **CIN0208 — Ciência de Dados**, do Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE).

## 1. Objetivo

Investigar como fatores como renda, escolaridade, idade, sexo, localização e situação de trabalho estão associados à prática suficiente de atividade física no lazer.

A proposta consiste em desenvolver um modelo de **classificação binária utilizando K-Nearest Neighbors (KNN)** e avaliar o impacto de diferentes técnicas de pré-processamento em seu desempenho.

A análise busca identificar padrões associados às desigualdades sociais, sem estabelecer relações causais.

## 2. Base de Dados

**Fonte:** Instituto Brasileiro de Geografia e Estatística (IBGE)  
**Pesquisa:** Pesquisa Nacional de Saúde — 2019  
**Formato original:** Arquivo TXT de largura fixa  
**Registros inicialmente extraídos:** 293.726  
**Registros após tratamento:** 88.456  
**Variáveis selecionadas:** 10 características preditoras e 1 variável-alvo

Foram considerados entrevistados com idade igual ou superior a 18 anos e informações suficientes para a construção da variável-alvo.

### Variáveis utilizadas

| Variável | Código PNS | Descrição |
|---|---|---|
| Idade | `C008` | Idade do entrevistado |
| Sexo | `C006` | Sexo registrado na pesquisa |
| Escolaridade | `VDD004A` | Maior nível de instrução alcançado |
| Renda per capita | `VDF003` | Rendimento domiciliar por pessoa |
| UF | `V0001` | Unidade da Federação |
| Situação de trabalho | `VDE001` e `VDE002` | Condição na força de trabalho e ocupação |
| Área de residência | `V0026` | Urbana ou rural |
| Raça/cor | `C009` | Cor ou raça autodeclarada |
| Estado civil | `C011` | Estado civil |
| Número de moradores | `V0022` | Quantidade de moradores do domicílio |

## 3. Construção da Variável-Alvo

A variável `ativo_lazer` representa a classificação da prática suficiente de atividade física no lazer.

| Valor | Classificação |
|---|---|
| `0` | Insuficientemente ativo |
| `1` | Suficientemente ativo |

Sua construção utiliza cinco informações do questionário:

| Código | Informação |
|---|---|
| `P034` | Prática de exercícios nos últimos três meses |
| `P035` | Frequência semanal |
| `P03701` | Horas por sessão |
| `P03702` | Minutos adicionais por sessão |
| `P036` | Modalidade de exercício praticada |

### Critérios adotados

O tempo semanal de exercício é calculado por:

**Minutos semanais = dias por semana × (60 × horas + minutos)**

A classificação considera:

- **Atividades moderadas:** pelo menos 150 minutos semanais.
- **Atividades vigorosas:** pelo menos 75 minutos semanais.
- **Ausência de prática ou frequência inferior a uma vez por semana:** classificação insuficientemente ativa.
- **Prática abaixo do limite correspondente:** classificação insuficientemente ativa.

As modalidades são classificadas conforme a regra operacional definida no projeto.

### Distribuição da variável-alvo

| Classificação | Registros | Percentual |
|---|---:|---:|
| Insuficientemente ativo (`0`) | 63.391 | 71,66% |
| Suficientemente ativo (`1`) | 25.065 | 28,34% |
| **Total** | **88.456** | **100%** |

A distribuição evidencia uma diferença entre as frequências das classes, aspecto relevante para os futuros experimentos de classificação.

## 4. Preparação e Tratamento dos Dados

O processamento dos microdados envolveu a extração das variáveis, interpretação dos códigos do dicionário, tratamento de informações ausentes e investigação de valores extremos.

### Valores ausentes

As perguntas sobre atividade física possuem dependências entre si. Por exemplo, entrevistados que declararam não praticar exercícios não precisavam responder às perguntas sobre duração e frequência.

Essas ausências foram identificadas como estruturais e tratadas de acordo com o funcionamento do questionário.

Também foram realizados os seguintes tratamentos:

- Reconstrução da situação de trabalho a partir de `VDE001` e `VDE002`.
- Remoção de 22 registros com renda per capita ausente.
- Remoção de 10 registros com raça/cor sem categoria válida.
- Restrição da análise a pessoas com 18 anos ou mais.

### Identificação de outliers

Foi utilizado o **Intervalo Interquartil (IQR)** para investigar valores estatisticamente atípicos.

Na análise da duração dos exercícios, foram removidos 48 registros com mais de seis horas diárias de atividade física, conforme critério de plausibilidade adotado pelo grupo.

Valores extremos de idade, renda e número de moradores foram preservados quando considerados plausíveis, evitando excluir informações potencialmente relevantes para a investigação das desigualdades sociais.

## 5. Análise Exploratória de Dados (EDA)

A EDA foi realizada para compreender a estrutura da base, identificar padrões e investigar as associações entre as características socioeconômicas e a variável-alvo.

### Técnicas aplicadas

| Técnica | Finalidade |
|---|---|
| Estatística descritiva | Resumir os valores e as distribuições |
| Análise univariada | Investigar cada característica individualmente |
| Análise bivariada | Comparar as características com a variável-alvo |
| Análise multivariada | Investigar idade, renda e atividade física simultaneamente |
| IQR | Identificar possíveis outliers |
| Histogramas | Visualizar distribuições numéricas |
| Gráficos de barras | Comparar percentuais entre grupos |
| Mapas de calor | Identificar padrões em cruzamentos de variáveis |
| Correlação de Spearman | Medir associações entre características numéricas |
| V de Cramér | Medir associações entre características categóricas |

## 6. Principais Resultados

### Escolaridade

A proporção de entrevistados classificados como suficientemente ativos aumentou conforme o nível de escolaridade.

| Escolaridade | Suficientemente ativos |
|---|---:|
| Sem instrução | 10,20% |
| Ensino médio completo | 33,97% |
| Ensino superior completo | 49,39% |

### Renda per capita

A análise identificou uma associação positiva entre renda e prática suficiente de atividade física no lazer.

| Grupo de renda | Suficientemente ativos |
|---|---:|
| Menor quartil | 21,28% |
| Maior quartil | 40,85% |

### Idade

A proporção de suficientemente ativos diminuiu progressivamente entre as faixas etárias.

| Faixa etária | Suficientemente ativos |
|---|---:|
| 18 a 29 anos | 39,40% |
| 30 a 44 anos | 32,57% |
| 45 a 59 anos | 25,58% |
| 60 a 74 anos | 21,24% |
| 75 anos ou mais | 11,71% |

### Área de residência

| Área | Suficientemente ativos |
|---|---:|
| Urbana | 31,63% |
| Rural | 17,27% |

### Análise multivariada

O cruzamento entre idade, renda e atividade física mostrou que, em todas as faixas etárias analisadas, os entrevistados dos grupos de maior renda apresentaram proporções superiores de prática suficiente.

Esse resultado indica que a associação com a renda permanece visível mesmo quando a amostra é dividida por faixa etária.

### Correlações e redundâncias

A correlação de Spearman identificou associações fracas a moderadas entre idade, renda e número de moradores.

Para as variáveis categóricas, foi utilizado o V de Cramér com correção de viés.

A maior associação encontrada ocorreu entre **escolaridade e área de residência**, com V de Cramér de aproximadamente **0,316**.

Não foram identificadas associações suficientemente fortes para justificar a remoção de características por redundância durante a EDA.

## 7. Estrutura do Projeto

O repositório está organizado para separar os dados originais, os dados intermediários, as análises e os experimentos.

| Diretório | Finalidade |
|---|---|
| `data/raw/` | Microdados originais e dicionário |
| `data/interim/` | Dados intermediários e base preparada para EDA |
| `data/processed/` | Dados preparados para etapas posteriores |
| `notebooks/` | Extração, EDA e experimentos |
| `src/` | Código reutilizável do projeto |

### Notebooks

**`01_extracao.ipynb`**

Extração dos microdados, interpretação do dicionário, limpeza e construção da variável-alvo.

**`02_eda.ipynb`**

Análise exploratória com estatísticas descritivas, distribuições, gráficos, análise multivariada e medidas de associação.

**Próxima etapa — Pré-processamento experimental**

Preparação de diferentes estratégias de transformação dos dados e avaliação posterior do KNN.

## 8. Tecnologias

- **Python:** linguagem utilizada no processamento e análise.
- **Pandas:** manipulação, tratamento e agrupamento dos dados.
- **NumPy:** operações numéricas e cálculos auxiliares.
- **Matplotlib:** construção de gráficos.
- **Seaborn:** visualizações estatísticas e mapas de calor.
- **SciPy:** testes e medidas estatísticas utilizadas na análise.
- **Jupyter Notebook:** desenvolvimento e documentação das análises.

A etapa de modelagem utilizará o ecossistema Scikit-learn.

## 9. Próximas Etapas

O desenvolvimento experimental seguirá os requisitos da disciplina, com foco no impacto das técnicas de pré-processamento sobre o desempenho do KNN.

Estão previstas:

1. Codificação das variáveis categóricas.
2. Normalização e padronização.
3. Seleção de atributos.
4. Redução de dimensionalidade com PCA.
5. Estratégias de balanceamento das classes.
6. Avaliação de pelo menos 150 configurações distintas de pré-processamento.
7. Validação cruzada estratificada com dez partições.
8. Comparação dos resultados por acurácia, F1-score, AUC-ROC e matriz de confusão.

O KNN será mantido com suas configurações padrão, conforme as exigências da disciplina. As transformações que dependem dos dados de treinamento serão ajustadas dentro de cada partição de treino, evitando vazamento de informações durante a validação cruzada.

## 10. Considerações Metodológicas

Os resultados da EDA descrevem a amostra tratada da PNS 2019, sem aplicação de ponderação amostral.

Consequentemente, os percentuais não devem ser interpretados como estimativas representativas de toda a população brasileira.

As relações identificadas são associações estatísticas e não estabelecem causalidade entre características socioeconômicas e prática de atividade física.

## 11. Status do Projeto

| Etapa | Situação |
|---|---|
| Definição do problema | Concluída |
| Seleção e extração das variáveis | Concluída |
| Construção da variável-alvo | Concluída |
| Tratamento inicial dos dados | Concluído |
| Análise Exploratória de Dados | Concluída |
| Pré-processamento experimental | Próxima etapa |
| Experimentos com KNN | Planejados |
| Comparação e discussão dos modelos | Pendente |

---

**Fonte dos dados:** Instituto Brasileiro de Geografia e Estatística (IBGE), Pesquisa Nacional de Saúde (PNS), 2019.

**Contexto acadêmico:** Centro de Informática (CIn) — Universidade Federal de Pernambuco (UFPE), disciplina CIN0208 — Ciência de Dados, 2026.2.
