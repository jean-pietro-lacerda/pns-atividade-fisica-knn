# Guia de pré-processamento

## 1. Visão geral

O módulo `src/preprocessamento.py` prepara os dados de `data/interim/pns_base_eda.parquet` para classificação com kNN. A entrada contém dez preditores e o alvo binário `ativo_lazer`:

- Numéricos: `idade`, `renda_per_capita` e `numero_moradores`.
- Categóricos: `sexo`, `escolaridade`, `uf`, `situacao_trabalho`, `area_residencia`, `raca_cor` e `estado_civil`.

O pipeline aplica imputação, tratamento de extremos e escala ou discretização, codificação das categorias, seleção de características e PCA, conforme a configuração escolhida. O kNN usa os parâmetros padrão do scikit-learn.

Para evitar vazamento de dados, todas as etapas que aprendem com os dados devem ser ajustadas apenas no treino de cada partição. Isso inclui imputação, limites de extremos, faixas de discretização, codificação, escala, seleção e PCA. Não forneça uma matriz previamente transformada à validação cruzada.

## 2. Como utilizar o pré-processamento

Execute o exemplo a partir da raiz do projeto:

```python
import pandas as pd
from sklearn.model_selection import cross_validate
from src.preprocessamento import (
    separar_xy,
    construir_pipeline,
    configuracoes_experimentais,
    validacao_cruzada,
)

base = pd.read_parquet("data/interim/pns_base_eda.parquet")
X, y = separar_xy(base)
if y.value_counts().min() < 10:
    raise ValueError("São necessárias ao menos dez observações por classe.")

receita = configuracoes_experimentais()[0]
resultado = cross_validate(
    construir_pipeline(**receita),
    X,
    y,
    cv=validacao_cruzada(),
    scoring=["accuracy", "f1", "roc_auc"],
    error_score="raise",
)
```

O exemplo avalia uma configuração. F1 considera a classe 1 como positiva; AUC-ROC utiliza os escores do estimador. Para calcular a matriz de confusão, use previsões fora da amostra.

As funções públicas do módulo são:

| Função | Uso |
| --- | --- |
| `separar_xy(base)` | Retorna cópias dos dez preditores (`X`) e do alvo inteiro (`y`). Valida as colunas obrigatórias, rejeita infinitos numéricos e exige alvo completo com valores 0/1, sem imputação. Uniformiza categorias como texto, preservando ausências. |
| `construir_preprocessador(...)` | Retorna um `ColumnTransformer` ainda não ajustado, com saída numérica densa. |
| `construir_pipeline(...)` | Retorna o pipeline ainda não ajustado: pré-processamento, seleção, PCA e kNN. |
| `configuracoes_experimentais()` | Retorna uma lista de 385 dicionários aceitos por `construir_pipeline`. |
| `validacao_cruzada()` | Retorna `StratifiedKFold` com dez partições, embaralhamento e semente 42. Requer ao menos dez observações por classe para representar ambas em cada partição. |

A classe pública `TratamentoExtremos(estrategia="preservar")` aplica as opções de tratamento de extremos descritas abaixo. Ela recebe dados numéricos já imputados na ordem dos preditores numéricos e não remove linhas.

## 3. Configurações disponíveis

| Parâmetro | Opções | Padrão e comportamento |
| --- | --- | --- |
| `imputacao` | `"median"`, `"mean"` | Padrão: `"median"`. Preenche ausências numéricas com mediana ou média do treino. Categorias recebem a moda do treino. |
| `escala` | `"standard"`, `"minmax"`, `"robust"` | Padrão: `"standard"`. Usa `StandardScaler`, `MinMaxScaler` ou `RobustScaler` após imputação e tratamento de extremos. |
| `extremos` | `"preservar"`, `"log_renda"`, `"iqr"` | Padrão: `"preservar"`. Mantém os valores, aplica `log1p` apenas à renda ou limita os valores numéricos a 1,5 IQR abaixo do primeiro quartil e acima do terceiro quartil, com limites aprendidos no treino. `log_renda` rejeita renda negativa. |
| `faixas` | `None`, `3`, `5` | Padrão: `None`. Discretiza os preditores numéricos por quantis, com saída one-hot. Empates podem reduzir o número efetivo de faixas. |
| `selecao` | `25`, `40`, `50`, `60`, `75`, `90`, `100` | Padrão: `100`, que desativa a seleção. Os demais valores indicam o percentual de colunas codificadas retidas por `SelectPercentile` com `f_classif`. |
| `pca` | `None` ou fração entre 0 e 1, sem incluir os extremos | Padrão: `None`, que desativa PCA. A fração define a variância a preservar; o PCA usa SVD completo em matriz densa centrada. |

`construir_preprocessador` recebe os quatro primeiros parâmetros; `construir_pipeline` recebe todos. Ao definir `faixas`, mantenha `escala="standard"` e `extremos="preservar"`: a discretização substitui essas transformações.

As categorias são codificadas com `OneHotEncoder`. Categorias novas geram indicadores zero no bloco correspondente. Colunas totalmente vazias no treino são preservadas pelo imputador.

A grade contém **385 configurações candidatas**:

- Nove bases contínuas: três escalas × três tratamentos de extremos.
- Duas bases discretizadas: três ou cinco faixas.
- Sete percentuais de seleção.
- Cinco opções de PCA: `None`, `0.8`, `0.9`, `0.95` e `0.99`.

Assim, `(9 + 2) × 7 × 5 = 385`. A grade fixa a imputação em mediana; a média continua disponível nas funções. Sem valores ausentes, essas duas opções de imputação produzem o mesmo resultado.

A seleção atua nas colunas codificadas, não diretamente nas dez variáveis originais. Configurações diferentes podem produzir representações ou geometrias equivalentes: limiares distintos de PCA podem reter a mesma dimensão, e a seleção pode eliminar diferenças entre as bases. Portanto, 385 configurações não garantem 385 transformações distintas; confira a contagem conforme os critérios da disciplina.

## 4. Testes e observações

Para executar as verificações, use a raiz do projeto:

```console
python tests/verificar_preprocessamento.py
```

O script verifica as 385 configurações em uma amostra com 240 registros de treino e 60 de validação, testa a integração com dez partições e transforma a base completa com o preprocessador padrão. Também verifica imputação, escalas, categorias novas, tratamento de extremos, parâmetros inválidos e estatísticas aprendidas apenas no treino.

A execução grava relatórios em `outputs/preprocessamento` e executa integralmente o notebook `notebooks/03_preprocessamento.ipynb`, salvando suas saídas. Os relatórios locais não precisam ser versionados. Essas verificações são funcionais e não substituem a comparação completa das 385 configurações.

O arquivo `outputs/preprocessamento/equivalencias_configuracoes.json`, gerado pelo script, registra equivalências observadas nos 300 registros da amostra. Esse diagnóstico não garante equivalência em toda a base ou em outras partições e não deve orientar escolhas usando dados reservados para avaliação final. A transformação da base completa feita pelo script também não deve ser reutilizada como entrada da validação cruzada.

Registre configurações, versões das bibliotecas, avisos e resultados dos experimentos. Reserve um conjunto de teste ou uma avaliação externa para a estimativa final: escolher e avaliar configurações nas mesmas partições pode produzir resultados otimistas. O balanceamento ainda depende de decisão da equipe e, se aplicado, deve usar apenas os dados de treino de cada partição.
