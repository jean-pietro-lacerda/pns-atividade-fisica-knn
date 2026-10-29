# Guia de integração do pré-processamento

A entrada é `data/interim/pns_base_eda.parquet`, com dez preditores e o alvo `ativo_lazer`. Não se deve fornecer uma matriz previamente transformada à validação cruzada: imputação, limites de extremos, encoding, escala, seleção e PCA devem ser ajustados em cada treino.

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

Este exemplo não foi usado para executar uma avaliação completa da base nesta revisão. O kNN permanece com todos os parâmetros padrão. F1 usa classe positiva 1; AUC-ROC utiliza os escores do estimador. A matriz de confusão requer previsões fora da amostra.

## Grade e interpretação

São 385 configurações candidatas: nove bases contínuas e duas discretizadas, sete percentuais de seleção e cinco alternativas de PCA. A imputação da grade é mediana; média continua disponível na API. Seleção atua nas colunas codificadas. Discretização por quantis substitui escala e extremos: ao definir `faixas`, mantenha os argumentos padrão de `escala` e `extremos`.

Combinações diferentes podem gerar representações ou geometrias equivalentes. `outputs/preprocessamento/equivalencias_configuracoes.json` contém uma auditoria diagnóstica em 300 registros reais, não uma garantia sobre todos os registros ou partições. A contagem de configurações equivalentes deve ser conferida com os critérios da disciplina.

## Execução das verificações

Na raiz do projeto:

```console
python tests/verificar_preprocessamento.py
```

O script verifica as 385 receitas em uma amostra de treino/validação, testa integração com dez partições, transforma a base completa com o preprocessador padrão e executa integralmente o notebook 03. Os resultados são diagnósticos funcionais, não a comparação completa das 385 receitas. Os relatórios locais em `outputs/preprocessamento` não precisam ser versionados.

Registre receitas, versões, avisos e resultados nos experimentos posteriores. Reserve teste ou avaliação externa para a estimativa final, pois selecionar e avaliar nas mesmas partições pode produzir otimismo. Balanceamento permanece pendente de decisão da equipe e deve se restringir ao treino. Preserve eventuais declarações de ferramentas exigidas pela disciplina; este guia não atribui autoria nem substitui tais exigências.
