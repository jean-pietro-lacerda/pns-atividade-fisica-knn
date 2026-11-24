import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)

from src.preprocessamento import (
    NUMERICAS,
    CATEGORICAS,
    separar_xy,
    construir_pipeline,
    validacao_cruzada,
)


# Ordem correta dos níveis de escolaridade
ORDEM_ESCOLARIDADE = [
    "Sem instrução",
    "Fundamental incompleto",
    "Fundamental completo",
    "Médio incompleto",
    "Médio completo",
    "Superior incompleto",
    "Superior completo",
]


def criar_modelo(usar_ordinal=False):
    """
    Cria o kNN com RobustScaler.

    usar_ordinal=False:
        Pré-processamento original, com One-Hot para escolaridade.

    usar_ordinal=True:
        Escolaridade passa a ser ordinal.
        As outras categorias continuam com One-Hot.

    O kNN permanece com os parâmetros padrão.
    """

    modelo = construir_pipeline(escala="robust")

    if not usar_ordinal:
        return modelo

    # Recuperar as etapas originais SEM ajustá-las.
    preprocessador_original = modelo.named_steps["preprocessamento"]

    transformadores = {
        nome: transformador
        for nome, transformador, colunas
        in preprocessador_original.transformers
    }

    pipeline_numerico = transformadores["numericas"]
    pipeline_categorico = transformadores["categoricas"]

    # Novo pipeline EXCLUSIVO para escolaridade.
    pipeline_escolaridade = Pipeline([
        (
            "imputacao",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[ORDEM_ESCOLARIDADE],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
        (
            "escala",
            MinMaxScaler(),
        ),
    ])

    outras_categoricas = [
        coluna
        for coluna in CATEGORICAS
        if coluna != "escolaridade"
    ]

    novo_preprocessador = ColumnTransformer(
        [
            (
                "numericas",
                pipeline_numerico,
                NUMERICAS,
            ),
            (
                "categoricas",
                pipeline_categorico,
                outras_categoricas,
            ),
            (
                "escolaridade_ordinal",
                pipeline_escolaridade,
                ["escolaridade"],
            ),
        ],
        remainder="drop",
    )

    # Substituir apenas a etapa de pré-processamento.
    modelo.set_params(preprocessamento=novo_preprocessador)

    return modelo


def calcular_metricas(modelo, X_validacao, y_validacao):
    previsoes = modelo.predict(X_validacao)

    indice_classe_1 = list(modelo.classes_).index(1)

    probabilidades = modelo.predict_proba(
        X_validacao
    )[:, indice_classe_1]

    metricas = {
        "acuracia": accuracy_score(
            y_validacao, previsoes
        ),
        "acuracia_balanceada": balanced_accuracy_score(
            y_validacao, previsoes
        ),
        "precisao_classe_1": precision_score(
            y_validacao,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "recall_classe_1": recall_score(
            y_validacao,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "recall_classe_0": recall_score(
            y_validacao,
            previsoes,
            pos_label=0,
            zero_division=0,
        ),
        "f1": f1_score(
            y_validacao,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_validacao,
            previsoes,
            beta=2,
            pos_label=1,
            zero_division=0,
        ),
        "auc": roc_auc_score(
            y_validacao,
            probabilidades,
        ),
    }

    matriz = confusion_matrix(
        y_validacao,
        previsoes,
        labels=[0, 1],
    )

    return metricas, matriz


def comparar_codificacoes():
    print("Carregando base...", flush=True)

    base = pd.read_parquet(
        "data/interim/pns_base_eda.parquet"
    )

    X, y = separar_xy(base)

    # Conferir se os nomes de escolaridade são os esperados.
    categorias_encontradas = set(
        X["escolaridade"].dropna().astype(str).unique()
    )

    categorias_esperadas = set(ORDEM_ESCOLARIDADE)

    categorias_inesperadas = (
        categorias_encontradas - categorias_esperadas
    )

    if categorias_inesperadas:
        raise ValueError(
            "Foram encontradas categorias de escolaridade "
            "que não estão na ordem definida: "
            f"{sorted(categorias_inesperadas)}"
        )

    # Mesma divisão 80/20 das avaliações anteriores.
    # O conjunto de teste continua reservado.
    X_treino, _, y_treino, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print(f"Registros de treinamento: {len(X_treino)}")
    print("Escalonamento numérico: RobustScaler")
    print("Validação: StratifiedKFold com 10 folds")
    print("kNN: parâmetros padrão")
    print()

    resultados = {
        "onehot": [],
        "ordinal": [],
    }

    matrizes = {
        "onehot": np.zeros((2, 2), dtype=int),
        "ordinal": np.zeros((2, 2), dtype=int),
    }

    cv = validacao_cruzada()

    # Os dois modelos utilizam exatamente os mesmos folds.
    for fold, (idx_treino, idx_validacao) in enumerate(
        cv.split(X_treino, y_treino),
        start=1,
    ):
        X_fold_treino = X_treino.iloc[idx_treino]
        y_fold_treino = y_treino.iloc[idx_treino]

        X_fold_validacao = X_treino.iloc[idx_validacao]
        y_fold_validacao = y_treino.iloc[idx_validacao]

        for nome, usar_ordinal in [
            ("onehot", False),
            ("ordinal", True),
        ]:
            # Modelo novo a cada fold.
            # Imputação, escala e encoding aprendem só no treino.
            modelo = criar_modelo(
                usar_ordinal=usar_ordinal
            )

            modelo.fit(
                X_fold_treino,
                y_fold_treino,
            )

            metricas, matriz = calcular_metricas(
                modelo,
                X_fold_validacao,
                y_fold_validacao,
            )

            resultados[nome].append(metricas)
            matrizes[nome] += matriz

            print(
                f"Fold {fold}/10 | {nome.upper()} concluído",
                flush=True,
            )

    # Apresentar os resultados de cada codificação.
    resumo = {}

    for nome in ["onehot", "ordinal"]:
        tabela = pd.DataFrame(resultados[nome])

        medias = tabela.mean()
        desvios = tabela.std(ddof=1)

        resumo[nome] = medias

        print(f"\n{'=' * 45}")
        print(f"RESULTADOS — {nome.upper()}")
        print("=" * 45)

        print("\nMédias:")
        print(medias.round(4).to_string())

        print("\nDesvios-padrão:")
        print(desvios.round(4).to_string())

        print("\nMatriz de confusão média:")
        print((matrizes[nome] / cv.n_splits).round(1))

    # Comparação direta.
    comparacao = pd.DataFrame(resumo).T

    print(f"\n{'=' * 55}")
    print("COMPARAÇÃO FINAL — ONE-HOT VS ORDINAL")
    print("=" * 55)

    print(comparacao.round(4).to_string())

    # Diferença média pareada: ordinal menos one-hot.
    # Calculada nos mesmos folds.
    tabela_onehot = pd.DataFrame(resultados["onehot"])
    tabela_ordinal = pd.DataFrame(resultados["ordinal"])

    diferencas = tabela_ordinal - tabela_onehot

    print("\nDIFERENÇA MÉDIA POR MÉTRICA")
    print("(ORDINAL - ONE-HOT)")
    print(diferencas.mean().round(4).to_string())

    print("\nInterpretação das diferenças:")
    print("Positivo = ordinal obteve valor maior.")
    print("Negativo = One-Hot obteve valor maior.")
    print("Diferença pequena não comprova superioridade.")

    print("\nAvaliação diagnóstica concluída.")
    print("Conjunto de teste reservado não utilizado.")
    print("Nenhum arquivo do projeto foi modificado.")


if __name__ == "__main__":
    comparar_codificacoes()