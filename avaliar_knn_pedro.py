
import numpy as np
import pandas as pd

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
from sklearn.model_selection import train_test_split

from src.preprocessamento import separar_xy, validacao_cruzada
from src.treinamento import treinar_knn


def avaliar_configuracoes():
    base = pd.read_parquet("data/interim/pns_base_eda.parquet")
    X, y = separar_xy(base)

    # Reproduz a mesma divisão 80/20 da execução anterior.
    # Os 20% de teste permanecem reservados.
    X_treino, _, y_treino, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    configuracoes = {
        "standard": {"escala": "standard"},
        "minmax": {"escala": "minmax"},
        "robust": {"escala": "robust"},
    }

    resumo = []

    for nome, configuracao in configuracoes.items():
        print(f"\n=== {nome.upper()} ===", flush=True)

        resultados = []
        matriz_total = np.zeros((2, 2), dtype=int)

        cv = validacao_cruzada()

        for fold, (idx_treino, idx_validacao) in enumerate(
            cv.split(X_treino, y_treino),
            start=1,
        ):
            X_fold_treino = X_treino.iloc[idx_treino]
            y_fold_treino = y_treino.iloc[idx_treino]

            X_fold_validacao = X_treino.iloc[idx_validacao]
            y_fold_validacao = y_treino.iloc[idx_validacao]

            # Novo pipeline para cada fold:
            # nada é ajustado usando os dados de validação.
            modelo = treinar_knn(
                X_fold_treino,
                y_fold_treino,
                configuracao=configuracao,
            )

            previsoes = modelo.predict(X_fold_validacao)

            indice_classe_1 = list(modelo.classes_).index(1)
            probabilidades = modelo.predict_proba(
                X_fold_validacao
            )[:, indice_classe_1]

            resultados.append({
                "acuracia": accuracy_score(
                    y_fold_validacao, previsoes
                ),
                "acuracia_balanceada": balanced_accuracy_score(
                    y_fold_validacao, previsoes
                ),
                "precisao_classe_1": precision_score(
                    y_fold_validacao, previsoes,
                    pos_label=1, zero_division=0
                ),
                "recall_classe_1": recall_score(
                    y_fold_validacao, previsoes,
                    pos_label=1, zero_division=0
                ),
                "recall_classe_0": recall_score(
                    y_fold_validacao, previsoes,
                    pos_label=0, zero_division=0
                ),
                "f1": f1_score(
                    y_fold_validacao, previsoes,
                    pos_label=1, zero_division=0
                ),
                "f2": fbeta_score(
                    y_fold_validacao, previsoes,
                    beta=2, pos_label=1,
                    zero_division=0
                ),
                "auc": roc_auc_score(
                    y_fold_validacao, probabilidades
                ),
            })

            matriz_total += confusion_matrix(
                y_fold_validacao,
                previsoes,
                labels=[0, 1],
            )

            print(f"Fold {fold}/10 concluído", flush=True)

        tabela = pd.DataFrame(resultados)
        medias = tabela.mean()
        desvios = tabela.std(ddof=1)

        print("\nMédias:")
        print(medias.round(4).to_string())

        print("\nDesvios-padrão:")
        print(desvios.round(4).to_string())

        print("\nMatriz de confusão média:")
        print((matriz_total / cv.n_splits).round(1))

        resumo.append({
            "escalonamento": nome,
            **medias.to_dict(),
        })

    print("\n=== COMPARAÇÃO FINAL ===")
    comparacao = pd.DataFrame(resumo).set_index("escalonamento")
    print(comparacao.round(4).to_string())

    print("\nAvaliação diagnóstica concluída.")
    print("Conjunto de teste reservado não utilizado.")


if __name__ == "__main__":
    avaliar_configuracoes()


