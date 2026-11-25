
from pathlib import Path
from time import perf_counter
import warnings

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.preprocessamento import (
    separar_xy,
    construir_pipeline,
)


RAIZ = Path(__file__).resolve().parent

BASE = RAIZ / "data/interim/pns_base_eda.parquet"

SAIDA = (
    RAIZ / "outputs/diagnostico_pedro/regressao_logistica"
)

# Resultados anteriores do kNN com os mesmos 10 folds.
REFERENCIA = {
    "acuracia": 0.6956,
    "acuracia_balanceada": 0.5763,
    "precisao": 0.4452,
    "recall": 0.3009,
    "f1": 0.3590,
    "f2": 0.3217,
    "auc": 0.6281,
    "average_precision": 0.3768,
    "fpr": 0.1483,
}


def executar():

    SAIDA.mkdir(parents=True, exist_ok=True)

    X, y = separar_xy(pd.read_parquet(BASE))

    assert X.shape[1] == 10

    # Mesma divisao utilizada nos experimentos anteriores.
    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    resultados = []
    registros = []

    print("REGRESSAO LOGISTICA")
    print("10 atributos | Robust | 10 folds")
    print(f"Desenvolvimento: {len(X_dev)} registros")
    print("Teste reservado NAO utilizado.")

    for fold, (tr, va) in enumerate(
        cv.split(X_dev, y_dev), 1
    ):

        original = construir_pipeline(
            escala="robust"
        )

        assert original.steps[-1][0] == "knn"

        # Reutiliza todo o pre-processamento.
        # Substitui apenas o estimador final.
        modelo = Pipeline(
            original.steps[:-1] + [
                ("regressao", LogisticRegression())
            ]
        )

        inicio = perf_counter()

        with warnings.catch_warnings(
            record=True
        ) as avisos:

            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            modelo.fit(
                X_dev.iloc[tr],
                y_dev.iloc[tr],
            )

        treino_s = perf_counter() - inicio

        convergiu = not any(
            issubclass(
                a.category,
                ConvergenceWarning,
            )
            for a in avisos
        )

        for a in avisos:
            print(
                f"Aviso fold {fold}: {a.message}"
            )

        inicio = perf_counter()

        real = y_dev.iloc[va].to_numpy()

        pred = modelo.predict(
            X_dev.iloc[va]
        )

        indice = list(
            modelo.classes_
        ).index(1)

        prob = modelo.predict_proba(
            X_dev.iloc[va]
        )[:, indice]

        previsao_s = perf_counter() - inicio

        tn, fp, fn, tp = confusion_matrix(
            real,
            pred,
            labels=[0, 1],
        ).ravel()

        resultado = {
            "fold": fold,

            "acuracia": accuracy_score(
                real, pred
            ),

            "acuracia_balanceada": (
                balanced_accuracy_score(
                    real, pred
                )
            ),

            "precisao": precision_score(
                real,
                pred,
                zero_division=0,
            ),

            "recall": recall_score(
                real,
                pred,
                zero_division=0,
            ),

            "f1": f1_score(
                real,
                pred,
                zero_division=0,
            ),

            "f2": fbeta_score(
                real,
                pred,
                beta=2,
                zero_division=0,
            ),

            "auc": roc_auc_score(
                real, prob
            ),

            "average_precision": (
                average_precision_score(
                    real, prob
                )
            ),

            "fpr": fp / (fp + tn),

            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,

            "convergiu": convergiu,

            "iteracoes": int(
                modelo.named_steps[
                    "regressao"
                ].n_iter_.max()
            ),

            "treino_s": treino_s,

            "previsao_e_proba_s": previsao_s,
        }

        resultados.append(resultado)

        registros.append(
            pd.DataFrame({
                "posicao_dev": va,
                "fold": fold,
                "real": real,
                "previsto": pred,
                "prob_ativo": prob,
            })
        )

        print(
            f"[{fold}/10] "
            f"F1={resultado['f1']:.4f} | "
            f"AUC={resultado['auc']:.4f} | "
            f"Recall={resultado['recall']:.4f} | "
            f"Convergiu={convergiu}",
            flush=True,
        )

    folds = pd.DataFrame(resultados)

    oof = pd.concat(
        registros,
        ignore_index=True,
    ).sort_values("posicao_dev")

    assert len(oof) == len(X_dev)
    assert oof["posicao_dev"].is_unique

    metricas = list(REFERENCIA)

    resumo = folds[metricas].agg(
        ["mean", "std"]
    ).T

    resumo.columns = [
        "media",
        "desvio_padrao",
    ]

    comparacao = pd.DataFrame({
        "knn": pd.Series(REFERENCIA),
        "regressao": resumo["media"],
    })

    comparacao["delta"] = (
        comparacao["regressao"]
        - comparacao["knn"]
    )

    matriz = folds[
        ["tn", "fp", "fn", "tp"]
    ].mean()

    print(
        "\n=== COMPARACAO COM KNN"
        " - MEDIA DOS 10 FOLDS ==="
    )

    print(
        comparacao.round(4).to_string()
    )

    print("\n=== MEDIA E DESVIO-PADRAO ===")

    print(
        resumo.round(4).to_string()
    )

    print("\n=== MATRIZ DE CONFUSAO MEDIA ===")

    print(
        matriz.round(1).to_string()
    )

    print("\n=== CONVERGENCIA E TEMPOS ===")

    print(
        "Folds sem aviso de convergencia: "
        f"{folds['convergiu'].sum()}/10"
    )

    print(
        folds[
            [
                "iteracoes",
                "treino_s",
                "previsao_e_proba_s",
            ]
        ].mean().round(4).to_string()
    )

    if not folds["convergiu"].all():

        print(
            "ATENCAO: verificar convergencia "
            "antes de concluir a comparacao."
        )

    # Salvar todos os resultados.
    folds.to_csv(
        SAIDA / "METRICAS_POR_FOLD.csv",
        index=False,
    )

    oof.to_csv(
        SAIDA / "PROBABILIDADES_OOF.csv",
        index=False,
    )

    resumo.to_csv(
        SAIDA / "RESUMO.csv"
    )

    comparacao.to_csv(
        SAIDA / "COMPARACAO_KNN.csv"
    )

    matriz.to_csv(
        SAIDA / "MATRIZ_MEDIA.csv"
    )

    print(f"\nResultados: {SAIDA.resolve()}")

    print("Diagnostico local.")
    print("Codigo oficial intacto.")
    print("Teste reservado NAO utilizado.")


if __name__ == "__main__":
    executar()
