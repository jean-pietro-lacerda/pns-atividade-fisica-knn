
from pathlib import Path

import numpy as np
import pandas as pd

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
    confusion_matrix,
    average_precision_score,
)

from src.preprocessamento import separar_xy
from src.treinamento import treinar_knn


BASE = Path("data/interim/pns_base_eda.parquet")

SAIDA = Path(
    "outputs/diagnostico_pedro/limiares"
)

# Limiar e quantidade minima de votos positivos.
LIMIARES = {
    0.5: 3,
    0.4: 2,
    0.2: 1,
    0.8: 4,
    1.0: 5,
}


def executar():

    SAIDA.mkdir(parents=True, exist_ok=True)

    X, y = separar_xy(pd.read_parquet(BASE))

    assert X.shape[1] == 10, (
        "Esperados 10 atributos."
    )

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

    print("10 atributos | Robust | kNN padrao | 10 folds")
    print(f"Desenvolvimento: {len(X_dev)} registros")
    print("Teste reservado NAO utilizado.")

    for fold, (tr, va) in enumerate(
        cv.split(X_dev, y_dev), 1
    ):

        modelo = treinar_knn(
            X_dev.iloc[tr],
            y_dev.iloc[tr],
            configuracao={"escala": "robust"},
        )

        knn = modelo.named_steps["knn"]

        assert knn.n_neighbors == 5
        assert knn.weights == "uniform"

        real = y_dev.iloc[va].to_numpy()

        # Uma unica busca de vizinhos por fold.
        indice = list(modelo.classes_).index(1)

        p = modelo.predict_proba(
            X_dev.iloc[va]
        )[:, indice]

        votos = np.rint(5 * p).astype(int)

        assert np.allclose(p, votos / 5), (
            "Escores inesperados."
        )

        assert ((votos >= 0) & (votos <= 5)).all()

        auc = roc_auc_score(real, p)

        ap = average_precision_score(real, p)

        registros.append(pd.DataFrame({
            "posicao_dev": va,
            "fold": fold,
            "real": real,
            "prob_ativo": p,
            "votos_ativos": votos,
        }))

        # Avaliar todos os limiares sem retreinar.
        for limiar, minimo in LIMIARES.items():

            pred = (votos >= minimo).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                real, pred, labels=[0, 1]
            ).ravel()

            resultados.append({
                "fold": fold,
                "limiar": limiar,
                "min_votos": minimo,

                "acuracia": accuracy_score(
                    real, pred
                ),

                "acuracia_balanceada": (
                    balanced_accuracy_score(
                        real, pred
                    )
                ),

                "precisao": precision_score(
                    real, pred, zero_division=0
                ),

                "recall": recall_score(
                    real, pred, zero_division=0
                ),

                "f1": f1_score(
                    real, pred, zero_division=0
                ),

                "f2": fbeta_score(
                    real, pred,
                    beta=2,
                    zero_division=0,
                ),

                "auc": auc,
                "average_precision": ap,
                "fpr": fp / (fp + tn),

                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            })

        print(
            f"[{fold}/10] "
            f"Cinco limiares avaliados | "
            f"AUC={auc:.4f}",
            flush=True,
        )

    folds = pd.DataFrame(resultados)

    oof = pd.concat(
        registros, ignore_index=True
    ).sort_values("posicao_dev")

    assert len(oof) == len(X_dev)
    assert oof["posicao_dev"].is_unique

    metricas = [
        "acuracia",
        "acuracia_balanceada",
        "precisao",
        "recall",
        "f1",
        "f2",
        "auc",
        "average_precision",
        "fpr",
    ]

    medias = folds.groupby(
        "limiar"
    )[metricas].mean().reindex(LIMIARES)

    dp = folds.groupby(
        "limiar"
    )[metricas].std().reindex(LIMIARES)

    matrizes = folds.groupby("limiar")[
        ["tn", "fp", "fn", "tp"]
    ].mean().reindex(LIMIARES)

    # Comparar ganhos e custos contra o limiar original.
    ganhos = pd.DataFrame({

        "ativos_recuperados_por_fold": (
            matrizes.loc[0.5, "fn"] - matrizes["fn"]
        ),

        "fp_adicionais_por_fold": (
            matrizes["fp"] - matrizes.loc[0.5, "fp"]
        ),

        "delta_f1": (
            medias["f1"] - medias.loc[0.5, "f1"]
        ),

        "delta_recall": (
            medias["recall"] - medias.loc[0.5, "recall"]
        ),
    })

    # Distribuicao dos votos por classe verdadeira.
    votos = pd.crosstab(
        oof["votos_ativos"],
        oof["real"],
    ).reindex(
        index=range(6),
        columns=[0, 1],
        fill_value=0,
    )

    votos.columns = [
        "inativos_reais",
        "ativos_reais",
    ]

    votos["pct_dos_ativos"] = (
        100
        * votos["ativos_reais"]
        / votos["ativos_reais"].sum()
    )

    votos["pct_dos_inativos"] = (
        100
        * votos["inativos_reais"]
        / votos["inativos_reais"].sum()
    )

    # Falsos negativos do kNN original:
    # pessoas ativas com zero, um ou dois votos.
    fn = votos.loc[0:2, ["ativos_reais"]].copy()

    fn["pct_dos_fn"] = (
        100
        * fn["ativos_reais"]
        / fn["ativos_reais"].sum()
    )

    for titulo, tabela in [

        ("DISTRIBUICAO DOS VOTOS (OOF)", votos),

        (
            "PERFIL DOS FALSOS NEGATIVOS (LIMIAR 0.5)",
            fn,
        ),

        (
            "COMPARACAO DOS LIMIARES - MEDIA DOS 10 FOLDS",
            medias,
        ),

        ("DESVIO-PADRAO DOS 10 FOLDS", dp),

        ("MATRIZES DE CONFUSAO MEDIAS", matrizes),

        (
            "GANHOS E CUSTOS CONTRA LIMIAR 0.5",
            ganhos,
        ),

    ]:

        print("\n=== " + titulo + " ===")
        print(tabela.round(4).to_string())

    # Salvar os resultados e probabilidades individuais.
    for nome, tabela, indice in [

        ("PROBABILIDADES_OOF", oof, False),

        ("METRICAS_POR_FOLD", folds, False),

        ("COMPARACAO_LIMIARES", medias, True),

        ("DESVIOS_PADRAO", dp, True),

        ("MATRIZES_MEDIAS", matrizes, True),

        ("GANHOS_E_CUSTOS", ganhos, True),

        ("DISTRIBUICAO_VOTOS", votos, True),

        ("PERFIL_FALSOS_NEGATIVOS", fn, True),

    ]:

        tabela.to_csv(
            SAIDA / f"{nome}.csv",
            index=indice,
        )

    print(f"\nResultados: {SAIDA.resolve()}")
    print("Diagnostico exploratorio.")
    print("Nao seleciona limiar definitivo.")
    print("Teste reservado NAO utilizado.")
    print("Codigo oficial NAO alterado.")


if __name__ == "__main__":
    executar()
