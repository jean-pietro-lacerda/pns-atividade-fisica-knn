
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
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
)

from src.preprocessamento import separar_xy
from src.treinamento import treinar_knn


# ============================================================
# CONFIGURACAO
# ============================================================

BASE = Path("data/interim/pns_base_eda.parquet")

SAIDA = Path(
    "outputs/diagnostico_pedro/retreinamento_3_escalas"
)

ESCALAS = ["standard", "minmax", "robust"]

N_FOLDS = 10
N_REPETICOES = 5
SEMENTE = 42

METRICAS = [
    "acuracia",
    "acuracia_balanceada",
    "precisao_ativo",
    "recall_ativo",
    "recall_inativo",
    "f1",
    "f2",
    "auc",
    "taxa_falsos_positivos",
]


# ============================================================
# EXECUTAR EXPERIMENTO
# ============================================================

def executar():

    SAIDA.mkdir(parents=True, exist_ok=True)

    # Identificar a base e a versao do experimento.
    assinatura = hashlib.sha256()

    for arquivo in [
        BASE,
        Path(__file__),
        Path("src/preprocessamento.py"),
        Path("src/treinamento.py"),
    ]:
        assinatura.update(arquivo.read_bytes())

    assinatura.update(sklearn.__version__.encode())
    assinatura.update(pd.__version__.encode())
    assinatura.update(np.__version__.encode())

    identificador = assinatura.hexdigest()[:16]

    checkpoint = (
        SAIDA / f"FOLDS_{identificador}.csv"
    )

    base = pd.read_parquet(BASE)

    X, y = separar_xy(base)

    assert X.shape[1] == 10, (
        "O experimento exige os 10 atributos."
    )

    # Preservar a mesma divisao 80/20 anterior.
    X_treino, _, y_treino, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEMENTE,
    )

    cv = RepeatedStratifiedKFold(
        n_splits=N_FOLDS,
        n_repeats=N_REPETICOES,
        random_state=SEMENTE,
    )

    # Todas as escalas utilizam as mesmas particoes.
    folds = list(
        cv.split(X_treino, y_treino)
    )

    total = len(ESCALAS) * len(folds)

    resultados = []

    if checkpoint.exists():
        resultados = pd.read_csv(
            checkpoint
        ).to_dict("records")

    concluidos = {
        (r["escala"], int(r["execucao"]))
        for r in resultados
    }

    print("=" * 60)
    print("RETREINAMENTO DOS 3 ESCALONADORES")
    print("=" * 60)

    print(f"Atributos: {X.shape[1]}")
    print(f"Registros de treino: {len(X_treino)}")
    print("kNN com parametros padrao")
    print("One-Hot Encoding")
    print(f"Folds: {N_FOLDS}")
    print(f"Repeticoes: {N_REPETICOES}")
    print(f"Total de ajustes: {total}")
    print(f"Ajustes recuperados: {len(concluidos)}")
    print("Teste reservado: NAO utilizado")

    for escala in ESCALAS:

        print("\n" + "=" * 60)
        print(f"ESCALONADOR: {escala.upper()}")
        print("=" * 60)

        for execucao, (
            idx_treino,
            idx_validacao,
        ) in enumerate(folds, start=1):

            repeticao = (
                (execucao - 1) // N_FOLDS
            ) + 1

            fold = (
                (execucao - 1) % N_FOLDS
            ) + 1

            chave = (escala, execucao)

            if chave in concluidos:
                continue

            X_fold_treino = X_treino.iloc[
                idx_treino
            ]

            y_fold_treino = y_treino.iloc[
                idx_treino
            ]

            X_fold_validacao = X_treino.iloc[
                idx_validacao
            ]

            y_fold_validacao = y_treino.iloc[
                idx_validacao
            ]

            # Novo pipeline em cada fold.
            # Pre-processamento ajustado somente no treino.

            inicio = perf_counter()

            modelo = treinar_knn(
                X_fold_treino,
                y_fold_treino,
                configuracao={"escala": escala},
            )

            tempo_treino = perf_counter() - inicio

            inicio = perf_counter()

            previsoes = modelo.predict(
                X_fold_validacao
            )

            tempo_previsao = perf_counter() - inicio

            inicio = perf_counter()

            indice_classe_1 = list(
                modelo.classes_
            ).index(1)

            probabilidades = modelo.predict_proba(
                X_fold_validacao
            )[:, indice_classe_1]

            tempo_probabilidade = (
                perf_counter() - inicio
            )

            tn, fp, fn, tp = confusion_matrix(
                y_fold_validacao,
                previsoes,
                labels=[0, 1],
            ).ravel()

            resultado = {
                "escala": escala,
                "execucao": execucao,
                "repeticao": repeticao,
                "fold": fold,

                "acuracia": accuracy_score(
                    y_fold_validacao, previsoes
                ),

                "acuracia_balanceada": (
                    balanced_accuracy_score(
                        y_fold_validacao, previsoes
                    )
                ),

                "precisao_ativo": precision_score(
                    y_fold_validacao,
                    previsoes,
                    pos_label=1,
                    zero_division=0,
                ),

                "recall_ativo": recall_score(
                    y_fold_validacao,
                    previsoes,
                    pos_label=1,
                    zero_division=0,
                ),

                "recall_inativo": recall_score(
                    y_fold_validacao,
                    previsoes,
                    pos_label=0,
                    zero_division=0,
                ),

                "f1": f1_score(
                    y_fold_validacao,
                    previsoes,
                    pos_label=1,
                    zero_division=0,
                ),

                "f2": fbeta_score(
                    y_fold_validacao,
                    previsoes,
                    beta=2,
                    pos_label=1,
                    zero_division=0,
                ),

                "auc": roc_auc_score(
                    y_fold_validacao,
                    probabilidades,
                ),

                "taxa_falsos_positivos": (
                    fp / (fp + tn)
                ),

                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,

                "tempo_treino_s": tempo_treino,
                "tempo_previsao_s": tempo_previsao,
                "tempo_probabilidade_s": (
                    tempo_probabilidade
                ),
            }

            resultados.append(resultado)
            concluidos.add(chave)

            # Salvar apos cada fold concluido.
            temporario = checkpoint.with_suffix(".tmp")

            pd.DataFrame(resultados).to_csv(
                temporario,
                index=False,
            )

            temporario.replace(checkpoint)

            print(
                f"[{len(concluidos)}/{total}] "
                f"{escala.upper()} | "
                f"Repeticao {repeticao}/5 | "
                f"Fold {fold}/10 | "
                f"F1={resultado['f1']:.4f} | "
                f"AUC={resultado['auc']:.4f} | "
                f"Recall={resultado['recall_ativo']:.4f}",
                flush=True,
            )

    tabela = pd.DataFrame(resultados)


    # ========================================================
    # MEDIAS GERAIS
    # ========================================================

    medias = tabela.groupby(
        "escala"
    )[METRICAS].mean().reindex(ESCALAS)

    print("\n" + "=" * 60)
    print("COMPARACAO FINAL - MEDIA DOS 50 FOLDS")
    print("=" * 60)

    print(medias.round(4).to_string())


    # ========================================================
    # ESTABILIDADE ENTRE FOLDS
    # ========================================================

    desvios = tabela.groupby(
        "escala"
    )[METRICAS].std(ddof=1).reindex(ESCALAS)

    print("\n" + "=" * 60)
    print("DESVIOS-PADRAO DOS 50 FOLDS")
    print("=" * 60)

    print(desvios.round(4).to_string())


    # ========================================================
    # RESULTADOS POR REPETICAO
    # ========================================================

    por_repeticao = tabela.groupby(
        ["escala", "repeticao"]
    )[METRICAS].mean()

    print("\n" + "=" * 60)
    print("RESULTADOS POR REPETICAO")
    print("=" * 60)

    print(por_repeticao.round(4).to_string())

    estabilidade = por_repeticao.groupby(
        level="escala"
    ).std(ddof=1).reindex(ESCALAS)

    print("\n" + "=" * 60)
    print("ESTABILIDADE ENTRE AS 5 REPETICOES")
    print("=" * 60)

    print(estabilidade.round(4).to_string())


    # ========================================================
    # COMPARACAO PAREADA
    # ========================================================

    comparacoes = []

    for metrica in [
        "f1",
        "auc",
        "recall_ativo",
        "acuracia_balanceada",
    ]:

        pares = tabela.pivot(
            index="execucao",
            columns="escala",
            values=metrica,
        )

        for escala_a, escala_b in [
            ("robust", "standard"),
            ("robust", "minmax"),
            ("standard", "minmax"),
        ]:

            delta = pares[escala_a] - pares[escala_b]

            comparacoes.append({
                "metrica": metrica,
                "comparacao": f"{escala_a} - {escala_b}",
                "diferenca_media": delta.mean(),
                "desvio_diferenca": delta.std(ddof=1),
                "folds_melhores": int((delta > 1e-12).sum()),
                "folds_piores": int((delta < -1e-12).sum()),
                "empates": int((delta.abs() <= 1e-12).sum()),
            })

    tabela_pareada = pd.DataFrame(comparacoes)

    print("\n" + "=" * 60)
    print("COMPARACAO PAREADA ENTRE ESCALONADORES")
    print("=" * 60)

    print(
        tabela_pareada.round(4).to_string(index=False)
    )


    # ========================================================
    # MATRIZES DE CONFUSAO
    # ========================================================

    print("\n" + "=" * 60)
    print("MATRIZES DE CONFUSAO MEDIAS POR FOLD")
    print("=" * 60)

    for escala in ESCALAS:

        dados = tabela[
            tabela["escala"] == escala
        ]

        matriz = np.array([
            [
                dados["tn"].mean(),
                dados["fp"].mean(),
            ],
            [
                dados["fn"].mean(),
                dados["tp"].mean(),
            ],
        ])

        print(f"\n{escala.upper()}:")
        print(matriz.round(1))


    # ========================================================
    # TEMPO COMPUTACIONAL
    # ========================================================

    tempos = tabela.groupby("escala")[
        [
            "tempo_treino_s",
            "tempo_previsao_s",
            "tempo_probabilidade_s",
        ]
    ].mean().reindex(ESCALAS)

    print("\n" + "=" * 60)
    print("TEMPO MEDIO POR FOLD (SEGUNDOS)")
    print("=" * 60)

    print(tempos.round(4).to_string())


    # ========================================================
    # COMPARACAO COM O PRIMEIRO EXPERIMENTO
    # ========================================================

    anteriores = pd.DataFrame(
        {
            "standard": {
                "acuracia": 0.6966,
                "f1": 0.3573,
                "auc": 0.6272,
                "recall_ativo": 0.2977,
            },
            "minmax": {
                "acuracia": 0.6926,
                "f1": 0.3503,
                "auc": 0.6242,
                "recall_ativo": 0.2924,
            },
            "robust": {
                "acuracia": 0.6956,
                "f1": 0.3590,
                "auc": 0.6281,
                "recall_ativo": 0.3009,
            },
        }
    ).T

    diferencas = (
        medias[anteriores.columns] - anteriores
    )

    print("\n" + "=" * 60)
    print("DIFERENCA: 50 FOLDS - 10 FOLDS ANTERIORES")
    print("=" * 60)

    print(diferencas.round(4).to_string())


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    medias.to_csv(
        SAIDA / "COMPARACAO_FINAL.csv"
    )

    desvios.to_csv(
        SAIDA / "DESVIOS_PADRAO.csv"
    )

    por_repeticao.to_csv(
        SAIDA / "RESULTADOS_POR_REPETICAO.csv"
    )

    estabilidade.to_csv(
        SAIDA / "ESTABILIDADE.csv"
    )

    tabela_pareada.to_csv(
        SAIDA / "COMPARACAO_PAREADA.csv",
        index=False,
    )

    tempos.to_csv(
        SAIDA / "TEMPOS.csv"
    )

    diferencas.to_csv(
        SAIDA / "DIFERENCAS_EXPERIMENTO_ANTERIOR.csv"
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTO CONCLUIDO")
    print("=" * 60)

    print(f"Total de ajustes: {len(tabela)}")
    print("Todos os 10 atributos foram mantidos.")
    print("Teste reservado NAO utilizado.")
    print("Nenhum codigo oficial foi alterado.")
    print(f"Resultados: {SAIDA.resolve()}")


if __name__ == "__main__":
    executar()
