
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    confusion_matrix,
)

from src.preprocessamento import (
    NUMERICAS,
    CATEGORICAS,
    separar_xy,
    construir_pipeline,
    validacao_cruzada,
)


# ============================================================
# CONFIGURACAO
# ============================================================

CAMINHO_BASE = Path("data/interim/pns_base_eda.parquet")

PASTA_SAIDA = Path(
    "outputs/diagnostico_pedro/combinacoes_atributos"
)

CONFIGURACOES = [
    ("nenhum", []),

    (
        "sem_numero_moradores_e_uf",
        ["numero_moradores", "uf"],
    ),

    (
        "sem_numero_moradores_e_raca_cor",
        ["numero_moradores", "raca_cor"],
    ),

    (
        "sem_uf_e_raca_cor",
        ["uf", "raca_cor"],
    ),

    (
        "sem_numero_moradores_uf_e_raca_cor",
        ["numero_moradores", "uf", "raca_cor"],
    ),
]


# ============================================================
# CONSTRUIR MODELO SEM OS ATRIBUTOS ESCOLHIDOS
# ============================================================

def criar_modelo(atributos_retirados):
    """
    Mantem o pipeline original, o RobustScaler,
    o One-Hot Encoding e o kNN com parametros padrao.

    Apenas exclui os atributos escolhidos da entrada
    do pre-processamento.

    Nao altera nenhum arquivo da pasta src.
    """

    modelo = construir_pipeline(escala="robust")

    preprocessador_original = modelo.named_steps[
        "preprocessamento"
    ]

    transformadores = {
        nome: transformador
        for nome, transformador, colunas
        in preprocessador_original.transformers
    }

    colunas_numericas = [
        coluna
        for coluna in NUMERICAS
        if coluna not in atributos_retirados
    ]

    colunas_categoricas = [
        coluna
        for coluna in CATEGORICAS
        if coluna not in atributos_retirados
    ]

    novo_preprocessador = ColumnTransformer(
        [
            (
                "numericas",
                transformadores["numericas"],
                colunas_numericas,
            ),
            (
                "categoricas",
                transformadores["categoricas"],
                colunas_categoricas,
            ),
        ],
        remainder="drop",
    )

    modelo.set_params(
        preprocessamento=novo_preprocessador
    )

    return modelo


# ============================================================
# CALCULAR METRICAS
# ============================================================

def calcular_metricas(y_real, previsoes, probabilidades):

    matriz = confusion_matrix(
        y_real,
        previsoes,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matriz.ravel()

    return {
        "acuracia": accuracy_score(
            y_real, previsoes
        ),

        "acuracia_balanceada": balanced_accuracy_score(
            y_real, previsoes
        ),

        "f1": f1_score(
            y_real,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),

        "auc": roc_auc_score(
            y_real, probabilidades
        ),

        "recall_ativo": recall_score(
            y_real,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),

        "recall_inativo": recall_score(
            y_real,
            previsoes,
            pos_label=0,
            zero_division=0,
        ),

        "taxa_falsos_positivos": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        ),
    }


# ============================================================
# ANALISAR OS GRUPOS SOCIOECONOMICOS
# ============================================================

def analisar_subgrupos(
    X_treino,
    y_treino,
    previsoes,
    nome_configuracao,
):
    """
    Examina os grupos usando as caracteristicas
    originais, mesmo quando o atributo foi retirado
    do modelo.

    Isso permite verificar se a retirada melhorou
    ou piorou o desempenho em cada grupo.
    """

    grupos = {
        "Renda ate R$ 500": (
            X_treino["renda_per_capita"] <= 500
        ),

        "Renda acima de R$ 3.000": (
            X_treino["renda_per_capita"] > 3000
        ),

        "Sem instrucao": (
            X_treino["escolaridade"] == "Sem instrução"
        ),

        "Fundamental incompleto": (
            X_treino["escolaridade"]
            == "Fundamental incompleto"
        ),

        "Superior completo": (
            X_treino["escolaridade"]
            == "Superior completo"
        ),

        "Area rural": (
            X_treino["area_residencia"] == "Rural"
        ),

        "Area urbana": (
            X_treino["area_residencia"] == "Urbana"
        ),
    }

    y_array = y_treino.to_numpy()
    registros = []

    for nome_grupo, mascara in grupos.items():

        indices = mascara.to_numpy()

        reais = y_array[indices]
        previstos = previsoes[indices]

        n = len(reais)

        if n == 0:
            continue

        ativos_reais = reais == 1
        inativos_reais = reais == 0

        n_ativos = int(ativos_reais.sum())
        n_inativos = int(inativos_reais.sum())

        vp = int(
            ((reais == 1) & (previstos == 1)).sum()
        )

        fp = int(
            ((reais == 0) & (previstos == 1)).sum()
        )

        recall_ativo = (
            100 * vp / n_ativos
            if n_ativos > 0
            else np.nan
        )

        taxa_falsos_positivos = (
            100 * fp / n_inativos
            if n_inativos > 0
            else np.nan
        )

        registros.append({
            "configuracao": nome_configuracao,
            "grupo": nome_grupo,
            "n_pessoas": n,
            "n_ativos_reais": n_ativos,
            "taxa_real_pct": 100 * ativos_reais.mean(),
            "taxa_prevista_pct": 100 * (previstos == 1).mean(),
            "recall_ativo_pct": recall_ativo,
            "falsos_positivos_pct": taxa_falsos_positivos,
        })

    return registros


# ============================================================
# EXECUTAR EXPERIMENTOS
# ============================================================

def executar():

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("TESTE DE COMBINACOES DE ATRIBUTOS")
    print("=" * 60)

    print("RobustScaler + One-Hot + kNN padrao")
    print("10 folds estratificados")
    print("Teste reservado de 20% NAO utilizado")
    print("Nenhum CSV anterior e necessario")
    print()

    base = pd.read_parquet(CAMINHO_BASE)

    X, y = separar_xy(base)

    X_treino, _, y_treino, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    cv = validacao_cruzada()

    # Mesmas particoes para todas as configuracoes.
    folds = list(
        cv.split(X_treino, y_treino)
    )

    resultados_folds = []
    resultados_gerais = []
    resultados_subgrupos = []

    total = len(CONFIGURACOES)

    for numero, (nome, retirados) in enumerate(
        CONFIGURACOES,
        start=1,
    ):

        print()
        print("=" * 60)
        print(f"[{numero}/{total}] {nome}")
        print(f"Atributos retirados: {retirados}")
        print("=" * 60)

        # Uma previsao para cada observacao de treino,
        # obtida apenas no fold em que ela foi validacao.
        previsoes_fora_amostra = np.full(
            len(X_treino),
            -1,
            dtype=int,
        )

        metricas_da_configuracao = []

        for numero_fold, (
            idx_treino,
            idx_validacao,
        ) in enumerate(folds, start=1):

            X_fold_treino = X_treino.iloc[idx_treino]
            y_fold_treino = y_treino.iloc[idx_treino]

            X_fold_validacao = X_treino.iloc[
                idx_validacao
            ]

            y_fold_validacao = y_treino.iloc[
                idx_validacao
            ]

            # Um novo modelo para cada fold.
            modelo = criar_modelo(retirados)

            modelo.fit(
                X_fold_treino,
                y_fold_treino,
            )

            previsoes = modelo.predict(
                X_fold_validacao
            )

            indice_classe_1 = list(
                modelo.classes_
            ).index(1)

            probabilidades = modelo.predict_proba(
                X_fold_validacao
            )[:, indice_classe_1]

            # Guardar previsoes fora da amostra.
            previsoes_fora_amostra[
                idx_validacao
            ] = previsoes

            metricas = calcular_metricas(
                y_fold_validacao,
                previsoes,
                probabilidades,
            )

            metricas["fold"] = numero_fold
            metricas["configuracao"] = nome

            resultados_folds.append(metricas)
            metricas_da_configuracao.append(metricas)

            print(
                f"  Fold {numero_fold}/10 concluido",
                flush=True,
            )

        if (previsoes_fora_amostra == -1).any():
            raise RuntimeError(
                "Existem observacoes sem previsao."
            )

        tabela_configuracao = pd.DataFrame(
            metricas_da_configuracao
        )

        colunas_metricas = [
            "acuracia",
            "acuracia_balanceada",
            "f1",
            "auc",
            "recall_ativo",
            "recall_inativo",
            "taxa_falsos_positivos",
        ]

        medias = tabela_configuracao[
            colunas_metricas
        ].mean()

        resultado = {
            "configuracao": nome,
            **medias.to_dict(),
        }

        resultados_gerais.append(resultado)

        grupos = analisar_subgrupos(
            X_treino,
            y_treino,
            previsoes_fora_amostra,
            nome,
        )

        resultados_subgrupos.extend(grupos)

        print()
        print("Metricas medias:")
        print(medias.round(4).to_string())

        # Salvar apos cada configuracao.
        pd.DataFrame(
            resultados_folds
        ).to_csv(
            PASTA_SAIDA / "METRICAS_POR_FOLD.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            resultados_gerais
        ).to_csv(
            PASTA_SAIDA / "COMPARACAO_PARCIAL.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            resultados_subgrupos
        ).to_csv(
            PASTA_SAIDA / "SUBGRUPOS_PARCIAL.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

        print("Resultados parciais salvos.")

    # ========================================================
    # COMPARACAO FINAL
    # ========================================================

    comparacao = pd.DataFrame(
        resultados_gerais
    )

    referencia = comparacao.loc[
        comparacao["configuracao"] == "nenhum"
    ].iloc[0]

    for metrica in [
        "f1",
        "auc",
        "recall_ativo",
        "acuracia_balanceada",
    ]:
        comparacao[
            f"delta_{metrica}"
        ] = (
            comparacao[metrica]
            - referencia[metrica]
        )

    print()
    print("=" * 65)
    print("COMPARACAO FINAL")
    print("=" * 65)

    print(
        comparacao[
            [
                "configuracao",
                "f1",
                "auc",
                "recall_ativo",
                "acuracia_balanceada",
                "taxa_falsos_positivos",
                "delta_f1",
                "delta_auc",
                "delta_recall_ativo",
            ]
        ].round(4).to_string(index=False)
    )

    comparacao.to_csv(
        PASTA_SAIDA / "COMPARACAO_FINAL.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # ========================================================
    # COMPARAR OS SUBGRUPOS COM A REFERENCIA
    # ========================================================

    subgrupos = pd.DataFrame(
        resultados_subgrupos
    )

    referencia_grupos = (
        subgrupos[
            subgrupos["configuracao"] == "nenhum"
        ]
        .set_index("grupo")["recall_ativo_pct"]
        .to_dict()
    )

    subgrupos["delta_recall_pp"] = (
        subgrupos["recall_ativo_pct"]
        - subgrupos["grupo"].map(referencia_grupos)
    )

    print()
    print("=" * 65)
    print("SUBGRUPOS - COMPARACAO DE RECALL")
    print("=" * 65)

    print(
        subgrupos[
            [
                "configuracao",
                "grupo",
                "recall_ativo_pct",
                "falsos_positivos_pct",
                "delta_recall_pp",
            ]
        ].round(2).to_string(index=False)
    )

    subgrupos.to_csv(
        PASTA_SAIDA / "SUBGRUPOS.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    print()
    print("=" * 60)
    print("EXPERIMENTO CONCLUIDO")
    print("=" * 60)

    print(f"Resultados salvos em: {PASTA_SAIDA.resolve()}")
    print("Conjunto de teste reservado NAO utilizado.")
    print("Nenhum codigo oficial foi alterado.")
    print()
    print(
        "ATENCAO: ganhos pequenos precisam de confirmacao. "
        "Nao escolher o modelo final apenas por esta comparacao."
    )


if __name__ == "__main__":
    executar()
