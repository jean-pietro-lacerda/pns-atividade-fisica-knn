"""Diagnóstico local: taxa REAL vs PREVISTA de atividade física por grupo.

Coloque este arquivo na raiz do repositório pns-atividade-fisica-knn.
Execute: .venv/Scripts/python.exe comparar_taxas_pedro.py

Não modifica módulos oficiais. Não utiliza os 20% reservados para teste.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessamento import separar_xy, validacao_cruzada
from src.treinamento import treinar_knn

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / "data" / "interim" / "pns_base_eda.parquet"
SAIDA = RAIZ / "outputs" / "diagnostico_pedro" / "taxas_reais_vs_previstas"

VARIAVEIS = [
    "idade", "renda_per_capita", "numero_moradores", "sexo",
    "escolaridade", "uf", "situacao_trabalho", "area_residencia",
    "raca_cor", "estado_civil",
]


def construir_grupos(x):
    """Faixas fixas, apenas para análise; não são usadas para treinar o modelo."""
    grupos = {}
    grupos["idade"] = pd.cut(
        x["idade"],
        bins=[17, 29, 44, 59, 74, np.inf],
        labels=["18-29", "30-44", "45-59", "60-74", "75 ou mais"],
    )
    grupos["renda_per_capita"] = pd.cut(
        x["renda_per_capita"],
        bins=[-np.inf, 500, 1000, 2000, 3000, np.inf],
        labels=[
            "Até R$ 500", "Mais de R$ 500 a 1.000",
            "Mais de R$ 1.000 a 2.000", "Mais de R$ 2.000 a 3.000",
            "Acima de R$ 3.000",
        ],
    )
    grupos["numero_moradores"] = pd.cut(
        x["numero_moradores"],
        bins=[0, 2, 3, 4, np.inf],
        labels=["1-2 moradores", "3 moradores", "4 moradores", "5 ou mais"],
    )
    for variavel in VARIAVEIS:
        if variavel not in grupos:
            grupos[variavel] = x[variavel]
    return grupos


def calcular_tabela(dados, variavel):
    """Calcula taxas por grupo usando APENAS previsões fora da amostra (OOF)."""
    registros = []
    for grupo, parte in dados.groupby("grupo", sort=False, dropna=False):
        real = parte["real"].to_numpy(dtype=int)
        previsto = parte["previsto"].to_numpy(dtype=int)

        vp = int(np.sum((real == 1) & (previsto == 1)))
        fn = int(np.sum((real == 1) & (previsto == 0)))
        fp = int(np.sum((real == 0) & (previsto == 1)))
        vn = int(np.sum((real == 0) & (previsto == 0)))
        n = len(parte)
        n_ativos = vp + fn
        n_inativos = vn + fp

        registros.append({
            "variavel": variavel,
            "grupo": str(grupo),
            "n": n,
            "n_ativos_reais": n_ativos,
            "n_inativos_reais": n_inativos,
            "taxa_ativa_real_pct": round(100 * n_ativos / n, 2),
            "taxa_ativa_prevista_pct": round(100 * (vp + fp) / n, 2),
            # Negativo = o modelo prevê menos ativos do que existem no grupo.
            "diferenca_prevista_menos_real_pp": round(100 * ((vp + fp) - n_ativos) / n, 2),
            "recall_ativo_pct": round(100 * vp / n_ativos, 2) if n_ativos else np.nan,
            "falsos_positivos_entre_inativos_pct": (
                round(100 * fp / n_inativos, 2) if n_inativos else np.nan
            ),
            "vp": vp, "fn": fn, "fp": fp, "vn": vn,
            "cautela_menos_100_ativos_ou_inativos": (
                n_ativos < 100 or n_inativos < 100
            ),
        })
    return pd.DataFrame(registros)


def executar():
    if not BASE.exists():
        raise FileNotFoundError(
            f"Base não encontrada em {BASE}.\n"
            "Coloque o arquivo Python na pasta PRINCIPAL do projeto."
        )

    base = pd.read_parquet(BASE)
    X, y = separar_xy(base)

    # Mesma separação 80/20 dos seus testes anteriores.
    # O conjunto de teste fica reservado: nem é usado nesta análise.
    X_treino, _, y_treino, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_treino = X_treino.reset_index(drop=True)
    y_treino = y_treino.reset_index(drop=True)

    previsoes = np.full(len(X_treino), -1, dtype=np.int8)
    coberturas = np.zeros(len(X_treino), dtype=np.int8)
    cv = validacao_cruzada()

    print("DIAGNOSTICO: TAXA REAL vs TAXA PREVISTA POR GRUPO")
    print("RobustScaler + One-Hot + kNN com parametros padrao")
    print("10 folds estratificados (mesmos folds dos testes anteriores)")
    print("Conjunto de teste de 20%: NAO utilizado\n", flush=True)

    for numero, (indices_treino, indices_validacao) in enumerate(
        cv.split(X_treino, y_treino), start=1
    ):
        # treinar_knn cria e ajusta um pipeline NOVO só com este treino.
        modelo = treinar_knn(
            X_treino.iloc[indices_treino],
            y_treino.iloc[indices_treino],
            configuracao={"escala": "robust"},
        )
        previsoes[indices_validacao] = modelo.predict(
            X_treino.iloc[indices_validacao]
        )
        coberturas[indices_validacao] += 1
        print(f"Fold {numero}/10 concluido", flush=True)

    if not np.all(coberturas == 1) or np.any(previsoes == -1):
        raise RuntimeError("Erro: alguma linha nao foi validada exatamente uma vez.")

    real = y_treino.to_numpy(dtype=int)
    matriz = confusion_matrix(real, previsoes, labels=[0, 1])
    print("\nMatriz de confusao AGREGADA:")
    print(matriz)
    print(f"Acuracia: {accuracy_score(real, previsoes):.4f}")
    print(f"Acuracia balanceada: {balanced_accuracy_score(real, previsoes):.4f}")
    print(f"Taxa REAL de ativos: {100 * np.mean(real == 1):.2f}%")
    print(f"Taxa PREVISTA de ativos: {100 * np.mean(previsoes == 1):.2f}%")
    print("\nDiferenca negativa (prevista - real) = subprevisao da classe ativa.")

    SAIDA.mkdir(parents=True, exist_ok=True)
    grupos = construir_grupos(X_treino)
    todas_tabelas = []

    for variavel in VARIAVEIS:
        dados = pd.DataFrame({
            "grupo": grupos[variavel].astype("string").fillna("Ausente"),
            "real": real,
            "previsto": previsoes,
        })
        tabela = calcular_tabela(dados, variavel)
        todas_tabelas.append(tabela)
        tabela.to_csv(
            SAIDA / f"{variavel}_taxas.csv",
            sep=";", index=False, encoding="utf-8-sig",
        )

        # Mostra as 4 principais na tela; salva as 10 em CSV.
        if variavel in ("idade", "renda_per_capita", "escolaridade", "area_residencia"):
            print(f"\n--- {variavel.upper()} ---")
            colunas = [
                "grupo", "n", "taxa_ativa_real_pct",
                "taxa_ativa_prevista_pct", "diferenca_prevista_menos_real_pp",
                "recall_ativo_pct", "falsos_positivos_entre_inativos_pct",
            ]
            print(tabela[colunas].to_string(index=False))

    consolidado = pd.concat(todas_tabelas, ignore_index=True)
    consolidado.to_csv(
        SAIDA / "TODOS_OS_GRUPOS.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )

    nota = [
        "# Diagnostico exploratorio: taxa real vs prevista por grupo",
        "",
        "PNS 2019; 80% de treino; 20% de teste reservado e nao utilizado.",
        "kNN padrao; RobustScaler; One-Hot; StratifiedKFold(10), embaralhado, semente 42.",
        "Cada observacao foi prevista fora da amostra no seu fold de validacao.",
        "Faixas fixas sao APENAS para descricao; nao foram oferecidas ao modelo.",
        "",
        "Taxa ativa real = (VP + FN) / total do grupo.",
        "Taxa ativa prevista = (VP + FP) / total do grupo.",
        "Diferenca pp = taxa prevista menos taxa real; negativa significa subprevisao.",
        "Recall ativo = VP / (VP + FN).",
        "Taxa de falsos positivos = FP / (FP + VN).",
        "",
        "Importante: taxas proximas nao provam boas previsoes individuais.",
        "Diferenças por grupo nao demonstram por si so causalidade, vies indevido ou discriminacao.",
        "Grupos com menos de 100 ativos ou menos de 100 inativos exigem cautela.",
        "Analise exploratoria; nao substituir validacao final nem usar o conjunto de teste.",
        "",
        f"Total avaliado: {len(real)}",
        f"Matriz de confusao agregada: {matriz.tolist()}",
        f"Taxa real de ativos: {100 * np.mean(real == 1):.2f}%",
        f"Taxa prevista de ativos: {100 * np.mean(previsoes == 1):.2f}%",
        "",
        "Para ver os resultados, abra TODOS_OS_GRUPOS.csv ou cada CSV individual.",
    ]
    (SAIDA / "LEIA_ME_RESULTADOS.md").write_text("\n".join(nota), encoding="utf-8")

    print(f"\nConcluido! TODOS OS RESULTADOS: {SAIDA}")
    print("Abra TODOS_OS_GRUPOS.csv para conferir as 10 variaveis.")
    print("Nenhum codigo oficial foi modificado; o teste reservado permaneceu intacto.")


if __name__ == "__main__":
    executar()
