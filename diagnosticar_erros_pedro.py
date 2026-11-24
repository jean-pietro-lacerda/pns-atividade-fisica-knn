"""Diagnóstico local de VP vs FN na PNS 2019.

Uso: salve este arquivo na raiz do repositório e execute
  .venv/Scripts/python.exe diagnosticar_erros_pedro.py

Não altera os módulos oficiais nem acessa o conjunto de teste reservado.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from src.preprocessamento import separar_xy, validacao_cruzada
from src.treinamento import treinar_knn

ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "data" / "interim" / "pns_base_eda.parquet"
SAIDA = ROOT / "outputs" / "diagnostico_pedro"

NUMERICAS = ["idade", "renda_per_capita", "numero_moradores"]
CATEGORICAS = [
    "sexo", "escolaridade", "uf", "situacao_trabalho",
    "area_residencia", "raca_cor", "estado_civil",
]


def tabela_grupos(ativos, coluna, tipo):
    """Recall condicionado à classe real 1 para cada grupo do atributo."""
    if tipo == "numerica":
        valores = pd.to_numeric(ativos[coluna], errors="coerce")
        if valores.nunique(dropna=True) < 2:
            grupo = valores.astype("string").fillna("Ausente")
        else:
            # Faixas definidas somente para DESCRIÇÃO das previsões fora da amostra.
            # Não há transformação de X nem reutilização desses limites no modelo.
            grupo = pd.qcut(valores, q=5, duplicates="drop")
            grupo = grupo.astype("string").fillna("Ausente")
    else:
        grupo = ativos[coluna].astype("string").fillna("Ausente")

    tab = pd.DataFrame({
        "grupo": grupo.to_numpy(),
        "acerto": ativos["acertou_classe_1"].to_numpy(),
    })
    resumo = (
        tab.groupby("grupo", dropna=False)
        .agg(n_ativos=("acerto", "size"), vp=("acerto", "sum"))
        .reset_index()
    )
    resumo["fn"] = resumo["n_ativos"] - resumo["vp"]
    resumo["recall_classe_1"] = (resumo["vp"] / resumo["n_ativos"]).round(4)
    resumo["percentual_dos_ativos"] = (
        100 * resumo["n_ativos"] / resumo["n_ativos"].sum()
    ).round(2)
    resumo["amostra_pequena_menos_100"] = resumo["n_ativos"] < 100
    return resumo.sort_values(["n_ativos", "grupo"], ascending=[False, True])


def main():
    if not DADOS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DADOS}. "
            "Salve este script na raiz do projeto."
        )

    base = pd.read_parquet(DADOS)
    X, y = separar_xy(base)

    # Mesma divisão estratificada de 80/20 dos diagnósticos anteriores.
    # Os 20% reservados NÃO participam do diagnóstico.
    X_treino, _, y_treino, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Cada previsão é produzida por um modelo que não viu a linha prevista.
    previsoes_oof = np.full(len(y_treino), -1, dtype=np.int8)
    contagem_validacoes = np.zeros(len(y_treino), dtype=np.int8)
    cv = validacao_cruzada()

    print("Diagnóstico: RobustScaler + One-Hot + kNN padrão")
    print("Validação cruzada estratificada: 10 folds")
    print("Conjunto de teste reservado: NÃO utilizado\n")

    for n, (idx_fit, idx_val) in enumerate(cv.split(X_treino, y_treino), 1):
        modelo = treinar_knn(
            X_treino.iloc[idx_fit],
            y_treino.iloc[idx_fit],
            configuracao={"escala": "robust"},
        )
        previsoes_oof[idx_val] = modelo.predict(X_treino.iloc[idx_val])
        contagem_validacoes[idx_val] += 1
        print(f"Fold {n}/10 concluído", flush=True)

    if not np.all(contagem_validacoes == 1):
        raise RuntimeError("Cada registro deveria ser validado exatamente uma vez.")

    resultado = X_treino.copy()
    resultado["real"] = y_treino.to_numpy(dtype=int)
    resultado["previsto"] = previsoes_oof

    matriz = confusion_matrix(
        resultado["real"], resultado["previsto"], labels=[0, 1]
    )
    print("\nMatriz de confusão agregada dos 10 folds (não é média):")
    print(matriz)

    ativos = resultado.loc[resultado["real"] == 1].copy()
    ativos["acertou_classe_1"] = (ativos["previsto"] == 1).astype(int)
    ativos["situacao"] = np.where(
        ativos["acertou_classe_1"] == 1, "VP", "FN"
    )

    print(f"\nAtivos reais avaliados: {len(ativos)}")
    print(f"Verdadeiros positivos (VP): {(ativos['situacao'] == 'VP').sum()}")
    print(f"Falsos negativos (FN): {(ativos['situacao'] == 'FN').sum()}")
    print(
        "Recall agregado da classe ativa: "
        f"{recall_score(resultado['real'], resultado['previsto'], pos_label=1):.4f}"
    )

    SAIDA.mkdir(parents=True, exist_ok=True)
    texto_relatorio = [
        "# Diagnóstico exploratório de erros: VP versus FN",
        "",
        "Dados: PNS 2019, 80% de treino; teste reservado não utilizado.",
        "kNN padrão, RobustScaler, One-Hot, previsões fora da amostra de 10 folds.",
        "Tabelas descrevem exclusivamente os indivíduos com classe real 1.",
        "Recall por grupo = VP / (VP + FN) dentro do grupo.",
        "Números de grupos pequenos (menos de 100 ativos) são especialmente instáveis.",
        "Diferenças descritivas não provam causa, discriminação ou efeito de atributos.",
        "",
        "## Matriz de confusão agregada",
        "",
        "```",
        str(matriz),
        "```",
        "",
    ]

    for coluna in NUMERICAS + CATEGORICAS:
        print(f"\n{'=' * 65}\nVARIÁVEL: {coluna}\n{'=' * 65}")
        texto_relatorio += [f"## {coluna}", ""]
        if coluna in NUMERICAS:
            resumo_numerico = (
                ativos.groupby("situacao")[coluna]
                .agg(["count", "mean", "median", "std", "min", "max"])
                .round(3)
            )
            print("\nResumo numérico dos ativos acertados (VP) e errados (FN):")
            print(resumo_numerico.to_string())
            resumo_numerico.to_csv(
                SAIDA / f"{coluna}_vp_vs_fn_resumo.csv", encoding="utf-8-sig"
            )
            texto_relatorio += ["Resumo de valores:", "", "```", resumo_numerico.to_string(), "```", ""]
            tipo = "numerica"
        else:
            tipo = "categorica"

        tabela = tabela_grupos(ativos, coluna, tipo)
        print("\nRecall da classe ativa por grupo (somente classe real = 1):")
        print(tabela.to_string(index=False))
        print("\nNota: grupos com menos de 100 ativos requerem cautela.")
        tabela.to_csv(
            SAIDA / f"{coluna}_recall_por_grupo.csv",
            index=False,
            encoding="utf-8-sig",
        )
        texto_relatorio += ["Recall por grupo:", "", "```", tabela.to_string(index=False), "```", ""]

    caminho_relatorio = SAIDA / "relatorio_diagnostico_erros.md"
    caminho_relatorio.write_text("\n".join(texto_relatorio), encoding="utf-8")

    print(f"\nConcluído! Tabelas e relatório salvos em: {SAIDA}")
    print("Não houve ajuste do modelo com dados de validação nem uso do teste.")
    print("Nenhum arquivo do código oficial foi alterado.")


if __name__ == "__main__":
    main()
