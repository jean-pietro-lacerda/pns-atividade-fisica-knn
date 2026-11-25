r"""Experimento LOCAL: ablação das 10 variáveis do kNN da PNS 2019.

Execute na RAIZ do repositório:
    .venv\Scripts\python.exe ablacao_atributos_pedro.py

Treina 11 configurações: todas as variáveis (referência) e uma versão
sem cada um dos 10 atributos. Cada configuração usa os MESMOS 10 folds.
Não altera o código do grupo e NÃO utiliza os 20% reservados para teste.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessamento import (
    NUMERICAS,
    CATEGORICAS,
    construir_pipeline,
    separar_xy,
    validacao_cruzada,
)

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / "data" / "interim" / "pns_base_eda.parquet"
SAIDA = RAIZ / "outputs" / "diagnostico_pedro" / "ablacao_atributos"
ATRIBUTOS = list(NUMERICAS) + list(CATEGORICAS)


def criar_modelo(atributo_retirado):
    """Conserva kNN/Robust/One-Hot; exclui UMA coluna antes do ajuste."""
    modelo = construir_pipeline(escala="robust")
    if atributo_retirado is None:
        return modelo

    if atributo_retirado not in ATRIBUTOS:
        raise ValueError(f"Atributo desconhecido: {atributo_retirado}")

    original = modelo.named_steps["preprocessamento"]
    novos_transformadores = []
    for nome, transformador, colunas in original.transformers:
        mantidas = [c for c in colunas if c != atributo_retirado]
        novos_transformadores.append((nome, transformador, mantidas))

    # A variável retirada não é passada a nenhum transformador.
    # Reutiliza as transformações originais nos demais atributos.
    modelo.set_params(
        preprocessamento=ColumnTransformer(
            novos_transformadores, remainder="drop"
        )
    )
    return modelo


def metricas(y_real, y_previsto, prob_classe_1):
    return {
        "acuracia": accuracy_score(y_real, y_previsto),
        "acuracia_balanceada": balanced_accuracy_score(y_real, y_previsto),
        "precisao_ativo": precision_score(
            y_real, y_previsto, pos_label=1, zero_division=0
        ),
        "recall_ativo": recall_score(
            y_real, y_previsto, pos_label=1, zero_division=0
        ),
        "recall_inativo": recall_score(
            y_real, y_previsto, pos_label=0, zero_division=0
        ),
        "f1": f1_score(y_real, y_previsto, pos_label=1, zero_division=0),
        "f2": fbeta_score(
            y_real, y_previsto, beta=2, pos_label=1, zero_division=0
        ),
        "auc": roc_auc_score(y_real, prob_classe_1),
    }


def resumo_subgrupos(X, y, previsto, configuracao):
    """Diagnósticos DESCRITIVOS dos grupos, inclusive quando atributo foi retirado.

    Os grupos abaixo vêm das características ORIGINAIS, nunca entram no treino
    como faixas novas. Recall e FPR condicionados à classe real dentro do grupo.
    """
    grupos = {
        "renda_ate_500": X["renda_per_capita"].le(500).to_numpy(),
        "renda_acima_3000": X["renda_per_capita"].gt(3000).to_numpy(),
        "fundamental_incompleto": X["escolaridade"].eq(
            "Fundamental incompleto"
        ).to_numpy(),
        "superior_completo": X["escolaridade"].eq(
            "Superior completo"
        ).to_numpy(),
        "area_rural": X["area_residencia"].eq("Rural").to_numpy(),
        "area_urbana": X["area_residencia"].eq("Urbana").to_numpy(),
    }
    linhas = []
    for nome, mascara in grupos.items():
        r = y[mascara]
        p = previsto[mascara]
        vp = int(np.sum((r == 1) & (p == 1)))
        fn = int(np.sum((r == 1) & (p == 0)))
        fp = int(np.sum((r == 0) & (p == 1)))
        vn = int(np.sum((r == 0) & (p == 0)))
        n = len(r)
        linhas.append({
            "configuracao": configuracao,
            "grupo": nome,
            "n": n,
            "n_ativos": vp + fn,
            "taxa_real_ativa_pct": round(100 * (vp + fn) / n, 2) if n else np.nan,
            "taxa_prevista_ativa_pct": round(100 * (vp + fp) / n, 2) if n else np.nan,
            "recall_ativo_pct": round(100 * vp / (vp + fn), 2) if vp + fn else np.nan,
            "fpr_inativos_pct": round(100 * fp / (fp + vn), 2) if fp + vn else np.nan,
            "vp": vp, "fn": fn, "fp": fp, "vn": vn,
        })
    return linhas


def executar():
    if not BASE.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {BASE}. Coloque este script na raiz do projeto."
        )
    base = pd.read_parquet(BASE)
    X, y = separar_xy(base)

    # Reproduz a divisão original. Teste de 20% fica completamente reservado.
    X_treino, _, y_treino, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_treino = X_treino.reset_index(drop=True)
    y_treino = y_treino.reset_index(drop=True)
    y_real = y_treino.to_numpy(dtype=int)

    cv = list(validacao_cruzada().split(X_treino, y_treino))
    configuracoes = [("todas_as_10", None)] + [
        (f"sem_{atributo}", atributo) for atributo in ATRIBUTOS
    ]
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas_folds = []
    linhas_resumo = []
    linhas_grupos = []

    print("ABLACAO DE ATRIBUTOS: UM POR VEZ")
    print("Referencia: 10 atributos; mais 10 modelos, cada um sem um atributo")
    print("RobustScaler + One-Hot + kNN padrao")
    print("10 folds estratificados e identicos para todas as configuracoes")
    print("Conjunto de teste de 20%: NAO utilizado")
    print("Total de ajustes previstos: 110\n", flush=True)

    for pos, (nome, retirado) in enumerate(configuracoes, start=1):
        print(f"\n[{pos}/11] {nome}", flush=True)
        previsoes_oof = np.full(len(X_treino), -1, dtype=int)
        probabilidades_oof = np.full(len(X_treino), np.nan, dtype=float)
        cobertura = np.zeros(len(X_treino), dtype=np.int8)
        metricas_desta_configuracao = []

        for numero, (idx_t, idx_v) in enumerate(cv, start=1):
            modelo = criar_modelo(retirado)
            modelo.fit(X_treino.iloc[idx_t], y_treino.iloc[idx_t])

            # Probabilidades calculadas UMA VEZ por fold, para evitar
            # refazer a busca dos vizinhos em predict() e predict_proba().
            probabilidades = modelo.predict_proba(X_treino.iloc[idx_v])
            classes = np.asarray(modelo.classes_)
            indice_1 = int(np.flatnonzero(classes == 1)[0])
            p1 = probabilidades[:, indice_1]
            # Com k=5 padrão (ímpar) e pesos uniformes, argmax equivale
            # à votação majoritária usada em KNeighborsClassifier.predict.
            previstos = classes[np.argmax(probabilidades, axis=1)].astype(int)

            previsoes_oof[idx_v] = previstos
            probabilidades_oof[idx_v] = p1
            cobertura[idx_v] += 1

            resultado = metricas(y_real[idx_v], previstos, p1)
            metricas_desta_configuracao.append(resultado)
            linhas_folds.append({
                "configuracao": nome, "atributo_retirado": retirado or "nenhum",
                "fold": numero, **resultado,
            })
            print(f"  Fold {numero}/10 concluido", flush=True)

        if not np.all(cobertura == 1) or np.any(previsoes_oof == -1):
            raise RuntimeError("Alguma linha não recebeu uma previsão OOF válida.")

        medias = pd.DataFrame(metricas_desta_configuracao).mean()
        desvios = pd.DataFrame(metricas_desta_configuracao).std(ddof=1)
        matriz = confusion_matrix(y_real, previsoes_oof, labels=[0, 1])

        linha_resumo = {
            "configuracao": nome,
            "atributo_retirado": retirado or "nenhum",
            **medias.to_dict(),
            **{f"desvio_{k}": v for k, v in desvios.to_dict().items()},
            "vp_total": int(matriz[1, 1]),
            "fn_total": int(matriz[1, 0]),
            "fp_total": int(matriz[0, 1]),
            "vn_total": int(matriz[0, 0]),
            "taxa_prevista_ativa_pct": 100 * float((previsoes_oof == 1).mean()),
        }
        linhas_resumo.append(linha_resumo)
        linhas_grupos.extend(
            resumo_subgrupos(X_treino, y_real, previsoes_oof, nome)
        )

        # Salva após CADA configuração: resultados parciais disponíveis
        # mesmo que a execução seja interrompida depois.
        pd.DataFrame(linhas_resumo).to_csv(
            SAIDA / "COMPARACAO_ABLACAO.csv",
            sep=";", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(linhas_folds).to_csv(
            SAIDA / "METRICAS_POR_FOLD.csv",
            sep=";", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(linhas_grupos).to_csv(
            SAIDA / "GRUPOS_SENSIVEIS.csv",
            sep=";", index=False, encoding="utf-8-sig"
        )
        print(
            f"  F1={medias['f1']:.4f} | AUC={medias['auc']:.4f} | "
            f"Recall ativo={medias['recall_ativo']:.4f} | "
            f"Recall inativo={medias['recall_inativo']:.4f}",
            flush=True,
        )

    resumo = pd.DataFrame(linhas_resumo)
    referencia = resumo.iloc[0]
    for metrica in ("acuracia_balanceada", "f1", "auc", "recall_ativo", "recall_inativo"):
        resumo[f"delta_{metrica}"] = resumo[metrica] - referencia[metrica]
    resumo.to_csv(
        SAIDA / "COMPARACAO_ABLACAO.csv",
        sep=";", index=False, encoding="utf-8-sig"
    )

    grupos = pd.DataFrame(linhas_grupos)
    ref_grupos = grupos[grupos["configuracao"] == "todas_as_10"].set_index("grupo")
    grupos["delta_recall_ativo_pp"] = grupos.apply(
        lambda r: r["recall_ativo_pct"] - ref_grupos.loc[r["grupo"], "recall_ativo_pct"],
        axis=1,
    )
    grupos["delta_fpr_pp"] = grupos.apply(
        lambda r: r["fpr_inativos_pct"] - ref_grupos.loc[r["grupo"], "fpr_inativos_pct"],
        axis=1,
    )
    grupos.to_csv(
        SAIDA / "GRUPOS_SENSIVEIS.csv",
        sep=";", index=False, encoding="utf-8-sig"
    )

    mostrar = [
        "atributo_retirado", "acuracia_balanceada", "f1", "auc",
        "recall_ativo", "recall_inativo", "delta_f1",
        "delta_recall_ativo", "delta_auc",
    ]
    print("\n=== COMPARACAO FINAL (medias dos mesmos 10 folds) ===")
    print(resumo[mostrar].round(4).to_string(index=False))
    print("\nDelta = metrica do modelo sem o atributo MENOS a referencia com 10 atributos.")
    print("Delta positivo de recall nao implica melhora geral: confira FPR/F1/AUC.")
    print(f"\nArquivos CSV salvos em: {SAIDA}")
    print("Nenhum arquivo oficial modificado; teste reservado permanece intocado.")


if __name__ == "__main__":
    executar()
