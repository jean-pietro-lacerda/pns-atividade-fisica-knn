"""Experimento LOCAL: 5 regioes + oversampling + PCA + cortes de decisao + kNN padrao.

Coloque este arquivo na raiz do projeto, ao lado de testar_regioes_balanceamento_pedro.py.
Nao altera nenhum arquivo oficial. Teste reservado nunca utilizado.
"""
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
PASTA = ROOT / "outputs" / "diagnostico_pedro"
ENTRADA = PASTA / "regioes_balanceamento" / "PROBABILIDADES_OOF.csv"
SAIDA = PASTA / "regioes_balanceamento_pca_limiares"
COMPONENTES = [25, 20, 15, 10, 5]
# Com cinco vizinhos e votos uniformes, estes sao todos os cortes distintos.
LIMIARES = {1.0: 5, 0.8: 4, 0.5: 3, 0.4: 2, 0.2: 1}
CHAVES = ["posicao_dev", "fold", "real"]
METRICAS = ["acuracia", "acuracia_balanceada", "precisao", "recall",
            "f1", "f2", "auc", "average_precision", "fpr", "fnr"]
MATRIZ = ["tn", "fp", "fn", "tp"]

REGIOES = {
    "Norte": [(11, "RO", "Rondonia"), (12, "AC", "Acre"), (13, "AM", "Amazonas"),
              (14, "RR", "Roraima"), (15, "PA", "Para"), (16, "AP", "Amapa"),
              (17, "TO", "Tocantins")],
    "Nordeste": [(21, "MA", "Maranhao"), (22, "PI", "Piaui"), (23, "CE", "Ceara"),
                 (24, "RN", "Rio Grande do Norte"), (25, "PB", "Paraiba"),
                 (26, "PE", "Pernambuco"), (27, "AL", "Alagoas"),
                 (28, "SE", "Sergipe"), (29, "BA", "Bahia")],
    "Sudeste": [(31, "MG", "Minas Gerais"), (32, "ES", "Espirito Santo"),
                (33, "RJ", "Rio de Janeiro"), (35, "SP", "Sao Paulo")],
    "Sul": [(41, "PR", "Parana"), (42, "SC", "Santa Catarina"),
            (43, "RS", "Rio Grande do Sul")],
    "Centro-Oeste": [(50, "MS", "Mato Grosso do Sul"),
                     (51, "MT", "Mato Grosso"), (52, "GO", "Goias"),
                     (53, "DF", "Distrito Federal")],
}


def normalizar(valor):
    s = unicodedata.normalize("NFKD", str(valor).strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


MAPA = {
    normalizar(valor): regiao
    for regiao, estados in REGIOES.items()
    for codigo, sigla, nome in estados
    for valor in (codigo, float(codigo), sigla, nome)
}


def agrupar_regioes(X):
    X = X.copy()
    uf = X["uf"]
    regiao = uf.map(lambda v: MAPA.get(normalizar(v)) if pd.notna(v) else np.nan)
    invalidos = uf[uf.notna() & regiao.isna()].unique()
    if len(invalidos):
        raise ValueError(f"UFs nao reconhecidas: {invalidos}")
    X["uf"] = regiao.astype(object).where(regiao.notna(), np.nan)
    return X


def indices_oversampling(y_treino, fold):
    """Replica positivos ate igualar negativos; MESMA regra do experimento anterior."""
    rng = np.random.default_rng(42 + fold)
    y = np.asarray(y_treino)
    maior, menor = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(maior) < len(menor):
        maior, menor = menor, maior
    extra = rng.choice(menor, len(maior) - len(menor), replace=True)
    return rng.permutation(np.concatenate([np.arange(len(y)), extra]))


def preparar_oof(g):
    g = g.copy()
    p = g["prob_ativo"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Escores invalidos: NaN, infinito ou fora de [0, 1].")
    votos = np.rint(p * 5).astype(int)
    if not np.allclose(p, votos / 5, atol=1e-8):
        raise ValueError("Esperados escores de kNN com 5 vizinhos e voto uniforme.")
    g["votos"] = votos
    return g


def avaliar(g, votos_minimos):
    real = g["real"].to_numpy(dtype=int)
    prob = g["prob_ativo"].to_numpy(dtype=float)
    predito = (g["votos"].to_numpy(dtype=int) >= votos_minimos).astype(int)
    tn, fp, fn, tp = confusion_matrix(real, predito, labels=[0, 1]).ravel()
    return {
        "acuracia": accuracy_score(real, predito),
        "acuracia_balanceada": balanced_accuracy_score(real, predito),
        "precisao": precision_score(real, predito, zero_division=0),
        "recall": recall_score(real, predito, zero_division=0),
        "f1": f1_score(real, predito, zero_division=0),
        "f2": fbeta_score(real, predito, beta=2, zero_division=0),
        "auc": roc_auc_score(real, prob),
        "average_precision": average_precision_score(real, prob),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def executar():
    caminho = ROOT / "data" / "interim" / "pns_base_eda.parquet"
    if not caminho.is_file():
        raise FileNotFoundError(f"Base nao encontrada: {caminho}")
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f"OOF anterior nao encontrado: {ENTRADA}\n"
            "Execute primeiro testar_regioes_balanceamento_pedro.py."
        )

    X, y = separar_xy(pd.read_parquet(caminho))
    if X.shape[1] != 10:
        raise ValueError(f"Esperados 10 atributos originais; recebidos {X.shape[1]}.")
    X, _, y, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    R = agrupar_regioes(X)
    splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(R, y))
    esperado = pd.concat([
        pd.DataFrame({"posicao_dev": va, "fold": fold,
                      "real": y.iloc[va].to_numpy(dtype=int)})
        for fold, (_, va) in enumerate(splits, 1)
    ], ignore_index=True).sort_values("posicao_dev").reset_index(drop=True)

    anterior = pd.read_csv(ENTRADA)
    registros = []
    referencias = [
        ("regioes_sem_balanceamento_sem_pca", "regioes"),
        ("regioes_balanceado_sem_pca", "regioes_balanceado"),
    ]
    for nome_novo, nome_antigo in referencias:
        g = (anterior[anterior.configuracao == nome_antigo]
             .sort_values("posicao_dev").reset_index(drop=True))
        if not np.array_equal(g[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
            raise ValueError(f"OOF da referencia desalinhado ou incompleto: {nome_antigo}")
        registros.append(preparar_oof(g[CHAVES + ["prob_ativo"]].assign(configuracao=nome_novo)))

    print(f"REGIOES + OVERSAMPLING + PCA + LIMIAR | {len(R)} registros | 10 folds")
    print("Referencia balanceada sem PCA recuperada | 50 novos ajustes")
    print("Sem dados sinteticos: oversampling por replicacao. Teste reservado intacto.", flush=True)

    variancias = []
    for fold, (tr, va) in enumerate(splits, 1):
        yt = y.iloc[tr].to_numpy(dtype=int)
        real = y.iloc[va].to_numpy(dtype=int)
        original = construir_pipeline(escala="robust")
        if [nome for nome, _ in original.steps][-2:] != ["pca", "knn"]:
            raise ValueError("Estrutura do pipeline oficial mudou; verifique manualmente.")
        pre = original[:-2]
        # Imputacao/escala/One-Hot aprendidos APENAS no treino real do fold.
        ztr = np.asarray(pre.fit_transform(R.iloc[tr], yt), dtype=float)
        zva = np.asarray(pre.transform(R.iloc[va]), dtype=float)
        if ztr.shape[1] != 31:
            raise ValueError(f"Esperadas 31 dimensoes apos encoding; obtidas {ztr.shape[1]}.")

        # Oversampling APENAS no treino, ANTES de ajustar o PCA.
        indices = indices_oversampling(yt, fold)
        ztr_balanceado = ztr[indices]
        y_balanceado = yt[indices]
        assert (y_balanceado == 0).sum() == (y_balanceado == 1).sum()

        # Um PCA de 25 componentes por fold, reutilizado para 5 cortes dimensionais.
        # O PCA aprende a geometria BALANCEADA, nunca ve o fold de validacao.
        pca = PCA(n_components=max(COMPONENTES), svd_solver="full", whiten=False)
        ptr = pca.fit_transform(ztr_balanceado)
        pva = pca.transform(zva)
        explicada = np.cumsum(pca.explained_variance_ratio_)

        for n in COMPONENTES:
            nome = f"regioes_balanceado_pca_{n}"
            knn = clone(original.named_steps["knn"])
            if knn.n_neighbors != 5 or knn.weights != "uniform":
                raise ValueError("kNN deveria usar 5 vizinhos e pesos uniformes.")
            knn.fit(np.ascontiguousarray(ptr[:, :n]), y_balanceado)
            classe_ativa = list(knn.classes_).index(1)
            prob = knn.predict_proba(np.ascontiguousarray(pva[:, :n]))[:, classe_ativa]
            g = preparar_oof(pd.DataFrame({
                "configuracao": nome, "posicao_dev": va,
                "fold": fold, "real": real, "prob_ativo": prob,
            }))
            registros.append(g)
            variancias.append({
                "configuracao": nome, "fold": fold,
                "componentes": n, "variancia_explicada": explicada[n - 1],
            })
            m05 = avaliar(g, 3)
            m04 = avaliar(g, 2)
            print(
                f"Fold {fold}/10 | PCA {n:>2} | Var={explicada[n-1]:.2%} | "
                f"AUC={m05['auc']:.4f} | "
                f"F1(0.5)={m05['f1']:.4f} | "
                f"F1(0.4)={m04['f1']:.4f} | "
                f"Recall(0.4)={m04['recall']:.4f} | "
                f"FPR(0.4)={m04['fpr']:.4f}",
                flush=True,
            )

    oof = pd.concat(registros, ignore_index=True)
    ordem = [nome for nome, _ in referencias] + [
        f"regioes_balanceado_pca_{n}" for n in COMPONENTES
    ]
    linhas = []
    for nome in ordem:
        g = oof[oof.configuracao == nome].sort_values("posicao_dev").reset_index(drop=True)
        if not np.array_equal(g[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
            raise ValueError(f"OOF desalinhado/incompleto: {nome}")
        for limiar, minimo in LIMIARES.items():
            for fold, parte in g.groupby("fold", sort=True):
                linhas.append({"configuracao": nome, "limiar": limiar,
                               "fold": int(fold), **avaliar(parte, minimo)})

    d = pd.DataFrame(linhas)
    gb = d.groupby(["configuracao", "limiar"], sort=False)
    medias = gb[METRICAS].mean()
    dp = gb[METRICAS].std()
    acumuladas = gb[MATRIZ].sum().astype(int)
    matrizes_medias = gb[MATRIZ].mean()
    referencia = "regioes_balanceado_sem_pca"
    delta = medias.sub(medias.loc[referencia], level="limiar")
    pares = []
    for nome in ordem:
        if nome == referencia:
            continue
        for limiar in LIMIARES:
            a = d[(d.configuracao == nome) & (d.limiar == limiar)].set_index("fold")
            b = d[(d.configuracao == referencia) & (d.limiar == limiar)].set_index("fold")
            for metrica in ["auc", "average_precision", "f1", "recall", "acuracia_balanceada", "fpr"]:
                diferenca = a[metrica] - b[metrica]
                melhora = -diferenca if metrica == "fpr" else diferenca
                pares.append({
                    "configuracao": nome, "limiar": limiar, "metrica": metrica,
                    "delta_medio": diferenca.mean(), "dp_delta": diferenca.std(),
                    "melhores": int((melhora > 1e-12).sum()),
                    "piores": int((melhora < -1e-12).sum()),
                    "empates": int((np.abs(melhora) <= 1e-12).sum()),
                })

    variancias_df = pd.DataFrame(variancias)
    var_media = variancias_df.groupby("configuracao", sort=False)[
        ["componentes", "variancia_explicada"]
    ].mean()
    # Uma tabela compacta, para facilitar a leitura no terminal.
    principais = medias.loc[(slice(None), [0.5, 0.4, 0.2]), [
        "acuracia", "precisao", "recall", "f1", "auc", "average_precision", "fpr"
    ]]

    SAIDA.mkdir(parents=True, exist_ok=True)
    tabelas = [
        ("RESUMO_PRINCIPAL", principais),
        ("COMPARACAO_FINAL", medias),
        ("DESVIOS_PADRAO", dp),
        ("MATRIZES_ACUMULADAS", acumuladas),
        ("MATRIZES_MEDIAS", matrizes_medias),
        ("VARIANCIA_EXPLICADA", var_media),
        ("DELTA_VS_BALANCEADO_SEM_PCA", delta),
        ("COMPARACAO_PAREADA", pd.DataFrame(pares)),
    ]
    for nome, tabela in tabelas:
        tabela.to_csv(SAIDA / f"{nome}.csv")
        if nome in ["RESUMO_PRINCIPAL", "MATRIZES_ACUMULADAS", "VARIANCIA_EXPLICADA"]:
            print(f"\n=== {nome} ===\n{tabela.round(4).to_string()}")
    d.to_csv(SAIDA / "METRICAS_POR_FOLD.csv", index=False)
    variancias_df.to_csv(SAIDA / "VARIANCIA_POR_FOLD.csv", index=False)
    oof.sort_values(["configuracao", "posicao_dev"]).to_csv(
        SAIDA / "PROBABILIDADES_OOF.csv", index=False
    )
    print("\nLIMIARES: 1.0=5 votos; 0.8=4; 0.5=3; 0.4=2; 0.2=1.")
    print("PCA ajustado apos replicar positivos, somente com os dados do treino.")
    print("AUC/AP nao variam com o limiar da mesma configuracao.")
    print("Exploratorio; folds dependentes. Sem escolha definitiva de limiar.")
    print("Teste reservado e arquivos oficiais intactos.")
    print("Resultados:", SAIDA)


if __name__ == "__main__":
    executar()
