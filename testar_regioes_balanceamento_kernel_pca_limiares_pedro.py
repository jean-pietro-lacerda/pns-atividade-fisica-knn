"""DIAGNOSTICO LOCAL: regioes + oversampling + Kernel PCA (RBF) + limiares.

Repeticao do ultimo experimento: mesma divisao de desenvolvimento, 10 folds,
10 atributos, RobustScaler + One-Hot, oversampling apenas no treino e kNN
padrao (5 vizinhos, voto uniforme). Muda somente PCA linear para Kernel PCA.

IMPORTANTE: Kernel PCA RBF exato na base balanceada inteira pode requerer
centenas de GB de RAM. Aqui se usa KernelPCA ajustado em uma AMOSTRA
BALANCEADA de ate 768 exemplos do treino por fold, com transformacao em lotes.
Esta e uma APROXIMACAO POR SUBAMOSTRAGEM, nao Kernel PCA exato em toda a base.
Todos os exemplos balanceados sao utilizados para TREINAR o kNN, e todos os
exemplos de validacao sao avaliados. Nao usa o teste reservado.

Os modelos PCA linear e sem PCA sao recuperados do OOF da execucao anterior,
sem retreinamento. As comparacoes sao exploratorias e nao contam para os 150
experimentos oficiais sem autorizacao da professora.
"""

from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
PASTA = ROOT / "outputs" / "diagnostico_pedro"
ENTRADA = PASTA / "regioes_balanceamento_pca_limiares" / "PROBABILIDADES_OOF.csv"
SAIDA = PASTA / "regioes_balanceamento_kernel_pca_limiares"

COMPONENTES = [25, 20, 15, 10, 5]
LIMIARES = {1.0: 5, 0.8: 4, 0.5: 3, 0.4: 2, 0.2: 1}
MAX_LANDMARKS = 768  # 384 ativos e 384 inativos, quando disponiveis
TAMANHO_LOTE = 1024
SEMENTE = 42
# Escolha fixa, sem ajuste nos folds de validacao: gamma=1/31.
# Kernel RBF: K(a,b)=exp(-gamma*||a-b||^2).
GAMMA = 1.0 / 31.0
CHAVES = ["posicao_dev", "fold", "real"]
METRICAS = ["acuracia", "acuracia_balanceada", "precisao", "recall",
            "f1", "f2", "auc", "average_precision", "fpr", "fnr"]
MATRIZ = ["tn", "fp", "fn", "tp"]

REGIOES = {
    "Norte": [(11, "RO", "Rondonia"), (12, "AC", "Acre"),
              (13, "AM", "Amazonas"), (14, "RR", "Roraima"),
              (15, "PA", "Para"), (16, "AP", "Amapa"),
              (17, "TO", "Tocantins")],
    "Nordeste": [(21, "MA", "Maranhao"), (22, "PI", "Piaui"),
                 (23, "CE", "Ceara"), (24, "RN", "Rio Grande do Norte"),
                 (25, "PB", "Paraiba"), (26, "PE", "Pernambuco"),
                 (27, "AL", "Alagoas"), (28, "SE", "Sergipe"),
                 (29, "BA", "Bahia")],
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
    """Exatamente o mesmo sorteio do script PCA linear anterior."""
    rng = np.random.default_rng(SEMENTE + fold)
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
        raise ValueError("Probabilidades fora de [0,1] ou nao finitas.")
    votos = np.rint(p * 5).astype(int)
    if not np.allclose(p, votos / 5, atol=1e-8):
        raise ValueError("Esperados votos uniformes de kNN com 5 vizinhos.")
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


def selecionar_marcos_balanceados(ztr, yt, fold):
    """Subamostra 50/50 do treino real; nao incorpora o fold de validacao.

    A replicacao 50/50 para o kNN e feita separadamente e permanece identica
    a do experimento anterior. A selecao de marcos nao modifica y do kNN.
    """
    rng = np.random.default_rng(100_000 + SEMENTE + fold)
    y = np.asarray(yt, dtype=int)
    zeros, uns = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    quantia = min(MAX_LANDMARKS // 2, len(zeros), len(uns))
    if quantia <= max(COMPONENTES):
        raise ValueError(f"Treino pequeno demais para {max(COMPONENTES)} componentes Kernel PCA.")
    amostra = np.concatenate([
        rng.choice(zeros, size=quantia, replace=False),
        rng.choice(uns, size=quantia, replace=False),
    ])
    return np.ascontiguousarray(ztr[rng.permutation(amostra)], dtype=float)


def transformar_em_lotes(kpca, dados, tamanho_lote=TAMANHO_LOTE):
    """Evita alocar toda a matriz kernel n_amostras x n_marcos de uma vez."""
    n = len(dados)
    out = np.empty((n, max(COMPONENTES)), dtype=np.float32)
    for ini in range(0, n, tamanho_lote):
        fim = min(n, ini + tamanho_lote)
        emb = kpca.transform(np.asarray(dados[ini:fim], dtype=float))
        if not np.isfinite(emb).all():
            raise ValueError(f"Kernel PCA gerou numeros nao finitos nas linhas {ini}:{fim}.")
        out[ini:fim] = emb.astype(np.float32)
    return out


def executar():
    caminho = ROOT / "data" / "interim" / "pns_base_eda.parquet"
    if not caminho.is_file():
        raise FileNotFoundError(f"Base nao encontrada: {caminho}")
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f"Resultados anteriores nao encontrados: {ENTRADA}\n"
            "Execute primeiro testar_regioes_balanceamento_pca_limiares_pedro.py."
        )
    X, y = separar_xy(pd.read_parquet(caminho))
    if X.shape[1] != 10:
        raise ValueError(f"Esperados 10 atributos originais; recebidos {X.shape[1]}.")
    X, _, y, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE
    )
    R = agrupar_regioes(X)
    splits = list(StratifiedKFold(
        n_splits=10, shuffle=True, random_state=SEMENTE
    ).split(R, y))
    esperado = pd.concat([
        pd.DataFrame({"posicao_dev": va, "fold": fold,
                      "real": y.iloc[va].to_numpy(dtype=int)})
        for fold, (_, va) in enumerate(splits, 1)
    ], ignore_index=True).sort_values("posicao_dev").reset_index(drop=True)

    antigos = pd.read_csv(ENTRADA)
    referencias = [
        "regioes_sem_balanceamento_sem_pca",
        "regioes_balanceado_sem_pca",
        "regioes_balanceado_pca_25",
        "regioes_balanceado_pca_20",
        "regioes_balanceado_pca_15",
        "regioes_balanceado_pca_10",
        "regioes_balanceado_pca_5",
    ]
    registros = []
    for nome in referencias:
        g = (antigos[antigos["configuracao"] == nome]
             .sort_values("posicao_dev").reset_index(drop=True))
        if len(g) != len(esperado) or not np.array_equal(
            g[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()
        ):
            raise ValueError(f"OOF antigo ausente ou desalinhado: {nome}")
        registros.append(preparar_oof(g[CHAVES + ["prob_ativo"]].assign(configuracao=nome)))

    print(f"REGIOES + OVERSAMPLING + KERNEL PCA (RBF) + LIMIARES | {len(R)} dev | 10 folds")
    print("Referencia: 7 configuracoes antigas, sem retreinamento; 50 novos ajustes kNN")
    print(f"Kernel PCA APROXIMADO por ate {MAX_LANDMARKS} marcos balanceados por fold; "
          f"RBF gamma={GAMMA:.6f}")
    print("Nenhum dado sintetico: oversampling apenas replica exemplos reais do treino.")
    print("Kernel PCA ajustado SO no treino, validacao sem vazamento, teste reservado intacto.",
          flush=True)

    detalhes = []
    for fold, (tr, va) in enumerate(splits, 1):
        yt = y.iloc[tr].to_numpy(dtype=int)
        real = y.iloc[va].to_numpy(dtype=int)
        original = construir_pipeline(escala="robust")
        if [nome for nome, _ in original.steps][-2:] != ["pca", "knn"]:
            raise ValueError("O pipeline oficial mudou: verificar estrutura manualmente.")
        pre = original[:-2]
        ztr = np.asarray(pre.fit_transform(R.iloc[tr], yt), dtype=float)
        zva = np.asarray(pre.transform(R.iloc[va]), dtype=float)
        if ztr.shape[1] != 31:
            raise ValueError(f"Esperadas 31 colunas apos One-Hot; recebidas {ztr.shape[1]}.")
        if not np.isfinite(ztr).all() or not np.isfinite(zva).all():
            raise ValueError("O pre-processador retornou valores nao finitos.")

        indices = indices_oversampling(yt, fold)
        zbal = ztr[indices]
        ybal = yt[indices]
        if int((ybal == 0).sum()) != int((ybal == 1).sum()):
            raise ValueError("Balanceamento 50/50 nao foi obtido.")

        marcos = selecionar_marcos_balanceados(ztr, yt, fold)
        kpca = KernelPCA(
            n_components=max(COMPONENTES),
            kernel="rbf", gamma=GAMMA,
            eigen_solver="arpack", fit_inverse_transform=False,
            random_state=SEMENTE + fold,
            n_jobs=1,
        )
        kpca.fit(marcos)
        ptr = transformar_em_lotes(kpca, zbal)
        pva = transformar_em_lotes(kpca, zva)
        for n in COMPONENTES:
            nome = f"regioes_balanceado_kernel_rbf_{n}"
            knn = clone(original.named_steps["knn"])
            if (knn.n_neighbors != 5 or knn.weights != "uniform"
                    or knn.p != 2):
                raise ValueError("kNN oficial deveria ter 5 vizinhos, voto uniforme e p=2.")
            knn.fit(np.ascontiguousarray(ptr[:, :n]), ybal)
            prob = knn.predict_proba(np.ascontiguousarray(pva[:, :n]))[
                :, list(knn.classes_).index(1)
            ]
            g = preparar_oof(pd.DataFrame({
                "configuracao": nome, "posicao_dev": va,
                "fold": fold, "real": real, "prob_ativo": prob,
            }))
            registros.append(g)
            detalhes.append({
                "configuracao": nome, "fold": fold,
                "componentes": n, "marcos": len(marcos),
                "gamma": GAMMA,
                # Nao e a variancia explicada do PCA linear.
                "autovalor_kernel_n": float(kpca.eigenvalues_[n - 1]),
            })
            m05, m04 = avaliar(g, 3), avaliar(g, 2)
            print(
                f"Fold {fold}/10 | Kernel RBF {n:>2} | "
                f"marcos={len(marcos)} | AUC={m05['auc']:.4f} | "
                f"F1(0.5)={m05['f1']:.4f} | F1(0.4)={m04['f1']:.4f} | "
                f"Recall(0.4)={m04['recall']:.4f} | FPR(0.4)={m04['fpr']:.4f}",
                flush=True,
            )
        del ptr, pva, zbal, ztr, zva

    oof = pd.concat(registros, ignore_index=True)
    ordem = referencias + [f"regioes_balanceado_kernel_rbf_{n}" for n in COMPONENTES]
    linhas = []
    for nome in ordem:
        g = oof[oof.configuracao == nome].sort_values("posicao_dev").reset_index(drop=True)
        if len(g) != len(esperado) or not np.array_equal(
            g[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()
        ):
            raise ValueError(f"Resultados incompletos ou fora da ordem: {nome}")
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
    ref = "regioes_balanceado_sem_pca"
    delta = medias.sub(medias.loc[ref], level="limiar")
    pares = []
    for nome in ordem:
        if nome == ref:
            continue
        # PCA linear pareado com Kernel PCA na MESMA dimensionalidade.
        ref_pareada = (nome.replace("kernel_rbf", "pca")
                       if "kernel_rbf" in nome else ref)
        for limiar in LIMIARES:
            a = d[(d.configuracao == nome) & (d.limiar == limiar)].set_index("fold")
            b = d[(d.configuracao == ref_pareada) & (d.limiar == limiar)].set_index("fold")
            for metrica in ["auc", "average_precision", "f1", "recall",
                            "acuracia_balanceada", "fpr"]:
                diferenca = a[metrica] - b[metrica]
                melhora = -diferenca if metrica == "fpr" else diferenca
                pares.append({
                    "configuracao": nome, "referencia": ref_pareada,
                    "limiar": limiar, "metrica": metrica,
                    "delta_medio": diferenca.mean(), "dp_delta": diferenca.std(),
                    "melhores": int((melhora > 1e-12).sum()),
                    "piores": int((melhora < -1e-12).sum()),
                    "empates": int((np.abs(melhora) <= 1e-12).sum()),
                })

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
        ("DELTA_VS_BALANCEADO_SEM_PCA", delta),
        ("COMPARACAO_PAREADA", pd.DataFrame(pares)),
    ]
    for nome, tabela in tabelas:
        tabela.to_csv(SAIDA / f"{nome}.csv")
        if nome in ["RESUMO_PRINCIPAL", "MATRIZES_ACUMULADAS"]:
            print(f"\n=== {nome} ===\n{tabela.round(4).to_string()}")
    d.to_csv(SAIDA / "METRICAS_POR_FOLD.csv", index=False)
    pd.DataFrame(detalhes).to_csv(SAIDA / "DETALHES_KERNEL_POR_FOLD.csv", index=False)
    oof.sort_values(["configuracao", "posicao_dev"]).to_csv(
        SAIDA / "PROBABILIDADES_OOF.csv", index=False
    )
    print("\nLIMIARES: 1.0=5 votos; 0.8=4; 0.5=3; 0.4=2; 0.2=1.")
    print("Kernel PCA de amostra balanceada, nao exato na base inteira; sem variancia PCA linear.")
    print("AUC/AP nao mudam com limiar; limiares escolhidos exploratoriamente.")
    print("Comparacoes de folds dependentes; sem conclusao de generalizacao independente.")
    print("Diagnostico extra: teste reservado e arquivos oficiais intactos.")
    print("Resultados:", SAIDA)


if __name__ == "__main__":
    executar()
