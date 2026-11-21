"""Testes básicos do módulo src/avaliacao.py."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from src.avaliacao import avaliar_modelo, avaliar_validacao_cruzada
from src.preprocessamento import CATEGORICAS, NUMERICAS
from src.treinamento import treinar_knn


def _base_sintetica(n=120):
    rng = np.random.default_rng(42)
    dados = {
        "idade": rng.integers(18, 80, size=n).astype(float),
        "renda_per_capita": rng.uniform(500, 10000, size=n),
        "numero_moradores": rng.integers(1, 6, size=n).astype(float),
    }
    for col in CATEGORICAS:
        dados[col] = rng.choice(["A", "B", "C"], size=n)

    X = pd.DataFrame(dados)[NUMERICAS + CATEGORICAS]
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]), name="ativo_lazer")
    return X, y


def main():
    X, y = _base_sintetica()

    #Testa avaliar_modelo isoladamente
    modelo = treinar_knn(X.iloc[:90], y.iloc[:90])
    metricas = avaliar_modelo(modelo, X.iloc[90:], y.iloc[90:])
    chaves_esperadas = {
        "acuracia", "acuracia_baseline", "precisao",
        "revocacao", "f1", "auc_roc", "vp", "fn", "fp", "vn",
    }
    assert set(metricas.keys()) == chaves_esperadas
    assert metricas["vp"] + metricas["fn"] + metricas["fp"] + metricas["vn"] == 30

    #Testa avaliar_validacao_cruzada (10 folds)
    resumo, df_folds = avaliar_validacao_cruzada(X, y, n_jobs=1)
    assert len(df_folds) == 10
    assert resumo["vp"] + resumo["fn"] + resumo["fp"] + resumo["vn"] == len(X)
    assert 0.0 <= resumo["f1_media"] <= 1.0
    assert 0.0 <= resumo["auc_roc_media"] <= 1.0

    print("Todos os testes de src/avaliacao.py passaram!")


if __name__ == "__main__":
    main()