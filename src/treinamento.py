
import pandas as pd

from src.preprocessamento import construir_pipeline, ALVO


def treinar_knn(X_treino, y_treino, configuracao=None):
    """Treina um kNN utilizando o pipeline existente."""

    if not isinstance(X_treino, pd.DataFrame):
        raise TypeError("X_treino deve ser um DataFrame.")

    if not isinstance(y_treino, pd.Series):
        raise TypeError("y_treino deve ser uma Series.")

    if len(X_treino) != len(y_treino):
        raise ValueError("X e y possuem tamanhos diferentes.")

    if not X_treino.index.equals(y_treino.index):
        raise ValueError("Os índices de X e y não correspondem.")

    if ALVO in X_treino.columns:
        raise ValueError("A variável-alvo não pode estar em X.")

    if y_treino.isna().any() or set(y_treino.unique()) != {0, 1}:
        raise ValueError("O treinamento exige as classes 0 e 1.")

    modelo = construir_pipeline(**(configuracao or {}))
    modelo.fit(X_treino, y_treino)

    return modelo
