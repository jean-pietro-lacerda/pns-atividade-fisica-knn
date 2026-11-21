"""Avaliação de modelos kNN no conjunto de teste e em validação cruzada."""

from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessamento import ALVO, validacao_cruzada
from src.treinamento import treinar_knn


def _validar_xy_avaliacao(X, y):
    """Valida tipos, tamanhos, índices e classes antes da avaliação."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X deve ser um DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y deve ser uma Series.")
    if len(X) != len(y):
        raise ValueError("X e y possuem tamanhos diferentes.")
    if not X.index.equals(y.index):
        raise ValueError("Os índices de X e y não correspondem.")
    if ALVO in X.columns:
        raise ValueError("A variável-alvo não pode estar em X.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("A avaliação exige as classes 0 e 1 sem ausentes.")


def avaliar_modelo(modelo, X_teste, y_teste):
    """Avalia um modelo já treinado nos dados de teste.

    Parameters
    ----------
    modelo : sklearn.pipeline.Pipeline
        Estimador já ajustado nos dados de treino.
    X_teste : pandas.DataFrame
        Preditores do conjunto de teste.
    y_teste : pandas.Series
        Rótulos reais (0 e 1) do conjunto de teste.

    Returns
    -------
    dict
        Dicionário com acurácia, baseline, precisão, revocação, F1-score,
        AUC-ROC e os quatro valores da matriz de confusão (VP, FN, FP, VN).
    """
    _validar_xy_avaliacao(X_teste, y_teste)

    y_pred = modelo.predict(X_teste)
    y_proba = modelo.predict_proba(X_teste)[:, 1]

    vn, fp, fn, vp = confusion_matrix(y_teste, y_pred, labels=[0, 1]).ravel()
    baseline = float(y_teste.value_counts(normalize=True).max())

    return {
        "acuracia": float(accuracy_score(y_teste, y_pred)),
        "acuracia_baseline": baseline,
        "precisao": float(precision_score(y_teste, y_pred, zero_division=0)),
        "revocacao": float(recall_score(y_teste, y_pred, zero_division=0)),
        "f1": float(f1_score(y_teste, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_teste, y_proba)),
        "vp": int(vp),
        "fn": int(fn),
        "fp": int(fp),
        "vn": int(vn),
    }


def _avaliar_um_fold(X, y, idx_treino, idx_teste, configuracao):
    """Treina e avalia um único fold da validação cruzada."""
    X_tr, y_tr = X.iloc[idx_treino], y.iloc[idx_treino]
    X_te, y_te = X.iloc[idx_teste], y.iloc[idx_teste]
    modelo = treinar_knn(X_tr, y_tr, configuracao=configuracao)
    return avaliar_modelo(modelo, X_te, y_te)


def avaliar_validacao_cruzada(X, y, configuracao=None, n_jobs=-2):
    """Executa validação cruzada estratificada (10 folds) para uma configuração.

    Parameters
    ----------
    X : pandas.DataFrame
        Preditores completos da base.
    y : pandas.Series
        Variável-alvo binária (0 e 1).
    configuracao : dict or None, default=None
        Parâmetros repassados para construir_pipeline.
    n_jobs : int, default=-2
        Número de processos paralelos para os folds (1 executa sequencial).

    Returns
    -------
    resumo : dict
        Média e desvio padrão das métricas e matriz de confusão acumulada.
    df_folds : pandas.DataFrame
        Métricas detalhadas de cada uma das 10 partições.
    """
    _validar_xy_avaliacao(X, y)

    cv = validacao_cruzada()
    folds = list(cv.split(X, y))

    resultados_folds = Parallel(n_jobs=n_jobs)(
        delayed(_avaliar_um_fold)(X, y, idx_tr, idx_te, configuracao)
        for idx_tr, idx_te in folds
    )

    df_folds = pd.DataFrame(resultados_folds)
    metricas_continuas = [
        "acuracia",
        "acuracia_baseline",
        "precisao",
        "revocacao",
        "f1",
        "auc_roc",
    ]

    resumo = {}
    for metrica in metricas_continuas:
        resumo[f"{metrica}_media"] = float(df_folds[metrica].mean())
        resumo[f"{metrica}_desvio"] = float(df_folds[metrica].std())

    for componente in ("vp", "fn", "fp", "vn"):
        resumo[componente] = int(df_folds[componente].sum())

    return resumo, df_folds