"""Preparação experimental da PNS. Ajustar sempre dentro da validação cruzada."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

NUMERICAS = ["idade", "renda_per_capita", "numero_moradores"]
CATEGORICAS = [
    "sexo",
    "escolaridade",
    "uf",
    "situacao_trabalho",
    "area_residencia",
    "raca_cor",
    "estado_civil",
]
ALVO = "ativo_lazer"


def separar_xy(base):
    """Separa os dez preditores e o alvo, sem modificar a base recebida.

    Parameters
    ----------
    base : pandas.DataFrame
        Base contendo NUMERICAS, CATEGORICAS e o alvo binário completo.

    Returns
    -------
    X : pandas.DataFrame
        Cópia dos preditores: numéricos float e categorias textuais com NaN.
    y : pandas.Series
        Cópia inteira de ativo_lazer.

    Raises
    ------
    ValueError
        Se faltarem colunas, o alvo for inválido ou houver infinito numérico.
    """
    faltantes = set(NUMERICAS + CATEGORICAS + [ALVO]) - set(base.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(faltantes)}")
    if base[ALVO].isna().any() or not base[ALVO].isin([0, 1]).all():
        raise ValueError(
            "Alvo deve estar completo e conter somente 0/1; não imputar."
        )
    X = base[NUMERICAS + CATEGORICAS].copy()
    for coluna in NUMERICAS:
        X[coluna] = X[coluna].astype(float)
    if np.isinf(X[NUMERICAS].to_numpy()).any():
        raise ValueError("Preditores numéricos contêm infinito.")
    # Uniformiza categorias, inclusive UF numérica, preservando ausências.
    for coluna in CATEGORICAS:
        X[coluna] = (
            X[coluna]
            .astype(object)
            .map(lambda valor: np.nan if pd.isna(valor) else str(valor))
        )
    return X, base[ALVO].astype(int).copy()


class TratamentoExtremos(TransformerMixin, BaseEstimator):
    """Alternativa experimental: preservar, log1p da renda ou limitar por IQR.

    Limites são aprendidos no fit. Nenhuma linha é removida. Log da renda
    reduz assimetria; limitar por IQR pode apagar diferenças legítimas.
    """

    def __init__(self, estrategia="preservar"):
        """Define estrategia: preservar (padrão), log_renda ou iqr."""
        self.estrategia = estrategia

    def fit(self, X, y=None):
        """Aprende limites de 1,5 IQR em X e retorna self.

        X deve ser uma matriz numérica 2D já imputada, na ordem NUMERICAS.
        y é ignorado; os limites são calculados apenas com o treino recebido.
        """
        if self.estrategia not in ("preservar", "log_renda", "iqr"):
            raise ValueError("Estratégia de extremos inválida")
        dados_numericos = np.asarray(X, dtype=float)
        self.n_features_in_ = dados_numericos.shape[1]
        q1, q3 = np.quantile(dados_numericos, [0.25, 0.75], axis=0)
        self.inferior_ = q1 - 1.5 * (q3 - q1)
        self.superior_ = q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        """Retorna cópia numérica de X com a estratégia ajustada.

        Preserva linhas e colunas; log1p atua na renda (segunda coluna).
        Rejeita renda negativa nessa estratégia e dimensão diferente do treino.
        """
        check_is_fitted(self, "inferior_")
        dados_numericos = np.array(X, dtype=float, copy=True)
        if dados_numericos.shape[1] != self.n_features_in_:
            raise ValueError("Número de características diferente do treino")
        if self.estrategia == "iqr":
            dados_numericos = np.clip(
                dados_numericos, self.inferior_, self.superior_
            )
        elif self.estrategia == "log_renda":
            if (dados_numericos[:, 1] < 0).any():
                raise ValueError("Renda negativa incompatível com log1p")
            dados_numericos[:, 1] = np.log1p(dados_numericos[:, 1])
        return dados_numericos


def construir_preprocessador(
    imputacao="median", escala="standard", extremos="preservar", faixas=None
):
    """Cria um ColumnTransformer ainda não ajustado.

    Parameters
    ----------
    imputacao : {"mean", "median"}, default="median"
        Estratégia numérica; categorias recebem a moda do treino.
    escala : {"standard", "minmax", "robust"}, default="standard"
        Escalonador aplicado após imputação e tratamento de extremos.
    extremos : {"preservar", "log_renda", "iqr"}, default="preservar"
        Estratégia aplicada às numéricas, sem remover observações.
    faixas : {None, 3, 5}, default=None
        Quantis com saída one-hot. Substituem escala e extremos, que devem
        manter os argumentos padrão. Empates podem reduzir as faixas efetivas.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Saída densa numérica. Categorias novas geram indicadores zero;
        colunas vazias no treino são preservadas pelo imputador.
    """
    if imputacao not in ("mean", "median"):
        raise ValueError("Imputação deve ser mean ou median")
    escalas = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if escala not in escalas or faixas not in (None, 3, 5):
        raise ValueError("Escala ou quantidade de faixas inválida")
    if faixas is not None and (
        escala != "standard" or extremos != "preservar"
    ):
        raise ValueError(
            "Faixas substituem escala/extremos; use os padrões desses parâmetros"
        )
    passos = [
        (
            "imputacao",
            SimpleImputer(strategy=imputacao, keep_empty_features=True),
        ),
        ("extremos", TratamentoExtremos(extremos)),
    ]
    if faixas is None:
        passos.append(("escala", escalas[escala]))
    else:
        passos.append(
            (
                "faixas",
                KBinsDiscretizer(
                    n_bins=faixas,
                    encode="onehot-dense",
                    strategy="quantile",
                    subsample=None,
                    random_state=42,
                ),
            )
        )
    pipeline_categorico = Pipeline(
        [
            (
                "imputacao",
                SimpleImputer(
                    strategy="most_frequent", keep_empty_features=True
                ),
            ),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numericas", Pipeline(passos), NUMERICAS),
            ("categoricas", pipeline_categorico, CATEGORICAS),
        ],
        remainder="drop",
    )


def construir_pipeline(
    imputacao="median",
    escala="standard",
    extremos="preservar",
    faixas=None,
    selecao=100,
    pca=None,
):
    """Cria o pipeline de preparação, seleção, PCA e kNN padrão.

    Parameters
    ----------
    imputacao, escala, extremos, faixas
        Encaminhados a construir_preprocessador, com os mesmos padrões.
    selecao : {25, 40, 50, 60, 75, 90, 100}, default=100
        Percentual de colunas codificadas retidas por f_classif.
        O valor 100 desativa a seleção supervisionada.
    pca : float or None, default=None
        Fração de variância no intervalo (0, 1); None desativa PCA.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Estimador não ajustado. Todas as etapas são aprendidas no treino;
        PCA usa SVD completo em matriz densa centrada.
    """
    if selecao not in (25, 40, 50, 60, 75, 90, 100):
        raise ValueError(
            "Seleção deve ser 25, 40, 50, 60, 75, 90 ou 100 por cento"
        )
    if pca is not None and (not 0 < pca < 1):
        raise ValueError("PCA deve ser fração entre 0 e 1")
    return Pipeline(
        [
            (
                "preprocessamento",
                construir_preprocessador(imputacao, escala, extremos, faixas),
            ),
            (
                "selecao",
                (
                    SelectPercentile(f_classif, percentile=selecao)
                    if selecao < 100
                    else "passthrough"
                ),
            ),
            (
                "pca",
                (
                    PCA(n_components=pca, svd_solver="full")
                    if pca is not None
                    else "passthrough"
                ),
            ),
            ("knn", KNeighborsClassifier()),
        ]
    )


def configuracoes_experimentais():
    """385 combinações candidatas; não garante 385 transformações distintas.

    Imputação fixada em mediana: mean/median equivalem sem ausentes.
    9 bases contínuas + 2 discretizadas, 7 seleções e 5 opções de PCA.
    Seleção atua nas colunas codificadas, não nas dez variáveis originais.
    Retorna uma lista de dicionários aceitos por construir_pipeline.

    PCA por variância pode reter a mesma dimensão para limiares diferentes;
    seleção pode eliminar diferenças entre bases. Auditar por fold, sem
    usar validação/teste para escolher receitas. kNN permanece padrão.
    """
    selecoes = (25, 40, 50, 60, 75, 90, 100)
    componentes = (None, 0.8, 0.9, 0.95, 0.99)
    receitas = []
    for escala, extremos, selecao, pca in product(
        ("standard", "minmax", "robust"),
        ("preservar", "log_renda", "iqr"),
        selecoes,
        componentes,
    ):
        receitas.append(
            dict(
                imputacao="median",
                escala=escala,
                extremos=extremos,
                selecao=selecao,
                pca=pca,
            )
        )
    for faixas, selecao, pca in product((3, 5), selecoes, componentes):
        receitas.append(
            dict(imputacao="median", faixas=faixas, selecao=selecao, pca=pca)
        )
    return receitas


def validacao_cruzada():
    """Retorna StratifiedKFold com dez partes, shuffle e semente 42.

    A aplicação requer ao menos dez observações por classe para representar
    ambas as classes em cada partição. Não recebe dados nem ajusta modelos.
    """
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
