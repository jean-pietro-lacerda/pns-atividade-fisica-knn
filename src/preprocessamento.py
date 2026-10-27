"""Preparação experimental da PNS. Ajustar sempre dentro da validação cruzada."""
from itertools import product
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, MinMaxScaler,
                                   RobustScaler, KBinsDiscretizer)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

NUMERICAS = ['idade', 'renda_per_capita', 'numero_moradores']
CATEGORICAS = ['sexo', 'escolaridade', 'uf', 'situacao_trabalho',
               'area_residencia', 'raca_cor', 'estado_civil']
ALVO = 'ativo_lazer'


def separar_xy(base):
    """Lista explícita de preditores: exclui alvo, perguntas P e derivados."""
    faltantes = set(NUMERICAS + CATEGORICAS + [ALVO]) - set(base.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes: {sorted(faltantes)}')
    if base[ALVO].isna().any() or not base[ALVO].isin([0, 1]).all():
        raise ValueError('Alvo deve estar completo e conter somente 0/1; não imputar.')
    X = base[NUMERICAS + CATEGORICAS].copy()
    for coluna in NUMERICAS:
        X[coluna] = X[coluna].astype(float)
    if np.isinf(X[NUMERICAS].to_numpy()).any():
        raise ValueError('Preditores numéricos contêm infinito.')
    # Uniformiza categorias e ausências, inclusive UF armazenada como número.
    for coluna in CATEGORICAS:
        X[coluna] = X[coluna].astype(object).map(
            lambda valor: np.nan if __import__('pandas').isna(valor) else str(valor))
    return X, base[ALVO].astype(int).copy()


class TratamentoExtremos(TransformerMixin, BaseEstimator):
    """Alternativa experimental: preservar, log1p da renda ou limitar por IQR.

    Limites são aprendidos no fit. Nenhuma linha é removida. Log da renda
    reduz assimetria; limitar por IQR pode apagar diferenças legítimas.
    """
    def __init__(self, estrategia='preservar'):
        self.estrategia = estrategia

    def fit(self, X, y=None):
        if self.estrategia not in ('preservar', 'log_renda', 'iqr'):
            raise ValueError('Estratégia de extremos inválida')
        a = np.asarray(X, dtype=float)
        self.n_features_in_ = a.shape[1]
        q1, q3 = np.quantile(a, [0.25, 0.75], axis=0)
        self.inferior_ = q1 - 1.5 * (q3 - q1)
        self.superior_ = q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        check_is_fitted(self, 'inferior_')
        a = np.array(X, dtype=float, copy=True)
        if a.shape[1] != self.n_features_in_:
            raise ValueError('Número de características diferente do treino')
        if self.estrategia == 'iqr':
            a = np.clip(a, self.inferior_, self.superior_)
        elif self.estrategia == 'log_renda':
            if (a[:, 1] < 0).any():
                raise ValueError('Renda negativa incompatível com log1p')
            a[:, 1] = np.log1p(a[:, 1])
        return a


def construir_preprocessador(imputacao='median', escala='standard',
                            extremos='preservar', faixas=None):
    """Recebe DataFrame de separar_xy. Faixas: None, 3 ou 5 quantis.

    Numéricas: imputação -> extremos -> escala OU faixas one-hot.
    Categorias: moda -> one-hot, sem ordem artificial; novas categorias
    viram vetor zero. keep_empty_features preserva colunas vazias no treino.
    """
    if imputacao not in ('mean', 'median'):
        raise ValueError('Imputação deve ser mean ou median')
    escalas = {'standard': StandardScaler(), 'minmax': MinMaxScaler(),
               'robust': RobustScaler()}
    if escala not in escalas or faixas not in (None, 3, 5):
        raise ValueError('Escala ou quantidade de faixas inválida')
    if faixas is not None and (escala != 'standard' or extremos != 'preservar'):
        raise ValueError('Faixas substituem escala/extremos; use os padrões desses parâmetros')
    passos = [('imputacao', SimpleImputer(strategy=imputacao, keep_empty_features=True)),
              ('extremos', TratamentoExtremos(extremos))]
    if faixas is None:
        passos.append(('escala', escalas[escala]))
    else:
        passos.append(('faixas', KBinsDiscretizer(n_bins=faixas, encode='onehot-dense',
                       strategy='quantile', subsample=None, random_state=42)))
    cat = Pipeline([('imputacao', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                    ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numericas', Pipeline(passos), NUMERICAS),
                              ('categoricas', cat, CATEGORICAS)], remainder='drop')


def construir_pipeline(imputacao='median', escala='standard', extremos='preservar',
                       faixas=None, selecao=100, pca=None):
    """Pipeline completo para Pedro/Rafael; kNN permanece inteiramente padrão.

    Seleção supervisionada e PCA são ajustadas no treino. PCA aceita fração
    de variância (0, 1), usando SVD completo sobre matriz densa centrada.
    """
    if selecao not in (25, 40, 50, 60, 75, 90, 100):
        raise ValueError('Seleção deve ser 25, 40, 50, 60, 75, 90 ou 100 por cento')
    if pca is not None and not 0 < pca < 1:
        raise ValueError('PCA deve ser fração entre 0 e 1')
    return Pipeline([
        ('preprocessamento', construir_preprocessador(imputacao, escala, extremos, faixas)),
        ('selecao', SelectPercentile(f_classif, percentile=selecao) if selecao < 100 else 'passthrough'),
        ('pca', PCA(n_components=pca, svd_solver='full') if pca is not None else 'passthrough'),
        ('knn', KNeighborsClassifier())])


def configuracoes_experimentais():
    """385 combinações candidatas; não garante 385 transformações distintas.

    Imputação fixada em mediana: mean/median equivalem sem ausentes.
    9 bases contínuas + 2 discretizadas, 7 seleções e 5 opções de PCA.
    Seleção atua nas colunas codificadas, não nas dez variáveis originais.
    PCA por variância pode reter a mesma dimensão para limiares diferentes;
    seleção pode eliminar diferenças entre bases. Auditar por fold, sem
    usar validação/teste para escolher receitas. kNN permanece padrão.
    """
    selecoes = (25, 40, 50, 60, 75, 90, 100)
    componentes = (None, .80, .90, .95, .99)
    receitas = []
    for esc, ext, sel, pca in product(
            ('standard', 'minmax', 'robust'),
            ('preservar', 'log_renda', 'iqr'), selecoes, componentes):
        receitas.append(dict(imputacao='median', escala=esc, extremos=ext,
                             selecao=sel, pca=pca))
    for faixas, sel, pca in product((3, 5), selecoes, componentes):
        receitas.append(dict(imputacao='median', faixas=faixas, selecao=sel, pca=pca))
    return receitas


def validacao_cruzada():
    """Dez partes estratificadas reproduzíveis; exigir >=10 casos por classe."""
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
