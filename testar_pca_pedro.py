from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_auc_score,
    average_precision_score, confusion_matrix, pairwise_distances,
)
from src.preprocessamento import separar_xy, construir_pipeline

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / 'data/interim/pns_base_eda.parquet'
SAIDA = RAIZ / 'outputs/diagnostico_pedro/pca'
COMPONENTES = [30, 20, 15, 10, 5]
REFERENCIA = {
    'acuracia': 0.6956, 'acuracia_balanceada': 0.5763,
    'precisao': 0.4452, 'recall': 0.3009,
    'f1': 0.3590, 'f2': 0.3217,
    'auc': 0.6281, 'average_precision': 0.3768,
    'fpr': 0.1483,
}


def geometria(treino, validacao, it, iv):
    # Mesma amostra de pares em todas as representacoes.
    # Diagnostico amostral, nao a busca real do kNN.
    d = pairwise_distances(
        validacao[iv], treino[it], metric='euclidean'
    )
    mediana = np.median(d, axis=1)
    validos = mediana > 0
    razao = d.min(axis=1)[validos] / mediana[validos]
    return {
        'distancia_media': d.mean(),
        'cv_distancias': d.std() / d.mean() if d.mean() > 0 else np.nan,
        'razao_min_mediana': razao.mean() if len(razao) else np.nan,
        'consultas_validas': int(validos.sum()),
    }


def executar():
    SAIDA.mkdir(parents=True, exist_ok=True)
    X, y = separar_xy(pd.read_parquet(BASE))
    assert X.shape[1] == 10

    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    resultados, distancias, registros = [], [], []

    print('KNN + PCA | 10 atributos | Robust | 10 folds')
    print(f'Desenvolvimento: {len(X_dev)} registros')
    print('50 novos ajustes de kNN. Teste reservado NAO utilizado.')

    for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev), 1):
        original = construir_pipeline(escala='robust')
        assert [nome for nome, _ in original.steps][-2:] == ['pca', 'knn']
        # Reutiliza os passos anteriores ao PCA e ao modelo.
        pre = Pipeline(original.steps[:-2])
        y_tr = y_dev.iloc[tr].to_numpy()
        real = y_dev.iloc[va].to_numpy()
        inicio = perf_counter()
        z_tr = np.asarray(pre.fit_transform(X_dev.iloc[tr], y_tr))
        z_va = np.asarray(pre.transform(X_dev.iloc[va]))
        tempo_pre = perf_counter() - inicio
        assert min(z_tr.shape) >= max(COMPONENTES)

        rng = np.random.default_rng(42 + fold)
        it = rng.choice(len(tr), min(2048, len(tr)), replace=False)
        iv = rng.choice(len(va), min(256, len(va)), replace=False)

        distancias.append({
            'fold': fold, 'configuracao': 'sem_pca',
            'dimensoes': z_tr.shape[1],
            **geometria(z_tr, z_va, it, iv),
        })

        # PCA ajustado SOMENTE no treino do fold.
        # Uma decomposicao fornece todos os prefixos de componentes.
        inicio = perf_counter()
        pca = PCA(n_components=max(COMPONENTES), svd_solver='full')
        pca.fit(z_tr)
        p_tr = pca.transform(z_tr)
        p_va = pca.transform(z_va)
        tempo_pca = perf_counter() - inicio
        variancia = np.cumsum(pca.explained_variance_ratio_)

        for n in COMPONENTES:
            nome = f'pca_{n}'
            a = np.ascontiguousarray(p_tr[:, :n])
            b = np.ascontiguousarray(p_va[:, :n])
            modelo = clone(original.named_steps['knn'])
            assert modelo.n_neighbors == 5 and modelo.weights == 'uniform'

            inicio = perf_counter()
            modelo.fit(a, y_tr)
            tempo_knn = perf_counter() - inicio
            inicio = perf_counter()
            prob = modelo.predict_proba(b)[:, list(modelo.classes_).index(1)]
            pred = (prob > 0.5).astype(int)
            tempo_pred = perf_counter() - inicio

            tn, fp, fn, tp = confusion_matrix(real, pred, labels=[0, 1]).ravel()
            r = {
                'fold': fold, 'configuracao': nome,
                'dimensoes_originais': z_tr.shape[1],
                'componentes': n,
                'variancia_explicada': variancia[n - 1],
                'acuracia': accuracy_score(real, pred),
                'acuracia_balanceada': balanced_accuracy_score(real, pred),
                'precisao': precision_score(real, pred, zero_division=0),
                'recall': recall_score(real, pred, zero_division=0),
                'f1': f1_score(real, pred, zero_division=0),
                'f2': fbeta_score(real, pred, beta=2, zero_division=0),
                'auc': roc_auc_score(real, prob),
                'average_precision': average_precision_score(real, prob),
                'fpr': fp / (fp + tn),
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'tempo_knn_s': tempo_knn, 'tempo_previsao_s': tempo_pred,
                'tempo_pre_compartilhado_s': tempo_pre,
                'tempo_pca_compartilhado_s': tempo_pca,
            }
            resultados.append(r)
            distancias.append({
                'fold': fold, 'configuracao': nome, 'dimensoes': n,
                **geometria(a, b, it, iv),
            })
            registros.append(pd.DataFrame({
                'posicao_dev': va, 'fold': fold, 'configuracao': nome,
                'real': real, 'previsto': pred, 'prob_ativo': prob,
            }))
            print(
                f'[{len(resultados)}/50] Fold {fold} | {nome} | '
                f'Variancia={variancia[n-1]:.2%} | '
                f'AUC={r["auc"]:.4f} | F1={r["f1"]:.4f} | '
                f'Recall={r["recall"]:.4f}', flush=True,
            )

    folds = pd.DataFrame(resultados)
    geo = pd.DataFrame(distancias)
    oof = pd.concat(registros, ignore_index=True)
    assert len(folds) == 50
    assert not oof.duplicated(['configuracao', 'posicao_dev']).any()
    assert len(oof) == len(COMPONENTES) * len(X_dev)

    ordem = ['sem_pca'] + [f'pca_{n}' for n in COMPONENTES]
    cols = list(REFERENCIA)
    medias = folds.groupby('configuracao')[cols].mean()
    medias.loc['sem_pca'] = pd.Series(REFERENCIA)
    medias = medias.reindex(ordem)
    delta = medias - medias.loc['sem_pca']
    dp = folds.groupby('configuracao')[cols].std()
    variancia = folds.groupby('configuracao')[['variancia_explicada']].mean()
    geo_medias = geo.groupby('configuracao')[
        ['dimensoes', 'distancia_media', 'cv_distancias', 'razao_min_mediana']
    ].mean().reindex(ordem)
    matrizes = folds.groupby('configuracao')[['tn', 'fp', 'fn', 'tp']].mean()
    matrizes.loc['sem_pca'] = [4319.0, 752.2, 1401.9, 603.3]
    matrizes = matrizes.reindex(ordem)

    tabelas = {
        'COMPARACAO_FINAL': medias,
        'DELTA_SEM_PCA': delta,
        'DESVIOS_PADRAO': dp,
        'VARIANCIA_EXPLICADA': variancia,
        'GEOMETRIA_MEDIA': geo_medias,
        'MATRIZES_MEDIAS': matrizes,
    }
    for nome, tabela in tabelas.items():
        print(f'\n=== {nome} ===')
        print(tabela.round(4).to_string())
        tabela.to_csv(SAIDA / f'{nome}.csv')
    folds.to_csv(SAIDA / 'METRICAS_POR_FOLD.csv', index=False)
    geo.to_csv(SAIDA / 'GEOMETRIA_POR_FOLD.csv', index=False)
    oof.to_csv(SAIDA / 'PROBABILIDADES_OOF.csv', index=False)
    print('\nSem PCA: referencia de metricas do experimento anterior.')
    print('Geometria sem PCA: calculada agora, sem novo ajuste de kNN.')
    print('Analise exploratoria. Codigo oficial e teste reservado intactos.')
    print(f'Resultados: {SAIDA.resolve()}')


if __name__ == '__main__':
    executar()
