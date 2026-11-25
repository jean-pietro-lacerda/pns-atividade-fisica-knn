from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from src.preprocessamento import separar_xy, construir_pipeline

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / 'data/interim/pns_base_eda.parquet'
PASTA = RAIZ / 'outputs/diagnostico_pedro'
SAIDA = PASTA / 'pca_regressao_logistica'
COMPONENTES = [30, 20, 15, 10, 5]
METRICAS = ['acuracia', 'acuracia_balanceada', 'precisao', 'recall',
            'f1', 'f2', 'auc', 'average_precision', 'fpr']
CONFUSAO = ['tn', 'fp', 'fn', 'tp']


def medir(real, pred, prob):
    tn, fp, fn, tp = confusion_matrix(real, pred, labels=[0, 1]).ravel()
    return dict(
        acuracia=accuracy_score(real, pred),
        acuracia_balanceada=balanced_accuracy_score(real, pred),
        precisao=precision_score(real, pred, zero_division=0),
        recall=recall_score(real, pred, zero_division=0),
        f1=f1_score(real, pred, zero_division=0),
        f2=fbeta_score(real, pred, beta=2, zero_division=0),
        auc=roc_auc_score(real, prob),
        average_precision=average_precision_score(real, prob),
        fpr=fp / (fp + tn), tn=tn, fp=fp, fn=fn, tp=tp,
    )


def carregar_referencia(X_dev, y_dev, particoes):
    # Reaproveita os escores e as previsoes do experimento SEM PCA.
    # Verifica alinhamento dos folds e classes antes da comparacao.
    caminho = PASTA / 'regressao_logistica/PROBABILIDADES_OOF.csv'
    if not caminho.exists():
        raise FileNotFoundError(f'Falta o CSV da regressao anterior: {caminho}')
    oof = pd.read_csv(caminho).sort_values('posicao_dev')
    if not np.array_equal(oof['posicao_dev'].to_numpy(), np.arange(len(X_dev))):
        raise ValueError('Posicoes da referencia nao correspondem ao desenvolvimento.')
    if not np.array_equal(oof['real'].to_numpy(), y_dev.to_numpy()):
        raise ValueError('Alvos da referencia nao correspondem a base atual.')
    registros = []
    for fold, (_, va) in enumerate(particoes, 1):
        dados = oof.iloc[va]
        if not (dados['fold'] == fold).all():
            raise ValueError(f'Particao diferente na referencia: fold {fold}')
        registros.append({'fold': fold, 'configuracao': 'sem_pca',
                          **medir(dados['real'], dados['previsto'], dados['prob_ativo'])})
    return pd.DataFrame(registros)


def executar():
    SAIDA.mkdir(parents=True, exist_ok=True)
    X, y = separar_xy(pd.read_parquet(BASE))
    assert X.shape[1] == 10
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    particoes = list(cv.split(X_dev, y_dev))
    referencia = carregar_referencia(X_dev, y_dev, particoes)
    resultados, registros = [], []
    print('REGRESSAO LOGISTICA + PCA | 10 atributos | Robust | 10 folds')
    print(f'Desenvolvimento: {len(X_dev)} registros')
    print('50 novos ajustes. Referencia sem PCA recuperada do CSV anterior.')
    print('Teste reservado NAO utilizado.')

    for fold, (tr, va) in enumerate(particoes, 1):
        original = construir_pipeline(escala='robust')
        assert [nome for nome, _ in original.steps][-2:] == ['pca', 'knn']
        pre = Pipeline(original.steps[:-2])
        y_tr = y_dev.iloc[tr].to_numpy()
        real = y_dev.iloc[va].to_numpy()
        inicio = perf_counter()
        z_tr = np.asarray(pre.fit_transform(X_dev.iloc[tr], y_tr))
        z_va = np.asarray(pre.transform(X_dev.iloc[va]))
        tempo_pre = perf_counter() - inicio
        assert min(z_tr.shape) >= max(COMPONENTES)
        # Mesmo PCA do diagnostico anterior: SVD completo, sem whitening.
        # Ajuste apenas no treino; prefixos da mesma decomposicao.
        inicio = perf_counter()
        pca = PCA(n_components=max(COMPONENTES), svd_solver='full')
        pca.fit(z_tr)
        p_tr, p_va = pca.transform(z_tr), pca.transform(z_va)
        tempo_pca = perf_counter() - inicio
        variancia = np.cumsum(pca.explained_variance_ratio_)

        for n in COMPONENTES:
            nome = f'pca_{n}'
            a = np.ascontiguousarray(p_tr[:, :n])
            b = np.ascontiguousarray(p_va[:, :n])
            modelo = LogisticRegression()
            inicio = perf_counter()
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter('always', ConvergenceWarning)
                modelo.fit(a, y_tr)
            tempo_treino = perf_counter() - inicio
            convergiu = not any(issubclass(w.category, ConvergenceWarning) for w in avisos)
            for w in avisos:
                print(f'Aviso fold {fold}, {nome}: {w.message}')
            inicio = perf_counter()
            pred = modelo.predict(b)
            prob = modelo.predict_proba(b)[:, list(modelo.classes_).index(1)]
            tempo_pred = perf_counter() - inicio
            r = {'fold': fold, 'configuracao': nome, 'componentes': n,
                 'dimensoes_originais': z_tr.shape[1],
                 'variancia_explicada': variancia[n-1],
                 **medir(real, pred, prob), 'convergiu': convergiu,
                 'iteracoes': int(modelo.n_iter_.max()), 'treino_s': tempo_treino,
                 'previsao_e_proba_s': tempo_pred,
                 'tempo_pre_compartilhado_s': tempo_pre,
                 'tempo_pca_compartilhado_s': tempo_pca}
            resultados.append(r)
            registros.append(pd.DataFrame({'posicao_dev': va, 'fold': fold,
                'configuracao': nome, 'real': real, 'previsto': pred, 'prob_ativo': prob}))
            print(f'[{len(resultados)}/50] Fold {fold} | {nome} | '
                  f'Variancia={variancia[n-1]:.2%} | AUC={r["auc"]:.4f} | '
                  f'F1={r["f1"]:.4f} | Recall={r["recall"]:.4f} | '
                  f'Convergiu={convergiu}', flush=True)

    novos = pd.DataFrame(resultados)
    assert len(novos) == 50
    oof = pd.concat(registros, ignore_index=True)
    assert len(oof) == len(COMPONENTES) * len(X_dev)
    assert not oof.duplicated(['configuracao', 'posicao_dev']).any()
    folds = pd.concat([referencia, novos], ignore_index=True)
    ordem = ['sem_pca'] + [f'pca_{n}' for n in COMPONENTES]
    grupo = folds.groupby('configuracao')
    medias = grupo[METRICAS].mean().reindex(ordem)
    dp = grupo[METRICAS].std().reindex(ordem)
    delta = medias - medias.loc['sem_pca']
    matrizes = grupo[CONFUSAO].mean().reindex(ordem)
    variancia = novos.groupby('configuracao')[['variancia_explicada']].mean().reindex(ordem)
    variancia.loc['sem_pca', 'variancia_explicada'] = 1.0
    pareadas = []
    for metrica in ['auc', 'average_precision', 'f1', 'recall']:
        p = folds.pivot(index='fold', columns='configuracao', values=metrica)
        for nome in ordem[1:]:
            d = p[nome] - p['sem_pca']
            pareadas.append({'configuracao': nome, 'metrica': metrica,
                'delta_medio': d.mean(), 'dp_delta': d.std(),
                'folds_melhores': int((d > 1e-12).sum()),
                'folds_piores': int((d < -1e-12).sum()),
                'empates': int((d.abs() <= 1e-12).sum())})
    convergencia = novos.groupby('configuracao').agg(
        folds_sem_aviso=('convergiu', 'sum'), iteracoes_medias=('iteracoes', 'mean'),
        treino_s=('treino_s', 'mean'), previsao_e_proba_s=('previsao_e_proba_s', 'mean'))
    tabelas = {'COMPARACAO_FINAL': medias, 'DELTA_SEM_PCA': delta,
               'DESVIOS_PADRAO': dp, 'VARIANCIA_EXPLICADA': variancia,
               'MATRIZES_MEDIAS': matrizes, 'CONVERGENCIA_E_TEMPOS': convergencia}
    # Compara tambem com o KNN + PCA anterior, se o CSV estiver disponivel.
    knn_csv = PASTA / 'pca/COMPARACAO_FINAL.csv'
    if knn_csv.exists():
        knn = pd.read_csv(knn_csv, index_col='configuracao').reindex(ordem)
        comparar = pd.DataFrame(index=ordem)
        for m in ['auc', 'average_precision', 'f1', 'recall']:
            comparar[f'{m}_knn'] = knn[m]
            comparar[f'{m}_regressao'] = medias[m]
            comparar[f'{m}_delta'] = medias[m] - knn[m]
        tabelas['COMPARACAO_KNN_PCA'] = comparar
    for nome, tabela in tabelas.items():
        print(f'\n=== {nome} ===')
        print(tabela.round(4).to_string())
        tabela.to_csv(SAIDA / f'{nome}.csv')
    pares = pd.DataFrame(pareadas)
    print('\n=== COMPARACAO_PAREADA ===')
    print(pares.round(4).to_string(index=False))
    pares.to_csv(SAIDA / 'COMPARACAO_PAREADA.csv', index=False)
    folds.to_csv(SAIDA / 'METRICAS_POR_FOLD.csv', index=False)
    oof.to_csv(SAIDA / 'PROBABILIDADES_OOF.csv', index=False)
    if not novos['convergiu'].all():
        print('ATENCAO: houve avisos de convergencia; investigar antes de concluir.')
    print('\nComparacao exploratoria; folds dependentes.')
    print('Limiar padrao mantido. Teste reservado e codigo oficial intactos.')
    print(f'Resultados: {SAIDA.resolve()}')


if __name__ == '__main__':
    executar()
