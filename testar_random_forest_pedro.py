from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
SAIDA = PASTA / 'random_forest'
CHAVES = ['posicao_dev', 'fold', 'real']
METRICAS = ['acuracia', 'acuracia_balanceada', 'precisao', 'recall',
            'f1', 'f2', 'auc', 'average_precision', 'fpr']


def avaliar(real, pred, prob):
    tn, fp, fn, tp = confusion_matrix(real, pred, labels=[0, 1]).ravel()
    return dict(acuracia=accuracy_score(real, pred),
                acuracia_balanceada=balanced_accuracy_score(real, pred),
                precisao=precision_score(real, pred, zero_division=0),
                recall=recall_score(real, pred, zero_division=0),
                f1=f1_score(real, pred, zero_division=0),
                f2=fbeta_score(real, pred, beta=2, zero_division=0),
                auc=roc_auc_score(real, prob),
                average_precision=average_precision_score(real, prob),
                fpr=fp/(fp+tn), tn=tn, fp=fp, fn=fn, tp=tp)


def carregar_referencia(pasta, esperado, knn=False):
    arquivo = PASTA / pasta / 'PROBABILIDADES_OOF.csv'
    d = pd.read_csv(arquivo, float_precision='round_trip')
    extra = 'votos_ativos' if knn else 'previsto'
    d = d[CHAVES + ['prob_ativo', extra]].sort_values('posicao_dev').reset_index(drop=True)
    if not np.array_equal(d[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
        raise ValueError(f'OOF desalinhado: {arquivo}. Nao comparar.')
    if not d.prob_ativo.between(0, 1).all():
        raise ValueError(f'Probabilidades invalidas: {arquivo}')
    if knn:
        if not d[extra].isin(range(6)).all() or not np.allclose(d.prob_ativo, d[extra]/5):
            raise ValueError('Votos do kNN invalidos.')
        d['previsto'] = (d[extra] >= 3).astype(int)
    elif not d.previsto.isin([0, 1]).all():
        raise ValueError('Previsoes da regressao invalidas.')
    return pd.DataFrame([dict(fold=f, **avaliar(g.real, g.previsto, g.prob_ativo))
                         for f, g in d.groupby('fold')]).set_index('fold')


def executar():
    X, y = separar_xy(pd.read_parquet(BASE))
    if X.shape[1] != 10:
        raise ValueError('Esperados 10 atributos.')
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(cv.split(X_dev, y_dev))
    esperado = pd.concat([pd.DataFrame(dict(posicao_dev=va, fold=f,
                    real=y_dev.iloc[va].to_numpy()))
                    for f, (_, va) in enumerate(splits, 1)], ignore_index=True)
    esperado = esperado.sort_values('posicao_dev').reset_index(drop=True)
    # Verifica os CSVs antes de iniciar o treinamento.
    refs = {'knn': carregar_referencia('limiares', esperado, knn=True),
            'regressao': carregar_referencia('regressao_logistica', esperado)}
    SAIDA.mkdir(parents=True, exist_ok=True)
    resultados, registros = [], []
    print('RANDOM FOREST | 10 atributos | Robust | 10 folds')
    print(f'Desenvolvimento: {len(X_dev)} registros | OOF anterior alinhado.')
    print('100 arvores; sem PCA ou balanceamento. Teste reservado NAO utilizado.')
    for fold, (tr, va) in enumerate(splits, 1):
        original = construir_pipeline(escala='robust')
        assert original.steps[-1][0] == 'knn'
        modelo = Pipeline(original.steps[:-1] + [
            ('floresta', RandomForestClassifier(random_state=42, n_jobs=-1))])
        inicio = perf_counter()
        modelo.fit(X_dev.iloc[tr], y_dev.iloc[tr])
        treino_s = perf_counter()-inicio
        inicio = perf_counter()
        real = y_dev.iloc[va].to_numpy()
        probabilidades = modelo.predict_proba(X_dev.iloc[va])
        # Regra padrao do Random Forest: classe de maior probabilidade.
        pred = modelo.classes_[probabilidades.argmax(axis=1)]
        prob = probabilidades[:, list(modelo.classes_).index(1)]
        previsao_s = perf_counter()-inicio
        r = dict(fold=fold, **avaliar(real, pred, prob),
                 treino_s=treino_s, previsao_e_proba_s=previsao_s)
        resultados.append(r)
        registros.append(pd.DataFrame(dict(posicao_dev=va, fold=fold, real=real,
                                           previsto=pred, prob_ativo=prob)))
        print(f'[{fold}/10] AUC={r["auc"]:.4f} | AP={r["average_precision"]:.4f} | '
              f'F1={r["f1"]:.4f} | Recall={r["recall"]:.4f} | Treino={treino_s:.2f}s', flush=True)

    folds = pd.DataFrame(resultados).set_index('fold')
    oof = pd.concat(registros, ignore_index=True).sort_values('posicao_dev').reset_index(drop=True)
    assert np.array_equal(oof[CHAVES].to_numpy(), esperado[CHAVES].to_numpy())
    comparacao = pd.DataFrame({nome: dados[METRICAS].mean() for nome, dados in refs.items()})
    comparacao['random_forest'] = folds[METRICAS].mean()
    comparacao['delta_vs_knn'] = comparacao.random_forest - comparacao.knn
    comparacao['delta_vs_regressao'] = comparacao.random_forest - comparacao.regressao
    resumo = folds[METRICAS].agg(['mean','std']).T
    resumo.columns = ['media','desvio_padrao']
    pares = []
    for nome, dados in refs.items():
        for metrica in ['auc','average_precision','f1','recall','acuracia_balanceada']:
            delta = folds[metrica] - dados[metrica]
            pares.append(dict(referencia=nome, metrica=metrica, delta_medio=delta.mean(),
                              dp_delta=delta.std(), folds_melhores=int((delta>1e-12).sum()),
                              folds_piores=int((delta< -1e-12).sum()),
                              empates=int((delta.abs()<=1e-12).sum())))
    pares = pd.DataFrame(pares)
    matriz = folds[['tn','fp','fn','tp']].mean()
    tempos = folds[['treino_s','previsao_e_proba_s']].mean()
    for titulo, tabela in [('COMPARACAO_FINAL',comparacao),('MEDIA_E_DESVIO_PADRAO',resumo),
                           ('COMPARACAO_PAREADA',pares),('MATRIZ_MEDIA',matriz),('TEMPOS',tempos)]:
        print(f'\n=== {titulo} ===')
        print(tabela.round(4).to_string())
        tabela.to_csv(SAIDA / f'{titulo}.csv', index=titulo!='COMPARACAO_PAREADA')
    folds.to_csv(SAIDA/'METRICAS_POR_FOLD.csv')
    oof.to_csv(SAIDA/'PROBABILIDADES_OOF.csv', index=False)
    print('\nComparacao exploratoria; folds dependentes. Limiar padrao mantido.')
    print('Teste reservado e codigo oficial intactos.')
    print(f'Resultados: {SAIDA.resolve()}')


if __name__ == '__main__':
    executar()
