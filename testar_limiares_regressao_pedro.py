from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

RAIZ = Path(__file__).resolve().parent
PASTA = RAIZ / 'outputs/diagnostico_pedro'
SAIDA = PASTA / 'limiares_regressao_logistica'
LIMIARES = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
CHAVES = ['posicao_dev', 'fold', 'real']
METRICAS = ['acuracia', 'acuracia_balanceada', 'precisao', 'recall',
            'f1', 'f2', 'auc', 'average_precision', 'fpr']


def carregar(pasta, extras):
    arquivo = PASTA / pasta / 'PROBABILIDADES_OOF.csv'
    if not arquivo.exists():
        raise FileNotFoundError(f'CSV anterior nao encontrado: {arquivo}')
    d = pd.read_csv(arquivo, float_precision='round_trip')
    cols = CHAVES + ['prob_ativo'] + extras
    if not set(cols).issubset(d.columns):
        raise ValueError(f'Colunas ausentes em {arquivo}: {set(cols)-set(d.columns)}')
    d = d[cols].sort_values('posicao_dev').reset_index(drop=True)
    if (len(d) != 70764 or not np.array_equal(d.posicao_dev, np.arange(70764))
            or set(d.fold) != set(range(1, 11)) or set(d.real) != {0, 1}
            or d[cols].isna().any().any() or not d.prob_ativo.between(0, 1).all()
            or not d.groupby('fold').real.nunique().eq(2).all()):
        raise ValueError(f'OOF invalido ou diferente do experimento original: {arquivo}')
    return d


def dividir(a, b):
    return a / b if b else 0.0


def metricas(y, pred, p, com_escores=True):
    y, pred = np.asarray(y), np.asarray(pred, dtype=bool)
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    recall = dividir(tp, tp + fn)
    fpr = dividir(fp, fp + tn)
    r = dict(acuracia=dividir(tp+tn, len(y)),
             acuracia_balanceada=(recall+1-fpr)/2,
             precisao=dividir(tp, tp+fp), recall=recall,
             f1=dividir(2*tp, 2*tp+fp+fn),
             f2=dividir(5*tp, 5*tp+fp+4*fn), fpr=fpr,
             tn=tn, fp=fp, fn=fn, tp=tp,
             taxa_prevista_ativo=dividir(tp+fp, len(y)))
    if com_escores:
        r.update(auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p))
    return r


def executar():
    reg = carregar('regressao_logistica', ['previsto'])
    knn = carregar('limiares', ['votos_ativos'])
    if not reg[CHAVES].equals(knn[CHAVES]):
        raise ValueError('Posicoes, alvos ou folds nao coincidem. Nao comparar.')
    if not reg.previsto.isin([0, 1]).all() or not knn.votos_ativos.isin(range(6)).all():
        raise ValueError('Previsoes originais ou votos invalidos.')
    if not np.allclose(knn.prob_ativo, knn.votos_ativos/5):
        raise ValueError('Votos e probabilidades do kNN nao coincidem.')
    p = reg.prob_ativo.to_numpy()
    y = reg.real.to_numpy()
    if not np.array_equal(reg.previsto[p != .5], (p[p != .5] > .5).astype(int)):
        raise ValueError('Previsoes da regressao nao correspondem ao limiar padrao.')
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = []
    for fold, d in reg.groupby('fold'):
        real = d.real.to_numpy()
        prob = d.prob_ativo.to_numpy()
        for t in LIMIARES:
            # No limiar original, preserva a previsao exata do estimador,
            # inclusive eventual empate em 0.5.
            pred = d.previsto.to_numpy() if t == .5 else prob >= t
            linhas.append(dict(fold=fold, limiar=t, **metricas(real, pred, prob)))
    folds = pd.DataFrame(linhas)
    grupos = folds.groupby('limiar')
    medias = grupos[METRICAS+['taxa_prevista_ativo']].mean().reindex(LIMIARES)
    dp = grupos[METRICAS].std().reindex(LIMIARES)
    matrizes = grupos[['tn','fp','fn','tp']].mean().reindex(LIMIARES)
    ganhos = pd.DataFrame({'ativos_recuperados_por_fold': matrizes.tp-matrizes.loc[.5,'tp'],
                           'fp_adicionais_por_fold': matrizes.fp-matrizes.loc[.5,'fp'],
                           'delta_f1': medias.f1-medias.loc[.5,'f1'],
                           'delta_recall': medias.recall-medias.loc[.5,'recall']})
    # Comparacao descritiva com o MESMO teto de FPR agregado.
    # O limiar da regressao e escolhido nos proprios OOF; nao e avaliacao independente.
    fprs, tprs, cortes = roc_curve(y, p, drop_intermediate=False)
    pares = []
    for t, votos in [(0.5,3),(0.4,2),(0.2,1)]:
        k = metricas(y, knn.votos_ativos >= votos, knn.prob_ativo, False)
        elegiveis = np.flatnonzero(fprs <= k['fpr'] + 1e-12)
        j = elegiveis[np.argmax(tprs[elegiveis])]
        corte = float(cortes[j])
        pred = p >= corte if np.isfinite(corte) and corte <= 1 else np.zeros(len(p), dtype=bool)
        r = metricas(y, pred, p, False)
        pares.append(dict(limiar_knn=t, limiar_reg=corte,
                          fpr_knn=k['fpr'], fpr_reg=r['fpr'],
                          recall_knn=k['recall'], recall_reg=r['recall'],
                          precisao_knn=k['precisao'], precisao_reg=r['precisao'],
                          f1_knn=k['f1'], f1_reg=r['f1'],
                          fp_knn=k['fp'], fp_reg=r['fp'], tp_knn=k['tp'], tp_reg=r['tp']))
    pareada = pd.DataFrame(pares)
    tabelas = [('COMPARACAO_LIMIARES', medias, True),
               ('DESVIOS_PADRAO', dp, True), ('MATRIZES_MEDIAS', matrizes, True),
               ('GANHOS_E_CUSTOS', ganhos, True),
               ('COMPARACAO_MESMO_TETO_FPR_OOF', pareada, False),
               ('METRICAS_POR_FOLD', folds, False)]
    print('Regressao sem PCA | 10 folds | 0 novos treinamentos')
    print(f'OOF alinhado: {len(reg)} registros | ativos: {int(y.sum())}')
    for nome, tabela, indice in tabelas:
        tabela.to_csv(SAIDA / f'{nome}.csv', index=indice)
        if nome != 'METRICAS_POR_FOLD':
            print(f'\n=== {nome} ===')
            print(tabela.round(4).to_string(index=indice))
    print('\nLIMIARES FIXOS: medias dos dez folds.')
    print('MESMO TETO FPR: metricas agregadas OOF, cortes exploratorios.')
    print('Nenhum limiar selecionado para uso definitivo.')
    print('Teste reservado NAO utilizado. Codigo oficial NAO alterado.')
    print(f'Resultados: {SAIDA}')


if __name__ == '__main__':
    executar()
