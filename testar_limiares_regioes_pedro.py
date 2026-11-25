from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
PASTA = ROOT / 'outputs/diagnostico_pedro'
ENTRADA = PASTA / 'regioes_balanceamento/PROBABILIDADES_OOF.csv'
SAIDA = PASTA / 'limiares_regioes'
LIMIARES = {0.5: 3, 0.4: 2, 0.2: 1}
CHAVES = ['posicao_dev', 'fold', 'real']
METRICAS = ['acuracia', 'acuracia_balanceada', 'precisao', 'recall',
            'f1', 'f2', 'auc', 'average_precision', 'fpr']
MATRIZ = ['tn', 'fp', 'fn', 'tp']


def avaliar(g, minimo):
    y = g.real.to_numpy()
    p = g.prob_ativo.to_numpy()
    pred = (g.votos.to_numpy() >= minimo).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acuracia=accuracy_score(y, pred),
        acuracia_balanceada=balanced_accuracy_score(y, pred),
        precisao=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        f2=fbeta_score(y, pred, beta=2, zero_division=0),
        auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
        fpr=fp/(fp+tn), tn=tn, fp=fp, fn=fn, tp=tp)


def executar():
    dados = pd.read_csv(ENTRADA)
    configs = ['uf_original', 'regioes', 'regioes_balanceado']
    grupos = {}
    for nome in configs:
        g = dados[dados.configuracao == nome].copy().sort_values('posicao_dev')
        g = g.reset_index(drop=True)
        if (len(g) != 70764 or not np.array_equal(g.posicao_dev, np.arange(70764))
            or set(g.fold) != set(range(1, 11)) or set(g.real) != {0, 1}
            or g.real.sum() != 20052):
            raise ValueError(f'OOF incompleto ou invalido: {nome}')
        p = g.prob_ativo.to_numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Probabilidades invalidas: {nome}')
        g['votos'] = np.rint(5*p).astype(int)
        if not np.allclose(p, g.votos/5):
            raise ValueError(f'Escores nao correspondem a 5 votos: {nome}')
        if grupos and not g[CHAVES].equals(grupos['uf_original'][CHAVES]):
            raise ValueError(f'OOF desalinhado: {nome}')
        grupos[nome] = g
    linhas = []
    for nome, g in grupos.items():
        cortes = {0.5: 3} if nome == 'regioes_balanceado' else LIMIARES
        for limiar, minimo in cortes.items():
            for fold, parte in g.groupby('fold'):
                linhas.append(dict(configuracao=nome, limiar=limiar, fold=fold,
                                    **avaliar(parte, minimo)))
    d = pd.DataFrame(linhas)
    gb = d.groupby(['configuracao', 'limiar'], sort=False)
    medias = gb[METRICAS].mean()
    dp = gb[METRICAS].std()
    acumuladas = gb[MATRIZ].sum()
    matrizes = gb[MATRIZ].mean()
    ref = medias.loc[('regioes', 0.5)]
    matriz_ref = acumuladas.loc[('regioes', 0.5)]
    ganhos = pd.DataFrame({'delta_f1': medias.f1-ref.f1,
        'delta_recall': medias.recall-ref.recall,
        'delta_fpr': medias.fpr-ref.fpr,
        'ativos_adicionais': acumuladas.tp-matriz_ref.tp,
        'fp_adicionais': acumuladas.fp-matriz_ref.fp})
    votos = pd.crosstab(grupos['regioes'].votos, grupos['regioes'].real).reindex(
        index=range(6), columns=[0,1], fill_value=0)
    votos.columns = ['inativos_reais', 'ativos_reais']
    print('5 regioes | Robust | kNN padrao | 0 novos treinamentos')
    print('OOF alinhado: 70764 registros | Teste reservado e codigo oficial intactos')
    SAIDA.mkdir(parents=True, exist_ok=True)
    for nome, tabela in [('COMPARACAO_FINAL', medias), ('DESVIOS_PADRAO', dp),
                         ('MATRIZES_ACUMULADAS', acumuladas), ('MATRIZES_MEDIAS', matrizes),
                         ('GANHOS_VS_REGIOES_PADRAO', ganhos), ('DISTRIBUICAO_VOTOS', votos)]:
        print('\n=== '+nome+' ===\n'+tabela.round(4).to_string())
        tabela.to_csv(SAIDA / (nome+'.csv'))
    d.to_csv(SAIDA/'METRICAS_POR_FOLD.csv', index=False)
    print('\nMedias dos 10 folds; matrizes acumuladas com numeros inteiros.')
    print('Comparacao exploratoria; nenhum limiar selecionado definitivamente.')
    print('Resultados:', SAIDA)


if __name__ == '__main__':
    executar()
