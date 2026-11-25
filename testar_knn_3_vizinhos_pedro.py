from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_auc_score, average_precision_score, confusion_matrix)
from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
PASTA = ROOT / 'outputs/diagnostico_pedro'
SAIDA = PASTA / 'regioes_k3'
REGIOES = {
    'Norte': [(11,'RO','Rondonia'),(12,'AC','Acre'),(13,'AM','Amazonas'),(14,'RR','Roraima'),(15,'PA','Para'),(16,'AP','Amapa'),(17,'TO','Tocantins')],
    'Nordeste': [(21,'MA','Maranhao'),(22,'PI','Piaui'),(23,'CE','Ceara'),(24,'RN','Rio Grande do Norte'),(25,'PB','Paraiba'),(26,'PE','Pernambuco'),(27,'AL','Alagoas'),(28,'SE','Sergipe'),(29,'BA','Bahia')],
    'Sudeste': [(31,'MG','Minas Gerais'),(32,'ES','Espirito Santo'),(33,'RJ','Rio de Janeiro'),(35,'SP','Sao Paulo')],
    'Sul': [(41,'PR','Parana'),(42,'SC','Santa Catarina'),(43,'RS','Rio Grande do Sul')],
    'Centro-Oeste': [(50,'MS','Mato Grosso do Sul'),(51,'MT','Mato Grosso'),(52,'GO','Goias'),(53,'DF','Distrito Federal')],
}

def normalizar(v):
    s = unicodedata.normalize('NFKD', str(v).strip().lower())
    return ''.join(c for c in s if not unicodedata.combining(c))

MAPA = {normalizar(v): reg for reg, estados in REGIOES.items()
        for codigo, sigla, nome in estados for v in (codigo, float(codigo), sigla, nome)}


def agrupar(X):
    X = X.copy()
    uf = X['uf']
    reg = uf.map(lambda v: MAPA.get(normalizar(v)) if pd.notna(v) else np.nan)
    desconhecidos = uf[uf.notna() & reg.isna()].unique()
    if len(desconhecidos):
        raise ValueError(f'UFs nao reconhecidas: {desconhecidos}. Confira a codificacao original.')
    X['uf'] = reg.astype(object).where(reg.notna(), np.nan)
    return X


CHAVES = ['posicao_dev', 'fold', 'real']
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
        fpr=fp/(fp+tn), fnr=fn/(fn+tp), tn=tn, fp=fp, fn=fn, tp=tp)


def votos(g, k):
    g = g.copy()
    p = g.prob_ativo.to_numpy()
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Escores invalidos.')
    g['votos'] = np.rint(k*p).astype(int)
    if not np.allclose(p, g.votos/k, atol=1e-10, rtol=0):
        raise ValueError(f'Escores nao correspondem a {k} votos uniformes.')
    g['k'] = k
    return g


def executar():
    X, y = separar_xy(pd.read_parquet(ROOT/'data/interim/pns_base_eda.parquet'))
    assert X.shape[1] == 10
    X, _, y, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    R = agrupar(X)
    splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(R, y))
    esperado = pd.concat([pd.DataFrame(dict(posicao_dev=va, fold=f, real=y.iloc[va].to_numpy()))
                            for f, (_,va) in enumerate(splits, 1)]).sort_values('posicao_dev').reset_index(drop=True)
    anterior = pd.read_csv(PASTA/'regioes_balanceamento/PROBABILIDADES_OOF.csv')
    ref = anterior[anterior.configuracao == 'regioes'].sort_values('posicao_dev').reset_index(drop=True)
    if not np.array_equal(ref[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
        raise ValueError('Referencia k=5 desalinhada ou incompleta.')
    registros = [votos(ref[CHAVES+['prob_ativo']], 5)]
    print(f'REGIOES | k=3 versus k=5 | {len(R)} registros | 10 novos ajustes')
    print('Diagnostico extra: k alterado. Sem PCA, pesos extras ou resampling. Teste reservado intacto.')
    for f, (tr, va) in enumerate(splits, 1):
        yt = y.iloc[tr].to_numpy()
        original = construir_pipeline(escala='robust')
        assert original.named_steps['selecao'] == 'passthrough' and original.named_steps['pca'] == 'passthrough'
        pre = original[:-1]
        ztr = pre.fit_transform(R.iloc[tr], yt)
        zva = pre.transform(R.iloc[va])
        if ztr.shape[1] != 31:
            raise ValueError(f'Esperadas 31 dimensoes; obtidas {ztr.shape[1]}.')
        modelo = original.named_steps['knn']
        assert modelo.n_neighbors == 5 and modelo.weights == 'uniform' and modelo.p == 2
        modelo.set_params(n_neighbors=3)
        modelo.fit(ztr, yt)
        p = modelo.predict_proba(zva)[:, list(modelo.classes_).index(1)]
        g = votos(pd.DataFrame(dict(posicao_dev=va, fold=f, real=y.iloc[va].to_numpy(), prob_ativo=p)), 3)
        registros.append(g)
        r = avaliar(g, 2)
        print(f'{f}/10 | k=3 | AUC={r["auc"]:.4f} | AP={r["average_precision"]:.4f} | F1={r["f1"]:.4f} | Recall={r["recall"]:.4f} | FPR={r["fpr"]:.4f}', flush=True)
    oof = pd.concat(registros, ignore_index=True)
    linhas = []
    for k in [5, 3]:
        g = oof[oof.k == k].sort_values('posicao_dev')
        if not np.array_equal(g[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
            raise ValueError(f'OOF k={k} desalinhado.')
        for minimo in range(k, 0, -1):
            for f, parte in g.groupby('fold'):
                linhas.append(dict(k=k, votos_minimos=minimo, regra='padrao' if minimo==k//2+1 else 'ajustada',
                                   fold=f, **avaliar(parte, minimo)))
    d = pd.DataFrame(linhas)
    metricas = [m for m in d.columns if m not in ['k','votos_minimos','regra','fold']+MATRIZ]
    gb = d.groupby(['k','votos_minimos','regra'], sort=False)
    medias = gb[metricas].mean()
    pares = []
    a = d[(d.k==3)&(d.regra=='padrao')].set_index('fold')
    b = d[(d.k==5)&(d.regra=='padrao')].set_index('fold')
    for m in metricas:
        delta=a[m]-b[m]
        melhora=-delta if m in ['fpr','fnr'] else delta
        pares.append(dict(metrica=m, delta_k3_menos_k5=delta.mean(), dp_delta=delta.std(),
                    melhores=int((melhora>1e-12).sum()), piores=int((melhora< -1e-12).sum()),
                    empates=int((melhora.abs()<=1e-12).sum())))
    distrib = oof.groupby(['k','votos','real']).size().unstack(fill_value=0)
    SAIDA.mkdir(parents=True, exist_ok=True)
    tabelas = [('COMPARACAO_FINAL', medias), ('DESVIOS_PADRAO', gb[metricas].std()),
                ('MATRIZES_ACUMULADAS', gb[MATRIZ].sum()), ('MATRIZES_MEDIAS', gb[MATRIZ].mean()),
                ('COMPARACAO_PAREADA_PADRAO', pd.DataFrame(pares)), ('DISTRIBUICAO_VOTOS', distrib)]
    for nome, tabela in tabelas:
        print('\n=== '+nome+' ===\n'+tabela.round(4).to_string())
        tabela.to_csv(SAIDA/(nome+'.csv'))
    d.to_csv(SAIDA/'METRICAS_POR_FOLD.csv', index=False)
    oof.sort_values(['k','posicao_dev']).to_csv(SAIDA/'PROBABILIDADES_OOF.csv', index=False)
    print('\nCada k tem sua propria escala discreta de votos. AUC/AP nao mudam com o corte.')
    print('Comparacao exploratoria; folds dependentes. k=3 nao conta para os experimentos oficiais.')
    print('Resultados:', SAIDA)


if __name__=='__main__':
    executar()
