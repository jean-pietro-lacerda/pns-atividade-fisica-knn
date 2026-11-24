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
SAIDA = PASTA / 'regioes_balanceamento'
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


def avaliar(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(acuracia=accuracy_score(y,pred), acuracia_balanceada=balanced_accuracy_score(y,pred),
        precisao=precision_score(y,pred,zero_division=0), recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0), f2=fbeta_score(y,pred,beta=2,zero_division=0),
        auc=roc_auc_score(y,p), average_precision=average_precision_score(y,p),
        fpr=fp/(fp+tn), tn=tn, fp=fp, fn=fn, tp=tp)


def balancear(y, fold):
    rng = np.random.default_rng(42+fold)
    y = np.asarray(y)
    a, b = np.flatnonzero(y==0), np.flatnonzero(y==1)
    if len(a)<len(b): a,b=b,a
    indices = np.concatenate([np.arange(len(y)), rng.choice(b, len(a)-len(b), replace=True)])
    return rng.permutation(indices)


def executar():
    X,y = separar_xy(pd.read_parquet(ROOT/'data/interim/pns_base_eda.parquet'))
    assert X.shape[1]==10
    X,_,y,_ = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    R = agrupar(X)
    print('Regioes no desenvolvimento:', R.uf.value_counts(dropna=False).to_dict())
    splits = list(StratifiedKFold(n_splits=10,shuffle=True,random_state=42).split(X,y))
    ref = pd.read_csv(PASTA/'limiares/PROBABILIDADES_OOF.csv').sort_values('posicao_dev').reset_index(drop=True)
    esperado = pd.concat([pd.DataFrame(dict(posicao_dev=va,fold=f,real=y.iloc[va].to_numpy()))
        for f,(_,va) in enumerate(splits,1)]).sort_values('posicao_dev').reset_index(drop=True)
    chaves=['posicao_dev','fold','real']
    if not np.array_equal(ref[chaves].to_numpy(),esperado[chaves].to_numpy()):
        raise ValueError('Referencia OOF desalinhada.')
    resultados=[]; registros=[]
    for f,g in ref.groupby('fold'):
        resultados.append(dict(configuracao='uf_original',fold=f,**avaliar(g.real,g.prob_ativo.to_numpy())))
    registros.append(ref[chaves+['prob_ativo']].assign(configuracao='uf_original'))
    configs=[('regioes',R,False),('uf_balanceado',X,True),('regioes_balanceado',R,True)]
    print('30 novos ajustes | Robust | kNN padrao | teste reservado intacto')
    for f,(tr,va) in enumerate(splits,1):
        yt=y.iloc[tr].to_numpy(); real=y.iloc[va].to_numpy()
        idx=balancear(yt,f)
        for nome,base,ros in configs:
            original=construir_pipeline(escala='robust')
            pre=original[:-1]; knn=original.named_steps['knn']
            # Aprende imputacao, escala e OHE apenas no treino REAL.
            ztr=pre.fit_transform(base.iloc[tr],yt)
            zva=pre.transform(base.iloc[va])
            ii=idx if ros else np.arange(len(yt))
            knn.fit(ztr[ii],yt[ii])
            p=knn.predict_proba(zva)[:,list(knn.classes_).index(1)]
            r=dict(configuracao=nome,fold=f,dimensoes=ztr.shape[1],
                   treino_0=int((yt[ii]==0).sum()),treino_1=int((yt[ii]==1).sum()),**avaliar(real,p))
            resultados.append(r)
            registros.append(pd.DataFrame(dict(configuracao=nome,posicao_dev=va,fold=f,real=real,prob_ativo=p)))
            print(f'{f}/10 | {nome} | dim={r["dimensoes"]} | treino={r["treino_0"]}/{r["treino_1"]} | AUC={r["auc"]:.4f} | F1={r["f1"]:.4f} | Recall={r["recall"]:.4f}',flush=True)
    d=pd.DataFrame(resultados)
    ordem=['uf_original']+[c[0] for c in configs]
    metricas=list(avaliar(np.array([0,1]),np.array([0.,1.])).keys())[:-4]
    medias=d.groupby('configuracao')[metricas].mean().reindex(ordem)
    dp=d.groupby('configuracao')[metricas].std().reindex(ordem)
    matrizes=d.groupby('configuracao')[['tn','fp','fn','tp']].mean().reindex(ordem)
    pares=[]
    for nome in ordem[1:]:
        for m in metricas:
            delta=d[d.configuracao==nome].set_index('fold')[m]-d[d.configuracao=='uf_original'].set_index('fold')[m]
            pares.append(dict(configuracao=nome,metrica=m,delta_medio=delta.mean(),dp_delta=delta.std(),maior=int((delta>1e-12).sum()),menor=int((delta< -1e-12).sum())))
    SAIDA.mkdir(parents=True,exist_ok=True)
    for nome,t in [('COMPARACAO_FINAL',medias),('DESVIOS_PADRAO',dp),('MATRIZES_MEDIAS',matrizes),('MATRIZES_ACUMULADAS',matrizes.mul(10).round().astype(int)),('DELTA_REFERENCIA',medias-medias.loc['uf_original']),('COMPARACAO_PAREADA',pd.DataFrame(pares))]:
        print('\n=== '+nome+' ===\n'+t.round(4).to_string()); t.to_csv(SAIDA/(nome+'.csv'))
    d.to_csv(SAIDA/'METRICAS_POR_FOLD.csv',index=False)
    pd.concat(registros,ignore_index=True).sort_values(['configuracao','posicao_dev']).to_csv(SAIDA/'PROBABILIDADES_OOF.csv',index=False)
    print('\nComparacao exploratoria; folds dependentes. Teste reservado e codigo oficial intactos.')
    print('Resultados:',SAIDA)

if __name__=='__main__':
    executar()
