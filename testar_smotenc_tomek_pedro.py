"""PNS 2019: diagnostico complementar de reamostragem para kNN.
Rode na raiz do repositorio. Nao altera arquivos oficiais nem avalia o teste.
--autoteste usa somente dados artificiais, em pasta temporaria.
"""
from pathlib import Path
import argparse, hashlib, json, tempfile, unicodedata
from time import perf_counter
import numpy as np
import pandas as pd
import sklearn, scipy, imblearn
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score)
from imblearn.over_sampling import SMOTENC, RandomOverSampler

ROOT = Path(__file__).resolve().parent
NUM = ['idade','renda_per_capita','numero_moradores']
CAT = ['sexo','escolaridade','uf','situacao_trabalho','area_residencia','raca_cor','estado_civil']
COLS = NUM + CAT
CONFIGS = ['knn','tomek','random_oversampling','smotenc','smotenc_tomek']
METRICAS = ['acuracia','precisao','recall','f1','fpr','auc','average_precision']
SEED = 42

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


MAPA = {normalizar(v): reg for reg, items in REGIOES.items()
        for cod, sigla, nome in items for v in (cod, float(cod), sigla, nome)}


def agrupar_uf(X):
    out = X.copy()
    reg = out['uf'].map(lambda v: MAPA.get(normalizar(v)) if pd.notna(v) else np.nan)
    desconhecidas = out.loc[out['uf'].notna() & reg.isna(), 'uf'].unique()
    if len(desconhecidas):
        raise ValueError(f'UF desconhecida: {desconhecidas!r}. Dados nao alterados.')
    out['uf'] = reg.astype(object).where(reg.notna(), np.nan)
    return out



def separar(base):
    X = base[COLS].copy()
    y = base['ativo_lazer']
    if not y.isin([0,1]).all():
        raise ValueError('Alvo incompleto ou invalido')
    for c in NUM:
        X[c] = X[c].astype(float)
    for c in CAT:
        X[c] = X[c].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    if np.isinf(X[NUM].to_numpy()).any():
        raise ValueError('Infinito na base')
    return X, y.astype(int)


def limpar(z,y):
    # Mesma regra de pares mutuos do diagnostico anterior. Remove somente
    # classe 0 (inativos), inclusive apos SMOTENC quando classes empatam.
    viz = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z)
    ids = viz.kneighbors(X=None, n_neighbors=1, return_distance=False)[:,0]
    manter = ~((ids[ids] == np.arange(len(y))) & (y != y[ids]) & (y == 0))
    return z[manter], y[manter]


def medir(y,p):
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(acuracia=accuracy_score(y,pred),precisao=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
        fpr=fp/(fp+tn),auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def ajustar(Xtr,Xva,yt,fold):
    # Ajuste exclusivamente no treino REAL. Equivalente ao pre-processamento
    # robust/mediana/moda/OHE do projeto, com extremos preservados.
    num=Pipeline([('imputacao',SimpleImputer(strategy='median',keep_empty_features=True)),
                  ('escala',RobustScaler())])
    cat=SimpleImputer(strategy='most_frequent',keep_empty_features=True)
    nt=num.fit_transform(Xtr[NUM]); nv=num.transform(Xva[NUM])
    ct=cat.fit_transform(Xtr[CAT]); cv=cat.transform(Xva[CAT])
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(ct)
    def codificar(n,c):
        return np.ascontiguousarray(np.column_stack([np.asarray(n,dtype=float),enc.transform(c)]),dtype=float)
    ztr=codificar(nt,ct); zva=codificar(nv,cv)
    # SMOTENC recebe numericas escaladas e categorias discretas, NAO OHE.
    mix=pd.DataFrame(nt,columns=NUM)
    for j,c in enumerate(CAT): mix[c]=ct[:,j].astype(str)
    if np.bincount(yt,minlength=2)[0] <= np.bincount(yt,minlength=2)[1]:
        raise ValueError('Esperava classe 0 majoritaria antes da reamostragem')
    ydict={}; probs={}; stats={}
    def executar(nome,z,yy,gerados=0,removidos=0):
        t=perf_counter()
        yy=np.asarray(yy,dtype=int)
        clf=KNeighborsClassifier().fit(z,yy)
        p=clf.predict_proba(zva)[:,1]
        probs[nome]=p
        stats[nome]={'treino_0':int((yy==0).sum()),'treino_1':int((yy==1).sum()),
                     'gerados_ou_duplicados':int(gerados),'removidos':int(removidos),
                     'segundos_knn':perf_counter()-t}
    print(f'Fold {fold} | kNN referencia',flush=True)
    executar('knn',ztr,yt)
    print(f'Fold {fold} | Tomek',flush=True)
    z,y=limpar(ztr,yt); executar('tomek',z,y,removidos=len(yt)-len(y))
    print(f'Fold {fold} | Random oversampling',flush=True)
    z,y=RandomOverSampler(sampling_strategy=1.0,random_state=SEED+fold).fit_resample(ztr,yt)
    executar('random_oversampling',z,y,gerados=len(y)-len(yt))
    print(f'Fold {fold} | SMOTENC',flush=True)
    sm=SMOTENC(categorical_features=list(range(len(NUM),len(COLS))),sampling_strategy=1.0,
               random_state=SEED+fold,k_neighbors=5)
    xm,ym=sm.fit_resample(mix,yt); ym=np.asarray(ym,dtype=int)
    assert np.bincount(ym,minlength=2)[0]==np.bincount(ym,minlength=2)[1]
    assert np.array_equal(ym[:len(yt)],yt)
    for c in CAT:
        assert set(xm[c]).issubset(set(mix[c])),c
    zs=codificar(xm[NUM],xm[CAT]); gerados=len(ym)-len(yt)
    executar('smotenc',zs,ym,gerados=gerados)
    print(f'Fold {fold} | SMOTENC + Tomek (remove classe 0)',flush=True)
    zl,yl=limpar(zs,ym)
    executar('smotenc_tomek',zl,yl,gerados=gerados,removidos=len(ym)-len(yl))
    return probs,stats


def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); m=len(p)
    out=np.empty(m); out[order]=np.minimum(1,np.maximum.accumulate((m-np.arange(m))*p[order])); return out


def executar(X,y,destino,n_folds=10):
    X=agrupar_uf(X).reset_index(drop=True); y=np.asarray(y,int)
    destino.mkdir(parents=True,exist_ok=True)
    cache=destino/'checkpoints'; cache.mkdir(exist_ok=True)
    versoes=dict(sklearn=sklearn.__version__,imblearn=imblearn.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__)
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(pd.util.hash_pandas_object(X,index=False).to_numpy().tobytes());h.update(y.tobytes())
    manifest=dict(assinatura=h.hexdigest(),folds=n_folds,configs=CONFIGS,seed=SEED,versoes=versoes)
    ident=cache/'manifesto.json'
    if ident.exists() and json.loads(ident.read_text())!=manifest:
        raise RuntimeError('Checkpoints de outra versao/base. Renomeie a pasta de saida antes de repetir.')
    if not ident.exists() and list(cache.glob('*.npz')): raise RuntimeError('Cache sem manifesto')
    ident.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    linhas=[]; oof=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=SEED).split(X,y),1):
        arq=cache/f'fold_{fold:02d}.npz'
        if arq.exists():
            with np.load(arq,allow_pickle=False) as a:
                assert np.array_equal(a['va'],va) and np.array_equal(a['real'],y[va])
                probs={c:a[c].copy() for c in CONFIGS}; stats=json.loads(str(a['stats'].item()))
            print(f'Fold {fold} recuperado.',flush=True)
        else:
            probs,stats=ajustar(X.iloc[tr],X.iloc[va],y[tr],fold)
            tmp=cache/f'tmp_{fold:02d}.npz'
            np.savez_compressed(tmp,va=va,real=y[va],stats=json.dumps(stats),**probs);tmp.replace(arq)
        for nome in CONFIGS:
            p=probs[nome]; m=medir(y[va],p)
            linhas.append(dict(configuracao=nome,fold=fold,**stats[nome],**m))
            oof.append(pd.DataFrame(dict(configuracao=nome,fold=fold,posicao_dev=va,real=y[va],prob_ativo=p)))
            print(f'{fold}/{n_folds} {nome:21s} AUC={m["auc"]:.4f} F1={m["f1"]:.4f} Recall={m["recall"]:.4f} FP={m["fp"]}',flush=True)
    df=pd.DataFrame(linhas); oo=pd.concat(oof,ignore_index=True)
    df.to_csv(destino/'METRICAS_POR_FOLD.csv',index=False)
    oo.to_csv(destino/'PROBABILIDADES_OOF.csv',index=False)
    grp=df.groupby('configuracao'); mat=['tn','fp','fn','tp']
    medias=grp[METRICAS].mean().reindex(CONFIGS); dp=grp[METRICAS].std().reindex(CONFIGS)
    mm=grp[mat].mean().reindex(CONFIGS); soma=grp[mat].sum().reindex(CONFIGS)
    for nome,t in [('MEDIAS',medias),('DESVIOS_PADRAO',dp),('MATRIZES_MEDIAS',mm),('MATRIZES_ACUMULADAS',soma)]:t.to_csv(destino/f'{nome}.csv')
    pooled=[]
    for nome,g in oo.groupby('configuracao'):
        g=g.sort_values('posicao_dev'); assert np.array_equal(g.posicao_dev,np.arange(len(y))) and np.array_equal(g.real,y)
        pooled.append(dict(configuracao=nome,**medir(g.real,g.prob_ativo)))
    pooled=pd.DataFrame(pooled).set_index('configuracao').reindex(CONFIGS); pooled.to_csv(destino/'METRICAS_OOF_GLOBAIS.csv')
    # Wilcoxon bilateral, mesmos folds, Holm por familia de 4 comparacoes;
    # F1 primaria predefinida; AUC secundaria. Folds nao independentes.
    w=[]
    for metrica in ['f1','auc']:
        inicio=len(w); ref=df[df.configuracao=='knn'].set_index('fold')[metrica]
        for nome in CONFIGS[1:]:
            delta=df[df.configuracao==nome].set_index('fold')[metrica]-ref
            delta=np.round(delta.to_numpy(),12)
            p=1.0 if np.all(delta==0) else float(wilcoxon(delta,alternative='two-sided').pvalue)
            w.append(dict(metrica=metrica,configuracao=nome,delta_medio=float(delta.mean()),p_bruto=p))
        for r,p in zip(w[inicio:],holm([r['p_bruto'] for r in w[inicio:]])):r['p_holm']=p
    pd.DataFrame(w).to_csv(destino/'WILCOXON_PAREADO.csv',index=False)
    resumo=pooled.copy(); resumo['delta_fp']=resumo.fp-resumo.loc['knn','fp'];resumo['delta_fn']=resumo.fn-resumo.loc['knn','fn']
    print('\n=== MEDIA DOS FOLDS ===\n',medias.round(4).to_string())
    print('\n=== OOF GLOBAL: COMPARACAO E MATRIZES ACUMULADAS ===\n',resumo.round(4).to_string())
    print('\n=== WILCOXON EXPLORATORIO (HOLM POR METRICA) ===\n',pd.DataFrame(w).round(5).to_string(index=False))
    print('\nResultados:',destino,'\nTeste final NAO utilizado. Folds parcialmente dependentes.',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--autoteste',action='store_true');a=ap.parse_args()
    if a.autoteste:
        rng=np.random.default_rng(SEED); n=400
        base=pd.DataFrame({'idade':rng.integers(18,85,n),'renda_per_capita':rng.integers(250,5000,n),'numero_moradores':rng.integers(1,7,n)})
        for c in CAT:base[c]=rng.choice(['A','B','C'],n)
        base['uf']=rng.choice(['26','35','41','11','50'],n)
        base['ativo_lazer']=(rng.random(n)<.28).astype(int)
        X,y=separar(base)
        with tempfile.TemporaryDirectory() as t:executar(X,y,Path(t),3)
        print('AUTOTESTE OK — dados artificiais; NAO sao resultados da PNS.')
        return
    X,y=separar(pd.read_parquet(ROOT/'data/interim/pns_base_eda.parquet'))
    if len(y)!=88456 or y.value_counts().to_dict()!={0:63391,1:25065}:
        raise ValueError('Base difere da referencia de 88.456 registros. Nao comparar automaticamente.')
    X,_,y,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    assert len(y)==70764 and y.value_counts().to_dict()=={0:50712,1:20052}
    print('PNS REAL | 5 regioes | 10 folds | 5 configuracoes | kNN padrao | teste reservado intacto',flush=True)
    executar(X,y,ROOT/'outputs/diagnostico_pedro/smotenc_tomek')

if __name__=='__main__':main()
