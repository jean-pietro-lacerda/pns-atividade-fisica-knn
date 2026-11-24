"""TSTR exploratorio PNS: kNN treinado exclusivamente em exemplos gerados.
Requer testar_smotenc_tomek_pedro.py na mesma pasta (script anterior).
Nao modifica a base, src/ ou o teste reservado. --autoteste: dados ficticios.
"""
from pathlib import Path
import argparse, hashlib, json, tempfile
from time import perf_counter
import numpy as np
import pandas as pd
import sklearn, imblearn, scipy
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTENC
from scipy.stats import wilcoxon
from testar_smotenc_tomek_pedro import NUM, CAT, COLS, SEED, METRICAS, agrupar_uf, separar, medir, holm

ROOT = Path(__file__).resolve().parent
CONFIGS = ['real_72_28','tstr_proporcao_real','tstr_50_50']


def pipeline():
    return Pipeline([
        ('pre',ColumnTransformer([
            ('num',Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',RobustScaler())]),NUM),
            ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)
        ],remainder='drop')),
        ('knn',KNeighborsClassifier())])


def gerar(X,y,fold):
    # Ajuste do gerador APENAS no treino real do fold.
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    scale=RobustScaler(); nums=scale.fit_transform(imp.fit_transform(X[NUM]))
    cat=SimpleImputer(strategy='most_frequent',keep_empty_features=True).fit_transform(X[CAT])
    mix=pd.DataFrame(nums,columns=NUM)
    for j,c in enumerate(CAT): mix[c]=cat[:,j].astype(str)
    n=len(y); orig=np.bincount(y,minlength=2); bal=np.array([n//2,n-n//2]); need=np.maximum(orig,bal)
    if orig.min()<6: raise ValueError('SMOTENC requer pelo menos 6 registros por classe no treino')
    # SMOTENC gera as duas classes, nao so a minoritaria. Cada classe
    # recebe quantidade suficiente para ambos os cenarios TSTR.
    xm,ym=SMOTENC(categorical_features=list(range(len(NUM),len(COLS))),
        sampling_strategy={i:int(orig[i]+need[i]) for i in (0,1)},
        random_state=SEED+fold,k_neighbors=5).fit_resample(mix,y)
    ym=np.asarray(ym,int)
    assert np.array_equal(ym[:n],y)
    assert np.array_equal(np.bincount(ym[n:],minlength=2),need)
    # Verifica a ordenacao das linhas antes de descartar TODOS os originais.
    assert np.allclose(xm.iloc[:n][NUM].to_numpy(float),mix[NUM].to_numpy(float))
    assert np.array_equal(xm.iloc[:n][CAT].to_numpy(str),mix[CAT].to_numpy(str))
    z=xm.iloc[n:].reset_index(drop=True).copy(); labels=ym[n:]
    z[NUM]=scale.inverse_transform(z[NUM].to_numpy(float))
    # Variaveis discretas de contagem/idade permanecem inteiras.
    for c in ['idade','numero_moradores']:
        z[c]=z[c].round().clip(X[c].min(),X[c].max())
    for c in CAT:
        assert set(z[c]).issubset(set(mix[c])),c
    rng=np.random.default_rng(SEED+fold)
    pools=[rng.permutation(np.flatnonzero(labels==i)) for i in (0,1)]
    def amostra(count):
        idx=rng.permutation(np.concatenate([pools[i][:count[i]] for i in (0,1)]))
        assert len(idx)==n and np.array_equal(np.bincount(labels[idx],minlength=2),count)
        return z.iloc[idx].reset_index(drop=True),labels[idx]
    return amostra(orig),amostra(bal)


def rodar(X,y,out,folds):
    X=agrupar_uf(X).reset_index(drop=True); y=np.asarray(y,int)
    out.mkdir(parents=True,exist_ok=True); cache=out/'checkpoints'; cache.mkdir(exist_ok=True)
    h=hashlib.sha256(Path(__file__).read_bytes()); h.update((ROOT/'testar_smotenc_tomek_pedro.py').read_bytes());h.update(pd.util.hash_pandas_object(X,index=False).to_numpy().tobytes());h.update(y.tobytes())
    manifest=dict(assinatura=h.hexdigest(),folds=folds,seed=SEED,sklearn=sklearn.__version__,imblearn=imblearn.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__)
    mp=cache/'manifesto.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest: raise RuntimeError('Cache de outra base/versao: renomeie a pasta de resultados.')
    if not mp.exists() and list(cache.glob('*.npz')): raise RuntimeError('Cache sem manifesto')
    mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    linhas=[]; oof=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(n_splits=folds,shuffle=True,random_state=SEED).split(X,y),1):
        file=cache/f'fold_{fold:02d}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as a:
                assert np.array_equal(a['va'],va) and np.array_equal(a['real'],y[va])
                probs={c:a[c].copy() for c in CONFIGS}; counts=json.loads(str(a['counts'].item()))
            print(f'Fold {fold} recuperado',flush=True)
        else:
            Xt=X.iloc[tr]; yt=y[tr]; Xv=X.iloc[va]
            print(f'Fold {fold}/{folds} | gerando exemplos das duas classes',flush=True)
            t=perf_counter(); s1,s2=gerar(Xt,yt,fold); print(f'  Geracao: {perf_counter()-t:.1f}s',flush=True)
            probs={}; counts={}
            for nome,(xx,yy) in zip(CONFIGS,[(Xt,yt),s1,s2]):
                print(f'  Treinando {nome}',flush=True)
                # Preprocessador do classificador ajustado exclusivamente
                # nos seus proprios dados de treino (sinteticos no TSTR).
                model=pipeline().fit(xx,yy)
                probs[nome]=model.predict_proba(Xv)[:,1]
                counts[nome]={'treino_0':int((yy==0).sum()),'treino_1':int((yy==1).sum()),'linhas_originais_no_treino_knn':len(yy) if nome==CONFIGS[0] else 0}
            tmp=cache/f'tmp_{fold:02d}.npz'; np.savez_compressed(tmp,va=va,real=y[va],counts=json.dumps(counts),**probs);tmp.replace(file)
        for nome in CONFIGS:
            m=medir(y[va],probs[nome]); linhas.append(dict(configuracao=nome,fold=fold,**counts[nome],**m))
            oof.append(pd.DataFrame(dict(configuracao=nome,fold=fold,posicao_dev=va,real=y[va],prob_ativo=probs[nome])))
            print(f'{fold}/{folds} {nome:24} AUC={m["auc"]:.4f} F1={m["f1"]:.4f} Recall={m["recall"]:.4f} FP={m["fp"]}',flush=True)
    df=pd.DataFrame(linhas); oo=pd.concat(oof,ignore_index=True);grp=df.groupby('configuracao');mat=['tn','fp','fn','tp']
    df.to_csv(out/'METRICAS_POR_FOLD.csv',index=False);oo.to_csv(out/'PROBABILIDADES_OOF.csv',index=False)
    for nome,t in [('MEDIAS',grp[METRICAS].mean()),('DESVIOS_PADRAO',grp[METRICAS].std()),('MATRIZES_MEDIAS',grp[mat].mean()),('MATRIZES_ACUMULADAS',grp[mat].sum())]:t.reindex(CONFIGS).to_csv(out/f'{nome}.csv')
    glob=[]
    for nome,g in oo.groupby('configuracao'):
        g=g.sort_values('posicao_dev');assert np.array_equal(g.posicao_dev,np.arange(len(y))) and np.array_equal(g.real,y)
        glob.append(dict(configuracao=nome,**medir(g.real,g.prob_ativo)))
    glob=pd.DataFrame(glob).set_index('configuracao').reindex(CONFIGS);glob.to_csv(out/'METRICAS_OOF_GLOBAIS.csv')
    w=[]
    for metrica in ['auc','f1']:
        ref=df[df.configuracao==CONFIGS[0]].set_index('fold')[metrica]; rows=[]
        for nome in CONFIGS[1:]:
            d=np.round((df[df.configuracao==nome].set_index('fold')[metrica]-ref).to_numpy(),12)
            rows.append(dict(metrica=metrica,configuracao=nome,delta_medio=d.mean(),p_bruto=1. if np.all(d==0) else float(wilcoxon(d).pvalue)))
        for r,p in zip(rows,holm([r['p_bruto'] for r in rows])):r['p_holm']=p
        w.extend(rows)
    pd.DataFrame(w).to_csv(out/'WILCOXON_PAREADO.csv',index=False)
    print('\n=== MEDIA DOS FOLDS ===\n',grp[METRICAS].mean().reindex(CONFIGS).round(4).to_string())
    print('\n=== OOF GLOBAL ===\n',glob.round(4).to_string())
    print('\n=== WILCOXON EXPLORATORIO ===\n',pd.DataFrame(w).round(5).to_string(index=False))
    print('Resultados:',out,'\nTeste reservado NAO utilizado. Folds parcialmente dependentes.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--autoteste',action='store_true');args=ap.parse_args()
    if args.autoteste:
        r=np.random.default_rng(SEED);n=360
        b=pd.DataFrame({'idade':r.integers(18,85,n),'renda_per_capita':r.integers(200,5000,n),'numero_moradores':r.integers(1,8,n)})
        for c in CAT:b[c]=r.choice(['A','B','C'],n)
        b['uf']=r.choice(['26','35','41','11','50'],n); b['ativo_lazer']=(r.random(n)<.28).astype(int)
        X,y=separar(b)
        with tempfile.TemporaryDirectory() as d:rodar(X,y,Path(d),3)
        print('AUTOTESTE OK. Dados ficticios, nao resultados PNS.');return
    X,y=separar(pd.read_parquet(ROOT/'data/interim/pns_base_eda.parquet'))
    if len(y)!=88456 or y.value_counts().to_dict()!={0:63391,1:25065}:raise ValueError('Base difere da referencia; nao comparar automaticamente.')
    X,_,y,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    assert len(y)==70764 and y.value_counts().to_dict()=={0:50712,1:20052}
    print('PNS REAL | TSTR SMOTENC | 5 regioes | 10 folds | kNN padrao | teste final intacto',flush=True)
    rodar(X,y,ROOT/'outputs/diagnostico_pedro/tstr_smotenc',10)

if __name__=='__main__':main()