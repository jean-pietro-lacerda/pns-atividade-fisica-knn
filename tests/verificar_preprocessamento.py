from pathlib import Path
import sys, os, json, hashlib, warnings
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
import pandas as pd
import sklearn, nbformat
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
from nbclient import NotebookClient
from sklearn.model_selection import train_test_split, cross_validate
from src.preprocessamento import *
OUT = ROOT / 'outputs/preprocessamento'
OUT.mkdir(parents=True, exist_ok=True)
checks = []
def ok(name):
    checks.append(name)
    print('OK:', name, flush=True)
path = ROOT/'data/interim/pns_base_eda.parquet'
before = hashlib.sha256(path.read_bytes()).hexdigest()
df = pd.read_parquet(path)
assert df.shape == (88456, 11) and not df.isna().any().any()
X, y = separar_xy(df)
assert list(X) == NUMERICAS + CATEGORICAS
assert ALVO not in X
assert all(r['imputacao'] == 'median' for r in configuracoes_experimentais())
ok('Base real lida; 88456 x 11; sem nulos; dez preditores explicitos')
a,b,ya,yb = train_test_split(X,y,train_size=240,test_size=60,stratify=y,random_state=42)
recipes = configuracoes_experimentais()
assert len(recipes) == len({json.dumps(r,sort_keys=True) for r in recipes}) == 385
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    representations = []
    geometries = []
    recipe_audit = []
    for recipe in recipes:
        m = construir_pipeline(**recipe).fit(a,ya)
        z = m[:-1].transform(b)
        assert np.isfinite(z).all() and z.shape[0] == 60
        assert m.predict(b).shape == (60,)
        assert m.predict_proba(b).shape == (60,2)
        assert m.named_steps['knn'].get_params() == KNeighborsClassifier().get_params()
        # Probes include train and held-out rows; diagnostic only, no ranking.
        probe = m[:-1].transform(pd.concat([a,b]))
        group = next((i for i,v in enumerate(representations)
                      if v.shape == probe.shape and np.allclose(v,probe,rtol=1e-7,atol=1e-9)),None)
        if group is None:
            group = len(representations); representations.append(probe)
        # Pairwise distances: also detect rotations/translations/global scaling
        # that preserve kNN neighborhoods. All 300 real probes, not predictions.
        from scipy.spatial.distance import pdist
        distances = pdist(probe)
        distances /= max(distances.max(),np.finfo(float).eps)
        geometry = next((i for i,v in enumerate(geometries)
                         if np.allclose(v,distances,rtol=1e-7,atol=1e-9)),None)
        if geometry is None:
            geometry = len(geometries); geometries.append(distances)
        recipe_audit.append(dict(recipe=recipe, columns=probe.shape[1],
            representation_group=group, geometry_group=geometry))
    (OUT/'equivalencias_configuracoes.json').write_text(json.dumps({
        'combinations':len(recipes), 'observed_representations':len(representations),
        'observed_normalized_geometries':len(geometries),
        'train_rows':240,'validation_rows':60,'rtol':1e-7,'atol':1e-9,
        'scope':'Diagnostic on fixed real sample; not proof for all rows/folds',
        'recipes':recipe_audit},ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(geometries) >= 150, len(geometries)
    ok(f'{len(recipes)} receitas testadas; {len(representations)} matrizes e {len(geometries)} geometrias normalizadas observadas em 300 registros reais')
    np.testing.assert_allclose(construir_preprocessador(imputacao='mean').fit_transform(a), construir_preprocessador(imputacao='median').fit_transform(a))
    ok('Mean/median equivalentes sem ausentes; mean segue disponivel fora da grade')
    for scale in ('standard','minmax','robust'):
        p = construir_preprocessador(escala=scale).fit(a,ya)
        z = p.transform(a)[:,:3]
        if scale == 'standard':
            np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-12)
            np.testing.assert_allclose(z.std(axis=0),1)
        if scale == 'minmax':
            np.testing.assert_allclose(z.min(axis=0),0,atol=1e-12)
            np.testing.assert_allclose(z.max(axis=0),1)
        cat = p.transform(a)[:,3:]
        assert set(np.unique(cat)) <= {0,1}
        np.testing.assert_allclose(cat.sum(axis=1),7)
        num = p.named_transformers_['numericas']
        snapshot = num.named_steps['imputacao'].statistics_.copy()
        shifted = b.copy(); shifted['renda_per_capita'] = 1e12
        shifted['uf'] = 'categoria_nova'
        assert np.isfinite(p.transform(shifted)).all()
        np.testing.assert_array_equal(snapshot,num.named_steps['imputacao'].statistics_)
    ok('Standard media 0/desvio 1; MinMax [0,1]; Robust; one-hot binario; categoria nova; transform nao reajusta')
    for strategy in ('mean','median'):
        missing = a.copy()
        missing.loc[missing.index[:10], NUMERICAS+CATEGORICAS] = np.nan
        for c in NUMERICAS+CATEGORICAS:
            missing[c] = missing[c].astype(object) if c in CATEGORICAS else missing[c]
        p = construir_preprocessador(imputacao=strategy).fit(missing,ya)
        assert np.isfinite(p.transform(missing)).all()
        stats = p.named_transformers_['numericas'].named_steps['imputacao'].statistics_
        expected = missing[NUMERICAS].mean() if strategy=='mean' else missing[NUMERICAS].median()
        np.testing.assert_allclose(stats, expected)
        empty = a.copy(); empty[NUMERICAS+CATEGORICAS] = np.nan
        assert np.isfinite(construir_preprocessador(imputacao=strategy).fit_transform(empty,ya)).all()
    ok('Ausencias injetadas: media/mediana e moda; todas as colunas vazias preservadas')
    bad = df.iloc[:20].copy(); bad.loc[bad.index[0],ALVO] = pd.NA
    try: separar_xy(bad)
    except ValueError: pass
    else: raise AssertionError('Alvo ausente aceito')
    for kwargs in ({'escala':'invalida'}, {'imputacao':'moda'}, {'faixas':4}, {'faixas':3,'escala':'minmax'}, {'selecao':20}, {'pca':1}):
        try: construir_pipeline(**kwargs)
        except ValueError: pass
        else: raise AssertionError(kwargs)
    ok('Rejeicao de alvo ausente e parametros invalidos')
    for ext in ('preservar','log_renda','iqr'):
        t = TratamentoExtremos(ext).fit(a[NUMERICAS])
        raw = b[NUMERICAS].to_numpy()
        expected = raw.copy()
        if ext == 'log_renda': expected[:,1] = np.log1p(raw[:,1])
        if ext == 'iqr': expected = np.clip(raw,t.inferior_,t.superior_)
        np.testing.assert_allclose(t.transform(b[NUMERICAS]),expected)
        q1,q3 = np.quantile(a[NUMERICAS],[.25,.75],axis=0)
        np.testing.assert_allclose(t.superior_,q3+1.5*(q3-q1))
    catpipe = construir_preprocessador().fit(a,ya).named_transformers_['categoricas']
    missingcat = b[CATEGORICAS].copy(); missingcat.iloc[0,:] = np.nan
    np.testing.assert_array_equal(catpipe.named_steps['imputacao'].transform(missingcat)[0],catpipe.named_steps['imputacao'].statistics_)
    ok('Preservar, log1p e IQR com resultados esperados; limites somente do treino; moda categorica conferida')
    scores = cross_validate(construir_pipeline(),a,ya,cv=validacao_cruzada(),scoring=['accuracy','f1','roc_auc'],error_score='raise',return_estimator=True)
    assert all(np.isfinite(scores['test_'+s]).all() for s in ('accuracy','f1','roc_auc'))
    for (tr,te),est in zip(validacao_cruzada().split(a,ya),scores['estimator']):
        stats=est.named_steps['preprocessamento'].named_transformers_['numericas'].named_steps['imputacao'].statistics_
        np.testing.assert_allclose(stats,a.iloc[tr][NUMERICAS].median())
    ok('Integracao cross_validate: dez folds estratificados; estatisticas conferidas por treino; tres metricas finitas')
    p = construir_preprocessador().fit(X,y)
    full = p.transform(X)
    assert np.isfinite(full).all()
    ok(f'Preprocessador padrao na base completa: {full.shape}, {full.nbytes} bytes')
    nb = nbformat.read(ROOT/'notebooks/03_preprocessamento.ipynb',as_version=4)
    nbformat.validate(nb)
    client = NotebookClient(nb,timeout=600,resources={'metadata':{'path':str(ROOT)}})
    client.kernel_manager_class = __import__('jupyter_client').KernelManager
    client.execute()
    nbformat.write(nb,ROOT/'notebooks/03_preprocessamento.ipynb')
    images = sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
    assert images >= 2, f'Boxplots nao capturados: {images}'
    ok(f'Notebook executado por nbclient; {images} imagens PNG capturadas')
assert hashlib.sha256(path.read_bytes()).hexdigest() == before
ok('SHA256 da base inalterado')
for protected,digest in json.loads((OUT/'preservacao_revisao.json').read_text()).items():
    assert hashlib.sha256((ROOT/protected).read_bytes()).hexdigest() == digest
ok('Notebooks 01/02 e Parquet preservados por SHA256')
report={'python':sys.version,'executable':sys.executable,'sklearn':sklearn.__version__,'checks':checks,'sha256':before,'warnings':sorted(set(str(w.message) for w in captured)),'cv_smoke_means':{s:float(scores['test_'+s].mean()) for s in ('accuracy','f1','roc_auc')}}
report['configuration_audit'] = dict(combinations=len(recipes), observed_representations=len(representations), observed_normalized_geometries=len(geometries), rtol=1e-7, atol=1e-9, scope='Fixed 300-row real sample only')
(OUT/'validacao_final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('VALIDACAO CONCLUIDA',flush=True)