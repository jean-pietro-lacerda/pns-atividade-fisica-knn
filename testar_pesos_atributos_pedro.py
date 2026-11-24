from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_auc_score, average_precision_score, confusion_matrix)
from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
PASTA = ROOT / 'outputs/diagnostico_pedro'
SAIDA = PASTA / 'regioes_pesos_limiares'
REF = 'regioes_sem_peso_extra'
RELEVANTES = dict(escolaridade=2., idade=2., sexo=2., renda_per_capita=2.)
REDUZIDOS = dict(numero_moradores=.5, raca_cor=.5, uf=.5)
CONFIGS = {
    'A_escolaridade_2': dict(escolaridade=2.),
    'B_escolaridade_4': dict(escolaridade=4.),
    'C_relevantes_2': RELEVANTES,
    'D_relevantes_2_escolaridade_4': {**RELEVANTES, 'escolaridade': 4.},
    'E_reduzir_interferencia': REDUZIDOS,
    'F_relevantes_2_reduzidos': {**RELEVANTES, **REDUZIDOS},
}
LIMIARES = {0.5: 3, 0.4: 2, 0.2: 1}
CHAVES = ['posicao_dev', 'fold', 'real']
METRICAS = ['acuracia', 'acuracia_balanceada', 'precisao', 'recall',
            'f1', 'f2', 'auc', 'average_precision', 'fpr']
MATRIZ = ['tn', 'fp', 'fn', 'tp']

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


def preparar_oof(g):
    g = g.copy()
    prob = g.prob_ativo.to_numpy()
    if not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
        raise ValueError('Probabilidades invalidas.')
    g['votos'] = np.rint(prob * 5).astype(int)
    if not np.allclose(prob, g.votos / 5):
        raise ValueError('Probabilidades nao correspondem a cinco votos uniformes.')
    return g


def mapear_colunas(ct, n_colunas, atributos):
    """Identifica os blocos usando o ColumnTransformer realmente ajustado.

    Numericas: uma saida por atributo, na ordem original.
    Categoricas: todas as dummies de cada atributo recebem o mesmo peso.
    Falha explicitamente se a estrutura nao for a esperada.
    """
    grupos = np.full(n_colunas, '', dtype=object)
    for nome, trans, cols in ct.transformers_:
        if nome == 'remainder':
            if len(cols) and trans != 'drop':
                raise ValueError('Remainder inesperado.')
            continue
        cols = list(cols)
        faixa = ct.output_indices_[nome]
        largura = faixa.stop - faixa.start
        if not largura:
            continue
        etapas = [t for _, t in trans.steps] if hasattr(trans, 'steps') else [trans]
        encs = [t for t in etapas if isinstance(t, OneHotEncoder)]
        if encs:
            enc = encs[0]
            if len(encs) != 1 or enc.drop_idx_ is not None or enc.min_frequency is not None or enc.max_categories is not None:
                raise ValueError('Esperado OneHotEncoder completo, sem agrupamento ou drop.')
            etiquetas = [col for col, cats in zip(cols, enc.categories_) for _ in cats]
        else:
            etiquetas = cols
        if len(etiquetas) != largura or not set(etiquetas).issubset(atributos):
            raise ValueError(f'Mapeamento inesperado no bloco {nome}: {etiquetas}')
        grupos[faixa] = etiquetas
    if set(grupos) != set(atributos):
        raise ValueError(f'Mapeamento incompleto: {grupos}')
    return grupos


def executar():
    X, y = separar_xy(pd.read_parquet(ROOT / 'data/interim/pns_base_eda.parquet'))
    assert X.shape[1] == 10
    X, _, y, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    R = agrupar(X)
    splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(R, y))
    esperado = pd.concat([pd.DataFrame(dict(posicao_dev=va, fold=f, real=y.iloc[va].to_numpy()))
                          for f, (_, va) in enumerate(splits, 1)]).sort_values('posicao_dev').reset_index(drop=True)
    anterior = pd.read_csv(PASTA / 'regioes_balanceamento/PROBABILIDADES_OOF.csv')
    ref = anterior[anterior.configuracao == 'regioes'].sort_values('posicao_dev').reset_index(drop=True)
    if not np.array_equal(ref[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
        raise ValueError('Referencia das regioes desalinhada ou incompleta.')
    registros = [preparar_oof(ref[CHAVES + ['prob_ativo']].assign(configuracao=REF))]
    pesos_linhas = []
    print(f'REGIOES + PESOS + LIMIARES | {len(R)} registros | Robust | 10 folds')
    print(f'{len(CONFIGS)*len(splits)} novos ajustes | sem PCA ou resampling | teste reservado intacto', flush=True)
    for fold, (tr, va) in enumerate(splits, 1):
        yt = y.iloc[tr].to_numpy()
        real = y.iloc[va].to_numpy()
        original = construir_pipeline(escala='robust')
        assert [nome for nome, _ in original.steps] == ['preprocessamento', 'selecao', 'pca', 'knn']
        assert original.named_steps['selecao'] == 'passthrough' and original.named_steps['pca'] == 'passthrough'
        pre = original[:-1]
        ztr = np.asarray(pre.fit_transform(R.iloc[tr], yt), dtype=float)
        zva = np.asarray(pre.transform(R.iloc[va]), dtype=float)
        if ztr.shape[1] != 31:
            raise ValueError(f'Esperadas 31 dimensoes; obtidas {ztr.shape[1]}.')
        grupos = mapear_colunas(pre.named_steps['preprocessamento'], ztr.shape[1], R.columns)
        if fold == 1:
            print('Colunas por atributo:', pd.Series(grupos).value_counts().to_dict(), flush=True)
        for nome, pesos in CONFIGS.items():
            if not set(pesos).issubset(R.columns):
                raise ValueError(f'Atributos desconhecidos: {pesos}')
            # Peso w na distancia quadratica = multiplicar TODAS as colunas
            # do atributo por sqrt(w), depois de escala e OHE.
            fator = np.sqrt([pesos.get(a, 1.) for a in grupos])
            if not np.isfinite(fator).all() or (fator <= 0).any():
                raise ValueError('Pesos devem ser positivos e finitos.')
            for a in R.columns:
                pesos_linhas.append(dict(configuracao=nome, fold=fold, atributo=a,
                    peso=pesos.get(a, 1.), fator_coluna=np.sqrt(pesos.get(a, 1.)),
                    n_colunas=int((grupos == a).sum())))
            knn = clone(original.named_steps['knn'])
            assert knn.n_neighbors == 5 and knn.weights == 'uniform' and knn.p == 2
            knn.fit(np.ascontiguousarray(ztr * fator), yt)
            prob = knn.predict_proba(np.ascontiguousarray(zva * fator))[:, list(knn.classes_).index(1)]
            g = preparar_oof(pd.DataFrame(dict(configuracao=nome, posicao_dev=va, fold=fold, real=real, prob_ativo=prob)))
            registros.append(g)
            r = avaliar(g, 2)
            print(f'{fold}/10 | {nome} | AUC={r["auc"]:.4f} | AP={r["average_precision"]:.4f} | F1(0.4)={r["f1"]:.4f} | Recall(0.4)={r["recall"]:.4f} | FPR(0.4)={r["fpr"]:.4f}', flush=True)
    oof = pd.concat(registros, ignore_index=True)
    ordem = [REF] + list(CONFIGS)
    linhas = []
    for nome in ordem:
        g = oof[oof.configuracao == nome].sort_values('posicao_dev')
        if not np.array_equal(g[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
            raise ValueError(f'OOF desalinhado: {nome}')
        for limiar, minimo in LIMIARES.items():
            for fold, parte in g.groupby('fold'):
                linhas.append(dict(configuracao=nome, limiar=limiar, fold=fold, **avaliar(parte, minimo)))
    d = pd.DataFrame(linhas)
    gb = d.groupby(['configuracao', 'limiar'], sort=False)
    medias, dp = gb[METRICAS].mean(), gb[METRICAS].std()
    acumuladas, matrizes = gb[MATRIZ].sum(), gb[MATRIZ].mean()
    pares = []
    for nome in ordem[1:]:
        for limiar in LIMIARES:
            a = d[(d.configuracao == nome) & (d.limiar == limiar)].set_index('fold')
            b = d[(d.configuracao == REF) & (d.limiar == limiar)].set_index('fold')
            for m in ['auc', 'average_precision', 'f1', 'recall', 'acuracia_balanceada', 'fpr']:
                delta = a[m] - b[m]
                melhora = -delta if m == 'fpr' else delta
                pares.append(dict(configuracao=nome, limiar=limiar, metrica=m, delta_medio=delta.mean(),
                    dp_delta=delta.std(), melhores=int((melhora > 1e-12).sum()),
                    piores=int((melhora < -1e-12).sum()), empates=int((melhora.abs() <= 1e-12).sum())))
    SAIDA.mkdir(parents=True, exist_ok=True)
    tabelas = [('COMPARACAO_FINAL', medias), ('DESVIOS_PADRAO', dp),
               ('MATRIZES_ACUMULADAS', acumuladas), ('MATRIZES_MEDIAS', matrizes),
               ('DELTA_VS_MESMO_LIMIAR_REFERENCIA', medias.sub(medias.loc[REF], level='limiar')),
               ('COMPARACAO_PAREADA', pd.DataFrame(pares))]
    for nome, tabela in tabelas:
        print('\n=== ' + nome + ' ===\n' + tabela.round(4).to_string())
        tabela.to_csv(SAIDA / (nome + '.csv'))
    d.to_csv(SAIDA / 'METRICAS_POR_FOLD.csv', index=False)
    pd.DataFrame(pesos_linhas).to_csv(SAIDA / 'PESOS_POR_ATRIBUTO.csv', index=False)
    oof.sort_values(['configuracao','posicao_dev']).to_csv(SAIDA / 'PROBABILIDADES_OOF.csv', index=False)
    print('\nAUC/AP calculadas sobre escores, nao sobre classes previstas.')
    print('Comparacao exploratoria; folds dependentes. Teste reservado e codigo oficial intactos.')
    print('Resultados:', SAIDA)


if __name__ == '__main__':
    executar()
