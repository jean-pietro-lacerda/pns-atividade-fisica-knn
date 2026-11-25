"""Diagnostico local: kNN 7/11/13, regioes, sem balanceamento/PCA.

Mantem os 10 atributos; compara pesos padrao e a ponderacao F previamente explorada.
Mostra TODOS os cortes de voto e escolhas com o MESMO TETO de FPR do k=5.
Nao altera os scripts oficiais, nao acessa o conjunto de teste reservado.
"""
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, fbeta_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs' / 'diagnostico_pedro'
SAIDA = BASE / 'regioes_knn_7_11_13'
KS = (7, 11, 13)
CONFIGS = {
    'regioes_simples': {},
    # Ponderacao F: melhor AUC na experiencia anterior de pesos, ganho modesto.
    # Peso significa multiplicador da CONTRIBUICAO quadratica de cada atributo.
    'regioes_pesos_F': {
        'escolaridade': 2.0, 'idade': 2.0, 'sexo': 2.0,
        'renda_per_capita': 2.0, 'numero_moradores': 0.5,
        'raca_cor': 0.5, 'uf': 0.5,
    },
}
CHAVES = ['posicao_dev', 'fold', 'real']
MATRIZ = ['tn', 'fp', 'fn', 'tp']
METRICAS = ['acuracia', 'acuracia_balanceada', 'precisao', 'recall',
            'f1', 'f2', 'auc', 'average_precision', 'fpr', 'fnr']

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
        raise ValueError(f'UFs desconhecidas: {desconhecidos}')
    X['uf'] = reg.astype(object).where(reg.notna(), np.nan)
    return X


def mapear_colunas(ct, n_colunas, atributos):
    grupos = np.full(n_colunas, '', dtype=object)
    for nome, trans, cols in ct.transformers_:
        if nome == 'remainder':
            if len(cols) and trans != 'drop':
                raise ValueError('Colunas remainder inesperadas.')
            continue
        cols = list(cols)
        faixa = ct.output_indices_[nome]
        largura = faixa.stop - faixa.start
        if largura == 0:
            continue
        etapas = [t for _, t in trans.steps] if hasattr(trans, 'steps') else [trans]
        encs = [e for e in etapas if isinstance(e, OneHotEncoder)]
        if encs:
            enc = encs[0]
            if (len(encs) != 1 or enc.drop_idx_ is not None or
                enc.min_frequency is not None or enc.max_categories is not None):
                raise ValueError('Esperado OneHotEncoder completo, sem drop/agregacao.')
            etiquetas = [a for a, cats in zip(cols, enc.categories_) for _ in cats]
        else:
            etiquetas = cols
        if len(etiquetas) != largura or not set(etiquetas).issubset(atributos):
            raise ValueError(f'Saidas inesperadas no bloco {nome}: {etiquetas}')
        grupos[faixa] = etiquetas
    if set(grupos) != set(atributos):
        raise ValueError(f'Algum atributo nao foi mapeado: {set(atributos)-set(grupos)}')
    return grupos


def acrescentar_votos(df, k):
    df = df.copy()
    prob = df['prob_ativo'].to_numpy(dtype=float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError('Probabilidades fora de [0,1].')
    v = np.rint(prob * k).astype(int)
    if not np.allclose(prob, v / k, atol=1e-10, rtol=0):
        raise ValueError(f'Escores nao correspondem a votos uniformes com k={k}.')
    df['votos'] = v
    df['k'] = k
    return df


def avaliar(g, minimo):
    y = g['real'].to_numpy()
    p = g['prob_ativo'].to_numpy()
    pred = (g['votos'].to_numpy() >= minimo).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'acuracia': accuracy_score(y, pred),
        'acuracia_balanceada': balanced_accuracy_score(y, pred),
        'precisao': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'f2': fbeta_score(y, pred, beta=2, zero_division=0),
        'auc': roc_auc_score(y, p),
        'average_precision': average_precision_score(y, p),
        'fpr': fp / (tn + fp),
        'fnr': fn / (tp + fn),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def referencia_k5(esperado):
    """Reutiliza escores OOF, sem refazer o k=5 nem consumir o teste."""
    caminho = BASE / 'regioes_balanceamento' / 'PROBABILIDADES_OOF.csv'
    if not caminho.exists():
        raise FileNotFoundError(
            f'Faltou a referencia k=5: {caminho}\n'
            'Execute o diagnostico de regioes/balanceamento antes deste script.')
    anterior = pd.read_csv(caminho)
    ref = anterior[anterior['configuracao'] == 'regioes'].copy()
    ref = ref.sort_values('posicao_dev').reset_index(drop=True)
    if not np.array_equal(ref[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
        raise ValueError('Referencia antiga nao esta alinhada com os mesmos folds.')
    return acrescentar_votos(
        ref[CHAVES + ['prob_ativo']].assign(configuracao='referencia_k5'), 5)


def tabela_agregada(d):
    """Contagens OOF somadas. As outras metricas sao medias dos 10 folds."""
    idx = ['configuracao', 'k', 'votos_minimos', 'limiar_efetivo', 'regra']
    gb = d.groupby(idx, sort=False)
    t = gb[METRICAS].mean().join(gb[MATRIZ].sum())
    # Frações exatas derivadas das contagens OOF, para aplicar o teto de FPR.
    t['fpr_oof'] = t['fp'] / (t['fp'] + t['tn'])
    t['recall_oof'] = t['tp'] / (t['tp'] + t['fn'])
    t['f1_oof'] = 2 * t['tp'] / (2*t['tp'] + t['fp'] + t['fn'])
    return t.reset_index()


def comparacao_tetos(tabela):
    """Escolhe um voto minimo por k respeitando os tetos EXATOS do k=5.

    Empate: maior recall -> maior F1 -> menor FPR. Essa escolha e
    EXPLORATORIA dentro do desenvolvimento, nao uma avaliacao independente.
    """
    ref = tabela[tabela['configuracao'] == 'referencia_k5']
    tetos = {
        'conservador_k5_padrao': float(ref.loc[ref.votos_minimos == 3, 'fpr_oof'].iloc[0]),
        'moderado_k5_2_votos': float(ref.loc[ref.votos_minimos == 2, 'fpr_oof'].iloc[0]),
    }
    saida = []
    for label, teto in tetos.items():
        for (config, k), grupo in tabela.groupby(['configuracao', 'k'], sort=False):
            elegiveis = grupo[grupo['fpr_oof'] <= teto + 1e-12]
            if elegiveis.empty:
                raise ValueError(f'Sem limiar elegivel: {config} k={k} teto={teto:.4f}')
            melhor = elegiveis.sort_values(
                ['recall_oof', 'f1_oof', 'fpr_oof'],
                ascending=[False, False, True], kind='stable').iloc[0].to_dict()
            melhor['perfil'] = label
            melhor['teto_fpr'] = teto
            saida.append(melhor)
    t = pd.DataFrame(saida)
    refv = t[t.configuracao == 'referencia_k5'].set_index('perfil')
    for metrica in ['recall_oof', 'fpr_oof', 'f1_oof', 'auc', 'average_precision', 'fp', 'fn', 'tp']:
        t[f'delta_{metrica}_vs_k5'] = t.apply(
            lambda r: r[metrica] - refv.loc[r['perfil'], metrica], axis=1)
    t['domina_referencia_em_recall_e_fpr'] = (
        (t['delta_recall_oof_vs_k5'] >= -1e-12) &
        (t['delta_fpr_oof_vs_k5'] <= 1e-12) &
        ((t['delta_recall_oof_vs_k5'] > 1e-12) |
         (t['delta_fpr_oof_vs_k5'] < -1e-12))
    )
    return t


def executar():
    caminho = ROOT / 'data' / 'interim' / 'pns_base_eda.parquet'
    X, y = separar_xy(pd.read_parquet(caminho))
    if X.shape[1] != 10:
        raise ValueError(f'Esperados 10 atributos originais; vieram {X.shape[1]}.')
    X, _, y, _ = train_test_split(X, y, test_size=0.20,
                                  stratify=y, random_state=42)
    R = agrupar(X)
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(R, y))
    esperado = pd.concat([
        pd.DataFrame({'posicao_dev': va, 'fold': f, 'real': y.iloc[va].to_numpy()})
        for f, (_, va) in enumerate(folds, 1)
    ]).sort_values('posicao_dev').reset_index(drop=True)
    oofs = [referencia_k5(esperado)]
    print(f'kNN k={KS} | {len(R)} dev | 10 folds | 60 novos ajustes')
    print('UF -> 5 regioes | RobustScaler | sem PCA | sem oversampling | 10 atributos')
    print('Versoes: regioes simples e pesos F; referencia k=5 recuperada.')
    print('Teste reservado e codigo oficial intactos.\n', flush=True)
    pesos_registros = []
    for fold, (tr, va) in enumerate(folds, 1):
        yt = y.iloc[tr].to_numpy()
        real = y.iloc[va].to_numpy()
        pipe = construir_pipeline(escala='robust')
        if [nome for nome, _ in pipe.steps] != ['preprocessamento', 'selecao', 'pca', 'knn']:
            raise ValueError('A estrutura do pipeline oficial mudou.')
        if pipe.named_steps['selecao'] != 'passthrough' or pipe.named_steps['pca'] != 'passthrough':
            raise ValueError('Este diagnostico nao admite selecao/PCA.')
        pre = pipe[:-1]
        ztr = np.asarray(pre.fit_transform(R.iloc[tr], yt), dtype=float)
        zva = np.asarray(pre.transform(R.iloc[va]), dtype=float)
        if ztr.shape[1] != 31:
            raise ValueError(f'Esperadas 31 dimensoes apos One-Hot, vieram {ztr.shape[1]}.')
        grupos = mapear_colunas(pre.named_steps['preprocessamento'], ztr.shape[1], R.columns)
        if fold == 1:
            print('Dimensoes por atributo:', pd.Series(grupos).value_counts().to_dict(), flush=True)
        base_knn = pipe.named_steps['knn']
        if not (base_knn.n_neighbors == 5 and base_knn.weights == 'uniform' and base_knn.p == 2):
            raise ValueError('Esperado KNeighborsClassifier padrao, distancia euclidiana.')
        for config, pesos in CONFIGS.items():
            fatores = np.sqrt(np.array([pesos.get(a, 1.0) for a in grupos], dtype=float))
            if not np.isfinite(fatores).all() or (fatores <= 0).any():
                raise ValueError('Pesos invalidos.')
            if fold == 1:
                for a in R.columns:
                    pesos_registros.append({
                        'configuracao': config, 'atributo': a,
                        'peso_distancia_quadratica': pesos.get(a, 1.0),
                        'numero_colunas': int((grupos == a).sum())})
            treino = np.ascontiguousarray(ztr * fatores)
            validacao = np.ascontiguousarray(zva * fatores)
            for k in KS:
                model = clone(base_knn).set_params(n_neighbors=k)
                model.fit(treino, yt)
                prob = model.predict_proba(validacao)[:, list(model.classes_).index(1)]
                g = acrescentar_votos(pd.DataFrame({
                    'configuracao': config, 'posicao_dev': va, 'fold': fold,
                    'real': real, 'prob_ativo': prob}), k)
                oofs.append(g)
                r = avaliar(g, (k // 2) + 1)
                print(f'Fold {fold}/10 | {config} | k={k:2} | '
                      f'AUC={r["auc"]:.4f} AP={r["average_precision"]:.4f} '
                      f'F1={r["f1"]:.4f} Recall={r["recall"]:.4f} '
                      f'FPR={r["fpr"]:.4f}', flush=True)

    oof = pd.concat(oofs, ignore_index=True)
    linhas = []
    for (config, k), grupo in oof.groupby(['configuracao', 'k'], sort=False):
        grupo = grupo.sort_values('posicao_dev')
        if not np.array_equal(grupo[CHAVES].to_numpy(), esperado[CHAVES].to_numpy()):
            raise ValueError(f'Predicoes OOF desalinhadas: {config}, k={k}.')
        for votos in range(k, 0, -1):
            for fold, parte in grupo.groupby('fold'):
                linhas.append({
                    'configuracao': config, 'k': k, 'votos_minimos': votos,
                    'limiar_efetivo': votos / k,
                    'regra': 'padrao' if votos == (k//2 + 1) else 'ajustada',
                    'fold': fold, **avaliar(parte, votos),
                })
    d = pd.DataFrame(linhas)
    resumo = tabela_agregada(d)
    escolhidos = comparacao_tetos(resumo)
    pares = []
    ref = d[(d.configuracao=='referencia_k5') & (d.regra=='padrao')].set_index('fold')
    for (config, k), grupo in d[d.regra=='padrao'].groupby(['configuracao','k'], sort=False):
        if config == 'referencia_k5':
            continue
        g = grupo.set_index('fold')
        for metrica in ['auc','average_precision','f1','recall','fpr','acuracia']:
            delta = g[metrica] - ref[metrica]
            favorece = -delta if metrica == 'fpr' else delta
            pares.append({
                'configuracao':config, 'k':k, 'metrica':metrica,
                'delta_medio_vs_k5':delta.mean(), 'dp_delta':delta.std(),
                'melhores':int((favorece>1e-12).sum()),
                'piores':int((favorece< -1e-12).sum()),
                'empates':int((favorece.abs()<=1e-12).sum()),
            })

    SAIDA.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA/'METRICAS_POR_FOLD.csv', index=False)
    oof.sort_values(['configuracao','k','posicao_dev']).to_csv(
        SAIDA/'PROBABILIDADES_OOF.csv', index=False)
    resumo.to_csv(SAIDA/'TODOS_OS_LIMIARES.csv', index=False)
    escolhidos.to_csv(SAIDA/'LIMIARES_COM_TETO_FPR.csv', index=False)
    pd.DataFrame(pares).to_csv(SAIDA/'COMPARACAO_PAREADA_PADRAO.csv', index=False)
    pd.DataFrame(pesos_registros).to_csv(SAIDA/'PESOS_ATRIBUTOS.csv', index=False)

    apresentacao = ['configuracao','k','votos_minimos','acuracia','precisao','recall',
                    'f1','auc','average_precision','fpr','tn','fp','fn','tp']
    print('\n=== VOTACAO PADRAO - k5, k7, k11, k13 ===')
    print(resumo[resumo.regra=='padrao'][apresentacao].round(4).to_string(index=False))
    print('\n=== MELHOR RECALL SEM ULTRAPASSAR O FPR DA REFERENCIA k5 ===')
    destacar = ['perfil','configuracao','k','votos_minimos','teto_fpr',
                'fpr_oof','recall_oof','f1_oof','acuracia','precisao','auc',
                'average_precision','tn','fp','fn','tp',
                'delta_recall_oof_vs_k5','delta_fpr_oof_vs_k5',
                'domina_referencia_em_recall_e_fpr']
    print(escolhidos[destacar].round(4).to_string(index=False))
    print('\n=== TODOS OS CORTES (incluindo matrizes) ===')
    print(resumo[apresentacao].round(4).to_string(index=False))
    print('\n=== PARES POR FOLD - VOTACAO PADRAO ===')
    print(pd.DataFrame(pares).round(4).to_string(index=False))
    print('\nLEITURA: se a AUC subir, o modelo discrimina melhor; um recall maior')
    print('so e vantajoso se o FPR continuar aceitavel. Os tetos comparam com k=5.')
    print('Os limiares elegidos usam os mesmos dados de desenvolvimento: exploratorios.')
    print('k=7/11/13 altera hiperparametro, entao NAO e experimento oficial.')
    print('Resultados completos:', SAIDA)


if __name__ == '__main__':
    executar()
