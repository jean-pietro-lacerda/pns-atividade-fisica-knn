"""PNS 2019 | Comparacao controlada de tres variantes do kNN (diagnostico extra).

1. KNN k=5 uniforme (referencia); 2. kNN k=5 weights='distance';
3. KNN k=5 uniforme apos Tomek Links (remove apenas classe majoritaria).

Mesma divisao 80/20, 10 folds, 10 atributos, UF -> 5 regioes,
RobustScaler + OneHot, SEM balanceamento, SEM PCA, SEM tempo como feature.

Ajustes e Tomek ocorrem apenas no treino do fold. O teste reservado fica
intacto. Comparacoes de FPR iguais sao exploratorias, pois os limiares sao
escolhidos sobre as proprias predicoes OOF (nao sao desempenho independente).

Sem dependencia adicional de imbalanced-learn: Tomek Links implementado com
vizinhos proximos MUTUOS de classes opostas na matriz escalonada de cada fold.
Apenas observacoes da classe majoritaria sao removidas (regra usual 'auto').
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data' / 'interim' / 'pns_base_eda.parquet'
SAIDA = ROOT / 'outputs' / 'diagnostico_pedro' / 'comparacao_distancia_tomek'
CONFIGS = ('uniforme', 'distancia', 'tomek')
CORTES = (0.5, 0.4)
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


def probabilidade_distance(d, labels):
    """Equivalente a sklearn weights='distance', inclusive distancias zero."""
    d = np.asarray(d, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pesos = np.divide(1.0, d, out=np.zeros_like(d), where=(d != 0))
    zeros = (d == 0)
    com_zero = zeros.any(axis=1)
    pesos[com_zero] = zeros[com_zero].astype(float)
    return np.sum(pesos * labels, axis=1) / np.sum(pesos, axis=1)


def aplicar_tomek(Xtreino, ytreino):
    """Detecta pares de vizinhos mutuamente mais proximos e classes opostas.

    Usa kneighbors(X=None) para excluir inequivocamente cada ponto de sua
    propria lista de vizinhos, importante quando ha linhas duplicadas.
    Em empates de distancia, a escolha do vizinho segue a biblioteca.
    Remove apenas a classe majoritaria, nunca os positivos minoritarios.
    """
    y = np.asarray(ytreino, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('As duas classes precisam estar no treino para Tomek.')
    maior = int(np.argmax(np.bincount(y, minlength=2)))
    vizinhos = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    vizinhos.fit(Xtreino)
    viz = vizinhos.kneighbors(X=None, n_neighbors=1, return_distance=False)[:, 0]
    ids = np.arange(len(y))
    mutuos = (viz[viz] == ids) & (y != y[viz])
    remover = mutuos & (y == maior)
    manter = ~remover
    return manter, {
        'treino_antes': int(len(y)), 'classe_maioritaria': maior,
        'n_tomek_removidos': int(remover.sum()),
        'pct_tomek_removidos': float(remover.mean() * 100),
        'treino_depois': int(manter.sum()),
        'positivos_antes': int((y == 1).sum()),
        'positivos_depois': int((y[manter] == 1).sum()),
        'negativos_antes': int((y == 0).sum()),
        'negativos_depois': int((y[manter] == 0).sum()),
    }


def medir(real, prob, limiar):
    real = np.asarray(real, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(real, pred, labels=[0, 1]).ravel()
    return dict(
        acuracia=float(accuracy_score(real, pred)),
        precisao=float(precision_score(real, pred, zero_division=0)),
        recall=float(recall_score(real, pred, zero_division=0)),
        f1=float(f1_score(real, pred, zero_division=0)),
        fpr=float(fp / (fp + tn)) if (fp + tn) else np.nan,
        auc=float(roc_auc_score(real, prob)),
        average_precision=float(average_precision_score(real, prob)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def assinatura(X, y):
    """Previne reuso de checkpoints de outro conjunto/preprocessamento."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X.reset_index(drop=True), index=False)
             .to_numpy(dtype='uint64').tobytes())
    h.update(np.asarray(y, dtype='int8').tobytes())
    h.update(Path(__file__).read_bytes())
    h.update((ROOT / 'src' / 'preprocessamento.py').read_bytes())
    return h.hexdigest()


def ajustar_fold(X, y, tr, va, fold):
    yt = y[tr]
    pipe = construir_pipeline(escala='robust')
    if (list(pipe.named_steps) != ['preprocessamento', 'selecao', 'pca', 'knn']
        or pipe.named_steps['selecao'] != 'passthrough'
        or pipe.named_steps['pca'] != 'passthrough'):
        raise RuntimeError('O pipeline oficial mudou; revise antes de executar.')
    knn = pipe.named_steps['knn']
    if (knn.n_neighbors, knn.weights, knn.p) != (5, 'uniform', 2):
        raise RuntimeError('O kNN de referencia nao esta com parametros padrao.')
    print(f'Fold {fold}/10 | ajustando RobustScaler + OHE no treino...', flush=True)
    pre = pipe[:-1]
    ztr = np.ascontiguousarray(pre.fit_transform(X.iloc[tr], yt), dtype=float)
    zva = np.ascontiguousarray(pre.transform(X.iloc[va]), dtype=float)
    if not np.isfinite(ztr).all() or not np.isfinite(zva).all():
        raise ValueError('Valores nao finitos apos preprocessamento.')
    if fold == 1:
        print(f'  Colunas codificadas: {ztr.shape[1]}. 10 atributos originais.', flush=True)

    # MESMOS cinco vizinhos nas variantes uniforme e distancia: so muda voto.
    knn.fit(ztr, yt)
    dist, ind = knn.kneighbors(zva, n_neighbors=5, return_distance=True)
    labels = yt[ind]
    p_uniforme = np.mean(labels, axis=1)
    p_distancia = probabilidade_distance(dist, labels)

    print(f'Fold {fold}/10 | procurando Tomek Links SOMENTE no treino...', flush=True)
    manter, estat = aplicar_tomek(ztr, yt)
    if len(np.unique(yt[manter])) != 2:
        raise RuntimeError('A limpeza Tomek eliminou uma classe; interrompendo.')
    tomek_clf = KNeighborsClassifier(n_neighbors=5)
    tomek_clf.fit(ztr[manter], yt[manter])
    p_tomek = tomek_clf.predict_proba(zva)[:, list(tomek_clf.classes_).index(1)]
    for p in (p_uniforme, p_distancia, p_tomek):
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise RuntimeError('Escore fora do intervalo [0,1].')
    print(f'Fold {fold}/10 | removidos={estat["n_tomek_removidos"]} '
          f'({estat["pct_tomek_removidos"]:.2f}%) | '
          f'AUC ref={roc_auc_score(y[va],p_uniforme):.4f} '
          f'dist={roc_auc_score(y[va],p_distancia):.4f} '
          f'tomek={roc_auc_score(y[va],p_tomek):.4f}', flush=True)
    return dict(posicao_dev=va, fold=np.full(len(va), fold), real=y[va],
                uniforme=p_uniforme, distancia=p_distancia, tomek=p_tomek), estat


def escolher_limite_fpr(real, prob, teto):
    """Escolha RETROSPECTIVA do limiar nas proprias OOF; EXPLORATORIA!"""
    fprs, recalls, thresholds = roc_curve(real, prob, drop_intermediate=False)
    elegiveis = np.flatnonzero(np.isfinite(thresholds) & (fprs <= teto + 1e-12))
    if not len(elegiveis):
        raise RuntimeError('Nenhum limiar finito satisfaz o teto de FPR.')
    ordem = np.lexsort((fprs[elegiveis], -recalls[elegiveis]))
    melhor = int(elegiveis[ordem[0]])
    limiar = float(thresholds[melhor])
    met = medir(real, prob, limiar)
    if met['fpr'] > teto + 1e-10:
        raise RuntimeError('O limiar retrospectivo excedeu o teto de FPR.')
    return limiar, met


def criar_graficos(oof, destino):
    y = oof['real'].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for config in CONFIGS:
        fp, tp, _ = roc_curve(y, oof[config].to_numpy())
        auc = roc_auc_score(y, oof[config])
        ax.plot(fp, tp, linewidth=2, label=f'{config} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='0.7', linewidth=1)
    ax.set(xlabel='Taxa de falsos positivos (FPR)', ylabel='Recall (TPR)',
           title='PNS 2019 | kNN: uniforme, distância e Tomek Links', xlim=(0,1), ylim=(0,1))
    ax.grid(alpha=.25)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(destino / 'CURVAS_ROC.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    estilos = {'uniforme':'o', 'distancia':'s', 'tomek':'^'}
    for config in CONFIGS:
        for limiar in CORTES:
            met = medir(y, oof[config], limiar)
            ax.scatter(met['fpr']*100, met['recall']*100, s=90,
                       marker=estilos[config],
                       label=f'{config} / limiar {limiar:.1f}')
            ax.annotate(f'{met["f1"]:.3f}', (met['fpr']*100,met['recall']*100),
                        xytext=(6, 5), textcoords='offset points', fontsize=9)
    ax.set(xlabel='FPR (%) — menor é melhor', ylabel='Recall (%) — maior é melhor',
           title='Trade-off: recall × falsos positivos (rótulo = F1)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destino / 'TRADEOFF_RECALL_FPR.png', dpi=180)
    plt.close(fig)


def consolidar(oof, estat_tomek, destino):
    y = oof['real'].to_numpy(dtype=int)
    if not oof['posicao_dev'].is_unique:
        raise RuntimeError('Indices de validacao duplicados.')
    linhas = []
    folds = []
    for config in CONFIGS:
        p = oof[config].to_numpy(dtype=float)
        for limiar in CORTES:
            met = medir(y, p, limiar)
            linhas.append(dict(configuracao=config, limiar=limiar, n=len(y), **met))
            for fold, parte in oof.groupby('fold'):
                folds.append(dict(configuracao=config, limiar=limiar, fold=int(fold),
                                  n=len(parte), **medir(parte['real'], parte[config], limiar)))
    resumo = pd.DataFrame(linhas)
    por_fold = pd.DataFrame(folds)
    prim = resumo[resumo.configuracao == 'uniforme'].set_index('limiar')
    for m in ('acuracia','precisao','recall','f1','fpr','auc','average_precision'):
        resumo['delta_'+m+'_vs_uniforme_mesmo_limiar'] = [
            float(r[m]-prim.loc[r['limiar'],m]) for _,r in resumo.iterrows()]
    resumo['melhora_recall_sem_aumentar_fpr'] = [
        bool((r['recall']>=prim.loc[r['limiar'],'recall']-1e-12)
             and (r['fpr']<=prim.loc[r['limiar'],'fpr']+1e-12)
             and (r['recall']>prim.loc[r['limiar'],'recall']+1e-12
                  or r['fpr']<prim.loc[r['limiar'],'fpr']-1e-12))
        for _,r in resumo.iterrows()]

    # Retrocalibracao EXPLORATORIA para um FPR comparavel (nao desempenho
    # validado independentemente); a maior vantagem aqui e comparar curvas.
    comparacoes = []
    for limiar_ref in CORTES:
        mref = prim.loc[limiar_ref]
        teto = float(mref['fpr'])
        for config in CONFIGS:
            corte_escolhido, metricas = escolher_limite_fpr(y, oof[config], teto)
            comparacoes.append(dict(teto_nome=f'FPR_da_referencia_{limiar_ref:.1f}',
                                     teto_fpr=teto, configuracao=config,
                                     limiar_oof_retroescolhido=corte_escolhido,
                                     **metricas,
                                     delta_recall_vs_referencia=metricas['recall']-mref['recall'],
                                     delta_fpr_vs_referencia=metricas['fpr']-mref['fpr']))
    igual_fpr = pd.DataFrame(comparacoes)

    destino.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(destino/'RESUMO_TRES_MODELOS.csv', index=False, float_format='%.8f')
    por_fold.to_csv(destino/'METRICAS_POR_FOLD.csv', index=False, float_format='%.8f')
    igual_fpr.to_csv(destino/'FPR_IGUAL_EXPLORATORIO_NAO_VALIDADO.csv', index=False,
                     float_format='%.8f')
    pd.DataFrame(estat_tomek).to_csv(destino/'TOMEK_REMOVIDOS_POR_FOLD.csv', index=False)
    oof.sort_values('posicao_dev').to_csv(destino/'PREDICOES_OOF.csv',
                                             index=False, float_format='%.12f')
    criar_graficos(oof, destino)
    texto = '''COMPARACAO DIAGNOSTICA PNS 2019 — 3 configuracoes:
- uniforme: kNN n_neighbors=5, weights='uniform', padrao;
- distancia: MESMOS cinco vizinhos, weights='distance' (voto inverso a distancia);
- tomek: kNN uniforme apos remover apenas exemplos MAJORITARIOS
  de pares Tomek de treino, detectados apos RobustScaler e OneHot.

Fixo: 10 atributos, UF->5 regioes, RobustScaler, OHE, sem PCA,
sem oversampling; 10 folds estratificados seed 42 sobre desenvolvimento.
O teste reservado de 17.692 pessoas nao foi usado ou modificado.
Metodologia OOF: cada pessoa e avaliada no fold do qual ficou de fora;
metricas no RESUMO calculadas das previsoes OOF agregadas (nao media dos folds).

IMPORTANTE: cortes 0.5 e 0.4 sao fixos e comparaveis entre modelos,
mas pesos por distancia geram pontuacoes continuas; a mesma taxa de FPR
nao e garantida. FPR_IGUAL_EXPLORATORIO_NAO_VALIDADO escolhe o limiar
com a propria validacao OOF (otimismo de selecao). Somente exploracao:
nao apresentar como desempenho independente nem como ganho definitivo.

A sobreposicao real dos perfis pode ser inseparavel com as features atuais.
Tomek pode remover registros legitimos; remocao nao prova que sao ruido.
Com empates de distancia (perfis identicos), a escolha do vizinho e
sensivel a ordem; relatorio registra quantos foram removidos.
Nao ha inferencia causal nem teste de significancia nesta comparacao.
Nenhum script oficial, base ou codigo dos colegas foi alterado.
'''
    (destino/'LEIA_RESULTADOS.txt').write_text(texto, encoding='utf-8')
    print('\n=== RESULTADOS GLOBAIS OOF (70.764 no projeto real) ===', flush=True)
    cols = ['configuracao','limiar','acuracia','precisao','recall','f1','fpr','auc','average_precision','tn','fp','fn','tp']
    print(resumo[cols].round(4).to_string(index=False))
    print('\n=== MESMO TETO DE FPR — APENAS EXPLORATORIO / LIMIARES ESCOLHIDOS NA PROPRIA OOF ===')
    cols2 = ['teto_nome','configuracao','limiar_oof_retroescolhido',
             'recall','fpr','f1','delta_recall_vs_referencia','delta_fpr_vs_referencia']
    print(igual_fpr[cols2].round(4).to_string(index=False))
    print('\n=== TOTAL REMOVIDO POR TOMEK (por fold) ===')
    print(pd.DataFrame(estat_tomek)[['fold','n_tomek_removidos','pct_tomek_removidos']].to_string(index=False))
    print('\nArquivos e 2 graficos em:', destino)
    print('Teste reservado intacto. Esta e uma analise complementar, nao oficial.')


def executar(X, y, saida, n_folds=10, checkpoints=True):
    if X.shape[1] != 10:
        raise ValueError(f'Esperados 10 atributos de entrada; recebidos {X.shape[1]}')
    X = agrupar_uf(X)
    y = np.asarray(y, dtype=int)
    splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=42).split(X, y))
    print(f'COMPARACAO KNN | dev={len(y)} | folds={n_folds} | k=5')
    print('Modelos: UNIFORME | DISTANCE | TOMEK (somente treino; maioria)')
    print('Regioes | RobustScaler | OneHot | sem PCA/oversampling')
    print('Minutos semanais NAO sao features. Teste reservado NAO usado.\n', flush=True)

    cache = saida / 'checkpoints'
    chave = assinatura(X, y) if checkpoints else 'autoteste'
    if checkpoints:
        cache.mkdir(parents=True, exist_ok=True)
        manifesto = cache / 'identificador.json'
        if manifesto.exists():
            antigo = json.loads(manifesto.read_text(encoding='utf-8'))
            if antigo != {'assinatura': chave, 'folds': n_folds}:
                raise RuntimeError('Os checkpoints sao de uma versao diferente. '
                    'Renomeie a pasta outputs/.../checkpoints antes de repetir.')
        else:
            if list(cache.glob('fold_*.npz')):
                raise RuntimeError('Checkpoint sem identificador: verifique a pasta.')
            manifesto.write_text(json.dumps({'assinatura': chave, 'folds': n_folds}),
                                 encoding='utf-8')

    oof_registros = []
    todas_estat = []
    for fold, (tr, va) in enumerate(splits, 1):
        arq = cache / f'fold_{fold:02d}.npz'
        if checkpoints and arq.exists():
            with np.load(arq, allow_pickle=False) as z:
                if not np.array_equal(z['posicao_dev'], va) or not np.array_equal(z['real'],y[va]):
                    raise RuntimeError(f'Checkpoint desalinhado no fold {fold}.')
                reg = {c: z[c] for c in ('posicao_dev','fold','real') + CONFIGS}
                est = json.loads(str(z['estat_json'].item()))
            print(f'Fold {fold}/{n_folds} recuperado do checkpoint.', flush=True)
        else:
            reg, est = ajustar_fold(X, y, tr, va, fold)
            if checkpoints:
                np.savez_compressed(arq, **reg, estat_json=json.dumps(est))
        oof_registros.append(pd.DataFrame(reg))
        todas_estat.append(dict(fold=fold, **est))

    oof = pd.concat(oof_registros, ignore_index=True).sort_values('posicao_dev')
    if len(oof) != len(y) or not np.array_equal(oof['posicao_dev'].to_numpy(),np.arange(len(y))):
        raise RuntimeError('As predicoes OOF nao cobrem exatamente o desenvolvimento.')
    if not np.array_equal(oof['real'].to_numpy(), y):
        raise RuntimeError('As predicoes OOF nao correspondem ao Y original.')
    consolidar(oof, todas_estat, saida)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--autoteste', action='store_true', help='Testa com dados sinteticos, sem PNS')
    args = parser.parse_args()
    if args.autoteste:
        rng = np.random.default_rng(42)
        n = 360
        dados = pd.DataFrame({
            'idade': rng.integers(18, 85, n),
            'renda_per_capita': rng.integers(250, 5000, n),
            'numero_moradores': rng.integers(1, 7, n),
            'sexo': rng.choice(['Homem','Mulher'], n),
            'escolaridade': rng.choice(['Medio','Superior','Fundamental'], n),
            'uf': rng.choice(['26','35','41','11','50'], n),
            'situacao_trabalho': rng.choice(['Ocupado','Desocupado'], n),
            'area_residencia': rng.choice(['Urbana','Rural'], n),
            'raca_cor': rng.choice(['Parda','Branca','Preta'], n),
            'estado_civil': rng.choice(['Casado','Solteiro'], n),
        })
        yy = (rng.random(n) < (0.22 + .003*(50-dados['idade'].to_numpy()))).astype(int)
        d = np.array([[0.0],[0.10],[0.18],[2.0],[2.1],[3.0]])
        labs = np.array([0,1,1,0,0,0])
        manter, estat = aplicar_tomek(d, labs)
        assert estat['positivos_antes'] == estat['positivos_depois']
        assert len(manter) == len(labs)
        # Verificar calculo dos pesos inclusive zero, contra sklearn.
        zz = np.array([[0.],[0.],[.4],[1.],[2.],[3.]],dtype=float)
        ll = np.array([0,1,1,0,0,1])
        k = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(zz,ll)
        consultas = np.array([[0.],[.3],[1.1]])
        ds, ids = k.kneighbors(consultas)
        assert np.allclose(probabilidade_distance(ds,ll[ids]),k.predict_proba(consultas)[:,1])
        with tempfile.TemporaryDirectory(prefix='autoteste_tomek_') as temp:
            executar(dados, yy, Path(temp)/'saida', n_folds=3, checkpoints=False)
        print('\nAUTOTESTE CONCLUIDO. Nao foram usados dados reais da PNS.')
        return

    if not BASE.exists():
        raise FileNotFoundError(f'Base tratada nao encontrada: {BASE}')
    X, y = separar_xy(pd.read_parquet(BASE))
    X, _, y, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    if len(X) != 70764:
        raise RuntimeError(f'Esperados 70.764 no desenvolvimento; vieram {len(X)}.')
    executar(X.reset_index(drop=True), y.to_numpy(), SAIDA)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nERRO: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
