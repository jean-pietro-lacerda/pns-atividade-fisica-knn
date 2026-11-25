"""Diagnostico local: 5 x 10 folds, tres configuracoes e retomada por fold."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from src import preprocessamento

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / 'data/interim/pns_base_eda.parquet'
SAIDA = RAIZ / 'outputs/diagnostico_pedro/validacao_repetida'
N_FOLDS = 10
N_REPETICOES = 5
SEMENTE = 42
CONFIGS = {
    'nenhum': (),
    'sem_moradores_e_uf': ('numero_moradores', 'uf'),
    'sem_moradores_e_raca_cor': ('numero_moradores', 'raca_cor'),
}
METRICAS = ['acuracia', 'acuracia_balanceada', 'f1', 'auc', 'recall_ativo',
            'recall_inativo', 'precisao_ativo', 'fpr']


def criar_modelo(retirados):
    modelo = preprocessamento.construir_pipeline(escala='robust')
    if not retirados:
        return modelo
    original = modelo.named_steps['preprocessamento']
    transformadores = [(nome, etapa, [c for c in colunas if c not in retirados])
                       for nome, etapa, colunas in original.transformers]
    modelo.set_params(preprocessamento=ColumnTransformer(
        transformadores, remainder='drop', sparse_threshold=original.sparse_threshold))
    return modelo


def sha256(arquivo):
    h = hashlib.sha256()
    with open(arquivo, 'rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def gravar_json(arquivo, dados):
    temporario = arquivo.with_suffix('.tmp')
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf-8')
    temporario.replace(arquivo)


def salvar_csv(df, nome):
    df.to_csv(SAIDA / nome, index=False, sep=';', decimal=',', encoding='utf-8-sig')


def avaliar(modelo, X_t, y_t, X_v, y_v):
    inicio = perf_counter()
    modelo.fit(X_t, y_t)
    ajuste_s = perf_counter() - inicio
    inicio = perf_counter()
    # Mesma regra de previsao e calculo de AUC do teste de combinacoes.
    previsto = modelo.predict(X_v)
    prob = modelo.predict_proba(X_v)[:, list(modelo.classes_).index(1)]
    previsao_s = perf_counter() - inicio
    vn, fp, fn, vp = confusion_matrix(y_v, previsto, labels=[0, 1]).ravel()
    return dict(acuracia=float(accuracy_score(y_v, previsto)),
        acuracia_balanceada=float(balanced_accuracy_score(y_v, previsto)),
        f1=float(f1_score(y_v, previsto, zero_division=0)),
        auc=float(roc_auc_score(y_v, prob)),
        recall_ativo=float(recall_score(y_v, previsto, pos_label=1, zero_division=0)),
        recall_inativo=float(recall_score(y_v, previsto, pos_label=0, zero_division=0)),
        precisao_ativo=float(precision_score(y_v, previsto, zero_division=0)),
        fpr=float(fp / (fp + vn)), vn=int(vn), fp=int(fp), fn=int(fn), vp=int(vp),
        ajuste_s=ajuste_s, previsao_e_proba_s=previsao_s)


def resumir(linhas):
    df = pd.DataFrame(linhas).sort_values(['repeticao', 'fold', 'configuracao'])
    salvar_csv(df, 'METRICAS_POR_FOLD.csv')
    medias = df.groupby('configuracao', sort=False)[METRICAS].mean().reindex(CONFIGS)
    desvios = df.groupby('configuracao', sort=False)[METRICAS].std().reindex(CONFIGS)
    resumo = medias.join(desvios.add_prefix('dp_')).reset_index()
    ref = medias.loc['nenhum']
    for m in METRICAS:
        resumo['delta_' + m] = resumo[m] - ref[m]
    salvar_csv(resumo, 'COMPARACAO_FINAL.csv')
    reps = df.groupby(['configuracao', 'repeticao'], as_index=False)[METRICAS].mean()
    salvar_csv(reps, 'MEDIAS_POR_REPETICAO.csv')
    referencia = df[df.configuracao == 'nenhum'].set_index(['repeticao', 'fold'])
    deltas = []
    estabilidade = []
    for nome in CONFIGS:
        if nome == 'nenhum':
            continue
        alternativa = df[df.configuracao == nome].set_index(['repeticao', 'fold'])
        assert alternativa.index.equals(referencia.index)
        diferencas = alternativa[METRICAS] - referencia[METRICAS]
        deltas.append(diferencas.add_prefix('delta_').assign(configuracao=nome).reset_index())
        for m in ['f1', 'auc', 'recall_ativo', 'acuracia_balanceada', 'fpr']:
            d = diferencas[m]
            dr = d.groupby(level='repeticao').mean()
            # Para FPR, valores menores sao melhores. Empate com tolerancia numerica.
            direcao = -1 if m == 'fpr' else 1
            estabilidade.append(dict(configuracao=nome, metrica=m,
                delta_medio=float(d.mean()), dp_deltas=float(d.std(ddof=1)),
                minimo=float(d.min()), maximo=float(d.max()),
                folds_melhores=int((direcao*d > 1e-12).sum()),
                folds_piores=int((direcao*d < -1e-12).sum()),
                folds_empate=int((d.abs() <= 1e-12).sum()),
                repeticoes_melhores=int((direcao*dr > 1e-12).sum()),
                repeticoes_piores=int((direcao*dr < -1e-12).sum()),
                repeticoes_empate=int((dr.abs() <= 1e-12).sum())))
    salvar_csv(pd.concat(deltas, ignore_index=True), 'DELTAS_PAREADOS.csv')
    est = pd.DataFrame(estabilidade)
    salvar_csv(est, 'ESTABILIDADE.csv')
    print('\n=== COMPARACAO FINAL (medias dos 50 folds) ===')
    print(resumo[['configuracao','f1','auc','recall_ativo','acuracia_balanceada','fpr',
                  'delta_f1','delta_auc','delta_recall_ativo']].round(4).to_string(index=False))
    print('\n=== ESTABILIDADE (diferencas pareadas contra referencia) ===')
    print(est.round(4).to_string(index=False))
    print('\n=== MEDIAS POR REPETICAO ===')
    print(reps[['configuracao','repeticao','f1','auc','recall_ativo']].round(4).to_string(index=False))
    print('\nResultados em:', SAIDA)
    print('Analise exploratoria: folds dependentes, nao e teste de significancia.')
    print('Teste reservado NAO utilizado. Codigo oficial NAO alterado.')


def executar():
    if not BASE.is_file():
        raise FileNotFoundError(f'Base nao encontrada: {BASE}. Coloque o script na raiz.')
    SAIDA.mkdir(parents=True, exist_ok=True)
    checkpoint = SAIDA / 'checkpoint.json'
    # Impede misturar resultados de bases, implementacoes ou ambientes diferentes.
    meta = dict(base_sha256=sha256(BASE), script_sha256=sha256(Path(__file__)),
                preprocessamento_sha256=sha256(Path(preprocessamento.__file__)),
                python=sys.version, sklearn=sklearn.__version__, numpy=np.__version__,
                pandas=pd.__version__, n_folds=N_FOLDS, n_repeticoes=N_REPETICOES,
                semente=SEMENTE, configs={k:list(v) for k,v in CONFIGS.items()})
    if checkpoint.exists():
        estado = json.loads(checkpoint.read_text(encoding='utf-8'))
        if estado['meta'] != meta:
            raise ValueError('Checkpoint incompativel. Renomeie a pasta validacao_repetida para preservar os resultados antigos e execute novamente.')
    else:
        estado = {'meta': meta, 'linhas': []}
    X, y = preprocessamento.separar_xy(pd.read_parquet(BASE))
    X_dev, _, y_dev, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES, random_state=SEMENTE)
    folds = list(cv.split(X_dev, y_dev))
    linhas = estado['linhas']
    chaves = {(r['configuracao'], r['repeticao'], r['fold']) for r in linhas}
    total = len(folds) * len(CONFIGS)
    esperadas = {(nome, rep, fold) for nome in CONFIGS for rep in range(1,N_REPETICOES+1)
                 for fold in range(1,N_FOLDS+1)}
    if len(chaves) != len(linhas) or not chaves.issubset(esperadas):
        raise ValueError('Checkpoint possui chaves invalidas ou duplicadas.')
    print(f'VALIDACAO REPETIDA: {N_REPETICOES} x {N_FOLDS} folds; {total} ajustes', flush=True)
    print(f'Desenvolvimento: {len(X_dev)} registros. Teste reservado: NAO utilizado.', flush=True)
    print(f'Ajustes ja salvos: {len(chaves)}/{total}. Ctrl+C para interromper.', flush=True)
    try:
        for i, (t, v) in enumerate(folds):
            rep, fold = i//N_FOLDS+1, i%N_FOLDS+1
            for nome, retirados in CONFIGS.items():
                chave = (nome, rep, fold)
                if chave in chaves:
                    continue
                print(f'[{len(chaves)+1}/{total}] Repeticao {rep}/{N_REPETICOES} | Fold {fold}/{N_FOLDS} | {nome}', flush=True)
                modelo = criar_modelo(retirados)
                res = avaliar(modelo, X_dev.iloc[t], y_dev.iloc[t], X_dev.iloc[v], y_dev.iloc[v])
                linhas.append(dict(configuracao=nome, repeticao=rep, fold=fold, **res))
                gravar_json(checkpoint, estado)
                chaves.add(chave)
                print(f"  Salvo. F1={res['f1']:.4f} AUC={res['auc']:.4f} Recall={res['recall_ativo']:.4f}", flush=True)
    except KeyboardInterrupt:
        print('\nInterrompido. Execute o mesmo comando para retomar os folds ja salvos.', flush=True)
        return
    if chaves != esperadas:
        raise RuntimeError('Experimento incompleto.')
    resumir(linhas)


if __name__ == '__main__':
    executar()
