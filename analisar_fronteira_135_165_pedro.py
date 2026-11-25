"""Diagnostico LOCAL PNS 2019: faixa 135-165 min e alvo alternativo.

Nao altera a base oficial, repositorio, scripts de equipe ou teste reservado.
Requer data/interim/pns_base_eda_com_tempo.parquet, exportado do notebook
01_extracao apos criar df_final (ver instrucoes no terminal).

Hipotese alternativa: atividades MODERADAS passam a exigir 135 em vez
150 minutos/semana. Atividades vigorosas continuam com 75 minutos.
Minutos, modalidade e limite sao usados SOMENTE para definir/auditar o Y,
NUNCA como preditores. kNN k=5 default, 10 atributos, RobustScaler,
OHE, UF->5 regioes, sem PCA ou oversampling.
"""
from pathlib import Path
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from src.preprocessamento import separar_xy, construir_pipeline

ROOT = Path(__file__).resolve().parent
INTERIM = ROOT / 'data' / 'interim'
SAIDA = ROOT / 'outputs' / 'diagnostico_pedro' / 'fronteira_135_165'
COLS = ['idade', 'renda_per_capita', 'numero_moradores', 'sexo', 'escolaridade',
        'uf', 'situacao_trabalho', 'area_residencia', 'raca_cor', 'estado_civil']
ADICIONAIS = ['ativo_lazer', 'minutos_semanais', 'limite_minutos', 'P034', 'P035']
KS = 5
LIMITES_DECISAO = (0.5, 0.4)

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


MAPA = {normalizar(v): r for r, states in REGIOES.items()
        for cod, abbr, name in states for v in (cod, float(cod), abbr, name)}


def regioes(X):
    X = X.copy()
    traduz = X['uf'].map(lambda v: MAPA.get(normalizar(v)) if pd.notna(v) else np.nan)
    if traduz.isna().any():
        raise ValueError('UF nao reconhecida. Nao e seguro alterar o pre-processamento.')
    X['uf'] = traduz
    return X


def ler_bases():
    p = INTERIM / 'pns_base_eda.parquet'
    q = INTERIM / 'pns_base_eda_com_tempo.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Base oficial local nao encontrada: {p}')
    if not q.exists():
        raise FileNotFoundError(
            f'Falta a exportacao complementar: {q}\n\n'
            'No notebook notebooks/01_extracao.ipynb, DEPOIS da celula que cria '
            'df_final e ativo_lazer e DEPOIS da filtragem de adultos, execute:\n\n'
            "    from pathlib import Path\n"
            "    colunas = " + repr(COLS + ADICIONAIS) + "\n"
            "    pasta = Path(INTERIM)\n"
            "    pasta.mkdir(parents=True, exist_ok=True)\n"
            "    df_final[colunas].to_parquet(\n"
            "        pasta / 'pns_base_eda_com_tempo.parquet', index=False)\n\n"
            'Depois execute este script novamente. NAO altere pns_base_eda.parquet.')
    base, ext = pd.read_parquet(p).reset_index(drop=True), pd.read_parquet(q).reset_index(drop=True)
    faltam = set(COLS+ADICIONAIS) - set(ext.columns)
    if faltam:
        raise ValueError(f'Exportacao incompleta; colunas ausentes: {sorted(faltam)}')
    if len(base) != len(ext):
        raise ValueError(f'Alinhamento invalido: base oficial {len(base)} linhas; complementar {len(ext)} linhas.')
    if list(base.columns) != COLS + ['ativo_lazer'] and set(base.columns) != set(COLS + ['ativo_lazer']):
        raise ValueError('Base oficial possui colunas inesperadas. Interrompendo para evitar comparacao incorreta.')
    for col in COLS + ['ativo_lazer']:
        if col in ('idade', 'renda_per_capita', 'numero_moradores', 'ativo_lazer'):
            a, b = pd.to_numeric(base[col], errors='coerce'), pd.to_numeric(ext[col], errors='coerce')
            iguais = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                                equal_nan=True, atol=1e-8, rtol=1e-10)
        else:
            iguais = (base[col].astype('string').fillna('[NA]').str.strip().to_numpy() ==
                      ext[col].astype('string').fillna('[NA]').str.strip().to_numpy())
        if not bool(np.all(iguais)):
            pos = np.flatnonzero(~iguais)[:5].tolist()
            raise ValueError(f'ALINHAMENTO FALHOU em {col}, posicoes {pos}. '
                             'NAO vamos associar tempos a pessoas erradas. '
                             'Reexecute a exportacao apos a etapa final de limpeza.')
    return base, ext


def criar_alvo_alternativo(base, ext):
    y = pd.to_numeric(base['ativo_lazer'], errors='raise').to_numpy(dtype=int)
    tempo = pd.to_numeric(ext['minutos_semanais'], errors='coerce')
    limite = pd.to_numeric(ext['limite_minutos'], errors='coerce')
    pratica = ext['P034'].astype('string').str.replace(r'\.0$', '', regex=True).str.strip()
    freq = pd.to_numeric(ext['P035'], errors='coerce')
    regular = pratica.eq('1').fillna(False) & freq.between(1, 7).fillna(False)
    validos = regular & tempo.notna() & limite.isin([75, 150])
    calc = (tempo[validos] >= limite[validos]).astype(int).to_numpy()
    if not np.array_equal(y[np.flatnonzero(validos.to_numpy())], calc):
        raise ValueError('Rotulos originais nao batem com as regras de 150/75. '
                         'Confira se esta e a MESMA versao de 01_extracao e pns_base_eda.')
    if tempo[regular & ~validos].notna().any():
        raise ValueError('Existem praticantes regulares sem modalidade/limite valido: revise o notebook.')
    mudar = validos & limite.eq(150) & tempo.ge(135) & tempo.lt(150)
    if np.any(y[mudar.to_numpy()] != 0):
        raise ValueError('Esperava rotulo original 0 para moderados de 135 a 149 min.')
    alternativo = y.copy()
    alternativo[mudar.to_numpy()] = 1
    return y, alternativo, mudar.to_numpy(), tempo, limite


def predizer_oof(X, y_original, y_alt):
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X, y_original))
    saida = np.empty((len(X), 2), dtype=float)
    no_fold = np.zeros(len(X), dtype=int)
    for i, (tr, va) in enumerate(folds, 1):
        for j, (nome, y) in enumerate((('original', y_original), ('alternativo', y_alt))):
            clf = construir_pipeline(escala='robust')
            assert clf.named_steps['knn'].n_neighbors == 5
            assert clf.named_steps['knn'].weights == 'uniform'
            clf.fit(X.iloc[tr], y[tr])
            classes = list(clf.named_steps['knn'].classes_)
            if 1 not in classes: raise ValueError('Classe 1 ausente no fold.')
            saida[va, j] = clf.predict_proba(X.iloc[va])[:, classes.index(1)]
        no_fold[va] = i
        print(f'Fold {i}/10 | ajuste original + alternativo | n_validacao={len(va)}', flush=True)
    if not np.all(no_fold): raise RuntimeError('Nem todos os registros receberam predicao OOF.')
    return saida[:, 0], saida[:, 1], no_fold


def medidas(y, prob, corte):
    pred = (prob >= corte - 1e-12).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(limiar=corte, n=len(y), acuracia=accuracy_score(y, pred),
                precisao=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, prob),
                average_precision=average_precision_score(y, prob),
                fpr=fp/(tn+fp) if tn+fp else np.nan,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def salvar(fig, nome):
    fig.savefig(SAIDA / nome, dpi=180, bbox_inches='tight')
    plt.close(fig)


def graficos_faixa(dados):
    faixa = dados[dados['minutos_semanais'].between(135, 165, inclusive='both')].copy()
    faixa['intervalo'] = pd.cut(faixa['minutos_semanais'],
                               bins=[135, 140, 145, 150, 155, 160, 165.000001],
                               labels=['135–139', '140–144', '145–149', '150–154', '155–159', '160–165'],
                               right=False, include_lowest=True)
    if faixa.empty:
        print('Nenhuma pessoa na faixa 135–165 na dev; graficos desta faixa omitidos.')
        return faixa
    tab = pd.crosstab(faixa['intervalo'], faixa['real_original']).reindex(
        index=faixa['intervalo'].cat.categories, columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.bar(tab.index.astype(str), tab[0], label='Insuficiente (rótulo original)', color='#b9c6d6')
    ax.bar(tab.index.astype(str), tab[1], bottom=tab[0], label='Suficiente (rótulo original)', color='#158275')
    ax.axvline(2.5, color='#d18832', ls='--', lw=2, label='Fronteira moderada: 150 min')
    ax.set(title='PNS 2019: classes reais perto de 150 min/semana',
           ylabel='Registros na base de desenvolvimento', xlabel='Minutos de atividade (intervalos)')
    ax.legend(fontsize=9); fig.tight_layout(); salvar(fig, '01_classe_real_por_minutos.png')

    for corte in LIMITES_DECISAO:
        pred = (faixa['prob_original'] >= corte - 1e-12).astype(int)
        real = faixa['real_original'].to_numpy(dtype=int)
        status = np.select([ (real==0)&(pred==0), (real==0)&(pred==1),
                             (real==1)&(pred==0), (real==1)&(pred==1)],
                           ['VN','FP','FN','VP'], default='?')
        tab2 = pd.crosstab(faixa['intervalo'], status).reindex(
            index=faixa['intervalo'].cat.categories,
            columns=['VN', 'FP', 'FN', 'VP'], fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5.4))
        cores = {'VN':'#6587ad','FP':'#cc6b60','FN':'#e4a649','VP':'#208679'}
        bottom = np.zeros(len(tab2))
        for label in ['VN', 'FP', 'FN', 'VP']:
            ax.bar(tab2.index.astype(str), tab2[label], bottom=bottom,
                   color=cores[label], label=label)
            bottom += tab2[label].to_numpy()
        ax.axvline(2.5, color='#333333', ls='--', alpha=.7)
        ax.set(title=f'Classificação OOF original na faixa | limiar {corte:.1f}',
               ylabel='Registros', xlabel='Minutos semanais')
        ax.legend(ncol=4); fig.tight_layout()
        salvar(fig, f'02_confusao_faixa_limiar_{str(corte).replace(".", "_")}.png')

    tab3 = pd.crosstab(faixa['real_original'], faixa['real_alt']).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs=np.arange(2)
    ax.bar(xs-.18, [(faixa.real_original==0).sum(),(faixa.real_original==1).sum()],
           width=.36,label='Original',color='#91a7b8')
    ax.bar(xs+.18,[(faixa.real_alt==0).sum(),(faixa.real_alt==1).sum()],
           width=.36,label='Alternativo',color='#168276')
    ax.set_xticks(xs,['Insuficientes','Suficientes'])
    ax.set(ylabel='Registros',title='Rótulos na faixa de 135–165 min')
    ax.legend();fig.tight_layout();salvar(fig,'03_rotulos_antes_depois.png')
    return faixa


def graficos_metricas(medias):
    metricas = ['acuracia','precisao','recall','f1']
    x=np.arange(len(metricas));w=.20
    fig, ax=plt.subplots(figsize=(10,5.2))
    for i, (modelo,corte,cor) in enumerate([
            ('original',0.5,'#7185a1'),('alternativo',0.5,'#198679'),
            ('original',0.4,'#c58b63'),('alternativo',0.4,'#6baea1')]):
        r=medias[(medias['modelo']==modelo)&(np.isclose(medias['limiar'],corte))].iloc[0]
        ax.bar(x+(i-1.5)*w,[r[m] for m in metricas],width=w,
               label=f'{modelo.capitalize()} | corte {corte}',color=cor)
    ax.set_xticks(x,['Acurácia','Precisão','Recall','F1'])
    ax.set_ylim(0,1);ax.set_ylabel('Métrica calculada nas previsões OOF')
    ax.set_title('Mudança de alvo e de fronteira de decisão: hipóteses distintas')
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout();salvar(fig,'04_metricas_antes_depois.png')


def perfis_semelhantes(X_dev, dados, tempo, limite):
    # Descritivo: utiliza SOMENTE registros da dev e os 10 preditores.
    A = (limite.to_numpy() == 150) & (tempo.to_numpy() >= 135) & (tempo.to_numpy() < 150)
    B = (limite.to_numpy() == 150) & (tempo.to_numpy() >= 150) & (tempo.to_numpy() <= 165)
    a, b=np.flatnonzero(A), np.flatnonzero(B)
    if len(a)<5 or len(b)<5:
        return {'nota':f'Nao ha grupos moderados suficientes: abaixo={len(a)}, acima={len(b)}'}
    # Um pre-processador sem Y: usado somente para descrever a geometria dos perfis.
    prep=construir_pipeline(escala='robust')[:-1]
    z=np.asarray(prep.fit_transform(X_dev),dtype=float)
    viz=NearestNeighbors(n_neighbors=1).fit(z[b])
    dist,ix=viz.kneighbors(z[a]);dist=dist[:,0]; pares=b[ix[:,0]]
    rng=np.random.default_rng(42)
    dist_aleatoria=np.linalg.norm(z[a]-z[rng.choice(b,size=len(a),replace=True)],axis=1)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.hist(dist, bins=25, alpha=.7, label='Par mais próximo, outro lado de 150 min',color='#168276')
    ax.hist(dist_aleatoria,bins=25,alpha=.5,label='Par aleatório, outro lado de 150 min',color='#8495af')
    ax.set(title='Proximidade de perfis sociodemográficos (somente moderados)',
           xlabel='Distância no espaço RobustScaler + One-Hot',ylabel='Quantidade')
    ax.legend();fig.tight_layout();salvar(fig,'05_distancias_perfis_135_165.png')

    dados_a=X_dev.iloc[a].reset_index(drop=True)
    dados_b=X_dev.iloc[pares].reset_index(drop=True)
    atributos_cat=['sexo','escolaridade','uf','situacao_trabalho','area_residencia','raca_cor','estado_civil']
    resumo={'n_moderados_135_149':int(len(a)), 'n_moderados_150_165':int(len(b)),
            'distancia_mediana_mais_proximo':float(np.median(dist)),
            'distancia_mediana_aleatorio':float(np.median(dist_aleatoria)),
            'diferenca_mediana_idade':float(np.median(abs(dados_a['idade'].astype(float)-dados_b['idade'].astype(float)))),
            'diferenca_mediana_renda':float(np.median(abs(dados_a['renda_per_capita'].astype(float)-dados_b['renda_per_capita'].astype(float))))}
    for attr in atributos_cat:
        resumo[f'coincidencia_{attr}'] = float((dados_a[attr].to_numpy()==dados_b[attr].to_numpy()).mean())
    pd.Series(resumo).to_csv(SAIDA/'RESUMO_SEMELHANCA.csv',header=['valor'])

    # Projeção descritiva 2D; NÃO usada para treinar kNN nem provar separabilidade.
    amostra=np.concatenate([a,b])
    if len(amostra)>3000: amostra=rng.choice(amostra,size=3000,replace=False)
    # PCA ajustado apenas em dev. Não incluir minutos nem rotulos como atributos.
    xy=PCA(n_components=2,random_state=42).fit(z).transform(z[amostra])
    cores=np.where(tempo.iloc[amostra].to_numpy()<150,'#d59243','#158276')
    fig,ax=plt.subplots(figsize=(9,6))
    ax.scatter(xy[:,0],xy[:,1],c=cores,alpha=.27,s=15,edgecolors='none')
    ax.scatter([],[],c='#d59243',label='Moderada 135–149 min')
    ax.scatter([],[],c='#158276',label='Moderada 150–165 min')
    ax.set(title='Perfis da faixa moderada projetados em 2D (exploratório)',
           xlabel='Componente principal 1', ylabel='Componente principal 2')
    ax.legend();fig.tight_layout();salvar(fig,'06_projecao_pca_perfis.png')
    return resumo


def executar():
    base, ext=ler_bases()
    y0,y1,mudar,tempo_all,limite_all=criar_alvo_alternativo(base,ext)
    X,_=separar_xy(base)
    indices=np.arange(len(base))
    dev,_test=train_test_split(indices, test_size=.20, stratify=y0, random_state=42)
    X_dev=regioes(X.iloc[dev].reset_index(drop=True))
    a=y0[dev]; b=y1[dev]
    meta=ext.iloc[dev].reset_index(drop=True)
    tempo=pd.to_numeric(meta['minutos_semanais'],errors='coerce')
    limite=pd.to_numeric(meta['limite_minutos'],errors='coerce')
    troca=mudar[dev]
    print(f'Base desenvolvimento={len(dev)} | teste reservado={len(_test)} NAO USADO')
    print(f'Mudancas de rotulo 0 -> 1 no desenvolvimento: {troca.sum()}')
    print('kNN k=5 | cinco regioes | Robust | 10 folds | SEM usar tempo como preditor')
    print('Avaliacao descritiva e hipotese alternativa; criterio original preservado.',flush=True)
    SAIDA.mkdir(parents=True,exist_ok=True)
    p0,p1,folds=predizer_oof(X_dev,a,b)
    dados=pd.DataFrame({'posicao_dev':np.arange(len(dev)), 'fold':folds,
                        'minutos_semanais':tempo, 'limite_minutos':limite,
                        'real_original':a,'real_alt':b,
                        'prob_original':p0,'prob_alt':p1})
    for nome,prob in [('original',p0),('alt',p1)]:
        for corte in LIMITES_DECISAO:
            dados[f'pred_{nome}_{str(corte).replace(".","_")}']=(prob>=corte-1e-12).astype(int)
    # Sem IDs ou demais dados pessoais: o CSV tem apenas posicao local e resultados.
    dados.to_csv(SAIDA/'PREVISOES_OOF_DEV.csv',index=False)
    linhas=[]
    for nm,y,p in [('original',a,p0),('alternativo',b,p1)]:
        for corte in LIMITES_DECISAO:
            linhas.append({'modelo':nm, **medidas(y,p,corte)})
    resultado=pd.DataFrame(linhas)
    resultado.to_csv(SAIDA/'COMPARACAO_METRICAS.csv',index=False)
    print('\n=== METRICAS_GLOBAIS_OOF ===\n'+resultado.round(4).to_string(index=False))

    faixa=graficos_faixa(dados)
    graficos_metricas(resultado)
    resumo_perfis=perfis_semelhantes(X_dev,dados,tempo,limite)
    # Taxas na faixa com rotulo original (o alternativo muda o denominador).
    bloco=[]
    for nm,y,p in [('original',a,p0),('alternativo',b,p1)]:
        m=tempo.between(135,165).to_numpy()
        for corte in LIMITES_DECISAO:
            real=y[m]
            pre=(p[m]>=corte-1e-12).astype(int)
            tn,fp,fn,tp=confusion_matrix(real,pre,labels=[0,1]).ravel()
            bloco.append({'modelo':nm,'limiar':corte,'n_faixa':int(m.sum()),
                          'vn':int(tn),'fp':int(fp),'fn':int(fn),'vp':int(tp),
                          'recall_faixa':tp/(fn+tp) if fn+tp else np.nan})
    tabela_faixa=pd.DataFrame(bloco)
    tabela_faixa.to_csv(SAIDA/'CONFUSAO_FAIXA.csv',index=False)
    print('\n=== CONFUSAO_FAIXA_135_165 ===\n'+tabela_faixa.round(4).to_string(index=False))
    print('\n=== PERFIS ===\n',resumo_perfis)
    texto = (
        'DIAGNOSTICO LOCAL PNS 2019 - FRONTEIRA 135-165 MINUTOS\n'
        'Criterio original: atividade moderada >=150 min/semana; vigorosa >=75.\n'
        'Criterio alternativo EXPLORATORIO: atividade moderada >=135; vigorosa >=75.\n'
        f'Registros na dev: {len(dev)}; casos reclassificados 0->1 na dev: {int(troca.sum())}.\n'
        f'Registros entre 135 e 165 min na dev: {len(faixa)}.\n'
        'A nova definicao do alvo NAO demonstra melhora do modelo, pois muda a pergunta.\n'
        'Minutos, modalidade e limiar de atividade NAO entraram no X.\n'
        'O limiar 0.4/0.5 refere-se aos VOTOS DO kNN; 135/150 refere-se ao ROTULO.\n'
        'OOF com mesmas particoes originais para comparar as definicoes.\n'
        'PCA 2D e distancias sao exploratorios e NAO provam causalidade.\n'
        'Limites: comportamento humano pode depender de variaveis nao observadas,\n'
        'mas o estudo nao prova que esta seja a unica explicacao dos erros.\n'
        'Amostra nao ponderada pelo desenho amostral; sem inferencia populacional.\n'
        '\nMETRICAS:\n'+resultado.round(4).to_string(index=False)+'\n\n'
        'FAIXA:\n'+tabela_faixa.round(4).to_string(index=False)+'\n\n'
        'PERFIS:\n'+str(resumo_perfis)+'\n'
    )
    (SAIDA/'LEIA_ME_RESULTADOS.txt').write_text(texto,encoding='utf-8')
    print('\nPronto. Graficos e tabelas em:',SAIDA)


if __name__ == '__main__':
    executar()
