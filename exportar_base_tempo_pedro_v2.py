#!/usr/bin/env python3
"""Recupera minutos semanais a partir dos Parquets existentes da PNS.

NAO depende de PNS_2019.txt nem do Jupyter.
NAO altera o arquivo oficial. So gera uma exportacao complementar
SE todas as linhas forem alinhadas (9 preditores + alvo) com a base oficial.
Reproduz as regras de 01_extracao.ipynb (PNS 2019).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTERIM = ROOT / 'data' / 'interim'
OFICIAL = INTERIM / 'pns_base_eda.parquet'
FONTES = [INTERIM / 'pns_extraida.parquet', INTERIM / 'pns_limpa_pre_y.parquet']
DESTINO = INTERIM / 'pns_base_eda_com_tempo.parquet'

NUMERICOS = ['idade', 'renda_per_capita', 'numero_moradores']
CATEGORICOS = ['sexo', 'escolaridade', 'uf', 'situacao_trabalho',
               'area_residencia', 'raca_cor', 'estado_civil']
ATRIBUTOS = NUMERICOS + CATEGORICOS

MAPAS = {
    'sexo': {'1': 'Homem', '2': 'Mulher'},
    'area_residencia': {'1': 'Urbana', '2': 'Rural'},
    'escolaridade': {
        '1': 'Sem instrução', '2': 'Fundamental incompleto',
        '3': 'Fundamental completo', '4': 'Médio incompleto',
        '5': 'Médio completo', '6': 'Superior incompleto',
        '7': 'Superior completo',
    },
    'raca_cor': {
        '1': 'Branca', '2': 'Preta', '3': 'Amarela',
        '4': 'Parda', '5': 'Indígena',
    },
    'estado_civil': {
        '1': 'Casado(a)', '2': 'Divorciado(a) ou separado(a)',
        '3': 'Viúvo(a)', '4': 'Solteiro(a)',
    },
}

MODERADAS = {'01', '02', '07', '08', '09', '10', '11', '14', '16', '17'}
VIGOROSAS = {'03', '04', '05', '06', '12', '13', '15'}


def codigos(serie):
    """Normaliza codigos numericos sem alterar a ordem das linhas."""
    return serie.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def reconstruir(dados: pd.DataFrame) -> pd.DataFrame:
    necessarias = [
        'P034', 'P035', 'P03701', 'P03702', 'P036',
        'C008', 'VDF003', 'V0022', 'C006', 'VDD004A', 'V0001',
        'V0026', 'C009', 'C011',
    ]
    faltam = sorted(set(necessarias) - set(dados.columns))
    if faltam:
        raise RuntimeError(f'Fonte intermediaria incompleta: faltam {faltam}')

    bruto = dados.copy()
    p034 = codigos(bruto['P034'])
    freq = pd.to_numeric(bruto['P035'], errors='coerce')
    horas = pd.to_numeric(bruto['P03701'], errors='coerce')
    minutos = pd.to_numeric(bruto['P03702'], errors='coerce')

    # Notebooks/01_extracao: remover praticantes regulares que declararam
    # mais de 6 horas de atividade por dia (antes da formacao do alvo).
    excluir = (p034.eq('1').fillna(False)
               & freq.between(1, 7).fillna(False)
               & (horas * 60 + minutos).gt(360).fillna(False))
    bruto = bruto.loc[~excluir].copy()

    # Mesmo encadeamento final de selecoes do notebook.
    p034 = codigos(bruto['P034'])
    bruto = bruto.loc[p034.isin(['1', '2']).fillna(False)].copy()
    bruto = bruto.dropna(subset=['VDF003']).copy()

    nomes = {
        'C008': 'idade', 'C006': 'sexo', 'VDD004A': 'escolaridade',
        'VDF003': 'renda_per_capita', 'V0001': 'uf',
        'V0026': 'area_residencia', 'C009': 'raca_cor',
        'C011': 'estado_civil', 'V0022': 'numero_moradores',
    }
    bruto = bruto.rename(columns=nomes)
    for coluna, mapa in MAPAS.items():
        bruto[coluna] = codigos(bruto[coluna]).map(mapa)
    bruto = bruto.dropna(subset=['raca_cor']).copy()

    bruto['idade'] = pd.to_numeric(bruto['idade'], errors='coerce')
    bruto = bruto.loc[bruto['idade'] >= 18].copy()

    p034 = codigos(bruto['P034'])
    freq = pd.to_numeric(bruto['P035'], errors='coerce')
    horas = pd.to_numeric(bruto['P03701'], errors='coerce')
    minutos = pd.to_numeric(bruto['P03702'], errors='coerce')
    tempo = freq * (60 * horas + minutos)
    nao_praticam = p034.eq('2').fillna(False)
    frequencia_zero = p034.eq('1').fillna(False) & freq.eq(0).fillna(False)
    tempo.loc[nao_praticam | frequencia_zero] = 0

    modalidade = codigos(bruto['P036']).str.zfill(2)
    limite = np.select(
        [modalidade.isin(MODERADAS).fillna(False),
         modalidade.isin(VIGOROSAS).fillna(False)],
        [150.0, 75.0], default=np.nan,
    )

    # Os 150/75 minutos constituem o alvo ORIGINAL do projeto.
    y = pd.Series(pd.NA, index=bruto.index, dtype='Int64')
    y.loc[nao_praticam | frequencia_zero] = 0
    valido = (p034.eq('1').fillna(False) & freq.between(1, 7).fillna(False)
              & tempo.notna() & tempo.ge(0).fillna(False)
              & pd.Series(limite, index=bruto.index).notna())
    y.loc[valido] = (tempo.loc[valido] >= limite[valido.to_numpy()]).astype(int)

    bruto['minutos_semanais'] = tempo
    bruto['limite_minutos'] = limite
    bruto['ativo_lazer'] = y
    bruto['P034'] = p034
    bruto['P035'] = freq
    return bruto.reset_index(drop=True)


def verificar_linha_a_linha(reconstruida: pd.DataFrame, base: pd.DataFrame) -> None:
    if len(reconstruida) != len(base):
        raise RuntimeError(
            f'Bases com tamanhos diferentes: fonte reconstruida={len(reconstruida)} '
            f'vs oficial={len(base)}. NADA foi salvo.\n'
            'Talvez o parquet de origem seja de outra versao da extracao.'
        )
    # A situacao de trabalho depende de VDE001, recuperada no notebook via TXT.
    # Como o TXT nao esta presente, essa coluna vem da base oficial APOS
    # confirmar, na ordem exata, TODOS os demais 9 atributos e o alvo.
    conferidos = [c for c in ATRIBUTOS if c != 'situacao_trabalho'] + ['ativo_lazer']
    for col in conferidos:
        if col not in base.columns:
            raise RuntimeError(f'A base oficial nao tem a coluna {col!r}.')
        a = reconstruida[col].reset_index(drop=True)
        b = base[col].reset_index(drop=True)
        if col in NUMERICOS + ['ativo_lazer']:
            aa = pd.to_numeric(a, errors='coerce').to_numpy(dtype=float)
            bb = pd.to_numeric(b, errors='coerce').to_numpy(dtype=float)
            igual = np.isclose(aa, bb, rtol=1e-10, atol=1e-8, equal_nan=True)
        else:
            igual = (a.astype('string').fillna('<NA>').str.strip().to_numpy()
                     == b.astype('string').fillna('<NA>').str.strip().to_numpy())
        if not np.all(igual):
            pos = int(np.flatnonzero(~igual)[0])
            raise RuntimeError(
                f'DESALINHAMENTO na coluna {col!r}, registro {pos+1}: '
                f'fonte={a.iloc[pos]!r}; oficial={b.iloc[pos]!r}.\n'
                'Nao associe tempos a outras pessoas. Nenhum arquivo foi salvo.'
            )
    if reconstruida['ativo_lazer'].isna().any():
        raise RuntimeError('Ha rotulos indefinidos; nenhum arquivo foi salvo.')


def main():
    print('=== EXPORTACAO SEM TXT E SEM JUPYTER ===', flush=True)
    if not OFICIAL.is_file():
        raise FileNotFoundError(f'Base oficial ausente: {OFICIAL}')
    disponiveis = [p for p in FONTES if p.is_file()]
    if not disponiveis:
        arquivos = [str(p.relative_to(ROOT)) for p in sorted(INTERIM.glob('*.parquet'))] if INTERIM.exists() else []
        raise FileNotFoundError(
            'Nao encontrei nenhum destes arquivos intermediarios:\n'
            + '\n'.join('  '+str(p) for p in FONTES)
            + '\nParquets existentes em data/interim:\n'
            + ('\n'.join('  '+p for p in arquivos) if arquivos else '  (nenhum)')
            + '\n\nMostre esta lista para identificarmos a fonte correta.'
        )
    if DESTINO.exists():
        raise FileExistsError(
            f'A exportacao complementar ja existe em {DESTINO}. '
            'Nao vou sobrescrever automaticamente.'
        )
    base = pd.read_parquet(OFICIAL).reset_index(drop=True)
    erros = []
    for fonte in disponiveis:
        try:
            print(f'Lendo {fonte.name}...', flush=True)
            bruto = pd.read_parquet(fonte)
            reconstruida = reconstruir(bruto)
            print(f'Reconstruidos {len(reconstruida)} registros; validando linha a linha...', flush=True)
            verificar_linha_a_linha(reconstruida, base)
            # Agora pode usar a situacao de trabalho JA VALIDADA NA BASE OFICIAL:
            # em nenhuma circunstancia usar os minutos como preditores X.
            exportacao = base[ATRIBUTOS + ['ativo_lazer']].copy()
            for col in ['minutos_semanais', 'limite_minutos', 'P034', 'P035']:
                exportacao[col] = reconstruida[col].to_numpy()
            exportacao.to_parquet(DESTINO, index=False)
            print('\nSUCESSO: 88.456 registros na ordem correta!' if len(base) == 88456
                  else f'\nSUCESSO: {len(base)} registros na ordem correta!')
            print('Origem:', fonte.name)
            print('Arquivo:', DESTINO)
            print('Proximo comando:')
            print(r'.venv\Scripts\python.exe analisar_fronteira_135_165_pedro.py')
            return
        except (RuntimeError, ValueError, KeyError) as erro:
            erros.append(f'{fonte.name}: {erro}')
            print(f'Essa fonte nao passou na verificacao: {erro}', flush=True)
    raise RuntimeError('Nenhuma fonte intermediaria foi aprovada:\n' + '\n'.join(erros))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('\nERRO:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
