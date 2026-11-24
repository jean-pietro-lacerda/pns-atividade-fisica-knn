#!/usr/bin/env python3
"""Exporta minutos reais da PNS sem abrir/rodar o notebook no VS Code.

Reutiliza apenas as transformacoes de 01_extracao.ipynb, usando
pns_extraida.parquet se existir. Nao altera nenhum arquivo oficial.
Antes de gravar, compara linha a linha com pns_base_eda.parquet.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "01_extracao.ipynb"
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "PNS_2019.txt"
EXTRAIDA = INTERIM / "pns_extraida.parquet"
OFICIAL = INTERIM / "pns_base_eda.parquet"
DESTINO = INTERIM / "pns_base_eda_com_tempo.parquet"

PREDITORES = [
    "idade", "renda_per_capita", "numero_moradores", "sexo",
    "escolaridade", "uf", "situacao_trabalho", "area_residencia",
    "raca_cor", "estado_civil",
]
COLUNAS = PREDITORES + [
    "ativo_lazer", "minutos_semanais", "limite_minutos", "P034", "P035"
]

# Numero das celulas DE CODIGO da versao original? Estes sao indices
# absolutos (incluem markdown); exportacao deve ter sido acrescentada
# DEPOIS da celula que calcula o Y, portanto nao afeta indices abaixo.
CELULAS = [0, 11, 16, 25, 26, 27, 29, 31, 32, 33, 35, 36, 38, 40]
ANCORAS = {
    0: "ROOT =",
    11: "df_ativos",
    16: "minutos_diarios",
    25: 'df["VDE001"]',
    26: "df_tratado",
    27: "situacao_trabalho",
    29: "dropna",
    31: "df_visual",
    32: "mapa_sexo",
    33: "colunas_visualizacao",
    35: "raca_cor",
    36: "df_final",
    38: "minutos_semanais",
    40: "limite_minutos",
}


def executar_celula(celulas: list, indice: int, escopo: dict) -> None:
    cel = celulas[indice]
    codigo = cel.get("source", "")
    if isinstance(codigo, list):
        codigo = "".join(codigo)
    if cel.get("cell_type") != "code" or ANCORAS[indice] not in codigo:
        raise RuntimeError(
            f"O notebook difere da versao esperada na celula {indice+1}. "
            "Nao vou reconstruir uma base potencialmente incorreta."
        )
    try:
        exec(compile(codigo, str(NOTEBOOK) + f":celula_{indice+1}", "exec"), escopo)
    except Exception as exc:
        raise RuntimeError(
            f"Erro ao reproduzir a celula {indice+1} do notebook: {exc}"
        ) from exc


def comparar_com_oficial(gerada: pd.DataFrame, referencia: pd.DataFrame) -> None:
    if len(gerada) != len(referencia):
        raise RuntimeError(
            f"Quantidade de linhas difere: exportacao={len(gerada)}, "
            f"base oficial={len(referencia)}. Nenhum arquivo foi gravado."
        )
    for c in PREDITORES + ["ativo_lazer"]:
        if c not in referencia.columns:
            raise RuntimeError(f"A base oficial nao possui a coluna {c!r}.")
        a, b = gerada[c].reset_index(drop=True), referencia[c].reset_index(drop=True)
        if c in ("idade", "renda_per_capita", "numero_moradores", "ativo_lazer"):
            a = pd.to_numeric(a, errors="coerce")
            b = pd.to_numeric(b, errors="coerce")
            igual = np.isclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                equal_nan=True, rtol=1e-10, atol=1e-10
            )
        else:
            igual = a.astype("string").fillna("<NA>").eq(
                b.astype("string").fillna("<NA>")
            ).to_numpy(dtype=bool)
        if not bool(np.all(igual)):
            i = int(np.flatnonzero(~igual)[0])
            raise RuntimeError(
                f"DESALINHAMENTO em {c!r}, linha {i+1}: "
                f"exportacao={a.iloc[i]!r} vs oficial={b.iloc[i]!r}. "
                "Nada foi salvo. Verifique se as bases sao da mesma versao."
            )


def main() -> None:
    os.chdir(ROOT)

    print("=== EXPORTACAO COMPLEMENTAR DA PNS (SEM JUPYTER) ===", flush=True)
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Notebook nao encontrado: {NOTEBOOK}")
    if not OFICIAL.is_file():
        raise FileNotFoundError(
            f"Base oficial nao encontrada: {OFICIAL}\n"
            "Nao posso validar a ordem dos registros sem ela."
        )
    if not RAW.is_file():
        raise FileNotFoundError(
            f"Microdados nao encontrados: {RAW}\n"
            "O notebook usa esse TXT para recuperar situacao de trabalho."
        )

    print("Lendo notebook original e base de comparacao...", flush=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celulas = notebook["cells"]
    base_oficial = pd.read_parquet(OFICIAL)

    # Para nao reextrair todos os registros do TXT a toa.
    if EXTRAIDA.is_file():
        print("Usando base extraida ja existente (somente leitura).", flush=True)
        dados = pd.read_parquet(EXTRAIDA)
    else:
        print("Base extraida ausente; recriando em memoria a partir do TXT.", flush=True)
        escopo_ini = {
        "__name__": "__main__",
        "display": lambda *args, **kwargs: None,
        "txt_path": RAW,
        "pd": pd,
            }
        # Celula 2 define variaveis, celula 4 faz a extracao sem salvar.
        for indice, ancora in ((2, "variaveis_y"), (4, "linhas_invalidas")):
            cel = celulas[indice]
            codigo = cel.get("source", "")
            codigo = "".join(codigo) if isinstance(codigo, list) else codigo
            if ancora not in codigo:
                raise RuntimeError(f"Notebook inesperado na celula {indice+1}.")
            exec(compile(codigo, str(NOTEBOOK), "exec"), escopo_ini)
        dados = escopo_ini["df"]

    escopo = {
        "__name__": "__main__",
        "display": lambda *args, **kwargs: None,
        "df": dados.copy(),
    }
    for indice in CELULAS:
        # Celula 0 define os caminhos; nao sobrescreve df.
        # As demais reproduzem a limpeza e criacao original do alvo.
        executar_celula(celulas, indice, escopo)
        if indice == 0:
            escopo["txt_path"] = RAW
        print(f"Etapa do notebook {indice+1}/41 concluida.", flush=True)

    final = escopo["df_final"]
    faltantes = sorted(set(COLUNAS) - set(final.columns))
    if faltantes:
        raise RuntimeError(f"Faltam colunas de exportacao: {faltantes}")

    candidato = final[COLUNAS].reset_index(drop=True).copy()
    print("Validando TODOS os preditores e rotulos, linha a linha...", flush=True)
    comparar_com_oficial(candidato, base_oficial)

    if len(candidato) != 88456:
        raise RuntimeError(
            f"Esperados 88456 entrevistados, recebidos {len(candidato)}. "
            "Nenhum arquivo foi salvo."
        )
    if candidato["ativo_lazer"].isna().any():
        raise RuntimeError("Existem rotulos ausentes; nao sera salvo.")

    INTERIM.mkdir(parents=True, exist_ok=True)
    # Nao sobrescreve exportacao complementar caso ja exista.
    if DESTINO.exists():
        raise FileExistsError(
            f"Ja existe: {DESTINO}. Para evitar sobrescrita, nada foi gravado."
        )
    candidato.to_parquet(DESTINO, index=False)

    print("\nSUCESSO: exportacao criada e alinhamento verificado.")
    print("Arquivo:", DESTINO.resolve())
    print("Registros:", len(candidato))
    print("Distribuicao do alvo:", candidato["ativo_lazer"].value_counts().to_dict())
    print("\nProximo comando:")
    print(r".venv\Scripts\python.exe analisar_fronteira_135_165_pedro.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"\nERRO: {erro}", file=sys.stderr, flush=True)
        sys.exit(1)
