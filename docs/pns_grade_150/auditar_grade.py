"""Conferência offline das evidências do projeto CIN0208."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median, stdev
from zipfile import ZipFile


METRICAS = ("auc", "f1", "acuracia")
CELULAS = ("tn", "fp", "fn", "tp")
FOLDS = set(range(1, 11))


def ler_csv(pacote, nome):
    texto = pacote.read(nome).decode("utf-8-sig")
    separador = ";" if ";" in texto.splitlines()[0] else ","
    return list(csv.DictReader(texto.splitlines(), delimiter=separador))


def numero(valor):
    return float(valor.replace(",", "."))


def conferir(valor, esperado):
    assert isclose(valor, esperado, rel_tol=0, abs_tol=2e-12), (valor, esperado)


def verificar_historico(receita, linhas, original):
    chave = "escala" if "escala" in original[0] else "configuracao"
    origem = {
        int(linha["fold"]): linha
        for linha in original
        if linha[chave] == receita["configuracao_csv"]
        and ("limiar" not in linha or numero(linha["limiar"]) == 0.5)
        and ("repeticao" not in linha or int(linha["repeticao"]) == 1)
    }
    assert len(origem) == 10, receita["id"]
    for fold, valores in linhas.items():
        fonte = origem[fold]
        for metrica in METRICAS:
            conferir(numero(valores[metrica]), numero(fonte[metrica]))
        tn, fp, fn, tp = (int(valores[c]) for c in CELULAS)
        if all(c in fonte for c in CELULAS):
            assert (tn, fp, fn, tp) == tuple(int(fonte[c]) for c in CELULAS)
        else:
            conferir(tp / (tp + fn), numero(fonte["recall_ativo"]))
            conferir(tn / (tn + fp), numero(fonte["recall_inativo"]))


def verificar_escores(escores, linhas):
    assert set(escores) == set(linhas)
    for chave, hist in escores.items():
        assert set(hist) == set(range(6))
        negativos, positivos = zip(*(hist[votos] for votos in range(6)))
        n0, n1 = sum(negativos), sum(positivos)
        assert (sum(negativos[:3]), sum(negativos[3:]),
                sum(positivos[:3]), sum(positivos[3:])) == tuple(
            int(linhas[chave][c]) for c in CELULAS
        ), chave
        pares = sum(p * (2 * sum(negativos[:i]) + negativos[i])
                    for i, p in enumerate(positivos))
        conferir(numero(linhas[chave]["auc"]), pares / (2 * n0 * n1))


def auditar(pacote):
    assert pacote.testzip() is None
    configuracoes = ler_csv(pacote, "configuracoes.csv")
    resumo = ler_csv(pacote, "resumo_150.csv")
    por_fold = ler_csv(pacote, "metricas_1500_folds.csv")
    novas = ler_csv(pacote, "grade110/metricas_por_fold.csv")
    manifesto = json.loads(pacote.read("grade110/manifesto.json"))
    receitas_novas = {f"G{codigo:03d}": cfg for codigo, cfg in manifesto["recipes"]}
    assert len(receitas_novas) == len(manifesto["recipes"]) == 110
    assert len({json.dumps(r, sort_keys=True) for r in receitas_novas.values()}) == 110
    assinatura = manifesto["signature"]
    for campo, nome in (("runner", "codigo/grade_pns.py"),
                        ("pipeline", "codigo/src/preprocessamento.py")):
        assert hashlib.sha256(pacote.read(nome)).hexdigest() == assinatura[campo]
    assert (manifesto["split_seed"], manifesto["fold_seed"], assinatura["n"]) == (42, 42, 70764)

    ids = {linha["id"] for linha in configuracoes}
    assert len(configuracoes) == len(ids) == len(resumo) == 150
    assert len({linha["receita"] for linha in configuracoes}) == 150
    assert ids == {f"H{i:02d}" for i in range(1, 41)} | set(receitas_novas)
    resumidas = {linha["id"]: linha for linha in resumo}
    assert resumidas.keys() == ids
    grupos = defaultdict(dict)
    classes = {}
    for linha in por_fold:
        codigo, fold = linha["id"], int(linha["fold"])
        assert codigo in ids and fold not in grupos[codigo]
        grupos[codigo][fold] = linha
        tn, fp, fn, tp = (int(linha[c]) for c in CELULAS)
        assert min(tn, fp, fn, tp) >= 0
        n0, n1 = tn + fp, fn + tp
        assert (n0, n1) == classes.setdefault(fold, (n0, n1))
        assert all(0 <= numero(linha[k]) <= 1 for k in METRICAS)
        conferir(numero(linha["acuracia"]), (tn + tp) / (n0 + n1))
        conferir(numero(linha["f1"]), 2 * tp / (2 * tp + fp + fn))
    assert len(por_fold) == 1500 and grupos.keys() == ids
    assert set(classes) == FOLDS
    assert tuple(map(sum, zip(*classes.values()))) == (50712, 20052)
    assert max(sum(x) for x in classes.values()) - min(sum(x) for x in classes.values()) == 1
    assert all(max(col) - min(col) <= 1 for col in zip(*classes.values()))

    dados_g = {(f"G{int(x['id']):03d}", int(x["fold"])): x for x in novas}
    assert len(novas) == len(dados_g) == 1100
    tempos = defaultdict(list)
    dimensoes = defaultdict(list)
    arquivos = {}
    assinaturas = defaultdict(list)
    linhas_saida = []
    for receita in configuracoes:
        codigo = receita["id"]
        grupo = grupos[codigo]
        assert set(grupo) == FOLDS
        origem = receita["arquivo_origem"]
        if codigo.startswith("H"):
            origem = f"historico/{origem}"
        if origem not in arquivos:
            arquivos[origem] = (hashlib.sha256(pacote.read(origem)).hexdigest(), ler_csv(pacote, origem))
        hash_origem, linhas_originais = arquivos[origem]
        assert hash_origem == receita["sha256_csv"], codigo
        if codigo.startswith("H"):
            verificar_historico(receita, grupo, linhas_originais)
        else:
            assert json.loads(receita["receita"]) == receitas_novas[codigo]
            assert codigo == f"G{int(receita['configuracao_csv']):03d}"
            for fold, valores in grupo.items():
                fonte = dados_g[codigo, fold]
                for coluna in METRICAS + CELULAS:
                    conferir(numero(valores[coluna]), numero(fonte[coluna]))
                segundos, dimensao = numero(fonte["seconds"]), int(fonte["dim"])
                assert isfinite(segundos) and segundos > 0 and dimensao > 0
                tempos[codigo].append(segundos)
                dimensoes[codigo].append(dimensao)
        recalculado = {}
        for metrica in METRICAS:
            valores = [numero(linha[metrica]) for linha in grupo.values()]
            recalculado[f"{metrica}_media"] = mean(valores)
            recalculado[f"{metrica}_dp"] = stdev(valores)
        for celula in CELULAS:
            recalculado[f"{celula}_medio"] = mean(int(x[celula]) for x in grupo.values())
        for coluna, valor in recalculado.items():
            conferir(numero(resumidas[codigo][coluna]), valor)
        assinaturas[tuple(round(numero(grupo[fold][k]), 12)
                          for fold in sorted(FOLDS) for k in METRICAS + CELULAS)].append(codigo)
        linhas_saida.append({
            **receita, **recalculado,
            "segundos_media_cache": mean(tempos[codigo]) if tempos[codigo] else None,
            "dimensao_mediana": median(dimensoes[codigo]) if dimensoes[codigo] else None,
        })

    escores = defaultdict(dict)
    for linha in ler_csv(pacote, "frequencias_escores_110.csv"):
        chave = linha["id"], int(linha["fold"])
        votos = int(linha["votos_ativos"])
        assert votos not in escores[chave]
        freq = int(linha["inativos_reais"]), int(linha["ativos_reais"])
        assert min(freq) >= 0
        escores[chave][votos] = freq
    verificar_escores(escores, dados_g)
    return linhas_saida, classes, [x for x in assinaturas.values() if len(x) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidencias", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    with ZipFile(args.evidencias) as pacote:
        linhas, classes, repetidas = auditar(pacote)
    resultado = {"configuracoes": linhas, "classes_por_fold": classes,
                 "resultados_identicos": repetidas}
    if args.json:
        args.json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    for metrica in ("auc_media", "f1_media"):
        print(metrica)
        for linha in sorted(linhas, key=lambda x: -x[metrica])[:10]:
            print(linha["id"], linha[metrica], linha["receita"])
    print("Resultados identicos:", repetidas)


if __name__ == "__main__":
    main()
