"""Exporta rankings e figuras a partir da auditoria concluída."""

import csv
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auditar_grade import ler_csv


def escrever_ranking(destino, linhas, metrica):
    classificadas = sorted(linhas, key=lambda linha: (-linha[metrica], linha["id"]))
    colunas = ["posicao", *linhas[0]]
    with destino.open("w", newline="", encoding="utf-8-sig") as arquivo:
        csv_writer = csv.DictWriter(arquivo, fieldnames=colunas)
        csv_writer.writeheader()
        for posicao, linha in enumerate(classificadas, 1):
            csv_writer.writerow({"posicao": posicao, **linha})


def salvar(figura, destino):
    figura.savefig(destino, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figura)


def dispersao(linhas, destino):
    figura, eixo = plt.subplots(figsize=(8.5, 5))
    for prefixo, rotulo, cor in (("G", "110 receitas novas", "#137D8A"),
                                 ("H", "40 receitas históricas", "#AD5D1A")):
        grupo = [x for x in linhas if x["id"].startswith(prefixo)]
        eixo.scatter([x["auc_media"] for x in grupo], [x["f1_media"] for x in grupo],
                     alpha=0.66, label=rotulo, color=cor, edgecolor="none")
    anotacoes = {"H03": (-19, -27), "H15": (12, -23), "H17": (-23, 8),
                 "H19": (-34, 4), "G098": (5, 10), "G320": (6, -10)}
    for linha in linhas:
        if linha["id"] in anotacoes:
            eixo.annotate(linha["id"], (linha["auc_media"], linha["f1_media"]),
                          xytext=anotacoes[linha["id"]], textcoords="offset points",
                          fontsize=9, color="#142E43", fontweight="bold",
                          arrowprops={"arrowstyle": "-", "color": "#71808B", "lw": 0.65})
    eixo.set(title="AUC-ROC e F1 nas 150 configurações",
             xlabel="AUC-ROC média nos 10 folds", ylabel="F1 médio nos 10 folds")
    eixo.legend(loc="upper right", frameon=False)
    eixo.grid(alpha=0.18)
    salvar(figura, destino)


def roc(dados, destino):
    figura, eixo = plt.subplots(figsize=(7, 5.3))
    for id_receita, rotulo, cor in (
        ("G098", "G098 · maior AUC média entre as novas", "#137D8A"),
        ("G320", "G320 · maior F1 médio entre as novas", "#8F4A16"),
    ):
        acumulados = [[0, 0] for _ in range(6)]
        for linha in dados:
            if linha["id"] == id_receita:
                par = acumulados[int(linha["votos_ativos"])]
                par[0] += int(linha["inativos_reais"])
                par[1] += int(linha["ativos_reais"])
        n0, n1 = map(sum, zip(*acumulados))
        assert (n0, n1) == (50712, 20052)
        fpr, tpr = [0], [0]
        soma0 = soma1 = 0
        for neg, pos in reversed(acumulados):
            soma0 += neg
            soma1 += pos
            fpr.append(soma0 / n0)
            tpr.append(soma1 / n1)
        auc = sum((b - a) * (d + c) / 2
                  for a, b, c, d in zip(fpr[:-1], fpr[1:], tpr[:-1], tpr[1:]))
        eixo.plot(fpr, tpr, "o-", label=f"{rotulo} · AUC OOF {auc:.4f}".replace(".", ","),
                  color=cor, markersize=4)
    eixo.plot([0, 1], [0, 1], "--", color="#87929A", linewidth=1)
    eixo.set(title="Curvas ROC com os votos reunidos fora do treino",
             xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
             xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    eixo.legend(loc="lower right", frameon=False, fontsize=8)
    eixo.grid(alpha=0.18)
    salvar(figura, destino)


def main():
    caminho_json, evidencias, saida = map(Path, sys.argv[1:])
    linhas = json.loads(caminho_json.read_text(encoding="utf-8"))["configuracoes"]
    saida.mkdir(exist_ok=True, parents=True)
    for nome, metrica in (("ranking_auc.csv", "auc_media"), ("ranking_f1.csv", "f1_media")):
        escrever_ranking(saida / nome, linhas, metrica)
    dispersao(linhas, saida / "auc_f1_grade.png")
    with ZipFile(evidencias) as pacote:
        roc(ler_csv(pacote, "frequencias_escores_110.csv"), saida / "roc_receitas_novas.png")


if __name__ == "__main__":
    main()
