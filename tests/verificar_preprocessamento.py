"""Verificações funcionais com dados reais e execução integral do notebook 03."""

import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

import nbformat
import numpy as np
import pandas as pd
import sklearn
from jupyter_client import KernelManager
from nbclient import NotebookClient
from scipy.spatial.distance import pdist
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessamento import (
    ALVO,
    CATEGORICAS,
    NUMERICAS,
    TratamentoExtremos,
    construir_pipeline,
    construir_preprocessador,
    configuracoes_experimentais,
    separar_xy,
    validacao_cruzada,
)

OUT = ROOT / "outputs/preprocessamento"
checks = []


def ok(name):
    """Registra uma verificação concluída e informa seu nome no terminal."""
    checks.append(name)
    print("OK:", name, flush=True)


def grupo_equivalente(valores, referencias):
    """Retorna o índice de matriz equivalente ou None, com tolerâncias fixas."""
    return next(
        (
            indice
            for indice, referencia in enumerate(referencias)
            if referencia.shape == valores.shape
            and np.allclose(referencia, valores, rtol=1e-7, atol=1e-9)
        ),
        None,
    )


def executar_notebook():
    """Executa todas as células, salva as saídas e retorna o total de PNGs."""
    notebook = nbformat.read(
        ROOT / "notebooks/03_preprocessamento.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook, timeout=600, resources={"metadata": {"path": str(ROOT)}}
    )
    client.kernel_manager_class = KernelManager
    client.execute()
    nbformat.write(notebook, ROOT / "notebooks/03_preprocessamento.ipynb")
    return sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )


def main():
    """Executa as verificações e grava diagnósticos em outputs/preprocessamento."""
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    checks.clear()
    path = ROOT / "data/interim/pns_base_eda.parquet"
    protected_paths = [
        ROOT / "notebooks/01_extracao.ipynb",
        ROOT / "notebooks/02_eda.ipynb",
        path,
    ]
    protected_hashes = {
        protected: hashlib.sha256(protected.read_bytes()).hexdigest()
        for protected in protected_paths
    }
    before = protected_hashes[path]
    df = pd.read_parquet(path)
    assert df.shape == (88456, 11) and (not df.isna().any().any())
    X, y = separar_xy(df)
    assert list(X) == NUMERICAS + CATEGORICAS
    assert ALVO not in X
    assert all(
        (r["imputacao"] == "median" for r in configuracoes_experimentais())
    )
    ok("Base real lida; 88456 x 11; sem nulos; dez preditores explicitos")
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X, y, train_size=240, test_size=60, stratify=y, random_state=42
    )
    recipes = configuracoes_experimentais()
    assert (
        len(recipes)
        == len({json.dumps(r, sort_keys=True) for r in recipes})
        == 385
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        representations = []
        geometries = []
        recipe_audit = []
        for recipe in recipes:
            modelo = construir_pipeline(**recipe).fit(X_treino, y_treino)
            transformados = modelo[:-1].transform(X_validacao)
            assert np.issubdtype(transformados.dtype, np.number)
            assert (
                np.isfinite(transformados).all()
                and transformados.shape[0] == 60
            )
            assert modelo.predict(X_validacao).shape == (60,)
            assert modelo.predict_proba(X_validacao).shape == (60, 2)
            assert (
                modelo.named_steps["knn"].get_params()
                == KNeighborsClassifier().get_params()
            )
            probe = modelo[:-1].transform(pd.concat([X_treino, X_validacao]))
            # Inclui treino e validação apenas para diagnosticar equivalências.
            group = grupo_equivalente(probe, representations)
            if group is None:
                group = len(representations)
                representations.append(probe)
            distances = pdist(probe)
            distances /= max(distances.max(), np.finfo(float).eps)
            # A normalização detecta também escalas globais e rotações.
            geometry = grupo_equivalente(distances, geometries)
            if geometry is None:
                geometry = len(geometries)
                geometries.append(distances)
            recipe_audit.append(
                dict(
                    recipe=recipe,
                    columns=probe.shape[1],
                    representation_group=group,
                    geometry_group=geometry,
                )
            )
        (OUT / "equivalencias_configuracoes.json").write_text(
            json.dumps(
                {
                    "combinations": len(recipes),
                    "observed_representations": len(representations),
                    "observed_normalized_geometries": len(geometries),
                    "train_rows": 240,
                    "validation_rows": 60,
                    "rtol": 1e-07,
                    "atol": 1e-09,
                    "scope": "Diagnostic on fixed real sample; not proof for all rows/folds",
                    "recipes": recipe_audit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        assert len(geometries) >= 150, len(geometries)
        ok(
            f"{len(recipes)} receitas testadas; {len(representations)} matrizes e {len(geometries)} geometrias normalizadas observadas em 300 registros reais"
        )
        np.testing.assert_allclose(
            construir_preprocessador(imputacao="mean").fit_transform(X_treino),
            construir_preprocessador(imputacao="median").fit_transform(
                X_treino
            ),
        )
        ok(
            "Mean/median equivalentes sem ausentes; mean segue disponivel fora da grade"
        )
        for scale in ("standard", "minmax", "robust"):
            preprocessador = construir_preprocessador(escala=scale).fit(
                X_treino, y_treino
            )
            transformados = preprocessador.transform(X_treino)[:, :3]
            if scale == "standard":
                np.testing.assert_allclose(
                    transformados.mean(axis=0), 0, atol=1e-12
                )
                np.testing.assert_allclose(transformados.std(axis=0), 1)
            if scale == "minmax":
                np.testing.assert_allclose(
                    transformados.min(axis=0), 0, atol=1e-12
                )
                np.testing.assert_allclose(transformados.max(axis=0), 1)
            cat = preprocessador.transform(X_treino)[:, 3:]
            assert set(np.unique(cat)) <= {0, 1}
            np.testing.assert_allclose(cat.sum(axis=1), 7)
            num = preprocessador.named_transformers_["numericas"]
            snapshot = num.named_steps["imputacao"].statistics_.copy()
            shifted = X_validacao.copy()
            shifted["renda_per_capita"] = 1000000000000.0
            shifted["uf"] = "categoria_nova"
            assert np.isfinite(preprocessador.transform(shifted)).all()
            np.testing.assert_array_equal(
                snapshot, num.named_steps["imputacao"].statistics_
            )
        ok(
            "Standard media 0/desvio 1; MinMax [0,1]; Robust; one-hot binario; categoria nova; transform nao reajusta"
        )
        for strategy in ("mean", "median"):
            missing = X_treino.copy()
            missing.loc[missing.index[:10], NUMERICAS + CATEGORICAS] = np.nan
            for coluna in NUMERICAS + CATEGORICAS:
                missing[coluna] = (
                    missing[coluna].astype(object)
                    if coluna in CATEGORICAS
                    else missing[coluna]
                )
            preprocessador = construir_preprocessador(imputacao=strategy).fit(
                missing, y_treino
            )
            assert np.isfinite(preprocessador.transform(missing)).all()
            stats = (
                preprocessador.named_transformers_["numericas"]
                .named_steps["imputacao"]
                .statistics_
            )
            expected = (
                missing[NUMERICAS].mean()
                if strategy == "mean"
                else missing[NUMERICAS].median()
            )
            np.testing.assert_allclose(stats, expected)
            empty = X_treino.copy()
            empty[NUMERICAS + CATEGORICAS] = np.nan
            assert np.isfinite(
                construir_preprocessador(imputacao=strategy).fit_transform(
                    empty, y_treino
                )
            ).all()
        ok(
            "Ausencias injetadas: media/mediana e moda; todas as colunas vazias preservadas"
        )
        bad = df.iloc[:20].copy()
        bad.loc[bad.index[0], ALVO] = pd.NA
        try:
            separar_xy(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Alvo ausente aceito")
        for kwargs in (
            {"escala": "invalida"},
            {"imputacao": "moda"},
            {"faixas": 4},
            {"faixas": 3, "escala": "minmax"},
            {"selecao": 20},
            {"pca": 1},
        ):
            try:
                construir_pipeline(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(kwargs)
        ok("Rejeicao de alvo ausente e parametros invalidos")
        for ext in ("preservar", "log_renda", "iqr"):
            tratamento = TratamentoExtremos(ext).fit(X_treino[NUMERICAS])
            raw = X_validacao[NUMERICAS].to_numpy()
            expected = raw.copy()
            if ext == "log_renda":
                expected[:, 1] = np.log1p(raw[:, 1])
            if ext == "iqr":
                expected = np.clip(
                    raw, tratamento.inferior_, tratamento.superior_
                )
            np.testing.assert_allclose(
                tratamento.transform(X_validacao[NUMERICAS]), expected
            )
            q1, q3 = np.quantile(X_treino[NUMERICAS], [0.25, 0.75], axis=0)
            np.testing.assert_allclose(
                tratamento.superior_, q3 + 1.5 * (q3 - q1)
            )
        catpipe = (
            construir_preprocessador()
            .fit(X_treino, y_treino)
            .named_transformers_["categoricas"]
        )
        missingcat = X_validacao[CATEGORICAS].copy()
        missingcat.iloc[0, :] = np.nan
        np.testing.assert_array_equal(
            catpipe.named_steps["imputacao"].transform(missingcat)[0],
            catpipe.named_steps["imputacao"].statistics_,
        )
        ok(
            "Preservar, log1p e IQR com resultados esperados; limites somente do treino; moda categorica conferida"
        )
        scores = cross_validate(
            construir_pipeline(),
            X_treino,
            y_treino,
            cv=validacao_cruzada(),
            scoring=["accuracy", "f1", "roc_auc"],
            error_score="raise",
            return_estimator=True,
        )
        assert all(
            (
                np.isfinite(scores["test_" + s]).all()
                for s in ("accuracy", "f1", "roc_auc")
            )
        )
        for (indices_treino, indices_validacao), estimador in zip(
            validacao_cruzada().split(X_treino, y_treino), scores["estimator"]
        ):
            stats = (
                estimador.named_steps["preprocessamento"]
                .named_transformers_["numericas"]
                .named_steps["imputacao"]
                .statistics_
            )
            np.testing.assert_allclose(
                stats, X_treino.iloc[indices_treino][NUMERICAS].median()
            )
        ok(
            "Integracao cross_validate: dez folds estratificados; estatisticas conferidas por treino; tres metricas finitas"
        )
        preprocessador = construir_preprocessador().fit(X, y)
        full = preprocessador.transform(X)
        assert np.isfinite(full).all()
        ok(
            f"Preprocessador padrao na base completa: {full.shape}, {full.nbytes} bytes"
        )
        images = executar_notebook()
        assert images >= 2, f"Boxplots nao capturados: {images}"
        ok(f"Notebook executado por nbclient; {images} imagens PNG capturadas")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    ok("SHA256 da base inalterado")
    for protected, digest in protected_hashes.items():
        assert hashlib.sha256(protected.read_bytes()).hexdigest() == digest
    ok("Notebooks 01/02 e Parquet preservados por SHA256")
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "sklearn": sklearn.__version__,
        "checks": checks,
        "sha256": before,
        "warnings": sorted(set((str(w.message) for w in captured))),
        "cv_smoke_means": {
            s: float(scores["test_" + s].mean())
            for s in ("accuracy", "f1", "roc_auc")
        },
    }
    report["configuration_audit"] = dict(
        combinations=len(recipes),
        observed_representations=len(representations),
        observed_normalized_geometries=len(geometries),
        rtol=1e-07,
        atol=1e-09,
        scope="Fixed 300-row real sample only",
    )
    (OUT / "validacao_final.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("VALIDACAO CONCLUIDA", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
