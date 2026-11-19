
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessamento import separar_xy
from src.treinamento import treinar_knn


def executar_treinamentos():
    # Base ainda sem escalonamento, imputação ou PCA ajustados.
    base = pd.read_parquet("data/interim/pns_base_eda.parquet")

    X, y = separar_xy(base)

    # Uma única divisão para as três configurações.
    # O conjunto de teste permanece completamente reservado.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    configuracoes = {
        "standard": {"escala": "standard"},
        "minmax": {"escala": "minmax"},
        "robust": {"escala": "robust"},
    }

    modelos = {}

    for nome, configuracao in configuracoes.items():
        print(f"Treinando kNN com {nome}...", flush=True)

        # A função cria um pipeline novo e ajusta
        # todas as transformações SOMENTE em X_treino.
        modelo = treinar_knn(
            X_treino,
            y_treino,
            configuracao=configuracao,
        )

        # Confere quantas observações chegaram ao kNN.
        assert (
            modelo.named_steps["knn"].n_samples_fit_
            == len(X_treino)
        )

        modelos[nome] = modelo

        print(f"Concluído: {nome}", flush=True)

    print("\nTodos os treinamentos foram concluídos.")
    print(f"Registros de treino: {len(X_treino)}")
    print(f"Registros reservados: {len(X_teste)}")

    # Modelos disponíveis para utilização posterior.
    # Nenhum modelo ou dado é salvo automaticamente.
    return modelos


if __name__ == "__main__":
    executar_treinamentos()
