
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessamento import separar_xy
from src.treinamento import treinar_knn

base = pd.read_parquet(
    "data/interim/pns_base_eda.parquet"
)

X, y = separar_xy(base)

# Separar antes de qualquer ajuste do modelo
X_treino, _, y_treino, _ = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Amostra pequena apenas para testar o treinamento
X_piloto, _, y_piloto, _ = train_test_split(
    X_treino, y_treino,
    train_size=2000,
    stratify=y_treino,
    random_state=42
)

modelo = treinar_knn(X_piloto, y_piloto)

assert modelo.named_steps["knn"].n_samples_fit_ == 2000

print("Treinamento concluído com sucesso!")
print("Registros utilizados:", len(y_piloto))

