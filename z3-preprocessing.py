from datasets import load_from_disk
import os

base_dir = os.getcwd()
load_dir = os.path.join(base_dir, "dataset", "Filtered_dataset")

dataset = load_from_disk(load_dir)

# dataframe apenas para calcular as contagens
df = dataset.to_pandas()

print(f"\ncolunas do dataset:\n {df.columns}")

# colunas que serão removidas
cols_to_remove = [
    "dataset",
    "original_path",
    "Latitude",
    "Longitude",
    "Humidity",
    "Temperature",
    "ObjID",
    "Depth_max",
    "Depth_min"
]

# verifica quantidade por classe
print(f"\nQuantidade de amostras, por classe:\n  {df['original_label'].value_counts()}")

threshold = 1000
upper_threshold = 5000

# máscara das linhas que serão mantidas
counts = df["original_label"].value_counts()

mask = (df["original_label"].map(counts).between(threshold, upper_threshold))

# índices das linhas válidas
indices_validos = df.index[mask].tolist()

print(f"\ntamanho do dataset:\n {len(indices_validos)}")

# filtra o dataset ORIGINAL
dataset_filtrado = dataset.select(indices_validos)

# dataframe após o threshold
df_filtrado = df[mask]
print(f"\nQuantidade de amostras por classe após filtragem:\n {df_filtrado['original_label'].value_counts()}")

# remove colunas inúteis diretamente no Dataset
dataset_filtrado = dataset_filtrado.remove_columns(cols_to_remove)

# salva novamente em Arrow
save_dir = os.path.join(base_dir, "dataset", "Final_dataset")

dataset_filtrado.save_to_disk(save_dir)


