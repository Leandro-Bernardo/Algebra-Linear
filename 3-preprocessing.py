from datasets import load_from_disk
import pandas as pd
import os

base_dir = os.getcwd()
load_dir = os.path.join(base_dir, "dataset", "Filtered_dataset")

dataset = load_from_disk(load_dir)

# dataframe apenas para cálculos
df = dataset.to_pandas()

print(f"\nColunas do dataset:\n{df.columns.tolist()}")

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

cols_to_remove = [c for c in cols_to_remove if c in dataset.column_names]

print(f"\nQuantidade de amostras por classe (antes da filtragem):\n"f"{df['original_label'].value_counts()}")

threshold = 1000
upper_threshold = 5000

counts = df["original_label"].value_counts()

mask = df["original_label"].map(counts).between(threshold, upper_threshold)

df_filtrado = df[mask].copy()

print(f"\nQuantidade de amostras por classe após threshold:\n" f"{df_filtrado['original_label'].value_counts()}")

species_por_genus = (df_filtrado.groupby("Genus")["Species"].nunique().sort_values(ascending=False))

print(f"\nQuantidade de espécies distintas por gênero:\n {species_por_genus}")

genus_multispecies = species_por_genus[species_por_genus > 1].index

print(f"\nGêneros com mais de uma espécie: {len(genus_multispecies)}")

# mantém apenas linhas desses gêneros
df_filtrado = df_filtrado[df_filtrado["Genus"].isin(genus_multispecies)]

print(f"\nNúmero de amostras após filtro de gênero: {len(df_filtrado)}")

# relação gênero -> espécies
print(f"\nGêneros e suas espécies:")

for genus, species in (df_filtrado.groupby("Genus")["Species"].unique().items()):
    print(f"{genus}: {list(species)}")

print(f"\nContagem final por gênero:\n"f"{df_filtrado['Genus'].value_counts()}")

print(f"\nContagem final por espécie:\n" f"{df_filtrado['Species'].value_counts()}")

indices_validos = df_filtrado.index.tolist()

dataset_filtrado = dataset.select(indices_validos)

# remove colunas desnecessárias
dataset_filtrado = dataset_filtrado.remove_columns(cols_to_remove)

save_dir = os.path.join(base_dir, "dataset", "Final_dataset")

dataset_filtrado.save_to_disk(save_dir)

print(f"Tamanho final: {len(dataset_filtrado)} amostras")