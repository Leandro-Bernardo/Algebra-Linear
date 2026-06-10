from datasets import load_from_disk
from datasets import Image
from PIL import Image as PILImage
from io import BytesIO
from tqdm import tqdm
import os

# Diretórios
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dir = os.path.join(base_dir, "dataset", "Raw_dataset")
save_dir = os.path.join(base_dir, "dataset", "Filtered_dataset")

taxonomia = [
    "Kingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "Species",
]

# Carrega dataset (split train)
dataset = load_from_disk(load_dir)["train"]
print(f"Total inicial: {len(dataset)} registros")

# Desabilita a decodificação automática das imagens (mantém bytes brutos)
dataset = dataset.cast_column("image", Image(decode=False))

# Remove registros com taxonomia incompleta
dataset = dataset.filter(lambda x: all(x[col] is not None for col in taxonomia))
print(f"Após filtro taxonômico: {len(dataset)} registros")

# Valida integridade real de cada imagem
valid_indices = []
for idx in tqdm(range(len(dataset)), desc="Validando imagens"):
    try:
        img_data = dataset[idx]["image"]
        if img_data is None:
            continue
        img_bytes = img_data.get("bytes")
        if not img_bytes:
            continue
        img = PILImage.open(BytesIO(img_bytes))
        img.verify()  # Levanta exceção se a imagem estiver corrompida
        valid_indices.append(idx)
    except Exception as e:
        print(f"Imagem inválida no índice {idx}: {e}")

# Mantém apenas registros com imagens válidas
dataset = dataset.select(valid_indices)
print(f"Após validação de imagens: {len(dataset)} registros")

# Reabilita decodificação automática antes de salvar
dataset = dataset.cast_column("image", Image(decode=True))

# Salva dataset filtrado no mesmo formato HuggingFace (Arrow)
print(f"Salvando dataset filtrado em: {save_dir}")
dataset.save_to_disk(save_dir)

print(f"Total de registros salvos: {len(dataset)}")