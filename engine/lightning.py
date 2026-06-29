import torch, os, yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch import Generator
from torch.nn import ModuleDict
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from typing import Any, Dict, List, Tuple
#from .models import *
from math import ceil
import numpy as np
import matplotlib
matplotlib.use("Agg")  # renders plots only in memory
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import wandb
#import multiprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from .loss import HierarchicalTaxonomicLoss
import json
from pandas import DataFrame, read_parquet
import datasets
from PIL import Image, ImageOps
from .models import Model

def resize_with_padding(img, size=224):
    """
    Redimensiona uma imagem preservando sua proporção original e adiciona
    preenchimento (padding) para obter uma imagem quadrada de tamanho fixo.

    Esta abordagem evita distorções geométricas causadas por redimensionamentos
    diretos (ex.: 600x400 -> 224x224). O menor fator de escala é
    aplicado para que a imagem inteira caiba dentro do quadrado alvo e, em
    seguida, bordas são adicionadas de forma simétrica até atingir o tamanho
    final desejado.

    Args:
        img (PIL.Image.Image):
            Imagem de entrada.

        size (int, optional):
            Tamanho final da imagem quadrada em pixels.

    Returns:
        PIL.Image.Image:
            Imagem redimensionada e preenchida com bordas, possuindo dimensão
            final (size, size).
    """

    img.thumbnail((size, size))

    delta_w = size - img.size[0]
    delta_h = size - img.size[1]

    padding = (
                delta_w // 2,
                delta_h // 2,
                delta_w - delta_w // 2,
                delta_h - delta_h // 2
              )

    return ImageOps.expand(img, padding)

class _Preprocessing():
    def __init__(self, dataset: datasets.arrow_dataset.Dataset):
        self.dataset = dataset
        self.base_dir = os.getcwd()
        self.save_dir_base = os.path.join(self.base_dir, "dataset", "processed_dataset")
        os.makedirs(self.save_dir_base, exist_ok=True)

    def prepare_data(self):

            taxonomia = [
                        "Kingdom",
                        "Phylum",
                        "Class",
                        "Order",
                        "Family",
                        "Genus",
                    ] # "Species" não é considerada pois é a classe a ser prevista"

            ## PROCESSA A ÁRVORE TAXONOMICA
            # cria uma matriz para ser preenchida pela codificação da taxonomia
            tax_data = np.zeros((len(self.dataset), len(taxonomia)), dtype=np.int32)
            encoders = {}
            proximo_id_global = 1
            # codifica a arvore taxonomica, criando assim um ID global único para nó de cada nível
            for idx, col in enumerate(taxonomia):
                valores = self.dataset[col]
                unicos = sorted(set(valores))
                encoder = {}
                for i, v in enumerate(unicos):
                    encoder[v] = proximo_id_global + i
                encoders[col] = encoder
                tax_data[:, idx] = [encoder[v] for v in valores]
                proximo_id_global += len(unicos)
            # salva o mapper e o encoder da arvore taxonomica
            with open(os.path.join(self.save_dir_base, "encoders.json"), "w", encoding="utf-8") as f:
                json.dump(encoders, f, indent=4, ensure_ascii=False)
            DataFrame(tax_data, columns=taxonomia).to_parquet(os.path.join(self.save_dir_base, "tax_data.parquet"), index=False)

            ## PROCESSA AS IMAGENS (X) E AS LABELS (y)
            images = self.dataset["image"]
            labels = self.dataset["Species"]
            # cria o mapeamento em disco
            images_memmap = np.memmap(os.path.join(self.save_dir_base, "images.memmap"), dtype=np.uint8, mode="w+", shape=(len(images), 224, 224, 3))
            labels_memmap  = np.memmap(os.path.join(self.save_dir_base, "labels.memmap"), dtype=np.int32, mode="w+", shape=(len(labels),))
            # redimensiona as imagens para serem a entrada do extrator de características e salva no memmap
            IMG_SIZE = (224, 224)
            for i, img in enumerate(tqdm(images, desc="Salvando imagens")):
                # garante RGB
                img = img.convert("RGB")
                # redimensiona
                img = resize_with_padding(img, 224) #img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
                # converte para numpy uint8
                img_np = np.asarray(img, dtype=np.uint8)
                images_memmap[i] = img_np
            # finaliza de salvar as imagens no memmap (flush na memória)
            images_memmap.flush()

            # codifica as labels
            labels_unique = sorted(set(labels))
            labels_encoder = {v: i for i, v in enumerate(labels_unique)}
            labels = np.array([labels_encoder[v] for v in labels], dtype=np.int32)
            # finaliza de salvar as labels codificadas no memmap (flush na memória)
            labels_memmap[:] = labels[:]
            labels_memmap.flush()
            # salva o mapper
            with open(os.path.join(self.save_dir_base, "species_encoder.json"), "w", encoding="utf-8") as f:
                json.dump(labels_encoder, f, indent=4, ensure_ascii=False)

            num_total_nos_globais = proximo_id_global
            ## METADADOS
            metadata = {
                        "num_samples": len(images),
                        "num_classes": len(labels_unique),
                        "num_total_nos_globais": int(num_total_nos_globais),
                        "height": IMG_SIZE[0],
                        "width": IMG_SIZE[1],
                        "channels": 3,
                        "image_dtype": "uint8",
                        "label_dtype": "int32"
                        }

            with open(os.path.join(self.save_dir_base, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

class PlanktonDataset(torch.utils.data.Dataset):
    """
    Classe para gerenciar o dataset baseado em memmap.

    As imagens permanecem em disco e são carregadas sob demanda.
    """

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.N = metadata["num_samples"]
        self.H = metadata["height"]
        self.W = metadata["width"]
        self.C = metadata["channels"]
        self.images = np.memmap(os.path.join(data_dir, "images.memmap"), dtype=np.uint8, mode="r", shape=(self.N, self.H, self.W, self.C))
        self.labels = np.memmap(os.path.join(data_dir, "labels.memmap"), dtype=np.int32, mode="r", shape=(self.N,))
        self.tax_data = read_parquet(os.path.join(data_dir, "tax_data.parquet")).values.astype(np.int64)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        image = torch.from_numpy(self.images[idx].copy()).float()
        image = image.permute(2, 0, 1)
        image /= 255.0
        image = (image - self.mean) / self.std
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        tax = torch.tensor(self.tax_data[idx], dtype=torch.long)

        return image, tax, label

class Dataset(LightningDataModule):
    def __init__(self, sweep_configs, **kwargs ):
        super().__init__()
        self.sweep_configs = sweep_configs

    def _old_prepare_data(self):
        #base_dir = os.getcwd()
        #try:
            # Carrega o dataset preprocessado
            #load_path =  os.path.join(base_dir, "dataset", "processed_dataset") #torch.load(self.saved_samples_path) # TODO carregar untyped storage data aqui
            # with open(os.path.join(load_path, "metadata.json"), "r") as f:
            #     metadata = json.load(f)
            # self.num_classes, N, C, H, W = metadata["num_classes"], metadata["num_samples"], metadata["channels"], metadata["height"], metadata["width"]
            # images_memmap = np.memmap(os.path.join(load_path, "images.memmap"), dtype=np.uint8, mode="r+", shape=(N, H, W, C))
            # labels_memmap  = np.memmap(os.path.join(load_path, "labels.memmap"), dtype=np.int32, mode="r+", shape=(N,))
            # tax_data = read_parquet(os.path.join(load_path, "tax_data.parquet"))
           # self.dataset = PlanktonDataset(load_path)

        #except:
            # Carrega o dataset Raw para ser processado
            #load_raw_dataset_dir = os.path.join(base_dir, "dataset", "Final_dataset")
            #dataset = datasets.load_from_disk(load_raw_dataset_dir)
            # faz o preprocessamento das amostras
            #_Preprocessing(dataset).prepare_data()
            # load_path =  os.path.join(base_dir, "dataset", "processed_dataset")
            # self.dataset = PlanktonDataset(load_path)

            # with open(os.path.join(load_path, "metadata.json"), "r") as f:
            #     metadata = json.load(f)
            # self.num_classes, N, C, H, W = metadata["num_classes"], metadata["num_samples"], metadata["channels"], metadata["height"], metadata["width"]

            # # images_memmap = np.memmap(os.path.join(load_path, "images.memmap"), dtype=np.uint8, mode="r+", shape=(N, H, W, C))
            # # labels_memmap  = np.memmap(os.path.join(load_path, "labels.memmap"), dtype=np.int32, mode="r+", shape=(N,))
            # # tax_data = read_parquet(os.path.join(load_path, "tax_data.parquet"))

        # images_memmap = torch.from_numpy(images_memmap)
        # # normaliza e padroniza os valores baseados no treinamento da VGG
        # images_memmap = images_memmap.float() / 255.0
        # images_memmap = images_memmap.permute(0,3,1,2)
        # mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1) # média do dataset usado no treinamento da vgg
        # std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)  # desvio padrão usado no treinamento da vgg
        # images_memmap = (images_memmap - mean) / std

        # labels_memmap = torch.tensor(labels_memmap, dtype=torch.long)
        # tax_data = torch.tensor(tax_data.values, dtype=torch.long)
        # self.dataset = TensorDataset(images_memmap, labels_memmap, tax_data)
        pass
    def prepare_data(self):
        base_dir = os.getcwd()
        load_path = os.path.join(base_dir, "dataset", "processed_dataset")
        if not os.path.exists(os.path.join(load_path, "metadata.json")):
            raw_path = os.path.join(base_dir, "dataset", "Final_dataset")
            dataset = datasets.load_from_disk(raw_path)
            _Preprocessing(dataset).prepare_data()
        with open(os.path.join(load_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        self.num_classes = metadata["num_classes"]

    def setup(self, stage:str):
        load_path =  os.path.join(os.getcwd(), "dataset", "processed_dataset")
        self.dataset = PlanktonDataset(load_path)
        len_dataset = len(self.dataset)
        # ~60% ~20% ~20%
        n_train = ceil(0.6*len_dataset)
        n_val = ceil(0.2*len_dataset)
        n_test = len_dataset - n_train - n_val

        train_set, val_set, test_set = random_split(self.dataset, [n_train, n_val, n_test], generator = Generator().manual_seed(42))

        self.dataset_train = train_set
        self.dataset_val = val_set
        self.dataset_test = test_set

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = self.sweep_configs["batch_size"])#, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size = self.sweep_configs["batch_size"])#, shuffle=False, num_workers= 2, pin_memory=True, drop_last=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1,  shuffle=False)#, num_workers= 2, pin_memory=True, drop_last=False, persistent_workers=True)

class BaseModel(LightningModule):
    """
    Modelo genérico para gerenciar o treinamento
    Args:
        num_classes: quantidade de espécies diferentes à serem aprendidas pelo classificador.
        num_total_nos_globais: quantidade de nós na árvore taxonomica. Cada nó será representado por um embedding diferente na lookup table.
        embedding_dim: dimensão de cada embedding aprendido na

    """
    def __init__(self, *, num_classes: int, num_total_nos_globais: int, embedding_dim: int = 49, learning_rate: float = 1e-3, learning_rate_patience: int = 5, early_stopping_patience: int = 10, lambda_tax: float = 0.1, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.net = Model(num_classes_especie=num_classes, num_total_nos_globais=num_total_nos_globais, embedding_dim=embedding_dim, use_taxonomic_embedding=False)
        self.criterion = HierarchicalTaxonomicLoss(lambda_tax=lambda_tax, k=6, use_taxonomic_loss=False)
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.early_stopping_patience = early_stopping_patience

        self.metrics = ModuleDict({
            mode_name: MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
                "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
                "F1-score": F1Score(task="multiclass", num_classes=num_classes, average="macro")
            }) for mode_name in ["Train", "Val", "Test"]
        })
        self._inference_time = {"predictions": [], "targets": []}

    def configure_optimizers(self):
        # Passa self.net.parameters() explicitamente para garantir o rastreio dos gradientes
        optimizer = Adam(self.net.parameters(), lr=self.learning_rate)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', patience=self.learning_rate_patience)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": reduce_lr_on_plateau, "monitor": "Loss/Val"}
                }

    def forward(self, images: torch.Tensor, ids_taxonomicos: torch.Tensor):
        return self.net(images, ids_taxonomicos)

    def _any_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str):
        images, ids_taxonomicos, y_classe = batch
        # Forward retorna as saídas e os sigmas para a Loss Composta
        logits, _, sigmas = self(images, ids_taxonomicos)
        # Computa a perda composta
        total_loss, ce_loss, tax_loss = self.criterion(logits, y_classe, sigmas, ids_taxonomicos)
        # Logs das perdas individuais e globais
        self.log(f"Loss/{stage}", total_loss, prog_bar=True, on_epoch=True)
        self.log(f"Loss_CE/{stage}", ce_loss, on_epoch=True)
        self.log(f"Loss_Tax/{stage}", tax_loss, on_epoch=True)

        # Métricas
        metrics: MetricCollection = self.metrics[stage]
        output_metrics = metrics(logits, y_classe)

        self.log_dict({f'{metric_name}/{stage}/Step': value for metric_name, value in output_metrics.items()})
        return total_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        return self._any_step(batch, "Val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        images, ids_taxonomicos, y_classe = batch
        logits, _, _ = self(images, ids_taxonomicos=None) #logits, _, _ = self(images, ids_taxonomicos)
        preds = torch.argmax(logits, dim=1)

        metrics: MetricCollection = self.metrics["Test"]
        metrics(logits, y_classe)

        # Captura os dados para a matriz de confusão final
        for p, t in zip(preds.detach().cpu().numpy(), y_classe.detach().cpu().numpy()):
            self._inference_time["predictions"].append(p)
            self._inference_time["targets"].append(t)

    def _any_epoch_end(self, stage: str):
        metrics: MetricCollection = self.metrics[stage]
        self.log_dict({f'{metric_name}/{stage}/Epoch': value for metric_name, value in metrics.compute().items()}, on_step=False, on_epoch=True)
        metrics.reset()

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")

    def on_train_end(self):
        self.eval()
        self.trainer.test(model=self, datamodule=self.trainer.datamodule, ckpt_path="best")

        preds = np.array(self._inference_time["predictions"])
        targets = np.array(self._inference_time["targets"])

        cm = confusion_matrix(targets, preds)

        # CORREÇÃO 1: Aumentamos o figsize para 20x18. Isso dá espaço real para as 48 classes respirarem.
        fig, ax = plt.subplots(figsize=(20, 18))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # CORREÇÃO 2:
        # - values_format="d" garante que números inteiros não virem notação científica (ex: 1e+02).
        # - text_kw={"fontsize": 6} reduz drasticamente o tamanho do texto interno de cada quadrado.
        # NOTA: Se mesmo com tamanho 6 ficar poluído, você pode passar include_values=False para esconder os números e avaliar puramente pela cor.
        disp.plot(
            ax=ax,
            colorbar=True,
            cmap="Blues",
            values_format="d",
            text_kw={"fontsize": 6},
            include_values=True
        )

        # CORREÇÃO 3: Ajusta os eixos para que as 48 labels fiquem legíveis e sem sobreposição
        ax.set_title("Confusion Matrix — Best Model (Test Set)", fontsize=16, pad=20)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.xticks(rotation=45) # Rotaciona os números do eixo X para não colidirem

        plt.tight_layout()

        # Envia para o Weights & Biases com alta qualidade
        if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            self.logger.experiment.log({"confusion_matrix/Test/BestModel": wandb.Image(fig)})

        # CORREÇÃO 4: Aumentamos o DPI para 300 para salvar o arquivo local em altíssima resolução (HD)
        plt.savefig("confusion_matrix_best_model.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        self._inference_time = {"predictions": [], "targets": []}