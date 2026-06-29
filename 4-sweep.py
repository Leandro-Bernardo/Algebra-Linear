# FIX PARA ERRO DE CERTIFICADO CORROMPIDO NO WINDOWS
import ssl
try:
    import certifi
    ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: self.load_verify_locations(certifi.where())
except ImportError:
    # Caso o pacote certifi não esteja visível, ignora a carga para não travar o import do aiohttp
    ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: None

import torch
import wandb
import json
import os
import yaml

from pytorch_lightning import Trainer
from engine.lightning import Dataset, BaseModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Desativa logs redundantes de consoles em sub-threads
os.environ["WANDB_CONSOLE"] = "off"

torch.set_float32_matmul_precision('high')

BASE_DIR = os.getcwd()
CHECKPOINT_SAVE_PATH = os.path.join(BASE_DIR, "checkpoints")
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
ID = None


def train_iteration():
    """
    Função alvo executada automaticamente pelo wandb.agent.
    A cada chamada, o WandB injeta uma combinação nova de hiperparâmetros.
    """
    with wandb.init() as run:
        configs = run.config
        logger = WandbLogger(project="Linear-Algebra", experiment=run)
        checkpoint_callback = ModelCheckpoint(
            dirpath=CHECKPOINT_SAVE_PATH,
            filename=f"sweep_{run.name}_" + "{epoch}-{Loss/Val:.2f}",
            save_top_k=1,
            monitor='Loss/Val',
            mode='min',
            enable_version_counter=False,
            save_last=False,
            save_weights_only=True
        )

        data_module = Dataset(sweep_configs=dict(configs))
        data_module.prepare_data()
        load_path = os.path.join(os.getcwd(), "dataset", "processed_dataset")
        with open(os.path.join(load_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        num_classes = metadata["num_classes"]
        num_total_nos_globais = metadata["num_total_nos_globais"]
        embedding_dim = 49
        lambda_tax = configs.get("lambda_tax", 0.1)
        lr_atual = configs.get("learning_rate", 1e-3)
        model = BaseModel(
                        num_classes=num_classes,
                        num_total_nos_globais=num_total_nos_globais,
                        embedding_dim=embedding_dim,
                        learning_rate=lr_atual,
                        learning_rate_patience=10,
                        early_stopping_patience=EARLY_STOP_PATIENCE,
                        lambda_tax=lambda_tax
                        )

        trainer = Trainer(
                        logger=logger,
                        accelerator="gpu",
                        devices=1,
                        max_epochs=MAX_EPOCHS,
                        callbacks=[
                            checkpoint_callback,
                            LearningRateMonitor(logging_interval='epoch'),
                            EarlyStopping(
                                monitor="Loss/Val",
                                mode="min",
                                patience=EARLY_STOP_PATIENCE
                            )
                        ],
                        gradient_clip_val=configs.get("gradient_clip", 1.0),
                        gradient_clip_algorithm="value",
                        log_every_n_steps=1,
                        num_sanity_val_steps=2,
                        enable_progress_bar=True,
                        detect_anomaly=False
                        )

        trainer.fit(model=model, datamodule=data_module)


def main():
    yaml_path = os.path.join(BASE_DIR, "sweep_config.yml")
    with open(yaml_path, "r") as f:
        sweep_definition = yaml.load(f, Loader=yaml.FullLoader)
    id = sweep_definition["sweep_id"]
    sweep_id = wandb.sweep(sweep_definition, project="Linear-Algebra")
    wandb.agent(sweep_id, function=train_iteration, count=10) # count=10 significa que ele testará 10 combinações diferentes do yaml antes de parar


if __name__ == "__main__":
    main()