import os

import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from dataset import DataModule
from model import ResNet, resnet_configs

for model_name in resnet_configs.keys():
    config = EasyDict(
        model_name=model_name,
        pretrained=True,
        output_size=200,
        replace_conv1=True,
        replace_maxpool=True,
        lr=0.1,
        max_epochs=50,
        weight_decay=1e-4,
        batch_size=128,
        seed=12345,
        num_workers=int(os.cpu_count() / 2),
    )
    seed_everything(config.seed)
    wandb.init(project="resnet-tinyimagenet", entity="chrisliu298", config=config)

    datamodule = DataModule(
        batch_size=config.batch_size, num_workers=config.num_workers
    )
    model = ResNet(
        model_name=config.model_name,
        pretrained=config.pretrained,
        output_size=config.output_size,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        replace_conv1=config.replace_conv1,
        replace_maxpool=config.replace_maxpool,
    )

    trainer = Trainer(
        gpus=-1,
        max_epochs=config.max_epochs,
        logger=WandbLogger(
            save_dir="wandb/",
            project="tiny-imagenet",
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
        benchmark=True,
    )
    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(model, datamodule=datamodule)
    wandb.finish(quiet=True)
