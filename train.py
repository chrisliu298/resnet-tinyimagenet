import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import DataModule
from model import ResNet

seed_everything(12345)
batch_size = 128
num_workers = int(os.cpu_count() / 2)

datamodule = DataModule(batch_size=batch_size, num_workers=num_workers)
model = ResNet(lr=0.1, batch_size=batch_size)

trainer = Trainer(
    gpus=-1,
    max_epochs=50,
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ],
    benchmark=True,
)

trainer.fit(model, datamodule=datamodule)
result = trainer.test(model, datamodule=datamodule)
