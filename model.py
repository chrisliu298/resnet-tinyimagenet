import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)

resnet_configs = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50": resnext50_32x4d,
    "resnext101": resnext101_32x8d,
    "wide_resnet50": wide_resnet50_2,
    "wide_resnet101": wide_resnet101_2,
}


class ResNet(LightningModule):
    def __init__(
        self,
        model_name="resnet18",
        pretrained=True,
        output_size=200,
        lr=0.1,
        batch_size=128,
        weight_decay=1e-4,
        replace_conv1=True,
        replace_maxpool=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.model_name in resnet_configs.keys()
        self.model = resnet_configs[self.hparams.model_name](
            pretrained=self.hparams.pretrained
        )
        if self.hparams.replace_conv1:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
        if self.hparams.replace_maxpool:
            self.model.maxpool = nn.Identity()
        self.model.fc = (
            nn.Linear(512, self.hparams.output_size)
            if self.hparams.model_name in ["resnet18", "resnet34"]
            else nn.Linear(2048, self.hparams.output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self(x)
        pred = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, y)
        acc = accuracy(pred, y)
        if stage:
            self.log(f"{stage}_loss", loss, logger=True, prog_bar=True)
            self.log(f"{stage}_acc", acc, logger=True, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "train")
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        self.log("avg_train_loss", avg_loss, logger=True)
        self.log("avg_train_acc", avg_acc, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        self.log("avg_val_loss", avg_loss, logger=True)
        self.log("avg_val_acc", avg_acc, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        self.log("avg_test_loss", avg_loss, logger=True)
        self.log("avg_test_acc", avg_acc, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        steps_per_epoch = 100000 // self.hparams.batch_size + 1
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
