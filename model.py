import copy
from multiprocessing.sharedctypes import Value
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
    wide_resnet50_2,
    wide_resnet101_2,
)
from torchvision.models import efficientnet_b7

resnet_configs = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "wide_resnet50": wide_resnet50_2,
    "wide_resnet101": wide_resnet101_2,
}


class BaseModel(LightningModule):
    def __init__(self, output_size, lr, batch_size, weight_decay, optimizer, momentum):
        super().__init__()

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
        optimizer_fns = {
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        if self.hparams.optimizer in ["rmsprop", "sgd"]:
            optimizer = optimizer_fns[self.hparams.optimizer](
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError("Optimizer {self.hparams.optimizer} is invalid.")
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=50000 // self.hparams.batch_size + 1,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class ResNet(BaseModel):
    def __init__(
        self,
        model_name,
        pretrained,
        keep_conv1,
        keep_maxpool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        assert self.hparams.model_name in resnet_configs.keys()
        self.model = resnet_configs[self.hparams.model_name](
            pretrained=self.hparams.pretrained
        )
        if not self.hparams.keep_conv1:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
        if not self.hparams.keep_maxpool:
            self.model.maxpool = nn.Identity()
        self.model.fc = (
            nn.Linear(512, self.hparams.output_size)
            if self.hparams.model_name in ["resnet18", "resnet34"]
            else nn.Linear(2048, self.hparams.output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class EfficientNet(BaseModel):
    def __init__(
        self,
        model_name,
        pretrained,
        keep_conv1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = efficientnet_b7(pretrained=self.hparams.pretrained)
        if not self.hparams.keep_conv1:
            self.model.features[0] = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
        self.model.classifier[1] = nn.Linear(2560, self.hparams.output_size)

    def forward(self, x):
        out = self.model(x)
        return out


# class CNN(BaseModel):
#     def __init__(self, n_units, dropout, **kwargs):
#         super().__init__(**kwargs)
#         self.save_hyperparameters()
#         self._n_units = copy.copy(n_units)
#         self._layers = []
#         # [3, 16, 16, 16, 10]
#         for i in range(1, len(n_units) - 1):
#             layer = nn.Conv2d(n_units[i - 1], n_units[i], 3)
#             self._layers.append(layer)
#             name = f"conv{i}"
#             self.add_module(name, layer)
#             # layer = nn.MaxPool2d(2)
#             # self._layers.append(layer)
#             # name = f"maxpool{i}"
#             # self.add_module(name, layer)
#             if dropout > 0.0:
#                 layer = nn.Dropout(dropout)
#                 self._layers.append(layer)
#                 name = f"dropout{i}"
#                 self.add_module(name, layer)

#         # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(n_units[-2] * 4, n_units[-1])
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2)

#     def forward(self, x):
#         # x = self.maxpool(self.relu(self._layers[0](x)))
#         x = self.relu(self.maxpool(self._layers[0](x)))
#         # x = self.relu(self._layers[0](x))
#         for layer in self._layers[1:]:
#             # x = self.maxpool(self.relu(layer(x)))
#             x = self.relu(self.maxpool(layer(x)))
#             # x = self.relu(layer(x))
#         out = self.fc(self.flatten(x))
#         return out
