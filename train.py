# -*- coding: utf-8 -*-

# !nvidia-smi
# !pip3 install -q pytorch-lightning lightning-bolts torchinfo
# !rm *.zip*
# !rm -rf tiny-imagenet-200/
# !wget -nv http://cs231n.stanford.edu/tiny-imagenet-200.zip
# !unzip -q tiny-imagenet-200.zip
# !rm -rf lightning_logs/

import glob
import os
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy

seed_everything(12345)
batch_size = 128
num_workers = int(os.cpu_count() / 2)


id_dict = {}
for i, line in enumerate(open("/content/tiny-imagenet-200/wnids.txt", "r")):
    id_dict[line.replace("\n", "")] = i


class WideResNet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(2048, 200)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        pred = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, y)
        acc = accuracy(pred, y)
        self.log("train_loss", loss, logger=True)
        self.log("train_acc", acc, logger=True)
        return {"loss": loss, "train_acc": acc}

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

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = ceil(100000 / batch_size)
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


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/content/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = self.filenames[idx]
        image = Image.open(image_path)
        label = self.id_dict[image_path.split("/")[4]]
        if self.transform:
            image = self.transform(image)
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/content/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for _, line in enumerate(
            open("/content/tiny-imagenet-200/val/val_annotations.txt", "r")
        ):
            a = line.split("\t")
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = self.filenames[idx]
        image = Image.open(image_path)
        label = self.cls_dic[image_path.split("/")[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [122.4602, 114.2571, 101.3639]],
            std=[x / 255.0 for x in [70.4915, 68.5601, 71.8054]],
        ),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [122.4602, 114.2571, 101.3639]],
            std=[x / 255.0 for x in [70.4915, 68.5601, 71.8054]],
        ),
    ]
)
train_dataset = TrainTinyImageNetDataset(id=id_dict, transform=train_transforms)
test_dataset = TestTinyImageNetDataset(id=id_dict, transform=test_transforms)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

model = WideResNet(lr=0.1)
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

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
result = trainer.test(model, dataloaders=test_dataloader)

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/
