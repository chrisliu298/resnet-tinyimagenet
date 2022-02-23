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
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy

seed_everything(12345)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)


batch_size = 128

id_dict = {}
for i, line in enumerate(open("/content/tiny-imagenet-200/wnids.txt", "r")):
    id_dict[line.replace("\n", "")] = i


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
        for i, line in enumerate(
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
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
)
# val_dataloader = DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
# )

test_dataset = TestTinyImageNetDataset(id=id_dict, transform=test_transforms)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
)


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=200)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

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
        steps_per_epoch = ceil(100000 / BATCH_SIZE)
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


model = LitResnet(lr=0.05)

trainer = Trainer(
    gpus=-1,
    progress_bar_refresh_rate=10,
    max_epochs=30,
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
    benchmark=True,
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
trainer.test(model, dataloaders=test_dataloader)
