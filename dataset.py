from glob import glob

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import normal
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob("/content/tiny-imagenet-200/train/*/*/*.JPEG")
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


class TestDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob("/content/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.class_dict = {}
        for _, line in enumerate(
            open("/content/tiny-imagenet-200/val/val_annotations.txt", "r")
        ):
            a = line.split("\t")
            img, cls_id = a[0], a[1]
            self.class_dict[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = self.filenames[idx]
        image = Image.open(image_path)
        label = self.class_dict[image_path.split("/")[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label


class DataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.id_dict = {}
        for i, line in enumerate(open("/content/tiny-imagenet-200/wnids.txt", "r")):
            self.id_dict[line.replace("\n", "")] = i

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [122.4602, 114.2571, 101.3639]],
            std=[x / 255.0 for x in [70.4915, 68.5601, 71.8054]],
        )
        self.train_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = TrainDataset(
            id=self.id_dict, transform=self.train_transform
        )
        self.test_dataset = TestDataset(id=self.id_dict, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
