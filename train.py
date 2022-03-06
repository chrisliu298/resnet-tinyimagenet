import os

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from cmd_args import parse_args
from dataset import DataModule
from model import ResNet, resnet_configs, CNN


def main():
    args = parse_args()
    if args.model_name == "all":
        for model_name in resnet_configs.keys():
            args.model_name = model_name
            setup(args)
            train(args)
    else:
        setup(args)
        train(args)


def setup(args):
    seed_everything(args.seed)
    if args.use_wandb:
        wandb.init(project=args.project_name, entity="chrisliu298", config=vars(args))


def train(args):
    refresh_rate = 10 if args.verbose else 0

    datamodule = DataModule(
        path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=int(os.cpu_count() / 2),
    )
    if "cnn" in args.model_name:
        model_spec = args.model_name.split("-")[-1]
        n_units = [3] + [int(x) for x in model_spec.split("x")] + [args.output_size]
        model = CNN(
            n_units=n_units,
            dropout=0.0,
            output_size=args.output_size,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
    else:
        model = ResNet(
            model_name=args.model_name,
            pretrained=args.pretrained,
            output_size=args.output_size,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            keep_conv1=args.keep_conv1,
            keep_maxpool=args.keep_maxpool,
        )
    # if args.verbose:
    summary(model, input_size=(1, 3, 64, 64))

    trainer = Trainer(
        gpus=-1,
        max_epochs=args.max_epochs,
        logger=WandbLogger(
            save_dir="wandb/",
            project="tiny-imagenet",
        )
        if args.use_wandb
        else True,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=refresh_rate),
            ModelCheckpoint(
                dirpath=args.model_path,
                filename="{epoch}_{avg_val_acc}",
                monitor="avg_val_acc",
                save_top_k=1,
                mode="max",
                every_n_epochs=1,
            ),
        ],
        benchmark=True,
        enable_model_summary=False,
        # auto_scale_batch_size="power",
    )
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    if args.use_wandb:
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
