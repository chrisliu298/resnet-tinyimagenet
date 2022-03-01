import os

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from cmd_args import parse_args
from dataset import DataModule
from model import ResNet, resnet_configs


def main():
    args = parse_args()
    if args.model == "all":
        for model_name in resnet_configs.keys():
            args.model = model_name
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
    datamodule = DataModule(
        batch_size=args.batch_size, num_workers=int(os.cpu_count() / 2)
    )
    model = ResNet(
        model_name=args.model,
        pretrained=args.pretrained,
        output_size=args.output_size,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        keep_conv1=args.keep_conv1,
        keep_maxpool=args.keep_maxpool,
    )
    if args.verbose:
        summary(model, input_size=(1, 3, 64, 64))

    trainer = Trainer(
        gpus=-1,
        max_epochs=args.max_epochs,
        logger=WandbLogger(
            save_dir="wandb/",
            project="tiny-imagenet",
        )
        if args.use_wandb
        else None,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=0),
        ],
        benchmark=True,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    if args.use_wandb:
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
