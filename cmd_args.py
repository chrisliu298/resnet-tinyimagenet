import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--project_name", type=str, default="resnet-tinyimagenet")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--output_size", type=int, default=200)
parser.add_argument("--keep_conv1", action="store_true")
parser.add_argument("--keep_maxpool", action="store_true")
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--weight_decay", type=int, default=5e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--model", type=str, default="all")
parser.add_argument("--verbose", type=int, default=1)


def parse_args():
    args = parser.parse_args()
    return args
