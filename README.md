# resnet-tinyimagenet

I trained ResNet models using the [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) dataset. This implementation uses [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

```python
hparams = {
    "pretrained": True,
    "output_size": 200,
    "lr": 0.1,
    "max_epochs": 50,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "seed": 12345,
}
```

```bash
python3 train.py \
    --dataset_path="tiny-imagenet-200" \
    --pretrained \
    --model="resnet50" \
    --verbose=1
```


|  Model   | Test accuracy |
| :------: | :-----------: |
| resnet18 |    0.6847     |
| resnet34 |    0.7009     |
| resnet50 |    0.7214     |


## Acknowledgment

The dataset loading code in [`dataset.py`](dataset.py) is adapted from https://github.com/pranavphoenix/TinyImageNetLoader.