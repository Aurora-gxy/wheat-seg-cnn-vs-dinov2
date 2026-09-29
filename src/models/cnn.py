"""CNN baseline: DeepLabV3+ with a ResNet-50 encoder (ImageNet pre-trained).

Implemented with segmentation_models_pytorch. The whole network is fine-tuned.
DeepLabV3+ downsamples by 16, so inputs are padded from 350 to 352 internally.
"""
from typing import Dict

import torch.nn as nn

from .torch_base import TorchSegmenter


class DeepLabV3Plus(TorchSegmenter):
    name = "deeplabv3plus"
    pad_multiple = 16

    def __init__(self, encoder: str = "resnet50", pretrained: bool = True, **kwargs):
        self.encoder = encoder
        self.pretrained = pretrained
        super().__init__(**kwargs)

    def build_net(self) -> nn.Module:
        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus(
            encoder_name=self.encoder,
            encoder_weights="imagenet" if self.pretrained else None,
            classes=2,
        )

    def model_kwargs(self) -> Dict:
        # When re-loading a checkpoint the trained weights replace everything,
        # so there is no need to download ImageNet weights again.
        return {"encoder": self.encoder, "pretrained": False}
