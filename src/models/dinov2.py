"""Foundation-model approach: frozen DINOv2 ViT backbone + a small trainable segmentation head.

The backbone (loaded through timm, weights from the Hugging Face Hub) is never updated.
Only the head is trained, so training is fast and needs little memory.

Input 350 x 350 is exactly 25 x 25 patches of 14 pixels, so no padding is needed.
The backbone gives a 25 x 25 feature map (1/14 of the image resolution); the head turns it
into a full-resolution 2-class prediction.

Two heads are compared:
  * "linear"  : BatchNorm + 1x1 conv on the patch features, then bilinear upsampling.
                Measures how good the frozen features are on their own.
  * "decoder" : light progressive-upsampling decoder (25 -> 50 -> 100 -> 200 -> 350) plus a
                tiny full-resolution branch on the RGB image, to recover thin leaf edges that
                the 1/14-resolution features cannot represent.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_base import TorchSegmenter

BACKBONES = {
    "vits14": "vit_small_patch14_dinov2.lvd142m",   # ~22 M parameters, 384-dim features
    "vitb14": "vit_base_patch14_dinov2.lvd142m",    # ~86 M parameters, 768-dim features
}


def conv_bn_relu(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class LinearHead(nn.Module):
    def __init__(self, c_in: int, n_classes: int = 2):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.cls = nn.Conv2d(c_in, n_classes, 1)

    def forward(self, feats: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        logits = self.cls(self.bn(feats))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class LightDecoder(nn.Module):
    def __init__(self, c_in: int, n_classes: int = 2):
        super().__init__()
        self.proj = conv_bn_relu(c_in, 256, k=1)
        self.up1 = conv_bn_relu(256, 128)
        self.up2 = conv_bn_relu(128, 64)
        self.up3 = conv_bn_relu(64, 32)
        self.detail = nn.Sequential(conv_bn_relu(3, 16), conv_bn_relu(16, 16))   # full-res RGB branch
        self.fuse = conv_bn_relu(32 + 16, 32)
        self.cls = nn.Conv2d(32, n_classes, 1)

    def forward(self, feats: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = self.proj(feats)
        for block in (self.up1, self.up2, self.up3):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([x, self.detail(image)], dim=1))
        return self.cls(x)


class DinoSegNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, n_layers: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.n_layers = n_layers
        for p in self.backbone.parameters():
            p.requires_grad = False

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the patch features of the last `n_layers` transformer blocks -> (B, C, H/14, W/14)."""
        with torch.no_grad():
            maps = self.backbone.forward_intermediates(
                x, indices=self.n_layers, norm=True, output_fmt="NCHW", intermediates_only=True)
        return torch.cat(maps, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x), x)


class DINOv2Segmenter(TorchSegmenter):
    pad_multiple = 14

    def __init__(self, backbone: str = "vits14", head: str = "linear", n_layers: int = 4,
                 pretrained: bool = True, **kwargs):
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {list(BACKBONES)}")
        if head not in ("linear", "decoder"):
            raise ValueError("head must be 'linear' or 'decoder'")
        self.backbone_name, self.head_name = backbone, head
        self.n_layers, self.pretrained = n_layers, pretrained
        self.name = f"dinov2_{head}"
        super().__init__(**kwargs)

    def build_net(self) -> nn.Module:
        import timm

        backbone = timm.create_model(BACKBONES[self.backbone_name], pretrained=self.pretrained,
                                     dynamic_img_size=True)
        c_in = backbone.embed_dim * self.n_layers
        head = LinearHead(c_in) if self.head_name == "linear" else LightDecoder(c_in)
        return DinoSegNet(backbone, head, self.n_layers)

    def set_train_mode(self) -> None:
        self.net.train()
        self.net.backbone.eval()          # frozen backbone always stays in inference mode

    def model_kwargs(self) -> Dict:
        return {"backbone": self.backbone_name, "head": self.head_name,
                "n_layers": self.n_layers, "pretrained": False}
