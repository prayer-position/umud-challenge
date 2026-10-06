import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet34",
    num_classes: int = 1,
    encoder_weights: str = "imagenet",
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
