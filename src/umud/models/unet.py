import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet34",
    num_classes: int = 1,
    encoder_weights: str | None = "imagenet",
    arch: str = "Unet",
) -> nn.Module:
    return smp.create_model(
        arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def build_model_from_config(config: dict, encoder_weights: str | None = "imagenet") -> nn.Module:
    return build_unet(
        encoder_name=config["encoder"],
        num_classes=config["num_classes"],
        encoder_weights=encoder_weights,
        arch=config.get("arch", "Unet"),
    )
