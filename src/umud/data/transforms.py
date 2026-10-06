import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: tuple[int, int], enable_flip: bool = False) -> A.Compose:
    h, w = image_size
    transforms = [A.Resize(h, w)]
    if enable_flip:
        # Off by default: confirm in notebooks/01_eda.ipynb that left/right
        # orientation doesn't carry meaning (e.g. consistent probe direction)
        # before enabling.
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms += [
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def build_val_transform(image_size: tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
