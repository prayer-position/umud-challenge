from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from umud import utils
from umud.data import schema
from umud.data.dataset import MaskDataset
from umud.data.transforms import build_train_transform, build_val_transform
from umud.engine import DiceBCELoss, evaluate, save_checkpoint, train_one_epoch
from umud.models.unet import build_unet

POOL_DIRS = {
    "apo": (schema.APO_IMAGE_DIR, schema.APO_MASK_DIR),
    "fasc": (schema.FASC_IMAGE_DIR, schema.FASC_MASK_DIR),
}


def run_training(
    pool: str,
    config: dict,
    subset: int | None = None,
    epochs: int | None = None,
) -> Path:
    if pool not in POOL_DIRS:
        raise ValueError(f"Unknown pool {pool!r}, expected one of {list(POOL_DIRS)}")

    utils.set_seed(config["seed"])
    device = utils.get_device()
    image_size = tuple(config["image_size"])

    image_dir, mask_dir = POOL_DIRS[pool]
    train_tf = build_train_transform(image_size)
    val_tf = build_val_transform(image_size)

    n_total = len(MaskDataset(image_dir, mask_dir, image_size=image_size, transform=None))
    indices = list(range(n_total))
    if subset is not None:
        indices = indices[:subset]

    train_idx, val_idx = train_test_split(
        indices, test_size=config["val_frac"], random_state=config["seed"]
    )

    train_ds = Subset(
        MaskDataset(image_dir, mask_dir, image_size=image_size, transform=train_tf), train_idx
    )
    val_ds = Subset(
        MaskDataset(image_dir, mask_dir, image_size=image_size, transform=val_tf), val_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = build_unet(
        encoder_name=config["encoder"], num_classes=config["num_classes"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = DiceBCELoss()

    n_epochs = epochs if epochs is not None else config["epochs"]
    checkpoint_path = schema.PROJECT_ROOT / "outputs" / "checkpoints" / f"{pool}_best.pt"

    best_dice = -1.0
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(model, val_loader, loss_fn, device)
        print(
            f"[{pool}] epoch {epoch}/{n_epochs} "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"val_dice={metrics['dice']:.4f}"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            save_checkpoint(model, checkpoint_path)

    return checkpoint_path
