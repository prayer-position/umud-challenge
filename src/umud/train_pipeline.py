import json
import math
import time
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from umud import utils
from umud.data import schema
from umud.data.dataset import MaskDataset
from umud.data.transforms import build_train_transform, build_val_transform
from umud.engine import DiceBCELoss, evaluate, save_checkpoint, train_one_epoch
from umud.evaluation import FascicleAngleEvaluator
from umud.models.unet import build_model_from_config

POOL_DIRS = {
    "apo": (schema.APO_IMAGE_DIR, schema.APO_MASK_DIR),
    "fasc": (schema.FASC_IMAGE_DIR, schema.FASC_MASK_DIR),
}

# Optional config keys and their defaults (the defaults reproduce the
# original pipeline):
#   arch: Unet                  any segmentation_models_pytorch architecture
#   line_px: 0                  >0: line-preserving mask resize, dilated to ~line_px wide
#   cache: false                decode + resize every image once, keep in memory
#   amp: false                  mixed-precision training (CUDA only)
#   flip: false                 horizontal-flip augmentation
#   scheduler: none             "cosine": cosine LR decay with linear warmup
#   warmup_epochs: 1
#   early_stopping_patience: null
#   select_metric: dice         "angle_mae": pick the checkpoint by fascicle angle error


def split_indices(n_total: int, config: dict, subset: int | None = None) -> tuple[list[int], list[int]]:
    indices = list(range(n_total))
    if subset is not None:
        indices = indices[:subset]
    train_idx, val_idx = train_test_split(indices, test_size=config["val_frac"], random_state=config["seed"])
    return train_idx, val_idx


def _cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_training(
    pool: str,
    config: dict,
    subset: int | None = None,
    epochs: int | None = None,
    run_name: str | None = None,
) -> Path:
    if pool not in POOL_DIRS:
        raise ValueError(f"Unknown pool {pool!r}, expected one of {list(POOL_DIRS)}")

    utils.set_seed(config["seed"])
    device = utils.get_device()
    image_size = tuple(config["image_size"])
    line_px = config.get("line_px", 0)
    cache = config.get("cache", False)
    amp = config.get("amp", False) and device.type == "cuda"
    select_metric = config.get("select_metric", "dice")
    run_name = run_name or pool

    image_dir, mask_dir = POOL_DIRS[pool]
    train_tf = build_train_transform(image_size, enable_flip=config.get("flip", False))
    val_tf = build_val_transform(image_size)

    train_idx, val_idx = split_indices(len(schema.list_image_files(image_dir)), config, subset)
    t0 = time.time()
    train_ds = MaskDataset(image_dir, mask_dir, image_size, train_tf, line_px=line_px, indices=train_idx, cache=cache)
    val_ds = MaskDataset(image_dir, mask_dir, image_size, val_tf, line_px=line_px, indices=val_idx, cache=cache)
    angle_evaluator = FascicleAngleEvaluator(val_ds) if select_metric == "angle_mae" else None
    print(f"[{run_name}] data ready in {time.time() - t0:.0f}s ({len(train_ds)} train / {len(val_ds)} val)")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = DiceBCELoss()
    scaler = torch.amp.GradScaler("cuda") if amp else None

    n_epochs = epochs if epochs is not None else config["epochs"]
    scheduler = None
    if config.get("scheduler", "none") == "cosine":
        scheduler = _cosine_with_warmup(optimizer, config.get("warmup_epochs", 1), n_epochs)
    patience = config.get("early_stopping_patience")

    checkpoint_dir = schema.PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"

    best_score = -math.inf
    best_epoch = 0
    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        metrics = evaluate(model, val_loader, loss_fn, device, amp=amp)
        if angle_evaluator is not None:
            metrics.update(angle_evaluator(model, device, batch_size=config["batch_size"]))
        if scheduler is not None:
            scheduler.step()

        # Higher is better for dice, lower for angle error.
        score = -metrics["angle_mae"] if select_metric == "angle_mae" else metrics["dice"]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            save_checkpoint(model, checkpoint_path)

        history.append({"epoch": epoch, "train_loss": train_loss, **metrics, "seconds": time.time() - t0})
        extra = ""
        if angle_evaluator is not None:
            extra = f" angle_mae={metrics['angle_mae']:.2f} miss={metrics['miss_rate']:.3f}"
        print(
            f"[{run_name}] epoch {epoch}/{n_epochs} "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"val_dice={metrics['dice']:.4f}{extra} ({time.time() - t0:.0f}s){' *' if improved else ''}",
            flush=True,
        )
        (checkpoint_dir / f"{run_name}_history.json").write_text(json.dumps(history, indent=1))

        if patience is not None and epoch - best_epoch >= patience:
            print(f"[{run_name}] early stop: no improvement since epoch {best_epoch}")
            break

    return checkpoint_path
