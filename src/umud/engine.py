from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.bce(logits, targets) + (1 - dice_score(logits, targets))


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler: torch.amp.GradScaler | None = None) -> float:
    """One training epoch; mixed precision when a GradScaler is given."""
    model.train()
    running_loss = 0.0
    # disable=None hides the bar when output isn't a terminal (e.g. Kaggle logs).
    for images, masks in tqdm(loader, desc="train", leave=False, disable=None):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
        loss = loss_fn(logits.float(), masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp: bool = False) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    for images, masks in tqdm(loader, desc="val", leave=False, disable=None):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images).float()
        loss = loss_fn(logits, masks)

        running_loss += loss.item() * images.size(0)
        running_dice += dice_score(logits, masks).item() * images.size(0)

    n = len(loader.dataset)
    return {"loss": running_loss / n, "dice": running_dice / n}


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device) -> None:
    model.load_state_dict(torch.load(path, map_location=device))
