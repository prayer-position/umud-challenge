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


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        logits = model(images)
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
