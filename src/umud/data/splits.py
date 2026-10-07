"""Train/validation splits.

The training pools contain near-duplicate frames (the organizers confirm
duplicates; 75% of fascicle validation images had a training image with
thumbnail correlation > 0.95 under a random split). `grouped` splits keep
each cluster of near-duplicates on one side.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from umud.data import schema
from umud.data.dataset import load_image

DUPLICATE_CORRELATION = 0.95
_THUMB_SIZE = (64, 40)  # (w, h)


def _thumbnail(path: Path) -> np.ndarray:
    gray = cv2.cvtColor(load_image(path), cv2.COLOR_RGB2GRAY)
    thumb = cv2.resize(gray, _THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (thumb - thumb.mean()) / (thumb.std() + 1e-6)


def near_duplicate_groups(paths: list[Path], threshold: float = DUPLICATE_CORRELATION) -> np.ndarray:
    """Cluster id per image: images linked by thumbnail correlation above
    `threshold` (transitively) share a cluster."""
    with ThreadPoolExecutor() as pool:
        thumbs = np.stack(list(pool.map(_thumbnail, paths)))
    corr = thumbs @ thumbs.T / thumbs.shape[1]
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(corr > threshold, k=1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(paths))])


def _cached_groups(image_dir: Path, paths: list[Path]) -> np.ndarray:
    cache = schema.PROJECT_ROOT / "outputs" / "cache" / f"groups_{image_dir.name}.json"
    names = [p.name for p in paths]
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["names"] == names:
            return np.array(data["groups"])
    groups = near_duplicate_groups(paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"names": names, "groups": groups.tolist()}))
    return groups


def _grouped_split(indices: list[int], groups: np.ndarray, config: dict) -> tuple[list[int], list[int]]:
    """Whole clusters go to validation, in seeded random order, until it
    holds val_frac of the images. Clusters larger than half that target stay
    in training: chained near-duplicates can form one huge cluster (657 of
    2761 fascicle images), which would otherwise dominate validation."""
    target = config["val_frac"] * len(indices)
    members: dict[int, list[int]] = {}
    for idx, group in zip(indices, groups):
        members.setdefault(int(group), []).append(idx)
    order = sorted(members)
    np.random.default_rng(config["seed"]).shuffle(order)
    val: list[int] = []
    for group in order:
        if len(val) >= target:
            break
        if len(members[group]) <= target / 2:
            val.extend(members[group])
    val_set = set(val)
    return [i for i in indices if i not in val_set], sorted(val)


def split_indices(image_dir: Path, config: dict, subset: int | None = None) -> tuple[list[int], list[int]]:
    """Train/val indices into `schema.list_image_files(image_dir)`.

    config["split"]: "random" (default, per image) or "grouped" (clusters of
    near-duplicate frames stay together)."""
    paths = schema.list_image_files(Path(image_dir))
    indices = list(range(len(paths)))
    if subset is not None:
        indices = indices[:subset]
    if config.get("split", "random") == "grouped":
        return _grouped_split(indices, _cached_groups(Path(image_dir), paths)[indices], config)
    train_idx, val_idx = train_test_split(indices, test_size=config["val_frac"], random_state=config["seed"])
    return train_idx, val_idx
