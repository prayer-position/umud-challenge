import numpy as np

from umud.data.splits import _grouped_split


def test_grouped_split_keeps_clusters_together_and_skips_giant_ones():
    # One giant cluster (0) of 60 images and 40 clusters of one image each.
    indices = list(range(100))
    groups = np.array([0] * 60 + list(range(1, 41)))
    train, val = _grouped_split(indices, groups, {"val_frac": 0.15, "seed": 0})
    assert sorted(train + val) == indices
    assert len(val) == 15
    assert not set(range(60)) & set(val)  # the giant cluster stays in train


def test_grouped_split_never_splits_a_cluster():
    indices = list(range(90))
    groups = np.repeat(np.arange(30), 3)
    train, val = _grouped_split(indices, groups, {"val_frac": 0.2, "seed": 1})
    val_groups, train_groups = set(groups[val]), set(groups[train])
    assert not val_groups & train_groups
