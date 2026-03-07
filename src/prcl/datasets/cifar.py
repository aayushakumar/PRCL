"""CIFAR-10/100 dataset wrappers with poison-aware support."""

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


class PoisonAwareDataset(Dataset):
    """Wraps a torchvision dataset for SSL pretraining.

    During pretraining: returns (transformed_views, sample_index) — NO labels.
    Tracks poison indices as metadata for controlled evaluation only.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transform=None,
        poison_indices: np.ndarray | None = None,
        poison_fn=None,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.poison_indices = set(poison_indices.tolist()) if poison_indices is not None else set()
        self.poison_fn = poison_fn
        self._poison_mask = np.zeros(len(base_dataset), dtype=bool)
        if poison_indices is not None:
            self._poison_mask[poison_indices] = True

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _label = self.base_dataset[idx]  # label ignored during pretraining

        # Apply poison trigger if this sample is poisoned
        if idx in self.poison_indices and self.poison_fn is not None:
            img = self.poison_fn(img)

        # Apply SSL transform (e.g., TwoViewTransform)
        if self.transform is not None:
            img = self.transform(img)

        return img, idx

    @property
    def poison_mask(self) -> np.ndarray:
        """Boolean mask of poisoned samples (for evaluation only)."""
        return self._poison_mask

    @property
    def num_poisoned(self) -> int:
        return int(self._poison_mask.sum())

    @property
    def labels(self) -> np.ndarray:
        """Access labels for downstream eval only — never during pretraining."""
        if hasattr(self.base_dataset, "targets"):
            return np.array(self.base_dataset.targets)
        elif hasattr(self.base_dataset, "labels"):
            return np.array(self.base_dataset.labels)
        raise AttributeError("Base dataset has no labels attribute")


def get_cifar10(
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True,
) -> datasets.CIFAR10:
    """Load raw CIFAR-10 (no transform applied here — transform is set on the wrapper)."""
    return datasets.CIFAR10(root=data_dir, train=train, download=download, transform=None)


def get_cifar100(
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True,
) -> datasets.CIFAR100:
    return datasets.CIFAR100(root=data_dir, train=train, download=download, transform=None)


def build_ssl_dataloader(
    dataset_name: str,
    data_dir: str,
    transform,
    batch_size: int = 256,
    num_workers: int = 4,
    subset_size: int | None = None,
    poison_indices: np.ndarray | None = None,
    poison_fn=None,
    shuffle: bool = True,
) -> tuple[DataLoader, PoisonAwareDataset]:
    """Build a DataLoader for SSL pretraining.

    Returns (dataloader, wrapped_dataset) so callers can access poison metadata.
    """
    if dataset_name == "cifar10":
        base = get_cifar10(data_dir, train=True)
    elif dataset_name == "cifar100":
        base = get_cifar100(data_dir, train=True)
    elif dataset_name == "stl10":
        from prcl.datasets.stl10 import get_stl10
        base = get_stl10(data_dir, split="train+unlabeled")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if subset_size is not None and subset_size < len(base):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(base), size=subset_size, replace=False)
        base = Subset(base, indices)

    wrapped = PoisonAwareDataset(
        base_dataset=base,
        transform=transform,
        poison_indices=poison_indices,
        poison_fn=poison_fn,
    )

    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader, wrapped


def build_eval_dataloader(
    dataset_name: str,
    data_dir: str,
    transform,
    batch_size: int = 256,
    num_workers: int = 4,
    train: bool = True,
) -> DataLoader:
    """Build a DataLoader for downstream evaluation (with labels)."""
    if dataset_name == "cifar10":
        base = get_cifar10(data_dir, train=train)
    elif dataset_name == "cifar100":
        base = get_cifar100(data_dir, train=train)
    elif dataset_name == "stl10":
        from prcl.datasets.stl10 import get_stl10
        split = "train" if train else "test"
        base = get_stl10(data_dir, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Simple wrapper that applies transform and returns (img, label)
    class EvalDataset(Dataset):
        def __init__(self, ds, tfm):
            self.ds = ds
            self.tfm = tfm

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, label = self.ds[idx]
            if self.tfm is not None:
                img = self.tfm(img)
            return img, label

    ds = EvalDataset(base, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
