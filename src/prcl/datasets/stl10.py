"""STL-10 dataset wrapper with poison-aware support for SSL pretraining."""

from torchvision import datasets


def get_stl10(
    data_dir: str = "./data",
    split: str = "train+unlabeled",
    download: bool = True,
) -> datasets.STL10:
    """Load raw STL-10 (no transform — transform set on wrapper).

    STL-10 is 96×96 RGB. The 'train+unlabeled' split gives ~105k images
    which is the standard for SSL pretraining. The 'train' split has 5k
    labeled images; 'test' has 8k.
    """
    return datasets.STL10(
        root=data_dir, split=split, download=download, transform=None
    )
