"""SimCLR augmentation pipeline — dataset-aware standard contrastive views."""

from torchvision import transforms


def get_simclr_transform(dataset_name: str = "cifar10", train: bool = True):
    """Return the SimCLR two-view transform for the specified dataset.

    When `train=True`, returns a `TwoViewTransform` that produces (view1, view2).
    When `train=False`, returns a simple eval transform (single view, no augmentation).
    """
    if dataset_name in ("cifar10", "cifar100"):
        img_size = 32
        blur_kernel = 3  # small kernel for 32x32
    elif dataset_name == "stl10":
        img_size = 96
        blur_kernel = 9
    else:
        img_size = 32
        blur_kernel = 3

    if train:
        base_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=blur_kernel)],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])
        return TwoViewTransform(base_transform)
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])


class TwoViewTransform:
    """Applies the same base transform independently twice to produce two views."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)
