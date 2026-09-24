from torchvision import datasets, transforms


def get_cifar10(data_dir: str = "./data"):
    normalize = [
        transforms.ToTensor(), # [0, 1], shape (3, 32, 32)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -> [-1, 1]
    ]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip()] + normalize)
    test_transform = transforms.Compose(normalize)

    train_set = datasets.CIFAR10(data_dir,
                                 train=True,
                                 download=True,
                                 transform=train_transform)
    val_set = datasets.CIFAR10(data_dir,
                                train=False,
                                download=True,
                                transform=test_transform)

    return train_set, val_set


if __name__ == "__main__":
    train_set, val_set = get_cifar10()

    print(f"Train: {len(train_set)} samples, Test: {len(val_set)} samples")

    x, y = train_set[0]
    print(f"Sample shape: {tuple(x.shape)}, dtype: {x.dtype}, label: {y}")
    print(f"Values range: [{x.min():.3f}, {x.max():.3f}]")

    labels = set(train_set.targets)
    assert labels == set(range(10)), f"Unexpected labels: {labels}"
    print(f"Labels OK (0-9), {len(train_set.classes)} classes, class-conditioning ready")
