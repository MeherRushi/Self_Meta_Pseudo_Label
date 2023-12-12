from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loaders(BATCH_SIZE, labeled_dir, unlabeled_dir, test_dir, transform, simple_transform):
    """
    Function to get DataLoader instances for labeled, unlabeled, and test datasets.
    """
    labeled_data = ImageFolder(root=labeled_dir, transform=get_transform())
    unlabeled_data = ImageFolder(root=unlabeled_dir, transform=simple_transform())
    unlabeled_data_augmented = ImageFolder(root=unlabeled_dir, transform=get_transform())
    test_data = ImageFolder(root=test_dir, transform=get_transform())

    labeled_train_loader = DataLoader(labeled_data, batch_size=BATCH_SIZE, shuffle=True)
    unlabeled_train_loader = DataLoader(unlabeled_data, batch_size=BATCH_SIZE, shuffle=True)
    unlabeled_train_loader_augmented = DataLoader(unlabeled_data_augmented, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Labeled: {len(labeled_train_loader)} | Unlabeled_Augmented: {len(unlabeled_train_loader_augmented)} | Unlabeled: {len(unlabeled_train_loader)} | Test: {len(test_loader)}")

    return labeled_train_loader, unlabeled_train_loader_augmented, unlabeled_train_loader, test_loader
