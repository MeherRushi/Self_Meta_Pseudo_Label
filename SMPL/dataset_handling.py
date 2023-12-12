import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

def get_cifar10_data():
    """
    Function to get CIFAR-10 train and test data.
    """
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
    )
    return train_data, test_data

def load_batches(file_path):
    """
    Function to load batches from CIFAR-10 data files.
    """
    with open(file_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def create_classification_directory(input_dir, output_dir, subdirectory):
    """
    Function to create a classification directory structure for CIFAR-10.
    """
    os.makedirs(output_dir, exist_ok=True)
    for label_id, class_name in enumerate(subdirectory):
        class_name = class_name.decode('utf-8')
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

def Copy_train_Images(input_dir, output_dir):
    """
    Function to copy and save train images to the specified directory.
    """
    for batch_id in range(1, 6):
        batch_data = load_batches(os.path.join(input_dir, f'data_batch_{batch_id}'))
        for i, (image, label) in enumerate(zip(batch_data[b'data'], batch_data[b'labels'])):
            class_name = meta[b'label_names'][label].decode('utf-8')
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)
            image_filename = f'{batch_id}_{i + 1}.png'
            output_path = os.path.join(output_dir, class_name, image_filename)
            plt.imsave(output_path, image)

def Images_in_directory(input_dir):
    """
    Function to get the number of items in a directory.
    """
    from pathlib import Path
    dir = Path(input_dir)
    num_items = len(list(dir.glob('*/*.png')))
    print(f"Number of items in {dir} directory: {num_items}")

def Copy_test_Images(input_dir, output_dir):
    """
    Function to copy and save test images to the specified directory.
    """
    batch_data = load_batches(os.path.join(input_dir, 'test_batch'))
    for i, (image, label) in enumerate(zip(batch_data[b'data'], batch_data[b'labels'])):
        class_name = meta[b'label_names'][label].decode('utf-8')
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        image_filename = f'{i + 1}.png'
        output_path = os.path.join(output_dir, class_name, image_filename)
        plt.imsave(output_path, image)

def simple_dataset(input_dir):
    """
    Function to create a simple ImageFolder dataset.
    """
    from torchvision import datasets
    data = datasets.ImageFolder(root=input_dir)
    return data

def get_subdirectories(parent_dir):
    """
    Function to get subdirectories in a parent directory.
    """
    import os
    from pathlib import Path
    for subdirectory in os.listdir(parent_dir):
        subdirectory_path = os.path.join(parent_dir, subdirectory)
        if os.path.isdir(subdirectory_path):
            print("Subdirectory:", subdirectory_path)
    parent_dir = Path(parent_dir)
    subdirectories = [subdir for subdir in parent_dir.iterdir() if subdir.is_dir()]
    return subdirectories

def l_u_split(seed, source, l, u, ratio):
    """
    Function to split data into labeled and unlabeled based on a given ratio.
    """
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)

    import os
    import random
    from pathlib import Path

    source_dir = Path(source)
    labeled_dir = Path(l)
    unlabeled_dir = Path(u)

    labeled_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    split_ratio = ratio

    for class_name in os.listdir(source_dir):
        class_dir = source_dir / class_name

        labeled_class_dir = labeled_dir / class_name
        labeled_class_dir.mkdir(parents=True, exist_ok=True)

        unlabeled_class_dir = unlabeled_dir / class_name
        unlabeled_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(class_dir.glob("*.png"))

        random.shuffle(image_files)

        num_labeled = int(len(image_files) * split_ratio)
        num_unlabeled = len(image_files) - num_labeled

        for i in range(num_labeled):
            src_path = image_files[i]
            dest_path = labeled_class_dir / src_path.name
            src_path.rename(dest_path)

        for i in range(num_labeled, num_labeled + num_unlabeled):
            src_path = image_files[i]
            dest_path = unlabeled_class_dir / src_path.name
            src_path.rename(dest_path)
