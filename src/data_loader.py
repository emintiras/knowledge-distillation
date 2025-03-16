import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import logging

logger = logging.getLogger(__name__)

def get_data_loaders(batch_size, data_root='data/', seed=42):
    """
    Return data loaders for CIFAR-10

    Args:
    batch_size (int): Batch size for the data loaders.
    data_root (str): Path to store/download datasets.
    seed (int): Seed for reproducibility.

    Returns:
    dict: Data loaders for train and test splits of CIFAR-10.
    """
    try:
        logger.info('Initializing data loaders...')

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        logger.info('Loading the CIFAR-10 dataset')
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

        loaders = {
            'cifar_train' : DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'cifar_test' : DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

        logger.info(f'Data loaders initialized successfully with batch size {batch_size}.')
        logger.info(f'CIFAR-10 Train Size: {len(train_dataset)}, Test Size: {len(test_dataset)}')

        return loaders
    
    except Exception as e:
        logger.critical(f'Failed to initialize data loaders: {str(e)}', exc_info=True)
        raise