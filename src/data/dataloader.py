import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MRI_Dataset:
    """
    A class to handle loading and transforming the MRI dataset.
    """
    def __init__(self, train_dir, test_dir, batch_size=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.train_transforms = self._get_train_transforms()
        self.test_transforms = self._get_test_transforms()

    def _get_train_transforms(self):
        """
        Applies transformations to the training dataset.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_test_transforms(self):
        """
        Applies transformations to the testing dataset.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        """
        Creates and returns the training and testing DataLoaders.
        """
        train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        test_data = datasets.ImageFolder(self.test_dir, transform=self.test_transforms)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        return train_loader, test_loader