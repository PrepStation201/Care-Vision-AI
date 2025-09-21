import pytest
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from src.data.dataloader import MRI_Dataset

@pytest.fixture(scope="module")
def dummy_data_dir(tmpdir_factory):
    """Creates a temporary directory with dummy MRI images for testing."""
    base_dir = tmpdir_factory.mktemp("data")
    class_names = ['glioma', 'no_tumor']

    for split in ['training', 'testing']:
        for class_name in class_names:
            # Create directory: data/training/glioma
            class_dir = Path(base_dir) / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            # Create a dummy image file
            dummy_image = Image.new('RGB', (100, 100), color = 'red')
            dummy_image.save(class_dir / f"dummy_img_{split}_{class_name}.png")

    return str(base_dir)

def test_dataset_initialization(dummy_data_dir):
    """Tests if the MRI_Dataset class initializes correctly."""
    train_dir = Path(dummy_data_dir) / "training"
    test_dir = Path(dummy_data_dir) / "testing"
    dataset = MRI_Dataset(train_dir=str(train_dir), test_dir=str(test_dir), batch_size=2)
    assert dataset.batch_size == 2
    assert "training" in dataset.train_dir
    assert "testing" in dataset.test_dir

def test_get_loaders(dummy_data_dir):
    """Tests the get_loaders method."""
    train_dir = Path(dummy_data_dir) / "training"
    test_dir = Path(dummy_data_dir) / "testing"
    dataset = MRI_Dataset(train_dir=str(train_dir), test_dir=str(test_dir), batch_size=1)
    
    train_loader, test_loader = dataset.get_loaders()

    # Check if they are DataLoader objects
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check the number of batches
    assert len(train_loader) == 2 # 2 classes, batch size 1
    assert len(test_loader) == 2  # 2 classes, batch size 1

    # Check the shape and type of a batch
    images, labels = next(iter(train_loader))
    assert images.shape == (1, 3, 224, 224) # (batch_size, channels, H, W)
    assert labels.shape == (1,)
    assert images.dtype == torch.float32