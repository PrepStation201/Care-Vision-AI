import pytest
import torch
from src.utils.model_loader import load_model

def test_load_model_efficientnet():
    """Tests if the EfficientNet model loads correctly."""
    model = load_model("EfficientNet", num_classes=4)
    # Check if the final layer has the correct number of output features
    assert model.classifier[1].out_features == 4

def test_load_model_vgg16():
    """Tests if the VGG16 model loads correctly."""
    model = load_model("VGG16", num_classes=4)
    assert model.classifier[6].out_features == 4

def test_load_unsupported_model():
    """Tests if loading an unsupported model raises an error."""
    with pytest.raises(ValueError):
        load_model("UnsupportedModel", num_classes=4)

def test_model_forward_pass():
    """
    Tests if a loaded model can perform a forward pass and produce the correct output shape.
    """
    model = load_model("EfficientNet", num_classes=4)
    # Create a dummy input tensor
    dummy_input = torch.randn(2, 3, 224, 224) # (batch_size, channels, H, W)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check if the output has the correct shape: (batch_size, num_classes)
    assert output.shape == (2, 4)