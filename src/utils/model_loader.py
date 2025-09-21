# src/utils/model_loader.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from transformers import ViTForImageClassification, AutoImageProcessor, AutoConfig

def load_model_and_transform(name):
    """Loads the specified model architecture, weights, and required transform."""
    classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

    # ✅ Model paths
    vgg16_path = r"models/vgg16_brain_mri/vgg16_brain_mri.pth"
    resnet_path = r"models/resnet50_brain_mri/resnet50_brain_mri.pth"
    efficientnet_path = r"models/efficientnet_model/content/efficientnet_b0_brain_mri.pth"
    vit_path = r"models/vit_brain_model"

    if name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(4096, 4)
        )
        model.load_state_dict(torch.load(vgg16_path, map_location='cpu'))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return model, transform, classes

    elif name == "ResNet":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 4)
        )
        model.load_state_dict(torch.load(resnet_path, map_location='cpu'))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return model, transform, classes

    elif name == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
        model.load_state_dict(torch.load(efficientnet_path, map_location='cpu'))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return model, transform, classes

    elif name == "ViT":
        config = AutoConfig.from_pretrained(vit_path)
        model = ViTForImageClassification.from_pretrained(
            vit_path,
            config=config,
            local_files_only=True,
            use_safetensors=True
        )
        extractor = AutoImageProcessor.from_pretrained(vit_path)

        def transform_fn(image):
            # The ViT extractor handles resize, normalization, and conversion to tensor
            return extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return model, transform_fn, classes

    else:
        raise ValueError(f"Unknown model name: {name}")