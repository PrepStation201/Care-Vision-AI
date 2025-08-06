import torch
import torchvision.models as models
from torchvision import transforms
# from transformers import ViTForImageClassification, AutoFeatureExtractor, AutoConfig
from transformers import ViTForImageClassification, AutoImageProcessor, AutoConfig


import os
import torch.nn as nn

def load_model_and_transform(name):
    classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

    # ✅ Model paths
    vgg16_path = r"models/vgg16_brain_mri/vgg16_brain_mri.pth"
    resnet_path = r"models/resnet50_brain_mri/resnet50_brain_mri.pth"
    efficientnet_path = r"models/efficientnet_model/content/efficientnet_b0_brain_mri.pth"
    vit_path = r"models/vit_brain_model"

    if name == "VGG16":
        model = models.vgg16(pretrained=False)
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
        model = models.resnet50(pretrained=False)  # Make sure this matches the model you trained
        # model.fc = torch.nn.Linear(model.fc.in_features, 4)
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

        return model, transform, classes  # ✅ MUST return here


    elif name == "EfficientNet":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=False)
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
            use_safetensors=True   # ✅ Important
        )
        # extractor = AutoFeatureExtractor.from_pretrained(vit_path)
        extractor = AutoImageProcessor.from_pretrained(vit_path)


        def transform_fn(image):
            return extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return model, transform_fn, classes

    else:
        raise ValueError(f"Unknown model name: {name}")
