import timm
import torch.nn as nn
from .layers import AdaptiveConcatPool2d


class ClassificationSingleHeadMax(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=2):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nc, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationDounleHeadMax(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=2):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nc, nc//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//2, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationSingleHeadConcat(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=2):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2*nc, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationDounleHeadConcat(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=2):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2*nc, nc//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//2, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)
