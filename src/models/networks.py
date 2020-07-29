import timm
import torch.nn as nn
from .layers import AdaptiveConcatPool2d, Swish


class ClassificationSingleHeadMax(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nc, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationDounleHeadMax(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc, nc//2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//2, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationSingleHeadConcat(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*nc, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationDounleHeadConcat(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*nc, nc//2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//2, num_classes))

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class Generator(nn.Module):
    """
    Generator for WGAN-GP for 128x128 images
    """
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # Now instead of ConvTranspose2d we use
            # Upsample -> ReflectionPad2d -> Conv2d
            # presumably this should remove the typical GAN pattern
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # output of main module --> Image (Cx128x128)
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0))

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(nn.Module):
    """
    Discriminator for WGAN-GP for 128x128 images
    """
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because
            # our new penalized training objective (WGAN with gradient penalty)
            # is no longer valid
            # in this setting, since we penalize the norm of the critic's
            # gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization -->
            # using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            # output of main module --> State (1024x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)
