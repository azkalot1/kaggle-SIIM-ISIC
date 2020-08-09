import timm
import torch.nn as nn
from .layers import AdaptiveConcatPool2d
import torch


class ClassificationSingleHeadMax(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1, pretrained=True):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=pretrained)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nc, num_classes))

        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )

    def get_features(self, x):
        x = self.enc(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationDounleHeadMax(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1, pretrained=True):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=pretrained)
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
        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )

    def get_features(self, x):
        x = self.enc(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationSingleHeadConcat(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1, pretrained=True):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=pretrained)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*nc, num_classes))
        self.features = nn.Sequential(
            nn.AdaptiveConcatPool2d((1, 1)),
            nn.Flatten()
        )

    def get_features(self, x):
        x = self.enc(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)


class ClassificationDounleHeadConcat(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1, pretrained=True):
        super().__init__()
        m = timm.create_model(
            model_name,
            pretrained=pretrained)
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

        self.features = nn.Sequential(
            nn.AdaptiveConcatPool2d((1, 1)),
            nn.Flatten()
        )

    def get_features(self, x):
        x = self.enc(x)
        x = self.features(x)
        return x

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


class Generator_auxGAN(nn.Module):
    def __init__(self, n_classes=2, latent_dim=100, channels=3):
        super(Generator_auxGAN, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        # self.init_size = img_size // 4  # Initial size before upsampling
        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),

            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),

            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),

            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),

            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),

            # nn.BatchNorm2d(16, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(16, 8, 3, stride=1, padding=1),

            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_auxGAN(nn.Module):
    def __init__(self, img_size=128, channels=3, n_classes=2):
        super(Discriminator_auxGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 6

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        # print(out.size())
        out = out.view(out.shape[0], -1)
        # print(out.size())
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


class Generator_auxGAN_512(nn.Module):
    def __init__(self, n_classes=2, latent_dim=100, channels=3):
        super(Generator_auxGAN_512, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 1024 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels[:, 0]), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_auxGAN_512(nn.Module):
    def __init__(self, img_size=512, channels=3, n_classes=2):
        super(Discriminator_auxGAN_512, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 7

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(1024 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(1024 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
