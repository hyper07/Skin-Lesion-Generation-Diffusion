"""
Prebuilt Conditional GAN Model
StyleGAN-ish architecture designed for fine-tuning
"""

import torch
import torch.nn as nn
import math


class ConditionalGenerator(nn.Module):
    """
    Conditional generator (StyleGAN-ish but simplified).

    - Input: latent z (B, z_dim), labels (B)
    - Output: (B, 3, H, W) in [-1, 1]
    - Has label embeddings; designed for fine-tuning.
    """

    def __init__(self, z_dim, num_classes, img_size=256, base_channels=256):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, z_dim)

        self.fc = nn.Linear(z_dim, base_channels * 8 * 8)

        # Upsample from 8x8 to img_size
        num_upsamples = int(round(math.log2(img_size // 8)))
        channels = base_channels
        blocks = []
        for _ in range(num_upsamples):
            blocks += [
                nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
                nn.BatchNorm2d(channels // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels //= 2

        self.upsample_blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)
        x = z + label_vec
        x = self.fc(x).view(x.size(0), -1, 8, 8)
        x = self.upsample_blocks(x)
        x = self.to_rgb(x)
        return x


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator:
    - Embeds label -> spatial mask.
    - Concatenates mask with image.
    """

    def __init__(self, num_classes, img_size=256, base_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        in_channels = 4  # 3 (RGB) + 1 (label mask)
        layers = []
        layers += block(in_channels, base_channels, normalize=False)
        layers += block(base_channels, base_channels * 2)
        layers += block(base_channels * 2, base_channels * 4)
        layers += block(base_channels * 4, base_channels * 8)

        self.conv = nn.Sequential(*layers)

        down_factor = 2 ** 4
        final_size = img_size // down_factor
        self.adv = nn.Conv2d(base_channels * 8, 1, final_size, 1, 0)

    def forward(self, img, labels):
        mask = self.label_emb(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, mask], dim=1)
        feat = self.conv(x)
        out = self.adv(feat).view(-1)
        return out

