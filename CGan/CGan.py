# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Conditional BatchNorm --------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, embed_dim=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        embed_dim = embed_dim or num_features
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.gamma = nn.Linear(embed_dim, num_features)
        self.beta = nn.Linear(embed_dim, num_features)
        nn.init.ones_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, x, y):
        h = self.bn(x)
        e = self.embed(y)
        gamma = self.gamma(e).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta(e).unsqueeze(-1).unsqueeze(-1)
        return h * gamma + beta


# -------- Generator --------
class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.cbn1 = ConditionalBatchNorm2d(in_ch, num_classes)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.cbn2 = ConditionalBatchNorm2d(out_ch, num_classes)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, y):
        out = self.cbn1(x, y)
        out = F.relu(out, inplace=True)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.cbn2(out, y)
        out = F.relu(out, inplace=True)
        return self.conv2(out)


class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, base_ch=64, img_size=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, 4 * 4 * base_ch * 8)
        self.blocks = nn.ModuleList([
            GenBlock(base_ch * 8, base_ch * 4, num_classes),
            GenBlock(base_ch * 4, base_ch * 2, num_classes),
            GenBlock(base_ch * 2, base_ch * 1, num_classes),
        ])
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(base_ch, affine=True),
            nn.ReLU(True),
            nn.Conv2d(base_ch, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, y):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        for blk in self.blocks:
            h = blk(h, y)
        return self.to_rgb(h)


# -------- Discriminator --------
class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.downsample = downsample
        self.skip = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        ) if (downsample or in_ch != out_ch) else None

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        h = F.leaky_relu(self.conv2(h), 0.2, inplace=True)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        if self.skip is not None:
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return h + x if self.skip is not None else h


class Discriminator(nn.Module):
    def __init__(self, num_classes, base_ch=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiscBlock(3, base_ch, True),
            DiscBlock(base_ch, base_ch * 2, True),
            DiscBlock(base_ch * 2, base_ch * 4, True),
            DiscBlock(base_ch * 4, base_ch * 8, False),
        ])
        self.linear = nn.utils.spectral_norm(nn.Linear(base_ch * 8, 1))
        self.embed = nn.utils.spectral_norm(nn.Embedding(num_classes, base_ch * 8))

    def forward(self, x, y):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = h.mean(dim=[2, 3])
        out = self.linear(h).squeeze(1)
        proj = (self.embed(y) * h).sum(dim=1)
        return out + proj


# -------- Loss Functions --------
def d_hinge(real_logits, fake_logits):
    return F.relu(1. - real_logits).mean() + F.relu(1. + fake_logits).mean()

def g_hinge(fake_logits):
    return (-fake_logits).mean()
