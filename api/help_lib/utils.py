from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.generator import generate_samples
from helper_lib.model import get_model, train_vae_model
from helper_lib.utils import save_model
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Check which GPU is available
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

train_loader = get_data_loader('data/train', batch_size=64)
test_loader = get_data_loader('data/test', batch_size=64, train=False)
model = get_model("CNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trained_model = train_model(model, train_loader, criterion, optimizer, epochs=5)
evaluate_model(trained_model, test_loader, criterion)


vae = get_model("VAE")
criterion = nn.BCELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(vae, train_loader, criterion, optimizer, epochs=5)
generate_samples(vae, device, num_samples=10)