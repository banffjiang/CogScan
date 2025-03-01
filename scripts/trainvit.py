from utils import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from tqdm.notebook import tqdm
from data_processing import get_dataloaders 

seed_everything(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Data
train_loader, valid_loader, test_loader, num_classes = get_dataloaders()

# Define model
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.to(device)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, num_classes)
model = nn.DataParallel(model)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(7):
    model.train()
    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()