from utils import seed_everything
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from tqdm.notebook import tqdm
from preprocessing import train_loader, valid_loader, num_classes

seed_everything(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.to(device)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, num_classes)
model = nn.DataParallel(model)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(7):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()
    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc.item() / len(train_loader)
        epoch_loss += loss.item() / len(train_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}"):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc.item() / len(valid_loader)
            epoch_val_loss += val_loss.item() / len(valid_loader)

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(
        f"Epoch : {epoch+1} - Loss : {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f} - "
        f"Validation loss : {epoch_val_loss:.4f} - Validation Accuracy: {epoch_val_accuracy:.4f}\n"
    )

    scheduler.step()