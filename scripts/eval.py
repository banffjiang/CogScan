import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
import numpy as np
from preprocessing import classes
from trainvit import model, criterion, test_loader, device, train_accuracies, val_accuracies, train_losses, val_losses

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="Training")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(train_accuracies, label="Training")
plt.plot(val_accuracies, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


model.eval()
with torch.no_grad():
    test_accuracy = 0
    test_loss = 0
    for data, label in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        label = label.to(device)

        test_output = model(data)
        loss = criterion(test_output, label)

        acc = (test_output.argmax(dim=1) == label).float().mean()
        test_accuracy += acc.item() / len(test_loader)
        test_loss += loss.item() / len(test_loader)

    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")


    model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)

images = images.to(device)
labels = labels.to(device)

outputs = model(images)
_, preds = torch.max(outputs, 1)

images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, ax in enumerate(axes.ravel()):
    if i >= len(images):
        break
    img = images[i].permute(1, 2, 0)
    # Denormalize for visualization
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img = img.numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    gt = classes[labels[i]]
    pred = classes[preds[i]]
    ax.set_title(f"Ground Truth: {gt} | Prediction: {pred}")
    ax.axis("off")
plt.tight_layout()
plt.show()