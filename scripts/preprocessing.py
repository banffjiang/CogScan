import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from utils import seed_everything
seed_everything(42)




transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
raw_test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)

train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset_full, val_dataset_full = random_split(full_train_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

train_dataset = TransformDataset(train_dataset_full, transforms_train)
val_dataset = TransformDataset(val_dataset_full, transforms_val)
test_dataset = TransformDataset(raw_test_dataset, transforms_test)


classes = full_train_dataset.classes
num_classes = len(classes)
print("Number of classes:", num_classes)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_dataset, batch_size = 64 , shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False, num_workers=4)

print(f"Train Dataset size: {len(train_dataset)}")
print(f"Validation Dataset size: {len(val_dataset)}")
print(f"Test Dataset size: {len(test_dataset)}")
