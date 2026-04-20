import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import Net
from utils import sparsity_loss, compute_sparsity

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Data
# -------------------------------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# -------------------------------
# Model
# -------------------------------
model = Net().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

# -------------------------------
# Training for different lambdas
# -------------------------------
lambdas = [1e-5, 1e-4, 1e-3]

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")

    for epoch in range(8):
        model.train()
        total_loss = 0

        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            # ✅ FIXED LOSS (scaled)
            loss = criterion(outputs, y) + lam * sparsity_loss(model) / 1000

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss:.2f}")

    # -------------------------------
    # Sparsity check
    # -------------------------------
    sparsity = compute_sparsity(model)
    print(f"Sparsity: {sparsity:.2f}%")

# -------------------------------
# Save model
# -------------------------------
torch.save(model.state_dict(), "model.pth")

print("\nModel saved as model.pth")