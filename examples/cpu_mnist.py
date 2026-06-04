from itertools import cycle

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


torch.manual_seed(0)
device = torch.device("cpu")
batch_size = 64
total_steps = 30
eval_every = 10
lr = 0.1
history = []
evals = []

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(Subset(train_dataset, range(2048)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Subset(test_dataset, range(512)), batch_size=128)
train_iter = cycle(train_loader)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train_one_step():
    images, labels = next(train_iter)
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    model.train()
    return correct / total


for step in range(total_steps):
    loss = train_one_step()
    current_lr = optimizer.param_groups[0]["lr"]
    history.append((step, round(loss, 4), current_lr))
    if step % eval_every == 0:
        accuracy = evaluate()
        evals.append((step, round(accuracy, 4)))
    print(f"step={step} loss={loss:.4f} lr={current_lr:.5f}")

print("history", history)
print("evals", evals)
