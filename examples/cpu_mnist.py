import logging
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

torch.manual_seed(0)
device = torch.device("cpu")
batch_size = 64
num_epochs = 3
eval_every_epochs = 1
log_first_batches = 5
log_every_batches = 25
lr = 0.1
history = []
evals = []
batch_loss_trace = []
last_batch_summary = {}
epoch_loss_total = 0.0
epoch_examples = 0

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

model = models.vgg16(weights=None)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.classifier = nn.Sequential(nn.Linear(512, 10))
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def training_batches():
    total_batches = len(train_loader)
    for epoch in range(1, num_epochs + 1):
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            yield epoch, batch_index, total_batches, images, labels


def train_one_minibatch(images, labels):
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), int(labels.numel())


def track_training_loss(epoch, batch_index, total_batches, loss_value, batch_examples):
    global epoch_loss_total, epoch_examples, last_batch_summary
    epoch_loss_total += loss_value * batch_examples
    epoch_examples += batch_examples
    running_loss = epoch_loss_total / epoch_examples
    current_lr = optimizer.param_groups[0]["lr"]
    last_batch_summary = {
        "epoch": epoch,
        "batch": batch_index,
        "total_batches": total_batches,
        "batch_loss": round(loss_value, 4),
        "running_loss": round(running_loss, 4),
        "lr": current_lr,
        "epoch_examples": epoch_examples,
    }
    batch_loss_trace.append((epoch, batch_index, round(loss_value, 4), round(running_loss, 4), current_lr))
    if should_log_batch(batch_index, total_batches):
        log_training_progress(epoch, batch_index, total_batches, loss_value, running_loss, current_lr)


def should_log_batch(batch_index, total_batches):
    return batch_index <= log_first_batches or batch_index % log_every_batches == 0 or batch_index == total_batches


def log_training_progress(epoch, batch_index, total_batches, loss_value, running_loss, current_lr):
    logging.info(
        "progress epoch=%s/%s batch=%s/%s batch_loss=%.4f running_loss=%.4f lr=%.5f examples=%s",
        epoch,
        num_epochs,
        batch_index,
        total_batches,
        loss_value,
        running_loss,
        current_lr,
        epoch_examples,
    )


def maybe_finish_epoch(epoch, batch_index, total_batches):
    global epoch_loss_total, epoch_examples
    if batch_index != total_batches:
        return None
    average_loss = epoch_loss_total / epoch_examples
    current_lr = optimizer.param_groups[0]["lr"]
    history.append((epoch, round(average_loss, 4), current_lr))
    logging.info(
        "epoch=%s/%s train_loss=%.4f lr=%.5f",
        epoch,
        num_epochs,
        average_loss,
        current_lr,
    )
    epoch_loss_total = 0.0
    epoch_examples = 0
    return average_loss


def evaluate(epoch):
    model.eval()
    correct = 0
    total = 0
    logging.info("epoch=%s evaluation started test_examples=%s", epoch, len(test_loader.dataset))
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    accuracy = correct / total
    evals.append((epoch, round(accuracy, 4)))
    logging.info("epoch=%s evaluation accuracy=%.4f correct=%s total=%s", epoch, accuracy, correct, total)
    return accuracy


def maybe_evaluate(epoch, batch_index, total_batches):
    if batch_index == total_batches and epoch % eval_every_epochs == 0:
        return evaluate(epoch)
    return None


logging.info(
    (
        "starting CPU MNIST training epochs=%s eval_every_epochs=%s "
        "train_examples=%s test_examples=%s batches_per_epoch=%s "
        "log_first_batches=%s log_every_batches=%s"
    ),
    num_epochs,
    eval_every_epochs,
    len(train_loader.dataset),
    len(test_loader.dataset),
    len(train_loader),
    log_first_batches,
    log_every_batches,
)
for epoch, batch_index, total_batches, images, labels in training_batches():
    loss_value, batch_examples = train_one_minibatch(images, labels)
    track_training_loss(epoch, batch_index, total_batches, loss_value, batch_examples)
    maybe_finish_epoch(epoch, batch_index, total_batches)
    maybe_evaluate(epoch, batch_index, total_batches)

logging.info("training complete history=%s evals=%s", history, evals)
