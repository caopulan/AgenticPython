import gzip
import logging
from pathlib import Path
import struct
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


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


def read_mnist_images(path: Path) -> torch.Tensor:
    with open_maybe_gzip(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected MNIST image magic: {magic}")
        data = handle.read(count * rows * cols)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(count, 1, rows, cols)


def read_mnist_labels(path: Path) -> torch.Tensor:
    with open_maybe_gzip(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected MNIST label magic: {magic}")
        data = handle.read(count)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def open_maybe_gzip(path: Path):
    if path.exists():
        return path.open("rb")
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(f"MNIST file not found: {path}")


class MnistVGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(512, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(1))


mnist_root = Path("data") / "MNIST" / "raw"
train_images = read_mnist_images(mnist_root / "train-images-idx3-ubyte")
train_labels = read_mnist_labels(mnist_root / "train-labels-idx1-ubyte")
test_images = read_mnist_images(mnist_root / "t10k-images-idx3-ubyte")
test_labels = read_mnist_labels(mnist_root / "t10k-labels-idx1-ubyte")
train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=128)

model = MnistVGG16().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def training_batches():
    total_batches = len(train_loader)
    for epoch in range(1, num_epochs + 1):
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            yield epoch, batch_index, total_batches, images, labels


def train_one_minibatch(images, labels):
    model.train()
    images = prepare_images(images).to(device)
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


def summarize_current_grads():
    total_sq = 0.0
    max_abs = 0.0
    param_count = 0
    tensor_count = 0
    for _, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_sq += float((grad * grad).sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
        param_count += grad.numel()
        tensor_count += 1
    summary = {
        "epoch": last_batch_summary.get("epoch"),
        "batch": last_batch_summary.get("batch"),
        "total_batches": last_batch_summary.get("total_batches"),
        "batch_loss": last_batch_summary.get("batch_loss"),
        "running_loss": last_batch_summary.get("running_loss"),
        "lr": optimizer.param_groups[0]["lr"],
        "grad_l2": total_sq ** 0.5,
        "grad_max_abs": max_abs,
        "grad_tensors": tensor_count,
        "grad_params": param_count,
    }
    return summary


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
            images = prepare_images(images).to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    accuracy = correct / total
    evals.append((epoch, round(accuracy, 4)))
    logging.info("epoch=%s evaluation accuracy=%.4f correct=%s total=%s", epoch, accuracy, correct, total)
    return accuracy


def prepare_images(images):
    images = images.float().div(255.0)
    images = F.interpolate(images, size=(32, 32), mode="bilinear", align_corners=False)
    return images.sub(0.1307).div(0.3081)


def maybe_evaluate(epoch, batch_index, total_batches):
    if batch_index == total_batches and epoch % eval_every_epochs == 0:
        return evaluate(epoch)
    return None


logging.info(
    (
        "starting native CPython CPU MNIST training epochs=%s eval_every_epochs=%s "
        "train_examples=%s test_examples=%s batches_per_epoch=%s "
        "batch_size=%s model=MnistVGG16 lr=%.5f"
    ),
    num_epochs,
    eval_every_epochs,
    len(train_loader.dataset),
    len(test_loader.dataset),
    len(train_loader),
    batch_size,
    optimizer.param_groups[0]["lr"],
)
for epoch, batch_index, total_batches, images, labels in training_batches():
    loss_value, batch_examples = train_one_minibatch(images, labels)
    track_training_loss(epoch, batch_index, total_batches, loss_value, batch_examples)
    maybe_finish_epoch(epoch, batch_index, total_batches)
    maybe_evaluate(epoch, batch_index, total_batches)

logging.info("training complete history=%s evals=%s", history, evals)
