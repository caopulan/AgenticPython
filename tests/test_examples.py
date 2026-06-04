from pathlib import Path

from agenticpython.parser import parse_script


def test_cpu_mnist_is_epoch_based_and_reports_evaluation_accuracy():
    source = Path("examples/cpu_mnist.py").read_text(encoding="utf-8")

    assert "num_epochs =" in source
    assert "eval_every_epochs =" in source
    assert "batch_size = 64" in source
    assert "logging.basicConfig" in source
    assert "logging.info" in source
    assert "evaluation accuracy" in source
    assert "models.vgg16" in source
    assert "nn.Conv2d(1, 64" in source
    assert "nn.Linear(512, 10)" in source
    assert "log_every_batches =" in source
    assert "def training_batches():" in source
    assert "def train_one_minibatch(" in source
    assert "for epoch, batch_index, total_batches, images, labels in training_batches():" in source
    assert "train_one_epoch" not in source
    assert "total_steps =" not in source
    assert "Subset" not in source
    assert "DataLoader(train_dataset" in source
    assert "DataLoader(test_dataset" in source
    parse_script(source, filename="examples/cpu_mnist.py")
