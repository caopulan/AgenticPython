from pathlib import Path

from agenticpython.parser import parse_script


def test_cpu_mnist_is_epoch_based_and_reports_evaluation_accuracy():
    source = Path("examples/cpu_mnist.py").read_text(encoding="utf-8")

    assert "num_epochs =" in source
    assert "eval_every_epochs =" in source
    assert "logging.basicConfig" in source
    assert "logging.info" in source
    assert "evaluation accuracy" in source
    assert "total_steps =" not in source
    assert "Subset" not in source
    assert "DataLoader(train_dataset" in source
    assert "DataLoader(test_dataset" in source
    parse_script(source, filename="examples/cpu_mnist.py")
