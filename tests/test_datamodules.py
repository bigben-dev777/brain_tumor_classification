from pathlib import Path

import pytest
import torch

from src.data.brain_tumor_datamodule import BrainTumorDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_brain_tumor_datamodule(batch_size: int) -> None:
    """Test BrainTumorDataModule using an ImageFolder style dataset"""

    data_dir = Path('data/BRAINTUMOR')

    assert (data_dir / "train").exists()
    assert (data_dir / "test").exists()

    dm = BrainTumorDataModule(data_dir=str(data_dir), batch_size=batch_size)

    dm.prepare_data()

    dm.setup()

    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.data_test is not None

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    x,y = next(iter(train_loader))

    assert x.shape[0] == batch_size
    assert y.shape[0] == batch_size
