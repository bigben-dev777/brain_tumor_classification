import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from lightning import LightningDataModule
from torchvision import transforms
from typing import Optional


class BrainTumorDataModule(LightningDataModule):
    """DataModule for Brain Tumor dataset."""

    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return 4

    def prepare_data(self) -> None:
        """Download the Brain Tumor dataset if not already present."""
        datasets.ImageFolder(root=self.data_dir, transform=self.transforms)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the datasets for training, validation, and testing."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transforms)
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size

            self.data_train, self.data_val, self.data_test = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
        )