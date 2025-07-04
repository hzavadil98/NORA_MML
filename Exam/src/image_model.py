"""
===============================================================================
PyTorch Lightning Dataloaders and Model Definitions
===============================================================================

This module contains the implementation of PyTorch Lightning dataloaders and
model definitions. It provides reusable components for data loading and
model architecture, facilitating efficient training and evaluation workflows
using the PyTorch Lightning framework.

Classes and Functions:
----------------------
- Dataloaders for preparing and batching datasets.
- Model classes defining neural network architectures compatible with Lightning.

Usage:
------
Import the required dataloader or model class and integrate it into your
Lightning training pipeline.

Author: [Your Name]
Date: [Date]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix


class Image_Dataset(Dataset):
    def __init__(self, data_tensor, labels_tensor, transform=None):
        super().__init__()
        self.data = data_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Load and return a sample from the dataset
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class Image_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        path_to_data: Path,
        batch_size=32,
        val_ratio=0.2,
        num_workers=4,
        transform=None,
    ):
        super().__init__()
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        # Load the original image data and labels
        train_data = torch.load(self.path_to_data / "training_images.pth")
        train_labels = torch.load(self.path_to_data / "training_images_labels.pth")

        test_data = torch.load(self.path_to_data / "test_images.pth")
        test_labels = torch.load(self.path_to_data / "test_images_labels.pth")

        # Split the training data into training and validation sets
        train_data, validation_data, train_labels, validation_labels = train_test_split(
            train_data,
            train_labels,
            test_size=val_ratio,
            random_state=42,
            stratify=train_labels,
        )

        # Create datasets for training, validation, and testing
        self.train_dataset = Image_Dataset(
            train_data, train_labels, transform=self.transform
        )
        self.val_dataset = Image_Dataset(
            validation_data, validation_labels, transform=None
        )
        self.test_dataset = Image_Dataset(test_data, test_labels, transform=None)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class Image_Model(pl.LightningModule):
    def __init__(self, num_classes, drop_rate=0.5, inner_size=32, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.inner_size = inner_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(num_classes=self.num_classes, task="multiclass")
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.save_hyperparameters()

        # Use pretrained ResNet18 model
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.inner_size, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.inner_size, self.inner_size * 2, kernel_size=3),
            torch.nn.Dropout(self.drop_rate),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(self.inner_size * 2 * 5 * 5, self.inner_size * 4),
            torch.nn.Dropout(self.drop_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inner_size * 4, self.num_classes),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.accuracy(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", self.accuracy(logits, y), prog_bar=True)
        self.confusion_matrix.update(torch.argmax(logits, dim=1), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @rank_zero_only
    def on_test_epoch_end(self):
        cm = self.confusion_matrix.compute().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_title("Confusion Matrix")

        # Log confusion matrix to wandb
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
