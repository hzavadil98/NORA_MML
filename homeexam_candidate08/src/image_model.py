"""
===============================================================================
PyTorch Lightning Image Classification Components
===============================================================================

This module implements PyTorch Lightning components for image only classification task,
including data loading, model architecture, and training utilities. It provides
a complete pipeline for training and evaluating convolutional neural networks
on image datasets with automatic train/validation/test splits.

Classes:
--------
- Image_Dataset: Custom PyTorch Dataset for loading image tensors and labels
- Image_Dataloader: PyTorch Lightning DataModule for managing data splits and loading
- Conv2Net: Convolutional neural network architecture with 2 conv layers and dropout
- Image_Model: PyTorch Lightning Module wrapping Conv2Net with training logic

Features:
---------
- Automatic train/validation/test data splitting with stratification
- Configurable batch size, dropout rate, and model dimensions
- Built-in metrics tracking (accuracy, loss, confusion matrix)
- Weights & Biases integration for experiment logging
- Feature extraction capability for multimodal applications

Usage:
------
```python
from pathlib import Path
from image_model import Image_Dataloader, Image_Model

# Initialize data module
data_module = Image_Dataloader(
    path_to_data=Path("data/"),
    batch_size=32,
    val_ratio=0.2
)

# Initialize model
model = Image_Model(
    num_classes=10,
    learning_rate=0.001,
    inner_size=32
)

# Train with PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```
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
    """
    Custom PyTorch Dataset for loading image tensors and their corresponding labels.

    This dataset class handles pre-loaded image tensors and provides optional
    transformations during data loading. It's designed to work with tensor data
    that has been saved as .pth files.

    Args:
        data_tensor (torch.Tensor): Tensor containing image data
        labels_tensor (torch.Tensor): Tensor containing corresponding labels
        transform (callable, optional): Optional transform to be applied to samples

    Example:
        >>> data = torch.load("images.pth")
        >>> labels = torch.load("labels.pth")
        >>> dataset = Image_Dataset(data, labels, transform=some_transform)
        >>> sample, label = dataset[0]
    """

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
    """
    PyTorch Lightning DataModule for managing image dataset loading and splitting.

    This DataModule automatically handles train/validation/test splits from the
    provided training data, loads test data separately, and creates appropriate
    DataLoaders for each split. It expects data to be stored as .pth tensor files.

    Args:
        path_to_data (Path): Path to directory containing the .pth data files
        batch_size (int, optional): Batch size for all DataLoaders. Defaults to 32.
        val_ratio (float, optional): Fraction of training data to use for validation.
            Defaults to 0.2.
        num_workers (int, optional): Number of worker processes for data loading.
            Defaults to 4.
        transform (callable, optional): Transform to apply to training data only.
            Defaults to None.

    Expected Data Files:
        - training_images.pth: Training image tensors
        - training_images_labels.pth: Training labels
        - test_images.pth: Test image tensors
        - test_images_labels.pth: Test labels

    Example:
        >>> from pathlib import Path
        >>> data_module = Image_Dataloader(
        ...     path_to_data=Path("data/"),
        ...     batch_size=64,
        ...     val_ratio=0.15
        ... )
        >>> trainer.fit(model, data_module)
    """

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


class Conv2Net(torch.nn.Module):
    """
    Convolutional Neural Network with 2 convolutional layers for image classification.

    This network consists of two convolutional layers with ReLU activations and
    max pooling, followed by two fully connected layers. Dropout is applied for
    regularization. The architecture is designed for small-sized images of shape (1,28,28)
    (assumes final feature map size of 5x5 after convolutions).

    Architecture:
        - Conv2d(1, inner_size, kernel_size=3) + ReLU + MaxPool2d(2,2)
        - Conv2d(inner_size, inner_size*2, kernel_size=3) + Dropout + ReLU + MaxPool2d(2,2)
        - Linear(inner_size*2*5*5, inner_size*4) + Dropout + ReLU
        - Linear(inner_size*4, num_classes)

    Args:
        num_classes (int): Number of output classes
        drop_rate (float, optional): Dropout probability. Defaults to 0.5.
        inner_size (int, optional): Base number of filters/features. Defaults to 32.

    Example:
        >>> model = Conv2Net(num_classes=10, inner_size=64, drop_rate=0.3)
        >>> output = model(torch.randn(32, 1, 28, 28))  # batch_size=32, 28x28 grayscale
        >>> features = model.get_inner_features(torch.randn(32, 1, 28, 28))
    """

    def __init__(self, num_classes, drop_rate=0.5, inner_size=32):
        super().__init__()
        self.inner_size = inner_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.featurizer = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.inner_size, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.inner_size, self.inner_size * 2, kernel_size=3),
            torch.nn.Dropout(self.drop_rate),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = torch.nn.Linear(self.inner_size * 2 * 5 * 5, self.inner_size * 4)
        self.dropout2 = torch.nn.Dropout(self.drop_rate)
        self.relu3 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(self.inner_size * 4, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout2(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_inner_features(self, x):
        x = self.featurizer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout2(self.relu3(self.fc1(x)))
        return x


class Image_Model(pl.LightningModule):
    """
    PyTorch Lightning Module for image classification using Conv2Net architecture.

    This Lightning module wraps the Conv2Net model and provides complete training,
    validation, and testing functionality. It includes automatic metric tracking,
    confusion matrix generation, and Weights & Biases logging integration.

    Features:
        - Cross-entropy loss for multi-class classification
        - Accuracy and confusion matrix metrics
        - Adam optimizer with configurable learning rate
        - Automatic confusion matrix visualization and logging
        - Hyperparameter saving for reproducibility

    Args:
        num_classes (int): Number of output classes
        drop_rate (float, optional): Dropout probability for the underlying Conv2Net.
            Defaults to 0.5.
        inner_size (int, optional): Base number of filters/features for Conv2Net.
            Defaults to 32.
        learning_rate (float, optional): Learning rate for Adam optimizer.
            Defaults to 0.001.

    Example:
        >>> model = Image_Model(num_classes=10, learning_rate=0.001)
        >>> trainer = pl.Trainer(max_epochs=50)
        >>> trainer.fit(model, datamodule)
        >>> trainer.test(model, datamodule)
    """

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
        self.model = Conv2Net(
            num_classes=self.num_classes,
            drop_rate=self.drop_rate,
            inner_size=self.inner_size,
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
