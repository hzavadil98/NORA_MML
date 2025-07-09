from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from audio_model import Conv1Net
from image_model import Conv2Net
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix


class Multimodal_Dataset(Dataset):
    def __init__(
        self,
        image_tensor,
        audio_tensor,
        labels_tensor,
        image_transform=None,
        audio_transform=None,
    ):
        super().__init__()
        self.image_data = image_tensor
        self.audio_data = audio_tensor
        self.labels = labels_tensor
        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def __len__(self):
        # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and return a sample from the dataset
        image = self.image_data[idx]
        audio = self.audio_data[idx]
        label = self.labels[idx]
        if self.image_transform:
            image = self.image_transform(image)
        if self.audio_transform:
            audio = self.audio_transform(audio)
        return (image, audio), label


class Multimodal_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        path_to_data: Path,
        batch_size=32,
        val_ratio=0.2,
        num_workers=4,
        image_transform=None,
        audio_transform=None,
    ):
        super().__init__()
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_transform = image_transform
        self.audio_transform = audio_transform

        # Load the original audio data and labels
        train_image_data = torch.load(self.path_to_data / "training_images.pth").to(
            dtype=torch.float32
        )
        train_audio_data = torch.load(self.path_to_data / "training_audio.pth").to(
            dtype=torch.float32
        )
        train_labels = torch.load(self.path_to_data / "training_audio_labels.pth")

        test_image_data = torch.load(self.path_to_data / "test_images.pth").to(
            dtype=torch.float32
        )
        test_audio_data = torch.load(self.path_to_data / "test_audio.pth").to(
            dtype=torch.float32
        )
        test_labels = torch.load(self.path_to_data / "test_audio_labels.pth")

        # Split the training data into training and validation sets
        (
            train_image_data,
            validation_image_data,
            train_audio_data,
            validation_audio_data,
            train_labels,
            validation_labels,
        ) = train_test_split(
            train_image_data,
            train_audio_data,
            train_labels,
            test_size=val_ratio,
            random_state=42,
            stratify=train_labels,
        )

        # Create datasets for training, validation, and testing
        self.train_dataset = Multimodal_Dataset(
            train_image_data,
            train_audio_data,
            train_labels,
            image_transform=self.image_transform,
            audio_transform=self.audio_transform,
        )
        self.val_dataset = Multimodal_Dataset(
            validation_image_data,
            validation_audio_data,
            validation_labels,
            image_transform=None,
            audio_transform=None,
        )
        self.test_dataset = Multimodal_Dataset(
            test_image_data,
            test_audio_data,
            test_labels,
            image_transform=None,
            audio_transform=None,
        )

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


class AttentionFusion(torch.nn.Module):
    """Cross-modal attention fusion"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )
        self.layer_norm = torch.nn.LayerNorm(feature_dim)

    def forward(self, image_features, audio_features):
        # Reshape features for attention (batch_size, seq_len=1, feature_dim)
        image_features = image_features.unsqueeze(1)
        audio_features = audio_features.unsqueeze(1)

        # Cross-modal attention: image attends to audio
        img_attended, _ = self.attention(image_features, audio_features, audio_features)
        # Cross-modal attention: audio attends to image
        aud_attended, _ = self.attention(audio_features, image_features, image_features)

        # Combine attended features
        combined = img_attended + aud_attended
        combined = self.layer_norm(combined)
        return combined.squeeze(1)


class GatedFusion(torch.nn.Module):
    """Gated fusion with learnable gates"""

    def __init__(self, feature_dim):
        super().__init__()
        self.image_gate = torch.nn.Linear(feature_dim, feature_dim)
        self.audio_gate = torch.nn.Linear(feature_dim, feature_dim)
        self.fusion_gate = torch.nn.Linear(feature_dim * 2, feature_dim * 2)

    def forward(self, image_features, audio_features):
        # Learn gates for each modality
        img_gate = torch.sigmoid(self.image_gate(image_features))
        aud_gate = torch.sigmoid(self.audio_gate(audio_features))

        # Apply gates
        gated_img = image_features * img_gate
        gated_aud = audio_features * aud_gate

        # Combine and apply fusion gate
        combined = torch.cat([gated_img, gated_aud], dim=1)
        fusion_gate = torch.sigmoid(self.fusion_gate(combined))

        return combined * fusion_gate


class BilinearFusion(torch.nn.Module):
    """Bilinear pooling fusion"""

    def __init__(self, img_dim, aud_dim, output_dim):
        super().__init__()
        self.bilinear = torch.nn.Bilinear(img_dim, aud_dim, output_dim)

    def forward(self, image_features, audio_features):
        return self.bilinear(image_features, audio_features)


class FactorizedBilinearFusion(torch.nn.Module):
    """Factorized bilinear pooling for efficiency"""

    def __init__(self, img_dim, aud_dim, hidden_dim, output_dim):
        super().__init__()
        self.img_proj = torch.nn.Linear(img_dim, hidden_dim)
        self.aud_proj = torch.nn.Linear(aud_dim, hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, image_features, audio_features):
        img_proj = self.img_proj(image_features)
        aud_proj = self.aud_proj(audio_features)
        # Element-wise multiplication
        fused = img_proj * aud_proj
        return self.output_proj(fused)


class Multimodal_Model(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        fusion_method="concatenation",  # New parameter
        image_inner_size=32,
        image_dropout=0.5,
        audio_inner_size=32,
        audio_dropout=0.5,
        local_dropout=0.5,
        learning_rate=0.001,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.image_inner_size = image_inner_size
        self.image_dropout = image_dropout
        self.audio_inner_size = audio_inner_size
        self.audio_dropout = audio_dropout
        self.local_dropout = local_dropout
        self.learning_rate = learning_rate

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(num_classes=self.num_classes, task="multiclass")
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.save_hyperparameters()

        # Initialize modality-specific models
        self.image_model = Conv2Net(
            num_classes=num_classes,
            inner_size=image_inner_size,
            drop_rate=image_dropout,
        )

        self.audio_model = Conv1Net(
            num_classes=num_classes,
            inner_size=audio_inner_size,
            drop_rate=audio_dropout,
        )

        # Feature dimensions
        img_feat_dim = self.image_model.inner_size * 4
        aud_feat_dim = self.audio_model.inner_size * 4

        # Initialize fusion module based on method
        if fusion_method == "concatenation":
            self.fusion = None
            classifier_input_dim = img_feat_dim + aud_feat_dim
        elif fusion_method == "addition":
            # Ensure features have same dimension
            assert img_feat_dim == aud_feat_dim, (
                "Features must have same dimension for addition"
            )
            self.fusion = None
            classifier_input_dim = img_feat_dim
        elif fusion_method == "multiplication":
            assert img_feat_dim == aud_feat_dim, (
                "Features must have same dimension for multiplication"
            )
            self.fusion = None
            classifier_input_dim = img_feat_dim
        elif fusion_method == "maximum":
            assert img_feat_dim == aud_feat_dim, (
                "Features must have same dimension for maximum"
            )
            self.fusion = None
            classifier_input_dim = img_feat_dim
        elif fusion_method == "attention":
            assert img_feat_dim == aud_feat_dim, (
                "Features must have same dimension for attention"
            )
            self.fusion = AttentionFusion(img_feat_dim)
            classifier_input_dim = img_feat_dim
        elif fusion_method == "gated":
            assert img_feat_dim == aud_feat_dim, (
                "Features must have same dimension for gated fusion"
            )
            self.fusion = GatedFusion(img_feat_dim)
            classifier_input_dim = img_feat_dim * 2
        elif fusion_method == "bilinear":
            output_dim = 128
            self.fusion = BilinearFusion(img_feat_dim, aud_feat_dim, output_dim)
            classifier_input_dim = output_dim
        elif fusion_method == "factorized_bilinear":
            hidden_dim = 64
            output_dim = 128
            self.fusion = FactorizedBilinearFusion(
                img_feat_dim, aud_feat_dim, hidden_dim, output_dim
            )
            classifier_input_dim = output_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Final classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.local_dropout),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        image, audio = x
        image_features = self.image_model.get_inner_features(image)
        audio_features = self.audio_model.get_inner_features(audio)

        # Apply fusion method
        if self.fusion_method == "concatenation":
            combined_features = torch.cat((image_features, audio_features), dim=1)
        elif self.fusion_method == "addition":
            combined_features = image_features + audio_features
        elif self.fusion_method == "multiplication":
            combined_features = image_features * audio_features
        elif self.fusion_method == "maximum":
            combined_features = torch.max(image_features, audio_features)
        elif self.fusion_method in [
            "attention",
            "gated",
            "bilinear",
            "factorized_bilinear",
        ]:
            combined_features = self.fusion(image_features, audio_features)

        logits = self.classifier(combined_features)
        return logits

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
        ax.set_title(f"Confusion Matrix - {self.fusion_method.title()} Fusion")

        # Log confusion matrix to wandb
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
