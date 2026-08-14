import argparse
import copy
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MULTI-SCALE CNN
# ============================================================

class MultiScaleCNN(nn.Module):

    def __init__(self, in_channels=1, channels=32):
        super().__init__()

        def make_branch(kernel_size):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )

        self.branch3 = make_branch(3)
        self.branch5 = make_branch(5)
        self.branch7 = make_branch(7)

        self.pool = nn.MaxPool2d(2)

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch7 = self.branch7(x)

        x = torch.cat(
            [branch3, branch5, branch7],
            dim=1
        )

        x = self.pool(x)
        x = self.refine(x)

        return x


# ============================================================
# RESIDUAL ATTENTION FUSION
# ============================================================

class ResidualAttentionFusion(nn.Module):

    def __init__(self, channels=128):

        super().__init__()

        hidden = max(channels // 8, 8)

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.Sigmoid()
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        residual = x

        # Channel attention
        channel_weights = self.channel_attention(x)

        x = x * channel_weights

        # Spatial attention
        average_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        maximum_map = torch.max(
            x,
            dim=1,
            keepdim=True
        ).values

        spatial_input = torch.cat(
            [average_map, maximum_map],
            dim=1
        )

        spatial_weights = self.spatial_attention(
            spatial_input
        )

        x = x * spatial_weights

        x = self.project(x)

        # Residual connection
        x = x + residual

        return x


# ============================================================
# TRANSFORMER ENCODER
# ============================================================

class TransformerContext(nn.Module):

    def __init__(
        self,
        channels=128,
        heads=8,
        layers=2
    ):

        super().__init__()

        # Reduce the CNN feature map before the Transformer.
        #
        # Without pooling:
        # 112 x 112 = 12,544 tokens
        #
        # With 14 x 14 pooling:
        # 14 x 14 = 196 tokens
        #
        # This greatly reduces Transformer memory usage.

        self.spatial_pool = nn.AdaptiveAvgPool2d(
            (14, 14)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.norm = nn.LayerNorm(
            channels
        )

    def forward(self, feature_map):

        # Reduce spatial resolution
        feature_map = self.spatial_pool(
            feature_map
        )

        batch_size, channels, height, width = (
            feature_map.shape
        )

        # Convert feature map into Transformer tokens
        #
        # [B, C, H, W]
        #      ↓
        # [B, C, H*W]
        #      ↓
        # [B, H*W, C]

        tokens = feature_map.flatten(
            2
        )

        tokens = tokens.transpose(
            1,
            2
        )

        # Transformer encoder
        tokens = self.encoder(
            tokens
        )

        tokens = self.norm(
            tokens
        )

        # Global Transformer representation
        representation = tokens.mean(
            dim=1
        )

        return representation

# ============================================================
# ADAPTIVE FUSION
# ============================================================

class AdaptiveFusion(nn.Module):

    def __init__(self, channels=128):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(
                channels * 2,
                channels
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                channels,
                1
            ),

            nn.Sigmoid()
        )

        self.projection = nn.Sequential(

            nn.Linear(
                channels,
                channels
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.25)
        )

    def forward(
        self,
        cnn_vector,
        transformer_vector
    ):

        combined = torch.cat(
            [
                cnn_vector,
                transformer_vector
            ],
            dim=1
        )

        gate = self.gate(combined)

        fused = (
            gate * cnn_vector
            +
            (1 - gate) * transformer_vector
        )

        fused = self.projection(fused)

        return fused, gate


# ============================================================
# HYBRID CNN-TRANSFORMER MODEL
# ============================================================

class HybridXRayNet(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # Multi-scale CNN
        self.multiscale = MultiScaleCNN()

        # Attention
        self.attention = ResidualAttentionFusion(
            channels=128
        )

        # Transformer
        self.transformer = TransformerContext(
            channels=128,
            heads=8,
            layers=2
        )

        # CNN global representation
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Adaptive fusion
        self.fusion = AdaptiveFusion(
            channels=128
        )

        # Classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.35),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(
        self,
        x,
        return_gate=False
    ):

        # CNN feature extraction
        features = self.multiscale(x)

        # Attention
        attended_features = self.attention(
            features
        )

        # Transformer representation
        transformer_vector = self.transformer(
            attended_features
        )

        # CNN representation
        cnn_vector = self.cnn_pool(
            attended_features
        ).flatten(1)

        # Adaptive fusion
        fused, gate = self.fusion(
            cnn_vector,
            transformer_vector
        )

        # Classification
        logits = self.classifier(
            fused
        )

        if return_gate:
            return logits, gate

        return logits


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_transforms(image_size=224):

    train_transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            7
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    validation_transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    return (
        train_transform,
        validation_transform
    )


# ============================================================
# AUTOMATIC VALIDATION SPLIT
# ============================================================

def create_validation_split(
    data_dir,
    validation_ratio=0.10,
    seed=42
):

    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    validation_dir = data_dir / "val"

    # If validation already exists, don't recreate it
    if validation_dir.exists():

        print("Validation directory already exists.")

        return

    if not train_dir.exists():

        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    random_generator = random.Random(seed)

    class_directories = [
        directory
        for directory in train_dir.iterdir()
        if directory.is_dir()
    ]

    if len(class_directories) < 2:

        raise ValueError(
            "Expected at least two class folders."
        )

    print("\nCreating validation set...")
    print(
        f"Validation ratio: {validation_ratio * 100:.0f}%"
    )

    for class_directory in class_directories:

        images = [

            image
            for image in class_directory.iterdir()

            if image.is_file()
            and image.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            }
        ]

        random_generator.shuffle(images)

        validation_count = max(
            1,
            int(
                len(images)
                * validation_ratio
            )
        )

        validation_images = images[
            :validation_count
        ]

        destination_directory = (
            validation_dir
            / class_directory.name
        )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        for image in validation_images:

            destination = (
                destination_directory
                / image.name
            )

            # COPY, DON'T MOVE
            shutil.copy2(
                image,
                destination
            )

        print(
            f"{class_directory.name}: "
            f"{len(validation_images)} validation images"
        )

    print(
        f"\nValidation set created at: "
        f"{validation_dir}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders(
    data_dir,
    batch_size,
    image_size,
    num_workers,
    seed
):

    data_dir = Path(data_dir)

    # Automatically create validation set
    create_validation_split(
        data_dir,
        validation_ratio=0.10,
        seed=seed
    )

    train_transform, validation_transform = (
        create_transforms(image_size)
    )

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=validation_transform
    )

    test_dataset = None

    if (data_dir / "test").exists():

        test_dataset = datasets.ImageFolder(
            data_dir / "test",
            transform=validation_transform
        )

    if (
        train_dataset.class_to_idx
        != validation_dataset.class_to_idx
    ):

        raise ValueError(
            "Class mappings between train and val differ."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = None

    if test_dataset is not None:

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    )


# ============================================================
# TRAINING / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    true_labels = []
    predicted_labels = []
    probabilities = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        class_probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )[:, 1]
        )

        true_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        predicted_labels.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        probabilities.extend(
            class_probabilities.detach()
            .cpu()
            .numpy()
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    return {

        "loss": average_loss,

        "accuracy": accuracy,

        "recall": recall,

        "f1": f1,

        "y_true": np.array(
            true_labels
        ),

        "y_pred": np.array(
            predicted_labels
        ),

        "y_prob": np.array(
            probabilities
        )
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    class_names,
    device,
    output_directory
):

    result = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device
    )

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    else:

        roc_auc = float("nan")

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    metrics = {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        ),

        "confusion_matrix":
            matrix.tolist(),

        "classification_report":
            report
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_directory
        / "test_metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    # -------------------------
    # Confusion matrix plot
    # -------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "HybridXRayNet Confusion Matrix"
    )

    plt.colorbar()

    ticks = np.arange(
        len(class_names)
    )

    plt.xticks(
        ticks,
        class_names,
        rotation=45
    )

    plt.yticks(
        ticks,
        class_names
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.tight_layout()

    plt.savefig(
        output_directory
        / "confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # ROC curve
    # -------------------------

    if len(np.unique(y_true)) == 2:

        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                y_true,
                y_prob
            )
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC-AUC = {roc_auc:.3f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_directory
            / "roc_curve.png",
            dpi=200
        )

        plt.close()

    return metrics


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model(args):

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and not args.cpu
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(
        args.data_dir,
        args.batch_size,
        args.image_size,
        args.num_workers,
        args.seed
    )

    print("\n==============================")
    print("DATASET")
    print("==============================")

    print(
        f"Classes: {train_dataset.classes}"
    )

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    if test_dataset is not None:

        print(
            f"Test images: "
            f"{len(test_dataset)}"
        )

    print(
        f"Class mapping: "
        f"{train_dataset.class_to_idx}"
    )

    # -------------------------
    # Model
    # -------------------------

    model = HybridXRayNet(
        num_classes=len(
            train_dataset.classes
        )
    ).to(device)

    # -------------------------
    # Class weights
    # -------------------------

    class_counts = np.bincount(
        train_dataset.targets
    )

    class_weights = (
        len(train_dataset)
        /
        (
            len(class_counts)
            * class_counts
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    # -------------------------
    # Training
    # -------------------------

    best_f1 = -1

    best_model_state = None

    patience_counter = 0

    history = {

        "train_loss": [],
        "validation_loss": [],

        "train_accuracy": [],
        "validation_accuracy": [],

        "train_recall": [],
        "validation_recall": [],

        "train_f1": [],
        "validation_f1": []
    }

    output_directory = Path(
        args.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        validation_result = run_epoch(
            model,
            validation_loader,
            criterion,
            device
        )

        scheduler.step(
            validation_result["f1"]
        )

        history[
            "train_loss"
        ].append(
            float(
                train_result["loss"]
            )
        )

        history[
            "validation_loss"
        ].append(
            float(
                validation_result["loss"]
            )
        )

        history[
            "train_accuracy"
        ].append(
            float(
                train_result["accuracy"]
            )
        )

        history[
            "validation_accuracy"
        ].append(
            float(
                validation_result["accuracy"]
            )
        )

        history[
            "train_recall"
        ].append(
            float(
                train_result["recall"]
            )
        )

        history[
            "validation_recall"
        ].append(
            float(
                validation_result["recall"]
            )
        )

        history[
            "train_f1"
        ].append(
            float(
                train_result["f1"]
            )
        )

        history[
            "validation_f1"
        ].append(
            float(
                validation_result["f1"]
            )
        )

        print(
            f"\nEpoch "
            f"{epoch}/{args.epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_result['loss']:.4f}"
        )

        print(
            f"Train F1: "
            f"{train_result['f1']:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_result['loss']:.4f}"
        )

        print(
            f"Validation F1: "
            f"{validation_result['f1']:.4f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if (
            validation_result["f1"]
            > best_f1
        ):

            best_f1 = (
                validation_result["f1"]
            )

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            torch.save(
                {
                    "model_state_dict":
                        best_model_state,

                    "class_to_idx":
                        train_dataset.class_to_idx,

                    "image_size":
                        args.image_size
                },

                output_directory
                / "best_model.pth"
            )

            patience_counter = 0

            print(
                "✓ Best model saved."
            )

        else:

            patience_counter += 1

        # -------------------------
        # Early stopping
        # -------------------------

        if (
            patience_counter
            >= args.early_stopping_patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # -------------------------
    # Save history
    # -------------------------

    with open(
        output_directory
        / "training_history.json",
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # -------------------------
    # Load best model
    # -------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    # -------------------------
    # Plot training curves
    # -------------------------

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_directory
        / "loss_curve.png",
        dpi=200
    )

    plt.close()

    # -------------------------
    # Final test evaluation
    # -------------------------

    if test_loader is not None:

        evaluate_model(
            model,
            test_loader,
            train_dataset.classes,
            device,
            output_directory
        )

    else:

        print(
            "\nWARNING: No test directory found."
        )

    # -------------------------
    # Save configuration
    # -------------------------

    configuration = {

        "architecture":
            "HybridXRayNet",

        "image_size":
            args.image_size,

        "classes":
            train_dataset.classes,

        "class_to_idx":
            train_dataset.class_to_idx,

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "validation_ratio":
            0.10
    }

    with open(
        output_directory
        / "model_config.json",
        "w"
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4
        )

    print(
        "\n=============================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Model saved to: "
        f"{output_directory / 'best_model.pth'}"
    )


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=
        "Train HybridXRayNet"
    )

    parser.add_argument(
        "--data-dir",
        default="chest_xray"
    )

    parser.add_argument(
        "--output-directory",
        default="outputs"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--cpu",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    arguments = parse_arguments()

    train_model(
        arguments
    )