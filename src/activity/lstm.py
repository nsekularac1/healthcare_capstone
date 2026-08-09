"""Reusable PyTorch utilities for PAMAP2 activity recognition.

This module defines the LSTM model and reusable training/evaluation helpers
used by `04_activity_model.ipynb`.

Design goals
------------
- Keep the architecture compact and interpretable.
- Use a single-layer LSTM with hidden size 64 as the baseline.
- Support reproducible PyTorch training.
- Convert preprocessed NumPy arrays into DataLoaders.
- Track training and validation behavior across epochs.
- Evaluate with accuracy, weighted precision, recall, F1, and confusion matrix.
- Save and reload the trained model artifact.

The module expects input sequences shaped as:
    (samples, timesteps, features)

For this capstone, PAMAP2 preprocessing currently produces:
    timesteps = 100
    features = 19
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 15


@dataclass
class TrainingHistory:
    """Training and validation metrics collected across epochs."""

    train_loss: list[float]
    val_loss: list[float]
    train_accuracy: list[float]
    val_accuracy: list[float]


@dataclass
class EvaluationResult:
    """Container for activity-recognition evaluation outputs."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


class ActivityLSTM(nn.Module):
    """Single-direction LSTM classifier for PAMAP2 activity sequences."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        *,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be greater than zero.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")

        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        # PyTorch applies recurrent dropout only when num_layers > 1.
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""

        if x.ndim != 3:
            raise ValueError(
                "Expected input shaped (batch, timesteps, features)."
            )

        _, (hidden, _) = self.lstm(x)

        # Final hidden state from the last recurrent layer.
        final_hidden = hidden[-1]

        logits = self.classifier(final_hidden)
        return logits


def set_random_seed(
    seed: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Set common random seeds for reproducible training."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode improves reproducibility, though some GPU operations
    # may run more slowly.
    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except TypeError:
        # Compatibility fallback for older torch versions.
        pass


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def validate_sequence_arrays(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Validate preprocessed sequence and label arrays."""

    if X.ndim != 3:
        raise ValueError(
            "X must have shape (samples, timesteps, features)."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a one-dimensional class-label array."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if len(X) == 0:
        raise ValueError(
            "Sequence data cannot be empty."
        )

    if np.isnan(X).any():
        raise ValueError(
            "X contains NaN values. Complete preprocessing first."
        )


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    """Convert NumPy sequence arrays into a PyTorch DataLoader."""

    validate_sequence_arrays(
        X,
        y,
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_activity_model(
    *,
    input_size: int,
    num_classes: int,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = 0.0,
) -> ActivityLSTM:
    """Construct the PAMAP2 LSTM classifier."""

    return ActivityLSTM(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch."""

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(X_batch)
            loss = criterion(
                logits,
                y_batch,
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = y_batch.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == y_batch
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "DataLoader contained no samples."
        )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return (
        float(average_loss),
        float(accuracy),
    )


def train_activity_model(
    model: ActivityLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: torch.device | None = None,
    class_weights: torch.Tensor | None = None,
    verbose: bool = True,
) -> tuple[
    ActivityLSTM,
    TrainingHistory,
]:
    """Train the LSTM and record training/validation behavior."""

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if device is None:
        device = get_device()

    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(
            device
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_accuracy=[],
        val_accuracy=[],
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss, train_accuracy = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        history.train_loss.append(
            train_loss
        )

        history.val_loss.append(
            val_loss
        )

        history.train_accuracy.append(
            train_accuracy
        )

        history.val_accuracy.append(
            val_accuracy
        )

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

    return model, history


def predict_activity(
    model: ActivityLSTM,
    dataloader: DataLoader,
    *,
    device: torch.device | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Return true and predicted activity labels."""

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                y_batch.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    return (
        np.asarray(
            y_true,
            dtype=np.int64,
        ),
        np.asarray(
            y_pred,
            dtype=np.int64,
        ),
    )


def predict_activity_probabilities(
    model: ActivityLSTM,
    X: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return class-probability vectors for input sequences."""

    if X.ndim != 3:
        raise ValueError(
            "X must have shape (samples, timesteps, features)."
        )

    if np.isnan(X).any():
        raise ValueError(
            "X contains NaN values."
        )

    if device is None:
        device = get_device()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(device)
    model.eval()

    probability_batches = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            probability_batches.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(
        probability_batches,
        axis=0,
    )


def evaluate_activity_model(
    model: ActivityLSTM,
    dataloader: DataLoader,
    *,
    device: torch.device | None = None,
) -> EvaluationResult:
    """Evaluate the LSTM using weighted multiclass metrics."""

    y_true, y_pred = predict_activity(
        model,
        dataloader,
        device=device,
    )

    return EvaluationResult(
        accuracy=float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        precision=float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        confusion_matrix=confusion_matrix(
            y_true,
            y_pred,
        ),
        y_true=y_true,
        y_pred=y_pred,
    )


def compute_class_weights(
    y_train: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""

    y_train = np.asarray(
        y_train,
        dtype=np.int64,
    )

    counts = np.bincount(
        y_train,
        minlength=num_classes,
    ).astype(np.float64)

    if np.any(counts == 0):
        missing_classes = np.where(
            counts == 0
        )[0].tolist()

        raise ValueError(
            "Training data contains no examples for classes: "
            + ", ".join(
                map(str, missing_classes)
            )
        )

    total = counts.sum()

    weights = (
        total
        / (
            num_classes
            * counts
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def evaluation_to_dict(
    result: EvaluationResult,
) -> dict[str, Any]:
    """Convert evaluation results to a serializable dictionary."""

    return {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "confusion_matrix": (
            result.confusion_matrix.tolist()
        ),
        "samples": int(
            len(result.y_true)
        ),
    }


def save_activity_model(
    model: ActivityLSTM,
    path: str | Path,
    *,
    feature_names: list[str],
    class_names: list[str],
    window_size: int,
) -> Path:
    """Save the model state and reconstruction metadata."""

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "feature_names": feature_names,
        "class_names": class_names,
        "window_size": window_size,
    }

    torch.save(
        checkpoint,
        output_path,
    )

    return output_path


def load_activity_model(
    path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[
    ActivityLSTM,
    dict[str, Any],
]:
    """Load a saved activity model and its metadata."""

    model_path = Path(
        path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Activity model not found at: {model_path}"
        )

    if device is None:
        device = get_device()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "state_dict",
        "input_size",
        "hidden_size",
        "num_layers",
        "num_classes",
        "feature_names",
        "class_names",
        "window_size",
    }

    missing = required_keys.difference(
        checkpoint
    )

    if missing:
        raise ValueError(
            "Saved model checkpoint is missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    model = build_activity_model(
        input_size=int(
            checkpoint["input_size"]
        ),
        num_classes=int(
            checkpoint["num_classes"]
        ),
        hidden_size=int(
            checkpoint["hidden_size"]
        ),
        num_layers=int(
            checkpoint["num_layers"]
        ),
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(
        device
    )

    model.eval()

    metadata = {
        "feature_names": checkpoint[
            "feature_names"
        ],
        "class_names": checkpoint[
            "class_names"
        ],
        "window_size": checkpoint[
            "window_size"
        ],
        "input_size": checkpoint[
            "input_size"
        ],
        "hidden_size": checkpoint[
            "hidden_size"
        ],
        "num_layers": checkpoint[
            "num_layers"
        ],
        "num_classes": checkpoint[
            "num_classes"
        ],
    }

    return model, metadata


if __name__ == "__main__":
    # Lightweight architecture smoke test.
    set_random_seed()

    sample_model = build_activity_model(
        input_size=19,
        num_classes=12,
        hidden_size=64,
        num_layers=1,
    )

    sample_batch = torch.randn(
        8,
        100,
        19,
    )

    sample_logits = sample_model(
        sample_batch
    )

    print(
        "ActivityLSTM smoke test completed successfully."
    )

    print(
        "Input shape:",
        tuple(
            sample_batch.shape
        ),
    )

    print(
        "Output shape:",
        tuple(
            sample_logits.shape
        ),
    )

    print(
        "Device available:",
        get_device(),
    )
