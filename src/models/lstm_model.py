"""
LSTM Model Module
=================
PyTorch-based LSTM neural network for gold price prediction.

Features:
- Configurable regularization (dropout, weight decay, recurrent dropout)
- Optional batch/layer normalization
- Optional attention mechanism
- Residual connections support
"""

from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset


class Attention(nn.Module):
    """
    Self-attention mechanism for LSTM outputs (US-010).

    Allows the model to focus on important timesteps.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize attention layer.

        Args:
            hidden_size: Size of hidden states
            num_heads: Number of attention heads (must divide hidden_size)
            dropout: Attention dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            mask: Optional attention mask

        Returns:
            Attended output of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # Store for visualization

        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out(context)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return stored attention weights for visualization."""
        return self.attention_weights


class GoldLSTM(nn.Module):
    """
    LSTM neural network for gold price prediction.

    Architecture:
    - Multi-layer LSTM with configurable dropout
    - Optional layer/batch normalization
    - Optional self-attention mechanism
    - Optional residual connections
    - Fully connected layers for regression/classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        output_size: int = 1,
        bidirectional: bool = False,
        # New regularization options (US-008)
        recurrent_dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        # Attention options (US-010)
        use_attention: bool = False,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        # Residual connections (US-011)
        use_residual: bool = False,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability (default: 0.35 for better regularization)
            output_size: Number of outputs (1 for regression, 3 for classification)
            bidirectional: Whether to use bidirectional LSTM
            recurrent_dropout: Dropout on recurrent connections (0 to disable)
            use_layer_norm: Use LayerNorm after LSTM (US-009)
            use_batch_norm: Use BatchNorm after LSTM (US-009)
            use_attention: Add attention mechanism (US-010)
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout in attention layer
            use_residual: Add residual connections (US-011)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.use_attention = use_attention
        self.use_residual = use_residual

        # LSTM layers - build manually for recurrent dropout support
        lstm_dropout = dropout if num_layers > 1 and recurrent_dropout == 0 else 0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # Recurrent dropout (applied between timesteps)
        self.recurrent_dropout = nn.Dropout(recurrent_dropout) if recurrent_dropout > 0 else None

        # Output size after LSTM
        lstm_output_size = hidden_size * self.num_directions

        # Normalization layers (US-009)
        self.layer_norm = nn.LayerNorm(lstm_output_size) if use_layer_norm else None
        self.batch_norm = nn.BatchNorm1d(lstm_output_size) if use_batch_norm else None

        # Attention layer (US-010)
        self.attention = None
        if use_attention:
            self.attention = Attention(
                hidden_size=lstm_output_size,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
            )

        # Residual projection if dimensions don't match (US-011)
        self.residual_proj = None
        if use_residual and input_size != lstm_output_size:
            self.residual_proj = nn.Linear(input_size, lstm_output_size)

        # Fully connected layers with increased regularization
        fc_input_size = lstm_output_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state
            return_attention: If True, also return attention weights

        Returns:
            Output tensor of shape (batch, output_size)
            Optionally attention weights if return_attention=True
        """
        # Store input for residual connection
        residual_input = x

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # Apply recurrent dropout if enabled
        if self.recurrent_dropout is not None:
            lstm_out = self.recurrent_dropout(lstm_out)

        # Apply attention mechanism (US-010)
        if self.attention is not None:
            attended = self.attention(lstm_out)
            # Use attended output (average over sequence) instead of just last timestep
            lstm_out = attended

        # Residual connection (US-011)
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual_input)
            else:
                residual = residual_input
            lstm_out = lstm_out + residual

        # Take output from last timestep
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            last_output = torch.cat(
                (lstm_out[:, -1, :self.hidden_size], lstm_out[:, 0, self.hidden_size:]),
                dim=1,
            )
        else:
            last_output = lstm_out[:, -1, :]

        # Apply normalization (US-009)
        if self.layer_norm is not None:
            last_output = self.layer_norm(last_output)
        if self.batch_norm is not None:
            last_output = self.batch_norm(last_output)

        # Pass through fully connected layers
        output = self.fc(last_output)

        if return_attention and self.attention is not None:
            return output, self.attention.get_attention_weights()

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass."""
        if self.attention is not None:
            return self.attention.get_attention_weights()
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make prediction on numpy array.

        Args:
            x: Input array of shape (batch, seq_len, input_size)

        Returns:
            Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()
            output = self.forward(x_tensor)
            return output.cpu().numpy()


class DirectionWeightedMSE(nn.Module):
    """
    Direction-weighted MSE loss (US-013).

    Penalizes wrong direction predictions more heavily than magnitude errors.
    """

    def __init__(self, direction_weight: float = 2.0):
        super().__init__()
        self.direction_weight = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Base MSE
        mse = (pred - target) ** 2

        # Direction penalty: higher weight when predicted direction is wrong
        # Assumes pred and target are differences from previous value
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)
        direction_match = (pred_direction == target_direction).float()

        # Apply higher weight to wrong direction predictions
        weights = torch.where(
            direction_match == 1,
            torch.ones_like(mse),
            torch.full_like(mse, self.direction_weight),
        )

        return (weights * mse).mean()


class QuantileLoss(nn.Module):
    """
    Quantile loss for prediction intervals (US-013).
    """

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - pred
        return torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors,
        ).mean()


class LSTMTrainer:
    """
    Trainer class for LSTM model.

    Handles training loop, validation, early stopping, and model saving.

    Features:
    - Learning rate warmup (US-012)
    - Alternative loss functions (US-013)
    - Gradient accumulation (US-014)
    - Weight decay for L2 regularization
    """

    def __init__(
        self,
        model: GoldLSTM,
        learning_rate: float = 0.001,
        device: str = "auto",
        # Regularization (US-008)
        weight_decay: float = 1e-4,
        # Loss function (US-013)
        loss_type: str = "mse",
        direction_weight: float = 2.0,
        # Warmup (US-012)
        warmup_steps: int = 100,
        # Gradient accumulation (US-014)
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize trainer.

        Args:
            model: GoldLSTM model instance
            learning_rate: Initial learning rate
            device: Device to train on ('auto', 'cuda', or 'cpu')
            weight_decay: L2 regularization weight (default: 1e-4)
            loss_type: Loss function type ('mse', 'huber', 'direction_mse', 'quantile')
            direction_weight: Weight for wrong direction in direction_mse loss
            warmup_steps: Number of warmup steps for learning rate
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_step = 0

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer with weight decay (L2 regularization) (US-008)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function selection (US-013)
        self.loss_type = loss_type
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=1.0)
        elif loss_type == "direction_mse":
            self.criterion = DirectionWeightedMSE(direction_weight=direction_weight)
        elif loss_type == "quantile":
            self.criterion = QuantileLoss(quantile=0.5)
        else:
            logger.warning(f"Unknown loss type '{loss_type}', using MSE")
            self.criterion = nn.MSELoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def _get_lr_with_warmup(self) -> float:
        """Calculate learning rate with warmup (US-012)."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.current_step / self.warmup_steps)
        return self.learning_rate

    def _set_lr(self, lr: float):
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: int = 10,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Training batch size
            early_stopping: Patience for early stopping
            save_path: Path to save best model

        Returns:
            Training history dict
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.reshape(-1, 1)),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val.reshape(-1, 1)),
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        patience_counter = 0
        self.current_step = 0

        # Effective batch size with gradient accumulation (US-014)
        effective_batch_size = batch_size * self.gradient_accumulation_steps
        logger.info(
            f"Training on {self.device} for {epochs} epochs "
            f"(effective batch size: {effective_batch_size}, warmup: {self.warmup_steps} steps)"
        )

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            accumulated_loss = 0.0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Apply learning rate warmup (US-012)
                if self.current_step < self.warmup_steps:
                    warmup_lr = self._get_lr_with_warmup()
                    self._set_lr(warmup_lr)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Scale loss for gradient accumulation (US-014)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                # Update weights after accumulating gradients (US-014)
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += accumulated_loss
                    accumulated_loss = 0.0
                    self.current_step += 1

            # Handle remaining gradients
            if accumulated_loss > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += accumulated_loss

            train_loss /= (len(train_loader) // self.gradient_accumulation_steps + 1)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)

                # Only step scheduler after warmup
                if self.current_step >= self.warmup_steps:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - LR: {current_lr:.6f}"
                if val_loss:
                    msg += f" - Val Loss: {val_loss:.6f}"
                logger.info(msg)

            # Early stopping
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": epoch + 1,
            "final_lr": self.optimizer.param_groups[0]["lr"],
        }

    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return loss."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def save_model(self, path: str) -> None:
        """Save model state to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": {
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "bidirectional": self.model.bidirectional,
            },
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, device: str = "auto") -> Tuple[GoldLSTM, "LSTMTrainer"]:
        """
        Load model from file.

        Args:
            path: Path to model file
            device: Device to load model on

        Returns:
            Tuple of (model, trainer)
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["model_config"]

        model = GoldLSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            bidirectional=config["bidirectional"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.train_losses = checkpoint.get("train_losses", [])
        trainer.val_losses = checkpoint.get("val_losses", [])
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Model loaded from {path}")

        return model, trainer
