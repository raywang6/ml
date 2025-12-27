"""
Domain Adversarial Neural Network (DANN) for Session-Invariant Crypto Prediction

This module implements domain adaptation using adversarial training to learn
features that are invariant to trading sessions (US hours vs non-US hours).

Architecture:
    Input → Feature Extractor → [Predictor Head] → Return Prediction
                              ↘ [Domain Classifier with GRL] → Session Classification

The Gradient Reversal Layer (GRL) reverses gradients during backpropagation,
forcing the feature extractor to learn session-invariant representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Tuple, Optional, Dict, List
import joblib
import os


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for domain adversarial training.

    Forward pass: identity function
    Backward pass: reverses gradient and scales by lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor network.

    Learns representations that should be:
    1. Predictive of returns
    2. Invariant to trading session (via adversarial training)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.network(x)


class PredictorHead(nn.Module):
    """
    Prediction head for return forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DomainClassifier(nn.Module):
    """
    Domain classifier for session classification.

    Classifies whether input is from US trading hours or non-US hours.
    Used with GRL to learn session-invariant features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        self.grl = GradientReversalLayer()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )

    def forward(self, x, lambda_: float = 1.0):
        self.grl.set_lambda(lambda_)
        x = self.grl(x)
        return self.network(x)


class DANN(nn.Module):
    """
    Domain Adversarial Neural Network for Session-Invariant Prediction.

    Combines:
    - Feature extractor (shared)
    - Predictor head (return prediction)
    - Domain classifier (session classification with gradient reversal)

    Training objective:
        L_total = L_pred - λ * L_domain

    Where the negative sign on L_domain comes from the GRL.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        predictor_hidden: int = 32,
        domain_hidden: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        feature_dim = self.feature_extractor.output_dim

        self.predictor = PredictorHead(
            input_dim=feature_dim,
            hidden_dim=predictor_hidden,
            output_dim=1,
            dropout=dropout
        )

        self.domain_classifier = DomainClassifier(
            input_dim=feature_dim,
            hidden_dim=domain_hidden,
            dropout=dropout
        )

        self.input_dim = input_dim
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        lambda_: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]
            lambda_: Domain adaptation weight (increases during training)

        Returns:
            predictions: Return predictions [batch_size, 1]
            domain_logits: Domain classification logits [batch_size, 1]
            features: Extracted features [batch_size, feature_dim]
        """
        features = self.feature_extractor(x)
        predictions = self.predictor(features)
        domain_logits = self.domain_classifier(features, lambda_)

        return predictions, domain_logits, features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict returns only (for inference)."""
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            predictions = self.predictor(features)
        return predictions


class SignSensitiveLoss(nn.Module):
    """
    Custom loss that penalizes sign mismatches more heavily.
    Similar to the LightGBM objective in lgbmr.py.
    """

    def __init__(self, sign_penalty: float = 3.0):
        super().__init__()
        self.sign_penalty = sign_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Base MSE loss
        mse = (predictions - targets) ** 2

        # Sign mismatch penalty
        sign_mismatch = (torch.sign(predictions) != torch.sign(targets)).float()
        penalty_multiplier = 1.0 + self.sign_penalty * sign_mismatch

        weighted_loss = mse * penalty_multiplier

        if sample_weights is not None:
            weighted_loss = weighted_loss * sample_weights.unsqueeze(1)

        return weighted_loss.mean()


class DANNWrapper:
    """
    Sklearn-like wrapper for DANN model.

    Provides fit/predict interface compatible with the existing training pipeline.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = [256, 128, 64],
        predictor_hidden: int = 32,
        domain_hidden: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        n_epochs: int = 100,
        patience: int = 10,
        lambda_schedule: str = 'linear',  # 'linear', 'exp', 'constant'
        lambda_max: float = 1.0,
        sign_penalty: float = 3.0,
        domain_loss_weight: float = 0.1,
        device: str = 'auto',
        verbose: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.predictor_hidden = predictor_hidden
        self.domain_hidden = domain_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.lambda_schedule = lambda_schedule
        self.lambda_max = lambda_max
        self.sign_penalty = sign_penalty
        self.domain_loss_weight = domain_loss_weight
        self.verbose = verbose

        # Device setup
        if device == 'auto':
            self.device = torch.device('mps' if torch.backends.mps.is_available()
                                       else 'cuda' if torch.cuda.is_available()
                                       else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.model_namecard = {}

    def _compute_lambda(self, epoch: int, n_epochs: int) -> float:
        """Compute domain adaptation weight based on schedule."""
        p = epoch / n_epochs

        if self.lambda_schedule == 'linear':
            return self.lambda_max * p
        elif self.lambda_schedule == 'exp':
            # Gradual increase: λ = 2/(1+exp(-γp)) - 1
            gamma = 10
            return self.lambda_max * (2 / (1 + np.exp(-gamma * p)) - 1)
        else:  # constant
            return self.lambda_max

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domain_labels: np.ndarray,
        validation_data: Optional[Tuple] = None,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit the DANN model.

        Args:
            X: Features [n_samples, n_features]
            y: Target returns [n_samples,]
            domain_labels: Session labels (1=US hours, 0=non-US) [n_samples,]
            validation_data: Tuple of (X_val, y_val, domain_labels_val)
            sample_weight: Sample weights [n_samples,]
        """
        # Infer input dimension
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        # Initialize model
        self.model = DANN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            predictor_hidden=self.predictor_hidden,
            domain_hidden=self.domain_hidden,
            dropout=self.dropout
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        domain_tensor = torch.FloatTensor(domain_labels).reshape(-1, 1).to(self.device)

        if sample_weight is not None:
            sw_tensor = torch.FloatTensor(sample_weight).to(self.device)
        else:
            sw_tensor = None

        # Validation data
        if validation_data is not None:
            X_val, y_val, domain_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            domain_val_tensor = torch.FloatTensor(domain_val).reshape(-1, 1).to(self.device)

        # Loss functions
        pred_criterion = SignSensitiveLoss(sign_penalty=self.sign_penalty)
        domain_criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=self.verbose
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        n_batches = (len(X) + self.batch_size - 1) // self.batch_size

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_pred_loss = 0
            epoch_domain_loss = 0

            # Compute lambda for this epoch
            lambda_ = self._compute_lambda(epoch, self.n_epochs)

            # Shuffle data
            indices = torch.randperm(len(X))

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(X))
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                domain_batch = domain_tensor[batch_indices]

                if sw_tensor is not None:
                    sw_batch = sw_tensor[batch_indices]
                else:
                    sw_batch = None

                optimizer.zero_grad()

                # Forward pass
                predictions, domain_logits, _ = self.model(X_batch, lambda_)

                # Compute losses
                pred_loss = pred_criterion(predictions, y_batch, sw_batch)
                domain_loss = domain_criterion(domain_logits, domain_batch)

                # Total loss (GRL already handles the negative sign for domain loss)
                total_loss = pred_loss + self.domain_loss_weight * domain_loss

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_pred_loss += pred_loss.item()
                epoch_domain_loss += domain_loss.item()

            # Validation
            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred, val_domain, _ = self.model(X_val_tensor, lambda_)
                    val_pred_loss = pred_criterion(val_pred, y_val_tensor)
                    val_domain_loss = domain_criterion(val_domain, domain_val_tensor)
                    val_total_loss = val_pred_loss.item()

                # Learning rate scheduling
                scheduler.step(val_total_loss)

                # Early stopping
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if self.verbose and epoch % 10 == 0:
                    # Compute domain accuracy
                    domain_acc = ((torch.sigmoid(val_domain) > 0.5).float() == domain_val_tensor).float().mean()
                    print(f"Epoch {epoch}: pred_loss={epoch_pred_loss/n_batches:.4f}, "
                          f"domain_loss={epoch_domain_loss/n_batches:.4f}, "
                          f"val_loss={val_total_loss:.4f}, domain_acc={domain_acc:.2%}, "
                          f"λ={lambda_:.3f}")

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model_namecard['epoch'] = epoch
        self.model_namecard['best_val_loss'] = best_val_loss

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model.predict(X_tensor)

        return predictions.cpu().numpy().flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute RMSE on (X, y)."""
        predictions = self.predict(X)
        rmse = np.sqrt(np.mean((predictions - y.flatten()) ** 2))
        return rmse, rmse

    def save(self, path: str):
        """Save model."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        # Save model state
        torch.save(self.model.state_dict(), path + '_model.pt')

        # Save config
        config = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'predictor_hidden': self.predictor_hidden,
            'domain_hidden': self.domain_hidden,
            'dropout': self.dropout,
            'model_namecard': self.model_namecard
        }
        joblib.dump(config, path + '_config.pkl')

    @classmethod
    def load(cls, path: str) -> 'DANNWrapper':
        """Load model."""
        config = joblib.load(path + '_config.pkl')

        wrapper = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            predictor_hidden=config['predictor_hidden'],
            domain_hidden=config['domain_hidden'],
            dropout=config['dropout']
        )

        wrapper.model = DANN(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            predictor_hidden=config['predictor_hidden'],
            domain_hidden=config['domain_hidden'],
            dropout=config['dropout']
        ).to(wrapper.device)

        wrapper.model.load_state_dict(torch.load(path + '_model.pt', map_location=wrapper.device))
        wrapper.model_namecard = config['model_namecard']

        return wrapper


def prepare_dann(input_dim: int = None) -> Tuple[type, Dict]:
    """
    Prepare DANN model class and default parameters.

    Similar interface to prepare_lgbm() for easy swapping.
    """
    default_params = {
        'input_dim': input_dim,
        'hidden_dims': [256, 128, 64],
        'predictor_hidden': 32,
        'domain_hidden': 64,
        'dropout': 0.3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 512,
        'n_epochs': 100,
        'patience': 15,
        'lambda_schedule': 'exp',
        'lambda_max': 1.0,
        'sign_penalty': 3.0,
        'domain_loss_weight': 0.1,
        'verbose': True
    }

    return DANNWrapper, default_params
