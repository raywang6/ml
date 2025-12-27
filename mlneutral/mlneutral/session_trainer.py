"""
Session-Based Training Utilities

This module provides utilities for training models that are aware of
trading sessions (US hours vs non-US hours).

Two approaches are supported:
1. Domain Adaptation (DANN): Single model with adversarial loss for session invariance
2. Separate Models: Train distinct models for each session

US Trading Hours: UTC 13:00 - 21:00 (NYSE: 9:30 AM - 4:00 PM ET)
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
import os
import pickle
import gc

from .dann import DANNWrapper, prepare_dann
from .lgbmr import LGBMRegressor, prepare_lgbm


def add_session_labels(
    df: pl.DataFrame,
    datetime_col: str = 'datetime',
    us_start_hour: int = 13,  # UTC
    us_end_hour: int = 21,    # UTC
    session_col: str = 'is_us_session'
) -> pl.DataFrame:
    """
    Add trading session labels to dataframe.

    US Trading Hours: UTC 13:00 - 21:00
    (Corresponds roughly to NYSE hours: 9:30 AM - 4:00 PM ET)

    Args:
        df: Input DataFrame with datetime column
        datetime_col: Name of datetime column
        us_start_hour: Start of US session in UTC (default: 13)
        us_end_hour: End of US session in UTC (default: 21)
        session_col: Name for the session label column

    Returns:
        DataFrame with session label column added
    """
    df = df.with_columns(
        pl.col(datetime_col).dt.hour().alias('_hour')
    ).with_columns(
        ((pl.col('_hour') >= us_start_hour) & (pl.col('_hour') < us_end_hour))
        .cast(pl.Int8)
        .alias(session_col)
    ).drop('_hour')

    return df


def get_session_stats(
    df: pl.DataFrame,
    session_col: str = 'is_us_session'
) -> Dict:
    """Get statistics about session distribution."""
    total = df.height
    us_count = df.filter(pl.col(session_col) == 1).height
    non_us_count = total - us_count

    return {
        'total': total,
        'us_session': us_count,
        'non_us_session': non_us_count,
        'us_ratio': us_count / total if total > 0 else 0,
        'non_us_ratio': non_us_count / total if total > 0 else 0
    }


def train_dann_session(
    training_set: pl.DataFrame,
    target: str,
    features: List[str],
    save_path: str,
    config: Dict,
    name: str = '',
    session_col: str = 'is_us_session',
    use_sw: bool = False,
    dann_params: Optional[Dict] = None
) -> DANNWrapper:
    """
    Train a DANN model with domain adaptation for session invariance.

    Args:
        training_set: Training data with features, target, and session labels
        target: Target column name
        features: List of feature column names
        save_path: Directory to save model
        config: Training configuration
        name: Model name prefix
        session_col: Session label column name
        use_sw: Whether to use sample weights
        dann_params: Override default DANN parameters

    Returns:
        Trained DANNWrapper model
    """
    buff = config['training']['overlap_buff']

    # Prepare data
    training_set = training_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    training_set = training_set.fill_nan(None).drop_nulls(target)

    X_ = training_set.select(features).to_numpy()[:-buff]
    y_ = training_set.select(target).to_numpy().flatten()[:-buff]
    domain_ = training_set.select(session_col).to_numpy().flatten()[:-buff]

    N = len(X_)

    if use_sw:
        sw_ = training_set.select(f"sw_{target}").to_numpy().flatten()[:-buff]
    else:
        sw_ = None

    # Split dataset
    if N >= config['training']['min_datasize_thres']:
        ntrain = int(N - config['training']['min_datasize_thres'] / 2)
        nvalid = int(config['training']['min_datasize_thres'] / 12)
    else:
        ntrain = int(N / 2)
        nvalid = int(N / 12)

    # Get session stats
    stats = {
        'total': N,
        'us_session': (domain_ == 1).sum(),
        'non_us_session': (domain_ == 0).sum()
    }
    print(f"[DANN] Session stats: {stats}")

    # Prepare model
    _, default_params = prepare_dann(input_dim=len(features))
    if dann_params:
        default_params.update(dann_params)

    # Use last validation fold for final model
    idvalid = 5
    train_X = X_[:ntrain + nvalid * idvalid - buff]
    train_y = y_[:ntrain + nvalid * idvalid - buff]
    train_domain = domain_[:ntrain + nvalid * idvalid - buff]
    valid_X = X_[ntrain + nvalid * idvalid:ntrain + nvalid * (idvalid + 1)]
    valid_y = y_[ntrain + nvalid * idvalid:ntrain + nvalid * (idvalid + 1)]
    valid_domain = domain_[ntrain + nvalid * idvalid:ntrain + nvalid * (idvalid + 1)]

    if use_sw:
        sw = sw_[:ntrain + nvalid * idvalid - buff]
    else:
        sw = None

    print(f"[DANN] Training with {len(train_X)} samples, validating with {len(valid_X)} samples")

    # Train model
    model = DANNWrapper(**default_params)
    model.fit(
        train_X, train_y, train_domain,
        validation_data=(valid_X, valid_y, valid_domain),
        sample_weight=sw
    )

    # Save model
    model_path = os.path.join(save_path, f'{name}_{target}_dann')
    model.save(model_path)
    print(f"[DANN] Model saved to {model_path}")

    return model


def train_separate_sessions(
    training_set: pl.DataFrame,
    target: str,
    features: List[str],
    save_path: str,
    config: Dict,
    name: str = '',
    session_col: str = 'is_us_session',
    use_sw: bool = False,
    model_type: Literal['lgbm', 'dann'] = 'lgbm'
) -> Tuple:
    """
    Train separate models for US and non-US sessions.

    Args:
        training_set: Training data with features, target, and session labels
        target: Target column name
        features: List of feature column names
        save_path: Directory to save models
        config: Training configuration
        name: Model name prefix
        session_col: Session label column name
        use_sw: Whether to use sample weights
        model_type: 'lgbm' or 'dann'

    Returns:
        Tuple of (us_model, non_us_model)
    """
    buff = config['training']['overlap_buff']

    # Split by session
    us_data = training_set.filter(pl.col(session_col) == 1)
    non_us_data = training_set.filter(pl.col(session_col) == 0)

    print(f"[Separate] US session samples: {us_data.height}")
    print(f"[Separate] Non-US session samples: {non_us_data.height}")

    def train_session_model(data, session_name):
        data = data.with_columns(
            pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
        )
        data = data.fill_nan(None).drop_nulls(target)

        X_ = data.select(features).to_numpy()[:-buff] if data.height > buff else data.select(features).to_numpy()
        y_ = data.select(target).to_numpy().flatten()[:-buff] if data.height > buff else data.select(target).to_numpy().flatten()

        N = len(X_)
        if N < 1000:
            print(f"[Warning] Insufficient data for {session_name}: {N} samples")
            return None

        # Split dataset
        ntrain = max(int(N * 0.7), 100)
        nvalid = max(int(N * 0.15), 50)

        train_X = X_[:ntrain]
        train_y = y_[:ntrain]
        valid_X = X_[ntrain:ntrain + nvalid]
        valid_y = y_[ntrain:ntrain + nvalid]

        if model_type == 'lgbm':
            model_class, params = prepare_lgbm()
            params['learning_rate'] = 1e-4
            model = model_class(**params)
            model.fit(train_X, train_y, validation_data=(valid_X, valid_y))
        else:
            _, params = prepare_dann(input_dim=len(features))
            # For separate models, no domain labels needed
            model = DANNWrapper(**params)
            # Train without domain adaptation (all same domain)
            domain_train = np.zeros(len(train_X))
            domain_valid = np.zeros(len(valid_X))
            model.fit(train_X, train_y, domain_train,
                     validation_data=(valid_X, valid_y, domain_valid))

        # Save
        model_path = os.path.join(save_path, f'{name}_{target}_{session_name}')
        model.save(model_path)
        print(f"[Separate] {session_name} model saved to {model_path}")

        return model

    us_model = train_session_model(us_data, 'us_session')
    non_us_model = train_session_model(non_us_data, 'non_us_session')

    return us_model, non_us_model


def predict_with_session(
    test_data: pl.DataFrame,
    features: List[str],
    model_dir: str,
    name: str,
    target: str,
    session_col: str = 'is_us_session',
    mode: Literal['dann', 'separate_lgbm', 'separate_dann'] = 'dann'
) -> np.ndarray:
    """
    Generate predictions using session-aware models.

    Args:
        test_data: Test data with features and session labels
        features: Feature column names
        model_dir: Directory with saved models
        name: Model name prefix
        target: Target column name
        session_col: Session label column name
        mode: 'dann', 'separate_lgbm', or 'separate_dann'

    Returns:
        Array of predictions
    """
    X_test = test_data.select(features).to_numpy()
    session_labels = test_data.select(session_col).to_numpy().flatten()

    if mode == 'dann':
        # Single DANN model
        model_path = os.path.join(model_dir, f'{name}_{target}_dann')
        model = DANNWrapper.load(model_path)
        predictions = model.predict(X_test)

    else:
        # Separate models
        predictions = np.zeros(len(X_test))

        # US session predictions
        us_mask = session_labels == 1
        if us_mask.sum() > 0:
            model_path = os.path.join(model_dir, f'{name}_{target}_us_session')
            if mode == 'separate_lgbm':
                model_class, _ = prepare_lgbm()
                us_model = model_class.load(model_path)
            else:
                us_model = DANNWrapper.load(model_path)
            predictions[us_mask] = us_model.predict(X_test[us_mask])

        # Non-US session predictions
        non_us_mask = session_labels == 0
        if non_us_mask.sum() > 0:
            model_path = os.path.join(model_dir, f'{name}_{target}_non_us_session')
            if mode == 'separate_lgbm':
                model_class, _ = prepare_lgbm()
                non_us_model = model_class.load(model_path)
            else:
                non_us_model = DANNWrapper.load(model_path)
            predictions[non_us_mask] = non_us_model.predict(X_test[non_us_mask])

    return predictions


class SessionAwareTrainer:
    """
    Unified trainer for session-aware models.

    Supports three training modes:
    1. 'dann': Domain Adversarial Neural Network with session as domain
    2. 'separate_lgbm': Separate LightGBM models per session
    3. 'separate_dann': Separate DANN models per session (no domain adaptation)
    """

    def __init__(
        self,
        mode: Literal['dann', 'separate_lgbm', 'separate_dann'] = 'dann',
        us_start_hour: int = 13,
        us_end_hour: int = 21,
        session_col: str = 'is_us_session',
        dann_params: Optional[Dict] = None
    ):
        self.mode = mode
        self.us_start_hour = us_start_hour
        self.us_end_hour = us_end_hour
        self.session_col = session_col
        self.dann_params = dann_params or {}

    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add session labels to data."""
        return add_session_labels(
            df,
            us_start_hour=self.us_start_hour,
            us_end_hour=self.us_end_hour,
            session_col=self.session_col
        )

    def train(
        self,
        training_set: pl.DataFrame,
        target: str,
        features: List[str],
        save_path: str,
        config: Dict,
        name: str = '',
        use_sw: bool = False
    ):
        """
        Train session-aware model(s).

        Args:
            training_set: Training data (will add session labels if missing)
            target: Target column name
            features: Feature column names
            save_path: Directory to save models
            config: Training configuration
            name: Model name prefix
            use_sw: Whether to use sample weights

        Returns:
            Trained model(s)
        """
        # Add session labels if not present
        if self.session_col not in training_set.columns:
            training_set = self.prepare_data(training_set)

        # Print session stats
        stats = get_session_stats(training_set, self.session_col)
        print(f"[SessionTrainer] Mode: {self.mode}")
        print(f"[SessionTrainer] Session distribution: "
              f"US={stats['us_ratio']:.1%}, Non-US={stats['non_us_ratio']:.1%}")

        if self.mode == 'dann':
            return train_dann_session(
                training_set, target, features, save_path, config, name,
                session_col=self.session_col,
                use_sw=use_sw,
                dann_params=self.dann_params
            )
        else:
            model_type = 'lgbm' if self.mode == 'separate_lgbm' else 'dann'
            return train_separate_sessions(
                training_set, target, features, save_path, config, name,
                session_col=self.session_col,
                use_sw=use_sw,
                model_type=model_type
            )

    def predict(
        self,
        test_data: pl.DataFrame,
        features: List[str],
        model_dir: str,
        name: str,
        target: str
    ) -> np.ndarray:
        """
        Generate predictions.

        Args:
            test_data: Test data (will add session labels if missing)
            features: Feature column names
            model_dir: Directory with saved models
            name: Model name prefix
            target: Target column name

        Returns:
            Array of predictions
        """
        # Add session labels if not present
        if self.session_col not in test_data.columns:
            test_data = self.prepare_data(test_data)

        return predict_with_session(
            test_data, features, model_dir, name, target,
            session_col=self.session_col,
            mode=self.mode
        )
