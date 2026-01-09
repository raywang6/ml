"""
Session-Based Domain Adaptation Training Script

This script trains models using domain adaptation to learn session-invariant features.
Two approaches are supported:

1. DANN (Domain Adversarial Neural Network):
   - Single model with adversarial loss
   - Learns features invariant to trading session
   - Uses gradient reversal to confuse session classifier

2. Separate Models:
   - Train distinct models for US and non-US sessions
   - Each model specializes in its session's patterns

US Trading Hours: UTC 13:00 - 21:00
"""

import polars as pl
import pandas as pd
import numpy as np
import os
import sys
import gc
import yaml
import pickle
from datetime import datetime
from typing import Optional, List, Literal

from mlneutral.feature_engineer import (
    transformFT,
    feature_selection_2step,
    process_test_data,
)
from mlneutral.session_trainer import (
    SessionAwareTrainer,
    add_session_labels,
    get_session_stats
)


class SessionDANNExperiment:
    """
    Experiment runner for session-aware domain adaptation training.
    """

    def __init__(
        self,
        data_path: str = 'data/features_mom.parquet',
        ret_path: str = 'data/local/ret.parquet',
        features: Optional[List[str]] = None,
        horizon: int = 4,
        train_config_path: str = 'mlneutral/train_config.yaml',
        output_dir: str = 'session_dann_results',
        target_col: str = 'target',
        mode: Literal['dann', 'separate_lgbm', 'separate_dann'] = 'dann',
        us_start_hour: int = 13,
        us_end_hour: int = 21
    ):
        """
        Initialize experiment.

        Args:
            data_path: Path to features parquet
            ret_path: Path to returns parquet
            features: Feature columns (None = auto-detect)
            horizon: Prediction horizon
            train_config_path: Training config YAML
            output_dir: Output directory
            target_col: Target column name
            mode: Training mode ('dann', 'separate_lgbm', 'separate_dann')
            us_start_hour: Start of US session (UTC hour)
            us_end_hour: End of US session (UTC hour)
        """
        self.data_path = data_path
        self.ret_path = ret_path
        self.horizon = horizon
        self.train_config_path = train_config_path
        self.output_dir = output_dir
        self.target_col = target_col
        self.mode = mode
        self.us_start_hour = us_start_hour
        self.us_end_hour = us_end_hour

        # Load data
        print(f"Loading data from {data_path}...")
        self.data = pl.read_parquet(data_path)

        # Auto-detect features
        if features is None:
            self.features = self._detect_features()
        else:
            self.features = features
        print(f"Using {len(self.features)} features")

        # Load config
        with open(train_config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)

        # Load returns
        if os.path.exists(ret_path):
            self.returns = pd.read_parquet(ret_path)
        else:
            self.returns = None

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/features", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/predictions", exist_ok=True)

        # Initialize trainer
        self.trainer = SessionAwareTrainer(
            mode=mode,
            us_start_hour=us_start_hour,
            us_end_hour=us_end_hour
        )

        # Results
        self.results = []

    def _detect_features(self) -> List[str]:
        """Auto-detect feature columns."""
        exclude_cols = {'datetime', 'symbol', 'date', 'time', 'target', 'ret', 'price',
                       'open', 'high', 'low', 'close', 'volume', 'out_of_range',
                       'is_us_session'}
        all_cols = set(self.data.columns)
        feature_cols = [c for c in all_cols if c not in exclude_cols
                       and not c.startswith('ret_')
                       and not c.startswith('label_')
                       and not c.startswith('sw_')]
        return sorted(feature_cols)

    def train_period(
        self,
        train_cutoff: datetime,
        dann_params: Optional[dict] = None
    ):
        """
        Train models for a single period.

        Args:
            train_cutoff: Training data cutoff date
            dann_params: Optional DANN parameters override
        """
        name = f"session_{train_cutoff.strftime('%Y%m%d')}"
        feature_dir = f"{self.output_dir}/features/features_{name}/"
        model_dir = f"{self.output_dir}/models/models_{name}/"

        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Filter training data
        train_data = self.data.filter(pl.col("datetime") < train_cutoff)
        print(f"\n{'='*60}")
        print(f"Training cutoff: {train_cutoff}")
        print(f"Training samples: {train_data.height}")

        if train_data.height < self.train_config['training']['min_datasize_thres'] / 2:
            print(f"Warning: Insufficient data ({train_data.height} rows)")
            return None, None

        # Add session labels
        train_data = self.trainer.prepare_data(train_data)
        stats = get_session_stats(train_data)
        print(f"Session distribution: US={stats['us_ratio']:.1%}, Non-US={stats['non_us_ratio']:.1%}")

        # Transform features
        train_data, _ = transformFT(
            train_data, pl.DataFrame(), self.features,
            save_ecdf=True, outputfolder=feature_dir
        )

        # Fill NaN
        train_data = train_data.with_columns(
            pl.col(col).replace({np.nan: 0}) for col in self.features
        )

        # Feature selection
        selected_data = feature_selection_2step(
            train_data, self.features, self.target_col,
            corr_drop_threshold=0.99,
            corr_merge_threshold=0.9,
            save_grouping=True,
            outputfolder=feature_dir
        )
        selected_data = selected_data.sort('datetime')

        # Re-add session labels after feature selection (it may have been dropped)
        if 'is_us_session' not in selected_data.columns:
            selected_data = self.trainer.prepare_data(selected_data)

        # Get selected features
        select_features = [ft for ft in selected_data.columns
                         if ft in self.features or 'cluster_' in ft]

        with open(f'{feature_dir}/feature_list.pkl', 'wb') as f:
            pickle.dump(select_features, f)

        print(f"Selected {len(select_features)} features")

        # Update DANN params if provided
        if dann_params:
            self.trainer.dann_params.update(dann_params)

        # Train
        print(f"\nTraining {self.mode} model...")
        model = self.trainer.train(
            selected_data,
            self.target_col,
            select_features,
            model_dir,
            self.train_config,
            name=name,
            use_sw=False
        )

        gc.collect()
        return feature_dir, model_dir

    def test_period(
        self,
        test_start: datetime,
        test_end: datetime,
        feature_dir: str,
        model_dir: str
    ) -> pl.DataFrame:
        """
        Test on a period.

        Args:
            test_start: Test period start
            test_end: Test period end
            feature_dir: Feature artifacts directory
            model_dir: Model directory

        Returns:
            DataFrame with predictions
        """
        name = os.path.basename(model_dir.rstrip('/')).replace('models_', '')

        # Filter test data
        test_data = self.data.filter(
            (pl.col("datetime") >= test_start) &
            (pl.col("datetime") < test_end)
        )

        if test_data.height == 0:
            return None

        # Load artifacts
        with open(f'{feature_dir}/feature_mapping.pkl', 'rb') as f:
            merge_groups = pickle.load(f)

        with open(f'{feature_dir}/feature_list.pkl', 'rb') as f:
            select_features = pickle.load(f)

        with open(f'{feature_dir}/allfeatures_order.pkl', 'rb') as f:
            all_features = pickle.load(f)

        # Process test data
        test_data = process_test_data(
            test_data, all_features, select_features,
            ecdf_folder=feature_dir,
            feature_mapping_file=f'{feature_dir}/feature_mapping.pkl',
            scaler_file=os.path.join(feature_dir, 'X_scaler'),
            outputfolder=feature_dir
        )
        test_data = test_data.sort('datetime').drop_nulls(self.target_col)

        # Generate predictions
        predictions = self.trainer.predict(
            test_data, select_features, model_dir, name, self.target_col
        )

        test_data = test_data.with_columns(
            pl.Series(name='prediction', values=predictions)
        )

        return test_data

    def compute_metrics(
        self,
        predictions_df: pl.DataFrame,
        test_start: datetime,
        test_end: datetime
    ) -> dict:
        """Compute performance metrics."""
        from scipy.stats import spearmanr

        metrics = {
            'period_start': test_start,
            'period_end': test_end,
            'n_samples': predictions_df.height,
            'mode': self.mode
        }

        pred = predictions_df['prediction'].to_numpy()
        target = predictions_df[self.target_col].to_numpy()
        session = predictions_df['is_us_session'].to_numpy() if 'is_us_session' in predictions_df.columns else None

        valid = ~(np.isnan(pred) | np.isnan(target))

        if valid.sum() > 10:
            # Overall metrics
            metrics['ic'] = np.corrcoef(pred[valid], target[valid])[0, 1]
            metrics['rank_ic'], _ = spearmanr(pred[valid], target[valid])
            metrics['sign_acc'] = ((np.sign(pred[valid]) == np.sign(target[valid])).sum() / valid.sum())

            # Per-session metrics
            if session is not None:
                # US session
                us_mask = valid & (session == 1)
                if us_mask.sum() > 10:
                    metrics['ic_us'] = np.corrcoef(pred[us_mask], target[us_mask])[0, 1]
                    metrics['sign_acc_us'] = ((np.sign(pred[us_mask]) == np.sign(target[us_mask])).sum() / us_mask.sum())

                # Non-US session
                non_us_mask = valid & (session == 0)
                if non_us_mask.sum() > 10:
                    metrics['ic_non_us'] = np.corrcoef(pred[non_us_mask], target[non_us_mask])[0, 1]
                    metrics['sign_acc_non_us'] = ((np.sign(pred[non_us_mask]) == np.sign(target[non_us_mask])).sum() / non_us_mask.sum())

        return metrics

    def run_walk_forward(
        self,
        start_date: str,
        end_date: str,
        train_day: int = 15,
        dann_params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Run walk-forward test.

        Args:
            start_date: First training cutoff (YYYY-MM-DD)
            end_date: Last training cutoff (YYYY-MM-DD)
            train_day: Day of month for cutoffs
            dann_params: Optional DANN parameters

        Returns:
            Results DataFrame
        """
        from dateutil.relativedelta import relativedelta

        # Generate dates
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=train_day)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(day=train_day)

        dates = []
        current = start
        while current <= end:
            dates.append(pd.Timestamp(current, tz="UTC").to_pydatetime())
            current += relativedelta(months=1)

        print(f"Walk-forward test: {len(dates)} periods")
        print(f"Mode: {self.mode}")

        all_predictions = []

        for i, train_cutoff in enumerate(dates):
            print(f"\n{'='*60}")
            print(f"Period {i+1}/{len(dates)}")

            # Train
            feature_dir, model_dir = self.train_period(train_cutoff, dann_params)
            if feature_dir is None:
                continue

            # Test
            test_start = train_cutoff
            test_end = train_cutoff + relativedelta(months=1)

            predictions = self.test_period(test_start, test_end, feature_dir, model_dir)
            if predictions is None:
                continue

            # Save predictions
            pred_path = f"{self.output_dir}/predictions/pred_{train_cutoff.strftime('%Y%m%d')}.parquet"
            predictions.write_parquet(pred_path)
            all_predictions.append(predictions)

            # Metrics
            metrics = self.compute_metrics(predictions, test_start, test_end)
            self.results.append(metrics)

            # Print
            print(f"\nPeriod Results:")
            print(f"  Overall IC: {metrics.get('ic', 0):.4f}")
            print(f"  US Session IC: {metrics.get('ic_us', 0):.4f}")
            print(f"  Non-US Session IC: {metrics.get('ic_non_us', 0):.4f}")
            print(f"  Sign Accuracy: {metrics.get('sign_acc', 0):.2%}")

            gc.collect()

        # Aggregate
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.output_dir}/results_{self.mode}.csv", index=False)

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY - Mode: {self.mode}")
        print(f"{'='*60}")
        print(f"Periods: {len(self.results)}")

        if len(self.results) > 0:
            print(f"\nOverall IC: {results_df['ic'].mean():.4f} ± {results_df['ic'].std():.4f}")
            if 'ic_us' in results_df.columns:
                print(f"US Session IC: {results_df['ic_us'].mean():.4f} ± {results_df['ic_us'].std():.4f}")
            if 'ic_non_us' in results_df.columns:
                print(f"Non-US Session IC: {results_df['ic_non_us'].mean():.4f} ± {results_df['ic_non_us'].std():.4f}")

        # Save combined predictions
        if all_predictions:
            combined = pl.concat(all_predictions)
            combined.write_parquet(f"{self.output_dir}/all_predictions_{self.mode}.parquet")

        return results_df


def main():
    """Run session-based training experiment."""

    # Configuration
    CONFIG = {
        # Data
        'data_path': 'data/features_mom.parquet',
        'ret_path': 'data/local/ret.parquet',

        # Dates
        'start_date': '2024-06-01',
        'end_date': '2024-12-01',
        'train_day': 15,

        # Model
        'horizon': 4,
        'target_col': 'target',

        # Training mode: 'dann', 'separate_lgbm', or 'separate_dann'
        'mode': 'dann',

        # US session hours (UTC)
        'us_start_hour': 13,
        'us_end_hour': 21,

        # Output
        'output_dir': 'session_dann_results',

        # DANN-specific parameters (optional)
        'dann_params': {
            'hidden_dims': [256, 128, 64],
            'learning_rate': 1e-3,
            'n_epochs': 100,
            'patience': 15,
            'lambda_schedule': 'exp',
            'lambda_max': 1.0,
            'domain_loss_weight': 0.1,
        }
    }

    print("="*60)
    print("Session-Based Domain Adaptation Training")
    print("="*60)
    for k, v in CONFIG.items():
        if k != 'dann_params':
            print(f"  {k}: {v}")
    print("="*60)

    # Run experiment
    exp = SessionDANNExperiment(
        data_path=CONFIG['data_path'],
        ret_path=CONFIG['ret_path'],
        horizon=CONFIG['horizon'],
        output_dir=CONFIG['output_dir'],
        target_col=CONFIG['target_col'],
        mode=CONFIG['mode'],
        us_start_hour=CONFIG['us_start_hour'],
        us_end_hour=CONFIG['us_end_hour']
    )

    results = exp.run_walk_forward(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        train_day=CONFIG['train_day'],
        dann_params=CONFIG['dann_params']
    )

    return exp, results


if __name__ == '__main__':
    exp, results = main()
