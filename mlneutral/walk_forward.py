"""
Walk-Forward Testing Framework for ML Neutral Strategy

This script implements monthly walk-forward testing:
1. Train on data up to cutoff date
2. Test on the following month
3. Track OOS performance metrics
4. Aggregate results across all periods
"""

import polars as pl
import pandas as pd
import numpy as np
import os
import pickle
import yaml
import gc
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from mlneutral.feature_engineer1 import (
    transformFT,
    feature_selection_2step,
    process_test_data,
)
from mlneutral.lgbmr import prepare_lgbm
from mlneutral.trainer import train_classifier
from parquet_reader import parallel_read_parquets



class WalkForwardTest:
    """
    Walk-forward testing framework for crypto return prediction.

    Attributes:
        data_path: Path to the features parquet file
        ret_path: Path to returns parquet file (for performance analysis)
        features: List of feature column names to use
        horizon: Prediction horizon in periods
        train_config_path: Path to training configuration YAML
        output_dir: Base directory for outputs
    """

    def __init__(
        self,
        vname: str = 'wf',
        data_config: Dict = None,
        features: Optional[List[str]] = None,
        horizon: int = 4,
        target_col: str = None,
        train_config_path: str = 'mlneutral/train_config.yaml',
        output_dir: str = 'walkforward_results',
        target_func: str = 'ret',
        delta_month: int = 3,
        anti_leaking_days = 1
    ):
        self.vname = vname
        self.data_config = data_config
        self.horizon = horizon
        self.train_config_path = train_config_path
        self.output_dir = output_dir
        self.target_col = target_col
        self.delta_month = delta_month
        self.antileaking_buff = anti_leaking_days

        if target_func == 'ret':
            self._load_target = self._load_target2
        else:
            self._load_target = self._load_target1

        ## Load data
        #print(f"Loading data from {data_path}...")
        #self.data = pl.read_parquet(data_path)
        self.universe = pd.read_parquet(data_config['universe'])

        # Auto-detect features if not provided
        if features is None:
            self.features = self._detect_features()
        else:
            self.features = features
        print(f"Using {len(self.features)} features")

        # Load training config
        with open(train_config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)

        ## Load returns for performance analysis
        self._load_returns()
        #if os.path.exists(ret_path):
        #    self.returns = pd.read_parquet(ret_path)
        #else:
        #    self.returns = None
        #   print(f"Warning: Returns file {ret_path} not found. Performance analysis will be limited.")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/features", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/predictions", exist_ok=True)

        # Results storage
        self.results: List[Dict] = []
        self.all_predictions = []

    def _load_all_features(self, symbols):
        res = None
        for key in self.data_config['features']:
            path = self.data_config['features'][key]
            temp = parallel_read_parquets(
                folder=path,
                # columns=['__index_level_0__']+perp_features_agg_factor,
                pattern="*.parquet",
                n_workers=8,
                # symbol_prefix="perp_",
                symbols=symbols,
            )
            if 'timestamp' in temp.columns:
                temp = temp.rename(mapping = {"timestamp": 'datetime'})
            elif 'date_time' in temp.columns:
                temp = temp.rename(mapping = {"date_time": 'datetime'})

            if res is None:
                res = temp
            else:
                res = res.join(temp, on = ['datetime', 'symbol'], how = 'left')
        return res

    def _load_returns(self):
        path = self.data_config['returns']
        res = parallel_read_parquets(
            folder=path,
            columns=['__index_level_0__', 'close'],
            pattern="*.parquet",
            n_workers=8,
            symbol_prefix="perp_",
        ).rename(mapping = {"timestamp": 'datetime'})
        res = res.select(['datetime', 'symbol', 'close']).pivot(
                on="symbol",
                index="datetime",
                values="close"
        ).sort('datetime')
        res = res.to_pandas().set_index('datetime')
        self.returns = res.pct_change()

    def _load_target1(self, symbols):
        ret = self.returns.loc[:, symbols]
        tar = ret.rolling(self.horizon).sum().shift(-self.horizon)
        tar = tar.subtract(tar.mean(axis=1),axis=0)
        tar = tar.div(tar.abs().sum(axis=1),axis=0)
        scalevol = (tar.shift(1) * ret.reindex_like(tar)).sum(axis=1).ewm(168,min_periods = 72).std()
        tar = (tar * 0.15/80).div( scalevol,axis=0).clip(-0.1,0.1)
        tar = pl.from_pandas(
            tar.reset_index().melt(id_vars="datetime", var_name="symbol", value_name=self.target_col)
        )
        return tar

    def _load_target2(self, symbols):
        ret = self.returns.loc[:, symbols]
        tar = ret.rolling(self.horizon).sum().shift(-self.horizon)
        tar = tar.subtract(tar.median(axis=1),axis=0).clip(-0.5,0.5)
        tar = pl.from_pandas(
            tar.reset_index().melt(id_vars="datetime", var_name="symbol", value_name=self.target_col)
        )
        return tar

    def _detect_features(self) -> List[str]:
        """Auto-detect feature columns from data."""
        exclude_cols = {'datetime', 'symbol', 'date', 'time', 'target', 'ret', 'price',
                       'open', 'high', 'low', 'close', 'volume', 'out_of_range'}
        btcf = self._load_all_features(['BTCUSDT'])
        all_cols = set(btcf.columns)
        feature_cols = [c for c in all_cols if c not in exclude_cols
                       and not c.startswith('ret_')
                       and not c.startswith('label_')
                       and not c.startswith('sw_')]
        return sorted(feature_cols)

    def generate_train_dates(
        self,
        start_date: str,
        end_date: str,
        train_day: int = 15
    ) -> List[datetime]:
        """
        Generate monthly training cutoff dates.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            train_day: Day of month for training cutoff

        Returns:
            List of cutoff datetimes
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=train_day)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(day=train_day)

        dates = []
        current = start
        while current <= end:
            dates.append(pd.Timestamp(current).to_pydatetime())
            current += relativedelta(months=self.delta_month)

        return dates

    def _load_universe(self, asofdate):
        filters = self.universe.loc[:asofdate]
        symbols = filters.loc[filters.index[-1]]
        symbols = list(symbols[symbols==1].index)
        return symbols

    def train_single_period(
        self,
        name: str,
        train_cutoff: datetime,
        max_hp_evals: int = 30
    ) -> Tuple[str, str]:
        """
        Train model for a single period.

        Args:
            train_cutoff: Training data cutoff date
            max_hp_evals: Maximum hyperparameter optimization evaluations

        Returns:
            Tuple of (feature_dir, model_dir)
        """
        feature_dir = f"{self.output_dir}/features/features_{name}/"
        model_dir = f"{self.output_dir}/models/models_{name}/"

        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        train_cutoff = train_cutoff - relativedelta(days = self.antileaking_buff)
        # Filter training data
        symbols = self._load_universe(train_cutoff)
        self.data = self._load_all_features(symbols)
        labels = self._load_target(symbols)
        self.data = self.data.join(labels, on = ['datetime', 'symbol'], how = 'left')
        train_data = self.data.filter(pl.col("datetime") < train_cutoff)
        print(f"Info: Complete features preparation for {len(symbols)} symbols.")
        if train_data.height < self.train_config['training']['min_datasize_thres'] / 2:
            print(f"Warning: Insufficient training data for {name} ({train_data.height} rows)")
            return None, None

        # Transform features
        train_data, _ = transformFT(
            train_data, pl.DataFrame(), self.features,
            save_ecdf=True, outputfolder=feature_dir
        )
        print("Info: Complete transform features.")

        # Fill NaN with 0 for features
        train_data = train_data.with_columns(
            pl.col(col).replace({np.nan: 0}) for col in self.features
        )

        # Feature selection
        selected_data = feature_selection_2step(
            train_data, self.features, self.target_col,
            corr_drop_threshold=0.8,
            corr_merge_threshold=0.99,
            save_grouping=True,
            outputfolder=feature_dir
        )
        selected_data = selected_data.sort('datetime')
        print("Info: Complete features selection.")

        # Get selected feature names
        select_features = [ft for ft in selected_data.columns
                         if ft in self.features or 'cluster_' in ft]

        # Save feature list
        with open(f'{feature_dir}/feature_list.pkl', 'wb') as f:
            pickle.dump(select_features, f)

        print(f"Selected {len(select_features)} features for {name}")

        # Prepare model
        lgbm_model, params = prepare_lgbm()
        params['learning_rate'] = 1e-2
        params['early_stopping_rounds'] = 50
        #params['subsample'] = 0.8
        #params['max_depth'] = 12
        params['bagging_freq'] = 5
        params['bagging_fraction'] = 0.8
        params['feature_fraction'] = 0.8
        #params['reg_lambda'] = 0
        #params['reg_alpha'] = 0
        params['min_data_in_leaf'] = 5000
        params['num_boost_round'] = 500

        # Update config
        self.train_config['training']['max_hp_evals'] = max_hp_evals

        # Train
        train_classifier(
            training_set=selected_data,
            target=self.target_col,
            features=select_features,
            modelClass=lgbm_model,
            params=params,
            save_path=model_dir,
            config=self.train_config,
            name=name,
            use_sw=False
        )

        print(f"Training complete for {name}")
        gc.collect()

        return feature_dir, model_dir

    def predict_single_period(
        self,
        name: str,
        test_start: datetime,
        test_end: datetime,
        feature_dir: str,
        model_dir: str
    ) -> pl.DataFrame:
        """
        Generate predictions for a test period.

        Args:
            test_start: Test period start date
            test_end: Test period end date
            feature_dir: Directory with saved feature artifacts
            model_dir: Directory with saved models

        Returns:
            DataFrame with predictions
        """
        #name = os.path.basename(model_dir.rstrip('/'))

        # Filter test data
        test_data = self.data.filter(
            (pl.col("datetime") >= test_start) &
            (pl.col("datetime") < test_end)
        )

        if test_data.height == 0:
            print(f"No test data for period {test_start} to {test_end}")
            return None

        # Load feature artifacts
        with open(f'{feature_dir}/feature_mapping.pkl', 'rb') as f:
            merge_groups = pickle.load(f)

        with open(f'{feature_dir}/feature_list.pkl', 'rb') as f:
            select_features = pickle.load(f)

        with open(f'{feature_dir}/allfeatures_order.pkl', 'rb') as f:
            all_features = pickle.load(f)

        # Process test data
        test_data = process_test_data(
            test_data,
            all_features,
            select_features,
            ecdf_folder=feature_dir,
            feature_mapping_file=f'{feature_dir}/feature_mapping.pkl',
            scaler_file=os.path.join(feature_dir, 'X_scaler'),
            outputfolder=feature_dir
        )
        print(f"Info: test data for {len(test_data['symbol'].unique())} symbols.")
        test_data = test_data.sort('datetime').drop_nulls(self.target_col)
        print(f"Info: test data after droping for {len(test_data['symbol'].unique())} symbols.")

        # Load models and predict
        lgbm_model, _ = prepare_lgbm()

        model1_path = os.path.join(model_dir, f'{name}_{self.target_col}__1')
        model2_path = os.path.join(model_dir, f'{name}_{self.target_col}__2')

        model1 = lgbm_model.load(model1_path)
        model2 = lgbm_model.load(model2_path)

        X_test = test_data.select(select_features).to_numpy()
        preds1 = model1.predict(X_test)
        preds2 = model2.predict(X_test)

        predictions = (preds1 + preds2) / 2

        test_data = test_data.with_columns(
            pl.Series(name='prediction1', values=preds1),
            pl.Series(name='prediction2', values=preds2),
            pl.Series(name='prediction', values=predictions),
        ).select(['datetime','symbol', self.target_col, 'prediction1','prediction2','prediction'])

        return test_data

    def compute_period_metrics(
        self,
        predictions_df: pl.DataFrame,
        test_start: datetime,
        test_end: datetime
    ) -> Dict:
        """
        Compute performance metrics for a test period.

        Args:
            predictions_df: DataFrame with predictions
            test_start: Period start date
            test_end: Period end date

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'period_start': test_start,
            'period_end': test_end,
            'n_samples': predictions_df.height,
        }

        # IC (Information Coefficient)
        pred_np = predictions_df['prediction'].to_numpy()
        target_np = predictions_df[self.target_col].to_numpy()

        valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
        if valid_mask.sum() > 10:
            ic = np.corrcoef(pred_np[valid_mask], target_np[valid_mask])[0, 1]
            metrics['ic'] = ic

            # Rank IC
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(pred_np[valid_mask], target_np[valid_mask])
            metrics['rank_ic'] = rank_ic

            # Sign accuracy
            sign_acc = ((np.sign(pred_np[valid_mask]) == np.sign(target_np[valid_mask])).sum()
                       / valid_mask.sum())
            metrics['sign_accuracy'] = sign_acc

        # If we have returns data, compute PnL metrics
        if self.returns is not None:
            try:
                sgl = predictions_df.select(['datetime', 'symbol', 'prediction']).to_pandas()
                sgl = pd.pivot_table(sgl, index='datetime', columns='symbol', values='prediction')

                # Align with returns
                common_dates = sgl.index.intersection(self.returns.index)
                common_cols = sgl.columns.intersection(self.returns.columns)

                if len(common_dates) > 0 and len(common_cols) > 0:
                    sgl_aligned = sgl.loc[common_dates, common_cols]
                    ret_aligned = self.returns.loc[common_dates, common_cols]

                    # PnL (signal * returns)
                    pnl = (sgl_aligned.shift() * ret_aligned).sum(axis=1)

                    if len(pnl) > 1 and pnl.std() > 0:
                        # Annualized metrics (assuming hourly data, ~8760 hours/year)
                        periods_per_year = 8760 / self.horizon  # Adjust for horizon
                        metrics['sharpe'] = pnl.mean() / pnl.std() * np.sqrt(periods_per_year)
                        metrics['total_return'] = pnl.sum()
                        metrics['volatility'] = pnl.std() * np.sqrt(periods_per_year)

                        # Turnover
                        to = sgl_aligned.diff().abs().sum(axis=1).mean()
                        metrics['turnover'] = to

                        # Max drawdown
                        cum_pnl = pnl.cumsum()
                        running_max = cum_pnl.expanding().max()
                        drawdown = running_max - cum_pnl
                        metrics['max_drawdown'] = drawdown.max()
            except Exception as e:
                print(f"Warning: Could not compute PnL metrics: {e}")

        return metrics

    def run(
        self,
        start_date: str,
        end_date: str,
        train_day: int = 15,
        max_hp_evals: int = 30,
        min_train_months: int = 6
    ) -> pd.DataFrame:
        """
        Run full walk-forward test.

        Args:
            start_date: First training cutoff date (YYYY-MM-DD)
            end_date: Last training cutoff date (YYYY-MM-DD)
            train_day: Day of month for cutoffs
            max_hp_evals: HP optimization iterations per period
            min_train_months: Minimum months of training data required

        Returns:
            DataFrame with results for all periods
        """
        train_dates = self.generate_train_dates(start_date, end_date, train_day)
        print(f"Walk-forward test: {len(train_dates)} periods from {start_date} to {end_date}")

        data_start = self.returns.index[0]  
        for i, train_cutoff in enumerate(train_dates):
            print(f"\n{'='*60}")
            print(f"Period {i+1}/{len(train_dates)}: Training cutoff {train_cutoff.strftime('%Y-%m-%d')}")
            print(f"{'='*60}")
            name = f"{self.vname}_{train_cutoff.strftime('%Y%m%d')}"
            # Check minimum training data
            months_of_data = (train_cutoff - data_start).days / 30
            if months_of_data < min_train_months:
                print(f"Skipping: Only {months_of_data:.1f} months of data (need {min_train_months})")
                continue

            # Train
            feature_dir, model_dir = self.train_single_period(
                name, train_cutoff, max_hp_evals=max_hp_evals
            )

            if feature_dir is None:
                continue

            # Define test period (next month)
            test_start = train_cutoff
            test_end = train_cutoff + relativedelta(months=self.delta_month)

            print(f"Testing period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

            # Predict
            predictions = self.predict_single_period(
                name, test_start, test_end, feature_dir, model_dir
            )

            if predictions is None or predictions.height == 0:
                print(f"No predictions generated for this period")
                continue

            # Save predictions
            pred_path = f"{self.output_dir}/predictions/pred_{train_cutoff.strftime('%Y%m%d')}.parquet"
            predictions.write_parquet(pred_path)
            self.all_predictions.append(predictions.select(['datetime','symbol', self.target_col, 'prediction1','prediction2','prediction']))

            # Compute metrics
            metrics = self.compute_period_metrics(predictions, test_start, test_end)
            self.results.append(metrics)

            # Print period summary
            print(f"\nPeriod Results:")
            print(f"  Samples: {metrics['n_samples']}")
            if 'ic' in metrics:
                print(f"  IC: {metrics['ic']:.4f}")
                print(f"  Rank IC: {metrics['rank_ic']:.4f}")
                print(f"  Sign Accuracy: {metrics['sign_accuracy']:.2%}")
            if 'sharpe' in metrics:
                print(f"  Sharpe: {metrics['sharpe']:.2f}")
                print(f"  Total Return: {metrics['total_return']:.4f}")

            gc.collect()

        # Aggregate results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.output_dir}/walk_forward_results.csv", index=False)

        # Print summary
        print(f"\n{'='*60}")
        print("WALK-FORWARD TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total periods: {len(self.results)}")

        if len(self.results) > 0 and 'ic' in results_df.columns:
            print(f"\nIC Statistics:")
            print(f"  Mean: {results_df['ic'].mean():.4f}")
            print(f"  Std: {results_df['ic'].std():.4f}")
            print(f"  Hit Rate (IC > 0): {(results_df['ic'] > 0).mean():.2%}")

            print(f"\nRank IC Statistics:")
            print(f"  Mean: {results_df['rank_ic'].mean():.4f}")

            print(f"\nSign Accuracy:")
            print(f"  Mean: {results_df['sign_accuracy'].mean():.2%}")

        if 'sharpe' in results_df.columns:
            print(f"\nSharpe Statistics:")
            print(f"  Mean: {results_df['sharpe'].mean():.2f}")
            print(f"  Std: {results_df['sharpe'].std():.2f}")

        # Save combined predictions
        if self.all_predictions:
            combined = pl.concat(self.all_predictions)
            combined.write_parquet(f"{self.output_dir}/all_predictions.parquet")
            print(f"\nCombined predictions saved: {combined.height} rows")

        return results_df

    def analyze_results(self, results_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detailed analysis of walk-forward results.

        Args:
            results_df: Results DataFrame (uses self.results if None)

        Returns:
            Dictionary with analysis metrics
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)

        analysis = {}

        if 'ic' in results_df.columns:
            analysis['ic_mean'] = results_df['ic'].mean()
            analysis['ic_std'] = results_df['ic'].std()
            analysis['ic_ir'] = results_df['ic'].mean() / results_df['ic'].std() if results_df['ic'].std() > 0 else 0
            analysis['ic_hit_rate'] = (results_df['ic'] > 0).mean()

        if 'sharpe' in results_df.columns:
            analysis['sharpe_mean'] = results_df['sharpe'].mean()
            analysis['sharpe_std'] = results_df['sharpe'].std()

        if 'total_return' in results_df.columns:
            analysis['cumulative_return'] = results_df['total_return'].sum()

        if 'max_drawdown' in results_df.columns:
            analysis['avg_max_dd'] = results_df['max_drawdown'].mean()
            analysis['worst_dd'] = results_df['max_drawdown'].max()

        return analysis


    def run_test(
        self,
        start_date: str,
        end_date: str,
        train_day: int = 15,
        max_hp_evals: int = 30,
        min_train_months: int = 6
    ) -> pd.DataFrame:
        """
        Run full walk-forward test.

        Args:
            start_date: First training cutoff date (YYYY-MM-DD)
            end_date: Last training cutoff date (YYYY-MM-DD)
            train_day: Day of month for cutoffs
            max_hp_evals: HP optimization iterations per period
            min_train_months: Minimum months of training data required

        Returns:
            DataFrame with results for all periods
        """
        train_dates = self.generate_train_dates(start_date, end_date, train_day)
        print(f"Walk-forward test: {len(train_dates)} periods from {start_date} to {end_date}")

        data_start = self.returns.index[0]
        for i, train_cutoff in enumerate(train_dates):
            print(f"\n{'='*60}")
            print(f"Period {i+1}/{len(train_dates)}: Training cutoff {train_cutoff.strftime('%Y-%m-%d')}")
            print(f"{'='*60}")
            name = f"{self.vname}_{train_cutoff.strftime('%Y%m%d')}"
            # Check minimum training data
            months_of_data = (train_cutoff - data_start).days / 30
            if months_of_data < min_train_months:
                print(f"Skipping: Only {months_of_data:.1f} months of data (need {min_train_months})")
                continue

            # Train
            #feature_dir, model_dir = self.train_single_period(
            #    name, train_cutoff, max_hp_evals=max_hp_evals
            #)
            feature_dir = f"{self.output_dir}/features/features_{name}/"
            model_dir = f"{self.output_dir}/models/models_{name}/"

            symbols = self._load_universe(train_cutoff)
            self.data = self._load_all_features(symbols)
            if self.target_col not in self.data.columns:
                labels = self._load_target(symbols)
                self.data = self.data.join(labels, on = ['datetime', 'symbol'], how = 'left')

            # Filter training data
            train_data = self.data.filter(pl.col("datetime") < train_cutoff)

            if train_data.height < self.train_config['training']['min_datasize_thres'] / 2:
                print(f"Warning: Insufficient training data for {name} ({train_data.height} rows)")
                continue

            # Define test period (next month)
            test_start = train_cutoff
            test_end = train_cutoff + relativedelta(months=self.delta_month)

            print(f"Testing period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

            # Predict
            predictions = self.predict_single_period(
                name, test_start, test_end, feature_dir, model_dir
            )

            if predictions is None or predictions.height == 0:
                print(f"No predictions generated for this period")
                continue

            # Save predictions
            pred_path = f"{self.output_dir}/predictions/pred_{train_cutoff.strftime('%Y%m%d')}.parquet"
            predictions.write_parquet(pred_path)
            self.all_predictions.append(predictions)

            # Compute metrics
            metrics = self.compute_period_metrics(predictions, test_start, test_end)
            self.results.append(metrics)

            # Print period summary
            print(f"\nPeriod Results:")
            print(f"  Samples: {metrics['n_samples']}")
            if 'ic' in metrics:
                print(f"  IC: {metrics['ic']:.4f}")
                print(f"  Rank IC: {metrics['rank_ic']:.4f}")
                print(f"  Sign Accuracy: {metrics['sign_accuracy']:.2%}")
            if 'sharpe' in metrics:
                print(f"  Sharpe: {metrics['sharpe']:.2f}")
                print(f"  Total Return: {metrics['total_return']:.4f}")

            gc.collect()

        # Aggregate results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.output_dir}/walk_forward_results.csv", index=False)

        # Print summary
        print(f"\n{'='*60}")
        print("WALK-FORWARD TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total periods: {len(self.results)}")

        if len(self.results) > 0 and 'ic' in results_df.columns:
            print(f"\nIC Statistics:")
            print(f"  Mean: {results_df['ic'].mean():.4f}")
            print(f"  Std: {results_df['ic'].std():.4f}")
            print(f"  Hit Rate (IC > 0): {(results_df['ic'] > 0).mean():.2%}")

            print(f"\nRank IC Statistics:")
            print(f"  Mean: {results_df['rank_ic'].mean():.4f}")

            print(f"\nSign Accuracy:")
            print(f"  Mean: {results_df['sign_accuracy'].mean():.2%}")

        if 'sharpe' in results_df.columns:
            print(f"\nSharpe Statistics:")
            print(f"  Mean: {results_df['sharpe'].mean():.2f}")
            print(f"  Std: {results_df['sharpe'].std():.2f}")

        # Save combined predictions
        if self.all_predictions:
            combined = pl.concat(self.all_predictions)
            combined.write_parquet(f"{self.output_dir}/all_predictions.parquet")
            print(f"\nCombined predictions saved: {combined.height} rows")

        return results_df


def main():
    """Main entry point for walk-forward testing."""
    import argparse

    parser = argparse.ArgumentParser(description='Walk-Forward Testing for ML Neutral Strategy')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data', type=str, default='data/features_mom.parquet', help='Data file path')
    parser.add_argument('--returns', type=str, default='data/local/ret.parquet', help='Returns file path')
    parser.add_argument('--horizon', type=int, default=4, help='Prediction horizon')
    parser.add_argument('--hp-evals', type=int, default=30, help='HP optimization iterations')
    parser.add_argument('--output', type=str, default='walkforward_results', help='Output directory')

    args = parser.parse_args()

    wf = WalkForwardTest(
        data_path=args.data,
        ret_path=args.returns,
        horizon=args.horizon,
        output_dir=args.output
    )

    results = wf.run(
        start_date=args.start,
        end_date=args.end,
        max_hp_evals=args.hp_evals
    )

    print("\nAnalysis:")
    analysis = wf.analyze_results(results)
    for k, v in analysis.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
