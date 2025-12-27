"""
Description: Script for fine tuning parameters for the CTA strategies
in projects/cta/strategy/strategies.py
Date: 2025-07-03
"""

import datetime as dt
import polars as pl
import pandas as pd
import numpy as np
import os, sys
import pickle
import optuna
import logging
from typing import Optional
from multiprocessing import Pool, cpu_count

TRANSACTION_COST = 0.0002  
from joblib import Parallel, delayed
from tqdm import tqdm 

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval

ROOT_DATA_PATH = "/data/crypto/bar1m/futures/"
from performance_metrics import PerformanceCalculator

from strategy.params_spaces import PARAMS_SPACES
from strategy.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define DEFAULT_METRICS locally
DEFAULT_METRICS = {
    "IC": {},
    "Sharpe_ratio": {},
    "Calmar_ratio": {},
    "total_return": {}, 
    "net_return": {},
    "max_drawdown": {},
    "hit_rate": {},
    "exp_ret_per_mcl": {},
    "profit_factor": {},
    "avg_long_return": {},
    "avg_short_return": {},
    "pct_in_the_market": {},
}

class SignalEvaluator:
    """ signal evaluation """

    def __init__(self, signal_df: pl.DataFrame, signal_col: str, n_bars: int = 1):
        """ initialize with data and signal DataFrames """
        if isinstance( signal_df, pd.DataFrame ):
            signal_df = pl.from_pandas(signal_df)

        self.signal_df = signal_df # market data with columns of signals
        self.signal_col = signal_col # columns to evaluate
        self.n_bars = n_bars # future bars to consider

    def _compute_mcl(self, signal: pl.Series, fut_ret: pl.Series) -> float:
        """ 
        Compute maximum consecutive loss on a signal.
        """
        
        signal_ret = signal * fut_ret
        wrong_way_mask = (signal * fut_ret <= 0)
        if not wrong_way_mask.any():
            return 0.0

        signal_neg_ret = pl.select(
                            pl.when(wrong_way_mask)
                            .then(signal_ret)
                            .otherwise(0.0)
                            .alias("result")
                        ).to_series()
    
        # Step 1: Mask for negative returns (True if < 0, False if == 0)
        is_negative = signal_neg_ret < 0
        
        # Step 2: Assign group IDs to consecutive negative streaks
        group_ids = (is_negative != is_negative.shift(1)).cum_sum() * is_negative
        
        # Step 3: Compute cumulative product (1 + r_i) for each streak
        cumprod = (
            pl.DataFrame({"returns": signal_neg_ret, "group": group_ids})
            .filter(pl.col("returns") < 0)
            .group_by("group")
            .agg(
                (1 + pl.col("returns")).product().alias("product")
            )
        )
        
        # Step 4: Find the minimum product (worst streak)
        min_product = cumprod.get_column("product").min()
        
        # Handle case where there are no negative returns
        if min_product is None:
            return 0.0
        
        # Step 5: Max loss = 1 - min_product
        max_loss = 1 - min_product

        return max_loss
    
    def _compute_max_drawdown(self, signal_returns: pl.DataFrame, pos_col: str) -> float:
        """
        Compute maximum drawdown for a strategy using cumulative returns.
        """
        # Get non-zero signal returns
        non_zero_returns = signal_returns.filter(pl.col(pos_col) != 0)[pos_col]
        
        if non_zero_returns.len() == 0:
            return 0.0
        
        # Calculate cumulative returns (1 + r_i)
        cum_returns = (1 + non_zero_returns).cum_prod()
        
        # Calculate running maximum (peak)
        running_max = cum_returns.cum_max()
        
        # Calculate drawdown at each point
        drawdown = (cum_returns - running_max) / running_max
        
        # Maximum drawdown (most negative value)
        max_drawdown = abs(drawdown.min()) if drawdown.min() < 0 else 0.0
        
        return max_drawdown

    def _compute_total_return(self, signal_returns: pl.DataFrame, pos_col: str) -> float:
        """ Compute total return for a strategy """
        # Get non-zero signal returns
        non_zero_returns = signal_returns.filter(pl.col(pos_col) != 0)[pos_col]
        
        if non_zero_returns.len() == 0:
            return 0.0
        
        # Calculate cumulative returns (1 + r_i)
        cum_returns = (1 + non_zero_returns).cum_prod()
        
        # Total return is the last value in cumulative returns
        total_return = cum_returns[-1] - 1

        return total_return

    def _compute_metrics(self, 
                        metrics: dict) -> dict:
        """ compute metrics based on future returns and signals 
        Args: 
            - metrics: list of metrics to compute, e.g. ["IC", "Sharpe_ratio", "hit_rate"]
        
        """

        res = { metric: {} for metric in metrics }

        self.signal_df = self.signal_df.with_columns(
            pl.col(f"{self.signal_col}_pos").alias("position"),
            pl.col("close_price").pct_change().fill_null(0.0).alias("cur_ret"),
        ).drop([f"{self.signal_col}_pos"])

        # compute signal returns
        self.signal_df = self.signal_df.with_columns([
            # Detect signal changes (entry/exit points)
            (pl.col("position") != pl.col("position").shift(1)).alias("position_change"),
            
            # Calculate transaction cost only when signal changes
            pl.when(pl.col("position") != pl.col("position").shift(1))
                .then(pl.lit(TRANSACTION_COST))
                .otherwise(pl.lit(0.0))
                .alias("transaction_cost")
        ])

        # Compute signal returns with transaction costs
        raw_return_col = f"{self.signal_col}_return"
        net_return_col = f"net_{self.signal_col}_return"
        self.signal_df = self.signal_df.with_columns([          
            # before transaction costs
            (pl.col("cur_ret") * pl.col("position")).alias(raw_return_col),

            # Net return after transaction costs
            (pl.col("cur_ret") * pl.col("position") - pl.col("transaction_cost")).alias(net_return_col)
        ])

        if "IC" in metrics:
            corr = (
                self.signal_df.select(pl.corr( "cur_ret", "position", method = "spearman" )).item()
            )
            res["IC"][self.signal_col] = corr

        if "hit_rate" in metrics:
            non_zero_mask = self.signal_df[f"{self.signal_col}_return"] != 0
            if non_zero_mask.sum() > 0:
                hit_rate = (self.signal_df[f"{self.signal_col}_return"] > 0).sum() / non_zero_mask.sum()
            else:
                hit_rate = 0
            res["hit_rate"][self.signal_col] = hit_rate

        if "Sharpe_ratio" in metrics:
            if non_zero_mask.sum() > 0:
                # signal_returns_ = signal_returns.filter(non_zero_mask)
                avg_ret = self.signal_df[net_return_col].mean()
                std_dev = self.signal_df[net_return_col].std()
                annual_factor = (365 * 24 * 4) ** 0.5 # assuming 15mins bar
                sharpe_ratio = avg_ret / std_dev * annual_factor if std_dev != 0 else -np.inf
            else:
                sharpe_ratio = 0
            res["Sharpe_ratio"][self.signal_col] = sharpe_ratio

        if "Calmar_ratio" in metrics:
            if non_zero_mask.sum() > 0:
                # Calculate annualized return
                avg_ret = self.signal_df[net_return_col].mean()
                annual_factor = 365 * 24 * 4  # 15-min periods per year
                annual_return = avg_ret * annual_factor
                
                # Calculate maximum drawdown
                max_drawdown = self._compute_max_drawdown(self.signal_df, net_return_col)
                
                # Calculate Calmar ratio
                if max_drawdown > 0:
                    calmar_ratio = annual_return / max_drawdown
                else:
                    calmar_ratio = np.inf if annual_return > 0 else 0
            else:
                calmar_ratio = 0
            res["Calmar_ratio"][self.signal_col] = calmar_ratio

        if "total_return" in metrics:
            total_return = self._compute_total_return(self.signal_df, raw_return_col)
            res["total_return"][self.signal_col] = total_return

        if "net_return" in metrics:
            net_return = self._compute_total_return(self.signal_df, net_return_col)
            res["net_return"][self.signal_col] = net_return

        if "max_drawdown" in metrics:
            max_drawdown = self._compute_max_drawdown(self.signal_df, net_return_col)
            res["max_drawdown"][self.signal_col] = max_drawdown


        if "exp_ret_per_mcl" in metrics: #TODO{come back to upgrade this}
            mcl = self._compute_mcl(self.signal_df["position"], self.signal_df["cur_ret"])
            avg_ret = self.signal_df[net_return_col].mean()
            res["exp_ret_per_mcl"][self.signal_col] = avg_ret / mcl if mcl != 0 else np.inf

        if "profit_factor" in metrics:
            profit_factor = self.signal_df.select(
                pl.col(net_return_col).filter(pl.col(net_return_col) > 0).sum() /
                pl.col(net_return_col).filter(pl.col(net_return_col) < 0).sum().abs()
            ).item()

            res["profit_factor"][self.signal_col] = profit_factor

        if "avg_long_return" in metrics:
            long_mask = self.signal_df[ "position" ] > 0
            avg_long_return = self.signal_df[net_return_col].filter(long_mask).mean() if long_mask.any() else 0
            res["avg_long_return"][self.signal_col] = avg_long_return

        if "avg_short_return" in metrics:
            short_mask = self.signal_df[ "position" ] < 0
            avg_short_return = self.signal_df[net_return_col].filter(short_mask).mean() if short_mask.any() else 0
            res["avg_short_return"][self.signal_col] = avg_short_return

        if "pct_in_the_market" in metrics:
            non_zero_mask = self.signal_df[net_return_col] != 0
            pct_in_the_market = non_zero_mask.sum() / self.signal_df.height if self.signal_df.height > 0 else 0
            res["pct_in_the_market"][self.signal_col] = pct_in_the_market

        return res
    
    def eval_signals(self, metrics: dict = None) -> pl.DataFrame:
        """ compute metrics for signals """

        # update metrics dict
        if not metrics:
            metrics = DEFAULT_METRICS.copy()

        res = self._compute_metrics(metrics)
        res = pl.from_dicts(
                [{"index": k, **v} for k, v in res.items()]
            )
        return res


@dataclass
class OptimizationResult:
    """ Container for optimization results """
    strategy_name: str
    best_params: dict
    best_score: float
    all_trials: pd.DataFrame
    val_performance: dict
    training_performance: dict
    param_stability: dict # Plateau Sharpe / mean Sharpe of all trials; a higher value indicates robustness of the best parameters


class StrategyParamTuner:
    """ Class for tuning strategy parameters using Optuna """

    def __init__(self, 
                 symbol: str,
                 data_df: pd.DataFrame | pl.DataFrame,
                 strategy_name: str, 
                 params_space: dict,
                 exit_params: dict = None,
                 interval: str = "15m",
                 target_metric: str = "Sharpe_ratio",
                 n_bars: int = 1,
                 ):
        
        """ Initialize with strategy class and data """
        self.symbol = symbol
        self.data = pl.from_pandas(data_df) if isinstance(data_df, pd.DataFrame) else data_df
        self.strategy_name = strategy_name
        self.params_spaces = params_space # e.g. {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.5}
        self.target_metric = target_metric
        self.n_bars = n_bars
        self.interval = interval
        self.default_metrics = DEFAULT_METRICS.copy()

        if exit_params:
            self.exit_params = exit_params
        # Initialize storage for results
        self.optimization_results = []
        self.best_params_history = []
        logger.info(f"Initialized optimizer for {self.strategy_name}")

    def _generate_signals(self, 
                          data: pl.DataFrame, 
                          params: dict,
                          liquidity_filter: bool = False,
                          liquidity_filter_path: str = None,
                          ) -> pl.DataFrame:
        """ Generate signals based on the strategy class and parameters """

        required_columns = ["datetime", "open_price", "high_price", "low_price", "close_price", "volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        features = { self.strategy_name: params }

        exit_params = {self.strategy_name: self.exit_params} if self.exit_params else None
        signals_df = SignalGenerator(data, exit_params = exit_params)

        # extensions = { self.strategy_name: 8 } # extend signals for 8 hours by default
        signals_df.equipFeatures(features)

        return signals_df


    def _create_trial_params(self, trial: optuna.Trial, params_space: dict) -> dict:
            """ Create parameters from trial based on parameter space definition
            Args:
                - trial: Optuna trial object
                - params_space: Dictionary defining the parameter space
                e.g. {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.5}
            
            """
            params = {}
            
            for param_name, param_config in params_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            step = param_config.get('step')
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step = param_config.get('step', 1)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                else:
                    # Old format: lambda function (for backward compatibility)
                    params[param_name] = param_config(trial)
            
            return params  
    
    def run_single_split_param_tuner(self, 
                     split_data: tuple[pl.DataFrame, pl.DataFrame],
                     optimizer_direction: str = "maximize",
                     sampler: optuna.samplers = optuna.samplers.NSGAIISampler, 
                     n_trials = 100) -> OptimizationResult:
        """ Run the parameter optimizer 
        Args:
            - optimizer_direction: "maximize" or "minimize"
            - sampler: Optuna sampler to use, e.g., NSGAIISampler (genetic algorithm)
            - n_trials: Number of trials to run
        Returns:
            - OptimizationResult: Contains best parameters, score, and other details
        """

        train_data, test_data = split_data
        train_period = f"{train_data['datetime'].min()} to {train_data['datetime'].max()}"
        
        logger.info(f"Optimizing {self.strategy_name} split: {train_period}")
        
        study = optuna.create_study(
                direction = optimizer_direction,
                sampler = sampler(seed = 42),
            )
        
        # Storage for all metrics across trials
        all_metrics_history = []
        
        def objective(trial: optuna.Trial) -> float: 
            """ Objective function for Optuna """

            params = self._create_trial_params(trial, self.params_spaces)

            # Generate signals
            signal_df = self._generate_signals(train_data, params)._df
            
            # Convert signal_df to polars if needed
            if isinstance(signal_df, pd.DataFrame):
                signal_df = pl.from_pandas(signal_df)

            # 存储这次试验的所有指标
            trial_metrics = PerformanceCalculator.calculate_all_metrics(signal_df, f"{self.strategy_name}_pos", interval = self.interval, transaction_cost=TRANSACTION_COST)
            trial_metrics['trial_number'] = trial.number
            all_metrics_history.append(trial_metrics)
            
            # 返回目标指标用于优化
            target_value = trial_metrics.get(self.target_metric, -999.0)

            return target_value

        # Optimize
        study.optimize(objective, n_trials = n_trials)

        best_params = study.best_params
        best_score = study.best_value
        
        all_trials = study.trials_dataframe()
        
        # 添加所有指标到试验数据框中
        metrics_df = pl.DataFrame(all_metrics_history)
        all_trials = pl.from_pandas(all_trials)

        # 将指标数据与试验数据合并
        cols_to_drop = [ "datetime_start", "datetime_complete", "state" ]
        all_trials = all_trials.join(
            metrics_df, 
            left_on="number", 
            right_on="trial_number", 
            how="left"
        ).drop(cols_to_drop)        
        # Sort by target metric 
        all_trials = all_trials.sort(self.target_metric, descending = True)
        
        # Parameter stability analysis
        param_cols = [col for col in all_trials.columns if col.startswith("params")]
        cols = ["number", "value"] + param_cols
        df_params  = all_trials.select(cols)

        ## 1. Rank transformation -> min-max normalization
        normalize_exprs = []
        for col in param_cols:
            col_rank = f"{col}_rank"
            rank_expr = pl.col(col).rank()
            normalize_exprs.append(
                ((rank_expr - rank_expr.min()) / (rank_expr.max() - rank_expr.min())).alias(col_rank)
            )
        df_params = df_params.with_columns(normalize_exprs)

        ## 2. Compute dist to best params vector
        cols_rank = [ col for col in df_params.columns if col.endswith("rank")]
        best_trial_vec = df_params.slice(0, 1).select(cols_rank)

        dist_expr = sum(
            (pl.col(col) - best_val)**2
            for col, best_val in zip(cols_rank, best_trial_vec)
        ).sqrt().alias("dist_to_best")

        df_params = df_params.with_columns(
            dist_expr
        )

        ## 3. Compute mean Sharpe of the plateau; plateau consists of the 10 nearest points with the best param
        ##    store 1 - |(plateau_Sharpe - best_score) / best_score|;
        ##    so score = 1 <=> plateau Sharpe is equal to the best Sharpe implying stability for best params, and vice versa.
        df_sort = df_params.sort("dist_to_best").slice(1, 10)
        plateau_Sharpe = df_sort.select(pl.col("value").mean()).item()
        
        best_params_score = 1 - abs((plateau_Sharpe - best_score) / best_score) if best_score != 0 else 0

        # 在validation set上评估最佳参数的所有指标
        if test_data is not None and test_data.height > 0:
            test_signal_df = self._generate_signals(test_data, best_params)._df
            if isinstance(test_signal_df, pd.DataFrame):
                test_signal_df = pl.from_pandas(test_signal_df)
                
            test_perf = PerformanceCalculator.calculate_all_metrics(test_signal_df, f"{self.strategy_name}_pos", interval = self.interval)


        result = OptimizationResult(
            strategy_name = self.strategy_name,
            best_params   = best_params,
            best_score    = best_score,
            all_trials    = all_trials,
            val_performance = test_perf,  # validation set上的所有指标
            training_performance = all_metrics_history[study.best_trial.number] if all_metrics_history else {},
            param_stability = best_params_score
        )

        self.optimization_results.append(result)
        self.best_params_history.append(best_params)

        logger.info(f"Optimization completed for {self.strategy_name}")
        logger.info(f"Best {self.target_metric}: {best_score:.4f}")
        logger.info(f"Best Parameter Score: {best_params_score:.4f}")

        # 打印最佳试验的所有指标
        if all_metrics_history:
            best_trial_metrics = all_metrics_history[study.best_trial.number]
            logger.info("Best trial all metrics:")
            for metric, value in best_trial_metrics.items():
                if metric not in ['trial_number', 'params']:
                    logger.info(f"  {metric}: {value:.4f}")
        
        return result




def load_symbol_data(symbol: str, interval: str = "15m") -> pl.DataFrame:
    """ Load symbol data from local parquet file """

    file_path = f"{ROOT_DATA_PATH}/{symbol}.parquet"
    logging.info( f"Loading {symbol} data" )
    cols = [ "start_tm", "quote", "vol" ]

    def _resample( df: pl.DataFrame, interval: int ) -> pl.DataFrame:
        """ Resample the DataFrame to the specified interval """
        
        df = (df.sort("datetime")
               .group_by_dynamic(
                   "datetime",
                   every = interval,
                   period = interval,
               )
               .agg(
                   [
                    pl.col("open_price").first().alias("open_price"),
                    pl.col("high_price").max().alias("high_price"),
                    pl.col("low_price").min().alias("low_price"),
                    pl.col("close_price").last().alias("close_price"),
                    pl.col("volume").sum().alias("volume"),
                    pl.col("volume_usd").sum().alias("volume_usd"),
                    pl.col("vwap").mean().alias("vwap")
                   ]
               )
        )

        return df

    try:
        df = pl.read_parquet(file_path)
        df = df.select( 
            pl.col('start_tm').alias('datetime'),
            pl.col('open').alias("open_price"),
            pl.col('high').alias("high_price"), 
            pl.col('low').alias("low_price"),
            pl.col('close').alias("close_price"),
            pl.col('vol').alias('volume'),
            pl.col('quote').alias('volume_usd'),
            pl.when((pl.col("quote") == 0) | (pl.col("vol") == 0))
            .then(pl.col("close"))
            .otherwise((pl.col("quote")/pl.col("vol")))
            .alias("vwap")
        )

        if interval:
            df = _resample(df, interval)
        # Add symbol column if missing
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit( symbol ).alias("symbol"))

        logging.info(f"Data loaded successfully")
        logging.info(f"Total bars: {df.height:,}")
        logging.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")

def single_train_val_split(df: pl.DataFrame,
                            train_start: Optional[pl.datetime] = None,
                            train_end: Optional[pl.datetime] = None,
                            ) -> tuple[pl.DataFrame, pl.DataFrame]:
    
    """ Split DataFrame into training and testing sets based on date range """

    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column for splitting.")
    if not train_start or train_start < df['datetime'].min():
        train_start = df['datetime'].min()
    if not train_end or train_end > df['datetime'].max():
        train_end = df['datetime'].max()

    train_data = df.filter(
        pl.col("datetime") < train_end,
        pl.col("datetime") >= train_start
    ).sort("datetime")

    val_data = df.filter( 
        pl.col("datetime") >= train_end
    ).sort("datetime")

    start_bar = train_data["datetime"].min()
    end_bar = train_data["datetime"].max()
    test_start_bar = val_data["datetime"].min()
    test_end_bar = val_data["datetime"].max()

    logging.info(f"Training data: {train_data.height:,} bars ({start_bar} to {end_bar})")
    logging.info(f"Testing data: {val_data.height:,} bars ({test_start_bar} to {test_end_bar})")

    return train_data, val_data

def optimize_single_symbol(args) -> tuple[ str, OptimizationResult ]:
    """Optimize parameters for a single symbol"""
    try:
        symbol, train_data, val_data, strategy_class, params_space, exit_params, target_metric, n_trials = args
        
        logging.info(f"Starting optimization for {symbol}...")
        
        # Create tuner for this symbol
        tuner = StrategyParamTuner(
            symbol = symbol,
            data_df = train_data,
            strategy_name = strategy_class,
            params_space = params_space,
            exit_params = exit_params,
            target_metric = target_metric,
            n_bars = 1
        )
        tuner.default_metrics = DEFAULT_METRICS.copy()
        
        TPEsampler = optuna.samplers.TPESampler
        TPEsampler.multivariate = True
        TPEsampler.constant_liar = True
        # Run optimization
        result = tuner.run_single_split_param_tuner(
            split_data = (train_data, val_data),
            optimizer_direction = "maximize",
            sampler = TPEsampler,
            # sampler = optuna.samplers.NSGAIISampler,
            n_trials = n_trials
        )
        
        print(f"Completed optimization for {symbol}: Best {target_metric} = {result.best_score:.4f}")

        return symbol, result
    
    except Exception as e:
        import traceback
        print(f"Error in {args}: {e}\n{traceback.format_exc()}")
        raise  # Re-raise to stop Pool.map

#######################################
###### Parallelization Utilities ######
#######################################

def _optimize_symbol_with_data_loading(args_tuple):
    """
    Module-level wrapper function for joblib that handles data loading per symbol
    
    Args:
        args_tuple: (symbol, train_start, train_end, strategy_class, params_space, target_metric, n_trials)
    
    Returns:
        tuple: (symbol, result) where result is OptimizationResult or error dict
    """
    symbol, train_start, train_end, strategy_class, params_space, target_metric, n_trials = args_tuple
    
    try:
        logging.info(f"Loading and preparing data for {symbol}...")
        
        # Load and split data for this symbol
        symbol_data = load_symbol_data(symbol)
        train_data, val_data = single_train_val_split(
            df = symbol_data, 
            train_start = train_start, 
            train_end = train_end
        )
                
        # Prepare arguments for optimization
        optimization_args = (
            symbol,
            train_data,
            val_data,
            strategy_class,
            params_space,
            target_metric,
            n_trials
        )
        
        # Run optimization
        symbol_result, result = optimize_single_symbol(optimization_args)
        
        if result is not None:
            logging.info(f"✅ Completed {symbol_result}: {result.best_score:.4f}")
            return symbol_result, result
        else:
            logging.error(f"❌ Failed {symbol}: optimization returned None")
            return symbol, {"error": "Optimization returned None"}
            
    except Exception as e:
        logging.error(f"❌ Error optimizing {symbol}: {e}")
        import traceback
        logging.debug(f"Full traceback for {symbol}:\n{traceback.format_exc()}")

        return symbol, {"error": str(e)}


def run_coinlevel_optimization(universe: list[str],
                            train_start: dt.datetime,
                            train_end: dt.datetime,
                            strategy_class: type,
                            params_space: dict,
                            target_metric: str = "Sharpe_ratio",
                            n_trials: int = 50, 
                            n_jobs: int = None) -> dict:
    """Run parameter optimization using joblib parallel processing
    
    Args:
        universe: List of symbols to optimize
        train_start: Training data start date
        train_end: Training data end date (validation starts from here)
        strategy_class: Strategy class to optimize
        params_space: Parameter space dictionary
        target_metric: Metric to optimize for
        n_trials: Number of trials per symbol
        n_jobs: Number of parallel jobs (-1 for all cores, None for auto)
    
    Returns:
        dict: Dictionary with symbol -> OptimizationResult mapping
    """
    
    if n_jobs is None:
        n_jobs = min(cpu_count() // 2, len(universe))
    elif n_jobs == -1:
        n_jobs = cpu_count()
    
    logging.info(f"Running joblib optimization for {len(universe)} symbols using {n_jobs} jobs")
    logging.info(f"Trials per symbol: {n_trials}")
    
    # Prepare failed symbols list for data loading issues
    valid_symbols = []
    failed_symbols = []
    
    # Quick validation of symbols (check if data files exist)
    for symbol in universe:
        try:
            file_path = f"{ROOT_DATA_PATH}/{symbol}.parquet"
            import os
            if os.path.exists(file_path):
                first_timestamp = pl.read_parquet(file_path)['start_tm'].min()
                # not enough data training data, skip
                if not first_timestamp or (first_timestamp > train_end - pd.DateOffset(months=6)):
                    logging.warning(f"Not enough training data for {symbol}, skipping...")
                    failed_symbols.append(symbol)
                    continue
                valid_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
                logging.warning(f"Data file not found for {symbol}")
        except Exception as e:
            failed_symbols.append(symbol)
            logging.warning(f"Error checking {symbol}: {e}")
    
    if not valid_symbols:
        logging.error("No valid symbols found!")
        return {}
    
    logging.info(f"Processing {len(valid_symbols)} valid symbols")
    if failed_symbols:
        logging.warning(f"Skipped symbols due to missing data: {failed_symbols}")
    
    # Prepare arguments for each symbol
    job_args = [
        (symbol, train_start, train_end, strategy_class, params_space, target_metric, n_trials)
        for symbol in valid_symbols
    ]
    
    # Run optimization in parallel using joblib
    try:
        if n_jobs == 1:
            # Sequential processing
            logging.info("Running sequential optimization...")
            results_list = [_optimize_symbol_with_data_loading(args) for args in job_args]
        else:
            # Parallel processing with joblib
            logging.info(f"Running parallel optimization with {n_jobs} jobs...")
            results_list = Parallel(
                n_jobs = n_jobs, 
                verbose = 1,  # Show progress
                backend = 'loky',  
                timeout = 1200  # 20 minute timeout per job
            )(
                delayed(_optimize_symbol_with_data_loading)(args) 
                for args in job_args
            )
    
    except Exception as e:
        import psutil
        logging.error(f"Joblib parallel processing failed: {e}")
        # Capture comprehensive error information
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'full_traceback': traceback.format_exc(),
            'system_info': {
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'active_processes': len(psutil.pids()),
            },
        }
        
        # Log everything
        logging.error("="*60)
        logging.error("JOBLIB PARALLEL PROCESSING FAILED - DETAILED ERROR")
        logging.error("="*60)
        logging.error(f"Error Type: {error_info['error_type']}")
        logging.error(f"Error Message: {error_info['error_message']}")
        logging.error(f"System Memory: {error_info['system_info']['memory_percent']:.1f}% used")
        logging.error(f"Available Memory: {error_info['system_info']['available_memory_gb']:.1f}GB")
        logging.error(f"CPU Count: {error_info['system_info']['cpu_count']}")
        logging.error(f"Active Processes: {error_info['system_info']['active_processes']}")
        logging.error(f"Requested n_jobs: {error_info['job_config']['n_jobs']}")
        logging.error(f"Universe size: {error_info['job_config']['universe_size']}")
        logging.error("Full Traceback:")
        logging.error(error_info['full_traceback'])
        logging.error("="*60)
        
        sys.exit()

    
    # Organize results into dictionary
    optimization_results = {}
    successful_count = 0
    
    for symbol, result in results_list:
        optimization_results[symbol] = result
        if not isinstance(result, dict) or "error" not in result:
            successful_count += 1
    
    # Summary
    total_symbols = len(universe)
    failed_data_loading = len(failed_symbols)
    failed_optimization = len(optimization_results) - successful_count
    
    logging.info(f"Optimization completed!")
    logging.info(f"Total symbols: {total_symbols}")
    logging.info(f"Successful optimizations: {successful_count}")
    logging.info(f"Failed data loading: {failed_data_loading}")
    logging.info(f"Failed optimization: {failed_optimization}")
    
    if failed_symbols:
        logging.warning(f"Symbols with data loading issues: {failed_symbols}")
    
    return optimization_results


#######################################
######  Adding Liquidity Filter  ######
#######################################
perf, metadata = {}, {}
def load_optimization_results(file_path):
    """Load optimization results from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded results from: {file_path}")
        print(f"   Strategy: {data['metadata']['strategy']}")
        print(f"   Total symbols: {data['metadata']['total_symbols']}")
        print(f"   Successful: {data['metadata']['successful_symbols']}")
        print(f"   Date created: {data['metadata']['timestamp']}")
        
        return data['optimization_results'], data['metadata']
    
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None

def get_result_after_liquidity_filter(
        symbol: str,
        params_search_res_root_path: str,
        strategy_class: type,
        liquidity_filter_path: str
    ) -> dict[pl.DataFrame]:
    """ 
    Compare the performance after adding liquidity filter.
    """

    strat_name = strategy_class.__name__
    
    # TODO:
    params_search_path = params_search_res_root_path + f"{strat_name}_54coins.pkl"
    perf, metadata = load_optimization_results(params_search_path)

    # for symbol in universe:
    symbol_data = load_symbol_data(symbol)
    train_start, train_end = metadata["train_start"], metadata["train_end"]

    _, val_data = single_train_val_split(
        df = symbol_data, 
        train_start = train_start, 
        train_end = train_end
    )
    
    if not isinstance(perf[symbol], OptimizationResult):
        print(f"❌ No performance data for {symbol}, skipping...")
        return None
    # read symbol best parameters
    training_best_params = perf[symbol].best_params

    # create an instance of StrategyParamTuner
    tuner = StrategyParamTuner(
        symbol = symbol,
        data_df = val_data,
        strategy_class = strategy_class,
        params_space = training_best_params,
        target_metric = 'sharpe_ratio', # doesn't matter here
    )

    signals_df = (tuner._generate_signals(val_data, training_best_params, liquidity_filter = True, liquidity_filter_path = liquidity_filter_path)
                    .select( ["datetime", f"{strat_name}_liquidity_filtered"] ) )

    eval = SignalEvaluator( 
    data_df = val_data,
    signal_df = signals_df,
    )

    res = eval.eval_signals()
    val_perf = perf[symbol].val_performance
    val_perf_df = pl.DataFrame({
        "index": list(val_perf.keys()),  # Use same column name as df's index column
        "val_performance": list(val_perf.values())
    })
    res = res.join(
                    val_perf_df,
                    on = "index",
                    how = "left"
                    )

    return res


if __name__ == "__main__":
    from strategy.strategies import (SuperTrendStrategy,
                                VolumeLongShort,
                                PSY_MOM_LongShort,
                                SWING_LongShort,
                                MR_LongShort,
                                SMA_Strategy,
                                Turtle_LongShort,
                                BrickLongShort,
                                SuperTrend_ADPEXIT_LongShort)
    # coin-level parallelization
    import os
    import time
    import gc
    import traceback

    path = "/home/whq/data/crypto/bar1m/futures"
    # get the full universe of coin names
    files = os.listdir(path)
    full_universe = [f.split(".")[0] for f in files if f.endswith(".parquet")]
    full_universe.sort()


    strategies = [  SuperTrendStrategy
                    # VolumeLongShort,
                    # PSY_MOM_LongShort,
                    # SWING_LongShort,
                    # MR_LongShort,
                    # SMA_Strategy,
                    # Turtle_LongShort,
                    # BrickLongShort,
                    # SuperTrend_ADPEXIT_LongShort]
                ]
    train_start = dt.datetime(2023, 1, 1)
    cutoff_date = dt.datetime(2024, 7, 1)


    for i, strat_ in enumerate(strategies):
        try:
            print(f"🚀 Processing strategy {i+1}/{len(strategies)}: {strat_.__name__}")
            
            params_space = PARAMS_SPACES[strat_.__name__]
            
            # 确保保存目录存在
            save_dir = "/home/whq/projects/files/result/params_search/"
            os.makedirs(save_dir, exist_ok=True)
            
            start_time = time.time()
            
            res_d = run_coinlevel_optimization(
                universe = full_universe,
                train_start = train_start,
                train_end = cutoff_date,
                strategy_class = strat_,
                params_space = params_space,
                target_metric = "Sharpe_ratio",
                n_trials = 50,
                n_jobs = 50
            )
            
            end_time = time.time()
            elapsed = (end_time - start_time) / 3600  # hours
            
            # 保存结果
            save_path = f"{save_dir}{strat_.__name__}_fullUniverse_results.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'optimization_results': res_d,
                    'metadata': {
                        'strategy': strat_.__name__,
                        'universe': full_universe,
                        'train_start': train_start,
                        'train_end': cutoff_date,
                        'target_metric': 'Sharpe_ratio',
                        'n_trials': 50,
                        'n_jobs': 50,
                        'timestamp': dt.datetime.now(),
                        'elapsed_hours': elapsed,
                        'total_symbols': len(res_d),
                        'successful_symbols': len([k for k, v in res_d.items() if hasattr(v, 'best_score')])
                    }
                }, f)
            
            successful = len([k for k, v in res_d.items() if hasattr(v, 'best_score')])
            print(f"✅ Completed {strat_.__name__} in {elapsed:.1f}h")
            print(f"   Results: {successful}/{len(full_universe)} symbols successful")
            print(f"   Saved to: {save_path}\n")
            
            # 清理内存
            del res_d
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing {strat_.__name__}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # 继续处理下一个策略
            gc.collect()
            continue

        print("🎉 All strategies completed!")
