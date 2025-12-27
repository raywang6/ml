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
import optuna
import logging
from typing import Optional
from multiprocessing import Pool, cpu_count

TRANSACTION_COST = 0.0002  
from joblib import Parallel, delayed
from tqdm import tqdm 

from dataclasses import dataclass
from runner import StrategyRunner
from concurrent.futures import ProcessPoolExecutor, as_completed

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval

ROOT_DATA_PATH = "/home/whq/data/crypto/bar1m/futures/"
from projects.cta.performance_metrics import PerformanceCalculator

from ParamsSearch.params_spaces import PARAMS_SPACES
from strategy.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """ Container for optimization results """
    strategy_name: str
    best_params: dict
    best_score: float
    all_trials: pd.DataFrame
    val_performance: dict
    training_performance: dict
    best_params_score: float # Plateau Sharpe / mean Sharpe of all trials; a higher value indicates robustness of the best parameters


class StrategyParamTuner:
    """ Class for tuning strategy parameters using Optuna """

    def __init__(self, 
                 symbol: str,
                 data_df: pd.DataFrame | pl.DataFrame,
                 strategy_name: str, 
                 params_space: dict,
                 interval: str = "15m",
                 exit_params: dict = None,
                 target_metric: str = "Sharpe_ratio",
                 n_bars: int = 1,
                 ):
        
        """ Initialize with strategy class and data """
        self.symbol = symbol
        self.data = pl.from_pandas(data_df) if isinstance(data_df, pd.DataFrame) else data_df
        self.strategy_name = strategy_name
        self.params_spaces = params_space # e.g. {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.5}
        self.interval = interval
        self.target_metric = target_metric
        self.n_bars = n_bars

        if exit_params:
            self.exit_params = exit_params

        # Initialize storage for results
        self.optimization_results = []
        self.best_params_history = []
        logger.info(f"Initialized optimizer for {self.strategy_name}")

    def _generate_signals(self, 
                          data: pl.DataFrame, 
                          params: dict,
                          ) -> pl.DataFrame:
        """ Generate signals based on the strategy class and parameters """

        required_columns = ["datetime", "open_price", "high_price", "low_price", "close_price", "volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        # Retrieve strat and exit params
        features = { self.strategy_name: params }
        exit_params = {self.strategy_name: self.exit_params} if self.exit_params else None
        signals_df = SignalGenerator(data, interval = self.interval, exit_params = exit_params)

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
            best_params_score = best_params_score
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
        symbol, train_data, val_data, strategy_class, params_space, interval, exit_params, target_metric, n_trials = args
        
        logging.info(f"Starting optimization for {symbol}...")
        
        # Create tuner for this symbol
        tuner = StrategyParamTuner(
            symbol = symbol,
            data_df = train_data,
            strategy_name = strategy_class,
            params_space = params_space,
            interval = interval,
            exit_params = exit_params,
            target_metric = target_metric,
            n_bars = 1
        )
        
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
        args_tuple: (symbol, train_start, train_end, strategy_class, params_space, interval, exit_params, target_metric, n_trials)
    
    Returns:
        tuple: (symbol, result) where result is OptimizationResult or error dict
    """
    symbol, train_start, train_end, strategy_class, params_space, interval, exit_params, target_metric, n_trials = args_tuple
    
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
            interval,
            exit_params,
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
                            interval: str = "15m",
                            exit_params: dict = None,
                            target_metric: str = "Sharpe_ratio",
                            n_trials: int = 100, 
                            n_jobs: int = None) -> dict:
    """Run parameter optimization using joblib parallel processing
    
    Args:
        universe: List of symbols to optimize
        train_start: Training data start date
        train_end: Training data end date (validation starts from here)
        strategy_class: Strategy class to optimize
        params_space: Parameter space dictionary
        interval: time interval of the data
        exit_params: Exit parameters for the strategy
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
        (symbol, train_start, train_end, strategy_class, params_space, interval, exit_params, target_metric, n_trials)
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

