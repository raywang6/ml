"""
Enhanced Parameter Tuning with Robustness Testing
Improvements:
1. Multi-objective optimization (performance + robustness)
2. Grid-based robustness testing instead of noise
3. Efficient parameter space exploration
4. Early stopping for poor performers
5. Cross-validation with walk-forward analysis
"""

import datetime as dt
import polars as pl
import pandas as pd
import numpy as np
import optuna
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import itertools
from scipy.stats import rankdata

from parameters_tuning import (
    StrategyParamTuner, OptimizationResult, load_symbol_data, 
    single_train_val_split, DEFAULT_METRICS, SignalEvaluator
)

logger = logging.getLogger(__name__)

ROBUST_TEST_SIZE = 96 * 30 * 6
PERFORMANCE_WEIGHT = 1.
ROBUSTNESS_WEIGHT = 0.0

@dataclass
class RobustnessResult:
    """Container for robustness analysis results"""
    base_params: dict
    robustness_score: float
    performance_stability: dict
    parameter_sensitivity: dict
    grid_results: pd.DataFrame

class EnhancedStrategyTuner(StrategyParamTuner):
    """Enhanced tuner with robustness testing and efficiency improvements"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robustness_grid_size = 0.1  # 10% parameter variation
        self.min_trials_for_robustness = 20
        
    def create_robustness_grid(self, base_params: dict, grid_size: float = 0.1) -> List[dict]:
        """
        Create parameter grid around best parameters for robustness testing.
        Uses grid movement instead of random noise for more systematic testing.
        
        Args:
            base_params: Best parameters found during optimization
            grid_size: Relative size of parameter variations (0.1 = ±10%)
        """
        grid_params = []
        
        for param_name, base_value in base_params.items():
            if param_name not in self.params_spaces:
                continue
                
            param_config = self.params_spaces[param_name]
            variations = []
            
            if param_config['type'] == 'float':
                # Create ±grid_size variations
                delta = abs(base_value * grid_size)
                variations = [
                    max(param_config['low'], base_value - delta),
                    base_value,
                    min(param_config['high'], base_value + delta)
                ]
            elif param_config['type'] == 'int':
                # Create ±1 or ±2 step variations for integers
                step = param_config.get('step', 1)
                delta = max(1, int(abs(base_value * grid_size)))
                variations = [
                    max(param_config['low'], base_value - delta * step),
                    base_value,
                    min(param_config['high'], base_value + delta * step)
                ]
            else:
                variations = [base_value]  # No variation for categorical
            
            grid_params.append((param_name, variations))
        
        # Generate all combinations
        param_names = [p[0] for p in grid_params]
        param_values = [p[1] for p in grid_params]
        
        robustness_params = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            robustness_params.append(param_dict)
        
        return robustness_params
    
    def evaluate_robustness(self, base_params: dict, test_data: pl.DataFrame) -> RobustnessResult:
        """
        Evaluate parameter robustness using grid-based testing.
        Tests performance stability across small parameter variations.
        """
        logger.info(f"Evaluating robustness for {self.strategy_name}")
        
        # Generate robustness grid
        robustness_params = self.create_robustness_grid(base_params, self.robustness_grid_size)
        
        results = []
        for params in robustness_params:
            try:
                # Generate signals with these parameters
                signal_df = self._generate_signals(test_data, params)._df
                                
                # Calculate performance metrics using SignalEvaluator instead
                evaluator = SignalEvaluator(signal_df, self.strategy_name)
                metrics_df = evaluator.eval_signals(DEFAULT_METRICS)
                
                # Convert to dict format
                metrics = {}
                for row in metrics_df.iter_rows(named=True):
                    metric_name = row['index']
                    metric_value = row.get(self.strategy_name, 0.0)
                    metrics[metric_name] = metric_value if metric_value is not None else 0.0
                
                result = {
                    'params': params,
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")
                continue
        
        if not results:
            logger.error("No successful robustness evaluations")
            return None
        
        results_df = pd.DataFrame(results)
        
        # Calculate robustness metrics
        performance_cols = [col for col in results_df.columns if col != 'params']
        
        # Robustness score: inverse of coefficient of variation for key metrics
        key_metrics = ['Sharpe_ratio']#, 'Calmar_ratio', 'total_return']
        robustness_scores = []
        
        for metric in key_metrics:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                # debug
                """
                if len(values) > 1 and values.std() > 0:
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                    robustness_scores.append(1 / (1 + cv))  # Higher score = more robust
                else:
                    robustness_scores.append(1.0)
                """
                robustness_scores.append(values.min())
        robustness_score = np.mean(robustness_scores) if robustness_scores else 0.0
        
        # Parameter sensitivity analysis
        param_sensitivity = {}
        for param_name in base_params.keys():
            if param_name in self.params_spaces:
                param_values = [r['params'][param_name] for r in results]
                sharpe_values = results_df['Sharpe_ratio'].values
                
                if len(set(param_values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(param_values, sharpe_values)[0, 1]
                    param_sensitivity[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    param_sensitivity[param_name] = 0.0
        
        # Performance stability
        performance_stability = {}
        for metric in key_metrics:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                performance_stability[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'cv': values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                }
        
        return RobustnessResult(
            base_params=base_params,
            robustness_score=robustness_score,
            performance_stability=performance_stability,
            parameter_sensitivity=param_sensitivity,
            grid_results=results_df
        )
    
    def multi_objective_optimization(self, 
                                   split_data: Tuple[pl.DataFrame, pl.DataFrame],
                                   n_trials: int = 100,
                                   performance_weight: float = 0.7,
                                   robustness_weight: float = 0.3) -> OptimizationResult:
        """
        Multi-objective optimization balancing performance and robustness.
        
        Args:
            performance_weight: Weight for performance score (0-1)
            robustness_weight: Weight for robustness score (0-1)
        """
        train_data, test_data = split_data
        
        logger.info(f"Starting multi-objective optimization for {self.strategy_name}")
        logger.info(f"Performance weight: {performance_weight}, Robustness weight: {robustness_weight}")
        
        # Storage for robustness evaluations
        robustness_cache = {}
        
        def objective(trial: optuna.Trial) -> float:
            params = self._create_trial_params(trial, self.params_spaces)
            
            try:
                # Generate signals on training data
                signal_df = self._generate_signals(train_data, params)._df
                
                # Calculate performance metrics using SignalEvaluator
                evaluator = SignalEvaluator(signal_df, self.strategy_name)
                metrics_df = evaluator.eval_signals(DEFAULT_METRICS)
                
                # Convert to dict format
                metrics = {}
                for row in metrics_df.iter_rows(named=True):
                    metric_name = row['index']
                    metric_value = row.get(self.strategy_name, 0.0)
                    metrics[metric_name] = metric_value if metric_value is not None else 0.0
                
                performance_score = metrics.get(self.target_metric, -999.0)
                
                # Early stopping for very poor performers
                if performance_score < -10:
                    return -999.0
                
                # Robustness evaluation (only for promising candidates)
                robustness_score = 0.0
                if trial.number >= self.min_trials_for_robustness and performance_score > 0:
                    # Use a subset of test data for robustness to save time
                    test_subset = test_data.sample(min(ROBUST_TEST_SIZE, test_data.height))
                    robustness_result = self.evaluate_robustness(params, test_subset)
                    
                    if robustness_result:
                        robustness_score = robustness_result.robustness_score
                        robustness_cache[trial.number] = robustness_result
                
                # Combined objective
                combined_score = (performance_weight * performance_score + 
                                robustness_weight * robustness_score)
                
                return combined_score
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -999.0
        
        # Create study with pruning for efficiency
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and evaluate on full test set
        best_params = study.best_params
        best_score = study.best_value
        
        # Full robustness evaluation on best parameters
        final_robustness = None
        if test_data.height > 0:
            final_robustness = self.evaluate_robustness(best_params, test_data)
        
        # Create enhanced result
        result = OptimizationResult(
            strategy_name=self.strategy_name,
            best_params=best_params,
            best_score=best_score,
            all_trials=study.trials_dataframe(),
            val_performance={},  # Will be filled by caller
            training_performance={},  # Will be filled by caller
            param_stability={},
        )
        
        # Add robustness information
        if final_robustness:
            result.robustness_result = final_robustness
        
        logger.info(f"Multi-objective optimization completed")
        logger.info(f"Best combined score: {best_score:.4f}")
        if final_robustness:
            logger.info(f"Robustness score: {final_robustness.robustness_score:.4f}")
        
        return result

def walk_forward_analysis(symbol: str,
                         data: pl.DataFrame,
                         strategy_name: str,
                         params_space: dict,
                         exit_params: dict,
                         window_months: int = 6,
                         step_months: int = 1,
                         target_metric: str = "Sharpe_ratio") -> Dict:
    """
    Perform walk-forward analysis for more robust parameter validation.
    
    Args:
        window_months: Training window size in months
        step_months: Step size for moving window in months
    """
    logger.info(f"Starting walk-forward analysis for {symbol}")
    
    # Create time windows
    start_date = data['datetime'].min()
    end_date = data['datetime'].max()
    
    windows = []
    current_start = start_date
    
    while current_start < end_date:
        train_end = current_start + pd.DateOffset(months=window_months)
        test_end = train_end + pd.DateOffset(months=step_months)
        
        if test_end > end_date:
            break
            
        train_data = data.filter(
            (pl.col("datetime") >= current_start) & 
            (pl.col("datetime") < train_end)
        )
        
        test_data = data.filter(
            (pl.col("datetime") >= train_end) & 
            (pl.col("datetime") < test_end)
        )
        
        if train_data.height > 100 and test_data.height > 10:  # Minimum data requirements
            windows.append((train_data, test_data, train_end))
        
        current_start += pd.DateOffset(months=step_months)
    
    logger.info(f"Created {len(windows)} walk-forward windows")
    
    # Run optimization on each window
    results = []
    for i, (train_data, test_data, period_end) in enumerate(windows):
        logger.info(f"Processing window {i+1}/{len(windows)} ending {period_end}")
        
        tuner = EnhancedStrategyTuner(
            symbol=symbol,
            data_df=train_data,
            strategy_name=strategy_name,
            params_space=params_space,
            exit_params=exit_params,
            target_metric=target_metric
        )
        
        # Use multi-objective optimization
        result = tuner.multi_objective_optimization(
            split_data=(train_data, test_data),
            n_trials=50,  # Fewer trials per window
            performance_weight=PERFORMANCE_WEIGHT,
            robustness_weight=ROBUSTNESS_WEIGHT
        )
        
        results.append({
            'period_end': period_end,
            'best_params': result.best_params,
            'best_score': result.best_score,
            'robustness_score': getattr(result, 'robustness_result', {}).robustness_score if hasattr(result, 'robustness_result') else 0.0
        })
    
    # Analyze consistency across periods
    if results:
        # Parameter stability across time
        param_consistency = {}
        for param_name in results[0]['best_params'].keys():
            values = [r['best_params'][param_name] for r in results]
            param_consistency[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else float('inf')
            }
        
        # Performance consistency
        scores = [r['best_score'] for r in results]
        robustness_scores = [r['robustness_score'] for r in results]
        
        summary = {
            'symbol': symbol,
            'num_periods': len(results),
            'avg_performance': np.mean(scores),
            'performance_std': np.std(scores),
            'avg_robustness': np.mean(robustness_scores),
            'param_consistency': param_consistency,
            'period_results': results
        }
        
        logger.info(f"Walk-forward analysis completed for {symbol}")
        logger.info(f"Average performance: {summary['avg_performance']:.4f} ± {summary['performance_std']:.4f}")
        logger.info(f"Average robustness: {summary['avg_robustness']:.4f}")
        
        return summary
    
    return {}

def efficient_parameter_search(universe: List[str],
                             train_start: dt.datetime,
                             train_end: dt.datetime,
                             strategy_name: str,
                             params_space: dict,
                             exit_params: dict,
                             target_metric: str = "Sharpe_ratio",
                             use_walk_forward: bool = False,
                             n_trials: int = 100) -> Dict:
    """
    Efficient parameter search with robustness testing.
    
    Args:
        use_walk_forward: Whether to use walk-forward analysis instead of single split
    """
    logger.info(f"Starting efficient parameter search for {len(universe)} symbols")
    
    results = {}
    
    for symbol in universe:
        try:
            logger.info(f"Processing {symbol}")
            
            # Load data
            data = load_symbol_data(symbol)
            if data.height < 1000:  # Skip symbols with insufficient data
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            if use_walk_forward:
                # Use walk-forward analysis
                result = walk_forward_analysis(
                    symbol=symbol,
                    data=data,
                    strategy_name=strategy_name,
                    params_space=params_space,
                    exit_params=exit_params,
                    target_metric=target_metric
                )
            else:
                # Use enhanced single-split optimization
                train_data, test_data = single_train_val_split(data, train_start, train_end)
                
                tuner = EnhancedStrategyTuner(
                    symbol=symbol,
                    data_df=train_data,
                    strategy_name=strategy_name,
                    params_space=params_space,
                    exit_params=exit_params,
                    target_metric=target_metric
                )
                
                result = tuner.multi_objective_optimization(
                    split_data=(train_data, test_data),
                    n_trials=n_trials,
                    performance_weight=PERFORMANCE_WEIGHT,
                    robustness_weight=ROBUSTNESS_WEIGHT
                )
            
            results[symbol] = result
            logger.info(f"Completed {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    logger.info(f"Efficient parameter search completed for {len(results)} symbols")
    return results

# Usage example functions
def create_optimized_params_space(base_space: dict, reduction_factor: float = 0.5) -> dict:
    """
    Create a more focused parameter space based on prior knowledge or results.
    Reduces search space for efficiency while maintaining coverage.
    """
    optimized_space = {}
    
    for param_name, param_config in base_space.items():
        if param_config['type'] in ['int', 'float']:
            # Reduce range by reduction_factor while keeping the same center
            low, high = param_config['low'], param_config['high']
            center = (low + high) / 2
            new_range = (high - low) * reduction_factor
            
            new_low = max(low, center - new_range / 2)
            new_high = min(high, center + new_range / 2)
            
            optimized_space[param_name] = {
                **param_config,
                'low': new_low if param_config['type'] == 'float' else int(new_low),
                'high': new_high if param_config['type'] == 'float' else int(new_high)
            }
        else:
            optimized_space[param_name] = param_config
    
    return optimized_space