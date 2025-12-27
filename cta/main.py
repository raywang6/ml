"""
Example Usage of Enhanced Parameter Tuning System
This script demonstrates how to use the improved parameter optimization
with robustness testing and efficient search strategies.
"""

import datetime as dt
import logging
import pickle
from pathlib import Path
import polars as pl
import pandas as pd
# Import the enhanced modules
from enhanced_parameters_tuning import (
    EnhancedStrategyTuner, efficient_parameter_search, 
    walk_forward_analysis, create_optimized_params_space
)
from parameters_tuning import StrategyParamTuner, load_symbol_data, single_train_val_split
from strategy.params_spaces import PARAMS_SPACES, EXIT_PARAMS
from tuning_config import (
    get_strategy_config, get_universe, evaluate_performance_level,
    DataConfig, OptimizationConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_basic_optimization():
    """
    Basic enhanced optimization with robustness testing
    This replaces the simple optimization in your original run.py
    """
    
    print("\n" + "="*60)
    print("Enhanced Single Symbol Optimization")
    print("="*60)

    # Configuration
    alltrials = pd.DataFrame()
    bestp = {}
    for strategy_name in PARAMS_SPACES:
        bestp[strategy_name] = {}
        for symbol in DataConfig.MAJOR_COINS:
    
            # Get strategy-specific configuration
            config = get_strategy_config(strategy_name, "optimization")

            # Load and prepare data
            data = load_symbol_data(symbol, DataConfig.DEFAULT_INTERVAL)
            train_data, val_data = single_train_val_split(
                data, 
                DataConfig.DEFAULT_TRAIN_START, 
                DataConfig.DEFAULT_TRAIN_END
            )

            logger.info(f"Training data: {train_data.height} bars")
            logger.info(f"Validation data: {val_data.height} bars")

            # Create enhanced tuner
            tuner = StrategyParamTuner(
                symbol=symbol,
                data_df=train_data,
                strategy_name=strategy_name,
                params_space=PARAMS_SPACES[strategy_name],
                exit_params=EXIT_PARAMS[strategy_name],
                target_metric=config.get("target_metric", "Sharpe_ratio")
            )

            # Run multi-objective optimization
            result = tuner.run_single_split_param_tuner( 
                     split_data = (train_data, val_data),
                     optimizer_direction = "maximize",
                     n_trials = config.get("n_trials", 100))
            
            bestp[strategy_name][symbol] = result.best_params
            alltrials = pd.concat([alltrials, result.all_trials.to_pandas()], ignore_index=True)
        alltrials.to_parquet("/home/moneyking/projects/mlframework/cta/result/alltrials_min.parqruet")
        with open("/home/moneyking/projects/mlframework/cta/result/bestp_min.pkl", "wb") as f:
            pickle.dump(bestp, f)

    """
    # Display results
    print(f"\nBest Parameters: {result.best_params}")
    print(f"Combined Score: {result.best_score:.4f}")

    if hasattr(result, 'robustness_result') and result.robustness_result:
        rob = result.robustness_result
        print(f"Robustness Score: {rob.robustness_score:.4f}")
        #
        print("\nParameter Sensitivity (higher = more sensitive):")
        for param, sensitivity in rob.parameter_sensitivity.items():
            sensitivity_level = "High" if sensitivity > 0.7 else "Medium" if sensitivity > 0.4 else "Low"
            print(f"  {param}: {sensitivity:.3f} ({sensitivity_level})")
        #
        print("\nPerformance Stability:")
        for metric, stats in rob.performance_stability.items():
            if metric in ["Sharpe_ratio", "Calmar_ratio"]:
                stability = "Stable" if stats['cv'] < 0.2 else "Moderate" if stats['cv'] < 0.5 else "Unstable"
                print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stability})")
    """
    return result


def run_basic_enhanced_optimization():
    """
    Basic enhanced optimization with robustness testing
    This replaces the simple optimization in your original run.py
    """
    
    print("\n" + "="*60)
    print("Enhanced Single Symbol Optimization")
    print("="*60)

    # Configuration
    alltrials = pd.DataFrame()
    bestp = {}
    for strategy_name in PARAMS_SPACES:
        bestp[strategy_name] = {}
        for symbol in DataConfig.MAJOR_COINS:
    
            # Get strategy-specific configuration
            config = get_strategy_config(strategy_name, "optimization")

            # Load and prepare data
            data = load_symbol_data(symbol, DataConfig.DEFAULT_INTERVAL)
            train_data, val_data = single_train_val_split(
                data, 
                DataConfig.DEFAULT_TRAIN_START, 
                DataConfig.DEFAULT_TRAIN_END
            )

            logger.info(f"Training data: {train_data.height} bars")
            logger.info(f"Validation data: {val_data.height} bars")

            # Create enhanced tuner
            tuner = EnhancedStrategyTuner(
                symbol=symbol,
                data_df=train_data,
                strategy_name=strategy_name,
                params_space=PARAMS_SPACES[strategy_name],
                exit_params=EXIT_PARAMS[strategy_name],
                target_metric=config.get("target_metric", "Sharpe_ratio")
            )

            # Run multi-objective optimization
            result = tuner.multi_objective_optimization(
                split_data=(train_data, val_data),
                n_trials=config.get("n_trials", 100),
                performance_weight=config.get("performance_weight", 0.9),
                robustness_weight=config.get("robustness_weight", 0.1)
            )
            bestp[strategy_name][symbol] = result.best_params
            alltrials = pd.concat([alltrials, result.all_trials], ignore_index=True)
        alltrials.to_parquet("/home/moneyking/projects/mlframework/cta/result/alltrials_base.parqruet")
        with open("/home/moneyking/projects/mlframework/cta/result/bestp_base.pkl", "wb") as f:
            pickle.dump(bestp, f)

    """
    # Display results
    print(f"\nBest Parameters: {result.best_params}")
    print(f"Combined Score: {result.best_score:.4f}")

    if hasattr(result, 'robustness_result') and result.robustness_result:
        rob = result.robustness_result
        print(f"Robustness Score: {rob.robustness_score:.4f}")
        #
        print("\nParameter Sensitivity (higher = more sensitive):")
        for param, sensitivity in rob.parameter_sensitivity.items():
            sensitivity_level = "High" if sensitivity > 0.7 else "Medium" if sensitivity > 0.4 else "Low"
            print(f"  {param}: {sensitivity:.3f} ({sensitivity_level})")
        #
        print("\nPerformance Stability:")
        for metric, stats in rob.performance_stability.items():
            if metric in ["Sharpe_ratio", "Calmar_ratio"]:
                stability = "Stable" if stats['cv'] < 0.2 else "Moderate" if stats['cv'] < 0.5 else "Unstable"
                print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stability})")
    """
    return result


def run_walk_forward_validation():
    """
    Walk-forward analysis for robust parameter validation
    This provides more reliable performance estimates than single train/test split
    """
    
    print("\n" + "="*60)
    print("Walk-Forward Analysis")
    print("="*60)
    
    symbol = "BTCUSDT"
    strategy_name = "SuperTrend"
    
    # Load data
    data = load_symbol_data(symbol, DataConfig.DEFAULT_INTERVAL)
    
    # Use recent data for faster demonstration
    data = data.filter(pl.col("datetime") >= dt.datetime(2023, 6, 1))
    
    logger.info(f"Running walk-forward analysis on {data.height} bars")
    
    # Get walk-forward configuration
    wf_config = get_strategy_config(strategy_name, "walk_forward")
    
    # Run walk-forward analysis
    wf_result = walk_forward_analysis(
        symbol=symbol,
        data=data,
        strategy_name=strategy_name,
        params_space=PARAMS_SPACES[strategy_name],
        exit_params=EXIT_PARAMS,
        window_months=wf_config.get("window_months", 6),
        step_months=wf_config.get("step_months", 1),
        target_metric="Sharpe_ratio"
    )
    
    if wf_result:
        print(f"\nWalk-Forward Results:")
        print(f"Number of periods tested: {wf_result['num_periods']}")
        print(f"Average performance: {wf_result['avg_performance']:.4f} ± {wf_result['performance_std']:.4f}")
        print(f"Average robustness: {wf_result['avg_robustness']:.4f}")
        
        print(f"\nParameter Consistency Across Time:")
        for param, stats in wf_result['param_consistency'].items():
            consistency = "Consistent" if stats['cv'] < 0.3 else "Moderate" if stats['cv'] < 0.6 else "Inconsistent"
            print(f"  {param}: {stats['mean']:.2f} ± {stats['std']:.2f} ({consistency})")
    
    return wf_result

def run_batch_optimization_with_ranking():
    """
    Efficient batch optimization with performance ranking
    This shows how to optimize multiple symbols efficiently
    """
    
    print("\n" + "="*60)
    print("Batch Optimization with Ranking")
    print("="*60)
    
    strategy_name = "SuperTrend"
    
    # Use test universe for demonstration
    universe = get_universe("test")  # ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    logger.info(f"Optimizing {len(universe)} symbols: {universe}")
    
    # Get optimized parameter space for efficiency
    optimized_params = create_optimized_params_space(
        PARAMS_SPACES[strategy_name], 
        reduction_factor=0.6
    )
    
    # Run batch optimization
    results = efficient_parameter_search(
        universe=universe,
        train_start=DataConfig.DEFAULT_TRAIN_START,
        train_end=DataConfig.DEFAULT_TRAIN_END,
        strategy_name=strategy_name,
        params_space=optimized_params,
        exit_params=EXIT_PARAMS,
        target_metric="Calmar_ratio",
        use_walk_forward=False,
        n_trials=50  # Reduced for demo
    )
    
    if results:
        print(f"\nOptimization Results:")
        
        # Create ranking
        symbol_scores = []
        for symbol, result in results.items():
            if hasattr(result, 'best_score'):
                symbol_scores.append((symbol, result.best_score))
        
        # Sort by score (descending)
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nSymbol Ranking (by score):")
        for i, (symbol, score) in enumerate(symbol_scores, 1):
            print(f"  {i}. {symbol}: {score:.4f}")
        
        # Show best parameters for top performer
        if symbol_scores:
            best_symbol = symbol_scores[0][0]
            best_result = results[best_symbol]
            print(f"\nBest Parameters for {best_symbol}:")
            for param, value in best_result.best_params.items():
                print(f"  {param}: {value}")
    
    return results

def main():
    """"""
    
    print("Enhanced Parameter Tuning System ")
    print("="*80)
    
    try:
        run_basic_optimization()
        #run_walk_forward_validation()
        
        print("\n" + "="*80)
        print("All tasks completed successfully!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error running: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()