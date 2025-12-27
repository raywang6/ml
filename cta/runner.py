"""
Enhanced Run Script with Improved Parameter Tuning
Features:
1. Multi-objective optimization (performance + robustness)
2. Grid-based robustness testing
3. Walk-forward analysis
4. Efficient parameter space exploration
5. Early stopping for poor performers
"""

import sys
import pickle
import datetime as dt
import polars as pl
import numpy as np
from pathlib import Path

from strategy.signal_generator import SignalGenerator
from enhanced_parameters_tuning import (
    EnhancedStrategyTuner, efficient_parameter_search, 
    walk_forward_analysis, create_optimized_params_space
)
from parameters_tuning import load_symbol_data, single_train_val_split
from strategy.params_spaces import PARAMS_SPACES, EXIT_PARAMS

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_enhanced_single_optimization():
    """Enhanced single symbol optimization with robustness testing"""
    
    # Configuration
    symbol = "BTCUSDT"
    interval = "15m"
    train_start = dt.datetime(2023, 1, 1)
    train_end = dt.datetime(2024, 7, 1)
    strategy_name = "SuperTrend"
    
    logger.info(f"Running enhanced optimization for {symbol}")
    
    # Load and split data
    data = load_symbol_data(symbol, interval)
    train_data, val_data = single_train_val_split(data, train_start, train_end)
    
    # Create enhanced tuner
    tuner = EnhancedStrategyTuner(
        symbol=symbol,
        data_df=train_data,
        strategy_name=strategy_name,
        params_space=PARAMS_SPACES[strategy_name],
        exit_params=EXIT_PARAMS,
        target_metric="Sharpe_ratio"
    )
    
    # Run multi-objective optimization
    result = tuner.multi_objective_optimization(
        split_data=(train_data, val_data),
        n_trials=100,
        performance_weight=0.7,  # 70% performance, 30% robustness
        robustness_weight=0.3
    )
    
    # Display results
    logger.info("=== OPTIMIZATION RESULTS ===")
    logger.info(f"Best parameters: {result.best_params}")
    logger.info(f"Best combined score: {result.best_score:.4f}")
    
    if hasattr(result, 'robustness_result') and result.robustness_result:
        rob_result = result.robustness_result
        logger.info(f"Robustness score: {rob_result.robustness_score:.4f}")
        logger.info("Parameter sensitivity:")
        for param, sensitivity in rob_result.parameter_sensitivity.items():
            logger.info(f"  {param}: {sensitivity:.4f}")
        
        logger.info("Performance stability:")
        for metric, stats in rob_result.performance_stability.items():
            logger.info(f"  {metric}: mean={stats['mean']:.4f}, cv={stats['cv']:.4f}")
    
    return result

def run_walk_forward_analysis():
    """Run walk-forward analysis for more robust validation"""
    
    symbol = "BTCUSDT"
    interval = "15m"
    strategy_name = "SuperTrend"
    
    logger.info(f"Running walk-forward analysis for {symbol}")
    
    # Load data
    data = load_symbol_data(symbol, interval)
    
    # Filter to recent data for faster testing
    data = data.filter(pl.col("datetime") >= dt.datetime(2023, 1, 1))
    
    # Run walk-forward analysis
    wf_result = walk_forward_analysis(
        symbol=symbol,
        data=data,
        strategy_name=strategy_name,
        params_space=PARAMS_SPACES[strategy_name],
        exit_params=EXIT_PARAMS,
        window_months=6,  # 6-month training windows
        step_months=1,    # 1-month steps
        target_metric="Sharpe_ratio"
    )
    
    # Display results
    if wf_result:
        logger.info("=== WALK-FORWARD ANALYSIS RESULTS ===")
        logger.info(f"Number of periods: {wf_result['num_periods']}")
        logger.info(f"Average performance: {wf_result['avg_performance']:.4f} ± {wf_result['performance_std']:.4f}")
        logger.info(f"Average robustness: {wf_result['avg_robustness']:.4f}")
        
        logger.info("Parameter consistency:")
        for param, stats in wf_result['param_consistency'].items():
            logger.info(f"  {param}: mean={stats['mean']:.4f}, cv={stats['cv']:.4f}")
    
    return wf_result

def run_efficient_batch_optimization():
    """Run efficient batch optimization across multiple symbols"""
    
    # Configuration
    train_start = dt.datetime(2023, 1, 1)
    train_end = dt.datetime(2024, 7, 1)
    strategy_name = "SuperTrend"
    
    # Test with a smaller universe first
    test_universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
    
    logger.info(f"Running efficient batch optimization for {len(test_universe)} symbols")
    
    # Create optimized parameter space (reduced for efficiency)
    optimized_params = create_optimized_params_space(
        PARAMS_SPACES[strategy_name], 
        reduction_factor=0.6  # 60% of original range
    )
    
    logger.info("Optimized parameter space:")
    for param, config in optimized_params.items():
        if config['type'] in ['int', 'float']:
            logger.info(f"  {param}: [{config['low']}, {config['high']}] (step: {config.get('step', 'auto')})")
    
    # Run efficient search
    results = efficient_parameter_search(
        universe=test_universe,
        train_start=train_start,
        train_end=train_end,
        strategy_name=strategy_name,
        params_space=optimized_params,
        exit_params=EXIT_PARAMS,
        target_metric="Calmar_ratio",  # Focus on risk-adjusted returns
        use_walk_forward=False,  # Set to True for more robust but slower analysis
        n_trials=75  # Reduced for efficiency
    )
    
    # Analyze results
    if results:
        logger.info("=== BATCH OPTIMIZATION RESULTS ===")
        
        # Summary statistics
        scores = []
        robustness_scores = []
        
        for symbol, result in results.items():
            if hasattr(result, 'best_score'):
                scores.append(result.best_score)
                if hasattr(result, 'robustness_result') and result.robustness_result:
                    robustness_scores.append(result.robustness_result.robustness_score)
        
        if scores:
            logger.info(f"Performance scores: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
            logger.info(f"Best performer: {max(results.keys(), key=lambda x: getattr(results[x], 'best_score', -999))}")
        
        if robustness_scores:
            logger.info(f"Robustness scores: mean={np.mean(robustness_scores):.4f}, std={np.std(robustness_scores):.4f}")
        
        # Save results
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/tmp/enhanced_optimization_results_{timestamp}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'metadata': {
                    'strategy': strategy_name,
                    'train_start': train_start,
                    'train_end': train_end,
                    'universe': test_universe,
                    'params_space': optimized_params,
                    'timestamp': dt.datetime.now()
                }
            }, f)
        
        logger.info(f"Results saved to: {output_file}")
    
    return results

def compare_robustness_methods():
    """Compare grid-based vs noise-based robustness testing"""
    
    symbol = "BTCUSDT"
    interval = "15m"
    strategy_name = "SuperTrend"
    
    # Load data
    data = load_symbol_data(symbol, interval)
    train_data, val_data = single_train_val_split(
        data, 
        dt.datetime(2023, 1, 1), 
        dt.datetime(2024, 7, 1)
    )
    
    # Get some baseline parameters (you can use previously optimized ones)
    baseline_params = {
        'window_size': 149,
        'ayami_multi': 1.0,
        'sideway_filter_lookback': 200,
        'extrem_filter': 4.0,
        'entry_threshold': 0.15
    }
    
    # Create tuner
    tuner = EnhancedStrategyTuner(
        symbol=symbol,
        data_df=val_data,
        strategy_name=strategy_name,
        params_space=PARAMS_SPACES[strategy_name],
        exit_params=EXIT_PARAMS,
        target_metric="Sharpe_ratio"
    )
    
    # Test grid-based robustness
    logger.info("Testing grid-based robustness...")
    grid_robustness = tuner.evaluate_robustness(baseline_params, val_data)
    
    if grid_robustness:
        logger.info("=== GRID-BASED ROBUSTNESS RESULTS ===")
        logger.info(f"Robustness score: {grid_robustness.robustness_score:.4f}")
        logger.info("Parameter sensitivity:")
        for param, sensitivity in grid_robustness.parameter_sensitivity.items():
            logger.info(f"  {param}: {sensitivity:.4f}")
        
        # Show performance stability
        logger.info("Performance stability (Sharpe ratio):")
        if 'Sharpe_ratio' in grid_robustness.performance_stability:
            stats = grid_robustness.performance_stability['Sharpe_ratio']
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Std: {stats['std']:.4f}")
            logger.info(f"  CV: {stats['cv']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return grid_robustness

def main():
    """Main execution function"""
    
    print("Enhanced Parameter Tuning System")
    print("=" * 50)
    
    # Choose which analysis to run
    analysis_type = input("""
Choose analysis type:
1. Enhanced single optimization
2. Walk-forward analysis  
3. Efficient batch optimization
4. Compare robustness methods
5. All analyses

Enter choice (1-5): """).strip()
    
    if analysis_type == "1":
        run_enhanced_single_optimization()
    elif analysis_type == "2":
        run_walk_forward_analysis()
    elif analysis_type == "3":
        run_efficient_batch_optimization()
    elif analysis_type == "4":
        compare_robustness_methods()
    elif analysis_type == "5":
        logger.info("Running all analyses...")
        run_enhanced_single_optimization()
        run_walk_forward_analysis()
        run_efficient_batch_optimization()
        compare_robustness_methods()
    else:
        logger.info("Running default: enhanced single optimization")
        run_enhanced_single_optimization()

if __name__ == "__main__":
    main()

# Jupyter notebook cells for interactive use
def notebook_example():
    """Example usage in Jupyter notebook"""
    
    # %% Enhanced Single Symbol Optimization
    result = run_enhanced_single_optimization()
    
    # %% Walk-Forward Analysis
    wf_result = run_walk_forward_analysis()
    
    # %% Batch Optimization
    batch_results = run_efficient_batch_optimization()
    
    # %% Robustness Comparison
    robustness_result = compare_robustness_methods()
    
    # %% Custom Analysis
    # You can also create custom analysis by directly using the classes:
    
    symbol = "BTCUSDT"
    data = load_symbol_data(symbol)
    train_data, val_data = single_train_val_split(
        data, 
        dt.datetime(2023, 1, 1), 
        dt.datetime(2024, 7, 1)
    )
    
    tuner = EnhancedStrategyTuner(
        symbol=symbol,
        data_df=train_data,
        strategy_name="SuperTrend",
        params_space=PARAMS_SPACES["SuperTrend"],
        exit_params=EXIT_PARAMS,
        target_metric="Calmar_ratio"
    )
    
    # Custom multi-objective optimization
    custom_result = tuner.multi_objective_optimization(
        split_data=(train_data, val_data),
        n_trials=150,
        performance_weight=0.8,  # Emphasize performance more
        robustness_weight=0.2
    )
    
    return custom_result