"""
Configuration file for enhanced parameter tuning
Centralized settings for optimization parameters and robustness testing
"""

import datetime as dt
from typing import Dict, List

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

class OptimizationConfig:
    """Configuration for parameter optimization"""
    
    # Default optimization settings
    DEFAULT_N_TRIALS = 10
    DEFAULT_PERFORMANCE_WEIGHT = 0.7
    DEFAULT_ROBUSTNESS_WEIGHT = 0.3
    DEFAULT_TARGET_METRIC = "Sharpe_ratio"
    
    # Early stopping settings
    MIN_TRIALS_FOR_ROBUSTNESS = 20
    POOR_PERFORMANCE_THRESHOLD = -10.0
    
    # Pruning settings
    PRUNER_STARTUP_TRIALS = 10
    PRUNER_WARMUP_STEPS = 5
    
    # Parallel processing
    MAX_WORKERS = 8
    TIMEOUT_PER_SYMBOL = 1200  # 20 minutes
    
    # Data requirements
    MIN_BARS_REQUIRED = 1000
    MIN_TEST_BARS = 100

# =============================================================================
# ROBUSTNESS TESTING CONFIGURATION  
# =============================================================================

class RobustnessConfig:
    """Configuration for robustness testing"""
    
    # Grid-based robustness testing
    DEFAULT_GRID_SIZE = 0.1  # 10% parameter variation
    MAX_GRID_COMBINATIONS = 27  # 3^3 for 3 parameters with 3 values each
    
    # Robustness evaluation settings
    ROBUSTNESS_SAMPLE_SIZE = 1000  # Use subset of data for speed
    KEY_METRICS_FOR_ROBUSTNESS = ["Sharpe_ratio", "Calmar_ratio", "total_return"]
    
    # Parameter sensitivity thresholds
    HIGH_SENSITIVITY_THRESHOLD = 0.7
    MEDIUM_SENSITIVITY_THRESHOLD = 0.4
    
    # Performance stability thresholds
    STABLE_CV_THRESHOLD = 0.2  # Coefficient of variation < 20% is considered stable
    UNSTABLE_CV_THRESHOLD = 0.5  # CV > 50% is considered unstable

# =============================================================================
# WALK-FORWARD ANALYSIS CONFIGURATION
# =============================================================================

class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    
    # Window settings
    DEFAULT_WINDOW_MONTHS = 6
    DEFAULT_STEP_MONTHS = 1
    MIN_TRAIN_MONTHS = 3
    MIN_TEST_DAYS = 30
    
    # Analysis settings
    MAX_PERIODS = 24  # Limit to 2 years of monthly steps
    TRIALS_PER_WINDOW = 50  # Reduced trials per window for efficiency

# =============================================================================
# STRATEGY-SPECIFIC CONFIGURATIONS
# =============================================================================

STRATEGY_CONFIGS = {
    "SuperTrend": {
        "optimization": {
            "n_trials": 150,
            "performance_weight": 0.75,
            "robustness_weight": 0.25,
            "target_metric": "Calmar_ratio"
        },
        "robustness": {
            "grid_size": 0.08,  # 8% variation for SuperTrend
            "key_params": ["window_size", "entry_threshold"],  # Most important params
            "stability_threshold": 0.15
        },
        "walk_forward": {
            "window_months": 8,  # Longer window for trend-following
            "step_months": 2
        }
    },
    
    "PSYMOMLongShort": {
        "optimization": {
            "n_trials": 100,
            "performance_weight": 0.8,
            "robustness_weight": 0.2,
            "target_metric": "Sharpe_ratio"
        },
        "robustness": {
            "grid_size": 0.12,  # 12% variation for momentum strategy
            "key_params": ["window_size", "ama_period"],
            "stability_threshold": 0.2
        },
        "walk_forward": {
            "window_months": 6,
            "step_months": 1
        }
    }
}

# =============================================================================
# DATA AND UNIVERSE CONFIGURATION
# =============================================================================

class DataConfig:
    """Configuration for data handling"""
    
    # Data paths
    ROOT_DATA_PATH = "/home/whq/data/crypto/bar1m/futures/"
    RESULTS_PATH = "/home/whq/projects/files/result/params_search/"
    
    # Default time periods
    DEFAULT_TRAIN_START = dt.datetime(2020, 1, 1)
    DEFAULT_TRAIN_END = dt.datetime(2024, 7, 1)
    DEFAULT_INTERVAL = "15m"
    
    # Universe definitions
    MAJOR_COINS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
        "XRPUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT"
    ]
    
    EXTENDED_UNIVERSE = ['SUIUSDT', 'AAVEUSDT', 'ADAUSDT', 'FETUSDT', 'WIFUSDT', 'ARUSDT', 'AXSUSDT', 'DOGEUSDT', 'PYTHUSDT', 'ARBUSDT', 'GALAUSDT', 'BNBUSDT', 'LINKUSDT', 'UNIUSDT', 
            '1000BONKUSDT', '1000FLOKIUSDT', '1000SHIBUSDT', 'ETCUSDT', 'ETHUSDT', 'BTCUSDT', 'PENDLEUSDT', 'SOLUSDT', 'TIAUSDT', 'TONUSDT', 'WLDUSDT', '1000PEPEUSDT', 'FILUSDT', 
            'ORDIUSDT', 'SEIUSDT', 'DYDXUSDT', 'POLUSDT', 'MKRUSDT', 'TAOUSDT', 'DOTUSDT', 'OPUSDT', 'RUNEUSDT', 'LTCUSDT', 'BCHUSDT', 'APEUSDT', 'ATOMUSDT', 'SANDUSDT', 'NEARUSDT', 
            'APTUSDT', 'AVAXUSDT', 'JTOUSDT', 'JUPUSDT', 'STXUSDT', 'XRPUSDT', 'LDOUSDT', 'TRXUSDT', 'CRVUSDT', 'INJUSDT', 'RENDERUSDT', 'ENAUSDT']
    
    # Test universe for development
    TEST_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

class PerformanceThresholds:
    """Thresholds for evaluating strategy performance"""
    
    # Minimum acceptable performance
    MIN_SHARPE_RATIO = 0.5
    MIN_CALMAR_RATIO = 0.3
    MIN_HIT_RATE = 0.45
    MAX_DRAWDOWN = 0.3  # 30%
    
    # Good performance benchmarks
    GOOD_SHARPE_RATIO = 1.0
    GOOD_CALMAR_RATIO = 1.0
    GOOD_HIT_RATE = 0.55
    GOOD_DRAWDOWN = 0.15  # 15%
    
    # Excellent performance benchmarks
    EXCELLENT_SHARPE_RATIO = 1.5
    EXCELLENT_CALMAR_RATIO = 2.0
    EXCELLENT_HIT_RATE = 0.6
    EXCELLENT_DRAWDOWN = 0.1  # 10%

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_strategy_config(strategy_name: str, config_type: str = "optimization") -> Dict:
    """Get configuration for a specific strategy"""
    if strategy_name in STRATEGY_CONFIGS:
        return STRATEGY_CONFIGS[strategy_name].get(config_type, {})
    else:
        # Return default configuration
        if config_type == "optimization":
            return {
                "n_trials": OptimizationConfig.DEFAULT_N_TRIALS,
                "performance_weight": OptimizationConfig.DEFAULT_PERFORMANCE_WEIGHT,
                "robustness_weight": OptimizationConfig.DEFAULT_ROBUSTNESS_WEIGHT,
                "target_metric": OptimizationConfig.DEFAULT_TARGET_METRIC
            }
        elif config_type == "robustness":
            return {
                "grid_size": RobustnessConfig.DEFAULT_GRID_SIZE,
                "key_params": [],
                "stability_threshold": RobustnessConfig.STABLE_CV_THRESHOLD
            }
        elif config_type == "walk_forward":
            return {
                "window_months": WalkForwardConfig.DEFAULT_WINDOW_MONTHS,
                "step_months": WalkForwardConfig.DEFAULT_STEP_MONTHS
            }
    return {}

def get_universe(universe_type: str = "test") -> List[str]:
    """Get trading universe based on type"""
    if universe_type == "major":
        return DataConfig.MAJOR_COINS
    elif universe_type == "extended":
        return DataConfig.EXTENDED_UNIVERSE
    elif universe_type == "test":
        return DataConfig.TEST_UNIVERSE
    else:
        return DataConfig.TEST_UNIVERSE

def evaluate_performance_level(metrics: Dict[str, float]) -> str:
    """Evaluate overall performance level based on metrics"""
    
    sharpe = metrics.get("Sharpe_ratio", 0)
    calmar = metrics.get("Calmar_ratio", 0)
    hit_rate = metrics.get("hit_rate", 0)
    max_dd = metrics.get("max_drawdown", 1)
    
    # Count excellent metrics
    excellent_count = 0
    if sharpe >= PerformanceThresholds.EXCELLENT_SHARPE_RATIO:
        excellent_count += 1
    if calmar >= PerformanceThresholds.EXCELLENT_CALMAR_RATIO:
        excellent_count += 1
    if hit_rate >= PerformanceThresholds.EXCELLENT_HIT_RATE:
        excellent_count += 1
    if max_dd <= PerformanceThresholds.EXCELLENT_DRAWDOWN:
        excellent_count += 1
    
    # Count good metrics
    good_count = 0
    if sharpe >= PerformanceThresholds.GOOD_SHARPE_RATIO:
        good_count += 1
    if calmar >= PerformanceThresholds.GOOD_CALMAR_RATIO:
        good_count += 1
    if hit_rate >= PerformanceThresholds.GOOD_HIT_RATE:
        good_count += 1
    if max_dd <= PerformanceThresholds.GOOD_DRAWDOWN:
        good_count += 1
    
    # Count minimum acceptable metrics
    min_count = 0
    if sharpe >= PerformanceThresholds.MIN_SHARPE_RATIO:
        min_count += 1
    if calmar >= PerformanceThresholds.MIN_CALMAR_RATIO:
        min_count += 1
    if hit_rate >= PerformanceThresholds.MIN_HIT_RATE:
        min_count += 1
    if max_dd <= PerformanceThresholds.MAX_DRAWDOWN:
        min_count += 1
    
    # Determine overall level
    if excellent_count >= 3:
        return "Excellent"
    elif good_count >= 3:
        return "Good"
    elif min_count >= 3:
        return "Acceptable"
    else:
        return "Poor"

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example of how to use configurations
    
    strategy = "SuperTrend"
    
    # Get strategy-specific configuration
    opt_config = get_strategy_config(strategy, "optimization")
    rob_config = get_strategy_config(strategy, "robustness")
    wf_config = get_strategy_config(strategy, "walk_forward")
    
    print(f"Configuration for {strategy}:")
    print(f"Optimization: {opt_config}")
    print(f"Robustness: {rob_config}")
    print(f"Walk-forward: {wf_config}")
    
    # Get universe
    universe = get_universe("major")
    print(f"Major coins universe: {universe}")
    
    # Evaluate performance example
    example_metrics = {
        "Sharpe_ratio": 1.2,
        "Calmar_ratio": 1.5,
        "hit_rate": 0.58,
        "max_drawdown": 0.12
    }
    
    performance_level = evaluate_performance_level(example_metrics)
    print(f"Performance level: {performance_level}")