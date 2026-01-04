"""
Walk-Forward Test Runner

This script runs the walk-forward testing framework for ML Neutral Strategy.
Modify the parameters below to customize your test.

Usage:
    conda activate ml  # Use 'ml' environment (has PyTorch + LightGBM working)
    python run_walkforward.py
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from walk_forward import WalkForwardTest

# Configuration
CONFIG = {

    # Walk-forward parameters
    'start_date': '2023-06-01',    # First training cutoff (YYYY-MM-DD)
    'end_date': '2025-10-20',      # Last training cutoff (YYYY-MM-DD)
    'train_day': 1,               # Day of month for cutoffs

    # Model parameters
    'horizon': 24,                  # Prediction horizon (in data periods)
    'target_col': 'target24',        # Target column name
    'target_func': 'ret',

    # Training parameters
    'max_hp_evals': 1,            # HP optimization iterations per period
    'min_train_months': 6,         # Minimum months of data to start training

    'delta_month': 3,
    # Output
    'output_dir': 'walkforward_results_weight_early50_24',

    # Features (None = auto-detect from data)
    'features': ['SR_dist_16w_mean', 'SR_dist_1w_mean', 'SR_dist_2w_mean', 'SR_dist_4w_mean', 'SR_dist_52w_mean', 'SR_dist_8w_mean', 'after_bottom_price_diff_1h_diff', 'after_bottom_price_diff_1h_mean', 'after_bottom_price_diff_1h_std', 'after_top_price_diff_1h_diff', 'after_top_price_diff_1h_mean', 'after_top_price_diff_1h_std', 'before_bottom_price_diff_1h_diff', 'before_bottom_price_diff_1h_mean', 'before_bottom_price_diff_1h_std', 'before_top_price_diff_1h_diff', 'before_top_price_diff_1h_mean', 'before_top_price_diff_1h_std', 'big_bottom_ret_inflow_ratio_120h', 'big_bottom_ret_inflow_ratio_120h_right', 'big_bottom_ret_inflow_ratio_24h', 'big_bottom_ret_inflow_ratio_24h_right', 'big_bottom_ret_inflow_ratio_360h', 'big_bottom_ret_inflow_ratio_360h_right', 'big_bottom_ret_inflow_ratio_720h', 'big_bottom_ret_inflow_ratio_720h_right', 'big_bottom_ret_inflow_ratio_72h', 'big_bottom_ret_inflow_ratio_72h_right', 'big_buy_ratio_std_120h', 'big_buy_ratio_std_120h_right', 'big_buy_ratio_std_24h', 'big_buy_ratio_std_24h_right', 'big_buy_ratio_std_360h', 'big_buy_ratio_std_360h_right', 'big_buy_ratio_std_720h', 'big_buy_ratio_std_720h_right', 'big_buy_ratio_std_72h', 'big_buy_ratio_std_72h_right', 'big_buysell_std_diff_120h', 'big_buysell_std_diff_120h_right', 'big_buysell_std_diff_24h', 'big_buysell_std_diff_24h_right', 'big_buysell_std_diff_360h', 'big_buysell_std_diff_360h_right', 'big_buysell_std_diff_720h', 'big_buysell_std_diff_720h_right', 'big_buysell_std_diff_72h', 'big_buysell_std_diff_72h_right', 'big_inflow_120h', 'big_inflow_120h_right', 'big_inflow_24h', 'big_inflow_24h_right', 'big_inflow_360h', 'big_inflow_360h_right', 'big_inflow_720h', 'big_inflow_720h_right', 'big_inflow_72h', 'big_inflow_72h_right', 'big_inflow_ratio_120h', 'big_inflow_ratio_120h_right', 'big_inflow_ratio_24h', 'big_inflow_ratio_24h_right', 'big_inflow_ratio_360h', 'big_inflow_ratio_360h_right', 'big_inflow_ratio_720h', 'big_inflow_ratio_720h_right', 'big_inflow_ratio_72h', 'big_inflow_ratio_72h_right', 'big_inflow_self_corr_120h', 'big_inflow_self_corr_120h_right', 'big_inflow_self_corr_24h', 'big_inflow_self_corr_24h_right', 'big_inflow_self_corr_360h', 'big_inflow_self_corr_360h_right', 'big_inflow_self_corr_720h', 'big_inflow_self_corr_720h_right', 'big_inflow_self_corr_72h', 'big_inflow_self_corr_72h_right', 'big_inflow_std_120h', 'big_inflow_std_120h_right', 'big_inflow_std_24h', 'big_inflow_std_24h_right', 'big_inflow_std_360h', 'big_inflow_std_360h_right', 'big_inflow_std_720h', 'big_inflow_std_720h_right', 'big_inflow_std_72h', 'big_inflow_std_72h_right', 'big_ret_inflow_corr_120h', 'big_ret_inflow_corr_120h_right', 'big_ret_inflow_corr_24h', 'big_ret_inflow_corr_24h_right', 'big_ret_inflow_corr_360h', 'big_ret_inflow_corr_360h_right', 'big_ret_inflow_corr_720h', 'big_ret_inflow_corr_720h_right', 'big_ret_inflow_corr_72h', 'big_ret_inflow_corr_72h_right', 'big_ret_inflow_leadlag_corr_diff_120h', 'big_ret_inflow_leadlag_corr_diff_120h_right', 'big_ret_inflow_leadlag_corr_diff_24h', 'big_ret_inflow_leadlag_corr_diff_24h_right', 'big_ret_inflow_leadlag_corr_diff_360h', 'big_ret_inflow_leadlag_corr_diff_360h_right', 'big_ret_inflow_leadlag_corr_diff_720h', 'big_ret_inflow_leadlag_corr_diff_720h_right', 'big_ret_inflow_leadlag_corr_diff_72h', 'big_ret_inflow_leadlag_corr_diff_72h_right', 'big_ret_inflow_residual_120h', 'big_ret_inflow_residual_120h_right', 'big_ret_inflow_residual_24h', 'big_ret_inflow_residual_24h_right', 'big_ret_inflow_residual_360h', 'big_ret_inflow_residual_360h_right', 'big_ret_inflow_residual_720h', 'big_ret_inflow_residual_720h_right', 'big_ret_inflow_residual_72h', 'big_ret_inflow_residual_72h_right', 'big_top_ret_inflow_ratio_120h', 'big_top_ret_inflow_ratio_120h_right', 'big_top_ret_inflow_ratio_24h', 'big_top_ret_inflow_ratio_24h_right', 'big_top_ret_inflow_ratio_360h', 'big_top_ret_inflow_ratio_360h_right', 'big_top_ret_inflow_ratio_720h', 'big_top_ret_inflow_ratio_720h_right', 'big_top_ret_inflow_ratio_72h', 'big_top_ret_inflow_ratio_72h_right', 'bin_mean_px_range_1h_diff', 'bin_mean_px_range_1h_mean', 'bin_mean_px_range_1h_std', 'bin_mean_px_std_1h_diff', 'bin_mean_px_std_1h_mean', 'bin_mean_px_std_1h_std', 'bin_ret_std_std_1h_diff', 'bin_ret_std_std_1h_mean', 'bin_ret_std_std_1h_std', 'dec_vol_px_range_1h_diff', 'dec_vol_px_range_1h_mean', 'dec_vol_px_range_1h_std', 'dec_vol_ret_std_1h_diff', 'dec_vol_ret_std_1h_mean', 'dec_vol_ret_std_1h_std', 'dec_vol_ret_sum_1h_diff', 'dec_vol_ret_sum_1h_mean', 'dec_vol_ret_sum_1h_std', 'high_px_abnormal_cnt_1h_diff', 'high_px_abnormal_cnt_1h_mean', 'high_px_abnormal_cnt_1h_std', 'high_px_volume_1h_diff', 'high_px_volume_1h_mean', 'high_px_volume_1h_std', 'high_px_volume_diff_1h_diff', 'high_px_volume_diff_1h_mean', 'high_px_volume_diff_1h_std', 'high_vol_px_range_1h_diff', 'high_vol_px_range_1h_mean', 'high_vol_px_range_1h_std', 'high_vol_ret_std_1h_diff', 'high_vol_ret_std_1h_mean', 'high_vol_ret_std_1h_std', 'high_vol_ret_sum_1h_diff', 'high_vol_ret_sum_1h_mean', 'high_vol_ret_sum_1h_std', 'imb_diff_vol_px_range_1h_diff', 'imb_diff_vol_px_range_1h_mean', 'imb_diff_vol_px_range_1h_std', 'imb_diff_vol_ret_std_1h_diff', 'imb_diff_vol_ret_std_1h_mean', 'imb_diff_vol_ret_std_1h_std', 'imb_diff_vol_ret_sum_1h_diff', 'imb_diff_vol_ret_sum_1h_mean', 'imb_diff_vol_ret_sum_1h_std', 'imb_vol_px_range_1h_diff', 'imb_vol_px_range_1h_mean', 'imb_vol_px_range_1h_std', 'imb_vol_ret_std_1h_diff', 'imb_vol_ret_std_1h_mean', 'imb_vol_ret_std_1h_std', 'imb_vol_ret_sum_1h_diff', 'imb_vol_ret_sum_1h_mean', 'imb_vol_ret_sum_1h_std', 'inc_vol_px_range_1h_diff', 'inc_vol_px_range_1h_mean', 'inc_vol_px_range_1h_std', 'inc_vol_ret_std_1h_diff', 'inc_vol_ret_std_1h_mean', 'inc_vol_ret_std_1h_std', 'inc_vol_ret_sum_1h_diff', 'inc_vol_ret_sum_1h_mean', 'inc_vol_ret_sum_1h_std', 'long_liquidation_dvol_1d', 'long_liquidation_dvol_7d', 'low_px_abnormal_cnt_1h_diff', 'low_px_abnormal_cnt_1h_mean', 'low_px_abnormal_cnt_1h_std', 'low_px_volume_1h_diff', 'low_px_volume_1h_mean', 'low_px_volume_1h_std', 'low_px_volume_diff_1h_diff', 'low_px_volume_diff_1h_mean', 'low_px_volume_diff_1h_std', 'low_vol_px_range_1h_diff', 'low_vol_px_range_1h_mean', 'low_vol_px_range_1h_std', 'low_vol_ret_std_1h_diff', 'low_vol_ret_std_1h_mean', 'low_vol_ret_std_1h_std', 'low_vol_ret_sum_1h_diff', 'low_vol_ret_sum_1h_mean', 'low_vol_ret_sum_1h_std', 'ov_1hcorr_1dmean', 'ov_1hcorr_1dstd', 'ov_1hcorr_1dtrend', 'pv_1hcorr_1dmean', 'pv_1hcorr_1dstd', 'pv_1hcorr_1dtrend', 'realized_volatility_1d', 'realized_volatility_3d', 'realized_volatility_7d', 'shock_volatility_1d', 'shock_volatility_3d', 'shock_volatility_7d', 'short_liquidation_dvol_1d', 'short_liquidation_dvol_7d', 'small_big_inflow_corr_120h', 'small_big_inflow_corr_120h_right', 'small_big_inflow_corr_24h', 'small_big_inflow_corr_24h_right', 'small_big_inflow_corr_360h', 'small_big_inflow_corr_360h_right', 'small_big_inflow_corr_720h', 'small_big_inflow_corr_720h_right', 'small_big_inflow_corr_72h', 'small_big_inflow_corr_72h_right', 'small_big_inflow_leadlag_corr_diff_120h', 'small_big_inflow_leadlag_corr_diff_120h_right', 'small_big_inflow_leadlag_corr_diff_24h', 'small_big_inflow_leadlag_corr_diff_24h_right', 'small_big_inflow_leadlag_corr_diff_360h', 'small_big_inflow_leadlag_corr_diff_360h_right', 'small_big_inflow_leadlag_corr_diff_720h', 'small_big_inflow_leadlag_corr_diff_720h_right', 'small_big_inflow_leadlag_corr_diff_72h', 'small_big_inflow_leadlag_corr_diff_72h_right', 'small_bottom_ret_inflow_ratio_120h', 'small_bottom_ret_inflow_ratio_120h_right', 'small_bottom_ret_inflow_ratio_24h', 'small_bottom_ret_inflow_ratio_24h_right', 'small_bottom_ret_inflow_ratio_360h', 'small_bottom_ret_inflow_ratio_360h_right', 'small_bottom_ret_inflow_ratio_720h', 'small_bottom_ret_inflow_ratio_720h_right', 'small_bottom_ret_inflow_ratio_72h', 'small_bottom_ret_inflow_ratio_72h_right', 'small_buy_ratio_std_120h', 'small_buy_ratio_std_120h_right', 'small_buy_ratio_std_24h', 'small_buy_ratio_std_24h_right', 'small_buy_ratio_std_360h', 'small_buy_ratio_std_360h_right', 'small_buy_ratio_std_720h', 'small_buy_ratio_std_720h_right', 'small_buy_ratio_std_72h', 'small_buy_ratio_std_72h_right', 'small_buysell_std_diff_120h', 'small_buysell_std_diff_120h_right', 'small_buysell_std_diff_24h', 'small_buysell_std_diff_24h_right', 'small_buysell_std_diff_360h', 'small_buysell_std_diff_360h_right', 'small_buysell_std_diff_720h', 'small_buysell_std_diff_720h_right', 'small_buysell_std_diff_72h', 'small_buysell_std_diff_72h_right', 'small_inflow_120h', 'small_inflow_120h_right', 'small_inflow_24h', 'small_inflow_24h_right', 'small_inflow_360h', 'small_inflow_360h_right', 'small_inflow_720h', 'small_inflow_720h_right', 'small_inflow_72h', 'small_inflow_72h_right', 'small_inflow_ratio_120h', 'small_inflow_ratio_120h_right', 'small_inflow_ratio_24h', 'small_inflow_ratio_24h_right', 'small_inflow_ratio_360h', 'small_inflow_ratio_360h_right', 'small_inflow_ratio_720h', 'small_inflow_ratio_720h_right', 'small_inflow_ratio_72h', 'small_inflow_ratio_72h_right', 'small_inflow_self_corr_120h', 'small_inflow_self_corr_120h_right', 'small_inflow_self_corr_24h', 'small_inflow_self_corr_24h_right', 'small_inflow_self_corr_360h', 'small_inflow_self_corr_360h_right', 'small_inflow_self_corr_720h', 'small_inflow_self_corr_720h_right', 'small_inflow_self_corr_72h', 'small_inflow_self_corr_72h_right', 'small_inflow_std_120h', 'small_inflow_std_120h_right', 'small_inflow_std_24h', 'small_inflow_std_24h_right', 'small_inflow_std_360h', 'small_inflow_std_360h_right', 'small_inflow_std_720h', 'small_inflow_std_720h_right', 'small_inflow_std_72h', 'small_inflow_std_72h_right', 'small_ret_inflow_corr_120h', 'small_ret_inflow_corr_120h_right', 'small_ret_inflow_corr_24h', 'small_ret_inflow_corr_24h_right', 'small_ret_inflow_corr_360h', 'small_ret_inflow_corr_360h_right', 'small_ret_inflow_corr_720h', 'small_ret_inflow_corr_720h_right', 'small_ret_inflow_corr_72h', 'small_ret_inflow_corr_72h_right', 'small_ret_inflow_leadlag_corr_diff_120h', 'small_ret_inflow_leadlag_corr_diff_120h_right', 'small_ret_inflow_leadlag_corr_diff_24h', 'small_ret_inflow_leadlag_corr_diff_24h_right', 'small_ret_inflow_leadlag_corr_diff_360h', 'small_ret_inflow_leadlag_corr_diff_360h_right', 'small_ret_inflow_leadlag_corr_diff_720h', 'small_ret_inflow_leadlag_corr_diff_720h_right', 'small_ret_inflow_leadlag_corr_diff_72h', 'small_ret_inflow_leadlag_corr_diff_72h_right', 'small_ret_inflow_residual_120h', 'small_ret_inflow_residual_120h_right', 'small_ret_inflow_residual_24h', 'small_ret_inflow_residual_24h_right', 'small_ret_inflow_residual_360h', 'small_ret_inflow_residual_360h_right', 'small_ret_inflow_residual_720h', 'small_ret_inflow_residual_720h_right', 'small_ret_inflow_residual_72h', 'small_ret_inflow_residual_72h_right', 'small_top_ret_inflow_ratio_120h', 'small_top_ret_inflow_ratio_120h_right', 'small_top_ret_inflow_ratio_24h', 'small_top_ret_inflow_ratio_24h_right', 'small_top_ret_inflow_ratio_360h', 'small_top_ret_inflow_ratio_360h_right', 'small_top_ret_inflow_ratio_720h', 'small_top_ret_inflow_ratio_720h_right', 'small_top_ret_inflow_ratio_72h', 'small_top_ret_inflow_ratio_72h_right', 'up_volatility_ratio_1d', 'up_volatility_ratio_3d', 'up_volatility_ratio_7d', 'vol_max_std_1h_diff', 'vol_max_std_1h_mean', 'vol_max_std_1h_std', 'weighted_skewness_mb_1h_diff', 'weighted_skewness_mb_1h_mean', 'weighted_skewness_mb_1h_std'],
    # Or specify explicitly:
    # 'features': ['get_upper_shadow', 'get_upper_shadow_diff1',
    #              'get_upper_shadow_diff2', 'get_upper_shadow_diff3'],
}

DATA_CONFIG = {
    'features':{
        'mb_factor': '/data/shared/data/mb_factor',
        'diff_scale_inflow': '/data/shared/data/perp_agg/diff_scale_inflow',
        'inflow_and_ret': '/data/shared/data/perp_agg/inflow_and_ret',
        'inflow_mismatch_corr': '/data/shared/data/perp_agg/inflow_mismatch_corr',
        'inflow_stats': '/data/shared/data/perp_agg/inflow_stats',
        'spot_diff_scale_inflow': '/data/shared/data/spot_agg/diff_scale_inflow',
        'spot_inflow_and_ret': '/data/shared/data/spot_agg/inflow_and_ret',
        'spot_inflow_mismatch_corr': '/data/shared/data/spot_agg/inflow_mismatch_corr',
        'spot_inflow_stats': '/data/shared/data/spot_agg/inflow_stats',
        'sr': '/data/shared/data/sr',
        'vnl/': '/data/shared/data/vnl',
    },
    'returns': '/home/ray/projects/data/sync_folder/factor/norm_1h',
    'universe': '/home/moneyking/projects/mlframework/mlneutral/data/filters.parquet',
}


def run():
    """Run the walk-forward test with configured parameters."""

    print("=" * 60)
    print("Walk-Forward Test Configuration")
    print("=" * 60)
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists(DATA_CONFIG['returns']):
        print(f"\nError: Data file not found: {DATA_CONFIG['returns']}")
        print("Please ensure your data file exists before running.")
        return None, None

    for hori in [12, 24, 48]:
        CONFIG['horizon'] = hori
        CONFIG['target_col'] = f'target{hori}'
        CONFIG['output_dir'] = f'walkforward_results_ret_prod_{hori}'
        # Initialize
        wf = WalkForwardTest(
            data_config = DATA_CONFIG,
            features=CONFIG['features'],
            horizon=CONFIG['horizon'],
            output_dir=CONFIG['output_dir'],
            target_col=CONFIG['target_col'],
            target_func=CONFIG['target_func'],
            delta_month=CONFIG['delta_month'],
            anti_leaking_days=max(1,int(CONFIG['horizon']/24))
        )
        # Run walk-forward test
        results = wf.run(
            start_date=CONFIG['start_date'],
            end_date=CONFIG['end_date'],
            train_day=CONFIG['train_day'],
            max_hp_evals=CONFIG['max_hp_evals'],
            min_train_months=CONFIG['min_train_months']
        )
    """
    # Run walk-forward test
    results = wf.run_test(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        train_day=CONFIG['train_day'],
        max_hp_evals=CONFIG['max_hp_evals'],
        min_train_months=CONFIG['min_train_months']
    )


    # Print final analysis
    print("\n" + "=" * 60)
    print("Final Analysis")
    print("=" * 60)
    analysis = wf.analyze_results(results)
    for k, v in analysis.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nResults saved to: {CONFIG['output_dir']}/")
    print(f"  - walk_forward_results.csv: Period-by-period metrics")
    print(f"  - all_predictions.parquet: Combined OOS predictions")
    print(f"  - predictions/: Individual period predictions")
    """
    return wf, results


if __name__ == '__main__':
    wf, results = run()
