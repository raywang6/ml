"""
Unified Performance Metrics Calculator
"""
import polars as pl
import numpy as np
from typing import Dict, Optional



class PerformanceCalculator:
    """Centralized performance metrics calculation"""

    @staticmethod
    def _calculate_annual_factor(interval: str) -> int:
        """
        Calculate annualization factor based on data interval
        
        Args:
            interval: Time interval string ('15m', '1h', '4h', '1d', etc.)
        
        Returns:
            Number of periods per year for this interval
        """
        interval_factors = {
            '1m': 365 * 24 * 60,      # 1 minute
            '5m': 365 * 24 * 12,      # 5 minutes  
            '15m': 365 * 24 * 4,      # 15 minutes (current default)
            '30m': 365 * 24 * 2,      # 30 minutes
            '1h': 365 * 24,           # 1 hour
            '2h': 365 * 12,           # 2 hours
            '4h': 365 * 6,            # 4 hours
            '8h': 365 * 3,            # 8 hours
            '12h': 365 * 2,           # 12 hours
            '1d': 365,                # 1 day
            '3d': 365 / 3,            # 3 days
            '1w': 52,                 # 1 week
            '1M': 12                  # 1 month
        }
    
        if interval not in interval_factors:
            raise ValueError(f"Unsupported interval: {interval}. Supported: {list(interval_factors.keys())}")
        
        return int(interval_factors[interval])
    
    @staticmethod
    def calculate_all_metrics(df: pl.DataFrame, 
                            position_col: str,
                            interval: str = '15m',
                            transaction_cost: float = 0.0002,
                            slippage: float = 0.0,
                            ) -> Dict[str, float]:
        """
        Calculate all performance metrics in one pass
        Note that we assume the following convention:
        if position_col = [ 0, 1, 1, 1, 0], it means the entry bar is at index 1 
        and the exit bar is at index 4.

        Entry bar: VWAP -> Close + transaction cost + slippage
        Exit bar: Open -> VWAP + transaction cost + slippage
        Args:
            - df: Input DataFrame with market data, position column from the signal and exit
            - position_col: Name of the column representing the position (e.g., long/short)
            - interval: Time interval for the data (default is '15m')
            - transaction_cost: Transaction cost per trade (default is 0.0002)
            - slippage: Slippage cost per trade (default is 0.0)
        """

        # Check required columns
        required_cols = [position_col, "close_price", "open_price", "vwap"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Detect entry and exit bar
        df = df.with_columns([  
            ((pl.col(position_col).shift(1) == 0) & (pl.col(position_col) != 0)).alias("is_entry"),
            ((pl.col(position_col).shift(1) != 0) & (pl.col(position_col) == 0)).alias("is_exit")
        ])

        df = df.with_columns(
            ((pl.col(position_col) != 0) & (~pl.col("is_entry"))).alias("is_holding"),
            pl.col("close_price").pct_change().fill_null(0.0).alias("close_to_close_ret")
        )

        # Compute returns for each bar type with proper execution costs
        df = df.with_columns(
            pl.when(pl.col("is_entry"))
            # Entry bar returns: VWAP to Close - transaction cost - slippage
            .then(
                (pl.col("close_price") / pl.col("vwap") - 1 ) 
                * pl.col(position_col) - transaction_cost - slippage
            )
            # Exit bar returns: Open to VWAP - transaction cost - slippage
            .when(pl.col("is_exit"))
            .then(
                (pl.col("vwap") / pl.col("open_price") - 1 ) 
                * pl.col(position_col).shift(1) - transaction_cost - slippage  # Use previous position for exit
            )
            # Normal holding bar
            .when(pl.col("is_holding"))
            .then(pl.col("close_to_close_ret") * pl.col(position_col))
            .otherwise(0)
            .alias("cur_bar_ret")
        )
                
        # Filter to active periods
        active_returns = df.filter(pl.col("cur_bar_ret") != 0)
        if active_returns.height == 0:
            return PerformanceCalculator._empty_metrics()

        cum_ret = (1 + pl.col("cur_bar_ret")).cum_prod().alias("cum_ret")
        df = df.with_columns(cum_ret)

        # 1. Total Return 
        total_return = df["cum_ret"][-1] - 1

        # 2. Sharpe Ratio
        avg_ret = df["cur_bar_ret"].mean()
        std_dev = df["cur_bar_ret"].std()
        annual_factor = PerformanceCalculator._calculate_annual_factor(interval)
        sharpe_ratio = (avg_ret / std_dev) * np.sqrt(annual_factor) if std_dev != 0 else 0.0

        # 3. Max Drawdown
        mdd = PerformanceCalculator._compute_max_drawdown(df, "cur_bar_ret")

        # 4. Calmar Ratio
        calmar_ratio = total_return / mdd if mdd > 0 else -float('inf')

        # 5. Hit Rate
        hit_rate = (df["cur_bar_ret"] > 0).sum() / active_returns.height

        # 6. IC
        IC = df.select(pl.corr( "cur_bar_ret", position_col, method = "spearman" )).item()

        # 7. Profit Factor
        gross_profit = df.filter(pl.col("cur_bar_ret") > 0).select(pl.col("cur_bar_ret").sum()).item()
        gross_loss   = df.filter(pl.col("cur_bar_ret") < 0).select(pl.col("cur_bar_ret").sum().abs()).item()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else -float('inf')

        # 8. Direction-specific
        long_positions = df.filter(pl.col(position_col) > 0)
        short_positions = df.filter(pl.col(position_col) < 0)
        
        avg_long_return = long_positions.select(pl.col("cur_bar_ret").mean()).item() if long_positions.height > 0 else 0.0
        avg_short_return = short_positions.select(pl.col("cur_bar_ret").mean()).item() if short_positions.height > 0 else 0.0

        # 9. Market Participation
        pct_in_market = active_returns.height / df.height

        return {
                "total_return": total_return,
                "Sharpe_ratio": sharpe_ratio,
                "max_drawdown": mdd,
                "Calmar_ratio": calmar_ratio,
                "hit_rate": hit_rate,
                "IC": IC,
                "profit_factor": profit_factor,
                "avg_long_return": avg_long_return,
                "avg_short_return": avg_short_return,
                "pct_in_the_market": pct_in_market,
            }
    
    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "total_return": 0.0, "Sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "Calmar_ratio": 0.0, "hit_rate": 0.0,
            "IC": 0.0, "profit_factor": 0.0, "avg_long_return": 0.0,
            "avg_short_return": 0.0, "pct_in_the_market": 0.0
        }

    @staticmethod
    def _compute_max_drawdown(df: pl.DataFrame, cur_bar_ret_col: str) -> float:
        """
        Compute maximum drawdown for a strategy using cumulative returns.
        """
        # Get non-zero signal returns
        non_zero_returns = df.filter(pl.col(cur_bar_ret_col) != 0)[cur_bar_ret_col]

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
