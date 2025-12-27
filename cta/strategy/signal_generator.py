"""
Description: Rewriting Ray's strategies in polars
Author: Haoqing Wu
Date: 2025-08-03
TODO: We assume using 15min bars for now, for which we rely on the computation of Sharpe, etc.
But we need to make it flexible for other timer interval.

"""
from typing import Union, Optional

import pandas as pd
import numpy as np

import polars as pl
import datetime as dt
from .exit_methods import apply_exit_methods
from performance_metrics import PerformanceCalculator

#########################
## Decorator Functions ##
def necessary_columns( *columns ):
    """ Decorator to check if the necessary columns are present in the DataFrame. """
    def decorator( func ):
        def wrapper( self, *args, **kwargs ):
            for col in columns:
                if col not in self.columns:
                    raise ValueError( f"DataFrame must contain a '{col}' column for {func.__name__} calculation" )
            return func( self, *args, **kwargs )
        return wrapper
    return decorator
#########################

##################### 
## Helper Function ##
def exp_weights( window: int, half_life: int ) -> np.ndarray:
    """
    Generate exponentially decaying weights over `window` trailing values, decaying by half each `half_life` index.

    Args:
        window: integer number of points in the trailing lookback period
        half_life: integer decay rate

    Returns:
        np.ndarray: array of weights
    """
    try:
        assert isinstance( window, int )
        if not window > 0:
            raise ValueError( "`window` must be a strictly positive integer" )
    except ( AttributeError, AssertionError ) as e:
        raise TypeError("`window` must be an integer type") from e
    try:
        assert isinstance( half_life, int )
        if not half_life > 0:
            raise ValueError("`half_life` must be a strictly positive integer")
    except ( AttributeError, AssertionError ) as e:
        raise TypeError("`half_life` must be an integer type") from e
    
    decay = np.log(2) / half_life

    return np.exp( -decay * np.arange( window ) )[ ::-1 ]  # reverse the order

#####################

class SignalGenerator:
    """ SignalGenerator class for generating trading signals; This behaves like a normal polars dataframe"""
    
    def __init__(self, 
                 data: Union[ pl.DataFrame, pl.LazyFrame, pd.DataFrame ], 
                 interval: str = "15m",
                 exit_params: Optional[dict[str, dict]] = None ):
        """ Initialize with proper data conversion """
        if isinstance( data, pd.DataFrame ):
            # Convert pandas to polars, preserving index if meaningful
            if hasattr( data.index, 'name' ) and data.index.name:
                data_reset = data.reset_index()
            elif not isinstance( data.index, pd.RangeIndex ):
                data_reset = data.reset_index() 
            else:
                data_reset = data.copy()
            self._lf = pl.from_pandas( data_reset ).lazy()
        elif isinstance( data, pl.DataFrame ):
            self._lf = data.lazy()  # Convert to lazy
        elif isinstance( data, pl.LazyFrame ):
            self._lf = data.clone()  # Already lazy
        else:
            raise TypeError( f"Expected pl.DataFrame, pl.LazyFrame, or pd.DataFrame, got {type(data)}" )
        
        self._is_collected = False
        self._df = None
        self.interval = interval
        if exit_params:
            self.exit_params = exit_params

    def _ensure_collected( self ) -> None:
        """ Collect the lazy frame if needed """
        if not self._is_collected:
            self._df = self._lf.collect()
            self._is_collected = True

    # Essential DataFrame interface
    @property
    def columns( self ) -> list[ str ]:
        return self._lf.collect_schema().names()
    
    @property
    def shape( self ) -> tuple[ int, int ]:
        self._ensure_collected()
        return self._df.shape
    
    @property
    def dtypes( self ) -> dict:
        """ Return a dictionary of column names and their data types """
        return { col: str( self._lf.schema[ col ] ) for col in self._lf.columns }
    
    def tail( self, n: int = 5 ) -> pl.DataFrame:
        """ Return the last few rows of the DataFrame """
        return self._lf.tail( n ).collect()
    
    def head( self, n: int = 5 ) -> pl.DataFrame:
        """ Return the first few rows of the DataFrame """
        return self._lf.head( n ).collect()
    
    def slice( self, start: int = None, end: int = None ) -> pl.DataFrame:
        """ 
        Slice the DataFrame from start to end.
        If start is None, it defaults to 0.
        If end is None, it defaults to the length of the DataFrame.
        """
        self._ensure_collected()
        if start is None:
            start = 0
        if end is None:
            end = self.shape[0]
        
        return self._df.slice( start, end - start )
    
    # DataFrame-like methods
    def with_columns( self, exprs: list[ pl.Expr ] ) -> 'SignalGenerator':
        """ Return new instance with added columns """
        new_lf = self._lf.with_columns( exprs )
        new_instance = SignalGenerator.__new__( SignalGenerator )
        new_instance._lf = new_lf
        new_instance._is_collected = False
        new_instance._df = None
        
        return new_instance
    
    def select( self, *args ) -> 'SignalGenerator':
        """ Return new instance with selected columns """
        new_lf = self._lf.select( *args )
        new_instance = SignalGenerator.__new__( SignalGenerator )
        new_instance._lf = new_lf
        new_instance._is_collected = False
        new_instance._df = None
        return new_instance
    
    def drop( self, columns ) -> 'SignalGenerator':
        """ Return new instance with dropped columns """
        new_lf = self._lf.drop( columns )
        new_instance = SignalGenerator.__new__( SignalGenerator )
        new_instance._lf = new_lf
        new_instance._is_collected = False
        new_instance._df = None
        return new_instance
    
    def __getitem__( self, cols ):
        """ Support df[ cols ] syntax - triggers collection """
        self._ensure_collected()
        return self._df[ cols ]
    
    def __setitem__( self, cols, value ):
        """ Support df[ cols ] = value syntax """
        if isinstance( value, ( pl.Series, pl.Expr ) ):
            self._lf = self._lf.with_columns( [ value.alias( cols ) ] )
        else:
            self._lf = self._lf.with_columns( [ pl.lit( value ).alias( cols ) ] )
        self._is_collected = False  # Mark as needing re-collection
    
    def to_pandas( self ) -> pd.DataFrame:
        """ Convert to pandas DataFrame """
        self._ensure_collected()
        return self._df.to_pandas()
    
    def to_polars( self ) -> pl.DataFrame:
        """ Get collected Polars DataFrame """
        self._ensure_collected()
        return self._df.clone()
    
    def collect( self ) -> 'SignalGenerator':
        """ Force collection and return self for method chaining """
        self._ensure_collected()
        return self

    ####################################
    ##### Time Interval Resampling #####
    ####################################
    # For strategies that use multiple time intervals
    def _resample(self, df: pl.DataFrame, interval: str) -> pl.DataFrame:
        """ 
        Resample the DataFrame to the specified interval using group_by_dynamic
        
        Args:
            df: Input DataFrame with datetime column
            interval: Polars duration string ('1d', '4h', '1h', etc.)
        """
        return (df.sort("datetime")
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
    
    def addMultiTimeframeIndicators(self, indicators: dict[str, list[pl.Expr]]) -> None:
        """
        Pre-compute multi-timeframe indicators with custom expressions
        
        Args:
            indicators: Dict mapping intervals to list of expressions
            
        Usage:
            sg.addMultiTimeframeIndicators({
                "1d": [
                    pl.col("close_price").rolling_mean(21).shift(1).alias("Daily_SMA_21"),
                    pl.col("close_price").ewm_mean(span=50).shift(1).alias("Daily_EMA_50")
                ],
                "4h": [
                    pl.col("close_price").rolling_mean(10).shift(1).alias("4H_SMA_10"),
                    atr_expression.shift(1).alias("4H_ATR_14")
                ]
            })
        """
        if hasattr(self, '_mtf_indicators_computed') and self._mtf_indicators_computed:
            return  # Already computed
            
        # Collect primary data once
        primary_df = self._lf.collect()
        
        # Process each interval
        for interval, expressions in indicators.items():
            if not expressions:
                continue
                
            # Resample to target interval
            resampled_df = self._resample(primary_df, interval)
            
            # Compute indicators on resampled data
            indicators_df = (
                resampled_df.lazy()
                .with_columns(expressions)  # Apply expressions directly
                .collect().sort("datetime")
            )
            
            # Join back to primary timeframe
            indicator_cols = [expr.meta.output_name() for expr in expressions]
            primary_df = (
                primary_df.sort("datetime")
                .join_asof(
                    indicators_df.select(["datetime"] + indicator_cols),
                    on="datetime",
                    strategy="backward"
                )
            )
        
        # Update the lazy frame
        self._lf = primary_df.lazy()
        self._mtf_indicators_computed = True


    ###############################
    ##### Technical Indicator #####
    ###############################
    @necessary_columns( "close_price" )
    def _getAyami( self, window_size: int = 21,
                  ) -> pl.Expr:
        """
        Compute the Ayami indicator, a type of moving average.
        Args:
            - window_size(int): the window size for computing Ayami
        """
        shunka = int( (window_size - 1)/ 2 )
        ayami_expr = pl.col( "close_price" ) + (pl.col("close_price") - pl.col("close_price").shift(shunka))
        Ayami_expr = ayami_expr.rolling_mean( window_size ).alias( f'Ayami' )

        return Ayami_expr

    def _getRSI( self, window_size: int = 21 ) -> pl.Expr:
        ''' 
        Compute RSI, see: https://www.investopedia.com/terms/r/rsi.asp
        
        Args: 
            - window_size(int): the parameter for computing RSI
        '''

        alpha = 1.0 / window_size
    
        # Calculate price changes
        price_change = pl.col( 'close_price' ).diff()
        
        # Calculate gains and losses using pl.when (not method on expression)
        gains = pl.when( price_change > 0 ).then( price_change ).otherwise( 0 )
        losses = pl.when( price_change < 0 ).then( -price_change ).otherwise( 0 )
        
        # Calculate average gains and losses
        avg_gains  = gains.ewm_mean( alpha = alpha, adjust = False )
        avg_losses = losses.ewm_mean( alpha = alpha, adjust = False )
        
        # Create row number for masking first window_size values
        row_nr = pl.int_range( pl.len() )
        
        # Calculate RSI
        return (
            pl.when( price_change.is_null() )
                .then( None )    
              .when( row_nr < window_size )
                .then( None )
              .when( avg_losses == 0 )
                .then( 100.0 )
              .when( avg_gains == 0 )
                .then( 0.0 )
              .otherwise( 100.0 - ( 100.0 / ( 1 + avg_gains / avg_losses ) ) )
              .alias( f'RSI_{window_size}' )
        )

    @necessary_columns( 'close_price' )
    def _getBBand( self, window_size: int = 21, sigma: int = 2 ) -> list[ pl.Expr ]:
        """ 
        Build expressions for Bollinger Bands calculation
        
        Args:
            - params (dict): Dictionary containing 'window_size' and 'sigma' parameters
        """

        # Calculate SMA and standard deviation
        sma_expr = pl.col( 'close_price' ).rolling_mean( window_size )
        std_expr = ( pl.col( 'close_price' )
                       .rolling_std( window_size )
                       .alias( f'RealizedAbsVol' )
        )
        sma_expr = sma_expr.alias( 'mid_band' )

        return (( sma_expr + sigma * std_expr ).alias( f'upper_band' ), 
                sma_expr,
                ( sma_expr - sigma * std_expr ).alias( f'lower_band' ),
        )

    @necessary_columns( "close_price", "high_price", "low_price" )
    def _getATR( self, period: int = 14 ) -> pl.Expr:
        """
        Build expressions for Average True Range (ATR) calculation

        Args:
            - period (int): The number of periods to use for the ATR calculation (default is 14)
        """

        # Calculate True Range components
        hl_expr = pl.col( "high_price" ) - pl.col( "low_price" )
        hc_expr = ( pl.col( "high_price" ) - pl.col( "close_price" ).shift( 1 ) ).abs()
        lc_expr = ( pl.col( "low_price" ) - pl.col( "close_price" ).shift( 1 ) ).abs()

        # True Range is the maximum of the three components
        tr_expr = pl.max_horizontal( [ hl_expr, hc_expr, lc_expr ] )

        # ATR using exponentially weighted moving average
        atr_raw = tr_expr.ewm_mean( alpha = 1.0 / period, adjust = False )
        
        # Create row index for masking the first row
        row_nr = pl.int_range( pl.len() )
        atr_expr = ( pl.when( row_nr == 0 )
                       .then( None ) 
                       .otherwise( atr_raw )
                )
        
        return atr_expr.alias( f'ATR_{period}' )

    @necessary_columns( "close_price", "open_price" )
    def _getIMI(self, lookback: int = 10) -> pl.Expr:
        """
        Calculate the Intraday Momentum Index (IMI) for a given lookback period.
        
        Args:
            - lookback (int): The number of periods to use for the IMI calculation.
        """
        # Calculate the price change
        price_change = pl.col("close_price") - pl.col("open_price")
        
        # Calculate the up and down movements
        up_move = pl.when(price_change > 0).then(price_change).otherwise(0)
        down_move = pl.when(price_change < 0).then(-price_change).otherwise(0)
        
        # Calculate the average up and down movements
        sum_up = up_move.rolling_sum(lookback)
        sum_down = down_move.rolling_sum(lookback)
        
        # Calculate the IMI
        imi_expr = (sum_up / (sum_up + sum_down) * 100).alias(f'IMI_{lookback}')

        return imi_expr

    def _getEffRatio( self, sideway_filter_lookback: int = 50 ) -> pl.Expr:
        """
        Calculate the Efficiency Ratio (ER) for a given lookback period.
        
        Args:
            - sideway_filter_lookback (int): The number of periods to use for the ER calculation.
        """
        # Calculate net changes and total changes
        net_changes = (pl.col("close_price") - pl.col("close_price").shift(sideway_filter_lookback)).abs()
        tot_changes = ((pl.col("close_price") - pl.col("close_price").shift(1)).abs()).rolling_sum(sideway_filter_lookback)
        
        # Calculate Efficiency Ratio
        eff_ratio = (net_changes / tot_changes).rolling_mean(sideway_filter_lookback).fill_null(0.0).alias('EfficiencyRatio')

        return eff_ratio


    ###############################
    ##### Style Factor Scores #####
    ###############################
    @necessary_columns( "close_price" )
    def _getMomScore( self, 
                     window_size: int = 180, 
                     half_life: int = 42,
                     lag: int = 3, ) -> pl.Expr:
        """
        Build expression for momentum scores for crypto assets.
        This function presumes the incoming data interval is 4-hourly.

        Args:
            lookback(int): Rolling window size (default: 30 days).
            half_life(int): Half-life for exponential decay (default: 7 days).
            lag(int): number of bars to skip (avoid microstructure noise), default to 1 day.
        """
        
        # Calculate weights
        weights = exp_weights( window = window_size, half_life = half_life )
        
        # Build expression for weighted momentum calculation
        returns = pl.col( 'close_price' ).pct_change().shift( lag )
        
        # Use map_batches for complex rolling calculations that need numpy operations
        def weighted_cumprod_batch( s: pl.Series ) -> pl.Series:
            values = s.to_numpy()
            result = np.full( len( values ), np.nan )
            
            for i in range( window_size - 1, len( values ) ):
                window_values = values[ i - window_size + 1 : i + 1 ]
                if not np.isnan( window_values ).any():
                    weighted_returns = window_values * weights[ -len( window_values ): ]
                    result[ i ] = ( np.cumprod( 1 + weighted_returns ) - 1 )[ -1 ]
            
            return pl.Series( result )
        
        col_name = f'MomScore_{window_size}_{half_life}_{lag}'

        return ( returns.map_batches( weighted_cumprod_batch )
                        .alias( col_name )
            )


    ############################
    ##### Strategy Signals #####
    ############################
    @necessary_columns( "close_price" )
    def _getSuperTrendSignals( self,
                               window_size: int = 21,
                               ayami_multi: float = 1.0,
                               sideway_filter_lookback: int = 50,
                               extrem_filter: float = 2.8,
                               entry_threshold: float = 0.55 # 入场阈值
                               ) -> list[ pl.Expr ]:
        """
        超级趋势策略
        """
    
        # 计算移动均线，类似于AMA，降低趋势变化的延迟
        Ayami = self._getAyami(window_size)
        
        # Initialize the SuperTrend signals
        n = self.shape[0]
        signals = np.zeros(n, dtype = np.int8)

        ATR = self._getATR(window_size)
        ATR_rollingMax = ATR.rolling_max(window_size).fill_null(0.0)

        avg_solid_bar_length = ((pl.col("close_price") - pl.col("open_price")).abs()
                                                                .rolling_mean(window_size)
                                                                .rolling_max(window_size)
                                                                .fill_null(0.0)
        )

        vol = (ATR_rollingMax + avg_solid_bar_length) / 2.0

        # 计算上轨和下轨
        upper_band = (Ayami + ayami_multi * vol).fill_null(0.0).alias(f'upper_band')
        lower_band = (Ayami - ayami_multi * vol).fill_null(0.0).alias(f'lower_band')

        # 计算趋势指标，初始化为0
        uptrend_cond   = pl.lit(0.0).alias(f'uptrend_cond')
        downtrend_cond = pl.lit(0.0).alias(f'downtrend_cond')

        # 更新趋势判断条件
        downtrend_cond = (pl.when(Ayami.shift(1) > downtrend_cond.shift(1))
                            .then(pl.max_horizontal(lower_band, downtrend_cond.shift(1)))
                            .otherwise(lower_band)
                            .fill_null(0.0)
                        ).alias(f'downtrend_cond')
        uptrend_cond = (pl.when(Ayami.shift(1) < uptrend_cond.shift(1))
                            .then(pl.min_horizontal(upper_band, uptrend_cond.shift(1)))
                            .otherwise(upper_band)
                            .fill_null(0.0)
                        ).alias(f'uptrend_cond')

        # Create a more efficient signal calculation using built-in Polars operations        
        # First, let's create expressions for the conditions
        # Signal = 1 when ayami > previous uptrend_cond  
        # Signal = -1 when ayami < previous downtrend_cond
        # Otherwise keep previous signal
        
        uptrend_condition = Ayami > uptrend_cond.shift(1)
        downtrend_condition = Ayami < downtrend_cond.shift(1)

        # Create the basic signal without state dependency first
        base_signal = (
            pl.when(uptrend_condition)
            .then(pl.lit(1))
            .when(downtrend_condition) 
            .then(pl.lit(-1))
            .otherwise(None)
            .forward_fill()  # Forward fill to maintain state
        ).alias("base_signal")

        # 过滤震荡
        net_changes = ( pl.col("close_price") - pl.col("close_price").shift(sideway_filter_lookback) ).abs()
        tot_changes = ( (pl.col("close_price") - pl.col("close_price").shift(1)).abs() ).rolling_sum(sideway_filter_lookback)
        eff_ratio = net_changes / tot_changes
        eff_ratio = eff_ratio.rolling_mean(sideway_filter_lookback).fill_null(0.0)

        # 处理极端情况
        extrem_cond = (eff_ratio > extrem_filter)

        shutdown_extrem_cond = (eff_ratio.shift(1) > entry_threshold) & ( entry_threshold > eff_ratio )
        no_open = extrem_cond
        no_open = ( pl.when(shutdown_extrem_cond)
                    .then(pl.lit(False))
                    .otherwise(no_open) 
                )
        
        signals = (pl.when( (eff_ratio > entry_threshold) & ~no_open )
                    .then(base_signal)
                    .otherwise(pl.lit(0))
                ).alias("SuperTrend")

        return signals
        
    def _getVolumeLongShortSignal( self,
                              window_size: int = 21,
        ) -> pl.Expr:
        """ 
        量价配合突破，买入
        """

        price_action = pl.col("close_price") - pl.col("open_price")
        volume_price_prod = pl.col("volume") * pl.col("close_price")

        # Calculate buy pressure using rolling operations
        buy_condition = price_action > 0
        buy_volume = pl.when(buy_condition).then(volume_price_prod).otherwise(0)
        buy_pressure = buy_volume.rolling_sum(window_size).alias('buy_pressure')

        sell_condition = price_action < 0
        sell_volume = pl.when(sell_condition).then(volume_price_prod).otherwise(0)
        sell_pressure = sell_volume.rolling_sum(window_size).alias('sell_pressure')

        buy_sell_diff = (buy_pressure - sell_pressure)

        x = int(np.ceil( window_size / 1.5 ))

        # WHQ: The following is improvised, as not sure what the original intent was
        buy_pressure_resistance = buy_sell_diff.rolling_max(x)
        buycross = (buy_sell_diff > buy_pressure_resistance).alias('buycross')

        ATR = self._getATR(window_size)
        ATR_rollingMax = ATR.rolling_max(window_size).fill_null(0.0)

        rolling_high = pl.col("high_price").rolling_max(window_size)
        upper_band = (rolling_high + ATR_rollingMax).alias('upper_band')

        # 买入信号
        buy_signals = ( pl.when(buycross & (pl.col("high_price") > upper_band)) 
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                    )
        

        # 做空信号
        shunka = int((window_size - 1) / 2)
        Ayami_ = buy_sell_diff + (buy_sell_diff - buy_sell_diff.shift(shunka)) 
        Ayami  = Ayami_.ewm_mean( alpha = 1 / window_size, adjust = False ).alias('Ayami').fill_null(0.0)
        Ayami_rolling_low = Ayami.rolling_min( window_size * 2 ).shift(1).fill_null(0.0)

        len2 = int(window_size / 5)
        mad = pl.col("close_price").rolling_mean(len2)

        ma_filter = (pl.col("close_price") < mad)
        short_entry = (Ayami.shift(1) > Ayami_rolling_low) & (Ayami < Ayami_rolling_low)

        short_signals = ( pl.when(short_entry & ma_filter)
                            .then(pl.lit(-1))
                            .otherwise(pl.lit(0))
                        )
        signals = (buy_signals + short_signals).alias("VolumeLongShort")

        return signals

    def _getSwingLongShortSignal( self,
                                  slow_window_size: int = 21,
                                  fast_window_size: int = 7,
                                  pivot_len: int = 20,
                                  total_len: int = 2,
        ) -> pl.Expr:
        """ Swing trade strategy """
        
        # 计算ATR和MACD差分
        # ATR = self._getATR(slow_window_size)
        # ATR_rollingMax = ATR.rolling_max(slow_window_size).fill_null(0.0)
        MACD_diff = (pl.col("close_price").ewm_mean(span  = fast_window_size) - pl.col("close_price").ewm_mean(span  = slow_window_size)
                    ).alias('MACD_diff')
        
        # 计算位置条件
        bar_loc_p_1_low = pl.col("low_price") > pl.col("low_price").shift(pivot_len) 
        bar_loc_p_2_low = pl.col("low_price").shift(total_len) > pl.col("low_price").shift(pivot_len)
        bar_loc_p_3_low = pl.col("low_price") > pl.col("low_price").shift(total_len)
        bar_loc_low = (bar_loc_p_1_low & bar_loc_p_2_low & bar_loc_p_3_low) # 底部逐渐抬高
        
        bar_loc_p_1_high = pl.col("high_price") < pl.col("high_price").shift(pivot_len)
        bar_loc_p_2_high = pl.col("high_price").shift(total_len) < pl.col("high_price").shift(pivot_len)
        bar_loc_p_3_high = pl.col("high_price") < pl.col("high_price").shift(total_len)
        bar_loc_high = (bar_loc_p_1_high & bar_loc_p_2_high & bar_loc_p_3_high)

        bar_loc_p_low = (pl.when(bar_loc_low)
                           .then(pl.col("low_price").shift(pivot_len))
                           .otherwise(None)
        )
        bar_loc_p_high = (pl.when(bar_loc_high)
                            .then(pl.col("high_price").shift(pivot_len))
                            .otherwise(None)
        )
        
        # 多头
        long_location = (
                        bar_loc_p_low.shift(1).is_not_null() & 
                        bar_loc_p_low.is_not_null() & 
                        (bar_loc_p_low.shift(1) > bar_loc_p_low)
                )
        long_filter = MACD_diff > MACD_diff.shift(pivot_len)
        long_signals = ( pl.when(long_location & long_filter)
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                        )
        
        ## 空头
        short_location = (
                         bar_loc_p_high.shift(1).is_not_null() & 
                         bar_loc_p_high.is_not_null() & 
                         (bar_loc_p_high.shift(1) < bar_loc_p_high)
                    )
        short_filter = MACD_diff < MACD_diff.shift(pivot_len)
        short_signals = ( pl.when(short_location & short_filter)
                        .then(pl.lit(-1))
                        .otherwise(pl.lit(0))
                        )
        
        # 组合信号
        signals = (long_signals + short_signals).alias("SwingLongShort")

        return signals

    def _getMRLongShortSignal(self,
                              lookback: int = 10,
                              sideway_filter_lookback: int = 50,
                              lower_boundary: int = 25,
                              upper_boundary: int = 75,
                              entry_threshold: float = 0.1
                              ) -> pl.Expr:
        
        """ IMI动量反转策略 """
        # 计算IMI指标
        imi = self._getIMI(lookback).fill_null(0.0).alias('IMI')

        long_crossover = (imi.shift(1) <= lower_boundary) & (imi > lower_boundary)
        short_crossover = (imi.shift(1) >= upper_boundary) & (imi < upper_boundary)

        # 创建基础信号，使用0而不是None
        base_signal = (
            pl.when(long_crossover)
            .then(pl.lit(1))
            .when(short_crossover)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))  # Use 0 instead of None
        ).alias("MRLongShort")

        return base_signal


    def _getSMAStratSignal( self,
                       window_size: int = 30, 
                       S: int = 2 # shrinkage coefficient
                       ) -> pl.Expr:
        """ Simple Moving Average (SMA) strategy signals """

        MA1 = (((pl.col("open_price") + pl.col("low_price") + pl.col("close_price")) / 3)
               .rolling_mean(int(window_size / S)).alias('MA1')
        )
        slow_MA1 = MA1.ewm_mean(span = int( window_size / S), adjust = False).alias('slow_MA1')

        MA2 = (((pl.col("open_price") + pl.col("low_price") + pl.col("close_price")) / 3)
               .rolling_mean(window_size).alias('MA2')
        )
        slow_MA2 = MA2.ewm_mean(span = window_size, adjust = False).alias('slow_MA2')

        DLH = slow_MA1 - slow_MA2
        MADLH = DLH.rolling_mean(int(window_size / S)).alias('MADLH')

        # 波动差大于周期内均值，短期指数均线大于周期内均值，长期指数均线大于周期内均值，短期指数均线大于长期均值；
        long_filter = (
            (DLH > 0) 
            &(DLH > MADLH)
            &(MA1 > slow_MA1)
            &(MA2 > slow_MA2)
            &(slow_MA1 > slow_MA2)
        )

        #波动差小于周期内均值，短期指数均线小于周期内均值，长期指数均线小于周期内均值，短期指数均线小于长期均值；
        short_filter = (
            (DLH < 0)
            &(DLH < MADLH)
            &(MA1 < slow_MA1)
            &(MA2 < slow_MA2)
            &(slow_MA1 < slow_MA2)
        )

        highest_highs = (pl.col("high_price")
                            .shift(1)
                            .rolling_max(window_size = window_size)
        )
        lowest_lows = (pl.col("low_price")
                            .shift(1)
                            .rolling_min(window_size = window_size)
        )

        signals = (pl.when(long_filter & (pl.col("high_price") > highest_highs))
                    .then(1)
                    .when(short_filter & (pl.col("low_price") < lowest_lows))
                    .then(-1)
                    .otherwise(0)
                    .alias("SMAStrat")
        )

        return signals
    
    def _getTurtleLongShortSignals( self,
                                    window_size: int = 20, # bolength
                                    length: int = 21,
                                    fslength: int = 65,
                                    ) -> pl.Expr:
        """
        Turtle trading strategy signals.
        This is a placeholder for the actual implementation.
        """
        
        # 1. 计算唐奇安通道
        DonchianHigh = pl.col("high_price").rolling_max(window_size)
        DonchianLow = pl.col("low_price").rolling_min(window_size)
        DonchianMid = ((DonchianHigh + DonchianLow) / 2.0).alias('DonchianMid')
        
        # 2. 判断中轨变化
        isMidUp = (DonchianMid > DonchianMid.shift(1)).fill_null(False)
        isMidDown = (DonchianMid < DonchianMid.shift(1)).fill_null(False)

        # 3. 计算波动率归一化
        max_high_pclose = pl.max_horizontal(pl.col("high_price"), pl.col("close_price").shift(1))
        min_low_pclose = pl.min_horizontal(pl.col("low_price"), pl.col("close_price").shift(1))

        rms_tr_1  = (max_high_pclose / min_low_pclose).log().pow(2)
        rms_tr    = rms_tr_1.rolling_sum(length).sqrt()
        ma_rms_tr = rms_tr.rolling_mean(fslength) 

        # 4. 震荡检测
        # For each group where HL changed, we want to accumulate values
        # Use window functions to handle state-like behavior

        # Count changes in each group (represents 'x' counter)
        up_change_groups = isMidUp.cast(pl.Int32).cum_sum()
        up_group_size = isMidUp.cast(pl.Int32).rolling_sum(3)

        down_change_groups = isMidDown.cast(pl.Int32).cum_sum()
        down_group_size = isMidDown.cast(pl.Int32).rolling_sum(3)

        # Rolling sum of Donchian mid band values when changed (represents 'sumAG')
        hl_sum_when_MidUp = (pl.when(isMidUp)
                                .then(DonchianMid)
                                .otherwise(0)
                                .rolling_sum(3))
        hl_sum_when_MidDown = (pl.when(isMidDown)
                                .then(DonchianMid)
                                .otherwise(0)
                                .rolling_sum(3))

        # Calculate average when group_size > 2 (equivalent to x > 2)
        DonchianMidAvg_when_MidUp = (pl.when(up_group_size > 2)
                                    .then(hl_sum_when_MidUp / up_group_size)
                                    .otherwise(None)
                                    .forward_fill()  # Carry forward last valid average
        )

        DonchianMidAvg_when_MidDown = (pl.when(down_group_size > 2)
                                        .then(hl_sum_when_MidDown / down_group_size)
                                        .otherwise(None)
                                        .forward_fill()  # Carry forward last valid average
        )

        # Detect changes in HL average (equivalent to condRHL_average)
        is_up_changed = (DonchianMidAvg_when_MidUp.shift(1) != DonchianMidAvg_when_MidUp.shift(2)).fill_null(False)
        is_down_changed = (DonchianMidAvg_when_MidDown.shift(1) != DonchianMidAvg_when_MidDown.shift(2)).fill_null(False)
        

        rhl_average_up = (pl.when(is_up_changed)
                        .then(DonchianMidAvg_when_MidUp.shift(1))  # Take previous value when change detected
                        .otherwise(0.0) # TODO: Is this correct?
        )
        rhl_average_down = (pl.when(is_down_changed)
                            .then(DonchianMidAvg_when_MidDown.shift(1))  # Take previous value when change detected
                            .otherwise(0.0) 
        )
        long_cond  = (
            (DonchianMidAvg_when_MidUp.shift(1) <= rhl_average_up) &
            (rhl_average_up < DonchianMidAvg_when_MidUp)
        ).alias('swing_condition')

        short_cond = (
            (DonchianMidAvg_when_MidDown.shift(2) > rhl_average_down) &
            (rhl_average_down > DonchianMidAvg_when_MidDown)
        )

        low_vol_cond = rms_tr < ma_rms_tr
        high_vol_cond = rms_tr > ma_rms_tr

        long_signals = (pl.when((low_vol_cond | high_vol_cond) & long_cond)
                        .then(1)
                        .otherwise(0)
        )

        short_signals = (pl.when((low_vol_cond | high_vol_cond) & short_cond)
                        .then(-1)
                        .otherwise(0)
        )

        signals = (long_signals + short_signals).alias("TurtleLongShort")

        return signals

    @necessary_columns("high_price", "low_price", "close_price", "volume")
    def _getTurtleLongShort2Signals(self,
                                 long_window_size: int = 288, 
                                 short_window_size: int = 16,
                                 turtle_er_threshold: float = 0.3,
                                 soup_er_threshold: float = 0.1,
                                 volume_threshold: float = 1.5,

        ) -> pl.Expr:
        """
        长周期：海龟策略bet突破
        短周期：海龟策略bet假突破后反向
        震荡检测：ER + 暴量
        """

        long_DonchianHigh = pl.col("high_price").rolling_max(long_window_size).shift(1)
        long_DonchianLow = pl.col("low_price").rolling_min(long_window_size).shift(1)

        short_DonchianHigh = pl.col("high_price").rolling_max(short_window_size).shift(1)
        short_DonchianLow = pl.col("low_price").rolling_min(short_window_size).shift(1)
        
        eff_ratio = self._getEffRatio(long_window_size)

        volume_ratio = pl.col("volume") / pl.col("volume").rolling_mean(long_window_size)

        # 海龟策略
        # 1. 长周期突破
        # 2. ER大于阈值
        # 3. 成交量突破
        Turtle_long_cond = pl.col("close_price") > long_DonchianHigh
        Turtle_short_cond = pl.col("close_price") < long_DonchianLow

        Turtle_long_signal = (pl.when( Turtle_long_cond
                                        & (eff_ratio > turtle_er_threshold)
                                        & (volume_ratio > volume_threshold)
                                ).then(1).otherwise(0)
        )

        Turtle_short_signal = (pl.when( Turtle_short_cond
                                        & (eff_ratio > turtle_er_threshold)
                                        & (volume_ratio > volume_threshold)
                                ).then(-1).otherwise(0)
        )
        Turtle_signals = (Turtle_long_signal + Turtle_short_signal)
        Turtle_signals = Turtle_signals.replace(0, None)
        Turtle_signals = Turtle_signals.forward_fill().fill_null(0)  # Forward fill and fill nulls with 0

        # 过滤波动
        Turtle_signals = (pl.when(eff_ratio <= turtle_er_threshold)
                          .then(0)
                          .otherwise(Turtle_signals)
        ).alias("Turtle_signals")

        # 海龟汤策略
        # 多头陷阱：过去4小时突破后跌回
        long_trap = (pl.col("high_price").shift(16) > short_DonchianHigh.shift(16)) & (pl.col("close_price") < short_DonchianHigh.shift(16))
        # 空头陷阱：过去4小时跌破收回
        short_trap = (pl.col("low_price").shift(16) < short_DonchianLow.shift(16)) & (pl.col("close_price") > short_DonchianLow.shift(16))

        Turtle_soup_short_signal = pl.when( long_trap 
                                            & (eff_ratio < soup_er_threshold)
                                    ).then(-1).otherwise(0)
        Turtle_soup_short_signal = Turtle_soup_short_signal.forward_fill()
        Turtle_soup_long_signal = pl.when( short_trap 
                                            & (eff_ratio < soup_er_threshold)
                                    ).then(1).otherwise(0)
        
        Turtle_soup_signals = (Turtle_soup_long_signal + Turtle_soup_short_signal)
        Turtle_soup_signals = Turtle_soup_signals.replace(0, None)
        Turtle_soup_signals = Turtle_soup_signals.forward_fill().fill_null(0)  # Forward fill and fill nulls with 0
        Turtle_soup_signals = (pl.when((eff_ratio > soup_er_threshold) | (volume_ratio > volume_threshold))
                                .then(0)
                                .otherwise(Turtle_soup_signals)
        ).alias("Turtle_soup_signals")

        # 组合信号，优先海龟策略
        signals = (pl.when(Turtle_signals != 0)
                    .then(Turtle_signals)
                    .otherwise(Turtle_soup_signals)
                    .alias("TurtleLongShort2")
        )

        return signals
    
    def _getBrickLongShortSignals(self,
                                  window_size: int = 21,
                                  ) -> pl.Expr:
        """ TODO """
        
        SMA = pl.col("close_price").rolling_mean(window_size)
        cond_up = (pl.col("close_price") > SMA).fill_null(False)
        cond_down = (pl.col("close_price") < SMA).fill_null(False)

        cond_consec_up = pl.col("close_price") > pl.col("close_price").shift(1)
        cond_consec_down = pl.col("close_price") < pl.col("close_price").shift(1)
        
        # initialize status count
        status_init = pl.lit(2000) # ？？
        status_delta = (pl.when(cond_up & cond_consec_up)
                        .then(1)
                        .when(cond_up & cond_consec_down)
                        .then(0)
                        .when(cond_down & cond_consec_down)
                        .then(-1)
                        .when(cond_down & cond_consec_up)
                        .then(0)
        )

        status = (status_init + status_delta.cum_sum()).alias('status')
        vix_3  = (status.log() / status.shift(window_size).log()).fill_null(0.0)

        up_status = vix_3.rolling_max(window_size).shift(1)
        down_status = vix_3.rolling_min(window_size).shift(1)

        signals = (pl.when(up_status < vix_3)
                   .then(1)
                   .when(down_status > vix_3)
                   .then(-1)
                   .otherwise(0)
                   .alias("BrickLongShort")
        )

        return signals

    @necessary_columns("close_price", "datetime")
    def _getPSYMOMLongShortSignals(self,
                                    window_size: int = 48,
                                    ama_period: int = 14,
                                    MADDlength: int = 5,
                                    trigger: float = 0.0,
                                    
        ) -> pl.Expr:
        """
        Args:
            - window_size (int): The number of periods to use for the PSY calculation.
            - ama_period (int): The period for the Adaptive Exponential Moving Average (AEMA).
            - half_life (int): Half-life for exponential decay in momentum score calculation.
            - MADDlength (int): The period for the daily SMA used in the MADD calculation.
            - trigger (float): The trigger level for entry signals.
        """

        # mom = pl.col("close_price").pct_change().shift(1).ewm_mean(span = window_size)
        mom = 0.5 * (pl.col("close_price") - pl.col("close_price").shift(int(window_size / 3)))
        mom = mom.fill_null(0.0)

        up_bars = (pl.col("close_price") > pl.col("close_price").shift(1)).cast(pl.Int8).rolling_sum(window_size)
        psy = (up_bars / window_size).fill_null(0.0).alias('PSY')

        new_mom = (psy * mom).fill_null(0.0).alias('new_mom')

        mltp1 = pl.lit(2 / (1 + window_size))

        max_new_mom = new_mom.rolling_max(ama_period)
        min_new_mom = new_mom.rolling_min(ama_period)
        range_new_mom = max_new_mom - min_new_mom
    
        mltp2 = (pl.when(range_new_mom == 0.0)
                .then(0.0)  # Handle division by zero
                .otherwise(
                    ((new_mom - min_new_mom) - (max_new_mom - new_mom)).abs() / range_new_mom
                )
        )
        
        rate = (mltp1 * (pl.lit(1) + mltp2)).fill_null(0.0).alias('rate')

        def compute_aema_recursive(rate_vals, new_mom_vals):
            """递归计算AEMA，初始值为0"""
            n = len(rate_vals)
            aema_vals = np.zeros(n)
            aema_vals[0] = new_mom_vals[0]
            
            for i in range(1, n):
                aema_vals[i] = aema_vals[i-1] + rate_vals[i] * (new_mom_vals[i] - aema_vals[i-1])
            
            return aema_vals

        # 使用map_batches应用递归函数
        aema = (
            pl.struct([rate.alias("rate"), new_mom.alias("new_mom")])
            .map_batches(
                lambda s: pl.Series(
                    "AEMA", 
                    compute_aema_recursive(
                        s.struct.field("rate").to_numpy(), 
                        s.struct.field("new_mom").to_numpy()
                    )
                ), 
                return_dtype = pl.Float64
            )
            .alias('AEMA')
        )

        # 计算MADD和触发条件
        # compute daily SMA
        daily_sma_expr = (pl.col("close_price")
                          .rolling_mean(MADDlength)
                          .shift(1)
                          .alias('daily_sma')
                        )

        self.addMultiTimeframeIndicators( {"1d": [daily_sma_expr]} )

        long_entry = pl.col("close_price") > pl.col("daily_sma")
        long_entry2 = (trigger < aema) &  (trigger > aema.shift(1))

        long_signals = (pl.when(long_entry & long_entry2)
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
        )

        short_entry = pl.col("close_price") < pl.col("daily_sma")
        short_entry2 = (trigger > aema) &  (trigger < aema.shift(1))
        short_signals = (pl.when(short_entry & short_entry2)
                        .then(pl.lit(-1))
                        .otherwise(pl.lit(0))
        )

        signals = (long_signals + short_signals).alias("PSYMOMLongShort")

        return signals

    def _getPSYVOLLongShortSignals(self,
                                    window_size: int = 21,
                                    ma_window_size: int = 10,
        ) -> pl.Expr:

        """
        """
        
        net_changes = (pl.col("close_price") / pl.col("close_price").shift(1) - 1)
        rv = net_changes.pow(2).rolling_sum(window_size).alias('RV')

        # 计算正负波动率数据
        pos_net_changes = pl.when(net_changes > 0).then(net_changes).otherwise(0.0)
        neg_net_changes = pl.when(net_changes < 0).then(-net_changes).otherwise(0.0)

        rv_pos = pos_net_changes.pow(2).rolling_sum(window_size).alias('RV_pos')
        rv_neg = neg_net_changes.pow(2).rolling_sum(window_size).alias('RV_neg')
        # 计算RSJ指标
        rsj = (rv_pos - rv_neg) / rv
        rsj = rsj.fill_null(0.0).alias('RSJ')

        # 计算累计波动率
        up_vol_cum = (pl.when(net_changes > 0)
                      .then(pl.col("volume"))
                        .otherwise(0.0)
        ).rolling_sum(window_size).alias('up_vol_cum')
        down_vol_cum = (pl.when(net_changes < 0)
                        .then(pl.col("volume"))
                        .otherwise(0.0)
        ).rolling_sum(window_size).alias('down_vol_cum')

        vol_factor = (up_vol_cum - down_vol_cum) / (up_vol_cum + down_vol_cum)

        shunka = int((ma_window_size -1) / 2)
        Ayami_ = vol_factor + (vol_factor - rsj.shift(shunka)).fill_null(0.0)
        Ayami  = Ayami_.rolling_mean( shunka )

        long_entry = (rsj.shift(1) < 0) & (rsj > 0)
        short_entry = (rsj.shift(1) > 0) & (rsj < 0)

        long_signals = (pl.when(long_entry & (Ayami > 0))
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
        )

        short_signals = (pl.when(short_entry & (Ayami < 0))
                         .then(pl.lit(-1))
                         .otherwise(pl.lit(0))
        )

        signals = (long_signals + short_signals).alias("PSYVOLLongShort")
        
        return signals

    def _getSuperTrendADPExitSignals(self,
                                    window_size: int = 21, # boLength
                                    vol_length: int = 20, # 
                                    maria_multi: float = 2.25,
        ) -> pl.Expr:

        """
        """

        # 1. 计算唐奇安通道
        DonchianHigh_c = pl.col("close_price").rolling_max(window_size)
        DonchianLow_c = pl.col("close_price").rolling_min(window_size)
        DonchianMid_c = ((DonchianHigh_c + DonchianLow_c) / 2.0).alias('DonchianMid_c')
        # DonchianMid_c_smooth = DonchianMid_c.rolling_mean(3)
        
        
        max_high_pclose = pl.max_horizontal(pl.col("high_price"), pl.col("close_price").shift(1))
        min_low_pclose = pl.min_horizontal(pl.col("low_price"), pl.col("close_price").shift(1))

        rms_tr_1  = (max_high_pclose / min_low_pclose).log().pow(2)
        rms_tr    = rms_tr_1.rolling_sum(vol_length).sqrt() * 500
        atr = self._getATR(vol_length).fill_null(0.0)

        up = DonchianMid_c + (maria_multi * atr)
        down = DonchianMid_c - (maria_multi * atr)

        Nagai_up = pl.lit(0.0)  # Initialize with zero
        Nagai_up = (pl.when(pl.col("close_price").shift(1) > Nagai_up.shift(1))
                    .then( pl.max_horizontal(down, Nagai_up.shift(1)) )
                    .otherwise(down)
        )

        Nagai_down = pl.lit(0.0)  # Initialize with zero
        Nagai_down = (pl.when(pl.col("close_price").shift(1) < Nagai_down.shift(1))
                      .then( pl.min_horizontal(up, Nagai_down.shift(1)) )
                      .otherwise(up)
        )

        # 更新趋势判断条件
        base_signals = (pl.when(pl.col("close_price") > Nagai_down.shift(1))
                            .then(1)
                        .when(pl.col("close_price") < Nagai_up.shift(1))
                            .then(-1)
                        .otherwise(None)
                        )
        base_signals = base_signals.forward_fill().alias('base_signals')
        long_filter = (base_signals == 1)
        short_filter = (base_signals == -1)
        
        
        
        # 比较close与当天开盘价
        is_open = (pl.col("datetime").dt.hour() == 0) & (pl.col("datetime").dt.minute() == 0)
        day_open = pl.when(is_open).then(pl.col("open_price")).otherwise(None).forward_fill()
        higher_than_day_open = (pl.col("close_price") >= (day_open + maria_multi * atr))
        lower_than_day_open = (pl.col("close_price") <= (day_open - maria_multi * atr))

        long_signals = (pl.when(long_filter & higher_than_day_open)
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
        )
        
        short_signals = (pl.when(short_filter & lower_than_day_open)
                        .then(pl.lit(-1))
                        .otherwise(pl.lit(0))
        )

        signals = (long_signals + short_signals).alias("SuperTrendADPExit")

        return signals

    def _getRSIBBReversalSignals(self,
                            window_size: int = 21,
                            over_bought: float = 70,
                            over_sold: float = 30,
                            sigma: int = 2,
                            ) -> pl.Expr:
        """ RSI + BB 抄底 """
        
        
        rsi = self._getRSI(window_size).fill_null(50.0).alias('RSI')
        upper_band, _, lower_band = self._getBBand(window_size, sigma)

        # 多头条件：rsi低于超卖线且价格跌破下轨后回升
        long_cond = ((rsi < over_sold) 
                     & (pl.col("close_price").shift(1) < lower_band.shift(1))
                     & (pl.col("close_price") >= lower_band)
        )
        # 空头条件：rsi高于超买线且价格突破上轨后回落
        short_cond = ((rsi > over_bought)
                      & (pl.col("close_price").shift(1) > upper_band.shift(1))
                      & (pl.col("close_price") <= upper_band)
        )

        long_signals = (pl.when(long_cond)
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
        )
        short_signals = (pl.when(short_cond)
                        .then(pl.lit(-1))
                        .otherwise(pl.lit(0))
        )

        signals = (long_signals + short_signals).alias("RSIBBReversal")

        return signals

    def _getBollEnhancedSignals(self,
                                window_size: int = 21,   
                                sigma: int = 2,
                                X: int = 1,

                                ) -> pl.Expr:
        """ TODO """
        upper_band, mid_band, lower_band = self._getBBand(window_size, sigma)
        ema_long = pl.col("close_price").ewm_mean(span = window_size * X).alias('ema_long')

        BB_ratio = (pl.col("close_price") - lower_band) / (upper_band - lower_band)

        BB_width = (upper_band - lower_band) / upper_band * 100

        long_cond1 = pl.lit(0)
        long_cond2 = pl.lit(0)
        long_cond3 = pl.lit(0)

        is_up_mid_band = (mid_band > mid_band.shift(1)).fill_null(False)
        is_up_upper_band = (upper_band > upper_band.shift(1)).fill_null(False)
        is_up_ema_long = (ema_long > ema_long.shift(1)).fill_null(False)

        long_cond1 = (pl.when(is_up_mid_band & (pl.col("close_price") > mid_band.shift(1)))
                      .then(mid_band)
                      .otherwise(long_cond1.shift(1))
        )
        long_cond2 = (pl.when(is_up_upper_band & (pl.col("high_price") > upper_band))
                      .then(upper_band)
                      .otherwise(long_cond2.shift(1))
        )
        long_cond3 = (pl.when(is_up_ema_long & (pl.col("close_price") > ema_long))
                      .then(ema_long)
                      .otherwise(long_cond3.shift(1))
        )

        short_cond1 = pl.lit(0)
        short_cond2 = pl.lit(0)
        short_cond3 = pl.lit(0)
    
        is_down_mid_band = (mid_band < mid_band.shift(1)).fill_null(False)
        is_down_lower_band = (lower_band < lower_band.shift(1)).fill_null(False)
        is_down_ema_long = (ema_long < ema_long.shift(1)).fill_null(False)

        short_cond1 = (pl.when(is_down_mid_band & (pl.col("close_price") < mid_band.shift(1)))
                       .then(mid_band)
                       .otherwise(short_cond1.shift(1))
        )
        short_cond2 = (pl.when(is_down_lower_band & (pl.col("low_price") < lower_band))
                       .then(lower_band)
                       .otherwise(short_cond2.shift(1))
        )
        short_cond3 = (pl.when(is_down_ema_long & (pl.col("close_price") < ema_long))
                       .then(ema_long)
                       .otherwise(short_cond3.shift(1))
        )

        long_signals = (pl.when(long_cond1 > long_cond1.shift(1) 
                                & (long_cond2 > long_cond2.shift(1))
                                & (long_cond3 > long_cond3.shift(1))
                                & BB_ratio > 0.5
                                )
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                        )
        
        short_signals = (pl.when(short_cond1 < short_cond1.shift(1)
                                 & (short_cond2 < short_cond2.shift(1))
                                 & (short_cond3 < short_cond3.shift(1))
                                 & BB_ratio < 0.5
                                    )
                        .then(pl.lit(-1))
                        .otherwise(pl.lit(0))
                        )
        signals = (long_signals + short_signals).alias("BollEnhanced")

        return signals


    def _getAlligatorSignals(self,
                             lookback_window: int = 15,
                             jaw_length: int = 13,
                             teeth_length: int = 8,
                             lips_length: int = 5,
                             numberbars: int = 3,
                             allow_rev_signals: bool = True,
                             ) -> pl.Expr:
        """ 默认15分钟，但用4小时bar来计算鳄鱼线 
        Args:
            - lookback_window (int): 用于计算背离的回溯窗口
            - jaw_length (int): 鳄鱼颚线的长度
            - teeth_length (int): 鳄鱼齿线的长度
            - lips_length (int): 鳄鱼唇线的长度
            - numberbars (int): 用于计算背离的bar数量
            - allow_rev_signals (bool): 是否允许背离信号
        """
        # TODO: double-check this strategy
        
        fast_sma = pl.col("close_price").rolling_mean(5).alias('fast_sma')
        slow_sma = pl.col("close_price").rolling_mean(34).alias('slow_sma')

        ao = fast_sma - slow_sma
        mav_up = (ao > 0).cast(pl.Int32).rolling_sum(32)
        mav_down = (ao < 0).cast(pl.Int32).rolling_sum(32)

        lowest_ao = ao.rolling_min(lookback_window).shift(1).alias('lowest_ao')
        highest_ao = ao.rolling_max(lookback_window).shift(1).alias('highest_ao')

        lowest_lows = pl.col("low_price").rolling_min(lookback_window).shift(1).alias('lowest_lows')
        highest_highs = pl.col("high_price").rolling_max(lookback_window).shift(1).alias('highest_highs')

        jaw_expr = pl.col("close_price").ewm_mean(jaw_length).alias('jaw')
        teeth_expr = pl.col("close_price").ewm_mean(teeth_length).alias('teeth')
        lips_expr = pl.col("close_price").ewm_mean(lips_length).alias('lips')
        self.addMultiTimeframeIndicators( {"4h": [jaw_expr, teeth_expr, lips_expr]} )

        # basic signals
        long_base_cond = (
            (ao > ao.shift(1)) &                    # AO上升
            (pl.col("lips") > pl.col("teeth")) &    # 唇线 > 齿线
            (pl.col("teeth") >= pl.col("jaw")) &    # 齿线 >= 颚线  
            (mav_down > numberbars)                 # AO之前向下足够长时间
        )
        
        short_base_cond = (
            (ao < ao.shift(1)) &                    # AO下降
            (pl.col("lips") < pl.col("teeth")) &    # 唇线 < 齿线
            (pl.col("teeth") <= pl.col("jaw")) &    # 齿线 <= 颚线
            (mav_up > numberbars)                   # AO之前向上足够长时间
        )

        if allow_rev_signals:
            # 检测背离事件
            bullish_divergence = (
                (pl.col("low_price") < lowest_lows) & 
                (ao > lowest_ao)
            )
            
            bearish_divergence = (
                (pl.col("high_price") > highest_highs) &
                (ao < highest_ao) 
            )

            # 使用cumsum和差分来创建信号组ID (每次背离开始一个新组)
            bullish_group_id = bullish_divergence.cast(pl.Int32).cum_sum()
            bearish_group_id = bearish_divergence.cast(pl.Int32).cum_sum()
            
            # 为每个组创建计数器 (组内的位置)
            bullish_counter = (
                pl.int_range(pl.len())
                .over(bullish_group_id)  # 每个组内重新计数
            )
            
            bearish_counter = (
                pl.int_range(pl.len()) 
                .over(bearish_group_id)
            )

            # 信号激活条件：
            # 1. 发生了背离事件 (group_id > 0)
            # 2. 在lookback_window时间内 (counter <= lookback_window) 
            # 3. 价格满足条件 (close vs lips)
            bullish_signal_active = (
                (bullish_group_id > 0) &              # 至少发生过一次背离
                (bullish_counter <= lookback_window) & # 在过去lookback_window个bar里发生过背离
                (pl.col("close_price") > pl.col("lips")) # price > lips
            )
            
            bearish_signal_active = (
                (bearish_group_id > 0) &
                (bearish_counter <= lookback_window) &
                (pl.col("close_price") < pl.col("lips"))
            )
            
            # 处理两种信号同时激活时都关闭的逻辑
            both_active = bullish_signal_active & bearish_signal_active
            bullish_signal_active = (pl.when(both_active)
                                        .then(False)
                                        .otherwise(bullish_signal_active)
            )
            bearish_signal_active = (pl.when(both_active)
                                        .then(False)
                                        .otherwise(bearish_signal_active)
            )
            
    
            # 合并所有信号
            combined_long = long_base_cond | bullish_signal_active
            combined_short = short_base_cond | bearish_signal_active
            
            signals = (
                pl.when(combined_long).then(1)
                .when(combined_short).then(-1)
                .otherwise(0)
                .alias("Alligator")
            )
            
        else:
            # 没有背离信号时，只使用基础信号
            signals = (
                pl.when(long_base_cond).then(1)
                .when(short_base_cond).then(-1)
                .otherwise(0)
                .alias("Alligator")
            )

        return signals


    ############################
    ##### Signal Extension #####
    ############################
    """ 主要用于部分策略信号过于稀疏 """
    # 1. Minimal Holding period
    def _create_signal_extension_expr(self, signal_col: str, persistence_hours: int) -> pl.Expr:
        """
        Helper function for signal extension
        """
        periods = persistence_hours * 4  # 15分钟转周期数

        def extend_signal_numpy(signal_series: pl.Series) -> pl.Series:
            """NumPy实现的信号延续逻辑"""
            signals = signal_series.to_numpy()
            result = signals.copy()
            n = len(signals)

            i = 0
            while i < n:
                if signals[i] != 0:  # 发现非零信号
                    current_signal = signals[i]

                    # 寻找下一个非零信号的位置
                    next_signal_pos = None
                    for j in range(i + 1, min(i + periods + 1, n)):
                        if signals[j] != 0:
                            next_signal_pos = j
                            break

                    # 延续信号到下一个信号位置或periods限制
                    end_pos = next_signal_pos if next_signal_pos else min(i + periods, n)
                    for k in range(i + 1, end_pos):
                        if signals[k] == 0:  # 只覆盖空信号位置
                            result[k] = current_signal

                    i = next_signal_pos if next_signal_pos else end_pos
                else:
                    i += 1

            return pl.Series(result)

        return (
            pl.col(signal_col)
            .fill_null(0)
            .map_batches(extend_signal_numpy, return_dtype=pl.Int32)
            .alias(signal_col)
        )

    def _applySignalsExtension(self, 
                                extend_signals: dict[str, Union[bool, int]]
                                ) -> None:
        """
        Extend signal persistence for a given number of hours.
        This is done once completed the whole signal generation dataframe.
        
        Args:
            extend_signals (dict): Dictionary with signal columns and persistence hours.
                - dict: {"signal_col": hours} signal lasting hours
                - True: extend all signals by 8 hours
                - False: do not extend signals
                - None: do not extend signals        
        """
        

        extended_expressions = []

        for signal_col, persistence_hours in extend_signals.items():
            if isinstance(persistence_hours, bool):
                persistence_hours = 8 if persistence_hours else 0

            if signal_col in self.columns and persistence_hours > 0:
                # 转换为LazyFrame进行处理
                extended_expr = self._create_signal_extension_expr(signal_col, persistence_hours)
                extended_expressions.append(extended_expr)
            else:
                print(f"{signal_col} do not existed in the dataframe.")

        # 一次性应用所有信号延续
        if extended_expressions:
            self._df = (
                self._df
                .lazy()
                .with_columns(extended_expressions)
                .collect()
            )
    
    
    ########################
    ###### Aggregation #####
    ########################
    def _buildFeatureExpressions( self, features: dict ) -> list[ pl.Expr ]:
        """
        Build a list of Polars expressions from feature specifications.
        This allows batch processing of multiple features in one operation.
        """

        expressions = []
        ## derived features from Binance data

        # Expression builders for each feature type
        expr_builders = {

            ## Strategy Signals
            "SuperTrend": self._getSuperTrendSignals,
            "VolumeLongShort": self._getVolumeLongShortSignal,
            "SwingLongShort": self._getSwingLongShortSignal,
            "MRLongShort": self._getMRLongShortSignal,
            "SMAStrat": self._getSMAStratSignal,
            "TurtleLongShort": self._getTurtleLongShortSignals,
            "TurtleLongShort2": self._getTurtleLongShort2Signals,
            "BrickLongShort": self._getBrickLongShortSignals,
            "PSYMOMLongShort": self._getPSYMOMLongShortSignals,
            "PSYVOLLongShort": self._getPSYVOLLongShortSignals,  
            "SuperTrendADPExit": self._getSuperTrendADPExitSignals,
            "RSIBBReversal": self._getRSIBBReversalSignals,
            "BollEnhanced": self._getBollEnhancedSignals,
            "Alligator": self._getAlligatorSignals,
        }
        
        # Collect expressions from all requested features
        for feature, params in features.items():
            if feature in expr_builders:
                feature_exprs = expr_builders[feature]( **params or {})
                if isinstance(feature_exprs, list):
                    expressions.extend( feature_exprs )
                else:
                    expressions.append( feature_exprs )
        
        return expressions
    
    def equipFeatures(self, 
                    features: dict[str, dict], 
                    extend_signals: Union[ dict, bool ] = None,
                    ) -> None:
        """
        Lazy-Optimized feature engineering using batch Polars expressions.
        Args: 
            features (dict): Dictionary of feature/signal parameters specifications.
            extend_signals (dict or list, optional): Specifications for extending signals.
                - dict: {"signal_col": hours} signal lasting hours
                - True: extend all signals by 8 hours
                - None: do not extend signals
            exit_logics (dict or bool, optional): Specifications for exit logic.
        """

        # Handle both dict and list inputs
        if isinstance(features, dict):
            feature_list = [features]
        elif isinstance(features, list):
            feature_list = features
        else:
            raise TypeError("features must be dict or list of dicts")

        # Build all expressions from all feature/signal dictionaries
        expressions = []
        for feature_dict in feature_list:
            expressions.extend(self._buildFeatureExpressions(feature_dict))
        
        # Apply all features in lazy mode
        if expressions:
            self._lf = self._lf.with_columns( expressions )

        # Collect the data first to see what columns were actually created
        self._df = self._lf.collect()
        self._is_collected = True

        # if extend_signals:
        #     self._applySignalsExtension(extend_signals)

        # Added exits to original signals
        if self.exit_params:
            for signal_col, params in self.exit_params.items():
                self._df = apply_exit_methods(self._df, signal_col, params)

            self._lf = self._df.lazy()
            self._is_collected = True
            
    ##########################
    ###### Visualization #####
    ##########################
    def plot_strategy_signals(self, 
                        strategy_name: str,
                        additional_columns: list = None,
                        start_date: dt.datetime = None,
                        end_date: dt.datetime = None,
                        figsize: tuple = (15, 10),
                        show_equity_curve: bool = True,
                        show_stop_lines: bool = True) -> None:
        """
        Plot strategy signals with price, equity curve (in separate subplot), and stop loss lines.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import numpy as np
        except ImportError:
            raise ImportError("plotly and numpy are required for plotting. Install with: pip install plotly numpy")
        
        # Ensure data is collected
        self._ensure_collected()
        df = self._df.clone()

        # Filter data by date range if provided
        if start_date:
            df = df.filter(pl.col("datetime") >= start_date)
        if end_date:
            df = df.filter(pl.col("datetime") <= end_date)
        
        # Check if required columns exist
        required_cols = ["datetime", "close_price", strategy_name]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert Polars DataFrame to Pandas for Plotly compatibility
        df_pd = df.to_pandas()
        
        # Determine subplot configuration
        if show_equity_curve:
            subplot_titles = ["Price Chart with Signals", "Equity Curve"]
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1,  # ✅ Increased spacing to prevent overlap
                row_heights=[0.7, 0.3],  # Price chart gets more space
                shared_xaxes=True,  # ✅ Share x-axes for perfect alignment
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]  # ✅ Explicit specs
            )
            price_row = 1
            equity_row = 2
        else:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": False}]]
            )
            price_row = 1
            equity_row = None
        
        # Get position column
        position_col = f"{strategy_name}_pos" if f"{strategy_name}_pos" in df_pd.columns else strategy_name
        position_signals = df_pd[position_col].values
        
        # Add close price
        fig.add_trace(
            go.Scatter(
                x=df_pd["datetime"],
                y=df_pd["close_price"],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=price_row, col=1
        )
        
        # Add stop loss lines ONLY when positions exist
        if show_stop_lines:
            # Long stop prices - only when in long position
            long_stop_col = f"{strategy_name}_long_stop"
            if long_stop_col in df_pd.columns:
                long_mask = (df_pd[position_col] == 1) & (~df_pd[long_stop_col].isna())
                if long_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df_pd.loc[long_mask, "datetime"],
                            y=df_pd.loc[long_mask, long_stop_col],
                            mode='lines',
                            name='Long Stop',
                            line=dict(color='green', width=2, dash='dash'),
                            opacity=0.8
                        ),
                        row=price_row, col=1
                    )
            
            # Short stop prices - only when in short position
            short_stop_col = f"{strategy_name}_short_stop"
            if short_stop_col in df_pd.columns:
                short_mask = (df_pd[position_col] == -1) & (~df_pd[short_stop_col].isna())
                if short_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df_pd.loc[short_mask, "datetime"],
                            y=df_pd.loc[short_mask, short_stop_col],
                            mode='lines',
                            name='Short Stop',
                            line=dict(color='red', width=2, dash='dash'),
                            opacity=0.8
                        ),
                        row=price_row, col=1
                    )
        
        # Add additional columns if specified
        if additional_columns:
            colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, col in enumerate(additional_columns):
                if col in df_pd.columns:
                    color = colors[i % len(colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=df_pd["datetime"],
                            y=df_pd[col],
                            mode='lines',
                            name=col,
                            line=dict(color=color, width=1.5)
                        ),
                        row=price_row, col=1
                    )
                else:
                    print(f"Warning: Column '{col}' not found in dataframe")
        
        # Add position rectangles to both subplots
        dates = df_pd["datetime"]
        current_position = 0
        start_date_pos = None

        for i, position in enumerate(position_signals):
            if position != current_position:
                # End previous position area
                if current_position != 0 and start_date_pos is not None:
                    color = 'rgba(0, 255, 0, 0.2)' if current_position == 1 else 'rgba(255, 0, 0, 0.2)'
                    # Add to price chart
                    fig.add_vrect(
                        x0=start_date_pos,
                        x1=dates.iloc[i-1] if i > 0 else dates.iloc[i],
                        fillcolor=color,
                        opacity=1,
                        layer="below",
                        line_width=0,
                        row=price_row, col=1
                    )
                    # Add to equity chart if it exists
                    if equity_row is not None:
                        fig.add_vrect(
                            x0=start_date_pos,
                            x1=dates.iloc[i-1] if i > 0 else dates.iloc[i],
                            fillcolor=color,
                            opacity=1,
                            layer="below",
                            line_width=0,
                            row=equity_row, col=1
                        )
                
                # Start new position area
                current_position = position
                start_date_pos = dates.iloc[i] if position != 0 else None

        # Handle the last position
        if current_position != 0 and start_date_pos is not None:
            color = 'rgba(0, 255, 0, 0.2)' if current_position == 1 else 'rgba(255, 0, 0, 0.2)'
            fig.add_vrect(
                x0=start_date_pos,
                x1=dates.iloc[-1],
                fillcolor=color,
                opacity=1,
                layer="below",
                line_width=0,
                row=price_row, col=1
            )
            if equity_row is not None:
                fig.add_vrect(
                    x0=start_date_pos,
                    x1=dates.iloc[-1],
                    fillcolor=color,
                    opacity=1,
                    layer="below",
                    line_width=0,
                    row=equity_row, col=1
                )
        
        # Add signal markers at position starts
        pos_start_long = []
        pos_start_short = []
        
        prev_signal = 0
        for i, signal in enumerate(position_signals):
            if signal != prev_signal and signal != 0:
                if signal == 1:
                    pos_start_long.append((dates.iloc[i], df_pd["close_price"].iloc[i]))
                elif signal == -1:
                    pos_start_short.append((dates.iloc[i], df_pd["close_price"].iloc[i]))
            prev_signal = signal
        
        # Add buy signal markers
        if pos_start_long:
            long_dates, long_prices = zip(*pos_start_long)
            fig.add_trace(
                go.Scatter(
                    x=long_dates,
                    y=long_prices,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        color='green',
                        size=12,
                        line=dict(color='darkgreen', width=2)
                    )
                ),
                row=price_row, col=1
            )
        
        # Add sell signal markers
        if pos_start_short:
            short_dates, short_prices = zip(*pos_start_short)
            fig.add_trace(
                go.Scatter(
                    x=short_dates,
                    y=short_prices,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        color='red',
                        size=12,
                        line=dict(color='darkred', width=2)
                    )
                ),
                row=price_row, col=1
            )
        
        # Add custom legend entries for position areas
        if any(p != 0 for p in position_signals):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color='rgba(0, 255, 0, 0.3)'),
                    showlegend=True,
                    name='Long Position'
                ),
                row=price_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers', 
                    marker=dict(size=10, color='rgba(255, 0, 0, 0.3)'),
                    showlegend=True,
                    name='Short Position'
                ),
                row=price_row, col=1
            )
        
        # Add equity curve in separate subplot if requested
        if show_equity_curve and equity_row is not None:
            # Calculate forward returns and strategy returns
            cur_bar_ret = df.select(
                pl.col("close_price").pct_change().alias("cur_bar_ret")
            )["cur_bar_ret"].to_numpy()
            
            positions = df[position_col].to_numpy()
            
            # Calculate strategy returns
            strategy_returns = cur_bar_ret * positions
            # Replace NaN with 0
            strategy_returns = np.nan_to_num(strategy_returns, 0)
            
            # Calculate cumulative returns (equity curve)
            equity_curve = np.cumprod(1 + strategy_returns) - 1
            
            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=df_pd["datetime"],
                    y=equity_curve * 100,  # Convert to percentage
                    mode='lines',
                    name='Strategy Return (%)',
                    line=dict(color='purple', width=2),
                    showlegend=True
                ),
                row=equity_row, col=1
            )
            
            # Add horizontal line at 0%
            fig.add_hline(
                y=0, 
                line_dash="dot", 
                line_color="gray", 
                opacity=0.5,
                row=equity_row, col=1
            )
        
        # ✅ CRITICAL: Set x-axis range to ensure perfect alignment
        if len(df_pd) > 0:
            x_min = df_pd["datetime"].min()
            x_max = df_pd["datetime"].max()
            
            # Apply same range to both subplots
            fig.update_xaxes(range=[x_min, x_max], row=price_row, col=1)
            if equity_row is not None:
                fig.update_xaxes(range=[x_min, x_max], row=equity_row, col=1)
        
        # ✅ IMPORTANT: Set axes titles - only x-title on bottom subplot
        fig.update_yaxes(title_text="Price", row=price_row, col=1)
        
        if show_equity_curve and equity_row is not None:
            # Add "Time" to top subplot but with smaller font and more spacing
            fig.update_xaxes(
                title_text="Time", 
                title_font_size=10,  # Smaller font for top subplot
                title_standoff=15,   # More space between axis and title
                row=price_row, col=1
            )
            # Main "Time" title on bottom subplot
            fig.update_xaxes(title_text="Time", row=equity_row, col=1)
            fig.update_yaxes(title_text="Return (%)", row=equity_row, col=1)
        else:
            fig.update_xaxes(title_text="Time", row=price_row, col=1)
        
        # Update layout
        title_text = f'{strategy_name} Strategy Signals'
        if show_equity_curve:
            title_text += ' with Equity Curve'
        
        fig.update_layout(
            title=title_text,
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            hovermode='x unified',
            showlegend=True,
            # ✅ LEGEND POSITIONED COMPLETELY OUTSIDE PLOT AREA
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left", 
                x=1.05,  # ✅ Further right to be completely separate
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)  # Smaller font to save space
            ),
            # ✅ INCREASED RIGHT MARGIN FOR LEGEND SPACE
            margin=dict(
                l=60,
                r=200,  # ✅ More space for legend
                t=80,
                b=60
            )
        )
        
        # ✅ ENSURE MATCHING GRID SETTINGS FOR PERFECT ALIGNMENT
        grid_settings = dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        fig.update_xaxes(**grid_settings)
        fig.update_yaxes(**grid_settings)
        
        # ✅ REMOVE X-AXIS TICKS FROM TOP SUBPLOT TO AVOID CLUTTER
        if show_equity_curve and equity_row is not None:
            fig.update_xaxes(showticklabels=False, row=price_row, col=1)
            fig.update_xaxes(showticklabels=True, row=equity_row, col=1)
        
        fig.show()
        
        # Print summary statistics
        total_signals = (df_pd[position_col] != 0).sum()
        buy_signals = (df_pd[position_col] == 1).sum()
        sell_signals = (df_pd[position_col] == -1).sum()

        print(f"\n📊 Strategy Signal Summary for {strategy_name}:")
        print(f"Total signals: {total_signals}")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"Time in the market: {total_signals/len(df_pd)*100:.2f}%")
        
        if show_equity_curve:
            # Calculate equity curve for stats
            cur_bar_ret = df.select(
                pl.col("close_price").pct_change().alias("cur_bar_ret")
            )["cur_bar_ret"].to_numpy()
            positions = df[position_col].to_numpy()
            strategy_returns = cur_bar_ret * positions
            strategy_returns = np.nan_to_num(strategy_returns, 0)
            equity_curve = np.cumprod(1 + strategy_returns) - 1
            
            final_return = equity_curve[-1]
            max_return = equity_curve.max()
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = ((equity_curve - peak) / peak) if peak != 0 else 0
            max_drawdown = drawdown.min()
            
            print(f"Final Return: {final_return:.2%}")
            print(f"Max Return: {max_return:.2%}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
            # Calculate Sharpe ratio(assuming 365 days for crypto)
            if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
                annual_factor = PerformanceCalculator._calculate_annual_factor(self.interval)
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(annual_factor)
                print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # def plot_strategy_signals(self, 
    #                     strategy_name: str,
    #                     additional_columns: list = None,
    #                     start_date: dt.datetime = None,
    #                     end_date: dt.datetime = None,
    #                     figsize: tuple = (15, 8)) -> None:
    #     """
    #     Plot strategy signals with price and optional additional columns using Plotly.
        
    #     Args:
    #         strategy_name (str): Name of the strategy signal column to plot
    #         additional_columns (list): List of additional column names to plot (optional)
    #         start_date (str): Start date for filtering data (format: 'YYYY-MM-DD')
    #         end_date (str): End date for filtering data (format: 'YYYY-MM-DD')
    #         figsize (tuple): Figure size (width, height)
    #     """
    #     try:
    #         import plotly.graph_objects as go
    #     except ImportError:
    #         raise ImportError("plotly is required for plotting. Install with: pip install plotly")
        
    #     # Ensure data is collected
    #     self._ensure_collected()
    #     df = self._df.clone()

    #     # Filter data by date range if provided
    #     if start_date:
    #         df = df.filter(pl.col("datetime") >= start_date)
    #     if end_date:
    #         df = df.filter(pl.col("datetime") <= end_date)
        
    #     # Check if required columns exist
    #     required_cols = ["datetime", "close_price", strategy_name]
    #     missing_cols = [col for col in required_cols if col not in df.columns]
    #     if missing_cols:
    #         raise ValueError(f"Missing required columns: {missing_cols}")
        
    #     # Convert Polars DataFrame to Pandas for Plotly compatibility
    #     df = df.to_pandas()
    #     # Create figure
    #     fig = go.Figure()
        
    #     # Add close price
    #     fig.add_trace(go.Scatter(
    #         x=df["datetime"],
    #         y=df["close_price"],
    #         mode='lines',
    #         name='Close Price',
    #         line=dict(color='blue', width=2)
    #     ))
        
    #     # Add additional columns if specified
    #     if additional_columns:
    #         colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    #         for i, col in enumerate(additional_columns):
    #             if col in df.columns:
    #                 color = colors[i % len(colors)]
    #                 fig.add_trace(go.Scatter(
    #                     x=df["datetime"],
    #                     y=df[col],
    #                     mode='lines',
    #                     name=col,
    #                     line=dict(color=color, width=1.5)
    #                 ))
    #             else:
    #                 print(f"Warning: Column '{col}' not found in dataframe")
        
    #     # Add position rectangles using fig.add_vrect
    #     position_signals = df[ f"{strategy_name}_pos"].values
    #     dates = df["datetime"]
        
    #     current_position = 0
    #     start_date_pos = None

    #     for i, position in enumerate(position_signals):
    #         if position != current_position:
    #             # End previous position area
    #             if current_position != 0 and start_date_pos is not None:
    #                 color = 'rgba(0, 255, 0, 0.2)' if current_position == 1 else 'rgba(255, 0, 0, 0.2)'
    #                 fig.add_vrect(
    #                     x0 = start_date_pos,
    #                     x1 = dates.iloc[i-1] if i > 0 else dates.iloc[i],
    #                     fillcolor = color,
    #                     opacity = 1,
    #                     layer = "below",
    #                     line_width = 0,
    #                 )
                
    #             # Start new position area
    #             current_position = position
    #             start_date_pos = dates.iloc[i] if position != 0 else None

    #     # Handle the last position
    #     if current_position != 0 and start_date_pos is not None:
    #         color = 'rgba(0, 255, 0, 0.2)' if current_position == 1 else 'rgba(255, 0, 0, 0.2)'
    #         fig.add_vrect(
    #             x0=start_date_pos,
    #             x1=dates.iloc[-1],
    #             fillcolor=color,
    #             opacity = 1,
    #             layer="below",
    #             line_width=0,
    #         )
        
    #     # Add pos markers at the start of each signal sequence
    #     strategy_pos = df[f"{strategy_name}_pos"].values
    #     pos_start_long = []
    #     pos_start_short = []
        
    #     prev_signal = 0
    #     for i, signal in enumerate(strategy_pos):
    #         if signal != prev_signal and signal != 0:
    #             if signal == 1:
    #                 pos_start_long.append((dates[i], df["close_price"].iloc[i]))
    #             elif signal == -1:
    #                 pos_start_short.append((dates[i], df["close_price"].iloc[i]))
    #         prev_signal = signal
        
    #     # Add buy signal markers
    #     if pos_start_long:
    #         long_dates, long_prices = zip(*pos_start_long)
    #         fig.add_trace(go.Scatter(
    #             x=long_dates,
    #             y=long_prices,
    #             mode='markers',
    #             name='Buy Signal',
    #             marker=dict(
    #                 symbol='triangle-up',
    #                 color='green',
    #                 size=12,
    #                 line=dict(color='darkgreen', width=2)
    #             )
    #         ))
        
    #     # Add sell signal markers
    #     if pos_start_short:
    #         short_dates, short_prices = zip(*pos_start_short)
    #         fig.add_trace(go.Scatter(
    #             x=short_dates,
    #             y=short_prices,
    #             mode='markers',
    #             name='Sell Signal',
    #             marker=dict(
    #                 symbol='triangle-down',
    #                 color='red',
    #                 size=12,
    #                 line=dict(color='darkred', width=2)
    #             )
    #         ))
        
    #     # Add custom legend entries for the shaded areas (only if there are positions)
    #     if any(p != 0 for p in position_signals):
    #         fig.add_trace(go.Scatter(
    #             x=[None], y=[None],
    #             mode='markers',
    #             marker=dict(size=10, color='rgba(0, 255, 0, 0.3)'),
    #             showlegend=True,
    #             name='Long Position'
    #         ))
    #         fig.add_trace(go.Scatter(
    #             x=[None], y=[None],
    #             mode='markers', 
    #             marker=dict(size=10, color='rgba(255, 0, 0, 0.3)'),
    #             showlegend=True,
    #             name='Short Position'
    #         ))
        
    #     # Update layout
    #     fig.update_layout(
    #         title=f'{strategy_name} Strategy Signals',
    #         xaxis_title='Time',
    #         yaxis_title='Price',
    #         width=figsize[0] * 80,
    #         height=figsize[1] * 80,
    #         hovermode='x unified',
    #         showlegend=True,
    #         legend=dict(
    #             yanchor="top",
    #             y=0.99,
    #             xanchor="left",
    #             x=0.01
    #         )
    #     )
        
    #     # Update axes
    #     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    #     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
    #     fig.show()
     
    #     # Print summary statistics
    #     total_signals = (df[f"{strategy_name}_pos"] != 0).sum()
    #     buy_signals = (df[f"{strategy_name}_pos"] == 1).sum()
    #     sell_signals = (df[f"{strategy_name}_pos"] == -1).sum()

    #     print(f"\n📊 Strategy Signal Summary for {strategy_name}:")
    #     print(f"Total signals: {total_signals}")
    #     print(f"Buy signals: {buy_signals}")
    #     print(f"Sell signals: {sell_signals}")
    #     print(f"Time in the market: {total_signals/len(df)*100:.2f}%")

    # ########################