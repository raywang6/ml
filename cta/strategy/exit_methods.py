"""
Exit Methods for Trading Strategies
Author: Haoqing Wu
Date: 2025-08-14
"""

import numpy as np
import polars as pl
from numba import njit

@njit
def calculate_exits_numba(open_arr, high_arr, low_arr, close_arr, vwap_arr, signal_arr,
                         long_sl_factor, short_sl_factor, liqka_decay_speed, liqka_floor,
                         k_stop_eff_ratio_floor, k_stop_adapt_ratio, period,
                         k_stop_accelerator, k_stop_eff_ratio_trigger,
                         p_stop_price_trigger, p_stop_retain_ratio,
                         r_stop_eff_ratio, cooldown_period, max_holding_time_allowed,
                         # Stop-loss type flags
                         enable_time_adaptive, 
                         enable_price_adaptive, 
                         enable_profit_protective, 
                         enable_reverse_converge,
                         enable_max_holding_time):
    """
    Numba-optimized exit calculation with configurable stop-loss types
    
    Args:
        OHLC, V arrays and signal array
        Parameter values for different stop-loss methods
        
        Stop-loss enable flags:
        - enable_time_adaptive: Time-based trailing stop with liqka decay
        - enable_price_adaptive: Price-based trailing stop with efficiency ratio
        - enable_profit_protective: Profit protection stop when above threshold
        - enable_reverse_converge: Reverse movement triggered stop tightening
        - enable_max_holding_time: Maximum holding time exceeded
    Returns:
        position, long_entry_price, short_entry_price, long_stop, short_stop
    """
    n = close_arr.shape[0]
    
    # Initialize status vectors
    position = np.zeros(n)
    long_entry_price = np.full(n, np.nan)
    short_entry_price = np.full(n, np.nan)
    holding_time = np.zeros(n)
    cooldown = 0
    to_open = False
    to_exit = False
    last_pos = 0

    # Compute efficiency ratio if needed for price adaptive stops
    eff_ratio = np.full(n, np.nan)
    if enable_price_adaptive:
        direction = np.abs(close_arr[period:] - close_arr[:-period])
        for i in range(period, n):
            volatility = np.sum(np.abs(close_arr[i - period + 1:i + 1] - close_arr[i - period:i]))
            if volatility > 0:
                eff_ratio[i] = direction[i - period] / volatility

    # Initialize vectors
    higher_after_entry = np.full(n, np.nan)
    lower_after_entry = np.full(n, np.nan)
    liqka = np.zeros(n)

    # Stop-price vectors
    long_t_stop_price = np.full(n, np.nan)  # Time adaptive
    short_t_stop_price = np.full(n, np.nan)
    long_k_stop_price = np.full(n, np.nan)  # Price adaptive
    short_k_stop_price = np.full(n, np.nan)
    long_p_stop_price = np.full(n, np.nan)  # Profit protective
    short_p_stop_price = np.full(n, np.nan)
    long_r_stop_price = np.full(n, np.nan)  # Reverse converge
    short_r_stop_price = np.full(n, np.nan)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)

    if signal_arr[0] != 0:
        to_open = True

    for i in range(1, n):
        if to_open:
            if position[i - 1] != 0:
                continue  # Skip instead of raising error in numba
                
            if signal_arr[i - 1] > 0:
                position[i] = 1
                long_entry_price[i] = vwap_arr[i]
                holding_time[i] = 1
            elif signal_arr[i - 1] < 0:
                position[i] = -1
                short_entry_price[i] = vwap_arr[i]
                holding_time[i] = 1
            to_open = False

        elif to_exit:
            if position[i - 1] != 0:
                position[i] = 0
                holding_time[i] = 0
                last_pos = position[i - 1]
                cooldown = 1
                to_exit = False

        else:  # No signals or exits at the prev bar
            position[i] = position[i - 1]
            if position[i] != 0:
                holding_time[i] = holding_time[i - 1] + 1

        # Compute stop-loss price levels based on enabled types
        # Long position exits
        if position[i] > 0:
            if np.isnan(long_entry_price[i]):
                long_entry_price[i] = long_entry_price[i - 1]

            stops_to_combine = []

            # 1. Time adaptive trailing stop
            if enable_time_adaptive:
                if np.isnan(lower_after_entry[i - 1]):
                    lower_after_entry[i] = max(long_entry_price[i], close_arr[i])
                else:
                    lower_after_entry[i] = max(lower_after_entry[i-1], close_arr[i])

                liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
                liqka[i] = min(liqka[i], liqka_floor)
                long_t_stop_price[i] = lower_after_entry[i] - (1 - liqka[i]) * close_arr[i] * long_sl_factor
                stops_to_combine.append(long_t_stop_price[i])

            # 2. Price adaptive trailing stop
            if enable_price_adaptive:
                if np.isnan(long_k_stop_price[i - 1]):
                    long_k_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
                else:
                    long_k_stop_price[i] = long_k_stop_price[i - 1]

                if close_arr[i] > open_arr[i]:
                    eff_val = eff_ratio[i] if not np.isnan(eff_ratio[i]) else k_stop_eff_ratio_floor
                    offset = (close_arr[i] - open_arr[i]) * max(eff_val, k_stop_eff_ratio_floor) * k_stop_adapt_ratio
                    long_k_stop_price[i] += offset

                    if k_stop_accelerator and (not np.isnan(eff_ratio[i])) and eff_ratio[i] > k_stop_eff_ratio_trigger:
                        long_k_stop_price[i] = long_k_stop_price[i] + (close_arr[i] - long_k_stop_price[i]) * eff_ratio[i]

                stops_to_combine.append(long_k_stop_price[i])

            # 3. Profit protective stop
            if enable_profit_protective:
                long_change = high_arr[i] / long_entry_price[i] - 1
                if long_change > p_stop_price_trigger:
                    cur_p_stop_price = (1 + long_change * p_stop_retain_ratio) * long_entry_price[i]
                    if np.isnan(long_p_stop_price[i - 1]):
                        long_p_stop_price[i] = cur_p_stop_price
                    else:
                        long_p_stop_price[i] = max(long_p_stop_price[i - 1], cur_p_stop_price)
                    stops_to_combine.append(long_p_stop_price[i])

            # 4. Reverse converge stop
            if enable_reverse_converge:
                if np.isnan(long_r_stop_price[i - 1]):
                    long_r_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
                else:
                    long_r_stop_price[i] = long_r_stop_price[i - 1]

                if close_arr[i] < open_arr[i]:
                    long_r_stop_price[i] = long_r_stop_price[i] + (open_arr[i] - close_arr[i]) * r_stop_eff_ratio

                stops_to_combine.append(long_r_stop_price[i])

            # Combine enabled stops
            if len(stops_to_combine) > 0:
                valid_stops = [s for s in stops_to_combine if not np.isnan(s)]
                if len(valid_stops) > 0:
                    long_stop[i] = max(valid_stops)

            # Trigger long exit
            if (not np.isnan(long_stop[i]) and close_arr[i] <= long_stop[i]): 
                to_exit = True

            # Holding Time constraint
            if enable_max_holding_time:
                if holding_time[i] > max_holding_time_allowed:
                    to_exit = True

        # Short position exits
        elif position[i] < 0:
            if np.isnan(short_entry_price[i]):
                short_entry_price[i] = short_entry_price[i - 1]

            stops_to_combine = []

            # 1. Time adaptive trailing stop
            if enable_time_adaptive:
                if np.isnan(higher_after_entry[i - 1]):
                    higher_after_entry[i] = short_entry_price[i]
                higher_after_entry[i] = min(higher_after_entry[i-1], close_arr[i])

                liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
                liqka[i] = min(liqka[i], liqka_floor)
                short_t_stop_price[i] = higher_after_entry[i] + (1 - liqka[i]) * close_arr[i] * short_sl_factor
                stops_to_combine.append(short_t_stop_price[i])

            # 2. Price adaptive trailing stop
            if enable_price_adaptive:
                if np.isnan(short_k_stop_price[i - 1]):
                    short_k_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
                else:
                    short_k_stop_price[i] = short_k_stop_price[i - 1]
                    
                if close_arr[i] < open_arr[i]:
                    eff_val = eff_ratio[i] if not np.isnan(eff_ratio[i]) else k_stop_eff_ratio_floor
                    offset = (close_arr[i] - open_arr[i]) * max(eff_val, k_stop_eff_ratio_floor) * k_stop_adapt_ratio
                    short_k_stop_price[i] += offset

                    if k_stop_accelerator and not np.isnan(eff_ratio[i]) and eff_ratio[i] > k_stop_eff_ratio_trigger:
                        short_k_stop_price[i] = short_k_stop_price[i] - (close_arr[i] - short_k_stop_price[i]) * eff_ratio[i]

                stops_to_combine.append(short_k_stop_price[i])

            # 3. Profit protective stop
            if enable_profit_protective:
                short_change = low_arr[i] / short_entry_price[i] - 1
                if short_change < -p_stop_price_trigger:
                    cur_p_stop_price = (1 + short_change * p_stop_retain_ratio) * short_entry_price[i]
                    if np.isnan(short_p_stop_price[i - 1]):
                        short_p_stop_price[i] = cur_p_stop_price
                    else:
                        short_p_stop_price[i] = min(short_p_stop_price[i - 1], cur_p_stop_price)
                    stops_to_combine.append(short_p_stop_price[i])

            # 4. Reverse converge stop
            if enable_reverse_converge:
                if np.isnan(short_r_stop_price[i - 1]):
                    short_r_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
                else:
                    short_r_stop_price[i] = short_r_stop_price[i - 1]

                if close_arr[i] > open_arr[i]:
                    short_r_stop_price[i] = short_r_stop_price[i] + (close_arr[i] - open_arr[i]) * r_stop_eff_ratio

                stops_to_combine.append(short_r_stop_price[i])

            # Combine enabled stops
            if len(stops_to_combine) > 0:
                valid_stops = [s for s in stops_to_combine if not np.isnan(s)]
                if len(valid_stops) > 0:
                    short_stop[i] = min(valid_stops)

            # Trigger short exit
            if not np.isnan(short_stop[i]) and close_arr[i] >= short_stop[i]:
                to_exit = True

            # Holding Time constraint
            if enable_max_holding_time:
                if holding_time[i] > max_holding_time_allowed:
                    to_exit = True


        # No position - check for new signals
        else:
            if signal_arr[i] != 0:
                # Check cooldown period
                if (signal_arr[i] == last_pos) and (cooldown < cooldown_period):
                    cooldown += 1
                    continue
                to_open = True
                cooldown = 0
            else:
                cooldown += 1

    return position, long_entry_price, short_entry_price, long_stop, short_stop


def apply_exit_methods(df: pl.DataFrame, signal_col: str, exit_params: dict) -> pl.DataFrame:
    """
    Apply exit methods to a signal column in a polars DataFrame
    
    Args:
        df: Polars DataFrame with OHLCV data
        signal_col: Name of the signal column to apply exits to
        exit_params: Dictionary with exit parameters and flags
        
    Returns:
        DataFrame with added exit-related columns
    """
    # Extract OHLCV arrays
    open_arr = np.ascontiguousarray(df.select(pl.col("open_price")).to_numpy().flatten(), dtype=np.float64)
    high_arr = np.ascontiguousarray(df.select(pl.col("high_price")).to_numpy().flatten(), dtype=np.float64)
    low_arr = np.ascontiguousarray(df.select(pl.col("low_price")).to_numpy().flatten(), dtype=np.float64)
    close_arr = np.ascontiguousarray(df.select(pl.col("close_price")).to_numpy().flatten(), dtype=np.float64)
    vwap_arr = np.ascontiguousarray(df.select(pl.col("vwap")).to_numpy().flatten(), dtype=np.float64)
    signal_arr = np.ascontiguousarray(df.select(pl.col(signal_col).fill_null(0)).to_numpy().flatten(), dtype=np.float64)
    
    # ✅ EXPLICIT PARAMETER TYPE CONVERSION
    long_sl_factor = np.float64(exit_params.get('long_sl_factor', 0.02))
    short_sl_factor = np.float64(exit_params.get('short_sl_factor', 0.02))
    liqka_decay_speed = np.float64(exit_params.get('liqka_decay_speed', 0.2))
    liqka_floor = np.float64(exit_params.get('liqka_floor', 0.01))
    k_stop_eff_ratio_floor = np.float64(exit_params.get('k_stop_eff_ratio_floor', 0.1))
    k_stop_adapt_ratio = np.float64(exit_params.get('k_stop_adapt_ratio', 0.5))
    period = np.int64(exit_params.get("ER_period", 32))
    
    # ✅ BOOLEAN CONVERSION
    k_stop_accelerator_param = exit_params.get("k_stop_accelerator", True)
    k_stop_accelerator = True if k_stop_accelerator_param else False
    
    k_stop_eff_ratio_trigger = np.float64(exit_params.get("k_stop_eff_ratio_trigger", 0.5))
    p_stop_price_trigger = np.float64(exit_params.get('p_stop_price_trigger', 0.5))
    p_stop_retain_ratio = np.float64(exit_params.get('p_stop_retain_ratio', 0.4))
    r_stop_eff_ratio = np.float64(exit_params.get('r_stop_eff_ratio', 0.5))
    cooldown_period = np.int64(exit_params.get("cooldown_period", 32))
    max_holding_time_allowed = np.int64(exit_params.get("max_holding_time_allowed", 64))
    
    # ✅ BOOLEAN FLAGS
    enable_time_adaptive = True if exit_params.get('enable_time_adaptive', True) else False
    enable_price_adaptive = True if exit_params.get('enable_price_adaptive', True) else False
    enable_profit_protective = True if exit_params.get('enable_profit_protective', True) else False
    enable_reverse_converge = True if exit_params.get('enable_reverse_converge', True) else False
    enable_max_holding_time = True if exit_params.get('enable_max_holding_time', True) else False

    # Call numba function
    position, long_entry_price, short_entry_price, long_stop, short_stop = calculate_exits_numba(
        open_arr, high_arr, low_arr, close_arr, vwap_arr, signal_arr,
        long_sl_factor, short_sl_factor, liqka_decay_speed, liqka_floor,
        k_stop_eff_ratio_floor, k_stop_adapt_ratio, period,
        k_stop_accelerator, k_stop_eff_ratio_trigger,
        p_stop_price_trigger, p_stop_retain_ratio,
        r_stop_eff_ratio, cooldown_period, max_holding_time_allowed,
        enable_time_adaptive, enable_price_adaptive, 
        enable_profit_protective, enable_reverse_converge,
        enable_max_holding_time
    )
    
    # Add results to DataFrame
    result_df = df.with_columns([
        pl.Series(f"{signal_col}_pos", position),
        pl.Series(f"{signal_col}_long_entry_price", long_entry_price),
        pl.Series(f"{signal_col}_short_entry_price", short_entry_price),
        pl.Series(f"{signal_col}_long_stop", long_stop),
        pl.Series(f"{signal_col}_short_stop", short_stop)
    ])
    
    return result_df


####################
# Legacy code below
# @njit
# def _calculate_pos_w_exits_numba(open_arr, high_arr, low_arr, close_arr, vwap_arr, signal_arr,
#                                  long_sl_factor, short_sl_factor, liqka_decay_speed, liqka_floor,
#                                  k_stop_eff_ratio_floor, k_stop_adapt_ratio, period,
#                                  k_stop_accelerator, k_stop_eff_ratio_trigger,
#                                  p_stop_price_trigger, p_stop_retain_ratio,
#                                  r_stop_eff_ratio, cooldown_period):
#     """
#     Numba-optimized version of position calculation with exits
#     All parameters are passed individually since numba doesn't support dicts
#     """
#     n = close_arr.shape[0]
    
#     # Initialize status vectors
#     position = np.zeros(n)
#     long_entry_price = np.full(n, np.nan)
#     short_entry_price = np.full(n, np.nan)
#     holding_time = np.zeros(n)
#     cooldown = 0
#     to_open = False
#     to_exit = False
#     last_pos = 0

#     # Compute ER (efficiency ratio)
#     eff_ratio = np.full(n, np.nan)
#     direction = np.abs(close_arr[period:] - close_arr[:-period])
    
#     for i in range(period, n):
#         volatility = np.sum(np.abs(close_arr[i - period + 1:i + 1] - close_arr[i - period:i]))
#         if volatility > 0:
#             eff_ratio[i] = direction[i - period] / volatility

#     # Initialize vectors
#     higher_after_entry = np.full(n, np.nan)
#     lower_after_entry = np.full(n, np.nan)
#     liqka = np.zeros(n)

#     # Stop-price vectors
#     long_t_stop_price = np.full(n, np.nan)
#     short_t_stop_price = np.full(n, np.nan)
#     long_k_stop_price = np.full(n, np.nan)
#     short_k_stop_price = np.full(n, np.nan)
#     long_p_stop_price = np.full(n, np.nan)
#     short_p_stop_price = np.full(n, np.nan)
#     long_r_stop_price = np.full(n, np.nan)
#     short_r_stop_price = np.full(n, np.nan)
#     long_stop = np.full(n, np.nan)
#     short_stop = np.full(n, np.nan)

#     if signal_arr[0] != 0:
#         to_open = True

#     for i in range(1, n):
#         if to_open:
#             if position[i - 1] != 0:
#                 continue  # Skip instead of raising error in numba
                
#             if signal_arr[i - 1] > 0:
#                 position[i] = 1
#                 long_entry_price[i] = vwap_arr[i]
#                 holding_time[i] = 1
#             elif signal_arr[i - 1] < 0:
#                 position[i] = -1
#                 short_entry_price[i] = vwap_arr[i]
#                 holding_time[i] = 1
#             to_open = False

#         elif to_exit:
#             if position[i - 1] != 0:
#                 position[i] = 0
#                 holding_time[i] = 0
#                 last_pos = position[i - 1]
#                 cooldown = 1
#                 to_exit = False

#         else:  # No signals or exits at the prev bar
#             position[i] = position[i - 1]
#             if position[i] != 0:
#                 holding_time[i] = holding_time[i - 1] + 1

#         # Compute sl/tp price levels
#         # Long Exits
#         if position[i] > 0:
#             if np.isnan(long_entry_price[i]):
#                 long_entry_price[i] = long_entry_price[i - 1]

#             ## 1. time adaptive trailing sl
#             if np.isnan(lower_after_entry[i - 1]):
#                 lower_after_entry[i] = max(long_entry_price[i], close_arr[i])
#             else:
#                 lower_after_entry[i] = max(lower_after_entry[i-1], close_arr[i])

#             liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
#             liqka[i] = min(liqka[i], liqka_floor)
#             long_t_stop_price[i] = lower_after_entry[i] - (1 - liqka[i]) * close_arr[i] * long_sl_factor

#             ## 2. price adaptive trailing sl
#             if np.isnan(long_k_stop_price[i - 1]):
#                 long_k_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
#             else:
#                 long_k_stop_price[i] = long_k_stop_price[i - 1]

#             if close_arr[i] > open_arr[i]:
#                 eff_val = eff_ratio[i] if not np.isnan(eff_ratio[i]) else k_stop_eff_ratio_floor
#                 offset = (close_arr[i] - open_arr[i]) * max(eff_val, k_stop_eff_ratio_floor) * k_stop_adapt_ratio
#                 long_k_stop_price[i] += offset

#                 if k_stop_accelerator and not np.isnan(eff_ratio[i]) and eff_ratio[i] > k_stop_eff_ratio_trigger:
#                     long_k_stop_price[i] = long_k_stop_price[i] + (close_arr[i] - long_k_stop_price[i]) * eff_ratio[i]

#             ## 3. profit protective stop loss
#             long_change = high_arr[i] / long_entry_price[i] - 1
#             if long_change > p_stop_price_trigger:
#                 cur_p_stop_price = (1 + long_change * p_stop_retain_ratio) * long_entry_price[i]
#                 if np.isnan(long_p_stop_price[i - 1]):
#                     long_p_stop_price[i] = cur_p_stop_price
#                 else:
#                     long_p_stop_price[i] = max(long_p_stop_price[i - 1], cur_p_stop_price)

#             ## 4. reverse converge stop loss
#             if np.isnan(long_r_stop_price[i - 1]):
#                 long_r_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
#             else:
#                 long_r_stop_price[i] = long_r_stop_price[i - 1]

#             if close_arr[i] < open_arr[i]:
#                 long_r_stop_price[i] = long_r_stop_price[i] + (open_arr[i] - close_arr[i]) * r_stop_eff_ratio

#             # Combine all stop losses
#             stops = np.array([long_t_stop_price[i], long_k_stop_price[i], long_p_stop_price[i], long_r_stop_price[i]])
#             valid_stops = stops[~np.isnan(stops)]
#             if len(valid_stops) > 0:
#                 long_stop[i] = np.max(valid_stops)

#             # Trigger long exit
#             if not np.isnan(long_stop[i]) and close_arr[i] <= long_stop[i]:
#                 to_exit = True

#         # Short Exits
#         elif position[i] < 0:
#             if np.isnan(short_entry_price[i]):
#                 short_entry_price[i] = short_entry_price[i - 1]

#             ## 1. time adaptive trailing sl
#             if np.isnan(higher_after_entry[i - 1]):
#                 higher_after_entry[i] = short_entry_price[i]
#             higher_after_entry[i] = min(higher_after_entry[i-1], close_arr[i])

#             liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
#             liqka[i] = min(liqka[i], liqka_floor)
#             short_t_stop_price[i] = higher_after_entry[i] + (1 - liqka[i]) * close_arr[i] * short_sl_factor

#             ## 2. price adaptive trailing sl
#             if np.isnan(short_k_stop_price[i - 1]):
#                 short_k_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
#             else:
#                 short_k_stop_price[i] = short_k_stop_price[i - 1]
                
#             if close_arr[i] < open_arr[i]:
#                 eff_val = eff_ratio[i] if not np.isnan(eff_ratio[i]) else k_stop_eff_ratio_floor
#                 offset = (close_arr[i] - open_arr[i]) * max(eff_val, k_stop_eff_ratio_floor) * k_stop_adapt_ratio
#                 short_k_stop_price[i] += offset

#                 if k_stop_accelerator and not np.isnan(eff_ratio[i]) and eff_ratio[i] > k_stop_eff_ratio_trigger:
#                     short_k_stop_price[i] = short_k_stop_price[i] - (close_arr[i] - short_k_stop_price[i]) * eff_ratio[i]

#             ## 3. profit protective stop loss
#             short_change = low_arr[i] / short_entry_price[i] - 1
#             if short_change < -p_stop_price_trigger:
#                 cur_p_stop_price = (1 + short_change * p_stop_retain_ratio) * short_entry_price[i]
#                 if np.isnan(short_p_stop_price[i - 1]):
#                     short_p_stop_price[i] = cur_p_stop_price
#                 else:
#                     short_p_stop_price[i] = min(short_p_stop_price[i - 1], cur_p_stop_price)

#             ## 4. reverse converge stop loss
#             if np.isnan(short_r_stop_price[i - 1]):
#                 short_r_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
#             else:
#                 short_r_stop_price[i] = short_r_stop_price[i - 1]

#             if close_arr[i] > open_arr[i]:
#                 short_r_stop_price[i] = short_r_stop_price[i] + (close_arr[i] - open_arr[i]) * r_stop_eff_ratio

#             # Combine all stop losses
#             stops = np.array([short_t_stop_price[i], short_k_stop_price[i], short_p_stop_price[i], short_r_stop_price[i]])
#             valid_stops = stops[~np.isnan(stops)]
#             if len(valid_stops) > 0:
#                 short_stop[i] = np.min(valid_stops)

#             # Trigger short exit
#             if not np.isnan(short_stop[i]) and close_arr[i] >= short_stop[i]:
#                 to_exit = True

#         # No position
#         else:
#             if signal_arr[i] != 0:
#                 # if still in cooldown period
#                 if (signal_arr[i] == last_pos) and (cooldown < cooldown_period):
#                     cooldown += 1
#                     continue
#                 to_open = True  # open a pos at the next bar
#                 cooldown = 0  # cooldown over
#             else:
#                 cooldown += 1

#     return position, long_entry_price, short_entry_price, long_stop, short_stop


# class ExitMethods:
#     """
#     Exit methods for trading strategies including signal extension,
#     stop-losses, take-profits, and time-based exits.
#     """
    
#     def __init__(self, signal_generator, exit_params: dict[str, dict] = None):
#         """
#         Initialize with reference to SignalGenerator instance
#         It calculates the stop loss/take profit prices from the initial position,
#         then applies the specified exit signals.


#         Args:
#             signal_generator: SignalGenerator instance to apply exits to
#             exit_params: Dictionary of signals to apply exit methods to; 
#                 - keys: signal column names
#                 - values: parameters for exit methods
#         """
#         self.sg = signal_generator
#         self.exit_params = exit_params if exit_params else {}

#         if self.exit_params:
#             self._add_pos_w_exits()
#             self.sg._df = self._df  # Direct assignment
#             self.sg._lf = self._df.lazy()  # Update lazy frame too
#             self.sg._is_collected = True  # Mark as collected

#     @property
#     def columns(self):
#         return self.sg._df.columns
    
#     @property
#     def _df(self):
#         return self.sg._df
    
#     @_df.setter
#     def _df(self, value):
#         self.sg._df = value

#     def _calculate_pos_w_exits(self,                         
#                                 open: np.ndarray,
#                                 high: np.ndarray,
#                                 low: np.ndarray,
#                                 close: np.ndarray, 
#                                 vwap:np.ndarray,
#                                 signal: np.ndarray,
#                                 params:dict, 
#                                 *args,
#                                 **kwargs) -> dict[np.ndarray]:
        
#         """ compute position vector based on signals and exit criteria """
        
#         # Extract parameters for numba function (can't pass dict to numba)
#         long_sl_factor = params.get('long_sl_factor', 0.02)
#         short_sl_factor = params.get('short_sl_factor', 0.02)
#         liqka_decay_speed = params.get('liqka_decay_speed', 0.2)
#         liqka_floor = params.get('liqka_floor', 0.01)
#         k_stop_eff_ratio_floor = params.get('k_stop_eff_ratio_floor', 0.1)
#         k_stop_adapt_ratio = params.get('k_stop_adapt_ratio', 0.5)
#         period = params.get("ER_period", 32)
#         k_stop_accelerator = params.get("k_stop_accelerator", True)
#         k_stop_eff_ratio_trigger = params.get("k_stop_eff_ratio_trigger", 0.5)
#         p_stop_price_trigger = params.get('p_stop_price_trigger', 0.5)
#         p_stop_retain_ratio = params.get('p_stop_retain_ratio', 0.4)
#         r_stop_eff_ratio = params.get('r_stop_eff_ratio', 0.5)
#         cooldown_period = params.get("cooldown_period", 32)

#         # Ensure all arrays are proper numpy float64 types for numba
#         open_arr = np.asarray(open, dtype=np.float64)
#         high_arr = np.asarray(high, dtype=np.float64)
#         low_arr = np.asarray(low, dtype=np.float64)
#         close_arr = np.asarray(close, dtype=np.float64)
#         vwap_arr = np.asarray(vwap, dtype=np.float64)
#         signal_arr = np.asarray(signal, dtype=np.float64)


#         # Call the numba-optimized function
#         position, long_entry_price, short_entry_price, long_stop, short_stop = _calculate_pos_w_exits_numba(
#             open_arr, high_arr, low_arr, close_arr, vwap_arr, signal_arr,
#             float(long_sl_factor), float(short_sl_factor), float(liqka_decay_speed), float(liqka_floor),
#             float(k_stop_eff_ratio_floor), float(k_stop_adapt_ratio), int(period),
#             bool(k_stop_accelerator), float(k_stop_eff_ratio_trigger),
#             float(p_stop_price_trigger), float(p_stop_retain_ratio),
#             float(r_stop_eff_ratio), int(cooldown_period)
#         )
        
#         return {
#             "pos": position,
#             "long_entry_price": long_entry_price,
#             "short_entry_price": short_entry_price,
#             "long_stop": long_stop,
#             "short_stop": short_stop
#         }


#         # n = close.shape[0]

#         # # initialize status vectors
#         # position          = np.zeros(n)        # position vector (1 for long, -1 for short, 0 for no position)
#         # long_entry_price  = np.full(n, np.nan) # new long entry price after exits
#         # short_entry_price = np.full(n, np.nan) # new short entry price after exits
#         # holding_time      = np.zeros(n)        # holding time vector
#         # cooldown          = 0                  # cooldown counter after exits
#         # to_open           = False              # a flag to open a new position at the next bar
#         # to_exit           = False              # a flag to exit at the next bar
#         # last_pos          = 0                  # keep track of last direction


#         # # load parameters
#         # long_sl_factor = params.get('long_sl_factor', 0.02)
#         # short_sl_factor = params.get('short_sl_factor', 0.02)
#         # liqka_decay_speed = params.get('liqka_decay_speed', 0.2)
#         # liqka_floor = params.get('liqka_floor', 0.01)

#         # k_stop_eff_ratio_floor = params.get('k_stop_eff_ratio_floor', 0.1)
#         # k_stop_adapt_ratio = params.get('k_stop_adapt_ratio', 0.5)
#         # period = params.get("ER_period", 32)
#         # k_stop_accelerator = params.get("k_stop_accelerator", True)
#         # k_stop_eff_ratio_trigger = params.get("k_stop_eff_ratio_trigger", 0.5)

#         # p_stop_price_trigger = params.get('p_stop_price_trigger', 0.5)  # 50% profit trigger
#         # p_stop_retain_ratio  = params.get('p_stop_retain_ratio', 0.4)  # Retain 40% of profit

#         # r_stop_eff_ratio = params.get('r_stop_eff_ratio', 0.5)  # Reverse converge stop loss efficiency ratio
#         # cooldown_period = params.get("cooldown_period", 32)

#         # # Compute ER
#         # eff_ratio = np.full(n, np.nan)
#         # direction = np.abs(close[period: ] - close[ :-period ] ).flatten()
#         # volatility = np.array([
#         #             np.sum(np.abs(close[i - period + 1: i + 1] - close[i - period: i]))
#         #             for i in range(period, n)
#         # ])
#         # eff_ratio[period: ] = direction / volatility

#         # # initialize vectors
#         # higher_after_entry = np.full(n, np.nan)
#         # lower_after_entry  = np.full(n, np.nan)
#         # liqka = np.zeros(n)

#         # # Stop-price vectors
#         # long_t_stop_price  = np.full(n, np.nan)
#         # short_t_stop_price = np.full(n, np.nan) 
        
#         # long_k_stop_price  = np.full(n, np.nan)
#         # short_k_stop_price = np.full(n, np.nan)

#         # long_p_stop_price  = np.full(n, np.nan)
#         # short_p_stop_price = np.full(n, np.nan)

#         # long_r_stop_price  = np.full(n, np.nan)
#         # short_r_stop_price = np.full(n, np.nan)

#         # long_stop = np.full(n, np.nan)
#         # short_stop = np.full(n, np.nan)

#         # if signal[0] != 0:
#         #     to_open = True

#         # for i in range(1, n):

#         #     if to_open:
#         #         if position[i - 1] != 0:
#         #             raise ValueError("Cannot open a new position when there is an existing position.")
                
#         #         if signal[i - 1] > 0:
#         #             position[i] = 1
#         #             long_entry_price[i] = vwap[i]
#         #             holding_time[i] = 1
#         #         elif signal[i - 1] < 0:
#         #             position[i] = -1
#         #             short_entry_price[i] = vwap[i]
#         #             holding_time[i] = 1
#         #         else:
#         #             raise ValueError("Bug detected: the previous signal has to be nonzero to set the flag True")
#         #         to_open = False

#         #     elif to_exit:
#         #         if position[i - 1] != 0:
#         #             position[i] = 0
#         #             holding_time[i] = 0
#         #             last_pos = position[i - 1]
#         #             cooldown = 1
#         #             to_exit = False
#         #         else:
#         #             raise ValueError("Cannot exit position when there is no position to exit.")

#         #     else: # No signals or exits at the prev bar
#         #         position[i] = position[i - 1]
#         #         if position[i] != 0:
#         #             holding_time[i] = holding_time[i - 1] + 1

#         #     # Compute sl/tp price levels
#         #     # Long Exits
#         #     if position[i] > 0:
#         #         if np.isnan(long_entry_price[i]):
#         #             long_entry_price[i] = long_entry_price[i - 1]

#         #         ## 1. time adaptive trailing sl
#         #         if np.isnan(lower_after_entry[i - 1]):
#         #             lower_after_entry[i] = np.nanmax([long_entry_price[i], close[i]])
#         #         else:
#         #             lower_after_entry[i] = np.nanmax([lower_after_entry[i-1], close[i]])

#         #         liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
#         #         liqka[i] = np.nanmin([liqka[i], liqka_floor])
#         #         long_t_stop_price[i] = lower_after_entry[i] - (1 - liqka[i]) * close[i] * long_sl_factor

#         #         ## 2. price adaptive trailing sl
#         #         if np.isnan(long_k_stop_price[i - 1]):
#         #             long_k_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
#         #         else:
#         #             long_k_stop_price[i] = long_k_stop_price[i - 1]

#         #         if close[i] > open[i]:
#         #             offset = (close[i] - open[i]) * np.nanmax( [eff_ratio[i], k_stop_eff_ratio_floor]) * k_stop_adapt_ratio
#         #             long_k_stop_price[i] += offset

#         #             if k_stop_accelerator and eff_ratio[i] and eff_ratio[i] > k_stop_eff_ratio_trigger:
#         #                 long_k_stop_price[i] = long_k_stop_price[i] + (close[i] - long_k_stop_price[i]) * eff_ratio[i]

#         #         ## 3. profit protective stop loss
#         #         long_change = high[i] / long_entry_price[i] - 1
#         #         if long_change > p_stop_price_trigger:
#         #             cur_p_stop_price = ( 1 + long_change * p_stop_retain_ratio) * long_entry_price[i]
#         #             if np.isnan(long_p_stop_price[i - 1]):
#         #                 long_p_stop_price[i] = cur_p_stop_price
#         #             else:
#         #                 long_p_stop_price[i] = np.nanmax([long_p_stop_price[i - 1], cur_p_stop_price])

#         #         ## 4. reverse converge stop loss
#         #         if np.isnan(long_r_stop_price[i - 1]):
#         #             long_r_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
#         #         else:
#         #             long_r_stop_price[i] = long_r_stop_price[i - 1]

#         #         if close[i] < open[i]:
#         #             long_r_stop_price[i] = long_r_stop_price[i] + (open[i] - close[i]) * r_stop_eff_ratio

#         #         long_stop[i] = np.nanmax([long_t_stop_price[i], 
#         #                                   long_k_stop_price[i], 
#         #                                   long_p_stop_price[i], 
#         #                                   long_r_stop_price[i]]
#         #                         )

#         #         # 触发多单止损止盈
#         #         if close[i] <= long_stop[i]:
#         #             to_exit = True

#         #     # Short Exits
#         #     elif position[i] < 0:
#         #         # update short entry price
#         #         if np.isnan(short_entry_price[i]):
#         #             short_entry_price[i] = short_entry_price[i - 1]

#         #         ## 1. time adaptive trailing sl
#         #         if np.isnan(higher_after_entry[i - 1]):
#         #             higher_after_entry[i] = short_entry_price[i]
#         #         higher_after_entry[i] = np.nanmin([ higher_after_entry[i-1], close[i]])

#         #         liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
#         #         liqka[i] = np.nanmin([liqka[i], liqka_floor])
#         #         short_t_stop_price[i] = higher_after_entry[i] + (1 - liqka[i]) * close[i] * short_sl_factor

#         #         ## 2. price adaptive trailing sl
#         #         if np.isnan(short_k_stop_price[i - 1]):
#         #             short_k_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
#         #         else:
#         #             short_k_stop_price[i] = short_k_stop_price[i - 1]
#         #         if close[i] < open[i]:
#         #             offset = (close[i] - open[i]) * np.nanmax( [eff_ratio[i], k_stop_eff_ratio_floor]) * k_stop_adapt_ratio
#         #             short_k_stop_price[i] += offset

#         #             if k_stop_accelerator and eff_ratio[i] and eff_ratio[i] > k_stop_eff_ratio_trigger:
#         #                 short_k_stop_price[i] = short_k_stop_price[i] - (close[i] - short_k_stop_price[i]) * eff_ratio[i]

#         #         ## 3. profit protective stop loss
#         #         short_change = low[i] / short_entry_price[i] - 1
#         #         if short_change < -p_stop_price_trigger:
#         #             cur_p_stop_price = ( 1 + short_change * p_stop_retain_ratio) * short_entry_price[i]
#         #             if np.isnan(short_p_stop_price[i - 1]):
#         #                 short_p_stop_price[i] = cur_p_stop_price
#         #             else:
#         #                 short_p_stop_price[i] = np.nanmin([short_p_stop_price[i - 1], cur_p_stop_price])

#         #         ## 4. reverse converge stop loss
#         #         if np.isnan(short_r_stop_price[i - 1]):
#         #             short_r_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
#         #         else:
#         #             short_r_stop_price[i] = short_r_stop_price[i - 1]

#         #         if close[i] > open[i]:
#         #             short_r_stop_price[i] = short_r_stop_price[i] + (close[i] - open[i]) * r_stop_eff_ratio

#         #         short_stop[i] = np.nanmin([
#         #             short_t_stop_price[i],
#         #             short_k_stop_price[i],
#         #             short_p_stop_price[i],
#         #             short_r_stop_price[i],
#         #         ])

#         #         # 触发空单止损
#         #         if close[i] >= short_stop[i]:
#         #             to_exit = True

#         #     # No position
#         #     else:
#         #         if signal[i] != 0:
#         #             # if still in cooldown period
#         #             if (signal[i] == last_pos) and (cooldown < cooldown_period):
#         #                 cooldown += 1
#         #                 continue
#         #             to_open = True # open a pos at the next bar
#         #             cooldown = 0 # cooldown over
#         #         else: 
#         #             cooldown += 1 

#         # return {
#         #         "pos": position,
#         #         "long_entry_price": long_entry_price,
#         #         "short_entry_price": short_entry_price,
#         #         # "holding_time": holding_time,

#         #         # "long_t_stop_price": long_t_stop_price,
#         #         # "short_t_stop_price": short_t_stop_price,
#         #         # "long_k_stop_price": long_k_stop_price,
#         #         # "short_k_stop_price": short_k_stop_price,
#         #         # "long_p_stop_price": long_p_stop_price,
#         #         # "short_p_stop_price": short_p_stop_price,
#         #         # "long_r_stop_price": long_r_stop_price,
#         #         # "short_r_stop_price": short_r_stop_price,
#         #         "long_stop": long_stop,
#         #         "short_stop": short_stop
#         # }

#     def _add_pos_w_exits(self):
#         for signal_col in self.exit_params:
#             open                = self._df.select(pl.col("open_price")).to_numpy().flatten()
#             high                = self._df.select(pl.col("high_price")).to_numpy().flatten()
#             low                 = self._df.select(pl.col("low_price")).to_numpy().flatten()
#             close               = self._df.select(pl.col("close_price")).to_numpy().flatten()
#             vwap                = self._df.select(pl.col("vwap")).to_numpy().flatten()
#             signal              = self._df.select(pl.col(f"{signal_col}").fill_null(0)).to_numpy().flatten()
#             params              = self.exit_params[signal_col]

#             stop_d = self._calculate_pos_w_exits( open, 
#                                                  high, 
#                                                  low, 
#                                                  close, 
#                                                  vwap, 
#                                                  signal,
#                                                  params)

#             self._df = self._df.with_columns([
#                 pl.Series( f"{signal_col}_{name}", vals ) for name, vals in stop_d.items()  # ✅ List
#             ])


# # ########################
# # ### Legacy Code below
# # ### Decide to move to loop using numpy

# #     def _add_long_t_stop_price_expr(self, 
# #                                 signal_col: str, 
# #                                 params: dict) -> pl.Expr:
# #         """ 
# #         Time adaptive trailing stop price for long entry
# #         """
# #         entry_price = f"{signal_col}_long_entry_price"
# #         holding_time = f"{signal_col}_holding_time"

# #         # Exit Parameters
# #         long_sl_factor = params.get('long_sl_factor', 0.02)
# #         liqka_decay_speed = params.get('liqka_decay_speed', 0.2)
# #         liqka_floor = params.get('liqka_floor', 0.01)

# #         def _lower_after_entry(pos, close, entry):
# #             n = len(pos)
# #             s = [None] * n
# #             for i in range(1, n):
# #                 if pos[i]:
# #                     if s[i-1] is None:
# #                         s[i] = entry[i]
# #                     else:
# #                         s[i] = np.max(s[i-1], close[i])
# #             return s
        
# #         lower_after_entry = (
# #             pl.struct([pl.col(f"{signal_col}_pos"), pl.col("close_price"), pl.col(entry_price)] )
# #             .map_batches(lambda s: pl.Series(
# #                 _lower_after_entry(
# #                     s.struct.field(f"{signal_col}_pos"),
# #                     s.struct.field("close_price"),
# #                     s.struct.field(entry_price)
# #                 )
# #             ))
# #         )

# #         # lower_after_entry = pl.col(entry_price)
# #         # lower_after_entry = pl.max_horizontal(lower_after_entry.shift(1),
# #         #                                      pl.col("close_price")
# #         # )

# #         liqka = pl.lit(0.0)

# #         liqka = (liqka.shift(1, fill_value = 0) + liqka_decay_speed * pl.col(holding_time))
# #         liqka = pl.min_horizontal(liqka, liqka_floor)

# #         t_stop_price = lower_after_entry - (1 - liqka) * pl.col("close_price") * long_sl_factor

# #         return t_stop_price.alias(f"{signal_col}_long_t_stop_price")
    
# #     def _add_short_t_stop_price_expr(self, 
# #                                     signal_col: str, 
# #                                     params: dict) -> pl.Expr:
# #         """ 
# #         Time adaptive trailing stop price for short entry
# #         """

# #         entry_price = f"{signal_col}_short_entry_price"
# #         holding_time = f"{signal_col}_holding_time"

# #         # Exit Parameters
# #         short_sl_factor = params.get('short_sl_factor', 0.02)
# #         liqka_decay_speed = params.get('liqka_decay_speed', 0.01)
# #         liqka_floor = params.get('liqka_floor', 0.01)

# #         higher_after_entry = pl.col(entry_price)
# #         higher_after_entry = pl.min_horizontal(higher_after_entry.shift(1),
# #                                              pl.col("close_price")
# #         )

# #         liqka = pl.lit(0.0)

# #         liqka = (liqka.shift(1, fill_value = 0) + liqka_decay_speed * pl.col(holding_time))
# #         liqka = pl.min_horizontal(liqka, liqka_floor)

# #         t_stop_price = higher_after_entry + (1 - liqka) * pl.col("close_price") * short_sl_factor

# #         return t_stop_price.alias(f"{signal_col}_short_t_stop_price")

# #     def _add_long_k_stop_price_expr(self, 
# #                                signal_col: str,
# #                                params: dict) -> pl.Expr:
# #         """ 
# #         Price adaptive trailing stop price for long position
# #         """

# #         entry_price = f"{signal_col}_long_entry_price"

# #         # Exit Parameters
# #         long_sl_factor = params.get('long_sl_factor', 0.02)
# #         eff_ratio = params.get('eff_ratio', 0.3)
# #         k_stop_eff_ratio_floor = params.get('k_stop_eff_ratio_floor', 0.1)
# #         k_stop_adapt_ratio = params.get('k_stop_adapt_ratio', 0.5)
# #         k_stop_accelerator = params.get('k_stop_accelerator', False)
# #         k_stop_eff_ratio_trigger = params.get('k_stop_eff_ratio_trigger', 0.5)

# #         eff_ratio = self.sg._getEffRatio(32) # past 32 bars so about 8 hours

# #         # initialize stop price
# #         k_stop_price = pl.col(entry_price) * (1 - long_sl_factor)
        
# #         cond = (pl.col("close_price") > pl.col("open_price"))
# #         offset = (
# #             pl.when(cond)
# #             .then((pl.col("close_price") - pl.col("open_price")) 
# #                   * k_stop_adapt_ratio
# #                   * pl.max_horizontal(eff_ratio.abs(), k_stop_eff_ratio_floor)
# #                   )
# #             .otherwise(0)
# #         )

# #         # k_stop_price = (
# #         #     pl.when(cond)
# #         #     .then(k_stop_price.shift(1) + offset)
# #         #     pl.when(cond 
# #         #             & k_stop_accelerator
# #         #             & (eff_ratio > k_stop_eff_ratio_trigger))
# #         #     .then(k_stop_price.shift(1) + offset + (pl.col("close_price") - k_stop_price.shift(1)) * eff_ratio)
# #         #     .otherwise(k_stop_price.shift(1))
# #         # )

# #         return k_stop_price.alias(f"{signal_col}_long_k_stop_price")

# #     def _add_short_k_stop_price_expr(self,
# #                                 signal_col: str,
# #                                 params: dict) -> pl.Expr:
# #         """ Price adaptive trailing stop price for short position """

# #         entry_price = f"{signal_col}_short_entry_price"

# #         # Exit Parameters
# #         long_sl_factor = params.get('long_sl_factor', 0.02)
# #         eff_ratio = params.get('eff_ratio', 0.3)
# #         k_stop_eff_ratio_floor = params.get('k_stop_eff_ratio_floor', 0.1)
# #         k_stop_adapt_ratio = params.get('k_stop_adapt_ratio', 0.5)
# #         k_stop_accelerator = params.get('k_stop_accelerator', False)
# #         k_stop_eff_ratio_trigger = params.get('k_stop_eff_ratio_trigger', 0.5)

# #         eff_ratio = self.sg._getEffRatio(32) # past 32 bars so about 8 hours

# #         offset = (
# #                     pl.when(pl.col("close_price") < pl.col("open_price"))
# #                     .then((pl.col("close_price") - pl.col("open_price")) 
# #                         * k_stop_adapt_ratio
# #                         * pl.max_horizontal(eff_ratio.abs(), k_stop_eff_ratio_floor)
# #                         )
# #                     .otherwise(0)
# #                 )
# #         k_stop_price = k_stop_price.shift(1) + offset

# #         k_stop_price = (
# #             pl.when( k_stop_accelerator & (eff_ratio > k_stop_eff_ratio_trigger) )
# #             .then(k_stop_price - (pl.col("close_price") - k_stop_price.shift(1)) * eff_ratio)
# #             .otherwise(k_stop_price.shift(1))
# #         )
# #         return k_stop_price.alias(f"{signal_col}_short_k_stop_price")

# #     def _add_long_p_stop_price_expr(self,
# #                                signal_col: str,
# #                                params: dict) -> pl.Expr:
# #         """ 
# #         Price protective stop loss for long positions
# #         """

# #         entry_price = f"{signal_col}_long_entry_price"
# #         pos_col = f"{signal_col}_pos"

# #         # Exit Parameters
# #         p_stop_price_trigger = params.get('p_stop_price_trigger', 0.1) # 当前利润达到10%时启动保护性止损
# #         p_stop_retain_ratio = params.get('p_stop_retain_ratio', 0.5) # 保护性止损重新设定为利润的50%

# #         # Calculate profit from entry
# #         long_change = pl.col("high_price") / pl.col(entry_price) - 1
        
# #         # Only activate protective stop when in profit above threshold
# #         protection_active = (
# #             (pl.col(pos_col) == 1) &  # Long position
# #             (long_change > p_stop_price_trigger)
# #         )
        
# #         # initialize protective stop price
# #         cur_p_stop_price = (pl.when(protection_active)
# #                         .then((1 + long_change * p_stop_retain_ratio) * pl.col(entry_price))
# #                         .otherwise(None)
# #         )
# #         p_stop_price = cur_p_stop_price
# #         p_stop_price = pl.max_horizontal(p_stop_price.shift(1), cur_p_stop_price)

# #         return p_stop_price.alias(f"{signal_col}_long_p_stop_price")

# #     def _add_short_p_stop_price_expr(self,
# #                                 signal_col: str,
# #                                 params: dict) -> pl.Expr:
# #         """
# #         Price protective stop loss for short positions
# #         """

# #         entry_price = f"{signal_col}_short_entry_price"
# #         pos_col = f"{signal_col}_pos"

# #         # Exit Parameters
# #         p_stop_price_trigger = params.get('p_stop_price_trigger', 0.1) # 当前利润达到10%时启动保护性止损
# #         p_stop_retain_ratio = params.get('p_stop_retain_ratio', 0.5) # 保护性止损重新设定为利润的50%
        
# #         # Calculate profit from entry
# #         short_change = pl.col("low_price") / pl.col(entry_price) - 1
        
# #         # Only activate protective stop when in profit above threshold
# #         protection_active = (
# #             (pl.col(pos_col) == -1) &  # short position
# #             (short_change < -p_stop_price_trigger)
# #         )
        
# #         # initialize protective stop price
# #         cur_p_stop_price = (pl.when(protection_active)
# #                         .then((1 + short_change * p_stop_retain_ratio) * pl.col(entry_price))
# #                         .otherwise(None)
# #         )
# #         p_stop_price = cur_p_stop_price
# #         p_stop_price = pl.min_horizontal(p_stop_price.shift(1), cur_p_stop_price)

# #         return p_stop_price.alias(f"{signal_col}_short_p_stop_price")

# #     def _add_long_r_stop_price_expr(self,
# #                                 signal_col: str,
# #                                 params: dict) -> pl.Expr:
# #         """ reverse converge exit for long postion """

# #         long_sl_factor = params.get('long_sl_factor', 0.02)

# #         # initialize r_stop_price
# #         r_stop_price = pl.col( f"{signal_col}_long_entry_price" ) * (1 - long_sl_factor)
# #         r_stop_eff_ratio = params.get('r_stop_eff_ratio', 0.5)

# #         diff = pl.col("open_price") - pl.col("close_price")
# #         r_stop_price = (pl.when( diff > 0 )
# #                           .then( r_stop_price.shift(1) + diff * r_stop_eff_ratio ) # Price moves against pos, tighten sl
# #         )
        
# #         return r_stop_price.alias(f"{signal_col}_long_r_stop_price")

# #     def _add_short_r_stop_price_expr(self,
# #                                  signal_col: str,
# #                                  params: dict) -> pl.Expr:
# #         """ reverse converge exit for short position """

# #         short_sl_factor = params.get('short_sl_factor', 0.02)
        
# #         # initialize r_stop_price
# #         r_stop_price = pl.col( f"{signal_col}_short_entry_price" ) * (1 + short_sl_factor)
# #         r_stop_eff_ratio = params.get('r_stop_eff_ratio', 0.5)

# #         diff = pl.col("open_price") - pl.col("close_price")
# #         r_stop_price = (pl.when( diff < 0 )
# #                           .then( r_stop_price.shift(1) + diff * r_stop_eff_ratio ) # Price moves against pos, tighten sl
# #         )

# #         return r_stop_price.alias(f"{signal_col}_short_r_stop_price")


# #     def apply_exit_methods(self, *args, **kwargs) -> None:
# #         """
# #         Apply exit methods to the specified signal column
        
# #         """

# #         for signal_col in self.exit_signals:
# #             params = self.exit_signals[signal_col]

# #             long_t_stop_price = self._add_long_t_stop_price(signal_col, params)
# #             long_k_stop_price = self._add_long_k_stop_price_expr(signal_col, params)
# #             long_p_stop_price = self._add_long_p_stop_price_expr(signal_col, params)
# #             long_r_stop_price = self._add_long_r_stop_price_expr(signal_col, params)

# #             short_t_stop_price = self._add_short_t_stop_price(signal_col, params)
# #             short_k_stop_price = self._add_short_k_stop_price_expr(signal_col, params)
# #             short_p_stop_price = self._add_short_p_stop_price_expr(signal_col, params)
# #             short_r_stop_price = self._add_short_r_stop_price_expr(signal_col, params)

# #             long_exit = pl.max_horizontal(
# #                 long_t_stop_price,
# #                 long_k_stop_price,
# #                 long_p_stop_price,
# #                 long_r_stop_price
# #             ).alias(f"{signal_col}_long_exit")

# #             short_exit = pl.min_horizontal(
# #                 short_t_stop_price,
# #                 short_k_stop_price,
# #                 short_p_stop_price,
# #                 short_r_stop_price
# #             ).alias(f"{signal_col}_short_exit")

# #             # Add stop prices
# #             self._df = (
# #                 self._df
# #                 .lazy()
# #                 .with_columns([
# #                     long_t_stop_price,
# #                     long_k_stop_price,
# #                     long_p_stop_price,
# #                     long_r_stop_price,
# #                     short_t_stop_price,
# #                     short_k_stop_price,
# #                     short_p_stop_price,
# #                     short_r_stop_price,
# #                     long_exit,
# #                     short_exit
# #                 ])
# #                 .collect()
# #             )

# #     def _add_initial_pos(self) -> None:
# #         """
# #         - 如果前一个bar信号为1，那无论前一个bar的pos是什么，当前bar的pos都设为1
# #         （也就是说，1. 如果前一个bar的pos是0，那么开多。
# #                   2. 如果前一个bar的pos是1，那么维持pos不动。
# #                   3. 如果前一个bar的pos是-1，那么平掉之前的空单，反手做多）
# #         - 如果前一个bar信号为0，那么当前bar的pos取决于前一个bar的pos。
# #         - 如果前一个bar的信号为-1，那无论前一个bar的pos是什么，当前bar的pos都设为1
# #         （也就是说，1. 如果前一个bar的pos是0，那么开空。
# #                   2. 如果前一个bar的pos是-1，那么维持pos不动。
# #                   3. 如果前一个bar的pos是1，那么平掉之前的多单，反手做）
# #         """
        
# #         pos_exprs = []
# #         for signal_col in self.exit_signals:
# #             # Check if the signal column exists
# #             if signal_col not in self._df.columns:
# #                 raise ValueError(f"Signal column '{signal_col}' does not exist in the DataFrame.")
# #             pos_expr = pl.lit(0)
# #             long_pos_1 = (pl.col(signal_col).shift(1) == 1)
# #             long_pos_2 = (pl.col(signal_col).shift(1) == 0) & (pos_expr.shift(1) == 1)

# #             short_pos_1 = (pl.col(signal_col).shift(1) == -1)
# #             short_pos_2 = (pl.col(signal_col).shift(1) == 0) & (pos_expr.shift(1) == -1)

# #             pos_expr = (
# #                 pl.when(long_pos_1 | long_pos_2)
# #                 .then(1)
# #                 .when(short_pos_1 | short_pos_2)
# #                 .then(-1)
# #                 .otherwise(pos_expr)
# #                 .fill_null(0)
# #                 .alias(f"{signal_col}_pos")
# #             )

# #             pos_exprs.append(pos_expr)

# #         # Apply ALL columns in ONE operation
# #         self._df = (
# #             self._df
# #             .lazy()
# #             .with_columns(pos_exprs)
# #             .collect()
# #         )


# #     def _add_entry_price_cols(self) -> None:
# #         """
# #         Add separate entry price columns for long and short positions in one operation
# #         """
# #         exprs = []
# #         for signal_col in self.exit_signals:
# #             # Check if the signal column exists
# #             if signal_col not in self._df.columns:
# #                 raise ValueError(f"Signal column '{signal_col}' does not exist in the DataFrame.")

# #             # Create initial position column
# #             pos_col_expr = pl.col(f"{signal_col}_pos")
                        
# #             # Detect new position entries
# #             new_long_entry = (pos_col_expr == 1) & (pos_col_expr.shift(1) != 1)
# #             new_short_entry = (pos_col_expr == -1) & (pos_col_expr.shift(1) != -1)
            
# #             # Create long entry price (use VWAP for now, can be made configurable)
# #             long_entry_price = (
# #                 pl.when(new_long_entry)
# #                 .then(pl.col("vwap"))  
# #                 .otherwise(None)
# #                 .forward_fill()
# #             )
            
# #             # Only keep long entry price when in long position
# #             long_entry_price_final = (
# #                 pl.when(pos_col_expr == 1)
# #                 .then(long_entry_price)
# #                 .otherwise(None)
# #                 .alias(f"{signal_col}_long_entry_price")
# #             )
            
# #             # Create short entry price
# #             short_entry_price = (
# #                 pl.when(new_short_entry)
# #                 .then(pl.col("vwap")) 
# #                 .otherwise(None)
# #                 .forward_fill()
# #             )
            
# #             # Only keep short entry price when in short position
# #             short_entry_price_final = (
# #                 pl.when(pos_col_expr == -1)
# #                 .then(short_entry_price)
# #                 .otherwise(None)
# #                 .alias(f"{signal_col}_short_entry_price")
# #             )

# #             # Add holding time column
# #             pos_group = ((pos_col_expr != pos_col_expr.shift(1, fill_value=0))
# #                             .cum_sum()
# #                             .alias("pos_group")
# #             )
            
# #             holding_time = (pos_group.cum_count()
# #                                     .over(pos_group)
# #             )

# #             holding_time = (pl.when(pos_col_expr == 0)
# #                             .then(0)
# #                             .otherwise(holding_time)
# #                             .alias(f"{signal_col}_holding_time")
# #             )
# #             # Add all expressions to the list
# #             exprs.extend([long_entry_price_final, short_entry_price_final, holding_time])
        
# #         # Apply ALL columns in ONE operation
# #         self._df = (
# #             self._df
# #             .lazy()
# #             .with_columns(exprs)
# #             .collect()
# #         )

# #     def _calculate_stop_prices(self,
# #                          position: np.ndarray, 
# #                          open: np.ndarray,
# #                          high: np.ndarray,
# #                          low: np.ndarray,
# #                          close: np.ndarray, 
# #                          long_entry_price: np.ndarray,
# #                          short_entry_price: np.ndarray,
# #                          holding_time: np.ndarray,
# #                          params:dict, 
# #                          *args,
# #                          **kwargs) -> dict[np.ndarray]:
# #         """ compute stop prices 
# #         Args:
# #             - position(np.ndarray): The current position (1 for long, -1 for short, 0 for no position)
# #             - OHLCV
# #             - params(dict): The parameters for stop price calculation

# #         Returns:
# #             dict[np.ndarray]: A dictionary containing the stop prices
# #         """

# #         n = close.shape[0]

# #         # load parameters
# #         long_sl_factor = params.get('long_sl_factor', 0.02)
# #         short_sl_factor = params.get('short_sl_factor', 0.02)
# #         liqka_decay_speed = params.get('liqka_decay_speed', 0.2)
# #         liqka_floor = params.get('liqka_floor', 0.01)

# #         k_stop_eff_ratio_floor = params.get('k_stop_eff_ratio_floor', 0.1)
# #         k_stop_adapt_ratio = params.get('k_stop_adapt_ratio', 0.5)
# #         period = params.get("ER_period", 32)
# #         k_stop_accelerator = params.get("k_stop_accelerator", True)
# #         k_stop_eff_ratio_trigger = params.get("k_stop_eff_ratio_trigger", 0.5)

# #         p_stop_price_trigger = params.get('p_stop_price_trigger', 0.5)  # 50% profit trigger
# #         p_stop_retain_ratio  = params.get('p_stop_retain_ratio', 0.4)  # Retain 40% of profit

# #         r_stop_eff_ratio = params.get('r_stop_eff_ratio', 0.5)  # Reverse converge stop loss efficiency ratio
        
# #         cooldown_period = params.get("cooldown_period", 32)

# #         # Compute ER
# #         eff_ratio = np.full(n, np.nan)
# #         direction = np.abs(close[period: ] - close[ :-period ] ).flatten()
# #         volatility = np.array([
# #                     np.sum(np.abs(close[i - period + 1: i + 1] - close[i - period: i]))
# #                     for i in range(period, n)
# #         ])
# #         eff_ratio[period: ] = direction / volatility

# #         # initialize vectors
# #         higher_after_entry = np.full(n, np.nan)
# #         lower_after_entry  = np.full(n, np.nan)
# #         liqka = np.zeros(n)

# #         # initialize status vectors
# #         new_long_entry_price  = np.full(n, np.nan) # new long entry price after exits
# #         new_short_entry_price = np.full(n, np.nan) # new short entry price after exits
# #         cooldown              = 0                  # cooldown period after exits
# #         to_exit               = False              # a flag to exit at the next bar

# #         # Stop-price vectors
# #         long_t_stop_price  = np.full(n, np.nan)
# #         short_t_stop_price = np.full(n, np.nan) 
        
# #         long_k_stop_price  = np.full(n, np.nan)
# #         short_k_stop_price = np.full(n, np.nan)

# #         long_p_stop_price  = np.full(n, np.nan)
# #         short_p_stop_price = np.full(n, np.nan)

# #         long_r_stop_price  = np.full(n, np.nan)
# #         short_r_stop_price = np.full(n, np.nan)

# #         long_stop = np.full(n, np.nan)
# #         short_stop = np.full(n, np.nan)

# #         for i in range(1, n):
            
# #             # Long Exits
# #             if position[i] > 0:

# #                 ## 1. time adaptive trailing sl
# #                 if np.isnan(lower_after_entry[i - 1]):
# #                     lower_after_entry[i] = np.nanmax([long_entry_price[i], close[i]])
# #                 else:
# #                     lower_after_entry[i] = np.nanmax([lower_after_entry[i-1], close[i]])

# #                 liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
# #                 liqka[i] = np.nanmin([liqka[i], liqka_floor])
# #                 long_t_stop_price[i] = lower_after_entry[i] - (1 - liqka[i]) * close[i] * long_sl_factor

# #                 ## 2. price adaptive trailing sl
# #                 if np.isnan(long_k_stop_price[i - 1]):
# #                     long_k_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
# #                 else:
# #                     long_k_stop_price[i] = long_k_stop_price[i - 1]

# #                 if close[i] > open[i]:
# #                     offset = (close[i] - open[i]) * np.nanmax( [eff_ratio[i], k_stop_eff_ratio_floor]) * k_stop_adapt_ratio
# #                     long_k_stop_price[i] += offset

# #                     if k_stop_accelerator and eff_ratio[i] and eff_ratio[i] > k_stop_eff_ratio_trigger:
# #                         long_k_stop_price[i] = long_k_stop_price[i] + (close[i] - long_k_stop_price[i]) * eff_ratio[i]

# #                 ## 3. profit protective stop loss
# #                 long_change = high[i] / long_entry_price[i] - 1
# #                 if long_change > p_stop_price_trigger:
# #                     cur_p_stop_price = ( 1 + long_change * p_stop_retain_ratio) * long_entry_price[i]
# #                     if np.isnan(long_p_stop_price[i - 1]):
# #                         long_p_stop_price[i] = cur_p_stop_price
# #                     else:
# #                         long_p_stop_price[i] = np.nanmax([long_p_stop_price[i - 1], cur_p_stop_price])

# #                 ## 4. reverse converge stop loss
# #                 if np.isnan(long_r_stop_price[i - 1]):
# #                     long_r_stop_price[i] = long_entry_price[i] * (1 - long_sl_factor)
# #                 else:
# #                     long_r_stop_price[i] = long_r_stop_price[i - 1]

# #                 if close[i] < open[i]:
# #                     long_r_stop_price[i] = long_r_stop_price[i] + (open[i] - close[i]) * r_stop_eff_ratio

# #                 long_stop[i] = np.nanmax([long_t_stop_price[i], 
# #                                           long_k_stop_price[i], 
# #                                           long_p_stop_price[i], 
# #                                           long_r_stop_price[i]]
# #                                 )

# #             # Short Exits
# #             elif position[i] < 0:

# #                 ## 1. time adaptive trailing sl
# #                 if np.isnan(higher_after_entry[i - 1]):
# #                     higher_after_entry[i] = short_entry_price[i]
# #                 higher_after_entry[i] = np.nanmin([ higher_after_entry[i-1], close[i]])

# #                 liqka[i] = liqka[i - 1] + liqka_decay_speed * holding_time[i]
# #                 liqka[i] = np.nanmin([liqka[i], liqka_floor])
# #                 short_t_stop_price[i] = higher_after_entry[i] + (1 - liqka[i]) * close[i] * short_sl_factor

# #                 ## 2. price adaptive trailing sl
# #                 if np.isnan(short_k_stop_price[i - 1]):
# #                     short_k_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
# #                 else:
# #                     short_k_stop_price[i] = short_k_stop_price[i - 1]
# #                 if close[i] < open[i]:
# #                     offset = (close[i] - open[i]) * np.nanmax( [eff_ratio[i], k_stop_eff_ratio_floor]) * k_stop_adapt_ratio
# #                     short_k_stop_price[i] += offset

# #                     if k_stop_accelerator and eff_ratio[i] and eff_ratio[i] > k_stop_eff_ratio_trigger:
# #                         short_k_stop_price[i] = short_k_stop_price[i] - (close[i] - short_k_stop_price[i]) * eff_ratio[i]

# #                 ## 3. profit protective stop loss
# #                 short_change = low[i] / short_entry_price[i] - 1
# #                 if short_change < -p_stop_price_trigger:
# #                     cur_p_stop_price = ( 1 + short_change * p_stop_retain_ratio) * short_entry_price[i]
# #                     if np.isnan(short_p_stop_price[i - 1]):
# #                         short_p_stop_price[i] = cur_p_stop_price
# #                     else:
# #                         short_p_stop_price[i] = np.nanmin([short_p_stop_price[i - 1], cur_p_stop_price])

# #                 ## 4. reverse converge stop loss
# #                 if np.isnan(short_r_stop_price[i - 1]):
# #                     short_r_stop_price[i] = short_entry_price[i] * (1 + short_sl_factor)
# #                 else:
# #                     short_r_stop_price[i] = short_r_stop_price[i - 1]

# #                 if close[i] > open[i]:
# #                     short_r_stop_price[i] = short_r_stop_price[i] + (close[i] - open[i]) * r_stop_eff_ratio

# #                 short_stop[i] = np.nanmin([
# #                     short_t_stop_price[i],
# #                     short_k_stop_price[i],
# #                     short_p_stop_price[i],
# #                     short_r_stop_price[i],
# #                 ])

# #         return {
# #                 "long_t_stop_price": long_t_stop_price,
# #                 "short_t_stop_price": short_t_stop_price,
# #                 "long_k_stop_price": long_k_stop_price,
# #                 "short_k_stop_price": short_k_stop_price,
# #                 "long_p_stop_price": long_p_stop_price,
# #                 "short_p_stop_price": short_p_stop_price,
# #                 "long_r_stop_price": long_r_stop_price,
# #                 "short_r_stop_price": short_r_stop_price,
# #                 "long_stop": long_stop,
# #                 "short_stop": short_stop
# #         }



# #     def _add_stop_prices(self) -> None:
# #         for signal_col in self.exit_signals:
# #             # Calculate stop prices
# #             position            = self._df.select(pl.col(f"{signal_col}_pos")).to_numpy().flatten()
# #             open                = self._df.select(pl.col("open_price")).to_numpy().flatten()
# #             high                = self._df.select(pl.col("high_price")).to_numpy().flatten()
# #             low                 = self._df.select(pl.col("low_price")).to_numpy().flatten()
# #             close               = self._df.select(pl.col("close_price")).to_numpy().flatten()
# #             long_entry_price    = self._df.select(pl.col(f"{signal_col}_long_entry_price")).to_numpy().flatten()
# #             short_entry_price   = self._df.select(pl.col(f"{signal_col}_short_entry_price")).to_numpy().flatten()
# #             holding_time        = self._df.select(pl.col(f"{signal_col}_holding_time")).to_numpy().flatten()
# #             params              = self.exit_signals[signal_col]

# #             stop_d = self._calculate_stop_prices(position, open, high, low, close, long_entry_price, short_entry_price, holding_time, params)

# #             self._df = self._df.with_columns(
# #                 pl.Series( f"{signal_col}_{name}", stop_prices ) for name, stop_prices in stop_d.items()
# #             )

# #     def _add_exited_pos(self) -> None:
# #         """ 
# #         通过计算好的止损/止盈价格，添加退出的仓位信息
# #         """

# #         for signal_col in self.exit_signals:
# #             long_exit_cond = (pl.col(f"{signal_col}_pos") == 1) & (pl.col(f"{signal_col}_long_stop") > pl.col("close_price"))
# #             short_exit_cond = (pl.col(f"{signal_col}_pos") == -1) & (pl.col(f"{signal_col}_short_stop") < pl.col("close_price"))


# #             new_pos_expr = (pl.when(long_exit_cond.shift(1)) # if prev long exit is activated, then exit the position in the next bar
# #                             .then(0)
# #                             .when(short_exit_cond.shift(1)) # if prev short exit is activated, then exit the position in the next bar
# #                             .then(0)
# #                             .otherwise(pl.col(f"{signal_col}_pos"))
# #             )

# #             self._df = self._df.with_columns(
# #                                 new_pos_expr.alias(f"{signal_col}_pos_AE")
# #                         )
# # """