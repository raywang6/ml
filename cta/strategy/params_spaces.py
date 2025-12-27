PARAMS_SPACES = {
    "SuperTrend": {
        "window_size": {'type': 'int', 'low': 21, 'high': 193, 'step': 4},
        "ayami_multi": {'type': 'float', 'low': 0.5, 'high': 2.0, 'step': 0.25},
        "sideway_filter_lookback": {'type': 'int', 'low': 48, 'high': 1000, 'step': 8}, # 288 * 15 minutes = 3 days
        "extrem_filter": {'type': 'float', 'low': 2, 'high': 10, 'step': 0.5},
        "entry_threshold": {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.05}
        },

    "VolumeLongShort": {
            "window_size": {'type': 'int', 'low': 5, 'high': 195, 'step': 5},
            # "ayami_multi": {'type':
    },

    "PSYMOMLongShort": {
        "window_size": {'type': 'int', 'low': 21, 'high': 193, 'step': 4},
        "ama_period": {'type': 'int', 'low': 4, 'high': 60, 'step': 4},  # AMA的周期
        'MADDlength': {'type': 'int', 'low': 2, 'high': 14, 'step': 1}, 
        # "trigger": {'type': 'float', 'low': -1e-4, 'high': 8e-3, 'step': 1e-5},  # 触发阈值
    },

    "PSYVOLLongShort": {
        "window_size": {'type': 'int', 'low': 21, 'high': 193, 'step': 4},
        "ma_window_size": {'type': 'int', 'low': 10, 'high': 60, 'step': 5},  # MA窗口长度
    },

    "SwingLongShort": {
        "pivot_len": {'type': 'int', 'low': 20, 'high': 100, 'step': 5},  # Pivot长度
        "total_len": {'type': 'int', 'low': 1, 'high': 20, 'step': 1},  # 总长度
        "slow_window_size": {'type': 'int', 'low': 18, 'high': 54, 'step': 3 }, # 慢速均线长度
        "fast_window_size": {'type': 'int', 'low': 6, 'high': 18, 'step': 2},  # 快速均线长度
    },

    "MRLongShort": {
        "lookback": {'type': 'int', 'low': 20, 'high': 190, 'step': 5},  # Lookback period for IMI
        "lower_boundary": {'type': 'int', 'low': 10, 'high': 35, 'step': 5}, # 上行带
        "upper_boundary": {'type': 'int', 'low': 60, 'high': 100, 'step': 5}, # 下行带
        "sideway_filter_lookback": {'type': 'int', 'low': 48, 'high': 1000, 'step': 8}, # 侧向过滤回溯期
        "entry_threshold": {'type': 'float', 'low': 0.05, 'high': 0.5, 'step': 0.05} # 进入阈值
    },

    "SMAStrat": {
        "window_size": {"type": "int", "low": 10, "high": 190, "step": 5},  # MA周期参数M
        "S": {"type": "int", "low": 1, "high": 10, "step": 1},  # 缩放参数S
    },

    "TurtleLongShort": {
        "window_size": { "type": "int", "low": 10, "high": 60, "step": 10 },  # 唐奇安通道长度
        'length': {'type': 'int', 'low': 7, 'high': 56, 'step': 7}, 
        'fslength': {'type': 'int', 'low': 60, 'high': 120, 'step': 5},
    },

    "TurtleLongShort2": {
        "long_window_size": { "type": "int", "low": 48, "high": 96 * 7, "step": 4 },  # 唐奇安通道长度
        "short_window_size": { "type": "int", "low": 12, "high": 88, "step": 4 },  # 短周期长度
        "turtle_er_threshold": {'type': 'float', 'low': 0.1, "high": 0.5, 'step': 0.1},  # 效率比阈值
        "soup_er_threshold": {'type': 'float', 'low': 0.05, "high": 0.1, 'step': 0.05},  # 效率比阈值
        "volume_threshold": {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.5},  # 成交量阈值
    },

    "BrickLongShort": {
        "window_size": {'type': 'int', 'low': 21, 'high': 289, 'step': 4},
    },

    "SuperTrendADPExit": {
        "window_size": { "type": "int", "low": 20, "high": 96 * 7, "step": 4 },  # 唐奇安通道长度
        'vol_length': {'type': 'int', 'low': 7, 'high': 70, 'step': 7}, 
        'maria_multi': {'type': 'float', 'low': 0.25, 'high': 2.0, 'step': 0.25},
    },

    "RSIBBReversal": {
        "window_size": { "type": "int", "low": 16, "high": 96 * 7, "step": 4 },  # 唐奇安通道长度
        "over_bought": {'type': 'int', 'low': 70, 'high': 100, 'step': 1},
        "over_sold": {'type': 'int', 'low': 0, 'high': 30, 'step': 1},
        "sigma": {'type': 'int', 'low': 1, 'high': 5, 'step': 1},
    }
}

EXIT_PARAMS = {
    "SuperTrend": {
                'long_sl_factor': 0.08, 
                'short_sl_factor': 0.08,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.01,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.08,
                'p_stop_retain_ratio': 0.4,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.7,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': False, 
                'r_stop_eff_ratio': 0,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': False,
                'max_holding_time_allowed': 0  
            },
    "VolumeLongShort": {
                'long_sl_factor': 0.04, 
                'short_sl_factor': 0.04,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.025,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.05,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.6,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.25,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 16*4
            },
    "PSYMOMLongShort": {
                'long_sl_factor': 0.05, 
                'short_sl_factor': 0.05,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.025,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.05,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.6,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.25,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 16*4
            },
    "PSYVOLLongShort": {
                'long_sl_factor': 0.05, 
                'short_sl_factor': 0.05,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.025,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.05,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.6,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.25,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 16*4
            },
    "SwingLongShort": {
                'long_sl_factor': 0.04, 
                'short_sl_factor': 0.04,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.05,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.03,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.7,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.33,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 8*4
            },
    "MRLongShort": {
                'long_sl_factor': 0.04, 
                'short_sl_factor': 0.04,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.05,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.03,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.7,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.33,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 8*4
            },
    "SMAStrat": {
                'long_sl_factor': 0.08, 
                'short_sl_factor': 0.08,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.01,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.08,
                'p_stop_retain_ratio': 0.4,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.7,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': False, 
                'r_stop_eff_ratio': 0,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': False,
                'max_holding_time_allowed': 0  
            },
    "TurtleLongShort": {
                'long_sl_factor': 0.05, 
                'short_sl_factor': 0.05,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.025,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.05,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.6,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.25,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 16*4
            },
    "TurtleLongShort2": {
                'long_sl_factor': 0.05, 
                'short_sl_factor': 0.05,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.025,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.05,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.6,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.25,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 16*4
            },
    "BrickLongShort": {
                'long_sl_factor': 0.05, 
                'short_sl_factor': 0.05,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.025,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.05,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.6,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.25,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 16*4
            },
    "SuperTrendADPExit": {
                'long_sl_factor': 0.08, 
                'short_sl_factor': 0.08,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.01,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.08,
                'p_stop_retain_ratio': 0.4,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.7,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': False, 
                'r_stop_eff_ratio': 0,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': False,
                'max_holding_time_allowed': 0  
            },
    "RSIBBReversal": {
                'long_sl_factor': 0.04, 
                'short_sl_factor': 0.04,

                'enable_price_adaptive': True,
                'liqka_decay_speed': 0.05,
                'liqka_floor': 0.4,
                
                'enable_profit_protective': True,
                'p_stop_price_trigger': 0.03,
                'p_stop_retain_ratio': 0.5,

                'enable_price_adaptive': True,
                'k_stop_adapt_ratio': 0.5,
                'k_stop_accelerator': False,
                'k_stop_eff_ratio_trigger': 0.7,
                'k_stop_eff_ratio_floor':0.2,

                'enable_reverse_converge': True, 
                'r_stop_eff_ratio': 0.33,

                'ER_period': 32,
                'cooldown_period': 32,

                'enable_max_holding_time': True,
                'max_holding_time_allowed': 8*4
            },

}

