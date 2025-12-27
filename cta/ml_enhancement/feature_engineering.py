"""
特征工程模块 - 为机器学习模型提取技术指标和市场特征
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union
import talib


class FeatureEngineer:
    """技术指标和市场特征提取器"""
    
    def __init__(self, data: pl.DataFrame):
        """
        初始化特征工程器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.clone()
        self._validate_data()
    
    def _validate_data(self):
        """验证数据格式"""
        required_cols = ['datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
    
    def add_basic_features(self) -> pl.DataFrame:
        """添加基础技术指标特征"""
        
        # 转换为numpy数组以使用talib
        close = self.data['close_price'].to_numpy()
        high = self.data['high_price'].to_numpy()
        low = self.data['low_price'].to_numpy()
        open_price = self.data['open_price'].to_numpy()
        volume = self.data['volume'].to_numpy()
        
        features = []
        
        # 价格相关特征
        features.extend([
            # 收益率
            pl.col('close_price').pct_change().alias('returns'),
            pl.col('close_price').pct_change().rolling_std(20).alias('volatility_20'),
            
            # 价格位置
            ((pl.col('close_price') - pl.col('low_price').rolling_min(20)) / 
             (pl.col('high_price').rolling_max(20) - pl.col('low_price').rolling_min(20))).alias('price_position_20'),
            
            # 成交量特征
            pl.col('volume').pct_change().alias('volume_change'),
            (pl.col('volume') / pl.col('volume').rolling_mean(20)).alias('volume_ratio_20'),
        ])
        
        # 使用talib计算技术指标
        sma_5 = talib.SMA(close, timeperiod=5)
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        ema_12 = talib.EMA(close, timeperiod=12)
        ema_26 = talib.EMA(close, timeperiod=26)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # RSI
        rsi_14 = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # ATR
        atr_14 = talib.ATR(high, low, close, timeperiod=14)
        
        # 应用polars特征
        result = self.data.with_columns(features)
        
        # 添加talib特征 - 使用更安全的方法
        talib_data = {
            'sma_5': sma_5,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'rsi_14': rsi_14,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'atr_14': atr_14,
        }
        
        # 逐个添加talib特征
        for name, values in talib_data.items():
            result = result.with_columns(pl.Series(name, values))
        
        # 计算衍生特征
        derived_features = [
            # 布林带相关
            ((pl.col('close_price') - pl.col('bb_lower')) / (pl.col('bb_upper') - pl.col('bb_lower'))).alias('bb_position'),
            (pl.col('bb_upper') - pl.col('bb_lower')).alias('bb_width'),
            
            # 移动平均线关系
            (pl.col('close_price') / pl.col('sma_20') - 1).alias('price_sma20_ratio'),
            (pl.col('sma_5') / pl.col('sma_20') - 1).alias('sma5_sma20_ratio'),
            (pl.col('ema_12') / pl.col('ema_26') - 1).alias('ema12_ema26_ratio'),
            
            # 动量指标
            (pl.col('rsi_14') - 50).alias('rsi_centered'),
            pl.col('macd_hist').diff().alias('macd_hist_change'),
        ]
        
        result = result.with_columns(derived_features)
        
        return result
    
    def create_all_features(self) -> pl.DataFrame:
        """创建所有特征"""
        result = self.add_basic_features()
        return result
    
    def get_feature_names(self, exclude_base: bool = True) -> List[str]:
        """获取特征名称列表"""
        
        base_cols = ['datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'volume_usd', 'vwap']
        
        if exclude_base:
            # 确保返回的都是字符串
            return [str(col) for col in self.data.columns if str(col) not in base_cols]
        else:
            return [str(col) for col in self.data.columns]