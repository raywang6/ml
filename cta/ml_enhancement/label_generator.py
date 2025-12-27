"""
标签生成模块 - 为机器学习模型生成预测目标标签
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import talib


class LabelGenerator:
    """机器学习标签生成器"""
    
    def __init__(self, data: pl.DataFrame):
        """
        初始化标签生成器
        
        Args:
            data: 包含OHLCV和技术指标的DataFrame
        """
        self.data = data.clone()
        self._validate_data()
    
    def _validate_data(self):
        """验证数据格式"""
        required_cols = ['datetime', 'open_price', 'high_price', 'low_price', 'close_price']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
    
    def generate_volatility_labels(self, 
                                 lookforward_periods: List[int] = [4, 12, 24],  # 1h, 3h, 6h for 15min data
                                 volatility_threshold: float = 0.02) -> pl.DataFrame:
        """
        生成波动率预测标签
        
        Args:
            lookforward_periods: 前瞻期数列表
            volatility_threshold: 波动率阈值
            
        Returns:
            包含波动率标签的DataFrame
        """
        
        result = self.data
        
        for periods in lookforward_periods:
            # 先添加基础波动率特征
            basic_features = [
                # 计算未来N期的已实现波动率
                pl.col('close_price')
                .pct_change()
                .rolling_std(periods)
                .shift(-periods)  # 向前看
                .alias(f'future_vol_{periods}'),
                
                # 当前波动率
                pl.col('close_price')
                .pct_change()
                .rolling_std(periods)
                .alias(f'current_vol_{periods}')
            ]
            
            result = result.with_columns(basic_features)
            
            # 然后添加基于这些特征的标签
            derived_labels = [
                # 波动率变化标签 (分类)
                pl.when(pl.col(f'future_vol_{periods}') > pl.col(f'current_vol_{periods}') * (1 + volatility_threshold))
                .then(pl.lit(1))  # 波动率显著上升
                .when(pl.col(f'future_vol_{periods}') < pl.col(f'current_vol_{periods}') * (1 - volatility_threshold))
                .then(pl.lit(-1))  # 波动率显著下降
                .otherwise(pl.lit(0))  # 波动率稳定
                .alias(f'vol_change_label_{periods}'),
                
                # 高波动率标签 (二分类)
                (pl.col(f'future_vol_{periods}') > pl.col(f'future_vol_{periods}').rolling_quantile(0.8, window_size=252))
                .cast(pl.Int8)
                .alias(f'high_vol_label_{periods}')
            ]
            
            result = result.with_columns(derived_labels)
        
        return result
    
    def generate_bollinger_breakout_labels(self, 
                                         bb_period: int = 20,
                                         bb_std: float = 2.0,
                                         lookforward_periods: List[int] = [4, 8, 16]) -> pl.DataFrame:
        """
        生成布林带突破概率标签
        
        Args:
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            lookforward_periods: 前瞻期数列表
            
        Returns:
            包含布林带突破标签的DataFrame
        """
        
        # 计算布林带
        close = self.data['close_price'].to_numpy()
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
        
        # 添加布林带到数据
        bb_features = [
            pl.Series('bb_upper_label', bb_upper),
            pl.Series('bb_middle_label', bb_middle),
            pl.Series('bb_lower_label', bb_lower),
        ]
        
        result = self.data
        for series in bb_features:
            result = result.with_columns(pl.lit(series).alias(series.name))
        
        for periods in lookforward_periods:
            # 先添加基础突破标签
            basic_breakout_labels = [
                # 未来是否突破上轨
                (pl.col('high_price')
                 .rolling_max(periods)
                 .shift(-periods) > pl.col('bb_upper_label')
                ).cast(pl.Int8).alias(f'upper_breakout_{periods}'),
                
                # 未来是否突破下轨
                (pl.col('low_price')
                 .rolling_min(periods)
                 .shift(-periods) < pl.col('bb_lower_label')
                ).cast(pl.Int8).alias(f'lower_breakout_{periods}')
            ]
            
            result = result.with_columns(basic_breakout_labels)
            
            # 然后添加基于这些标签的衍生标签
            derived_breakout_labels = [
                # 任意方向突破
                ((pl.col(f'upper_breakout_{periods}') == 1) | 
                 (pl.col(f'lower_breakout_{periods}') == 1)
                ).cast(pl.Int8).alias(f'any_breakout_{periods}'),
                
                # 突破方向 (-1: 下破, 0: 无突破, 1: 上破)
                pl.when(pl.col(f'upper_breakout_{periods}') == 1)
                .then(pl.lit(1))
                .when(pl.col(f'lower_breakout_{periods}') == 1)
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .alias(f'breakout_direction_{periods}')
            ]
            
            result = result.with_columns(derived_breakout_labels)
        
        return result
    
    def generate_trend_end_labels(self, 
                                trend_periods: List[int] = [20, 50],
                                profit_threshold: float = 0.02,
                                loss_threshold: float = 0.01,
                                lookforward_periods: List[int] = [8, 16, 32]) -> pl.DataFrame:
        """
        生成趋势结束预测标签
        
        Args:
            trend_periods: 趋势判断周期
            profit_threshold: 止盈阈值
            loss_threshold: 止损阈值
            lookforward_periods: 前瞻期数列表
            
        Returns:
            包含趋势结束标签的DataFrame
        """
        
        labels = []
        
        for trend_period in trend_periods:
            # 定义趋势方向
            trend_direction = (
                pl.when(pl.col('close_price') > pl.col('close_price').rolling_mean(trend_period))
                .then(pl.lit(1))  # 上升趋势
                .when(pl.col('close_price') < pl.col('close_price').rolling_mean(trend_period))
                .then(pl.lit(-1))  # 下降趋势
                .otherwise(pl.lit(0))  # 无明确趋势
                .alias(f'trend_direction_{trend_period}')
            )
            
            labels.append(trend_direction)
            
            for periods in lookforward_periods:
                # 计算未来收益率
                future_return = (
                    (pl.col('close_price').shift(-periods) / pl.col('close_price') - 1)
                    .alias(f'future_return_{periods}')
                )
                
                # 趋势结束标签 (基于止盈止损)
                trend_end_long = (
                    (pl.col(f'trend_direction_{trend_period}') == 1) &
                    ((pl.col(f'future_return_{periods}') < -loss_threshold) |  # 止损
                     (pl.col(f'future_return_{periods}') > profit_threshold))   # 止盈
                ).cast(pl.Int8).alias(f'trend_end_long_{trend_period}_{periods}')
                
                trend_end_short = (
                    (pl.col(f'trend_direction_{trend_period}') == -1) &
                    ((pl.col(f'future_return_{periods}') > loss_threshold) |   # 止损
                     (pl.col(f'future_return_{periods}') < -profit_threshold)) # 止盈
                ).cast(pl.Int8).alias(f'trend_end_short_{trend_period}_{periods}')
                
                # 任意趋势结束
                any_trend_end = (
                    (pl.col(f'trend_end_long_{trend_period}_{periods}') == 1) |
                    (pl.col(f'trend_end_short_{trend_period}_{periods}') == 1)
                ).cast(pl.Int8).alias(f'any_trend_end_{trend_period}_{periods}')
                
                labels.extend([future_return, trend_end_long, trend_end_short, any_trend_end])
        
        return self.data.with_columns(labels)
    
    def generate_time_to_event_labels(self, 
                                    event_thresholds: Dict[str, float] = {
                                        'small_move': 0.01,
                                        'medium_move': 0.02, 
                                        'large_move': 0.03
                                    },
                                    max_periods: int = 48) -> pl.DataFrame:
        """
        生成事件发生时间预测标签
        
        Args:
            event_thresholds: 事件阈值字典
            max_periods: 最大前瞻期数
            
        Returns:
            包含时间预测标签的DataFrame
        """
        
        labels = []
        
        for event_name, threshold in event_thresholds.items():
            # 计算到达阈值的时间
            def calculate_time_to_threshold(prices: pl.Series, threshold: float, max_periods: int) -> pl.Series:
                """计算价格到达阈值的时间"""
                prices_np = prices.to_numpy()
                n = len(prices_np)
                time_to_event = np.full(n, max_periods, dtype=np.int32)
                
                for i in range(n - max_periods):
                    current_price = prices_np[i]
                    
                    for j in range(1, min(max_periods + 1, n - i)):
                        future_price = prices_np[i + j]
                        price_change = abs(future_price / current_price - 1)
                        
                        if price_change >= threshold:
                            time_to_event[i] = j
                            break
                
                return pl.Series(time_to_event)
            
            # 上涨到达阈值时间
            time_to_up_move = (
                pl.col('close_price')
                .map_batches(lambda x: calculate_time_to_threshold(x, threshold, max_periods))
                .alias(f'time_to_up_{event_name}')
            )
            
            # 下跌到达阈值时间  
            time_to_down_move = (
                pl.col('close_price')
                .map_batches(lambda x: calculate_time_to_threshold(x, threshold, max_periods))
                .alias(f'time_to_down_{event_name}')
            )
            
            # 任意方向到达阈值时间
            time_to_any_move = (
                pl.min_horizontal([
                    pl.col(f'time_to_up_{event_name}'),
                    pl.col(f'time_to_down_{event_name}')
                ]).alias(f'time_to_any_{event_name}')
            )
            
            # 分类标签 (短期/中期/长期)
            time_category = (
                pl.when(pl.col(f'time_to_any_{event_name}') <= max_periods // 4)
                .then(pl.lit(0))  # 短期
                .when(pl.col(f'time_to_any_{event_name}') <= max_periods // 2)
                .then(pl.lit(1))  # 中期
                .when(pl.col(f'time_to_any_{event_name}') < max_periods)
                .then(pl.lit(2))  # 长期
                .otherwise(pl.lit(3))  # 不发生
                .alias(f'time_category_{event_name}')
            )
            
            labels.extend([time_to_up_move, time_to_down_move, time_to_any_move, time_category])
        
        return self.data.with_columns(labels)
    
    def generate_all_labels(self) -> pl.DataFrame:
        """生成所有标签"""
        
        # 逐步添加标签
        result = self.generate_volatility_labels()
        self.data = result
        
        result = self.generate_bollinger_breakout_labels()
        self.data = result
        
        result = self.generate_trend_end_labels()
        self.data = result
        
        result = self.generate_time_to_event_labels()
        
        return result
    
    def get_label_names(self) -> List[str]:
        """获取所有标签名称"""
        
        base_cols = ['datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        
        # 查找标签列 (包含特定关键词的列)
        label_keywords = ['label', 'breakout', 'trend_end', 'time_to', 'future_']
        
        label_cols = []
        for col in self.data.columns:
            if col not in base_cols and any(keyword in col for keyword in label_keywords):
                label_cols.append(col)
        
        return label_cols