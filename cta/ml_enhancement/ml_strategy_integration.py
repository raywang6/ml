"""
机器学习策略集成模块 - 将ML预测结果集成到现有策略中
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.signal_generator import SignalGenerator
from .feature_engineering import FeatureEngineer
from .label_generator import LabelGenerator
from .qlib_predictor import QlibPredictor


class MLEnhancedStrategy:
    """机器学习增强的交易策略"""
    
    def __init__(self, 
                 base_strategy_params: Dict,
                 ml_config: Dict = None,
                 exit_params: Dict = None):
        """
        初始化ML增强策略
        
        Args:
            base_strategy_params: 基础策略参数
            ml_config: ML配置参数
            exit_params: 退出策略参数
        """
        self.base_strategy_params = base_strategy_params
        self.ml_config = ml_config or self._default_ml_config()
        self.exit_params = exit_params
        
        # ML组件
        self.feature_engineer = None
        self.label_generator = None
        self.predictor = None
        
        # 预测结果
        self.predictions = {}
        self.ml_signals = {}
    
    def _default_ml_config(self) -> Dict:
        """默认ML配置"""
        return {
            'models': ['lgb'],
            'prediction_targets': {
                'volatility': {
                    'lookforward_periods': [4, 12, 24],
                    'threshold': 0.02
                },
                'breakout': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'lookforward_periods': [4, 8, 16]
                },
                'trend_end': {
                    'trend_periods': [20, 50],
                    'profit_threshold': 0.02,
                    'loss_threshold': 0.01,
                    'lookforward_periods': [8, 16, 32]
                }
            },
            'feature_config': {
                'use_basic_features': True,
                'use_volatility_features': True,
                'use_trend_features': True,
                'use_microstructure_features': True
            },
            'training_config': {
                'train_ratio': 0.7,
                'valid_ratio': 0.15,
                'min_samples': 1000
            }
        }
    
    def prepare_ml_data(self, data: pl.DataFrame) -> Tuple[pl.DataFrame, List[str], List[str]]:
        """
        准备ML训练数据
        
        Args:
            data: 原始OHLCV数据
            
        Returns:
            (enhanced_data, feature_names, label_names): 增强数据、特征名、标签名
        """
        
        print("Preparing ML data...")
        
        # 特征工程
        self.feature_engineer = FeatureEngineer(data)
        enhanced_data = self.feature_engineer.create_all_features()
        
        # 标签生成
        self.label_generator = LabelGenerator(enhanced_data)
        
        # 根据配置生成标签
        if 'volatility' in self.ml_config['prediction_targets']:
            vol_config = self.ml_config['prediction_targets']['volatility']
            enhanced_data = self.label_generator.generate_volatility_labels(
                lookforward_periods=vol_config['lookforward_periods'],
                volatility_threshold=vol_config['threshold']
            )
            self.label_generator.data = enhanced_data
        
        if 'breakout' in self.ml_config['prediction_targets']:
            breakout_config = self.ml_config['prediction_targets']['breakout']
            enhanced_data = self.label_generator.generate_bollinger_breakout_labels(
                bb_period=breakout_config['bb_period'],
                bb_std=breakout_config['bb_std'],
                lookforward_periods=breakout_config['lookforward_periods']
            )
            self.label_generator.data = enhanced_data
        
        if 'trend_end' in self.ml_config['prediction_targets']:
            trend_config = self.ml_config['prediction_targets']['trend_end']
            enhanced_data = self.label_generator.generate_trend_end_labels(
                trend_periods=trend_config['trend_periods'],
                profit_threshold=trend_config['profit_threshold'],
                loss_threshold=trend_config['loss_threshold'],
                lookforward_periods=trend_config['lookforward_periods']
            )
        
        # 获取特征和标签名称
        all_feature_names = self.feature_engineer.get_feature_names(exclude_base=True)
        
        # 过滤掉非数值列和不需要的列
        exclude_cols = {
            'symbol', 'datetime', 'instrument', 'end_tm',
            'open_price', 'high_price', 'low_price', 'close_price', 'volume'
        }
        
        feature_names = []
        for col in all_feature_names:
            # 首先检查是否在排除列表中
            if col in exclude_cols:
                continue
                
            # 然后检查是否在数据中存在且为数值类型
            if col in enhanced_data.columns:
                try:
                    col_data = enhanced_data[col]
                    if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
                        feature_names.append(col)
                except:
                    continue
        
        label_names = self.label_generator.get_label_names()
        
        print(f"Generated {len(feature_names)} features and {len(label_names)} labels")
        print(f"Filtered feature names (first 10): {feature_names[:10]}")
        print(f"Available columns: {enhanced_data.columns[:10]}...")
        
        return enhanced_data, feature_names, label_names
    
    def train_ml_models(self, 
                       data: pl.DataFrame,
                       symbol: str = "BTCUSDT") -> Dict:
        """
        训练ML模型
        
        Args:
            data: 训练数据
            symbol: 交易对符号
            
        Returns:
            训练结果
        """
        
        print("Training ML models...")
        
        # 准备数据
        enhanced_data, feature_names, label_names = self.prepare_ml_data(data)
        
        # 检查数据量
        if len(enhanced_data) < self.ml_config['training_config']['min_samples']:
            raise ValueError(f"Insufficient data: {len(enhanced_data)} < {self.ml_config['training_config']['min_samples']}")
        
        # 初始化预测器
        self.predictor = QlibPredictor()
        
        # 准备ML数据
        clean_feature_names = [str(name) for name in feature_names]
        clean_label_names = [str(name) for name in label_names]
        
        print(f"Feature names (first 5): {clean_feature_names[:5]}")
        print(f"Label names: {clean_label_names}")
        
        data_dict, _ = self.predictor.prepare_qlib_data(
            enhanced_data, clean_feature_names, clean_label_names, symbol
        )
        
        features_array = data_dict['features']
        labels_array = data_dict['labels']
        
        # 训练模型
        training_config = self.ml_config['training_config']
        results = self.predictor.train_models_numpy(
            features_array=features_array,
            labels_array=labels_array,
            feature_names=clean_feature_names,
            label_names=clean_label_names,
            train_ratio=training_config['train_ratio'],
            valid_ratio=training_config['valid_ratio'],
            model_types=self.ml_config['models']
        )
        
        print("ML model training completed")
        return results
    
    def generate_ml_predictions(self, data: pl.DataFrame, symbol: str = "BTCUSDT") -> Dict[str, np.ndarray]:
        """
        生成ML预测
        
        Args:
            data: 预测数据
            symbol: 交易对符号
            
        Returns:
            预测结果字典
        """
        
        if self.predictor is None or not self.predictor.models:
            raise ValueError("No trained models available. Please train models first.")
        
        # 准备特征数据 - 使用与训练时相同的特征工程器
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer(data)
        
        enhanced_data = self.feature_engineer.create_all_features()
        
        # 使用与训练时相同的特征名称和过滤逻辑
        all_feature_names = self.feature_engineer.get_feature_names(exclude_base=True)
        
        # 过滤掉非数值列和不需要的列
        exclude_cols = {
            'symbol', 'datetime', 'instrument', 'end_tm',
            'open_price', 'high_price', 'low_price', 'close_price', 'volume'
        }
        
        feature_names = []
        for col in all_feature_names:
            if col not in exclude_cols:
                try:
                    # 检查列是否为数值类型
                    col_data = enhanced_data[col]
                    if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
                        feature_names.append(col)
                except:
                    continue
        
        # 准备预测数据
        data_dict, _ = self.predictor.prepare_qlib_data(
            enhanced_data, feature_names, [], symbol
        )
        
        features_array = data_dict['features']
        print(f"Prediction features array shape: {features_array.shape}")
        
        # 生成预测
        predictions = self.predictor.predict_numpy(
            features_array=features_array,
            model_type=self.ml_config['models'][0]  # 使用第一个模型
        )
        
        self.predictions = predictions
        return predictions
    
    def create_ml_signals(self, predictions: Dict[str, np.ndarray]) -> Dict[str, pl.Series]:
        """
        基于ML预测创建交易信号
        
        Args:
            predictions: ML预测结果
            
        Returns:
            ML信号字典
        """
        
        signals = {}
        
        # 波动率信号
        vol_signals = self._create_volatility_signals(predictions)
        signals.update(vol_signals)
        
        # 突破信号
        breakout_signals = self._create_breakout_signals(predictions)
        signals.update(breakout_signals)
        
        # 趋势结束信号
        trend_end_signals = self._create_trend_end_signals(predictions)
        signals.update(trend_end_signals)
        
        self.ml_signals = signals
        return signals
    
    def _create_volatility_signals(self, predictions: Dict[str, np.ndarray]) -> Dict[str, pl.Series]:
        """创建波动率相关信号"""
        
        signals = {}
        
        # 查找波动率预测
        vol_predictions = {k: v for k, v in predictions.items() if 'vol_change_label' in k}
        
        for pred_name, pred_values in vol_predictions.items():
            # 波动率上升信号 (预测值 > 0.5 表示波动率将上升)
            vol_up_signal = pl.Series('vol_up_signal', (pred_values > 0.5).astype(int))
            
            # 波动率下降信号
            vol_down_signal = pl.Series('vol_down_signal', (pred_values < -0.5).astype(int))
            
            signals[f'vol_up_{pred_name}'] = vol_up_signal
            signals[f'vol_down_{pred_name}'] = vol_down_signal
        
        return signals
    
    def _create_breakout_signals(self, predictions: Dict[str, np.ndarray]) -> Dict[str, pl.Series]:
        """创建突破相关信号"""
        
        signals = {}
        
        # 查找突破预测
        breakout_predictions = {k: v for k, v in predictions.items() if 'breakout_direction' in k}
        
        for pred_name, pred_values in breakout_predictions.items():
            # 上突破信号
            upper_breakout_signal = pl.Series('upper_breakout_signal', (pred_values > 0.5).astype(int))
            
            # 下突破信号
            lower_breakout_signal = pl.Series('lower_breakout_signal', (pred_values < -0.5).astype(int))
            
            signals[f'upper_breakout_{pred_name}'] = upper_breakout_signal
            signals[f'lower_breakout_{pred_name}'] = lower_breakout_signal
        
        return signals
    
    def _create_trend_end_signals(self, predictions: Dict[str, np.ndarray]) -> Dict[str, pl.Series]:
        """创建趋势结束信号"""
        
        signals = {}
        
        # 查找趋势结束预测
        trend_end_predictions = {k: v for k, v in predictions.items() if 'any_trend_end' in k}
        
        for pred_name, pred_values in trend_end_predictions.items():
            # 趋势结束信号 (预测值 > 0.5 表示趋势可能结束)
            trend_end_signal = pl.Series('trend_end_signal', (pred_values > 0.5).astype(int))
            
            signals[f'trend_end_{pred_name}'] = trend_end_signal
        
        return signals
    
    def integrate_with_base_strategy(self, 
                                   data: pl.DataFrame,
                                   base_strategy_name: str = "SuperTrend") -> pl.DataFrame:
        """
        将ML信号与基础策略集成
        
        Args:
            data: 市场数据
            base_strategy_name: 基础策略名称
            
        Returns:
            集成ML信号的数据
        """
        
        print("Integrating ML signals with base strategy...")
        
        # 生成基础策略信号
        sg = SignalGenerator(data, exit_params=self.exit_params)
        
        # 配置基础策略
        features_config = {
            base_strategy_name: self.base_strategy_params
        }
        
        sg.equipFeatures(features_config)
        base_data = sg.to_polars()
        
        # 生成ML预测和信号
        try:
            predictions = self.generate_ml_predictions(data)
            ml_signals = self.create_ml_signals(predictions)
            
            # 将ML信号添加到数据中
            for signal_name, signal_series in ml_signals.items():
                # 确保长度匹配
                if len(signal_series) == len(base_data):
                    base_data = base_data.with_columns(signal_series.alias(signal_name))
        except Exception as e:
            print(f"Warning: ML prediction failed: {e}")
            print("Continuing with base strategy only...")
            # 创建默认的ML信号（全为0）
            ml_signals = {
                'vol_up_signal': pl.Series('vol_up_signal', [0] * len(base_data)),
                'vol_down_signal': pl.Series('vol_down_signal', [0] * len(base_data)),
                'upper_breakout_signal': pl.Series('upper_breakout_signal', [0] * len(base_data)),
                'lower_breakout_signal': pl.Series('lower_breakout_signal', [0] * len(base_data)),
                'trend_end_signal': pl.Series('trend_end_signal', [0] * len(base_data))
            }
            
            # 添加默认信号到数据中
            for signal_name, signal_series in ml_signals.items():
                base_data = base_data.with_columns(signal_series.alias(signal_name))
        
        # 创建集成信号
        integrated_signals = self._create_integrated_signals(base_data, base_strategy_name)
        
        # 添加集成信号到数据
        for signal_name, signal_expr in integrated_signals.items():
            base_data = base_data.with_columns(signal_expr.alias(signal_name))
        
        print("ML integration completed")
        return base_data
    
    def _create_integrated_signals(self, 
                                 data: pl.DataFrame, 
                                 base_strategy_name: str) -> Dict[str, pl.Expr]:
        """
        创建集成信号
        
        Args:
            data: 包含基础策略和ML信号的数据
            base_strategy_name: 基础策略名称
            
        Returns:
            集成信号表达式字典
        """
        
        signals = {}
        
        # ML增强的基础策略信号
        base_signal_col = base_strategy_name
        
        if base_signal_col in data.columns:
            # 查找实际存在的ML信号列
            available_cols = data.columns
            
            # 查找波动率信号
            vol_up_cols = [col for col in available_cols if 'vol_up_' in col]
            vol_down_cols = [col for col in available_cols if 'vol_down_' in col]
            
            # 查找突破信号
            upper_breakout_cols = [col for col in available_cols if 'upper_breakout_' in col]
            lower_breakout_cols = [col for col in available_cols if 'lower_breakout_' in col]
            
            # 策略1: ML波动率过滤（如果有波动率信号）
            if vol_up_cols and vol_down_cols:
                vol_up_col = vol_up_cols[0]  # 使用第一个找到的列
                vol_down_col = vol_down_cols[0]
                
                vol_enhanced_signal = (
                    pl.when(pl.col(vol_up_col).fill_null(0) == 1)
                    .then(pl.col(base_signal_col) * 1.2)  # 增强信号
                    .when(pl.col(vol_down_col).fill_null(0) == 1)
                    .then(pl.col(base_signal_col) * 0.8)  # 减弱信号
                    .otherwise(pl.col(base_signal_col))
                )
                signals['ml_vol_enhanced_signal'] = vol_enhanced_signal
            
            # 策略2: 突破确认（如果有突破信号）
            if upper_breakout_cols and lower_breakout_cols:
                upper_breakout_col = upper_breakout_cols[0]
                lower_breakout_col = lower_breakout_cols[0]
                
                breakout_confirmed_signal = (
                    pl.when(
                        (pl.col(base_signal_col) > 0) & 
                        (pl.col(upper_breakout_col).fill_null(0) == 1)
                    ).then(pl.lit(1))  # 确认做多
                    .when(
                        (pl.col(base_signal_col) < 0) & 
                        (pl.col(lower_breakout_col).fill_null(0) == 1)
                    ).then(pl.lit(-1))  # 确认做空
                    .otherwise(pl.lit(0))  # 无确认信号
                )
                signals['ml_breakout_confirmed_signal'] = breakout_confirmed_signal
            
            # 策略3: 简化的综合ML信号
            # 基于基础策略信号的简单增强
            ml_composite_signal = (
                pl.when(pl.col(base_signal_col) > 0)
                .then(pl.lit(1))
                .when(pl.col(base_signal_col) < 0)
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
            )
            signals['ml_composite_signal'] = ml_composite_signal
        
        return signals
    
    def backtest_ml_strategy(self, 
                           data: pl.DataFrame,
                           strategy_name: str = "ml_composite_signal") -> Dict:
        """
        回测ML增强策略
        
        Args:
            data: 包含策略信号的数据
            strategy_name: 策略信号列名
            
        Returns:
            回测结果
        """
        
        if strategy_name not in data.columns:
            raise ValueError(f"Strategy signal '{strategy_name}' not found in data")
        
        # 计算收益率
        returns = data.select([
            pl.col('datetime'),
            pl.col('close_price').pct_change().alias('market_return'),
            pl.col(strategy_name).alias('position')
        ]).with_columns([
            (pl.col('market_return') * pl.col('position').shift(1)).alias('strategy_return')
        ])
        
        # 计算累积收益
        returns = returns.with_columns([
            (1 + pl.col('strategy_return').fill_null(0)).cum_prod().alias('cumulative_return'),
            (1 + pl.col('market_return').fill_null(0)).cum_prod().alias('market_cumulative_return')
        ])
        
        # 计算统计指标
        strategy_returns = returns['strategy_return'].drop_nulls()
        
        if len(strategy_returns) == 0:
            return {'error': 'No valid strategy returns'}
        
        total_return = returns['cumulative_return'].tail(1).item() - 1
        volatility = strategy_returns.std()
        sharpe_ratio = strategy_returns.mean() / volatility * np.sqrt(252 * 24 * 4) if volatility > 0 else 0
        
        # 最大回撤
        cumulative = returns['cumulative_return'].to_numpy()
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 交易统计
        positions = data[strategy_name].to_numpy()
        position_changes = np.diff(positions, prepend=0)
        num_trades = np.sum(position_changes != 0)
        
        results = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'returns_data': returns
        }
        
        return results
    
    def save_ml_models(self, filepath: str):
        """保存ML模型"""
        if self.predictor:
            self.predictor.save_models(filepath)
    
    def load_ml_models(self, filepath: str):
        """加载ML模型"""
        if self.predictor is None:
            self.predictor = QlibPredictor()
        self.predictor.load_models(filepath)