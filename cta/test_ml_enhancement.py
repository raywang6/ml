"""
测试ML增强策略 - 使用真实BTCUSDT数据
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ml_enhancement import MLEnhancedStrategy, FeatureEngineer, LabelGenerator
from strategy.params_spaces import PARAMS_SPACES, EXIT_PARAMS

def load_btc_data():
    """加载BTCUSDT数据"""
    data_path = Path(__file__).parent / "data" / "BTCUSDT.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 读取parquet文件
    data = pl.read_parquet(data_path)
    
    print(f"数据形状: {data.shape}")
    print(f"列名: {data.columns}")
    print(f"数据时间范围: {data['end_tm'].min()} 到 {data['end_tm'].max()}")
    
    # 映射列名以匹配FeatureEngineer的期望
    column_mapping = {
        'end_tm': 'datetime',
        'open': 'open_price',
        'high': 'high_price', 
        'low': 'low_price',
        'close': 'close_price'
    }
    
    # 重命名列
    data = data.rename(column_mapping)
    
    return data

def test_feature_engineering():
    """测试特征工程"""
    print("=== 测试特征工程 ===")
    
    data = load_btc_data()
    
    # 使用最近2000条数据进行测试
    test_data = data.tail(2000)
    
    feature_engineer = FeatureEngineer(test_data)
    enhanced_data = feature_engineer.create_all_features()
    
    print(f"原始数据: {test_data.shape}")
    print(f"增强数据: {enhanced_data.shape}")
    
    feature_names = feature_engineer.get_feature_names()
    print(f"生成特征数: {len(feature_names)}")
    
    return enhanced_data

def test_label_generation():
    """测试标签生成"""
    print("\n=== 测试标签生成 ===")
    
    enhanced_data = test_feature_engineering()
    
    label_generator = LabelGenerator(enhanced_data)
    
    # 生成波动率标签
    data_with_labels = label_generator.generate_volatility_labels(
        lookforward_periods=[4, 12],
        volatility_threshold=0.015
    )
    
    # 生成突破标签
    label_generator.data = data_with_labels
    data_with_labels = label_generator.generate_bollinger_breakout_labels(
        lookforward_periods=[4, 8]
    )
    
    label_names = label_generator.get_label_names()
    print(f"生成标签数: {len(label_names)}")
    print(f"标签名称: {label_names}")
    
    return data_with_labels

def test_ml_strategy():
    """测试完整ML策略"""
    print("\n=== 测试ML增强策略 ===")
    
    data = load_btc_data()
    
    # 使用最近3000条数据
    test_data = data.tail(30000)
    
    # 配置策略参数
    base_strategy_params = {
        'window_size': 21,
        'ayami_multi': 1.0,
        'sideway_filter_lookback': 50,
        'extrem_filter': 2.8,
        'entry_threshold': 0.55
    }
    
    # 简化ML配置
    ml_config = {
        'models': ['lgb'],
        'prediction_targets': {
            'volatility': {
                'lookforward_periods': [4, 12],
                'threshold': 0.015
            },
            'breakout': {
                'bb_period': 20,
                'bb_std': 2.0,
                'lookforward_periods': [4, 8]
            }
        },
        'training_config': {
            'train_ratio': 0.7,
            'valid_ratio': 0.15,
            'min_samples': 1000
        }
    }
    
    # 创建ML增强策略
    ml_strategy = MLEnhancedStrategy(
        base_strategy_params=base_strategy_params,
        ml_config=ml_config,
        exit_params=EXIT_PARAMS
    )
    
    try:
        # 训练ML模型
        print("开始训练ML模型...")
        training_results = ml_strategy.train_ml_models(test_data, symbol="BTCUSDT")
        print(f"训练完成，共训练 {len(training_results)} 个标签模型")
        
        # 集成策略
        print("集成ML信号与基础策略...")
        integrated_data = ml_strategy.integrate_with_base_strategy(test_data)
        print(f"集成完成，数据形状: {integrated_data.shape}")
        
        # 显示可用信号
        signal_cols = [col for col in integrated_data.columns if 'signal' in col]
        print(f"可用信号: {signal_cols}")
        
        # 调试信息：检查各种信号的统计
        print("\n=== 信号统计 ===")
        if 'SuperTrend' in integrated_data.columns:
            supertrend_stats = integrated_data['SuperTrend'].value_counts().sort('SuperTrend')
            print(f"SuperTrend信号统计: {supertrend_stats}")
        
        if 'ml_composite_signal' in integrated_data.columns:
            ml_stats = integrated_data['ml_composite_signal'].value_counts().sort('ml_composite_signal')
            print(f"ML复合信号统计: {ml_stats}")
        
        # 检查ML相关列
        ml_cols = [col for col in integrated_data.columns if any(keyword in col for keyword in ['vol_', 'breakout_', 'ml_'])]
        print(f"ML相关列: {ml_cols[:10]}...")  # 只显示前10个
        
        # 回测ML策略
        if 'ml_composite_signal' in integrated_data.columns:
            print("开始回测ML复合策略...")
            backtest_results = ml_strategy.backtest_ml_strategy(
                integrated_data, 
                strategy_name='ml_composite_signal'
            )
            
            print("\n=== 回测结果 ===")
            print(f"总收益率: {backtest_results['total_return']:.2%}")
            print(f"年化波动率: {backtest_results['volatility']:.4f}")
            print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
            print(f"最大回撤: {backtest_results['max_drawdown']:.2%}")
            print(f"交易次数: {backtest_results['num_trades']}")
            
            return ml_strategy, integrated_data, backtest_results
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """主测试函数"""
    print("ML增强策略测试 - 使用BTCUSDT真实数据")
    print("=" * 50)
    
    try:
        # 测试特征工程
        test_feature_engineering()
        
        # 测试标签生成
        test_label_generation()
        
        # 测试完整ML策略
        ml_strategy, integrated_data, backtest_results = test_ml_strategy()
        
        if ml_strategy is not None:
            print("\n✅ 所有测试完成！")
            print("\n建议优化方向:")
            print("1. 调整特征工程参数")
            print("2. 优化标签定义")
            print("3. 尝试不同模型参数")
            print("4. 增加更多预测目标")
        else:
            print("\n❌ 测试失败，请检查错误信息")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()