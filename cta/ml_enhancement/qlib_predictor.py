"""
Qlib机器学习预测模块
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.data import D
    from qlib.model.gbdt import LGBModel
    from qlib.model.linear import LinearModel
    from qlib.model.ens import RFModel
    from qlib.workflow import R
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
except ImportError:
    print("Warning: qlib not installed. Please install with: pip install pyqlib")
    qlib = None


class QlibPredictor:
    """基于Qlib的机器学习预测器"""
    
    def __init__(self, 
                 provider_uri: str = "~/.qlib/qlib_data/crypto_data",
                 region: str = "crypto"):
        """
        初始化Qlib预测器
        
        Args:
            provider_uri: Qlib数据提供者URI
            region: 数据区域
        """
        self.provider_uri = provider_uri
        self.region = region
        self.models = {}
        self.predictions = {}
        
        if qlib is not None:
            self._init_qlib()
    
    def _init_qlib(self):
        """初始化Qlib"""
        try:
            qlib.init(provider_uri=self.provider_uri, region=self.region)
        except Exception as e:
            print(f"Qlib initialization failed: {e}")
            print("Using fallback mode without qlib integration")
    
    def prepare_qlib_data(self, 
                         data: pl.DataFrame,
                         feature_cols: List[str],
                         label_cols: List[str],
                         symbol: str = "BTCUSDT") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备Qlib格式的数据
        
        Args:
            data: 原始数据
            feature_cols: 特征列名
            label_cols: 标签列名
            symbol: 交易对符号
            
        Returns:
            (features_df, labels_df): 特征和标签DataFrame
        """
        
        n_rows = len(data)
        
        # 使用numpy数组直接创建数据
        features_data = []
        features_names = []
        
        for col in feature_cols:
            if col in data.columns:
                try:
                    values = data[col].to_numpy()
                    # 转换为float并处理NaN值
                    values = np.array(values, dtype=float)
                    values = np.nan_to_num(values, nan=0.0)
                    features_data.append(values)
                    features_names.append(str(col))
                except Exception as e:
                    print(f"Warning: Failed to convert feature column {col}: {e}")
                    # 使用默认值
                    features_data.append(np.zeros(n_rows))
                    features_names.append(str(col))
        
        # 处理标签数据
        labels_data = []
        labels_names = []
        
        for col in label_cols:
            if col in data.columns:
                try:
                    values = data[col].to_numpy()
                    # 转换为float，标签保留NaN用于过滤
                    values = np.array(values, dtype=float)
                    labels_data.append(values)
                    labels_names.append(str(col))
                except Exception as e:
                    print(f"Warning: Failed to convert label column {col}: {e}")
                    # 使用NaN填充
                    labels_data.append(np.full(n_rows, np.nan))
                    labels_names.append(str(col))
        
        # 直接返回numpy数据，绕过pandas问题
        if features_data:
            features_array = np.column_stack(features_data)
            print(f"Features array shape: {features_array.shape}")
        else:
            features_array = np.array([])
        
        if labels_data:
            labels_array = np.column_stack(labels_data)
            print(f"Labels array shape: {labels_array.shape}")
        else:
            labels_array = np.array([])
        
        # 使用简单的字典返回数据
        return {
            'features': features_array,
            'labels': labels_array,
            'feature_names': features_names,
            'label_names': labels_names
        }, None
    
    def create_model_configs(self) -> Dict[str, Dict]:
        """创建不同模型的配置"""
        
        configs = {
            'lgb': {
                'class': 'LGBModel',
                'module_path': 'qlib.model.gbdt',
                'kwargs': {
                    'loss': 'mse',
                    'colsample_bytree': 0.8879,
                    'learning_rate': 0.0421,
                    'subsample': 0.8789,
                    'lambda_l1': 205.6999,
                    'lambda_l2': 580.9768,
                    'max_depth': 8,
                    'num_leaves': 210,
                    'num_threads': 20,
                    'verbosity': -1,
                }
            },
            'linear': {
                'class': 'LinearModel',
                'module_path': 'qlib.model.linear',
                'kwargs': {}
            },
            'rf': {
                'class': 'RFModel', 
                'module_path': 'qlib.model.ens',
                'kwargs': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'random_state': 42
                }
            }
        }
        
        return configs
    
    def train_models(self,
                    features_df: pd.DataFrame,
                    labels_df: pd.DataFrame,
                    train_ratio: float = 0.7,
                    valid_ratio: float = 0.15,
                    model_types: List[str] = ['lgb']) -> Dict[str, Any]:
        """
        训练多个模型
        
        Args:
            features_df: 特征数据
            labels_df: 标签数据
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            model_types: 模型类型列表
            
        Returns:
            训练结果字典
        """
        
        if qlib is None:
            return self._train_models_fallback(features_df, labels_df, train_ratio, valid_ratio, model_types)
        
        results = {}
        
        # 数据分割
        n_samples = len(features_df)
        train_end = int(n_samples * train_ratio)
        valid_end = int(n_samples * (train_ratio + valid_ratio))
        
        train_features = features_df.iloc[:train_end]
        valid_features = features_df.iloc[train_end:valid_end]
        test_features = features_df.iloc[valid_end:]
        
        model_configs = self.create_model_configs()
        
        # 为每个标签训练模型
        for label_col in labels_df.columns:
            print(f"Training models for label: {label_col}")
            
            train_labels = labels_df[label_col].iloc[:train_end]
            valid_labels = labels_df[label_col].iloc[train_end:valid_end]
            test_labels = labels_df[label_col].iloc[valid_end:]
            
            # 移除缺失值
            train_mask = ~(train_labels.isna() | train_features.isna().any(axis=1))
            valid_mask = ~(valid_labels.isna() | valid_features.isna().any(axis=1))
            
            if train_mask.sum() < 100:  # 需要足够的训练样本
                print(f"Insufficient training samples for {label_col}, skipping...")
                continue
            
            label_results = {}
            
            for model_type in model_types:
                if model_type not in model_configs:
                    continue
                
                try:
                    # 创建模型
                    model_config = model_configs[model_type]
                    model = init_instance_by_config(model_config)
                    
                    # 准备训练数据
                    X_train = train_features[train_mask]
                    y_train = train_labels[train_mask]
                    X_valid = valid_features[valid_mask]
                    y_valid = valid_labels[valid_mask]
                    
                    # 训练模型
                    model.fit(X_train, y_train, X_valid, y_valid)
                    
                    # 预测
                    train_pred = model.predict(X_train)
                    valid_pred = model.predict(X_valid)
                    
                    # 评估
                    train_score = self._calculate_score(y_train, train_pred)
                    valid_score = self._calculate_score(y_valid, valid_pred)
                    
                    label_results[model_type] = {
                        'model': model,
                        'train_score': train_score,
                        'valid_score': valid_score,
                        'train_pred': train_pred,
                        'valid_pred': valid_pred
                    }
                    
                    print(f"  {model_type}: train_score={train_score:.4f}, valid_score={valid_score:.4f}")
                    
                except Exception as e:
                    print(f"  Error training {model_type} for {label_col}: {e}")
                    continue
            
            results[label_col] = label_results
        
        self.models = results
        return results
    
    def _train_models_fallback(self,
                              features_df: pd.DataFrame,
                              labels_df: pd.DataFrame,
                              train_ratio: float,
                              valid_ratio: float,
                              model_types: List[str]) -> Dict[str, Any]:
        """使用sklearn作为fallback的模型训练"""
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
        except ImportError:
            print("sklearn not available for fallback")
            return {}
        
        results = {}
        
        # 数据分割
        n_samples = len(features_df)
        train_end = int(n_samples * train_ratio)
        valid_end = int(n_samples * (train_ratio + valid_ratio))
        
        train_features = features_df.iloc[:train_end]
        valid_features = features_df.iloc[train_end:valid_end]
        
        # 模型映射
        model_map = {
            'lgb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression()
        }
        
        for label_col in labels_df.columns:
            print(f"Training fallback models for label: {label_col}")
            
            train_labels = labels_df[label_col].iloc[:train_end]
            valid_labels = labels_df[label_col].iloc[train_end:valid_end]
            
            # 移除缺失值
            train_mask = ~(train_labels.isna() | train_features.isna().any(axis=1))
            valid_mask = ~(valid_labels.isna() | valid_features.isna().any(axis=1))
            
            if train_mask.sum() < 100:
                continue
            
            label_results = {}
            
            for model_type in model_types:
                if model_type not in model_map:
                    continue
                
                try:
                    model = model_map[model_type]
                    
                    X_train = train_features[train_mask]
                    y_train = train_labels[train_mask]
                    X_valid = valid_features[valid_mask]
                    y_valid = valid_labels[valid_mask]
                    
                    model.fit(X_train, y_train)
                    
                    train_pred = model.predict(X_train)
                    valid_pred = model.predict(X_valid)
                    
                    train_score = r2_score(y_train, train_pred)
                    valid_score = r2_score(y_valid, valid_pred)
                    
                    label_results[model_type] = {
                        'model': model,
                        'train_score': train_score,
                        'valid_score': valid_score,
                        'train_pred': train_pred,
                        'valid_pred': valid_pred
                    }
                    
                    print(f"  {model_type}: train_score={train_score:.4f}, valid_score={valid_score:.4f}")
                    
                except Exception as e:
                    print(f"  Error training {model_type}: {e}")
                    continue
            
            results[label_col] = label_results
        
        self.models = results
        return results
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算模型评分"""
        try:
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
        except:
            # 简单的相关系数作为fallback
            return np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    
    def predict(self, 
               features_df: pd.DataFrame,
               label_names: Optional[List[str]] = None,
               model_type: str = 'lgb') -> Dict[str, np.ndarray]:
        """
        使用训练好的模型进行预测
        
        Args:
            features_df: 特征数据
            label_names: 要预测的标签名称列表
            model_type: 模型类型
            
        Returns:
            预测结果字典
        """
        
        if not self.models:
            raise ValueError("No trained models available. Please train models first.")
        
        predictions = {}
        
        if label_names is None:
            label_names = list(self.models.keys())
        
        for label_name in label_names:
            if label_name not in self.models:
                print(f"No model found for label: {label_name}")
                continue
            
            if model_type not in self.models[label_name]:
                print(f"No {model_type} model found for label: {label_name}")
                continue
            
            try:
                model = self.models[label_name][model_type]['model']
                pred = model.predict(features_df)
                predictions[label_name] = pred
                
            except Exception as e:
                print(f"Error predicting {label_name} with {model_type}: {e}")
                continue
        
        self.predictions = predictions
        return predictions
    
    def save_models(self, filepath: str):
        """保存训练好的模型"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
    
    def load_models(self, filepath: str):
        """加载训练好的模型"""
        with open(filepath, 'rb') as f:
            self.models = pickle.load(f)
    
    def get_feature_importance(self, 
                              label_name: str, 
                              model_type: str = 'lgb',
                              top_k: int = 20) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            label_name: 标签名称
            model_type: 模型类型
            top_k: 返回前k个重要特征
            
        Returns:
            特征重要性字典
        """
        
        if label_name not in self.models or model_type not in self.models[label_name]:
            return {}
        
        model = self.models[label_name][model_type]['model']
        
        try:
            if hasattr(model, 'feature_importances_'):
                # sklearn模型
                importances = model.feature_importances_
                feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importances))]
            elif hasattr(model, 'get_feature_importance'):
                # qlib模型
                importances = model.get_feature_importance()
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            else:
                return {}
            
            # 创建特征重要性字典并排序
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # 返回前k个
            return dict(list(sorted_importance.items())[:top_k])
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}
    
    def train_models_numpy(self,
                          features_array: np.ndarray,
                          labels_array: np.ndarray,
                          feature_names: List[str],
                          label_names: List[str],
                          train_ratio: float = 0.7,
                          valid_ratio: float = 0.15,
                          model_types: List[str] = ['lgb']) -> Dict[str, Any]:
        """
        使用numpy数组训练模型
        
        Args:
            features_array: 特征数组
            labels_array: 标签数组
            feature_names: 特征名称
            label_names: 标签名称
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            model_types: 模型类型列表
            
        Returns:
            训练结果字典
        """
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
        except ImportError:
            print("sklearn not available")
            return {}
        
        results = {}
        
        # 数据分割
        n_samples = len(features_array)
        train_end = int(n_samples * train_ratio)
        valid_end = int(n_samples * (train_ratio + valid_ratio))
        
        X_train = features_array[:train_end]
        X_valid = features_array[train_end:valid_end]
        X_test = features_array[valid_end:]
        
        # 模型映射
        model_map = {
            'lgb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression()
        }
        
        # 为每个标签训练模型
        for i, label_name in enumerate(label_names):
            print(f"Training models for label: {label_name}")
            
            if i >= labels_array.shape[1]:
                continue
                
            y_train = labels_array[:train_end, i]
            y_valid = labels_array[train_end:valid_end, i]
            y_test = labels_array[valid_end:, i]
            
            # 移除缺失值
            train_mask = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
            valid_mask = ~np.isnan(y_valid) & ~np.isnan(X_valid).any(axis=1)
            
            if train_mask.sum() < 100:
                print(f"Insufficient training samples for {label_name}, skipping...")
                continue
            
            label_results = {}
            
            for model_type in model_types:
                if model_type not in model_map:
                    continue
                
                try:
                    model = model_map[model_type]
                    
                    X_train_clean = X_train[train_mask]
                    y_train_clean = y_train[train_mask]
                    X_valid_clean = X_valid[valid_mask]
                    y_valid_clean = y_valid[valid_mask]
                    
                    # 训练模型
                    model.fit(X_train_clean, y_train_clean)
                    
                    # 预测
                    train_pred = model.predict(X_train_clean)
                    valid_pred = model.predict(X_valid_clean) if len(X_valid_clean) > 0 else []
                    
                    # 评估
                    train_score = r2_score(y_train_clean, train_pred)
                    valid_score = r2_score(y_valid_clean, valid_pred) if len(valid_pred) > 0 else 0.0
                    
                    label_results[model_type] = {
                        'model': model,
                        'train_score': train_score,
                        'valid_score': valid_score,
                        'train_pred': train_pred,
                        'valid_pred': valid_pred
                    }
                    
                    print(f"  {model_type}: train_score={train_score:.4f}, valid_score={valid_score:.4f}")
                    
                except Exception as e:
                    print(f"  Error training {model_type} for {label_name}: {e}")
                    continue
            
            results[label_name] = label_results
        
        self.models = results
        return results
    
    def predict_numpy(self, 
                     features_array: np.ndarray,
                     label_names: Optional[List[str]] = None,
                     model_type: str = 'lgb') -> Dict[str, np.ndarray]:
        """
        使用numpy数组进行预测
        
        Args:
            features_array: 特征数组
            label_names: 要预测的标签名称列表
            model_type: 模型类型
            
        Returns:
            预测结果字典
        """
        
        if not self.models:
            raise ValueError("No trained models available. Please train models first.")
        
        predictions = {}
        
        if label_names is None:
            label_names = list(self.models.keys())
        
        for label_name in label_names:
            if label_name not in self.models:
                print(f"No model found for label: {label_name}")
                continue
            
            if model_type not in self.models[label_name]:
                print(f"No {model_type} model found for label: {label_name}")
                continue
            
            try:
                model = self.models[label_name][model_type]['model']
                pred = model.predict(features_array)
                predictions[label_name] = pred
                
            except Exception as e:
                print(f"Error predicting {label_name} with {model_type}: {e}")
                continue
        
        self.predictions = predictions
        return predictions