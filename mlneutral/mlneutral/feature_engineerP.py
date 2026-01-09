
import polars as pl
import numpy as np
from scipy.stats import norm, kurtosis
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler
from .types import List, PathType
import pickle
import joblib
import os


def pl_compute_sample_weights(tname):
    # prepare sw
    tret = pl.col(tname).abs()
    tret = (tret/ tret.quantile(0.99))* 100
    tret = pl.when(tret > 100).then(tret.sqrt() * 10).otherwise(tret)
    return tret.cast(int) + 1

def compute_sample_weights(y):
    # prepare sw
    tret = np.abs(y)
    tret = (tret/ np.quantile(tret, 0.99))* 100
    tret[tret > 100] = np.sqrt(tret[tret > 100]) * 10
    return tret.astype(int) + 1

#%% transform targets
def generate_targets(
    data: pl.DataFrame,
    dtname: str,
    pxname: str,
    horizons: List[int],
    save_ecdf: bool = None,
    outputfolder: str = None,
    use_relative: bool = True
    ) -> pl.DataFrame:
    data = data.sort(dtname).with_columns(
            ((pl.col(pxname).shift(-j)-pl.col(pxname))/pl.col(pxname)).alias(f"ret_T{j}") for j in horizons
        )
    if use_relative:
        #transform
        for j in horizons:
            col = f"ret_T{j}"
            dropna_col = data.select(col).drop_nulls().to_series().to_numpy()
            if kurtosis(dropna_col) > 5:    
                print(f"[debug]: target horizon {j} ecdf transformed")
                # Append the two buffer values.
                extended = np.concatenate([dropna_col, np.array([np.inf])])
                # Build the ECDF using the extended data.
                ecdf = ECDF(extended)
                # Transform the training data: compute ECDF on original non-null data.
                temp_factor = ecdf(dropna_col)
                norm_factor = norm.ppf(temp_factor)
                # Replace non-null entries in the train column with the transformed values.
                train_col_vals = data[col].to_numpy().copy()
                non_null_idx = np.where(~np.isnan(train_col_vals))[0]
                train_col_vals[non_null_idx] = norm_factor
                data = data.with_columns(pl.Series(name=col, values=train_col_vals))
                if save_ecdf and outputfolder is not None:
                    with open(f'{outputfolder}/{col}.pkl', 'wb') as f:
                        pickle.dump(ecdf, f)
    data = data.with_columns([
        pl.col(f"ret_T{j}").cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(f"ret_T{j}")
        for j in horizons
    ])
    return data

#%% transform features

def transformFT(
    trainDF: pl.DataFrame,
    testDF: pl.DataFrame,
    features: List[str],
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    ts_lookback: int = 336,
    save_scaler: bool = False,
    outputfolder: str = None,
    max_category: int = 100,
    valid_row_pct: float = 0.6,
    clip_zscore: float = 5.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Transform features for cross-sectional stock return prediction.
    
    Pipeline: TS Z-score (per symbol, lookback window) → Pooled StandardScaler
    
    Parameters
    ----------
    trainDF, testDF : pl.DataFrame
        Train and test dataframes
    features : List[str]
        Feature column names to transform
    datetime_col : str
        Column name for timestamp
    symbol_col : str
        Column name for stock symbol (used for time-series grouping)
    ts_lookback : int
        Lookback window for time-series z-score (default 336)
    save_scaler : bool
        Whether to save scaler and feature info
    outputfolder : str
        Where to save artifacts
    max_category : int
        Features with <= this many unique values are treated as categorical (not scaled)
    valid_row_pct : float
        Minimum fraction of non-null features required to keep a row
    clip_zscore : float
        Clip z-scores to [-clip, +clip] to handle outliers
        
    Returns
    -------
    trainDF, testDF : transformed DataFrames
    """
    
    # --- Step 1: Clean infinities ---
    trainDF = trainDF.with_columns([
        pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None})
        for c in features
    ])
    if testDF.height > 0:
        testDF = testDF.with_columns([
            pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None})
            for c in features
        ])
    
    # --- Step 2: Filter rows with too many nulls ---
    null_threshold = int((1 - valid_row_pct) * len(features))
    trainDF = trainDF.with_columns(
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in features]).alias("_null_count")
    ).filter(pl.col("_null_count") <= null_threshold).drop("_null_count")
    
    # --- Step 3: Identify categorical vs continuous features ---
    cate_feats = [f for f in features if trainDF[f].n_unique() <= max_category]
    cont_feats = [f for f in features if f not in cate_feats]
    
    print(f"[transformFT] Categorical: {len(cate_feats)}, Continuous: {len(cont_feats)}")
    print(f"[transformFT] Pipeline: TS Z-score (lookback={ts_lookback}) → Pooled StandardScaler")
    
    # --- Step 4: Time-series Z-score (per symbol, rolling window) ---
    # Sort by symbol and datetime for proper rolling calculation
    trainDF = trainDF.sort([symbol_col, datetime_col])
    
    trainDF = trainDF.with_columns([
        (
            (pl.col(f) - pl.col(f).rolling_mean(window_size=ts_lookback, min_periods=1).over(symbol_col)) /
            (pl.col(f).rolling_std(window_size=ts_lookback, min_periods=1).over(symbol_col) + 1e-8)
        ).clip(-clip_zscore, clip_zscore).alias(f)
        for f in cont_feats
    ])
    
    if testDF.height > 0:
        testDF = testDF.sort([symbol_col, datetime_col])
        testDF = testDF.with_columns([
            (
                (pl.col(f) - pl.col(f).rolling_mean(window_size=ts_lookback, min_periods=1).over(symbol_col)) /
                (pl.col(f).rolling_std(window_size=ts_lookback, min_periods=1).over(symbol_col) + 1e-8)
            ).clip(-clip_zscore, clip_zscore).alias(f)
            for f in cont_feats
        ])
    
    # --- Step 5: Pooled StandardScaler normalization ---
    # Save null positions
    null_masks_train = {col: trainDF[col].is_null().to_numpy() for col in cont_feats}
    
    train_array = trainDF.select(cont_feats).fill_null(0).to_numpy().astype(np.float32)
    
    X_scaler = StandardScaler()
    scaled_train = X_scaler.fit_transform(train_array).astype(np.float32)
    
    # Update trainDF with scaled values, preserving nulls
    for i, col in enumerate(cont_feats):
        new_vals = scaled_train[:, i]
        new_vals = np.where(null_masks_train[col], np.nan, new_vals)
        trainDF = trainDF.with_columns(pl.Series(name=col, values=new_vals))
    
    # Transform test data
    if testDF.height > 0:
        null_masks_test = {col: testDF[col].is_null().to_numpy() for col in cont_feats}
        test_array = testDF.select(cont_feats).fill_null(0).to_numpy().astype(np.float32)
        scaled_test = X_scaler.transform(test_array).astype(np.float32)
        
        for i, col in enumerate(cont_feats):
            new_vals = scaled_test[:, i]
            new_vals = np.where(null_masks_test[col], np.nan, new_vals)
            testDF = testDF.with_columns(pl.Series(name=col, values=new_vals))
    
    # --- Step 6: Save artifacts ---
    if save_scaler and outputfolder:
        os.makedirs(outputfolder, exist_ok=True)
        pickle.dump(X_scaler, open(f'{outputfolder}/X_scaler.pkl', 'wb'))
        pickle.dump(features, open(f'{outputfolder}/allfeatures_order.pkl', 'wb'))
        pickle.dump({
            'cont_feats': cont_feats,
            'cate_feats': cate_feats,
            'ts_lookback': ts_lookback,
            'clip_zscore': clip_zscore,
        }, open(f'{outputfolder}/feature_info.pkl', 'wb'))
        print(f"[transformFT] Saved artifacts to {outputfolder}")
    
    return trainDF, testDF
    
# selection
# compute corr, screening the group

def feature_selection_best(
        df: pl.DataFrame, 
        features: List[str],
        target_col: str, 
        corr_threshold: float = 0.5
    ) -> pl.DataFrame:
    """
        特征选择函数
        1. 计算特征间相关性，去除高相关性特征
        2. 选择与目标变量最相关的特征
        参数：
        - df: 包含特征和目标变量的DataFrame
        - target_col: 目标变量列名
        - corr_threshold: 特征间相关性阈值，默认0.5
        返回：
        - 筛选后的DataFrame
    """
    # 计算相关性矩阵
    conditions = [~pl.col(col).is_nan() for col in features + [target_col]]
    # Combine the conditions using reduce to form a single filter expression
    corr_matrix = df.filter(pl.reduce(lambda a, b: a & b, conditions)).select(features+[target_col]).corr()
    # 存储待删除特征
    to_drop = set()
    # 筛选高相关特征对
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > corr_threshold:
                # 计算与目标的相关性
                corr_i = abs(corr_matrix[i, -1])  # 最后一列是目标
                corr_j = abs(corr_matrix[j, -1])
                # 保留相关性更高的特征
                if corr_i > corr_j:
                    to_drop.add(features[j])
                else:
                    to_drop.add(features[i])
    # 执行特征筛选
    selected_features = [f for f in features if f not in to_drop]
    # 返回筛选后的DataFrame（包含目标列）
    return df.select([target_col] + selected_features)

def feature_selection_avg(
        df: pl.DataFrame, 
        features: List[str],
        target_col: str, 
        corr_threshold: float = 0.5,
        save_grouping: bool = False, outputfolder: str = None
    ) -> pl.DataFrame:
    """
        将高相关特征聚类并生成平均特征
        
        参数：
        - df: 输入DataFrame（仅包含数值特征）
        - threshold: 特征相关性阈值，默认0.5
        
        返回：
        - 包含新生成特征和原始非相关特征的DataFrame
    """
    # 计算相关性矩阵
    conditions = [~pl.col(col).is_nan() for col in features + [target_col]]
    # Combine the conditions using reduce to form a single filter expression
    corr_matrix = df.filter(pl.reduce(lambda a, b: a & b, conditions)).select(features+[target_col]).corr()
    # 使用并查集寻找相关特征组
    parent = list(range(len(features)))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
    # 遍历上三角矩阵建立连接
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix[i,j]) >= corr_threshold:
                union(i, j)
    
    # 构建特征分组
    clusters = {}
    for idx in range(len(features)):
        root = find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(features[idx])
    
    # 过滤出需要合并的组（至少2个特征）
    merge_groups = [g for g in clusters.values() if len(g) > 1]
    
    # 生成新特征
    new_df = df.clone()
    for group_id, subfeats in enumerate(merge_groups):
        # 计算平均特征
        new_df = new_df.with_columns(
            pl.mean_horizontal(subfeats).alias(f"cluster_{subfeats[0]}")
        )
        # 移除原始特征（可选）
        new_df = new_df.drop(subfeats)
    if save_grouping and outputfolder is not None:
        with open(f'{outputfolder}/feature_mapping.pkl', 'wb') as f:
            pickle.dump(merge_groups, f)
    return new_df


def feature_selection_filter_avg(
        df: pl.DataFrame, 
        features: List[str],
        target_col: str, 
        corr_threshold: float = 0.5,
        ic_threshold: float = 0.005,
        save_grouping: bool = False, outputfolder: str = None
    ) -> pl.DataFrame:
    """
        将高相关特征聚类并生成平均特征
        
        参数：
        - df: 输入DataFrame（仅包含数值特征）
        - threshold: 特征相关性阈值，默认0.5
        
        返回：
        - 包含新生成特征和原始非相关特征的DataFrame
    """
    # 计算相关性矩阵
    conditions = [~pl.col(col).is_nan() for col in features + [target_col]]
    # Combine the conditions using reduce to form a single filter expression
    corr_matrix = df.filter(pl.reduce(lambda a, b: a & b, conditions)).select(features+[target_col]).corr()
    # 使用并查集寻找相关特征组
    parent = list(range(len(features)))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
    # 遍历上三角矩阵建立连接
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix[i,j]) >= corr_threshold:
                union(i, j)
    
    # 构建特征分组
    clusters = {}
    for idx in range(len(features)):
        root = find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(features[idx])
    
    # 过滤出需要合并的组（至少2个特征）
    merge_groups = [g for g in clusters.values() if len(g) > 1]
    
    # 生成新特征
    new_df = df.clone()
    for group_id, rsubfeats in enumerate(merge_groups):
        subfeats = []
        maxic = -1
        maxfeat = rsubfeats[0]
        for feat in rsubfeats:
            thisic = abs(corr_matrix[-1,feat])
            if thisic > maxic:
                maxic = thisic
                maxfeat = feat
            if thisic >= ic_threshold:
                subfeats.append(feat)
        if len(subfeats) == 0:
            subfeats = [maxfeat]
        # 计算平均特征
        new_df = new_df.with_columns(
            pl.mean_horizontal(subfeats).alias(f"cluster_{subfeats[0]}")
        )
        # 移除原始特征（可选）
        new_df = new_df.drop(rsubfeats)
        merge_groups[group_id] = subfeats
    if save_grouping and outputfolder is not None:
        with open(f'{outputfolder}/feature_mapping.pkl', 'wb') as f:
            pickle.dump(merge_groups, f)
    return new_df


def feature_selection_2step(
        df: pl.DataFrame, 
        features: List[str],
        target_col: str, 
        corr_drop_threshold: float = 0.8,
        corr_merge_threshold: float = 0.6,
        save_grouping: bool = False, outputfolder: str = None
    ) -> pl.DataFrame:
    """
        将高相关特征聚类并生成平均特征
        
        参数：
        - df: 输入DataFrame（仅包含数值特征）
        - threshold: 特征相关性阈值，默认0.5
        
        返回：
        - 包含新生成特征和原始非相关特征的DataFrame
    """
    ## drop nonsense
    #todrop = [i for i in features if df[i].std() == 0]
    #df = df.drop(todrop)
    # 计算相关性矩阵
    conditions = [~pl.col(col).is_nan() for col in features + [target_col]]
    # Combine the conditions using reduce to form a single filter expression
    corr_matrix = df.filter(pl.reduce(lambda a, b: a & b, conditions)).select(features+[target_col]).corr()
    # 存储待删除特征
    to_drop = set()
    # 筛选高相关特征对
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > corr_drop_threshold:
                # 计算与目标的相关性
                corr_i = abs(corr_matrix[i, -1])  # 最后一列是目标
                corr_j = abs(corr_matrix[j, -1])
                # 保留相关性更高的特征
                if corr_i > corr_j:
                    to_drop.add(features[j])
                else:
                    to_drop.add(features[i])
    new_df = df.drop(to_drop)
    selected_features = [f for f in features if f not in to_drop]
    # 使用并查集寻找相关特征组
    parent = list(range(len(selected_features)))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
    # 遍历上三角矩阵建立连接
    conditions = [~pl.col(col).is_nan() for col in selected_features + [target_col]]
    corr_matrix = new_df.filter(pl.reduce(lambda a, b: a & b, conditions)).select(selected_features+[target_col]).corr()
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            if abs(corr_matrix[i,j]) >= corr_merge_threshold:
                union(i, j)
    
    # 构建特征分组
    clusters = {}
    for idx in range(len(selected_features)):
        root = find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append((selected_features[idx], np.sign(corr_matrix[root,idx])))
    
    # 过滤出需要合并的组（至少2个特征）
    merge_groups = [g for g in clusters.values() if len(g) > 1]
    
    # 生成新特征
    for group_id, subfeats in enumerate(merge_groups):
        # 计算平均特征
        new_df = new_df.with_columns(
            pl.mean_horizontal([pl.col(cname) * csign for cname,csign in subfeats]).alias(f"cluster_{subfeats[0][0]}")
        )
        # 移除原始特征（可选）
        new_df = new_df.drop([i[0] for i in subfeats])
        merge_groups[group_id] = subfeats
    if save_grouping and outputfolder is not None:
        with open(f'{outputfolder}/feature_mapping.pkl', 'wb') as f:
            pickle.dump(merge_groups, f)
    return new_df

    
def process_test_data(
    testDF: pl.DataFrame, 
    allfeatures: List[str],
    features: List[str],
    scaler_file: str,
    feature_mapping_file: str,
    feature_info_file: str,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """
    处理测试数据，步骤包括：
      1. 清理无穷值
      2. 时间序列Z-score标准化（每个symbol，滚动窗口）
      3. 使用训练时保存的StandardScaler进行pooled标准化
      4. 根据训练时保存的合并组（merge_groups），将高度相关的特征取均值合并

    参数:
      testDF: 待处理的测试数据（polars DataFrame）
      allfeatures: 原始特征列名称列表（合并前，用于scaler）
      features: 最终需要处理的特征列名称列表（合并后）
      scaler_file: 存储StandardScaler的文件路径
      feature_mapping_file: 存储特征合并映射（merge_groups）的pickle文件路径
      feature_info_file: 存储特征信息（cont_feats, cate_feats, ts_lookback）的pickle文件路径
      datetime_col: 时间列名称
      symbol_col: 股票代码列名称

    返回:
      处理后的测试数据（polars DataFrame）
    """
    # 复制一份数据用于处理
    new_test = testDF.clone()
    
    # --- 加载特征信息 ---
    try:
        with open(feature_info_file, "rb") as f:
            feature_info = pickle.load(f)
        cont_feats = feature_info.get('cont_feats', [])
        cate_feats = feature_info.get('cate_feats', [])
        ts_lookback = feature_info.get('ts_lookback', 336)
        clip_zscore = feature_info.get('clip_zscore', 3.0)
        # 只保留在allfeatures中的
        cont_feats = [f for f in cont_feats if f in allfeatures]
        cate_feats = [f for f in cate_feats if f in allfeatures]
    except Exception as e:
        print(f"加载特征信息文件 {feature_info_file} 失败: {e}")
        raise e
    
    # --- 第一步：清理无穷值 ---
    for col in allfeatures:
        if col in new_test.columns:
            new_test = new_test.with_columns(
                pl.col(col).cast(pl.Float64).replace({np.inf: None, -np.inf: None}).alias(col)
            )
    
    # --- 第二步：时间序列Z-score（每个symbol，滚动窗口） ---
    new_test = new_test.sort([symbol_col, datetime_col])
    
    new_test = new_test.with_columns([
        (
            (pl.col(f) - pl.col(f).rolling_mean(window_size=ts_lookback, min_periods=1).over(symbol_col)) /
            (pl.col(f).rolling_std(window_size=ts_lookback, min_periods=1).over(symbol_col) + 1e-8)
        ).clip(-clip_zscore, clip_zscore).alias(f)
        for f in cont_feats
    ])
    
    # --- 第三步：使用StandardScaler进行pooled标准化 ---
    try:
        X_scaler = pickle.load(open(scaler_file, "rb"))
    except Exception as e:
        print(f"加载scaler文件 {scaler_file} 失败: {e}")
        raise e
    
    if cont_feats:
        # 保存null位置
        null_masks = {col: new_test[col].is_null().to_numpy() for col in cont_feats}
        
        # 填充null为0进行transform
        test_array = new_test.select(cont_feats).fill_null(0).to_numpy().astype(np.float32)
        scaled_test = X_scaler.transform(test_array).astype(np.float32)
        
        # 更新特征值，保留原始null位置
        for i, col in enumerate(cont_feats):
            new_vals = scaled_test[:, i]
            new_vals = np.where(null_masks[col], np.nan, new_vals)
            new_test = new_test.with_columns(pl.Series(name=col, values=new_vals))
    
    # --- 第四步：合并高度相关特征 ---
    try:
        with open(feature_mapping_file, "rb") as f:
            merge_groups = pickle.load(f)
    except Exception as e:
        print(f"加载特征映射文件 {feature_mapping_file} 失败: {e}")
        merge_groups = []
    
    if merge_groups:
        for group in merge_groups:
            # 对分组内的特征取水平均值，新特征名为 "cluster_{group[0][0]}"
            new_test = new_test.with_columns(
                pl.mean_horizontal([pl.col(cname) * csign for cname, csign in group]).alias(f"cluster_{group[0][0]}")
            )
            # 删除原始的分组合并特征
            new_test = new_test.drop([i[0] for i in group])
    
    # --- 第五步：最终填充null为0 ---
    new_test = new_test.with_columns(
        pl.col(f).fill_nan(None).fill_null(0.0).alias(f) for f in features if f in new_test.columns
    )
    
    return new_test

    
def downsample_sequences(features, target, factor=4):
    """
    Downsamples a 3D features array and a 2D target array along the temporal dimension (T).
    Args:
        features (numpy.ndarray): 3D input array of shape (T, seq_len, N).
        target (numpy.ndarray): 2D target array of shape (T, 1).
        factor (int): The downsampling factor for the temporal dimension.
    Returns:
        tuple: Downsampled features (new_T, seq_len, N) and target (new_T, 1).
    """
    # Truncate to ensure T is divisible by the factor
    T = features.shape[0]
    new_T = T // factor
    features_trunc = features[:new_T * factor, :, :]
    target_trunc = target[:new_T * factor]
    # Downsample features (3D)
    features_reshaped = features_trunc.reshape(new_T, factor, features.shape[1], features.shape[2])
    features_down = features_reshaped[:, 0, :, :]    
    # Downsample target (2D)
    target_reshaped = target_trunc.reshape(new_T, factor)
    target_down = target_reshaped[:, 0]    
    return features_down, target_down


def downsample_sequences2(features, target, factor=4, shift=0):
    """
    Downsamples a 3D features array and a 2D target array along the temporal dimension (T).
    Args:
        features (numpy.ndarray): 3D input array of shape (T, seq_len, N).
        target (numpy.ndarray): 2D target array of shape (T, 1).
        factor (int): The downsampling factor for the temporal dimension.
        shift (int): The offset within each window to sample from (default: 0).
    Returns:
        tuple: Downsampled features (new_T, seq_len, N) and target (new_T, 1).
    """
    # Ensure shift is within [0, factor-1]
    shift = shift % factor
    
    # Truncate to ensure T is divisible by the factor
    T = features.shape[0]
    new_T = T // factor
    features_trunc = features[:new_T * factor, :, :]
    target_trunc = target[:new_T * factor]
    
    # Downsample features (3D)
    features_reshaped = features_trunc.reshape(new_T, factor, features.shape[1], features.shape[2])
    features_down = features_reshaped[:, shift, :, :]  # Select shift-th element from each window
    
    # Downsample target (2D)
    target_reshaped = target_trunc.reshape(new_T, factor)
    target_down = target_reshaped[:, shift]
    
    return features_down, target_down.reshape(-1, 1)  # Ensure target is 2D


def generate_test_targets(
        testDF: pl.DataFrame, 
        save_path: PathType,
        ecdf_folder: PathType,
        name: str, 
        pxname: str,
        dtname: str, 
        horizon: int
    ) -> pl.DataFrame:
    """
    Load the saved y_transformer and transform y_test using it.

    Args:
        y_test (array-like): The test target values to transform.
        save_path (str): Directory where the transformer was saved.
        name (str): Name used when saving the transformer.
        target (str): Target name used when saving the transformer.

    Returns:
        y_test_transformed (array-like): Transformed y_test.
    """
    target = f'ret_T{horizon}'
    testDF = testDF.sort(dtname).with_columns(
            ((pl.col(pxname).shift(-horizon)-pl.col(pxname))/pl.col(pxname)).alias(target)
        ).with_columns(
        pl.col(target).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(target)
    )
    ecdf_file = f"{ecdf_folder}/{target}.pkl"
    if os.path.exists(ecdf_file):
        print(f"[debug]: target horizon {horizon} ecdf transformed")
        with open(ecdf_file, "rb") as f:
            ecdf = pickle.load(f)
        test_nonnull = testDF.select(target).drop_nulls().to_series().to_numpy()
        if test_nonnull.size > 0:
            # 对非空数据进行ECDF转换，再用正态分布逆函数处理
            temp_factor_test = ecdf(test_nonnull)
            norm_factor_test = norm.ppf(temp_factor_test)
            # 替换当前列中的非空值
            test_col_vals = testDF[target].to_numpy().copy()
            non_null_idx_test = np.where(~np.isnan(test_col_vals))[0]
            test_col_vals[non_null_idx_test] = norm_factor_test
            testDF = testDF.with_columns(pl.Series(name=target, values=test_col_vals))
            testDF = testDF.with_columns(
                    pl.col(target).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(target)
                    ) 
    transformer_path = os.path.join(save_path, '_'.join(['y_transformer', name, target]))
    y_transformer = joblib.load(transformer_path)
    # Transform the test set
    test_nonnull = testDF.select(target).drop_nulls().to_series().to_numpy()
    test_col_vals = testDF[target].to_numpy().copy()
    non_null_idx_test = np.where(~np.isnan(test_col_vals))
    y_test_transformed = y_transformer.transform(test_nonnull.reshape(-1,1))[:,0]
    test_col_vals[non_null_idx_test] = y_test_transformed
    testDF = testDF.with_columns(pl.Series(name='label_' + target, values=test_col_vals))
    return testDF

def screen_features(
        df: pl.DataFrame, 
        features: List[str],
        target_col: str, 
        corr_threshold: float = 0.01
    ) -> pl.DataFrame:
    """
        特征选择函数
        1. 计算特征间相关性，去除高相关性特征
        2. 选择与目标变量最相关的特征
        参数：
        - df: 包含特征和目标变量的DataFrame
        - target_col: 目标变量列名
        - corr_threshold: 特征间相关性阈值，默认0.5
        返回：
        - 筛选后的DataFrame
    """
    # 计算相关性矩阵
    conditions = [~pl.col(col).is_nan() for col in features + [target_col]]
    # Combine the conditions using reduce to form a single filter expression
    corr_matrix = df.filter(conditions).select(features+[target_col]).corr()
    # 存储待删除特征
    keep = []
    # 筛选高相关特征对
    for i in range(len(corr_matrix) - 1):
        # 计算与目标的相关性
        corr_i = abs(corr_matrix[i, -1])  # 最后一列是目标
        if corr_i > corr_threshold:
            keep.append(features[i])
    # 返回筛选后的DataFrame（包含目标列）
    return keep
