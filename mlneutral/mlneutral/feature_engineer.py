
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
def transformFT(trainDF: pl.DataFrame, testDF: pl.DataFrame, features: List, save_ecdf: bool = False, outputfolder: str = None, max_category: int = 100):
    # Replace positive and negative infinity with NaN in both DataFrames.
    trainDF = trainDF.with_columns([
        pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None})
        for c in features
    ])
    if testDF.height > 0:
        testDF = testDF.with_columns([
            pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None})
            for c in features
        ])
    # keep categorical
    cate_feats = [i for i in features if len(trainDF[i].unique()) <= max_category]    
    # Process each column.
    ct = 0
    for col in features:
        if col in cate_feats:
            continue
        # Get non-null values from the training DataFrame as a numpy array.
        dropna_col = trainDF.select(col).drop_nulls().to_series().to_numpy()
        if dropna_col.size == 0:
            continue  # Skip columns that are completely null.
        # Check if kurtosis is high.
        if kurtosis(dropna_col) > 5:
            # Compute buffer values.
            buff_high = dropna_col.max() + dropna_col.std()
            buff_low = dropna_col.min() - dropna_col.std()
            # Append the two buffer values.
            extended = np.concatenate([dropna_col, np.array([buff_high, buff_low])])
            # Build the ECDF using the extended data.
            ecdf = ECDF(extended)
            # Transform the training data: compute ECDF on original non-null data.
            temp_factor = ecdf(dropna_col)
            norm_factor = norm.ppf(temp_factor)
            # Replace non-null entries in the train column with the transformed values.
            train_col_vals = trainDF[col].to_numpy().copy()
            non_null_idx = np.where(~np.isnan(train_col_vals))[0]
            train_col_vals[non_null_idx] = norm_factor
            trainDF = trainDF.with_columns(pl.Series(name=col, values=train_col_vals))
            # If the test DataFrame has rows, process its column similarly.
            if testDF.height > 0:
                test_nonnull = testDF.select(col).to_series().to_numpy()
                if test_nonnull.size > 0:
                    temp_factor_test = ecdf(test_nonnull)
                    norm_factor_test = norm.ppf(temp_factor_test)
                    test_col_vals = testDF[col].to_numpy()
                    non_null_idx_test = np.where(~np.isnan(test_col_vals))[0]
                    test_col_vals[non_null_idx_test] = norm_factor_test
                    new_test = new_test.with_column(pl.Series(name=col, values=test_col_vals))
            # Optionally save the ECDF to a pickle file.
            if save_ecdf and outputfolder is not None:
                with open(f'{outputfolder}/{col}.pkl', 'wb') as f:
                    pickle.dump(ecdf, f)
            ct += 1
    print(f"[debug]: total ecdf transformed {ct}")
    trainDF = trainDF.with_columns([
        pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(c)
        for c in features
    ])
    with open(f'{outputfolder}/allfeatures_order.pkl', 'wb') as f:
        pickle.dump(features, f)
    # preprocessing X
    non_cate_feats = [i for i in features if i not in cate_feats]
    X_ = trainDF.select(non_cate_feats).to_numpy()
    X_scaler = StandardScaler()
    X_scaler.fit(X_)
    joblib.dump((X_scaler,non_cate_feats), os.path.join(outputfolder, 'X_scaler'))
    scaled_array = X_scaler.transform(X_)
    # 更新每个特征列为标准化后的值
    for i, col in enumerate(non_cate_feats):
        trainDF = trainDF.with_columns(pl.Series(name=col, values=scaled_array[:, i]))
    if testDF.height > 0:
        testDF = testDF.with_columns([
            pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(c)
            for c in features
        ])    
        data_array = testDF.select(non_cate_feats).to_numpy()
        scaled_array = X_scaler.transform(data_array)
        # 更新每个特征列为标准化后的值
        for i, col in enumerate(non_cate_feats):
            testDF = testDF.with_columns(pl.Series(name=col, values=scaled_array[:, i]))
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
    ecdf_folder: str,
    feature_mapping_file: str,
    scaler_file: str,
    outputfolder: str = None
    ) -> pl.DataFrame:
    """
    处理测试数据，步骤包括：transformFT
      1. 对每个特征利用训练时保存的ECDF进行转换（transformFT）。
      2. 根据训练时保存的合并组（merge_groups），将高度相关的特征取均值合并。
      3. 使用训练时保存的StandardScaler对特征进行标准化。

    参数:
      testDF: 待处理的测试数据（polars DataFrame）。
      features: 需要处理的特征列名称列表。
      ecdf_folder: 存储ECDF文件的文件夹路径，每个特征对应一个文件 "{ecdf_folder}/{col}.pkl"。
      feature_mapping_file: 存储特征合并映射（merge_groups）的pickle文件路径。
      scaler_file: 存储StandardScaler的文件路径。
      outputfolder: 如果需要保存ECDF或者特征映射文件，则保存的输出文件夹路径（可选）。
      save_ecdf: 是否保存ECDF（默认False）。
      save_grouping: 是否保存特征合并映射（默认False）。

    返回:
      处理后的测试数据（polars DataFrame）。
    """
    # 复制一份数据用于处理
    new_test = testDF.clone()
    # --- 第一步：对每个特征进行ECDF转换 ---
    ct = 0
    for col in allfeatures:
        new_test = new_test.with_columns(
            pl.col(col).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(col)
        )
        ecdf_file = f"{ecdf_folder}/{col}.pkl"
        if os.path.exists(ecdf_file):
            with open(ecdf_file, "rb") as f:
                ecdf = pickle.load(f)
        else:
            continue
        test_nonnull = new_test.select(col).drop_nulls().to_series().to_numpy()
        if test_nonnull.size > 0:
            # 对非空数据进行ECDF转换，再用正态分布逆函数处理
            temp_factor_test = ecdf(test_nonnull)
            norm_factor_test = norm.ppf(temp_factor_test)
            # 替换当前列中的非空值
            test_col_vals = new_test[col].to_numpy().copy()
            non_null_idx_test = np.where(~np.isnan(test_col_vals))[0]
            test_col_vals[non_null_idx_test] = norm_factor_test
            new_test = new_test.with_columns(pl.Series(name=col, values=test_col_vals))
        ct += 1
    new_test = new_test.with_columns([
            pl.col(c).cast(pl.Float64).replace({np.inf: None, -np.inf: None, np.nan: None}).alias(c)
            for c in allfeatures
        ])  
    print(f"[debug]: total ecdf transformed {ct}")
    new_test = new_test.with_columns(
        pl.sum_horizontal([
            pl.col(col).is_null().cast(pl.Int32) for col in allfeatures
        ]).alias("out_of_range")
    )
    # --- 第二步：标准化 ---
    try:
        X_scaler, non_cate_feats = joblib.load(scaler_file)
    except Exception as e:
        raise Exception(f"加载Scaler文件 {scaler_file} 失败: {e}")
    data_array = new_test.select(non_cate_feats).to_numpy()
    scaled_array = X_scaler.transform(data_array)
    # 更新每个特征列为标准化后的值
    for i, col in enumerate(non_cate_feats):
        new_test = new_test.with_columns(pl.Series(name=col, values=scaled_array[:, i]))
    # --- 第三步：合并高度相关特征 ---
    try:
        with open(feature_mapping_file, "rb") as f:
            merge_groups = pickle.load(f)
    except Exception as e:
        print(f"加载特征映射文件 {feature_mapping_file} 失败: {e}")
        merge_groups = []
    if merge_groups:
        # 复制数据，方便后续操作
        new_df = new_test.clone()
        for group in merge_groups:
            # 对分组内的特征取水平均值，新特征名为 "cluster_{group[0]}"
            new_df = new_df.with_columns(
                pl.mean_horizontal([pl.col(cname) * csign for cname,csign in group]).alias(f"cluster_{group[0][0]}")
            )
            # 可选：删除原始的分组合并特征
            new_df = new_df.drop([i[0] for i in group])
        new_test = new_df
    new_test = new_test.with_columns(
            pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
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
