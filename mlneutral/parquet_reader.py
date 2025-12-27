"""
Parallel Parquet Reader - 多线程版本
高效并行读取parquet文件的工具模块
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union
import polars as pl


def _extract_symbol(filename: str, symbol_prefix: Optional[str] = None, symbol_pattern: Optional[str] = None) -> Optional[str]:
    """
    从文件名提取symbol
    
    支持的格式:
        - perp_BTCUSDT.parquet -> BTCUSDT (prefix模式)
        - coin_BTCUSDT_2022H1.parquet -> BTCUSDT (pattern模式)
    
    Args:
        filename: 文件名 (不含路径)
        symbol_prefix: 前缀，如 "perp_"
        symbol_pattern: 正则模式，如 r"coin_(\w+)_\d{4}H\d"
    
    Returns:
        提取的symbol，匹配不到返回None
    """
    import re
    
    stem = Path(filename).stem
    
    # 优先使用正则模式
    if symbol_pattern:
        match = re.search(symbol_pattern, stem)
        if match:
            return match.group(1)
        return None  # 匹配不到返回None
    
    # 使用前缀模式
    if symbol_prefix and stem.startswith(symbol_prefix):
        return stem[len(symbol_prefix):]
    
    # 没有指定任何模式，返回整个stem
    if symbol_prefix is None and symbol_pattern is None:
        return stem
    
    return None  # 指定了模式但匹配不到


def _read_single_parquet(
    filepath: Path,
    schema: Optional[Dict] = None,
    schema_overrides: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    rename_index: Optional[str] = None,
    symbol_prefix: Optional[str] = None,
    symbol_pattern: Optional[str] = None,
    sort_by: Optional[str] = None,
) -> pl.DataFrame:
    """
    读取单个parquet文件 (worker函数)
    """
    # 从文件名提取symbol
    symbol = _extract_symbol(filepath.name, symbol_prefix, symbol_pattern)
    
    # 构建读取参数
    kwargs = {}
    if schema is not None:
        kwargs["schema"] = schema
    if schema_overrides is not None:
        kwargs["schema_overrides"] = schema_overrides
    if columns is not None:
        kwargs["columns"] = columns
    
    df = pl.read_parquet(filepath, **kwargs)
    
    # 处理pandas保存的index列
    if rename_index and "__index_level_0__" in df.columns:
        df = df.rename({"__index_level_0__": rename_index})
    
    # 如果没有symbol列，添加symbol列
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
    
    # 排序
    if sort_by and sort_by in df.columns:
        df = df.sort(sort_by)
    
    return df


def parallel_read_parquets(
    folder: str,
    pattern: str = "*.parquet",
    n_workers: int = None,
    schema: Optional[Dict] = None,
    schema_overrides: Optional[Dict] = None,
    columns: Optional[List[str]] = None,
    rename_index: Optional[str] = "timestamp",
    symbol_prefix: Optional[str] = None,
    symbol_pattern: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    return_dict: bool = False,
    sort_by: Optional[str] = None,
    verbose: bool = True,
) -> Union[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """
    多线程并行读取parquet文件
    
    Args:
        folder: 文件夹路径
        pattern: 文件匹配模式，默认 "*.parquet"
        n_workers: 线程数，默认 min(32, CPU核心数 * 2)
        schema: 完整schema定义 (所有列)
        schema_overrides: 部分列类型覆盖 (只改指定列)
        columns: 只读取指定列 (节省内存)
        rename_index: 将__index_level_0__重命名，设None禁用
        symbol_prefix: 从文件名提取symbol时去掉的前缀，如 "perp_"
        symbol_pattern: 正则模式提取symbol，如 r"coin_(\w+)_\d{4}H\d" (优先级高于prefix)
        symbols: 只读取指定的symbols列表，如 ["BTCUSDT", "ETHUSDT"]，None表示读取全部
        return_dict: True返回字典{symbol: df}，False返回合并的大DataFrame(默认)
        sort_by: 排序列名
        verbose: 是否打印进度
    
    Returns:
        合并后的DataFrame (默认) 或 Dict[symbol, DataFrame]
    
    Example:
        # perp_BTCUSDT.parquet 格式
        >>> df = parallel_read_parquets(
        ...     "/data/features",
        ...     pattern="perp_*.parquet",
        ...     symbol_prefix="perp_",
        ... )
        
        # coin_BTCUSDT_2022H1.parquet 格式
        >>> df = parallel_read_parquets(
        ...     "/data/features",
        ...     pattern="coin_*.parquet",
        ...     symbol_pattern=r"coin_(\w+)_\d{4}H\d",
        ... )
        
        # 只读取指定symbols
        >>> df = parallel_read_parquets(
        ...     "/data/features",
        ...     pattern="perp_*.parquet",
        ...     symbol_prefix="perp_",
        ...     symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        ... )
    """
    folder_path = Path(folder)
    files = sorted(folder_path.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")
    
    # 过滤掉不匹配symbol_prefix或symbol_pattern的文件
    if symbol_prefix is not None or symbol_pattern is not None:
        files = [
            f for f in files
            if _extract_symbol(f.name, symbol_prefix, symbol_pattern) is not None
        ]
        if not files:
            raise FileNotFoundError(f"No files matching symbol_prefix='{symbol_prefix}' or symbol_pattern='{symbol_pattern}'")
    
    # 过滤指定symbols
    if symbols is not None:
        symbols_set = set(symbols)
        files = [
            f for f in files
            if _extract_symbol(f.name, symbol_prefix, symbol_pattern) in symbols_set
        ]
        if not files:
            raise FileNotFoundError(f"No files found for symbols: {symbols}")
        if verbose:
            print(f"Filtered to {len(files)} files for {len(symbols_set)} symbols")
    
    if n_workers is None:
        n_workers = min(32, os.cpu_count() * 2, len(files))
    
    if verbose:
        print(f"Reading {len(files)} files with {n_workers} threads...")
    
    # 多线程读取
    dfs = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _read_single_parquet,
                f,
                schema,
                schema_overrides,
                columns,
                rename_index,
                symbol_prefix,
                symbol_pattern,
                sort_by,
            ): f
            for f in files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            filepath = futures[future]
            try:
                df = future.result()
                dfs.append(df)
                
                if verbose and i % 50 == 0:
                    print(f"  Loaded {i}/{len(files)} files...")
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                raise
    
    if verbose:
        print(f"Done. Loaded {len(dfs)} files.")
    
    # 合并成单个DataFrame
    combined = pl.concat(dfs)
    
    if sort_by and sort_by in combined.columns:
        combined = combined.sort([sort_by, "symbol"])
    
    # 返回字典或合并后的df
    if return_dict:
        return {sym: grp for sym, grp in combined.group_by("symbol")}
    
    return combined


def get_parquet_schema(filepath: str) -> Dict:
    """获取parquet文件的schema"""
    return pl.read_parquet_schema(filepath)


def make_float32_schema(schema: Dict) -> Dict:
    """将所有Float64转为Float32 (省内存)"""
    return {
        col: pl.Float32 if dtype == pl.Float64 else dtype
        for col, dtype in schema.items()
    }


def list_symbols(
    folder: str,
    pattern: str = "*.parquet",
    symbol_prefix: Optional[str] = None,
    symbol_pattern: Optional[str] = None,
) -> List[str]:
    """
    列出文件夹中所有可用的symbols
    
    Args:
        folder: 文件夹路径
        pattern: 文件匹配模式
        symbol_prefix: 前缀模式
        symbol_pattern: 正则模式
    
    Returns:
        排序后的symbols列表
    
    Example:
        >>> symbols = list_symbols("/data", pattern="perp_*.parquet", symbol_prefix="perp_")
        >>> print(symbols)
        ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', ...]
    """
    folder_path = Path(folder)
    files = folder_path.glob(pattern)
    
    symbols = set()
    for f in files:
        sym = _extract_symbol(f.name, symbol_prefix, symbol_pattern)
        if sym is not None:  # 过滤掉不匹配的文件
            symbols.add(sym)
    
    return sorted(symbols)


# ============ 使用示例 ============
if __name__ == "__main__":
    FOLDER = "/path/to/your/parquet/folder"
    
    # ===== 查看可用symbols =====
    symbols = list_symbols(FOLDER, pattern="perp_*.parquet", symbol_prefix="perp_")
    print(f"Available symbols: {symbols}")  # ['BTCUSDT', 'ETHUSDT', ...]
    
    # ===== 格式1: perp_BTCUSDT.parquet =====
    df = parallel_read_parquets(
        FOLDER,
        pattern="perp_*.parquet",
        n_workers=16,
        symbol_prefix="perp_",
        rename_index="timestamp",
    )
    
    # ===== 格式2: coin_BTCUSDT_2022H1.parquet =====
    # 同一个symbol的多个时间段文件会自动合并
    df = parallel_read_parquets(
        FOLDER,
        pattern="coin_*.parquet",
        n_workers=16,
        symbol_pattern=r"coin_(\w+)_\d{4}H\d",  # 正则提取symbol
        rename_index="timestamp",
        sort_by="timestamp",  # 建议排序，确保时间顺序
    )
    
    # ===== 只读取指定symbols =====
    df = parallel_read_parquets(
        FOLDER,
        pattern="perp_*.parquet",
        symbol_prefix="perp_",
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],  # 只读这3个
    )
    
    # coin格式也支持
    df = parallel_read_parquets(
        FOLDER,
        pattern="coin_*.parquet",
        symbol_pattern=r"coin_(\w+)_\d{4}H\d",
        symbols=["BTCUSDT", "ETHUSDT"],  # 会读取这两个symbol的所有时间段文件
        sort_by="timestamp",
    )
    
    # 按symbol筛选
    btc = df.filter(pl.col("symbol") == "BTCUSDT")
    eth = df.filter(pl.col("symbol") == "ETHUSDT")
    
    # 带schema_overrides
    df = parallel_read_parquets(
        FOLDER,
        pattern="perp_*.parquet",
        schema_overrides={
            "__index_level_0__": pl.Datetime("us"),
        },
        symbol_prefix="perp_",
    )
    
    # 只读取部分列
    df = parallel_read_parquets(
        FOLDER,
        pattern="perp_*.parquet",
        columns=["__index_level_0__", "buy_proportion_1h_perp", "buy_concentration_1h_perp"],
        symbol_prefix="perp_",
    )
    
    # 如果需要字典形式
    data_dict = parallel_read_parquets(
        FOLDER,
        pattern="perp_*.parquet",
        symbol_prefix="perp_",
        return_dict=True,
    )
    
    # 使用Float32省内存
    schema = get_parquet_schema(f"{FOLDER}/perp_BTCUSDT.parquet")
    schema_f32 = make_float32_schema(schema)
    df = parallel_read_parquets(
        FOLDER,
        pattern="perp_*.parquet",
        schema=schema_f32,
        symbol_prefix="perp_",
    )
