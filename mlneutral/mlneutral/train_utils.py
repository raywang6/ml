
from datetime import datetime, timedelta
import polars as pl

def generate_target(ret, horizon):
    tar = ret.rolling(horizon).sum().shift(-horizon)
    tar = tar.subtract(tar.mean(axis=1),axis=0)
    tar = tar.div(tar.abs().sum(axis=1),axis=0)
    scalevol = (tar.shift(1) * ret.reindex_like(tar)).sum(axis=1).ewm(168,min_periods = 72).std()
    return (tar * tarvol).div( scalevol.clip(scalevol.max()/5),axis=0).clip(-0.1,0.1)
    #return tar.clip(-0.5,0.5)

def generate_features(sub_strats, target, with_hist = 4):
    """
    Converts a dictionary of feature DataFrames (pandas) into a long-format Polar DataFrame.
    Parameters:
        sub_strats (dict): key = feature name, value = pandas DataFrame (index=datetime, columns=symbols)
    Returns:
        pl.DataFrame: A Polar DataFrame with columns: datetime, symbol, f1, f2, ...
    """
    long_df = ret.reset_index().melt(id_vars='datetime', var_name='symbol', value_name='target')
    feature_dfs = [pl.from_pandas(long_df)]
    for feature_name, df in sub_strats.items():
        # Reset index to turn datetime index into a column
        long_df = df.reset_index().melt(id_vars='datetime', var_name='symbol', value_name=feature_name)
        feature_dfs.append(pl.from_pandas(long_df))
    # Start from the first and merge all others on datetime and symbol
    result = feature_dfs[0]
    for feature_name, feat_df in zip(sub_strats.keys(), feature_dfs[1:]):
        for lag in range(1, with_hist+1):
            feat_df = feat_df.with_columns(
                    pl.col(feature_name).diff()
                    .shift(lag)
                    .over('symbol')
                    .alias(f"{feature_name}_diff{lag}")
                )
        result = result.join(feat_df, on=['datetime', 'symbol'], how='left')
        #result = result.drop(["datetime_right", "symbol_right"])
    non_key_cols = [col for col in result.columns if col not in ['datetime', 'symbol']]
    result = result.filter(
       pl.col('target').is_not_null()
    )
    return result

def generate_first_days(start, end, day = 15):
    # Helper function to parse and truncate a date
    def truncate_to_first_day(date):
        if isinstance(date, str):
            dt = datetime.strptime(date, "%Y-%m-%d")
        else:
            dt = date
        return dt.replace(day=day)
    # Parse and truncate start and end dates
    start_trunc = truncate_to_first_day(start)
    end_trunc = truncate_to_first_day(end)
    # Check if start is after end after truncation
    if start_trunc > end_trunc:
        return pl.Series(dtype=pl.Datetime(time_unit = 'ns', time_zone = 'UTC'))
    # Generate the date range
    date_series = pl.date_range(
        start=start_trunc,
        end=end_trunc,
        interval="1mo",
        closed="both",
        eager=True,
        time_unit = 'ns', time_zone = 'UTC'
    )
    return date_series
