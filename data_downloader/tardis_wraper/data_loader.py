from __future__ import annotations

import os
import glob
import yaml
import re
import pathlib
import polars as pl
import gc
from typing import Iterable

from tardis_dev import datasets, get_exchange_details
import datetime as dt
import pandas as pd
import numpy as np
import time


from .types import (
    Callable,
    Dict,
    Any,
    DatetimeType,
    TimeType,
    DateType,
    FreqType,
    PathType,
    Generator
)

from .utils import (
    genenerate_dates,
    to_date
)

_cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_cwd, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)


def month_range(start: DateType, end: DateType, trunc_days: int):
    """
    Generator yielding (month_st, month_end) from start to end inclusive,
    stepping by month.
    """
    current = start
    nextmonth = current
    while current <= end:
        # Advance to the next month
        nextmonth += dt.timedelta(days=trunc_days)
        yield current, nextmonth - dt.timedelta(days=1)
        current = nextmonth

def get_hist_spot_bars_features(
        symbol: str,
        source: str,
        freq: FreqType,
        dtname: str,
        _read_csv: DateType,
        end_date: DateType,
        version_name: str,
        agg_func: Callable
    ) -> pl.DataFrame:
    """
    Loads large historical trades data in monthly chunks, generates bar features,
    saves them to disk, then re-loads all bar files and returns the concatenated result.
    """
    # Convert incoming dates to pandas Timestamps for easier monthly iteration
    dates = [i.date() for i in genenerate_dates(start_date, end_date)]

    output_dir = os.path.join(config['data']['data_dir'], f"binance/bars/{symbol}/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over months, generate bar data, and write to disk
    for tdate in dates:

        file_name = f"{tdate.year:04d}{tdate.month:02d}{tdate.day:02d}_{freq}_{version_name}.parquet"
        file_path = os.path.join(output_dir, file_name)

        if os.path.exists(file_path):
            continue

        trades_df = get_hist_spot_trades_data(
            symbol=symbol,
            source=source,
            start_date=tdate,
            end_date=tdate
        )
        trades_df = trades_df.filter(pl.col(dtname).is_between(tdate, tdate + dt.timedelta(days=1)))
        if trades_df.is_empty():
            continue

        bars_df = agg_func(
            data=trades_df,
            freq=freq,
            dtname=dtname,
        )
        bars_df.write_parquet(file_path)
        del trades_df, bars_df
        gc.collect()

    all_bar_files = sorted(glob.glob(os.path.join(output_dir, f"*_{freq}_{version_name}.parquet")))
    if not all_bar_files:
        print("No bar files found. Returning empty DataFrame.")
        return pl.DataFrame()

    bar_dfs = []
    for fp in all_bar_files:
        bar_dfs.append(pl.read_parquet(str(fp)))

    # Concatenate all monthly bars
    final_bars_df = pl.concat(bar_dfs, how="vertical")

    return final_bars_df



def get_hist_perp_bars_features(
        symbol: str,
        source: str,
        freq: FreqType,
        dtname: str,
        start_date: DateType,
        end_date: DateType,
        version_name: str,
        agg_func: Callablet
    ) -> pl.DataFrame:
    """
    Loads large historical trades data in monthly chunks, generates bar features,
    saves them to disk, then re-loads all bar files and returns the concatenated result.
    """
    # Convert incoming dates to pandas Timestamps for easier monthly iteration
    dates = [i.date() for i in genenerate_dates(start_date, end_date)]

    output_dir = os.path.join(config['data']['data_dir'], f"binance-futures/bars/{symbol}/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over months, generate bar data, and write to disk
    for tdate in dates:

        file_name = f"{tdate.year:04d}{tdate.month:02d}{tdate.day:02d}_{freq}_{version_name}.parquet"
        file_path = os.path.join(output_dir, file_name)

        if os.path.exists(file_path):
            continue

        trades_df = get_hist_perp_trades_data(
            symbol=symbol,
            source=source,
            start_date=tdate,
            end_date=tdate
        )
        trades_df = trades_df.filter(pl.col(dtname).is_between(tdate, tdate + dt.timedelta(days=1)))
        if trades_df.is_empty():
            continue

        bars_df = agg_func(
            data=trades_df,
            freq=freq,
            dtname=dtname,
        )
        bars_df.write_parquet(file_path)
        del trades_df, bars_df
        gc.collect()

    all_bar_files = sorted(glob.glob(os.path.join(output_dir, f"*_{freq}_{version_name}.parquet")))
    if not all_bar_files:
        print("No bar files found. Returning empty DataFrame.")
        return pl.DataFrame()

    bar_dfs = []
    for fp in all_bar_files:
        bar_dfs.append(pl.read_parquet(str(fp)))

    # Concatenate all monthly bars
    final_bars_df = pl.concat(bar_dfs, how="vertical")

    return final_bars_df



def get_hist_spot_trades_data(
        symbol: str,
        source: str,
        #period: FreqType,
        start_date: DateType,
        end_date: DateType,
    ) -> pl.DataFrame:
    assert source in ('tardis',)
    #assert period in ('5m',) # currently only support '5m\
    _download_data_from_tardis(
        sdate = start_date, edate = end_date,
        symbol = symbol,
        data_type = 'trades',
        exchange = 'binance',
    )
    if source == 'tardis':
        # save_path = os.path.join(config['data']['data_dir'], f"binance/trades/{symbol}/")
        save_path = os.path.join(config['data']['data_dir'], f"trades_spot_bn")
        fpatt = re.compile(r'.*binance_trades_(\d{4}-\d{2}-\d{2})_'+ symbol +'\.csv\.gz$')
        schema_overrides = {'price': pl.Float64, 'amount': pl.Float64}
        tdata = (
            _read_csv(dir_path=save_path,sdate=start_date,edate=end_date, schema_overrides = schema_overrides, filename_pattern = fpatt).with_columns((pl.col("timestamp")).cast(pl.Datetime).alias("datetime"))
                .select(
                    ["datetime", "symbol", 'side', 'price', 'amount']
                )
        )
    return tdata



def get_hist_perp_trades_data(
        symbol: str,
        source: str,
        start_date: DateType,
        end_date: DateType,
    ) -> pl.DataFrame:
    assert source in ('tardis',)
    #assert period in ('5m',) # currently only support '5m\
    _download_data_from_tardis(
        sdate = start_date, edate = end_date,
        symbol = symbol,
        data_type = 'trades',
        exchange = 'binance-futures',
    )
    if source == 'tardis':
        save_path = os.path.join(config['data']['data_dir'], f"trades_bn")
        schema_overrides = {'price': pl.Float64, 'amount': pl.Float64}
        fpatt = re.compile(r'.*binance-futures_trades_(\d{4}-\d{2}-\d{2})_'+ symbol +'\.csv\.gz$')
        tdata = (
            _read_csv(dir_path=save_path,sdate=start_date,edate=end_date, schema_overrides = schema_overrides, filename_pattern = fpatt).with_columns((pl.col("timestamp")).cast(pl.Datetime).alias("datetime"))
                .select(
                    ["datetime", "symbol", 'side', 'price', 'amount']
                )
        )
    return tdata



def get_hist_perp_tickers_data(
        symbol: str,
        source: str,
        start_date: DateType,
        end_date: DateType,
    ) -> pl.DataFrame:

    assert source in ('tardis',)
    #assert period in ('5m',) # currently only support '5m\
    _download_data_from_tardis(
        sdate = start_date, edate = end_date,
        symbol = symbol,
        data_type = 'derivative_ticker',
        exchange = 'binance-futures',
    )
    if source == 'tardis':
        save_path = os.path.join(config['data']['data_dir'], f"derivative_ticker_bn")
        schema_overrides = {'open_interest': pl.Float64, 'last_price': pl.Float64, 'index_price': pl.Float64, 'mark_price': pl.Float64}
        fpatt = re.compile(r'.*binance-futures_derivative_ticker_(\d{4}-\d{2}-\d{2})_'+ symbol +'\.csv\.gz$')
        tdata = (
            _read_csv(dir_path=save_path,sdate=start_date,edate=end_date, schema_overrides = schema_overrides).with_columns((pl.col("timestamp")).cast(pl.Datetime).alias("datetime"))
                .select(
                    ["datetime", "symbol", 'open_interest', 'last_price', 'index_price', 'mark_price']
                )
        )
    return tdata

def download_data_from_tardis(    
    sdate: str, edate: str,
    symbol: str,
    data_type: str,
    exchange: str,
    details = None
    ):
    return _download_data_from_tardis(    
        sdate, edate,
        symbol,
        data_type,
        exchange,
        details
    )

def _download_data_from_tardis(
    sdate: str, edate: str,
    symbol: str,
    data_type: str,
    exchange: str,
    details = None
    ):
    if exchange == 'binance-futures':
        if data_type == 'trades':
            save_path = os.path.join(config['data']['data_dir'], "trades_bn")
        elif data_type == "derivative_ticker":
            save_path = os.path.join(config['data']['data_dir'], "derivative_ticker_bn")
        elif data_type == "liquidations":
            save_path = os.path.join(config['data']['data_dir'], "liquidations")
        else:
            raise NotImplementedError()
    elif exchange == "binance":
        if data_type == 'trades':
            save_path = os.path.join(config['data']['data_dir'], "trades_spot_bn")
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    if details is None:
        details = get_exchange_details(exchange)
        details = pd.DataFrame(details["availableSymbols"])
        details['availableTo'] = pd.to_datetime(details['availableTo']).fillna(dt.datetime.now(datetime.UTC))

    sdate = str(max(dt.date.fromisoformat(details[details['id']==symbol.lower()]['availableSince'].values[0].split('T')[0]), dt.date.fromisoformat(sdate)))
    if details[details['id']==symbol.lower()]['availableTo'].values[0] is np.nan:
        to_date =str(dt.date.fromisoformat(edate))
    else:
        to_date = str(min(dt.date.fromisoformat(pd.to_datetime(details[details['id']==symbol.lower()]['availableTo'].values[0]).strftime('%Y-%m-%d')), dt.date.fromisoformat(edate)))
    dates = _get_tardis_dir_missing_dates(save_path, symbol, sdate, edate)
    if len(dates) > 1:
        print(f"start job {len(dates)}: from {dates[0]} to {dates[-1]}")
    for date in dates:
        std1, edd1 = _generate_one_date_range(date)
        datasets.download(
            exchange=exchange,
            # data_types=[ "trades", "quotes", "derivative_ticker", "book_snapshot_25", "book_snapshot_5", "liquidations"],
            data_types=[data_type],
            from_date=std1,
            to_date=edd1,
            symbols=[symbol],
            api_key=config['tardis']['key'],
            download_dir=save_path,
        )
        time.sleep(2)

def _generate_one_date_range(date: DateType):
    return (date.strftime('%Y-%m-%d'),(date + dt.timedelta(days=1)).strftime('%Y-%m-%d'))


def _get_tardis_dir_missing_dates(wkdir:str, symbol: str, sdate:str, edate:str):
    fs = glob.glob(os.path.join(wkdir, f'*_{symbol}.csv.gz'))
    existing_dates = []
    for f in fs:
        date = _extract_date(f)
        if date is not None:
            existing_dates.append(date)
    dates = genenerate_dates(sdate, edate)
    dates = [i for i in dates if i.strftime('%Y-%m-%d') not in existing_dates]
    return dates


def _extract_date(filename):
    # Define the regex pattern to match the date in the format YYYY-MM-DD
    pattern = r'\d{4}-\d{2}-\d{2}'
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    # If a match is found, return the matched string
    if match:
        return match.group(0)
    else:
        return None


def _extract_files_by_date(
        dir_path: PathType,
        suffix: str,
        sdate: DateType | None = None,
        edate: DateType | None = None,
        filename_pattern: str | re.Pattern | None = None,
    ) -> Generator[pathlib.Path]:
    dir_path = pathlib.Path(dir_path)
    sdate = to_date(sdate) if sdate is not None else dt.date(2020, 1, 1)
    edate = to_date(edate) if edate is not None else dt.date(2300, 12, 31)
    if not filename_pattern:
        files = (
            fl
            for fl in dir_path.glob(f"*{suffix}")
            if sdate <= dt.date.fromisoformat(_extract_date(fl.stem)) <= edate
        )
    else:
        filename_pattern = re.compile(filename_pattern)
        files = (
            fl
            for fl in dir_path.glob(f"*{suffix}")
            for extracted_date_list in [re.findall(filename_pattern, fl.name)]
            if extracted_date_list
            if sdate <= to_date(extracted_date_list[0]) <= edate
        )
    return files


def _read_csv(    
        dir_path: PathType,
        sdate: DateType | None = None,
        edate: DateType | None = None,
        time: TimeType | None = None,
        columns: list[str] | None = None,
        sort_cols: list[str] | None = None,
        filename_pattern: str | re.Pattern | None = None,
        schema_overrides: Dict[str,Any] | None = None
    ) -> pl.DataFrame:
    _files = _extract_files_by_date(
        dir_path=dir_path,
        suffix=".csv.gz",
        sdate=sdate,
        edate=edate,
        filename_pattern=filename_pattern,
    )
    _files_list = sorted(_files)
    res = pl.concat(
                    [
                        pl.read_csv(source=f, columns=columns, schema_overrides = schema_overrides).shrink_to_fit() 
                        for f in _files_list
                    ],
                    how="diagonal",
                )
    return res

