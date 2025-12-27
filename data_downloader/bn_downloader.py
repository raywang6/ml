#!/usr/bin/env python

"""
  script to download klines.
  set the absolute path destination folder for STORE_DIRECTORY, and run

  e.g. STORE_DIRECTORY=/data/ ./download-kline.py

"""
from datetime import *
import pandas as pd
import os, sys, re, shutil, glob
import json
from pathlib import Path
import urllib.request
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentTypeError


YEARS = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"]
DAILY_INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
TRADING_TYPE = ["spot", "um", "cm"]
KLINES_COLS = ["start_tm", "open","high","low","close","vol","end_tm", "quote","trades","buy_vol","buy_quote","ignore"]

SYMBOL_1K = ['1000PEPEUSDT', '1000SHIBUSDT']

MONTHS = list(range(1,13))
PERIOD_START_DATE = '2020-01-01'
BASE_URL = 'https://data.binance.vision/'
START_DATE = date(int(YEARS[0]), MONTHS[0], 1)
END_DATE = datetime.date(datetime.now())

def get_destination_dir(file_url, folder=None):
  store_directory = os.environ.get('STORE_DIRECTORY')
  if folder:
    store_directory = folder
  if not store_directory:
    store_directory = os.path.dirname(os.path.realpath(__file__))
  return os.path.join(store_directory, file_url)

def get_download_url(file_url):
  return "{}{}".format(BASE_URL, file_url)

def get_all_symbols(type):
  if type == 'um':
    response = urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo").read()
  elif type == 'cm':
    response = urllib.request.urlopen("https://dapi.binance.com/dapi/v1/exchangeInfo").read()
  else:
    response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
  return list(map(lambda symbol: symbol['symbol'], json.loads(response)['symbols']))

def download_file(base_path, file_name, date_range=None, folder=None):
  download_path = "{}{}".format(base_path, file_name)
  if folder:
    base_path = os.path.join(folder, base_path)
  if date_range:
    date_range = date_range.replace(" ","_")
    base_path = os.path.join(base_path, date_range)
  save_path = get_destination_dir(os.path.join(base_path, file_name), folder)
  

  if os.path.exists(save_path):
    print("\nfile already exists! {}".format(save_path))
    return
  
  # make the directory
  if not os.path.exists(base_path):
    Path(get_destination_dir(base_path)).mkdir(parents=True, exist_ok=True)

  try:
    download_url = get_download_url(download_path)
    dl_file = urllib.request.urlopen(download_url)
    length = dl_file.getheader('content-length')
    if length:
      length = int(length)
      blocksize = max(4096,length//100)

    with open(save_path, 'wb') as out_file:
      dl_progress = 0
      print("\nFile Download: {}".format(save_path))
      while True:
        buf = dl_file.read(blocksize)   
        if not buf:
          break
        dl_progress += len(buf)
        out_file.write(buf)
        done = int(50 * dl_progress / length)
        sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )    
        sys.stdout.flush()

  except urllib.error.HTTPError:
    print("\nFile not found: {}".format(download_url))
    pass

def convert_to_date_object(d):
  year, month, day = [int(x) for x in d.split('-')]
  date_obj = date(year, month, day)
  return date_obj

def get_start_end_date_objects(date_range):
  start, end = date_range.split()
  start_date = convert_to_date_object(start)
  end_date = convert_to_date_object(end)
  return start_date, end_date

def match_date_regex(arg_value, pat=re.compile(r'\d{4}-\d{2}-\d{2}')):
  if not pat.match(arg_value):
    raise ArgumentTypeError
  return arg_value

def check_directory(arg_value):
  if os.path.exists(arg_value):
    while True:
      option = input('Folder already exists! Do you want to overwrite it? y/n  ')
      if option != 'y' and option != 'n':
        print('Invalid Option!')
        continue
      elif option == 'y':
        shutil.rmtree(arg_value)
        break
      else:
        break
  return arg_value

def raise_arg_error(msg):
  raise ArgumentTypeError(msg)

def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
  trading_type_path = 'data/spot'
  if trading_type != 'spot':
    trading_type_path = f'data/futures/{trading_type}'
  if interval is not None:
    path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
  else:
    path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
  return path

def get_parser(parser_type):
  parser = ArgumentParser(description=("This is a script to download historical {} data").format(parser_type), formatter_class=RawTextHelpFormatter)
  parser.add_argument(
      '-s', dest='symbols', nargs='+',
      help='Single symbol or multiple symbols separated by space')
  parser.add_argument(
      '-y', dest='years', default=YEARS, nargs='+', choices=YEARS,
      help='Single year or multiple years separated by space\n-y 2019 2021 means to download {} from 2019 and 2021'.format(parser_type))
  parser.add_argument(
      '-m', dest='months', default=MONTHS,  nargs='+', type=int, choices=MONTHS,
      help='Single month or multiple months separated by space\n-m 2 12 means to download {} from feb and dec'.format(parser_type))
  parser.add_argument(
      '-d', dest='dates', nargs='+', type=match_date_regex,
      help='Date to download in [YYYY-MM-DD] format\nsingle date or multiple dates separated by space\ndownload from 2020-01-01 if no argument is parsed')
  parser.add_argument(
      '-startDate', dest='startDate', type=match_date_regex,
      help='Starting date to download in [YYYY-MM-DD] format')
  parser.add_argument(
      '-endDate', dest='endDate', type=match_date_regex,
      help='Ending date to download in [YYYY-MM-DD] format')
  parser.add_argument(
      '-folder', dest='folder', type=check_directory,
      help='Directory to store the downloaded data')
  parser.add_argument(
      '-skip-monthly', dest='skip_monthly', default=0, type=int, choices=[0, 1],
      help='1 to skip downloading of monthly data, default 0')
  parser.add_argument(
      '-skip-daily', dest='skip_daily', default=0, type=int, choices=[0, 1],
      help='1 to skip downloading of daily data, default 0')
  parser.add_argument(
      '-c', dest='checksum', default=0, type=int, choices=[0,1],
      help='1 to download checksum file, default 0')
  parser.add_argument(
      '-t', dest='type', required=True, choices=TRADING_TYPE,
      help='Valid trading types: {}'.format(TRADING_TYPE))

  if parser_type == 'klines':
    parser.add_argument(
      '-i', dest='intervals', default=INTERVALS, nargs='+', choices=INTERVALS,
      help='single kline interval or multiple intervals separated by space\n-i 1m 1w means to download klines interval of 1minute and 1week')


  return parser

def download_monthly_klines(trading_type, symbols, num_symbols, intervals, years, months, start_date, end_date, folder, checksum):
  current = 0
  date_range = None

  if start_date and end_date:
    date_range = start_date + " " + end_date

  if not start_date:
    start_date = START_DATE
  else:
    start_date = convert_to_date_object(start_date)

  if not end_date:
    end_date = END_DATE
  else:
    end_date = convert_to_date_object(end_date)

  print("Found {} symbols".format(num_symbols))

  for symbol in symbols:
    print("[{}/{}] - start download monthly {} klines ".format(current+1, num_symbols, symbol))
    for interval in intervals:
      for year in years:
        for month in months:
          current_date = convert_to_date_object('{}-{}-01'.format(year, month))
          if current_date >= start_date and current_date <= end_date:
            path = get_path(trading_type, "klines", "monthly", symbol, interval)
            file_name = "{}-{}-{}-{}.zip".format(symbol.upper(), interval, year, '{:02d}'.format(month))
            download_file(path, file_name, None, folder)

            if checksum == 1:
              checksum_path = get_path(trading_type, "klines", "monthly", symbol, interval)
              checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, year, '{:02d}'.format(month))
              download_file(checksum_path, checksum_file_name, None, folder)

    current += 1

def download_daily_klines(trading_type, symbols, num_symbols, intervals, dates, start_date, end_date, folder, checksum):
  current = 0
  date_range = None

  if start_date and end_date:
    date_range = start_date + " " + end_date

  if not start_date:
    start_date = START_DATE
  else:
    start_date = convert_to_date_object(start_date)

  if not end_date:
    end_date = END_DATE
  else:
    end_date = convert_to_date_object(end_date)

  #Get valid intervals for daily
  intervals = list(set(intervals) & set(DAILY_INTERVALS))
  print("Found {} symbols".format(num_symbols))

  for symbol in symbols:
    print("[{}/{}] - start download daily {} klines ".format(current+1, num_symbols, symbol))
    for interval in intervals:
      for date in dates:
        current_date = convert_to_date_object(date)
        if current_date >= start_date and current_date <= end_date:
          path = get_path(trading_type, "klines", "daily", symbol, interval)
          file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, date)
          download_file(path, file_name, None, folder)

          if checksum == 1:
            checksum_path = get_path(trading_type, "klines", "daily", symbol, interval)
            checksum_file_name = "{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, date)
            download_file(checksum_path, checksum_file_name, None, folder)

    current += 1

def download_daily_metrics(trading_type, symbols, num_symbols, intervals, dates, start_date, end_date, folder, checksum):
  current = 0
  date_range = None

  if start_date and end_date:
    date_range = start_date + " " + end_date

  if not start_date:
    start_date = START_DATE
  else:
    start_date = convert_to_date_object(start_date)

  if not end_date:
    end_date = END_DATE
  else:
    end_date = convert_to_date_object(end_date)

  #Get valid intervals for daily
  intervals = list(set(intervals) & set(DAILY_INTERVALS))
  print("Found {} symbols".format(num_symbols))

  for symbol in symbols:
    print("[{}/{}] - start download daily {} metrics ".format(current+1, num_symbols, symbol))
    for interval in intervals:
      for date in dates:
        current_date = convert_to_date_object(date)
        if current_date >= start_date and current_date <= end_date:
          path = get_path(trading_type, "metrics", "daily", symbol)
          file_name = "{}-metrics-{}.zip".format(symbol.upper(), date)
          download_file(path, file_name, None, folder)

          if checksum == 1:
            checksum_path = get_path(trading_type, "metrics", "daily", symbol)
            checksum_file_name = "{}-metrics-{}.zip.CHECKSUM".format(symbol.upper(), date)
            download_file(checksum_path, checksum_file_name, None, folder)

    current += 1


def to_datetime_mixed(ts):
    ts = int(ts)
    length = len(str(ts))
    if length > 15:
        # nanoseconds (typically ~10^18…10^19)
        return pd.to_datetime(ts, unit="ns")
    elif length > 12:
        # microseconds (~10^15…10^16)
        return pd.to_datetime(ts, unit="us")
    else:
        # milliseconds (~10^12…10^13)
        return pd.to_datetime(ts, unit="ms")

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove folder and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def get_all_fut_zip_paths(ins: str, freq: str, data_dir:str, pname: str) -> list[str]:
    "    Returns a list of absolute paths to all ZIP files for the given symbol `ins`."
    base_pattern = (
        f"{data_dir}"
        "/data/futures/um/daily/klines"
        f"/{ins}/{freq}/{pname}/{ins}-1m-*.zip"
    )
    # glob.glob returns a list of paths matching the pattern
    return glob.glob(base_pattern)

def update_1min_data(ins, start, end, lookback, data_dir):
    fpath = os.path.join(data_dir, 'futures', f"{ins}.parquet")
    
    # Check if existing data exists and determine start date
    if os.path.exists(fpath):
        prev_data = pd.read_parquet(fpath)
        last = max(start - lookback, prev_data['start_tm'].max() + timedelta(minutes=1))
        start = last
    else:
        start = start - lookback
        last = None
    
    # Download data
    parser = get_parser('klines')
    symbols = [ins]
    period = end - start
    dates = pd.date_range(start=start, end=end, freq='D').to_pydatetime().tolist()
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    
    download_daily_klines('um', symbols, 1, ['1m'], dates, start.strftime('%Y-%m-%d'), 
                         end.strftime('%Y-%m-%d'), data_dir, True)
    
    # Get file paths and check if any files exist
    fs = get_all_fut_zip_paths(ins, '1m', data_dir, '')
    
    # Handle case where no files are found (symbol discontinued or no new data)
    if not fs:
        print(f"No data files found for {ins}. Symbol may be discontinued or no new data available.")
        return
    
    # Filter files to only process those that might contain new data
    if last is not None:
        # Only process files that could potentially have data after 'last' timestamp
        # This is a heuristic - you might need to adjust based on your file naming convention
        filtered_fs = []
        for path in fs:
            # Extract date from filename or check file modification time
            # This assumes files are named with dates - adjust as needed
            try:
                # If you can determine from filename whether file has data after 'last', filter here
                # For now, we'll process all files but this can be optimized
                filtered_fs.append(path)
            except:
                filtered_fs.append(path)
        fs = filtered_fs
    
    # Process files
    res = pd.DataFrame()
    for path in fs:
        try:
            tmp = pd.read_csv(path, names=KLINES_COLS)
            
            # Skip empty files
            if tmp.empty:
                continue
                
            # Handle header row
            if tmp.loc[0, 'start_tm'] == 'open_time':
                tmp = tmp.drop(0)
                
            # Skip if no data after header removal
            if tmp.empty:
                continue
                
            # Convert timestamps
            nlen = len(str(tmp.loc[tmp.index[0], 'start_tm']))
            if nlen > 16:
                tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="ns")
                tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="ns")
            elif nlen > 13:
                tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="us")
                tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="us")
            else:
                tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="ms")
                tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="ms")
                
            res = pd.concat([res, tmp], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue
    
    # Handle case where res is empty after processing all files
    if res.empty:
        print(f"No valid data found for {ins} in the specified date range.")
        # If we have existing data but no new data, just return without error
        if last is not None:
            print(f"Existing data for {ins} is up to date.")
        return
    
    # Convert data types
    try:
        res = res.astype({
            "open": float, "high": float, "low": float, "close": float,
            "vol": float, "quote": float, "trades": int, 
            "buy_vol": float, "buy_quote": float, 'ignore': int
        })
    except Exception as e:
        print(f"Error converting data types: {e}")
        return
    
    res = res.sort_values('end_tm')
    
    # Handle merging with existing data
    data_updated = False
    
    if last is not None:
        # Filter new data to only include records after the last timestamp
        new_data = res[res['end_tm'] >= last]
        
        # If no new data after filtering, no need to update
        if new_data.empty:
            print(f"No new data to add for {ins}. Data is already up to date.")
            return
        
        # Check if we actually have new records (not just duplicates)
        original_count = len(prev_data)
        
        # Merge with existing data
        res = pd.concat([prev_data, new_data], ignore_index=True)
        res = res.sort_values('end_tm').drop_duplicates(subset=['end_tm'])
        
        # Check if we actually added new records
        if len(res) > original_count:
            data_updated = True
        else:
            print(f"No new unique data to add for {ins}. All data already exists.")
            return
    else:
        # New file, always update
        data_updated = True
    
    # Only save the data if it was actually updated
    if data_updated:
        try:
            res.to_parquet(fpath)
            new_records = len(res) - (len(prev_data) if last is not None else 0)
            print(f"Successfully updated data for {ins}. Added {new_records} new records. Total: {len(res)}")
        except Exception as e:
            print(f"Error saving data to {fpath}: {e}")
            return
    else:
        print(f"No update needed for {ins}. Data is already current.")


def get_all_spot_zip_paths(ins: str, freq: str, data_dir:str, pname: str) -> list[str]:
    "    Returns a list of absolute paths to all ZIP files for the given symbol `ins`."
    base_pattern = (
        f"{data_dir}"
        "/data/spot/daily/klines"
        f"/{ins}/{freq}/{pname}/{ins}-1m-*.zip"
    )
    # glob.glob returns a list of paths matching the pattern
    return glob.glob(base_pattern)

def update_1min_spotdata(ins, start, end, lookback, data_dir):
    fpath = os.path.join(data_dir, 'spot', f"{ins}.parquet")
    
    # Check if existing data exists and determine start date
    if os.path.exists(fpath):
        prev_data = pd.read_parquet(fpath)
        last = max(start - lookback, prev_data['start_tm'].max() + timedelta(minutes=1))
        start = last
    else:
        start = start - lookback
        last = None
    
    # Download data
    parser = get_parser('klines')
    symbols = [ins]
    period = end - start
    dates = pd.date_range(start=start, end=end, freq='D').to_pydatetime().tolist()
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    
    download_daily_klines('spot', symbols, 1, ['1m'], dates, start.strftime('%Y-%m-%d'), 
                         end.strftime('%Y-%m-%d'), data_dir, True)
    
    # Get file paths and check if any files exist
    fs = get_all_spot_zip_paths(ins, '1m', data_dir, '')
    
    # Handle case where no files are found (symbol discontinued or no new data)
    if not fs:
        print(f"No data files found for {ins}. Symbol may be discontinued or no new data available.")
        return
    
    # Filter files to only process those that might contain new data
    if last is not None:
        # Only process files that could potentially have data after 'last' timestamp
        # This is a heuristic - you might need to adjust based on your file naming convention
        filtered_fs = []
        for path in fs:
            # Extract date from filename or check file modification time
            # This assumes files are named with dates - adjust as needed
            try:
                # If you can determine from filename whether file has data after 'last', filter here
                # For now, we'll process all files but this can be optimized
                filtered_fs.append(path)
            except:
                filtered_fs.append(path)
        fs = filtered_fs
    
    # Process files
    res = pd.DataFrame()
    for path in fs:
        try:
            tmp = pd.read_csv(path, names=KLINES_COLS)
            
            # Skip empty files
            if tmp.empty:
                continue
                
            # Handle header row
            if tmp.loc[0, 'start_tm'] == 'open_time':
                tmp = tmp.drop(0)
                
            # Skip if no data after header removal
            if tmp.empty:
                continue
                
            # Convert timestamps
            nlen = len(str(tmp.loc[tmp.index[0], 'start_tm']))
            if nlen > 16:
                tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="ns")
                tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="ns")
            elif nlen > 13:
                tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="us")
                tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="us")
            else:
                tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="ms")
                tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="ms")
                
            res = pd.concat([res, tmp], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue
    
    # Handle case where res is empty after processing all files
    if res.empty:
        print(f"No valid data found for {ins} in the specified date range.")
        # If we have existing data but no new data, just return without error
        if last is not None:
            print(f"Existing data for {ins} is up to date.")
        return
    
    # Convert data types
    try:
        res = res.astype({
            "open": float, "high": float, "low": float, "close": float,
            "vol": float, "quote": float, "trades": int, 
            "buy_vol": float, "buy_quote": float, 'ignore': int
        })
    except Exception as e:
        print(f"Error converting data types: {e}")
        return
    
    res = res.sort_values('end_tm')
    
    # Handle merging with existing data
    data_updated = False
    
    if last is not None:
        # Filter new data to only include records after the last timestamp
        new_data = res[res['end_tm'] >= last]
        
        # If no new data after filtering, no need to update
        if new_data.empty:
            print(f"No new data to add for {ins}. Data is already up to date.")
            return
        
        # Check if we actually have new records (not just duplicates)
        original_count = len(prev_data)
        
        # Merge with existing data
        res = pd.concat([prev_data, new_data], ignore_index=True)
        res = res.sort_values('end_tm').drop_duplicates(subset=['end_tm'])
        
        # Check if we actually added new records
        if len(res) > original_count:
            data_updated = True
        else:
            print(f"No new unique data to add for {ins}. All data already exists.")
            return
    else:
        # New file, always update
        data_updated = True
    
    # Only save the data if it was actually updated
    if data_updated:
        try:
            res.to_parquet(fpath)
            new_records = len(res) - (len(prev_data) if last is not None else 0)
            print(f"Successfully updated data for {ins}. Added {new_records} new records. Total: {len(res)}")
        except Exception as e:
            print(f"Error saving data to {fpath}: {e}")
            return
    else:
        print(f"No update needed for {ins}. Data is already current.")

# [debug] to update!
def update_5min_oi(ins, start, end, lookback, data_dir):
  fpath = f"{data_dir}/{ins}.parquet"
  if os.path.exists(fpath):
    prev_data = pd.read_parquet(fpath)
    last = prev_data['end_tm'].max() + timedelta(seconds = 1)
    start = last
  else:
    start = start - lookback
    last = None
  parser = get_parser('klines')
  symbols = [ins]
  period = end - start
  dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
  dates = [date.strftime("%Y-%m-%d") for date in dates]
  download_daily_klines('um', symbols, 1, ['1m'], dates, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), data_dir, True)
  pname = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
  fs = get_all_fut_zip_paths(ins, '1m', data_dir, pname)
  res = pd.DataFrame()
  for path in fs:
      tmp = pd.read_csv(path, names = KLINES_COLS)
      if tmp.loc[0,'start_tm'] == 'open_time':
        tmp = tmp.drop(0)
        nlen = len(str(tmp.loc[tmp.index[0],'start_tm']))
        if nlen > 16:
          tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="ns")
          tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="ns")
        elif nlen > 13:
          tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="us")
          tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="us")
        else:
          tmp['start_tm'] = pd.to_datetime(tmp['start_tm'], unit="ms")
          tmp['end_tm'] = pd.to_datetime(tmp['end_tm'], unit="ms")
      res = pd.concat([res, tmp], ignore_index=True)
  res = res.astype({"open": float,"high": float,"low": float,"close": float,"vol": float, "quote": float,"trades": int,"buy_vol": float,"buy_quote": float, 'ignore': int})
  res = res.sort_values('end_tm')
  if last is not None:
    res = res[last:]
    res = pd.concat([prev_data, res], ignore_index=True).sort_values('end_tm').drop_duplicates(subset = ['end_tm'])
  res.to_parquet(f"{data_dir}/{ins}.parquet")


if __name__ == "__main__":
    parser = get_parser('klines')

    print("fetching all symbols from exchange")
    symbols = ['PEPEUSDT', 'SHIBUSDT']
    num_symbols = len(symbols)

    period = convert_to_date_object(datetime.today().strftime('%Y-%m-%d')) - convert_to_date_object(
        PERIOD_START_DATE)
    dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
    dates = [date.strftime("%Y-%m-%d") for date in dates]

    download_monthly_klines('spot', symbols, num_symbols, ['1m'], [2020,2021,2022,2023,2024,2025], list(range(1,13)), '2020-01-01', '2025-05-30', '/home/ubuntu/project2/home/ubuntu/crypto/mlfeatures/bar1m', True)




