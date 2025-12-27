
from datetime import datetime
import pandas as pd
from datetime import *
import os, sys, re, shutil
import json
from pathlib import Path
import urllib.request
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentTypeError

YEARS = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"]
DAILY_INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
TRADING_TYPE = ["spot", "um", "cm"]
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
  return os.path.join(store_directory, os.path.basename(file_url))

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
  download_path = "{}/{}".format(base_path, file_name)
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
  if not os.path.exists(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)

  try:
    download_url = get_download_url(download_path)
    dl_file = urllib.request.urlopen(download_url)
    length = dl_file.getheader('content-length')
    if length:
      length = int(length)
      blocksize = max(4096,length//100)

    with open(save_path, 'wb') as out_file:
      dl_progress = 0
      #print("\nFile Download: {}".format(save_path))
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
    #print("\nFile not found: {}".format(download_url))
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
  trading_type_path = '?prefix=data/spot'
  if trading_type != 'spot':
    trading_type_path = f'?prefix=data/futures/{trading_type}'
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

def download_daily_metrics(symbol, dates, start_date, end_date, folder, checksum):
    """
    Download daily metrics ZIPs from Binance Vision (data/futures/um/daily/metrics).
    """
    current = 0
    # Prepare date bounds
    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    for date in dates:
        current_date = convert_to_date_object(date)
        if start_date <= current_date <= end_date:
            # The metrics files are not symbol-specific; one file per date
            path = f"data/futures/um/daily/metrics/{symbol}"
            file_name = f"{symbol}-metrics-{date}.zip"
            #print(f"[{current+1}/{len(dates)}] Downloading metrics for {date}")
            download_file(path, file_name, None, folder)

            if checksum:
                checksum_file = f"{symbol}-metrics-{date}.zip.CHECKSUM"
                download_file(path, checksum_file, None, folder)

        current += 1


if __name__ == "__main__":
    parser = get_parser('metrics')  # adjust parser to accept 'metrics'
    args = parser.parse_args(sys.argv[1:])

    # dates: use provided or build range
    if args.dates:
        dates = args.dates
    else:
        period = convert_to_date_object(datetime.today().strftime('%Y-%m-%d')) - convert_to_date_object(PERIOD_START_DATE)
        dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
        dates = [date.strftime("%Y-%m-%d") for date in dates]

    if not args.symbols:
        print("fetching all symbols from exchange")
        symbols = get_all_symbols(args.type)
        num_symbols = len(symbols)
    else:
        symbols = args.symbols
        num_symbols = len(symbols)
    #symbols = ['1000BONKUSDT', '1000SATSUSDT', 'AI16ZUSDT', 'AIXBTUSDT', 'ALGOUSDT', 'BERAUSDT', 'BOMEUSDT', 'CAKEUSDT', 'CUSDT', 'EIGENUSDT', 'ENAUSDT', 'ENSUSDT', 'ERAUSDT', 'ETHFIUSDT', 'FARTCOINUSDT', 'HBARUSDT', 'HUMAUSDT', 'HYPEUSDT', 'ICPUSDT', 'INITUSDT', 'JTOUSDT', 'JUPUSDT', 'KAITOUSDT', 'LAUSDT', 'MOODENGUSDT', 'MUSDT', 'MYXUSDT', 'NEIROUSDT', 'OMUSDT', 'ONDOUSDT', 'PAXGUSDT', 'PENGUUSDT', 'PEOPLEUSDT', 'PNUTUSDT', 'POPCATUSDT', 'RAYSOLUSDT', 'RENDERUSDT', 'RESOLVUSDT', 'SAHARAUSDT', 'SOPHUSDT', 'SPKUSDT', 'SPXUSDT', 'SUSDT', 'SYRUPUSDT', 'TAOUSDT', 'TIAUSDT', 'TRBUSDT', 'TRUMPUSDT', 'TURBOUSDT', 'VIRTUALUSDT', 'WIFUSDT', 'XLMUSDT', 'ZORAUSDT', 'ZROUSDT']
    futuniverse = pd.read_parquet('/data/crypto/universe/' + 'bn_future_universe.parquet')
    symbols = [i.upper() for i in futuniverse['id']]
    num_symbols = len(symbols)
    for symbol in symbols:
        download_daily_metrics(
            symbol,
            dates,
            '2025-07-01',
            '2025-11-10', # args.endDate
            args.folder,
            args.checksum
        )
        print(f"complete {symbol}")
