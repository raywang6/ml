import pandas as pd
import cfg
from datetime import datetime, timedelta
from bn_downloader import *
import glob,os

futuniverse = pd.read_parquet(cfg.universe_folder + 'bn_future_universe.parquet')
spotuniverse = pd.read_parquet(cfg.universe_folder + 'bn_spot_universe.parquet')

end = datetime.now()
start = end - timedelta(days = 60)

symbols = [i.upper() for i in futuniverse['id']]
for ins in symbols:
    update_1min_data(ins, start, end, timedelta(days = 60), cfg.bar1m_folder)


symbols = [i.upper() for i in spotuniverse['id']]
for ins in symbols:
    update_1min_spotdata(ins, start, end, lookback, data_dir)
#


"""
#%% download futures
parser = get_parser('klines')
print("fetching all symbols from exchange")
exists = [os.path.basename(i).split('.parquet')[0] for i in glob.glob("/data/crypto/bar1m/futures/*")]
symbols = [i.upper() for i in futuniverse['id'] if i.upper() not in exists]
num_symbols = len(symbols)

period = convert_to_date_object(datetime.today().strftime('%Y-%m-%d')) - convert_to_date_object(PERIOD_START_DATE)
dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
dates = [date.strftime("%Y-%m-%d") for date in dates]

download_monthly_klines('um', symbols, num_symbols, ['1m'], [2020,2021,2022,2023,2024,2025], list(range(1,13)), '2020-01-01', '2025-06-30', '/data/crypto/bar1m', True)

#%% download spot
parser = get_parser('klines')

print("fetching all symbols from exchange")
exists = [os.path.basename(i).split('.parquet')[0] for i in glob.glob("/data/crypto/bar1m/spot/*")]
symbols = [i.upper() for i in spotuniverse['id'] if i.upper() not in exists]
num_symbols = len(symbols)

period = convert_to_date_object(datetime.today().strftime('%Y-%m-%d')) - convert_to_date_object(PERIOD_START_DATE)
dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
dates = [date.strftime("%Y-%m-%d") for date in dates]

download_monthly_klines('spot', symbols, num_symbols, ['1m'], [2020,2021,2022,2023,2024,2025], list(range(1,13)), '2020-01-01', '2025-05-30', '/data/crypto/bar1m', True)




#%% pack together
ins = symbols[0]
fpath = f"{data_dir}/{ins}.parquet"
prev_data = pd.read_parquet(fpath)
last = prev_data['end_tm'].max() + timedelta(seconds = 1)
start = last

parser = get_parser('klines')
period = end - start
dates = pd.date_range(end=datetime.today(), periods=period.days + 1).to_pydatetime().tolist()
dates = [date.strftime("%Y-%m-%d") for date in dates]
sins = ins
symbols = [sins]
download_daily_klines('spot', symbols, 1, ['1m'], dates, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), data_dir, True)
pname = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"

def get_all_spot_zip_paths(ins: str, freq: str, data_dir:str, pname: str) -> list[str]:
    "    Returns a list of absolute paths to all ZIP files for the given symbol `ins`."
    base_pattern = (
        f"{data_dir}"
        "/data/spot/monthly/klines"
        f"/{ins}/{freq}/{pname}/{ins}-1m-*.zip"
    )
    # glob.glob returns a list of paths matching the pattern
    return glob.glob(base_pattern)

for ins in ['1000CATUSDT', '1000CHEEMSUSDT', 'FLOKIUSDT', 'LUNCUSDT', 'PEPEUSDT', '1000SATSUSDT', 'SHIBUSDT', 'XECUSDT']:
    fs = get_all_spot_zip_paths(ins, '1m', '/data/crypto/bar1m', '')
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
    res.to_parquet(f"{data_dir}/{ins}.parquet")

#%%
"""
