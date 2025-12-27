from tardis_wraper import get_exchange_details
import pandas as pd
import datetime
import cfg

cut = pd.to_datetime('2022-01-01 00:00+0')



#%% future
futdetails = get_exchange_details('binance-futures')
futdetails = pd.DataFrame(futdetails["availableSymbols"]).query('type == "perpetual"')
futdetails['availableTo'] = pd.to_datetime(futdetails['availableTo']).fillna(datetime.datetime.now(datetime.UTC))
futdetails = futdetails[futdetails['availableTo'] > cut]
futdetails['base'] = futdetails['id'].apply(lambda x: x[-4:])
futdetails = futdetails[futdetails['base']=="usdt"]

#%%
spotdetails = get_exchange_details('binance')
spotdetails = pd.DataFrame(spotdetails["availableSymbols"])
spotdetails['availableTo'] = pd.to_datetime(spotdetails['availableTo']).fillna(datetime.datetime.now(datetime.UTC))
spotdetails = spotdetails[spotdetails['availableTo'] > cut]
spotdetails['base'] = spotdetails['id'].apply(lambda x: x[-4:])
spotdetails = spotdetails[spotdetails['base']=="usdt"]


futdetails.to_parquet(cfg.universe_folder + 'bn_future_universe.parquet')
spotdetails.to_parquet(cfg.universe_folder + 'bn_spot_universe.parquet')