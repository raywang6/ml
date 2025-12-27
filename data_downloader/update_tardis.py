
import pandas as pd
from tardis_wraper import download_data_from_tardis
import cfg
from time import sleep

futuniverse = pd.read_parquet(cfg.universe_folder + 'bn_future_universe.parquet')
spotuniverse = pd.read_parquet(cfg.universe_folder + 'bn_spot_universe.parquet')
sym1k = ['1000BONKUSDT', '1000FLOKIUSDT', '1000SHIBUSDT','1000PEPEUSDT','1000XECUSDT','1000RATSUSDT','1000LUNCUSDT']

complete = ['SUIUSDT', 'AAVEUSDT', 'ADAUSDT', 'FETUSDT', 'WIFUSDT', 'ARUSDT', 'AXSUSDT', 'DOGEUSDT', 'PYTHUSDT', 'ARBUSDT', 'GALAUSDT', 'BNBUSDT', 'LINKUSDT', 'UNIUSDT', 
'1000BONKUSDT', '1000FLOKIUSDT', '1000SHIBUSDT', 'ETCUSDT', 'ETHUSDT', 'BTCUSDT', 'PENDLEUSDT', 'SOLUSDT', 'TIAUSDT', 'TONUSDT', 'WLDUSDT', '1000PEPEUSDT', 'FILUSDT', 
'ORDIUSDT', 'SEIUSDT', 'DYDXUSDT', 'POLUSDT', 'MKRUSDT', 'TAOUSDT', 'DOTUSDT', 'OPUSDT', 'RUNEUSDT', 'LTCUSDT', 'BCHUSDT', 'APEUSDT', 'ATOMUSDT', 'SANDUSDT', 'NEARUSDT', 
'APTUSDT', 'AVAXUSDT', 'JTOUSDT', 'JUPUSDT', 'STXUSDT', 'XRPUSDT', 'LDOUSDT', 'TRXUSDT', 'CRVUSDT', 'ENAUSDT', 'INJUSDT', 'RENDERUSDT']


symbols = [i.upper() for i in futuniverse['id'].head(200).tail(101) if i.upper() not in complete]

# TAOUSDT
sdate = '2024-01-01'
edate = '2025-07-20'
err = []
# err = ['DEFIUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'CHZUSDT', 'ANKRUSDT', 'RVNUSDT', 'SFPUSDT', 'COTIUSDT', 'CHRUSDT', 'MANAUSDT', 'ALICEUSDT', 'HBARUSDT', 'ONEUSDT', 'DENTUSDT', 'CELRUSDT', 'HOTUSDT', 'MTLUSDT', 'OGNUSDT', 'NKNUSDT', 'ICPUSDT', 'BAKEUSDT', 'GTCUSDT', 'BTCDOMUSDT', 'TLMUSDT', 'IOTXUSDT', 'C98USDT', 'MASKUSDT', 'ATAUSDT', '1000XECUSDT', 'CELOUSDT', 'ARPAUSDT', 'CTSIUSDT']
#  ['1000LUNCUSDT', 'LUNA2USDT', 'LEVERUSDT', 'DODOXUSDT', 'BSVUSDT', 'TOKENUSDT', 'KASUSDT', 'ETHWUSDT', '1000RATSUSDT']
# CTSIUSDT
for symbol in symbols: #universe[1]
    if symbol in sym1k:
        spot_sym = symbol[4:]
    else:
        spot_sym = symbol
    try:
        download_data_from_tardis(    
                sdate, edate,
                spot_sym,
                'trades',
                'binance',
                details = spotuniverse
            )
        sleep(5)
        download_data_from_tardis(    
                sdate, edate,
                symbol,
                'trades',
                'binance-futures',
                details = futuniverse
            )
        sleep(5)
        download_data_from_tardis(    
                sdate, edate,
                symbol,
                data_type = 'derivative_ticker',
                exchange = 'binance-futures',
                details = futuniverse
            )
        download_data_from_tardis(    
                sdate, edate,
                symbol,
                data_type = 'liquidations',
                exchange = 'binance-futures',
                details = futuniverse
            )
    except:
        err.append(symbol)
        sleep(30)

# liquidation
