

import glob, os
import polars as pl
import pytz
import yaml
from datetime import datetime, timedelta
from bn_downloader import (
    update_1min_data, clear_folder,
    update_1min_spotdata
)
from srdist import run_srdist_updates

import cfg
from train_utils import(
    generate_features_from_config,
    generate_training_dates,
)
from feature_engineer import (
    generate_targets, transformFT, 
    feature_selection_2step,
    downsample_sequences
)
from gru import prepare_gru
UTCTZ = pytz.timezone('UTC')

data_dir = '/data/crypto'
end = datetime.now().replace(tzinfo=UTCTZ)
start = end - timedelta(days=40)
bar1m_dir = f'{data_dir}/bar1m/futures/'
lookback = timedelta(days=60)
universe = ['BTCUSDT', 'ADAUSDT',]# 'ATOMUSDT', 'BCHUSDT', 'DOGEUSDT', 'DYDXUSDT', 'ETHUSDT', 'FILUSDT', 
#'JASMYUSDT', 'LINKUSDT', 'LTCUSDT', 'GALAUSDT', 'RUNEUSDT', 'SOLUSDT', 'SUIUSDT', 'NEARUSDT', 'XRPUSDT', 
#'SUSHIUSDT', 'TRXUSDT', 'UNIUSDT', 'AVAXUSDT', 'AAVEUSDT', '1000PEPEUSDT', 'WLDUSDT', 'BNBUSDT', 
#'1000SHIBUSDT', 'ORDIUSDT', 'ETCUSDT', 'APEUSDT', 'COMPUSDT', 'DOTUSDT', 'SANDUSDT', 'MANAUSDT', '1INCHUSDT', 'TONUSDT', 'STXUSDT']

# Downloader
#%% future 1m -> 15m data
#or ins in universe:
#    update_1min_data(ins, start, end, lookback, bar1m_dir)

#clear_folder(f"{bar1m_dir}/data")

#%% oi
#oi_dir = f'{data_dir}/oiprod'

for ins in universe:
    # load data
    p1m = pl.read_parquet(os.path.join(bar1m_dir, f'{ins}.parquet'))
    p1m = p1m.with_columns(
        pl.col("trades").alias("cnt"),
        pl.col("end_tm").cast(pl.Datetime(time_zone = 'UTC', time_unit = 'ns')).alias("end_tm")
    )
    sorders = p1m.sort('end_tm').with_columns(
            ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("returns1m"),
            ).with_columns((pl.col('returns1m') ** 2).rolling_sum(window_size = 15).alias("rv"),
                    pl.col('quote').rolling_sum(window_size = 60).alias("mquote"),
                    pl.col('cnt').rolling_sum(window_size = 60).alias("mcnt"), 
                    pl.col('buy_quote').rolling_sum(window_size = 60).alias("mbquote")
        ).group_by_dynamic(
                    index_column="end_tm",
                    every="15m",  # Set the time interval for bars (e.g., 1 minute)
                    closed="left", # Close the interval on the right
                    label = 'right',
                ).agg([
                    (pl.col('buy_vol')).sum().alias("buy_vol"),
                    pl.col("close").last().alias("close"),
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("vol").sum().alias("volume"),
                    pl.col("quote").sum().alias("quote"),
                    (pl.col('returns1m') / pl.col('rv').sqrt()).pow(3).sum().alias("skew"),
                    pl.col('rv').last().alias("rv"),
                    (pl.col("returns1m").filter(pl.col("returns1m") > 0).var()/pl.col("returns1m").var()).alias("posv"),
                    (pl.col("returns1m").filter(pl.col("returns1m") < 0).var()/pl.col("returns1m").var()).alias("negv"),
                    (pl.col("returns1m").sum()/pl.col("returns1m").abs().sum()).alias("smooth"),                    
                    (pl.col('close') > pl.col('open')).cast(int).sum().alias("upct"),
                    (pl.col('close') < pl.col('open')).cast(int).sum().alias("downct"),
                    ((pl.col('close') > pl.col('open')).cast(int) * pl.col("quote")).sum().alias("upquote"),
                    ((pl.col('close') > pl.col('open')).cast(int) * pl.col("buy_quote")).sum().alias("upbuyquote"),
                    ((pl.col('close') > pl.col('open')).cast(int) * pl.col("cnt")).sum().alias("uptrades"),
                    ((pl.col('close') < pl.col('open')).cast(int) * pl.col("quote")).sum().alias("downquote"),
                    ((pl.col('close') < pl.col('open')).cast(int) * pl.col("buy_quote")).sum().alias("downbuyquote"),
                    ((pl.col('close') < pl.col('open')).cast(int) * pl.col("cnt")).sum().alias("downtrades"),
                ]).with_columns(
                    ((pl.col("close") / pl.col("close").shift(1)) - 1).shift(1).alias("returns"),
                    ((pl.col("quote") / pl.col("volume"))).alias("vwap")
                )
    #temp2 = pl.read_parquet(os.path.join(oi_dir, f'oi_{ins}.parquet')).with_columns(pl.col("tm").alias("end_tm"))
    #sorders = sorders.join(temp2.select(['openinterest','end_tm']), left_on=["end_tm"], right_on=["end_tm"], how="left")
    #sorders = sorders.with_columns(pl.col("volume").fill_nan(0).alias("volume")).unique(subset=["end_tm"]).sort('end_tm')
    sorders = generate_features_from_config(
                sorders,
                'end_tm',
                config = cfg.prod_24h_factors
            )
    sorders.write_parquet(f"{data_dir}/mlfeatures/mlupdates/mlfeatures_{ins}.parquet")

#%% spot 1m -> 15m data
#end = datetime.now().replace(tzinfo=UTCTZ)
#start = end - timedelta(days=20)
#spot1m_dir = f'{data_dir}/spot1m'
#lookback = timedelta(days=20)
#for ins in universe:
#    update_1min_spotdata(ins, start, end, lookback, spot1m_dir)

#clear_folder(f"{spot1m_dir}/data")
#%%

bar1m_dir = f'{data_dir}/bar1m/spot/'

for ins in universe:
    data1m = pl.read_parquet(f"{bar1m_dir}/{ins}.parquet").unique(subset=["end_tm"]).sort('end_tm')
    test = data1m.with_columns(
                    pl.col('quote').rolling_quantile(0.8, window_size = 1440).alias("lquote"),
                    pl.col('buy_quote').rolling_quantile(0.8, window_size = 1440).alias("lbuyquote"),
                    (pl.col('quote') - pl.col('buy_quote')).rolling_quantile(0.8, window_size = 1440).alias("lsellquote"),
                    pl.col("end_tm").cast(pl.Datetime(time_zone = 'UTC', time_unit = 'ns')).alias("end_tm")
                ).group_by_dynamic(
                    index_column="end_tm",
                    every="15m",  # Set the time interval for bars (e.g., 1 minute)
                    closed="left", # Close the interval on the right
                    label = 'right',
                ).agg([
                    pl.col("close").last().alias("close"),
                    pl.col("quote").sum().alias("quote"),
                    pl.col("buy_quote").sum().alias("buy_quote"),
                    pl.col("trades").sum().alias("trades"),
                    pl.col("vol").sum().alias("vol"),
                    ((pl.col('quote') > pl.col('lquote')).cast(float) * pl.col('vol')).sum().alias("bigvol"),    
                    ((pl.col('buy_quote') > pl.col('lbuyquote')).cast(float) * pl.col('vol')).sum().alias("bigbuyvol"),        
                    (((pl.col('quote') - pl.col('buy_quote')) > pl.col('lsellquote')).cast(float) * pl.col('vol')).sum().alias("bigsellvol"),        
                    ((pl.col('quote') > pl.col('lquote')).cast(float) * (pl.col("close")/pl.col("open") - 1)).sum().alias("bigvolret"),    
                    ((pl.col('buy_quote') > pl.col('lbuyquote')).cast(float) *(pl.col("close")/pl.col("open") - 1)).sum().alias("bigbuyvolret"),         
                    (((pl.col('quote') - pl.col('buy_quote')) > pl.col('lsellquote')).cast(float) * (pl.col("close")/pl.col("open") - 1)).sum().alias("bigsellvolret"),        
                    ((pl.col('close') > pl.col('open'))).cast(int).sum().alias("upct"),
                    (((pl.col('close') > pl.col('open'))).cast(int) * pl.col("vol")).sum().alias("upvol"),
                    (((pl.col('close') > pl.col('open'))).cast(int) * pl.col("buy_vol")).sum().alias("upbuyvol"),
                    (((pl.col('close') < pl.col('open'))).cast(int) * (pl.col("vol") - pl.col("buy_vol"))).sum().alias("downsellvol"),
                ]
                )
    test.write_parquet(f"{data_dir}/mlfeatures/mlupdates/spot_{ins}.parquet")

#%% srdist
#run_srdist_updates(universe, start - lookback, f"{data_dir}/mlfeatures/mlupdates/", wks = [1, 2, 4, 12, 52])


allfeatures = list(cfg.prod_24h_factors.keys())
#srfeatures = ['srdist1','srdist2','srdist4','srdist12','srdist52']
spotfeatures = ['bigvol_spot4q', 'bigvol_spot16q', 'bigvol_spot32q', 'bigvol_spot96q', 'tradesz_spot4q', 'tradesz_spot16q', 'tradesz_spot32q', 'tradesz_spot96q', 'inflow_spot4q', 'inflow_spot16q', 'inflow_spot32q', 'inflow_spot96q', 'bignetvol_spot4q', 'bignetvol_spot16q', 'bignetvol_spot32q', 'bignetvol_spot96q', 'bigbuyvolret_spot4q', 'bigbuyvolret_spot16q', 'bigbuyvolret_spot32q', 'bigbuyvolret_spot96q', 'bigsellvolret_spot4q', 'bigsellvolret_spot16q', 'bigsellvolret_spot32q', 'bigsellvolret_spot96q', 'netvolratio_spot4q', 'netvolratio_spot16q', 'netvolratio_spot32q', 'netvolratio_spot96q']


#%% train models
name = f"test"
with open('train_config.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

horizon = 4
fbasedir = f"models/features{int(horizon/4)}h_{name}/"
mbasedir = f"models/models_{int(horizon/4)}h_{name}/"
feature_dir = f"{fbasedir}{ins}/"
if not os.path.exists(fbasedir):
    os.mkdir(fbasedir)
if not os.path.exists(mbasedir):
    os.mkdir(mbasedir)
if not os.path.exists(feature_dir):
    os.mkdir(feature_dir)

feat1 = pl.read_parquet(f"{data_dir}/mlfeatures/mlupdates/mlfeatures_{ins}.parquet")
feat2 = pl.read_parquet(f"{data_dir}/mlfeatures/mlupdates/spot_{ins}.parquet")
feat2 = feat2.sort("end_tm").with_columns(
            bigvol_spot4q = pl.col("bigvol").rolling_sum(4),
            bigvol_spot16q = pl.col("bigvol").rolling_sum(16),
            bigvol_spot32q = pl.col("bigvol").rolling_sum(32),
            bigvol_spot96q = pl.col("bigvol").rolling_sum(96),
            tradesz_spot4q = (pl.col("quote").rolling_sum(4) / pl.col("trades").rolling_sum(4)),
            tradesz_spot16q = (pl.col("quote").rolling_sum(16) / pl.col("trades").rolling_sum(16)),
            tradesz_spot32q = (pl.col("quote").rolling_sum(32) / pl.col("trades").rolling_sum(32)),
            tradesz_spot96q = (pl.col("quote").rolling_sum(96) / pl.col("trades").rolling_sum(96)),
            inflow_spot4q = (pl.col("buy_quote") *2 - pl.col("quote")).rolling_sum(4),
            inflow_spot16q = (pl.col("buy_quote") *2 - pl.col("quote")).rolling_sum(16),
            inflow_spot32q = (pl.col("buy_quote") *2 - pl.col("quote")).rolling_sum(32),
            inflow_spot96q = (pl.col("buy_quote") *2 - pl.col("quote")).rolling_sum(96),
            bignetvol_spot4q = (pl.col("bigbuyvol") - pl.col("bigsellvol")).rolling_sum(4),
            bignetvol_spot16q = (pl.col("bigbuyvol") - pl.col("bigsellvol")).rolling_sum(16),
            bignetvol_spot32q = (pl.col("bigbuyvol") - pl.col("bigsellvol")).rolling_sum(32),
            bignetvol_spot96q = (pl.col("bigbuyvol") - pl.col("bigsellvol")).rolling_sum(96),
            bigbuyvolret_spot4q = pl.col("bigbuyvolret").rolling_sum(4),
            bigbuyvolret_spot16q = pl.col("bigbuyvolret").rolling_sum(16),
            bigbuyvolret_spot32q = pl.col("bigbuyvolret").rolling_sum(32),
            bigbuyvolret_spot96q = pl.col("bigbuyvolret").rolling_sum(96),
            bigsellvolret_spot4q = pl.col("bigsellvolret").rolling_sum(4),
            bigsellvolret_spot16q = pl.col("bigsellvolret").rolling_sum(16),
            bigsellvolret_spot32q = pl.col("bigsellvolret").rolling_sum(32),
            bigsellvolret_spot96q = pl.col("bigsellvolret").rolling_sum(96),
            netvolratio_spot4q = ((pl.col("upbuyvol") - pl.col("downsellvol"))/pl.col("vol")).rolling_sum(4),
            netvolratio_spot16q = ((pl.col("upbuyvol") - pl.col("downsellvol"))/pl.col("vol")).rolling_sum(16),
            netvolratio_spot32q = ((pl.col("upbuyvol") - pl.col("downsellvol"))/pl.col("vol")).rolling_sum(32),
            netvolratio_spot96q = ((pl.col("upbuyvol") - pl.col("downsellvol"))/pl.col("vol")).rolling_sum(96),
        )
temp = feat1.join(feat2, on = 'end_tm', how = 'left').sort("end_tm")
iallfeats = allfeatures + spotfeatures
temp, _ = transformFT(temp, pl.DataFrame(), iallfeats, save_ecdf = True, outputfolder = feature_dir)
temp = generate_targets(
    temp, 
    'end_tm',
    'close',
    [horizon],
    save_ecdf = True,
    outputfolder = feature_dir
    )
ff = feature_selection_2step(temp, iallfeats, f'ret_T{horizon}', corr_drop_threshold = 0.8, corr_merge_threshold = 0.6, save_grouping = True, outputfolder = feature_dir)
features = [ft for ft in ff.columns if ft in iallfeats or 'cluster_' in ft]
with open(f'{feature_dir}/feature_list.pkl', 'wb') as f:
    pickle.dump(features, f)

lgbmodel, params = prepare_gru()
params['feature_size'] = len(features)
params['nb_epoch'] = 50
params['learning_rate'] = 1e-3
params['dropout'] = 0.3
params['patience'] = 10
params['seq_len'] = horizon * 2
params['l2_lambda'] = 1e-4
params['min_delta'] = 1e-3
params['weight_decay'] = 0.004
train_config['training']['max_hp_evals'] = 30


model_path =f'{mbasedir}{ins}/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
if True:
    train_classifier(
        training_set = ff,
        target = f'ret_T{horizon}',
        features = features,
        modelClass = lgbmodel,
        params = params,
        save_path = model_path,
        config = train_config,
        name = name,
        sampler = downsample_sequences
    )
if False:
    train_classifier_simple(
                training_set = ff,
                target = f'ret_T{horizon}',
                features = features,
                modelClass = lgbmodel,
                params = params,
                save_path = model_path,
                config = train_config,
                name = name,
                batch_size = 512,
                use_sw = True
        )