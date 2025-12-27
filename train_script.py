import polars as pl
import cfg
from train_utils import(
    generate_features_from_config,
    generate_training_dates,
)

#%%% generate features
training_dates = generate_training_dates('2025-03-01', '2025-03-30')
train_cutoff = training_dates[0]
#for ins in subuni:
ins = 'BTCUSDT'
data1m = pl.read_parquet(f"{cfg.data_folder}/bar1m/futures/{ins}.parquet").unique(subset=["end_tm"]).sort('end_tm')
test = data1m.with_columns((pl.col("buy_quote").cast(float)/pl.col("trades").cast(float) > 3000).alias("lorder"),
                (pl.col("buy_quote").cast(float)/pl.col("trades").cast(float) < 1000).alias("sorder")
        ).sort('end_tm').with_columns(
            pl.col("open").cast(float).alias("open"),
            pl.col("close").cast(float).alias("close"),
            pl.col("high").cast(float).alias("high"),
            pl.col("low").cast(float).alias("low"),
            pl.col("vol").cast(float).alias("vol"),
            pl.col("buy_vol").cast(float).alias("buy_vol"),
            pl.col("quote").cast(float).alias("quote"),
            pl.col("buy_quote").cast(float).alias("buy_quote"),
            pl.col("trades").cast(float).alias("cnt"),
    ).with_columns(
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
                by = 'symbol'
            ).agg([
                (pl.col('buy_quote') * pl.col('sorder').cast(float)).sum().alias("sbuy_quote"),
                (pl.col('buy_vol')).sum().alias("buy_vol"),
                (pl.col('quote') * pl.col('sorder').cast(float)).sum().alias("squote"),
                pl.col("close").last().alias("close"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("vol").sum().alias("volume"),
                pl.col("quote").sum().alias("quote"),
                (pl.col('buy_quote') * pl.col('lorder').cast(float)).sum().alias("lbuy_quote"),
                (pl.col('quote') * pl.col('lorder').cast(float)).sum().alias("lquote"),
                (pl.col('returns1m') / pl.col('rv').sqrt()).pow(3).sum().alias("skew"),
                pl.col('rv').last().alias("rv"),
                pl.corr(pl.col("close"), pl.col("cnt")).alias("cpm"),
                pl.corr(pl.col("close"), pl.col("quote")).alias("cpv"),
                pl.corr(pl.col("close"), pl.col("buy_quote")).alias("cpbv"),
                pl.corr(pl.col("close"), pl.col("quote")/pl.col("cnt")).alias("cpvmr"),
                pl.corr(pl.col("returns1m"), pl.col("cnt")).alias("crm"),
                pl.corr(pl.col("returns1m"), pl.col("quote")).alias("crv"),
                pl.corr(pl.col("returns1m"), pl.col("buy_quote")).alias("crbv"),
                pl.corr(pl.col("returns1m"), pl.col("quote")/pl.col("cnt")).alias("crvmr"),
                pl.corr(pl.col("close"), pl.col("cnt") - pl.col("mcnt")).alias("cpac"),
                pl.corr(pl.col("close"), pl.col("quote") - pl.col("mquote")).alias("cpav"),
                pl.corr(pl.col("close"), pl.col("buy_quote") - pl.col("mbquote")).alias("cpabv"),
                pl.corr(pl.col("close"), (pl.col("quote")- pl.col("mquote"))/pl.col("cnt")).alias("cpavmr"),
                pl.corr(pl.col("returns1m"), pl.col("cnt") - pl.col("mcnt")).alias("cram"),
                pl.corr(pl.col("returns1m"), pl.col("quote") - pl.col("mquote")).alias("crav"),
                pl.corr(pl.col("returns1m"), pl.col("buy_quote") - pl.col("mbquote")).alias("crabv"),
                pl.corr(pl.col("returns1m"), (pl.col("quote")- pl.col("mquote"))/pl.col("cnt")).alias("cravmr"),
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
            ).filter(pl.col("end_tm") <= train_cutoff)
test = test.join(
    dataoi5m.select(['openinterest','tm','symbol']), left_on=["symbol", "end_tm"], right_on=["symbol", "tm"], how="left"
    ).with_columns(pl.col("volume").fill_nan(0).alias("volume"))
#sorders.write_parquet(f'production/bar15m/{ins}.parquet')
test = test.unique(
    subset=['symbol','end_tm'],         # Columns to check for duplicates (default: all columns)
    keep="last",        # "first", "last", or "any"
).sort('end_tm')
# 
# add srdist
srdist = get_srdist(enddate, ins, test['close'])
test = test.with_columns(**srdist)
temp = generate_features_from_config(
                temp,
                'end_tm',
                config = cfg.prod_24h_factors
            )

#temp.write_parquet(f"{cfg.data_folder}/mlfeatures/mlfeatures_{ins}.parquet")


    