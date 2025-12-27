import polars as pl
import pandas as pd
import numpy as np
import kmeans1d
from kneed import KneeLocator
from datetime import timedelta


def gen_SR_distance_week(close, vwp, start, week_cnt): # the span is how many weeks
    time = []
    data = []
    date = pd.Timestamp(start.date(), tz = 'UTC')
    while date < vwp.index[-1]:
        SSE = []
        flag = 0
        if week_cnt == 52:
            cluster_num = 4
            df = vwp[(vwp.index >= date - pd.Timedelta(f"{7*week_cnt}d")) & (vwp.index < date)]
            if len(df) <= 10:
                flag = 1
        else:
            for cluster_num in range(2, 9):
                df = vwp[(vwp.index >= date - pd.Timedelta(f"{7*week_cnt}d")) & (vwp.index < date)]
                if len(df) > 10:
                    clusters, centroids = kmeans1d.cluster(df, cluster_num)
                    SSE.append(sum([(df.iloc[i] - centroids[clusters[i]])**2 for i in range(len(df))]))
                else:
                    flag = 1
                    break
        if flag == 0:
            if week_cnt == 52:
                clusters, centroids = kmeans1d.cluster(df, 4)
            else:
                kl = KneeLocator(range(2, 9), SSE, curve="convex", direction="decreasing")
                if not kl.elbow:
                    print(f" {date} error, use 4 as default")
                    optimal_num = 4
                else:
                    optimal_num = kl.elbow
                clusters, centroids = kmeans1d.cluster(df, optimal_num)
            #change back to time weighted price
            twp = close[(close.index >= date) & (close.index < date + pd.Timedelta("1d"))]
            for i in range(len(twp)):
                time.append(twp.index[i])
                # change the distance to directional, with positive value being the current price is higher than the nearest core
                temp = sorted([(abs(twp.iloc[i] - centroids[j]), centroids[j]) for j in range(len(centroids))])
                # closest core
                c1 = temp[0][1]   
                # second closest core
                c2 = temp[1][1]
                dif_thre = min(abs(c1 - c2)/2, 0.2*c1) # this 0.2 is rather arbitrary
                dif = min(abs(twp.iloc[i] - c1), dif_thre)
                data.append((1 - dif/dif_thre) * np.sign(twp.iloc[i] - c1)) # In the end, I choose the linear function
        date += pd.Timedelta("1d")
    return pd.Series(data, index = time)


def run_srdist_updates(universe, start, data_dir, wks = [1, 2, 4, 12, 52]):
    cols = ['close',] + [f'srdist{wk}' for wk in wks]
    for ins in universe:
        prev = pd.read_parquet(f"{data_dir}/features/mlfeatures_{ins}.parquet").dropna(subset = ['close','vwap']).set_index('end_tm').sort_index()[['close','vwap']]
        temp = pd.read_parquet(f"{data_dir}/mlupdates/mlfeatures_{ins}.parquet").dropna(subset = ['close','vwap']).set_index('end_tm').sort_index()[['close','vwap']]
        cut = prev.index[-1] + pd.to_timedelta('1m')
        temp = pd.concat([prev, temp[cut:]])
        temp = temp[~temp.index.duplicated()].sort_index()
        for wk in wks:
            rs = gen_SR_distance_week(temp['close'], temp['vwap'], start, wk)
            temp[f'srdist{wk}'] = rs
            print(f'complete {ins} wk {wk}')
        temp = temp[cols]
        temp['symbol'] = ins
        temp.to_parquet(f"{data_dir}/mlupdates/srdist/srdist_{ins}.parquet")

        
