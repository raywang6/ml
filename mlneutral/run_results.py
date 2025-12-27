import polars as pl
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from parquet_reader import parallel_read_parquets

def to_weights(x, cut = 0.3):
    x = x.subtract(x.mean(axis=1),axis=0).div(x.std(axis=1),axis=0)
    xrank = x.rank(axis=1, pct = True)
    top = x.mask(xrank < 1 - cut, 0)
    top = 1.0/(1.0 + np.exp(-top))
    bot = x.mask(xrank > cut, 0)
    bot = 1.0/(1.0 + np.exp(-bot))
    wt = top + bot - 1.
    return wt

def csr(comb):
    return round(comb.mean()/comb.std() * np.sqrt(365*24),2)

#

preds = []
for ifile in glob.glob('/home/moneyking/projects/mlframework/mlneutral/walkforward_results_weight_lr5e-2_early20/predictions/pred_*.parquet'):
    tmp = pl.read_parquet(ifile).select(['datetime','symbol','prediction1','prediction2','prediction'])
    preds.append(tmp)
#%%
preds = pl.concat(preds).to_pandas()

#%%
path = '/home/ray/projects/data/sync_folder/factor/norm_1h'
res = parallel_read_parquets(
    folder=path,
    columns=['__index_level_0__', 'close'],
    pattern="*.parquet",
    n_workers=8,
    symbol_prefix="perp_",
).rename(mapping = {"timestamp": 'datetime'})
res = res.select(['datetime', 'symbol', 'close']).pivot(
        on="symbol",
        index="datetime",
        values="close"
).sort('datetime')
ret = res.to_pandas().set_index('datetime').pct_change()

#%%
sgl1 = pd.pivot_table(preds, index = 'datetime', columns = 'symbol', values = 'prediction1')
sgl2 = pd.pivot_table(preds, index = 'datetime', columns = 'symbol', values = 'prediction2')

#%%
vwindow = 1
#%% simple
ns1 = sgl1.subtract(sgl1.mean(axis=1),axis=0)
ns1 = ns1.div(ns1.abs().sum(axis=1),axis=0)

ns2 = sgl2.subtract(sgl2.mean(axis=1),axis=0)
ns2 = ns2.div(ns2.abs().sum(axis=1),axis=0)

sglc = (sgl1+sgl2)/2
nsc = sglc.subtract(sglc.mean(axis=1),axis=0)
nsc = nsc.div(nsc.abs().sum(axis=1),axis=0)

sgla = sgl1.mask(np.logical_and(sgl1 < 0, sgl2 > 0), 0)
sgla = sgla.mask(np.logical_and(sgl2 < 0, sgl1 > 0), 0)
nsa = sgla.subtract(sgla.mean(axis=1),axis=0)
nsa = nsa.div(nsa.abs().sum(axis=1),axis=0)

z1 = (ns1.shift() * ret).sum(axis = 1).loc[ns1.index]
ns1 = ns1.div(np.sqrt(365*24)*z1.ewm(168*vwindow).std()/0.15,axis=0)

z2 = (ns2.shift() * ret).sum(axis = 1).loc[ns2.index]
ns2 = ns2.div(np.sqrt(365*24)*z2.ewm(168*vwindow).std()/0.15,axis=0)

zc = (nsc.shift() * ret).sum(axis = 1).loc[nsc.index]
nsc = nsc.div(np.sqrt(365*24)*zc.ewm(168*vwindow).std()/0.15,axis=0)

za = (nsa.shift() * ret).sum(axis = 1).loc[nsc.index]
nsa = nsa.div(np.sqrt(365*24)*za.ewm(168*vwindow).std()/0.15,axis=0)

smooth = (ns1 * 0.5 + ns2 * 0.5).rolling(24).mean()

z1 = (ns1.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
z2 = (ns2.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
zc = (nsc.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
za = (nsa.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
zsmt = (smooth.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
comb = (z1+z2)/2
plt_df = pd.DataFrame({f'm1 {csr(z1)}': z1, f'm2 {csr(z2)}': z2, f'mc {csr(comb)}': comb, 
f'sglc {csr(zc)}': zc, f'sgla {csr(za)}': za,
f'sth {csr(zsmt)}': zsmt}
).resample('1d').sum()


fig,ax=plt.subplots()
plt_df.cumsum().plot(ax=ax, alpha = 0.5)
plt_df[f'sth {csr(zsmt)}'].cumsum().plot(ax=ax, alpha = 1)
fig.savefig("simple.png")

#%% rank
ns1 = sgl1.rank(axis=1,pct=True)
ns1 = ns1.subtract(ns1.mean(axis=1),axis=0)
ns1 = ns1.div(ns1.abs().sum(axis=1),axis=0)

ns2 = sgl2.rank(axis=1,pct=True)
ns2 = ns2.subtract(ns2.mean(axis=1),axis=0)
ns2 = ns2.div(ns2.abs().sum(axis=1),axis=0)

sglc = (sgl1+sgl2)/2
nsc = sglc.rank(axis=1,pct=True)
nsc = nsc.subtract(nsc.mean(axis=1),axis=0)
nsc = nsc.div(nsc.abs().sum(axis=1),axis=0)

sgla = sgl1.mask(np.logical_and(sgl1 < 0, sgl2 > 0), 0)
sgla = sgla.mask(np.logical_and(sgl2 < 0, sgl1 > 0), 0)
nsa = sgla.rank(axis=1,pct=True)
nsa = nsa.subtract(nsa.mean(axis=1),axis=0)
nsa = nsa.div(nsa.abs().sum(axis=1),axis=0)


z1 = (ns1.shift() * ret).sum(axis = 1).loc[ns1.index]
ns1 = ns1.div(np.sqrt(365*24)*z1.ewm(168 * vwindow).std()/0.15,axis=0)

z2 = (ns2.shift() * ret).sum(axis = 1).loc[ns2.index]
ns2 = ns2.div(np.sqrt(365*24)*z2.ewm(168* vwindow).std()/0.15,axis=0)

zc = (nsc.shift() * ret).sum(axis = 1).loc[nsc.index]
nsc = nsc.div(np.sqrt(365*24)*zc.ewm(168* vwindow).std()/0.15,axis=0)

za = (nsa.shift() * ret).sum(axis = 1).loc[nsc.index]
nsa = nsa.div(np.sqrt(365*24)*za.ewm(168* vwindow).std()/0.15,axis=0)

smooth = (ns1 * 0.5 + ns2 * 0.5).rolling(24).mean()

z1 = (ns1.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
z2 = (ns2.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
zc = (nsc.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
za = (nsa.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
zsmt = (smooth.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
comb = (z1+z2)/2
plt_df = pd.DataFrame({f'm1 {csr(z1)}': z1, f'm2 {csr(z2)}': z2, f'mc {csr(comb)}': comb, f'sglc {csr(zc)}': zc, f'sgla {csr(za)}': za, f'sth {csr(zsmt)}': zsmt}).resample('1d').sum()


fig,ax=plt.subplots()
plt_df.cumsum().plot(ax=ax)
fig.savefig("rank.png")


#%% complex
ns1 = to_weights(sgl1)

ns2 = to_weights(sgl2)

sglc = (sgl1+sgl2)/2
nsc = to_weights(sglc)

sgla = sgl1.mask(np.logical_and(sgl1 < 0, sgl2 > 0), 0)
sgla = sgla.mask(np.logical_and(sgl2 < 0, sgl1 > 0), 0)
nsa = to_weights(sgla)

z1 = (ns1.shift() * ret).sum(axis = 1).loc[ns1.index]
ns1 = ns1.div(np.sqrt(365*24)*z1.ewm(168*2).std()/0.15,axis=0)

z2 = (ns2.shift() * ret).sum(axis = 1).loc[ns2.index]
ns2 = ns2.div(np.sqrt(365*24)*z2.ewm(168*2).std()/0.15,axis=0)

zc = (nsc.shift() * ret).sum(axis = 1).loc[nsc.index]
nsc = nsc.div(np.sqrt(365*24)*zc.ewm(168*2).std()/0.15,axis=0)

za = (nsa.shift() * ret).sum(axis = 1).loc[nsc.index]
nsa = nsa.div(np.sqrt(365*24)*za.ewm(168*2).std()/0.15,axis=0)


z1 = (ns1.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
z2 = (ns2.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
zc = (nsc.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
za = (nsa.shift() * ret).sum(axis = 1)['2023-09-01':'2025-07-02']
comb = (z1+z2)/2
plt_df = pd.DataFrame({f'm1 {csr(z1)}': z1, f'm2 {csr(z2)}': z2, f'mc {csr(comb)}': comb, f'sglc {csr(zc)}': zc, f'sgla {csr(za)}': za}).resample('1d').sum()


fig,ax=plt.subplots()
plt_df.cumsum().plot(ax=ax)
fig.savefig("softmax.png")
