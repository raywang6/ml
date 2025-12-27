

import polars as pl
import numpy as np

def expr_moito(oiname, amtname, window = 72):
    return (pl.col(oiname).rolling_mean(window) / pl.col(amtname).rolling_mean(window))

def expr_avgoi(oiname, window = 72):
    return (pl.col(oiname).rolling_mean(window))

def expr_oiskew(oiname, window = 72):
    return pl.col(oiname).pct_change().clip(-1, 10).rolling_skew(window)

def expr_oistd(oiname, window = 72):
    return pl.col(oiname).pct_change().clip(-1, 10).rolling_std(window)

def expr_oitrendratio(oiname, window = 72):
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    return doi.rolling_sum(window) / doi.abs().rolling_sum(window)
    
def expr_oidrop(oiname, window = 168):
    ath = pl.col(oiname).rolling_max(window)
    return (ath - pl.col(oiname)) / ath

def expr_oiretcorr(oiname, cname, window = 8):
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    ret = pl.col(cname).pct_change().clip(-1, 2)
    return pl.rolling_corr(ret,doi, window_size = window)

def expr_oiautocorr(oiname, window = 672):
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    bf = doi.rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

def expr_oilevelautocorr(oiname, window = 672):
    bf = pl.col(oiname).rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

#%%
def expr_oiwp(oiname, cname, window=20):
    return (pl.col(cname) * pl.col(oiname)).rolling_mean(window) /  pl.col(oiname).rolling_mean(window)

def expr_oiamtcorr(oiname, aname, window=5):
    return pl.rolling_corr(pl.col(aname).diff(),pl.col(oiname).diff(), window_size = window)

def expr_oidivergence(oiname, cname, window=10):
    vol_ma = pl.col(cname).rolling_mean(window)
    oi_ma = pl.col(oiname).rolling_mean(window)
    return (vol_ma / oi_ma).pct_change(window)


def expr_doito(oiname, amtname, window = 72):
    return (pl.col(oiname).diff(window) / pl.col(amtname).rolling_mean(window))


# high pos ret, low pos ret, diff;
# high pos amt, etc.
# pos inc ret

def expr_amtlowoi(oiname, amtname, window = 96):
    low = pl.col(oiname).rolling_quantile(0.2, window_size = window)
    aal = (pl.col(amtname) * (pl.col(oiname) < low).cast(int)).rolling_sum(window)
    return aal / pl.col(amtname).rolling_sum(window)

def expr_amthighoi(oiname, amtname, window = 96):
    low = pl.col(oiname).rolling_quantile(0.8, window_size = window)
    aal = (pl.col(amtname) * (pl.col(oiname) > low).cast(int)).rolling_sum(window)
    return aal / pl.col(amtname).rolling_sum(window)

def expr_amtimbhloi(oiname, amtname, window = 96):
    low = pl.col(oiname).rolling_quantile(0.2, window_size = window)
    high = pl.col(oiname).rolling_quantile(0.8, window_size = window)
    aal = (pl.col(amtname) * (pl.col(oiname) < low).cast(int)).rolling_sum(window)
    aah = (pl.col(amtname) * (pl.col(oiname) > high).cast(int)).rolling_sum(window)
    return (aah - aal) / (aah + aal)

def expr_retlowoi(oiname, cname, window = 96):
    ret = pl.col(cname).pct_change().clip(-1, 2)
    low = pl.col(oiname).rolling_quantile(0.2, window_size = window)
    aal = (ret * (pl.col(oiname) < low).cast(int)).rolling_sum(window)
    return aal

def expr_rethighoi(oiname, cname, window = 96):
    ret = pl.col(cname).pct_change().clip(-1, 2)
    low = pl.col(oiname).rolling_quantile(0.8, window_size = window)
    aal = (ret * (pl.col(oiname) > low).cast(int)).rolling_sum(window)
    return aal

def expr_retimbhloi(oiname, cname, window = 96):
    ret = pl.col(cname).pct_change().clip(-1, 2)
    low = pl.col(oiname).rolling_quantile(0.2, window_size = window)
    high = pl.col(oiname).rolling_quantile(0.8, window_size = window)
    aal = (ret * (pl.col(oiname) < low).cast(int)).rolling_sum(window)
    aah = (ret * (pl.col(oiname) > high).cast(int)).rolling_sum(window)
    return (aah - aal)

def expr_retincoi(oiname, cname, window = 96):
    ret = pl.col(cname).pct_change().clip(-1, 2)
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    high = doi.rolling_quantile(0.8, window_size = window)
    aal = (ret * (doi > high).cast(int)).rolling_sum(window)
    return aal

def expr_retdecoi(oiname, cname, window = 96):
    ret = pl.col(cname).pct_change().clip(-1, 2)
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    low = doi.rolling_quantile(0.2, window_size = window)
    aal = (ret * (doi < low).cast(int)).rolling_sum(window)
    return aal
    
def expr_retimbhldoi(oiname, cname, window = 96):
    ret = pl.col(cname).pct_change().clip(-1, 2)
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    high = doi.rolling_quantile(0.8, window_size = window)
    low = doi.rolling_quantile(0.2, window_size = window)
    aal = (ret * (doi < low).cast(int)).rolling_sum(window)
    aah = (ret * (doi > high).cast(int)).rolling_sum(window)
    return (aah - aal)

def expr_amtincoi(oiname, amtname, window = 96):
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    high = doi.rolling_quantile(0.8, window_size = window)
    aal = (pl.col(amtname) * (doi > high).cast(int)).rolling_sum(window)
    return aal / pl.col(amtname).rolling_sum(window)

def expr_amtdecoi(oiname, amtname, window = 96):
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    low = doi.rolling_quantile(0.2, window_size = window)
    aal = (pl.col(amtname)  * (doi < low).cast(int)).rolling_sum(window)
    return aal / pl.col(amtname).rolling_sum(window)
    
def expr_amtimbhldoi(oiname, amtname, window = 96):
    doi = pl.col(oiname).pct_change().clip(-1, 10)
    high = doi.rolling_quantile(0.8, window_size = window)
    low = doi.rolling_quantile(0.2, window_size = window)
    aah = (pl.col(amtname) * (doi > high).cast(int)).rolling_sum(window)
    aal = (pl.col(amtname) * (doi < low).cast(int)).rolling_sum(window)
    return (aah - aal) / (aah + aal)


def expr_macd(cname, window = 9, swindow = 12, lwindow = 26):
    return ((pl.col(cname).ewm_mean(span=swindow) - pl.col(cname).ewm_mean(span=lwindow)) - (pl.col(cname).ewm_mean(span=swindow) - pl.col(cname).ewm_mean(span=lwindow)).ewm_mean(span=window))
    

def expr_smooth(cname, window):
    return pl.col(cname).rolling_mean(window)

def expr_bollingup(cname, window, band = 1):
    return (pl.col(cname).rolling_mean(window) + band * pl.col(cname).rolling_std(window))

# rev 
def expr_rev_vwap(cname, lname, hname, vname, window = 672, swindow = 8, cap = 0.2):
    volp = (pl.col(cname) + pl.col(lname) + pl.col(hname))/3 * pl.col(vname)
    volp2w = volp.rolling_sum(window) / pl.col(vname).rolling_sum(window)
    ma8 = pl.col(cname).rolling_mean(swindow)
    scale = cap - (ma8 / volp2w - 1).abs().clip( upper_bound = cap)
    sign = (ma8 - volp2w).sign()
    return sign * scale

def expr_rev_boll(cname, vname, window = 672, vswindow = 24, vlwindow = 168, awindow = 8, thres = 2.5):
    band = pl.col(cname).rolling_mean(window) 
    uband = band + band.rolling_std(window) * thres
    logv = pl.col(vname).clip(lower_bound = 1).log()
    lvcon = (logv.rolling_mean(vswindow) - logv.rolling_mean(vlwindow)) /logv.rolling_std(vlwindow)
    scale = ((uband - pl.col(cname)).clip( lower_bound = 0) / (uband - band)).abs()
    sign = (pl.col(cname) - band).sign()
    cond = (lvcon > 1).cast(int)
    return scale * sign * cond.rolling_max(awindow)


def expr_rev_candel(oname, cname, hname, lname, window = 4):
    phigh = pl.col(hname).rolling_max(window)
    popen = pl.col(oname).shift(window-1)
    plow = pl.col(lname).rolling_min(window)
    return pl.when(
            (phigh > popen) &
            (pl.col(cname).shift(window) < popen.shift(window)) &
            (phigh - pl.col(cname) > pl.col(cname) - plow)
        ).then(-1).when(
            (pl.col(cname) < popen) &
            (pl.col(cname).shift(window) > popen.shift(window)) &
            (pl.col(cname) - plow > phigh - pl.col(cname))
        ).then(1).otherwise(0)


def expr_vpconcentration(cname, amtname, window = 20, delta_pct = 0.01):
    vpc = pl.lit(0)
    for i in range(1, window+1):
        vpc += (pl.when((pl.col(cname) >= pl.col(cname).shift(i) * (1 - delta_pct))).then(pl.col(amtname).shift(i)).otherwise(0) - pl.when((pl.col(cname) <= pl.col(cname).shift(i) * (1 + delta_pct))).then(pl.col(amtname).shift(i)).otherwise(0))
    return vpc / pl.col(amtname).rolling_sum(window).shift()


def expr_vppoc(cname, hname, lname, amtname, window = 20, bins = 20):
    hpr = pl.col(hname).rolling_max(window)
    lpr = pl.col(lname).rolling_min(window)
    idbins = ((pl.col(cname) - lpr) / (hpr - lpr + 1e-9) * bins).floor().cast(int)
    vbins = []
    for i in range(0, bins):
        vbins.append(pl.when(idbins == i).then(pl.col(amtname)).otherwise(0).rolling_sum(window))
    return pl.concat_list(vbins).list.arg_max()

def expr_rev_amtsupp(cname, hname, lname, amtname, window=168):
    return (
        (pl.col(amtname) * (pl.col(cname) - pl.col(lname))).rolling_sum(window) /
        (pl.col(amtname) * (pl.col(hname) - pl.col(lname))).rolling_sum(window)
    )

def expr_rev_pullback(cname, lname, cwindow=20, rwindow=12):
    plow = pl.col(lname).rolling_min(cwindow)
    return (
        (pl.col(cname) - plow) / (plow + 1e-9)
    ).rolling_mean(rwindow)


def expr_rsi(cname, window = 24):
    pdiff = pl.col(cname).diff()
    upc = pl.when(pdiff < 0).then(pl.lit(0)).otherwise(pdiff)
    dpc = pl.when(pdiff > 0).then(pl.lit(0)).otherwise(pdiff * -1)
    supc = upc.ewm_mean(span = window)
    sdpc = dpc.ewm_mean(span = window)
    return - sdpc / supc


def expr_rev_lbreak(lname, window=24, thres = 0.005):
    support_level = pl.col(lname).rolling_min(window)
    return (
        (pl.col(lname) <= support_level * (1.+thres)).cast(int)
        .rolling_sum(window)
    ) 

def expr_rev_hbreak(hname, window=24, thres = 0.005):
    support_level = pl.col(hname).rolling_max(window)
    return (
        (pl.col(hname) >= support_level * (1.-thres)).cast(int)
        .rolling_sum(window)
    ) 

# mom
def expr_hlrange(hname, lname, cname, window = 800, ref_window = 4):
    high = pl.col(hname).rolling_max(window) 
    pos = 0.5 - (high - pl.col(cname).rolling_mean(ref_window)) / (high - pl.col(lname).rolling_min(window) + pl.lit(1e-8))
    return pos


def expr_zclose(cname, window = 168):
    bolldev = (pl.col(cname) - pl.col(cname).rolling_mean(window))/pl.col(cname).rolling_std(window)
    bolldev = bolldev.clip(lower_bound = -5, upper_bound = 5)
    return bolldev

def expr_lpm(cname, q = 0.75, window = 8):
    ret = pl.col(cname).pct_change()
    lpm = ret.rolling_quantile(quantile = q, interpolation = 'nearest', window_size = window) - ret
    lpm = pl.when(lpm <= 0).then(pl.lit(0)).otherwise(lpm) **2
    return lpm

def expr_momclock(cname, dtname, window = 24):
    ret = pl.col(cname).pct_change()
    hour = pl.col(dtname).dt.hour()
    return ret.rolling_mean(window).over(hour)

def expr_cosmom(cname, dtname, cycle = 2, window = 24, shift = 0):
    ret = pl.col(cname).pct_change()
    hour = pl.col(dtname).dt.hour()
    hscale = np.cos(hour * ((cycle/12 + shift/2) * 2 * np.pi))
    return (hscale * ret).rolling_mean(window)

def expr_streak(cname, window = 168):
    ret = pl.col(cname).pct_change()
    return ret.rolling_map(compute_streak, window_size = window)

def compute_streak(returns):
    streak, current_streak = 0, 0
    for return_value in returns:
        if return_value > 0:
            current_streak = current_streak + 1 if current_streak >= 0 else 1
        elif return_value < 0:
            current_streak = current_streak - 1 if current_streak <= 0 else -1
        else:
            current_streak = 0
        streak = streak if abs(streak) > abs(current_streak) else current_streak
    return streak


# ret stats
def expr_amtskew(amtname, window = 8):
    bf = pl.col(amtname).rolling_skew(window) * -1
    return bf

def expr_amtstd(amtname, window = 24):
    bf = pl.col(amtname).rolling_std(window)
    return bf

def expr_trendratio(cname, window = 24):
    ret = pl.col(cname).pct_change()
    return ret.rolling_sum(window) / ret.abs().rolling_sum(window)

def expr_pricemdd(cname, window = 168):
    ath = pl.col(cname).rolling_max(window)
    return (ath - pl.col(cname)) / ath
    
def expr_upvar(cname, window = 168):
    ret = pl.col(cname).pct_change()
    return pl.when(ret > 0).then(ret).otherwise(None).rolling_std(
            window_size=window,
            min_periods=1,         # adjust min_periods if needed
        )

# amt stats
def expr_zamihudilliq(amtname, cname, window = 8):
    ret = pl.col(cname).pct_change()
    bf = (ret / pl.col(amtname)).abs().rolling_mean(window) * -1
    return bf


def expr_amihudilliqhl(hname, lname, amtname, window=20):
    return (
        ((pl.col(hname) - pl.col(lname)).abs() / 
        (pl.col(amtname)).rolling_mean(window) + 1e-9)
    )


def expr_amtautocorr(amtname, window=14):
    return pl.rolling_corr(pl.col(amtname), pl.col(amtname).shift(), window_size = window)


def expr_mamt(amtname, window = 8):
    return (pl.col(amtname).rolling_mean(window) - pl.col(amtname).rolling_mean(window * 7)) / pl.col(amtname).rolling_std(window * 7)


def expr_amtinflow(cname, amtname, window = 168):
    ret = pl.col(cname).pct_change()
    retup= pl.when(ret > 0).then(ret.pow(2)).otherwise(0).rolling_sum(window)
    retdown= pl.when(ret < 0).then(ret.pow(2)).otherwise(0).rolling_sum(window)
    bf = (retup - retdown) / (retup + retdown).clip(1e-6) * (pl.col(amtname)/pl.col(amtname).shift(window) - 1).clip(-1,5)
    return bf

def expr_pvcorr(cname, amtname, window=14):
    price_trend = pl.col(cname).diff(window)
    volume_trend = pl.col(amtname).diff(window)
    return pl.rolling_corr(price_trend, volume_trend, window_size = window)

def expr_amtcum(oname, cname, amtname, window=20):
    return ((pl.col(cname) - pl.col(oname)) * pl.col(amtname)).rolling_sum(window)


def expr_amtcluster(amtname, window=24):
    intra_hour_vol = pl.col(amtname) / pl.col(amtname).rolling_sum(window)
    return -(intra_hour_vol * intra_hour_vol.log()).replace([-float('inf'),float('inf')], 0).rolling_sum(window)

def expr_shadowsz(oname, cname, hname, lname, window=5):
    body_size = (pl.col(cname) - pl.col(oname)).abs()
    shadow_ratio = (pl.col(hname) - pl.col(lname)) / (body_size + 1e-9)
    return shadow_ratio.rolling_mean(window)

def expr_amtresonance(cname, amtname, window=14):
    return (pl.col(cname).pct_change() * pl.col(amtname).pct_change().abs()).rolling_mean(window)


# vol
def expr_volskew(cname, vwindow = 24, window = 72):
    ret = pl.col(cname).pct_change()
    vol = ret.clip(lower_bound = -0.3, upper_bound=0.3).ewm_std(half_life = vwindow)
    return vol.rolling_skew(window)


def expr_volretcorr(cname, vwindow = 21, window = 8):
    ret = pl.col(cname).pct_change()
    vol = ret.clip(lower_bound = -0.3, upper_bound=0.3).ewm_std(half_life = vwindow)
    return pl.rolling_corr(ret,vol, window_size = window)

def expr_volautocorr(cname, vwindow = 24, window = 672):
    ret = pl.col(cname).pct_change()
    vol = ret.clip(lower_bound = -0.3, upper_bound=0.3).ewm_std(half_life = vwindow)
    bf = vol.rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

def expr_volcontraction(hname, lname, window=20):
    range_ratio = (pl.col(hname) - pl.col(lname)) / pl.col(lname)
    return (range_ratio.rolling_std(window) < range_ratio.rolling_mean(window)).cast(int)

def expr_tailr(cname, window=100):
    ret = pl.col(cname).pct_change()
    return (ret.rolling_skew(window) * 0.5 + ret.rolling_kurtosis(window) * 0.5)

def expr_tailc(cname, window=50, z=2):
    ret = pl.col(cname).pct_change()
    return (ret.abs() > z * ret.rolling_std(window)).cast(int).rolling_mean(window)

def expr_retvolr(cname, window=20):
    ret = pl.col(cname).pct_change()
    return ret.rolling_mean(window) / (ret.rolling_std(window) + 1e-9)

def expr_liqrank(window=30):
    dollar_volume = (pl.col('close') * pl.col('volume')).rolling_mean(window)
    return

def expr_retautocorr(cname, window = 672):
    ret = pl.col(cname).pct_change()
    bf = ret.rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

def expr_intrabarvolskew(cname, oname, hname, lname, window=20):
    open_to_close = (pl.col(cname)/pl.col(oname)).log()
    high_to_low = (pl.col(hname)/pl.col(lname)).log()
    return (high_to_low.rolling_std(window) - open_to_close.rolling_std(window))
    
def expr_voltailc(cname, window=30, z_threshold=2):
    ret = pl.col(cname).pct_change()
    vol = ret.rolling_std(window)
    return (ret.abs() > z_threshold * vol.shift(1)).cast(int)

def expr_lsvolr(cname, swindow=5, lwindow=60):
    ret = pl.col(cname).pct_change()
    return ret.rolling_std(swindow) / (ret.rolling_std(lwindow) + 1e-9)
    
# bins
def expr_amtbins_mom(cname, amtname, bins=5, window=10, ibin = 2):
    hamt = pl.col(amtname).rolling_max(window)
    lamt = pl.col(amtname).rolling_min(window)
    vol_bins = ((pl.col(amtname) - lamt) / (hamt - lamt + 1e-9) * bins).floor().cast(int)
    return pl.when(vol_bins == ibin).then(pl.col(cname).pct_change().rolling_mean(window)).otherwise(0)
    

def expr_prbins_flow(cname, hname, lname, amtname, window=20, bins=3, ibin = 2):
    hpr = pl.col(hname).rolling_max(window)
    lpr = pl.col(lname).rolling_min(window)
    pr_bins = ((pl.col(cname) - lpr) / (hpr - lpr + 1e-9) * bins).floor().cast(int)
    sgl = (pl.col(amtname) * pl.col(cname).diff().sign()).rolling_sum(window)
    return pl.when(pr_bins == ibin).then(sgl).otherwise(0)
    
def expr_prbins_mom(cname, hname, lname, window=20, bins=5, ibin = 4):
    hpr = pl.col(hname).rolling_max(window)
    lpr = pl.col(lname).rolling_min(window)
    pr_bins = ((pl.col(cname) - lpr) / (hpr - lpr + 1e-9) * bins).floor().cast(int)
    sret = (pl.col(cname).pct_change()).rolling_mean(window)
    return pl.when(pr_bins == ibin).then(sret).otherwise(0)


# pattern
def expr_pattern_tripletrend(oname, cname):
    cond1 = pl.col(cname) > pl.col(oname)
    cond2 = pl.col(cname).shift(1) > pl.col(oname).shift(1)
    cond3 = pl.col(cname).shift(2) > pl.col(oname).shift(2)
    return (
        (cond1 & cond2 & cond3).cast(int) -
        (~cond1 & ~cond2 & ~cond3).cast(int)
    )

def expr_pattern_wedge(hname, lname, window=20):
    highs = pl.col(hname).rolling_max(window)
    lows = pl.col(lname).rolling_min(window)
    return (
        ((highs - highs.shift(window)) < 
         (lows - lows.shift(window))).cast(int)
    )

def expr_pattern_engulf(oname, cname):
    bull_engulf = (
        (pl.col(cname) > pl.col(oname)) &
        (pl.col(cname) > pl.col(oname).shift(1)) &
        (pl.col(oname) < pl.col(cname).shift(1))
    )
    bear_engulf = (
        (pl.col(cname) < pl.col(oname)) &
        (pl.col(cname) < pl.col(oname).shift(1)) &
        (pl.col(oname) > pl.col(cname).shift(1))
    )
    return (bull_engulf.cast(int) - bear_engulf.cast(int))

def expr_fractal_dim(hname, lname, window=50):
    range_ = pl.col(hname) - pl.col(lname)
    return (range_.rolling_std(window) / range_.rolling_mean(window))


def expr_time_weighted_flow(cname, tname, window=24):
    tm_weight = np.cos(pl.col(tname).dt.hour() / 24 * 2 * np.pi)
    ret = pl.col(cname).pct_change()
    return (ret * tm_weight).rolling_sum(window)

def expr_fractal_memory(cname, window=24):
    ret = pl.col(cname).pct_change()
    rs = ret.rolling_std(4).log().rolling_mean(window) * 0.01854505 + ret.rolling_std(8).log().rolling_mean(window) * 0.02781757  + ret.rolling_std(12).log().rolling_mean(window) * 0.03324165 \
        + ret.rolling_std(16).log().rolling_mean(window) * 0.0370901 + ret.rolling_std(24).log().rolling_mean(window) * 0.04251418 + ret.rolling_std(72).log().rolling_mean(window) * 0.05721078 + ret.rolling_std(168).log().rolling_mean(window) * 0.06854544
    return rs

def expr_spectrum(cname, window=200):
    ret = pl.col(cname).pct_change()
    moments = [
        ret.abs().pow(q).rolling_mean(window)
        for q in [-3, 0, 3]
    ]
    return pl.max_horizontal(moments) - pl.min_horizontal(moments)

def expr_rvarskew(cname, window=20):
    returns = pl.col(cname).pct_change()
    return (
        (returns.rolling_var(window) / 2) - 
        returns.rolling_skew(window).pow(2)/8
    )

def expr_market_temperature(cname, hname, lname, aname, window=50):
    ke = pl.col(aname) * pl.col(cname).diff().pow(2)
    pe = (pl.col(hname) - pl.col(lname)).pow(2)
    return (ke.rolling_sum(window) / pe.rolling_sum(window)).log()

def expr_radiation(cname, aname, window=20):
    gamma = pl.col(aname) * pl.col(cname).diff().abs()
    return gamma.rolling_mean(window) * pl.col(cname).pct_change().rolling_skew(window)


def expr_retd1amtcorr(cname, amtname, nbar_p_hour = 4):
    ret = pl.col(cname).pct_change()
    return pl.rolling_corr(ret.shift(), pl.col(amtname), window_size = nbar_p_hour * 24)

def expr_retamtd1corr(cname, amtname, nbar_p_hour = 4):
    ret = pl.col(cname).pct_change()
    return pl.rolling_corr(ret, pl.col(amtname).shift(), window_size = nbar_p_hour * 24)



def expr_aroon(hname, lname, window=20):
    # Aroon Up = (pos_max + 1) / window * 100
    aroon_up = pl.col(hname).rolling_map(
            lambda s: s.arg_max(),
            window_size = window,
            min_periods=window
    )
    aroon_down = pl.col(lname).rolling_map(
            lambda s: s.arg_min(),
            window_size = window,
            min_periods=window
    )
    return (aroon_up - aroon_down) / window



def expr_bop(oname, cname, hname, lname, window): #Balance Of Power 
    bop = (pl.col(cname) - pl.col(oname)) / (pl.col(hname) - pl.col(lname) + 1e-7)
    return bop.rolling_mean(window)



def expr_k3(cname, hname, lname, window=20, swindow = 4):
    _lowest_low = pl.col(lname).rolling_min(window)
    _highest_high = pl.col(hname).rolling_max(window)
    _numerator = pl.col(cname) - _lowest_low
    _denominator = _highest_high - _lowest_low
    _smoothed_numerator = _numerator.rolling_sum(swindow)
    _smoothed_denominator = _denominator.rolling_sum(swindow)
    return _smoothed_numerator / _smoothed_denominator



def expr_secderiv(cname, window):
    ret = pl.col(cname).pct_change(window)
    return ret - ret.shift(window)



def expr_amtwgtskew(cname, amtname, window):
    mean = pl.col(cname).rolling_mean(window)
    p3 = ((pl.col(cname) - pl.col(cname).rolling_mean(window)) / pl.col(cname).rolling_std(window))**3 * pl.col(amtname)
    wskew = p3.rolling_sum(window) / pl.col(amtname).rolling_sum(window)
    return wskew

def expr_rethighamt(cname, amtname, window = 96):
    #high = pl.col(cname).rolling_quantile(0.8, window_size = window)
    ret = pl.col(cname).pct_change()
    low = pl.col(amtname).rolling_quantile(0.8, window_size = window)
    #aah = (pl.col(amtname) * (pl.col(cname) > high).cast(int)).rolling_sum(window)
    #amttt = pl.col(amtname).rolling_sum(window)
    aal = (ret * (pl.col(amtname) > low).cast(int)).rolling_sum(window)
    return aal

def expr_retlowamt(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    low = pl.col(amtname).rolling_quantile(0.2, window_size = window)
    aal = (ret * (pl.col(amtname) < low).cast(int)).rolling_sum(window)
    return aal

def expr_retaimb(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    low = pl.col(amtname).rolling_quantile(0.2, window_size = window)
    high = pl.col(amtname).rolling_quantile(0.8, window_size = window)
    aal = (ret * (pl.col(amtname) < low).cast(int)).rolling_sum(window)
    aah = (ret * (pl.col(amtname) > high).cast(int)).rolling_sum(window)
    return (aah - aal) / (aah + aal + 1e-7)


def expr_retstdaimb(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    low = pl.col(amtname).rolling_quantile(0.2, window_size = window)
    high = pl.col(amtname).rolling_quantile(0.8, window_size = window)
    aal = (ret * (pl.col(amtname) < low).cast(int)).rolling_std(window)
    aah = (ret * (pl.col(amtname) > high).cast(int)).rolling_std(window)
    return (aah - aal) / (aah + aal + 1e-7)


def expr_retdecamt(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    incamt = pl.col(amtname).pct_change() 
    low = incamt.rolling_quantile(0.2, window_size = window)
    aal = (ret * (incamt < low).cast(int)).rolling_sum(window)
    return aal

def expr_retincamt(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    incamt = pl.col(amtname).pct_change() 
    low = incamt.rolling_quantile(0.8, window_size = window)
    aal = (ret * (incamt > low).cast(int)).rolling_sum(window)
    return aal



def expr_after_highamt_ret(cname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(amtname).rolling_max(window)
    price_top = (
            (pl.col(amtname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="forward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(cname)).rolling_mean(window)
        / pl.col(cname).rolling_mean(window)
    )
    return bf

def expr_after_lowamt_ret(cname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(amtname).rolling_min(window)
    price_top = (
            (pl.col(amtname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="forward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(cname)).rolling_mean(window)
        / pl.col(cname).rolling_mean(window)
    )
    return bf


def expr_before_highamt_ret(cname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(amtname).rolling_max(window)
    price_top = (
            (pl.col(amtname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="backward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(cname)).rolling_mean(window)
        / pl.col(cname).rolling_mean(window)
    )
    return bf

def expr_before_lowamt_ret(cname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(amtname).rolling_min(window)
    price_top = (
            (pl.col(amtname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="backward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(cname)).rolling_mean(window)
        / pl.col(cname).rolling_mean(window)
    )
    return bf

def expr_netamt(amtname, bamtname, window = 8):
    return (2 * pl.col(bamtname) - pl.col(amtname)).rolling_mean(window)


# clock
def expr_usmom(cname, dtname, nday = 24):
    ret = pl.col(cname).pct_change()
    month = pl.col(dtname).dt.month()
    summer = (month.cast(int).is_in([3,4,5,6,7,8,9,10])).cast(int)
    tmint = (pl.col(dtname).dt.hour().cast(int) * 60 + pl.col(dtname).dt.minute().cast(int))
    summer_us = (tmint.is_between(720, 780) | tmint.is_between(1200, 1260)).cast(int)
    winter_us = (tmint.is_between(660, 720) | tmint.is_between(1140, 1200)).cast(int)
    inforet = (summer * summer_us * ret + (1-summer) * winter_us * ret).rolling_mean(96*nday)
    return inforet


# sr dist
def expr_madist(cname, dtname):
    windows = [7 * 96, 14 * 96, 21 * 96]
    not0 = (pl.col(dtname).dt.hour().cast(int) * 60 + pl.col(dtname).dt.minute().cast(int) != 0)
    mas = [pl.col(cname).rolling_mean(window) for window in windows]
    mas = [pl.when(not0).then(None).otherwise(ima).fill_null(strategy="forward", limit=95) for ima in mas]
    sses = [(pl.col(cname) - ima)**2 for ima in mas]
    minsse = pl.min_horizontal(sses)
    minmasks = [(sse == minsse).cast(int) for sse in sses]
    sses = [sse + minmask * 1e9 for sse,minmask in zip(sses, minmasks)]
    secondminsse = pl.min_horizontal(sses)
    secondminmasks = [(sse == secondminsse).cast(int) for sse in sses]
    c1 = pl.sum_horizontal([ma * minmask for ma,minmask in zip(mas,minmasks)])
    c2 = pl.sum_horizontal([ma * minmask for ma,minmask in zip(mas,secondminmasks)])
    difthre = pl.min_horizontal([(c2 - c1).abs()/2, 0.2 * c1])
    dc1 = pl.col(cname) - c1
    dif = pl.min_horizontal([dc1.abs(),difthre])
    return (1 - dif/difthre) * (dc1).sign()



def expr_adosc(cname, hname, lname, amtname, window, multiplier = 3, cum_days = 21) -> pl.Expr:
    """
      ADL = cumulative sum of Money Flow Volume
      ADOSC = EMA(ADL, fast) - EMA(ADL, slow)
    """
    mfm = ((pl.col(cname) - pl.col(lname)) - (pl.col(hname) - pl.col(cname))) / (pl.col(hname) - pl.col(lname)).clip(1e-6)
    mfv = mfm * pl.col(amtname)
    adl = mfv.rolling_sum(cum_days * 96 ) #.cumsum() #? maybe a cutoff using 1mth?
    ema_fast = adl.ewm_mean(span=window * multiplier)
    ema_slow = adl.ewm_mean(span=window)
    return (ema_fast - ema_slow) / pl.col(amtname).rolling_mean(96 * cum_days)




def expr_amt_mismatchv1(cname, amtname, window): 
    ret = pl.col(cname).pct_change()
    return pl.rolling_corr(ret.abs(), pl.col(amtname).shift(), window_size = window)

def expr_amt_mismatchr1(cname, amtname, window): 
    ret = pl.col(cname).pct_change()
    return pl.rolling_corr(ret.abs().shift(), pl.col(amtname), window_size = window)

def expr_amtmax(amtname, window, bootstrap_num = 500):
    def bootstrap(values):
        vol_arr = np.array(values)
        valid_vals = vol_arr[~np.isnan(vol_arr)]  # 过滤 NaN 值
        # 重采样计算逻辑
        samples = np.random.choice(
            valid_vals, 
            size=(bootstrap_num, len(valid_vals)),
            replace=True
        )
        max_samples = np.max(samples, axis=1)
        std_max = np.std(max_samples)
        return std_max / valid_vals.max() if valid_vals.max() != 0 else 0.0


def expr_amthighprice(cname, amtname, window = 96):
    high = pl.col(cname).rolling_quantile(0.8, window_size = window)
    #low = pl.col(cname).rolling_quantile(0.2, window_size = window)
    aah = (pl.col(amtname) * (pl.col(cname) > high).cast(int)).rolling_sum(window)
    amttt = pl.col(amtname).rolling_sum(window)
    #aal = (pl.col(amtname) * (pl.col(cname) < low).cast(int)).rolling_sum(window)
    return aah/amttt


def expr_amtlowprice(cname, amtname, window = 96):
    #high = pl.col(cname).rolling_quantile(0.8, window_size = window)
    low = pl.col(cname).rolling_quantile(0.2, window_size = window)
    #aah = (pl.col(amtname) * (pl.col(cname) > high).cast(int)).rolling_sum(window)
    amttt = pl.col(amtname).rolling_sum(window)
    aal = (pl.col(amtname) * (pl.col(cname) < low).cast(int)).rolling_sum(window)
    return aal/amttt

def expr_after_high_amt(hname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(hname).rolling_max(window)
    price_top = (
            (pl.col(hname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="forward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(amtname)).rolling_mean(window)
        / pl.col(amtname).rolling_mean(window)
    )
    return bf

def expr_after_low_amt(lname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(lname).rolling_min(window)
    price_top = (
            (pl.col(lname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="forward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(amtname)).rolling_mean(window)
        / pl.col(amtname).rolling_mean(window)
    )
    return bf


def expr_before_high_amt(hname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(hname).rolling_max(window)
    price_top = (
            (pl.col(hname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="backward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(amtname)).rolling_mean(window)
        / pl.col(amtname).rolling_mean(window)
    )
    return bf


def expr_before_low_amt(lname: str, amtname: str, window: int = 288, limit: int = 5):
    rmax = pl.col(lname).rolling_min(window)
    price_top = (
            (pl.col(lname) == rmax)
        )
    price_top_after = price_top.cast(int).replace(0, None).fill_null(strategy="backward", limit=limit)
    price_top_after = pl.when(price_top).then(None).otherwise(price_top_after)
    price_top_after = price_top_after.fill_null(strategy="zero")
    bf = (
        (price_top_after * pl.col(amtname)).rolling_mean(window)
        / pl.col(amtname).rolling_mean(window)
    )
    return bf
    
def expr_amthighret(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    low = ret.rolling_quantile(0.2, window_size = window)
    aal = (pl.col(amtname) * (ret < low).cast(int)).rolling_sum(window)
    return aal

def expr_amtlowret(cname, amtname, window = 96):
    ret = pl.col(cname).pct_change()
    low = ret.rolling_quantile(0.8, window_size = window)
    aal = (pl.col(amtname) * (ret > low).cast(int)).rolling_sum(window)
    return aal

def expr_amtstdhighprice(cname, amtname, window = 96):
    high = pl.col(cname).rolling_quantile(0.8, window_size = window)
    #low = pl.col(cname).rolling_quantile(0.2, window_size = window)
    aah = (pl.col(amtname) * (pl.col(cname) > high).cast(int)).rolling_std(window)
    amttt = pl.col(amtname).rolling_mean(window)
    #aal = (pl.col(amtname) * (pl.col(cname) < low).cast(int)).rolling_sum(window)
    return aah/amttt


def expr_amtstdlowprice(cname, amtname, window = 96):
    #high = pl.col(cname).rolling_quantile(0.8, window_size = window)
    low = pl.col(cname).rolling_quantile(0.2, window_size = window)
    #aah = (pl.col(amtname) * (pl.col(cname) > high).cast(int)).rolling_sum(window)
    amttt = pl.col(amtname).rolling_mean(window)
    aal = (pl.col(amtname) * (pl.col(cname) < low).cast(int)).rolling_std(window)
    return aal/amttt



def expr_natr(oname, cname, hname, lname, window):
    prev_close = pl.col(cname).shift(1)
    true_range = pl.max_horizontal(
        pl.col(hname) - pl.col(lname),
        (pl.col(hname) - prev_close).abs(),
        (pl.col(lname) - prev_close).abs()
    )
    atr = true_range.rolling_mean(window)
    natr = atr / pl.col(cname)
    return natr
