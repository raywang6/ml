import pandas as pd
from pandas import DataFrame, Series
import numpy as np
EPS = np.finfo(float).eps
class TA:
    @classmethod
    def SMA(cls, ohlc: DataFrame, period: int = 41, column: str = "close") -> Series:
        """
        Simple moving average - rolling mean in pandas lingo. Also known as 'MA'.
        The simple moving average (SMA) is the most basic of the moving averages used for trading.
        """

        return pd.Series(
            ohlc[column]
            .rolling(center=False, window=period, min_periods=period - 1)
            .mean(),
            name="{0} period SMA".format(period),
        )

    @classmethod
    def EMA(cls, ohlc: DataFrame, period: int = 9, column: str = "close") -> Series:
        """
        Exponential Weighted Moving Average - Like all moving average indicators, they are much better suited for trending markets.
        When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and vice-versa for a down trend.
        EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge their validity.
        """

        return pd.Series(
            ohlc[column]
            .ewm(ignore_na=False, min_periods=period - 1, span=period)
            .mean(),
            name="{0} period EMA".format(period),
        )
    

    @classmethod
    def TRIMA(cls, ohlc: DataFrame, period: int = 18) -> Series:
        """
        The Triangular Moving Average (TRIMA) [also known as TMA] represents an average of prices,
        but places weight on the middle prices of the time period.
        The calculations double-smooth the data using a window width that is one-half the length of the series.
        source: https://www.thebalance.com/triangular-moving-average-tma-description-and-uses-1031203
        """

        SMA = cls.SMA(ohlc, period).rolling(window=period).sum()
        return pd.Series(SMA / period, name="{0} period TRIMA".format(period))

    @classmethod
    def TEMA(cls, ohlc: DataFrame, period: int = 9) -> Series:
        """
        Triple exponential moving average - attempts to remove the inherent lag associated to Moving Averages by placing more weight on recent values.
        The name suggests this is achieved by applying a triple exponential smoothing which is not the case. The name triple comes from the fact that the
        value of an EMA (Exponential Moving Average) is triple.
        To keep it in line with the actual data and to remove the lag the value 'EMA of EMA' is subtracted 3 times from the previously tripled EMA.
        Finally 'EMA of EMA of EMA' is added.
        Because EMA(EMA(EMA)) is used in the calculation, TEMA needs 3 * period - 2 samples to start producing values in contrast to the period samples
        needed by a regular EMA.
        """

        triple_ema = 3 * cls.EMA(ohlc, period)
        ema_ema_ema = (
            cls.EMA(ohlc, period)
            .ewm(ignore_na=False, span=period)
            .mean()
            .ewm(ignore_na=False, span=period)
            .mean()
        )

        TEMA = (
            triple_ema
            - 3
            * cls.EMA(ohlc, period)
            .ewm(span=period)
            .mean()
            + ema_ema_ema
        )
        return pd.Series(TEMA, name="{0} period TEMA".format(period))
   
    @classmethod
    def ER(cls, ohlc: DataFrame, period: int = 10) -> Series:
        """The Kaufman Efficiency indicator is an oscillator indicator that oscillates between +100 and -100, where zero is the center point.
         +100 is upward forex trending market and -100 is downwards trending markets."""

        change = ohlc["close"].diff(period).abs()
        volatility = ohlc["close"].diff().abs().rolling(window=period).sum()

        return pd.Series(change / volatility, name="{0} period ER".format(period))
   
    @classmethod
    def DEMA(cls, ohlc: DataFrame, period: int = 9, column: str = "close") -> Series:
        """
        Double Exponential Moving Average - attempts to remove the inherent lag associated to Moving Averages
         by placing more weight on recent values. The name suggests this is achieved by applying a double exponential
        smoothing which is not the case. The name double comes from the fact that the value of an EMA (Exponential Moving Average) is doubled.
        To keep it in line with the actual data and to remove the lag the value 'EMA of EMA' is subtracted from the previously doubled EMA.
        Because EMA(EMA) is used in the calculation, DEMA needs 2 * period -1 samples to start producing values in contrast to the period
        samples needed by a regular EMA
        """

        DEMA = (
            2 * cls.EMA(ohlc, period)
            - cls.EMA(ohlc, period)
            .ewm(span=period)
            .mean()
        )

        return pd.Series(DEMA, name="{0} period DEMA".format(period))


    @classmethod
    def MACD(
        cls,
        ohlc: DataFrame,
        period_fast: int = 12,
        period_slow: int = 26,
        signal: int = 9,
    ) -> Series:
        """
        MACD, MACD Signal and MACD difference.
        The MACD Line oscillates above and below the zero line, which is also known as the centerline.
        These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course, depends on the direction of the moving average cross.
        Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as the shorter EMA diverges further from the longer EMA.
        This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA is below the 26-day EMA.
        Negative values increase as the shorter EMA diverges further below the longer EMA. This means downside momentum is increasing.

        Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.
        As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.
        A bullish crossover occurs when the MACD turns up and crosses above the signal line.
        A bearish crossover occurs when the MACD turns down and crosses below the signal line.
        """

        EMA_fast = pd.Series(
            ohlc["close"]
            .ewm(ignore_na=False, min_periods=period_slow - 1, span=period_fast)
            .mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            ohlc["close"]
            .ewm(ignore_na=False, min_periods=period_slow - 1, span=period_slow)
            .mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal).mean(), name="SIGNAL"
        )

        return pd.concat([MACD, MACD_signal], axis=1)

    @classmethod
    def ROC(cls, ohlc: DataFrame, period: int = 12) -> Series:
        """The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum,
        is a pure momentum oscillator that measures the percent change in price from one period to the next.
        The ROC calculation compares the current price with the price “n” periods ago."""

        return pd.Series(
            (ohlc["close"].diff(period) / ohlc["close"].shift(period)) , name="ROC"
        )

    @classmethod
    def RSI(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
        RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
        Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
        RSI can also be used to identify the general trend."""

        ## get the price diff
        delta = ohlc["close"].diff()

        ## positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # EMAs of ups and downs
        _gain = up.ewm(span=period, min_periods=period - 1).mean()
        _loss = down.abs().ewm(span=period, min_periods=period - 1).mean()

        RS = (_gain + EPS) / (_loss + EPS)
        return pd.Series(100 - (100 / (1 + RS)), name="RSI")


    @classmethod
    def HIGH_LOW(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """PEAK_To_Trough over a period"""
        highest_high = pd.Series(ohlc["close"].rolling(center=False, window=period).max(), name='high')
        lowest_low = pd.Series(ohlc["close"].rolling(center=False, window=period).min(), name= 'low')
    
        return pd.concat([highest_high, lowest_low], axis=1) 

    @classmethod
    def TR(cls, ohlc: DataFrame) -> Series:
        """True Range is the maximum of three price ranges.
        Most recent period's high minus the most recent period's low.
        Absolute value of the most recent period's high minus the previous close.
        Absolute value of the most recent period's low minus the previous close."""

        TR1 = pd.Series(ohlc["high"] - ohlc["low"]).abs()  # True Range = High less Low

        TR2 = pd.Series(
            ohlc["high"] - ohlc["close"].shift()
        ).abs()  # True Range = High less Previous Close

        TR3 = pd.Series(
            ohlc["close"].shift() - ohlc["low"]
        ).abs()  # True Range = Previous Close less Low

        _TR = pd.concat([TR1, TR2, TR3], axis=1)

        _TR["TR"] = _TR.max(axis=1)

        return pd.Series(_TR["TR"], name="TR")

    @classmethod
    def ATR(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """Average True Range is moving average of True Range."""

        TR = cls.TR(ohlc)
        return pd.Series(
            TR.rolling(center=False, window=period, min_periods=period - 1).mean(),
            name="{0} period ATR".format(period),
        )

    @classmethod
    def PEAKMINUSTROUGH(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """PEAK_To_Trough over a period"""
        highest_high = ohlc["high"].rolling(center=False, window=period).max()
        lowest_low = ohlc["low"].rolling(center=False, window=period).min()
        PEAKMINUSTROUGH = pd.Series(
            (highest_high - lowest_low),
            name="{0} period PEAKMINUSTROUGH".format(period),
        )
        return PEAKMINUSTROUGH

    @classmethod
    def PEAKMINUSTROUGHCLOSE(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """PEAK_To_Trough over a period"""
        highest_high = ohlc["close"].rolling(center=False, window=period).max()
        lowest_low = ohlc["close"].rolling(center=False, window=period).min()
        PEAKMINUSTROUGHCLOSE = pd.Series(
            (highest_high - lowest_low),
            name="{0} period PEAKMINUSTROUGHCLOSE".format(period),
        )
        return PEAKMINUSTROUGHCLOSE

    @classmethod
    def PRICESTD(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """Price std over a period"""
        return pd.Series( ohlc["close"].rolling(window=period).std() ,
                        name="{0} Price STD".format(period),
                        )

    
    @classmethod
    def BBANDS(
        cls, ohlc: DataFrame, period: int = 20, MA: Series = None, column: str = "close"
    ) -> Series:
        """
         Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
         Volatility is based on the standard deviation, which changes as volatility increases and decreases.
         The bands automatically widen when volatility increases and narrow when volatility decreases.

         This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
         Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
         """

        std = ohlc["close"].rolling(window=period).std()

        if not isinstance(MA, pd.core.series.Series):
            middle_band = pd.Series(cls.SMA(ohlc, period), name="BB_MIDDLE")
        else:
            middle_band = pd.Series(MA, name="BB_MIDDLE")

        upper_bb = pd.Series(middle_band + (2 * std), name="BB_UPPER")
        lower_bb = pd.Series(middle_band - (2 * std), name="BB_LOWER")

        return pd.concat([upper_bb, middle_band, lower_bb], axis=1)
    
    @classmethod
    def BBANDSSTDDIST(
        cls, ohlc: DataFrame, period: int = 20, MA: Series = None, column: str = "close"
    ) -> Series:
        """
         Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
         Volatility is based on the standard deviation, which changes as volatility increases and decreases.
         The bands automatically widen when volatility increases and narrow when volatility decreases.

         This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
         Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
         """

        std = ohlc["close"].rolling(window=period).std() + EPS

        if not isinstance(MA, pd.core.series.Series):
            middle_band = pd.Series(cls.SMA(ohlc, period), name="BB_MIDDLE")
        else:
            middle_band = pd.Series(MA, name="BB_MIDDLE")
        diff = (ohlc["close"] - middle_band)
        diff[diff.abs()<1e-6] = 0. # tho this thresh can be even high, just being safe from round issues

        bb_dist =pd.Series( (diff).divide(std), name="BB_DIST")
        return bb_dist

    @classmethod
    def TURNOVR(cls, ohlcv: DataFrame, period: int = 30) -> Series:
        return pd.Series(
            (ohlcv["close"] * ohlcv["volume"]).rolling(period).sum(), name="TURNOVR"
        )

    @classmethod
    def KC(
        cls, ohlc: DataFrame, period: int = 20, atr_period: int = 10, MA: Series = None
    ) -> Series:
        """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average.
        This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands.
        Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance.
        The channels are typically set two Average True Range values above and below the 20-day EMA.
        The exponential moving average dictates direction and the Average True Range sets channel width.
        Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel direction.
        Channels can also be used to identify overbought and oversold levels when the trend is flat."""

        if not isinstance(MA, pd.core.series.Series):
            middle = pd.Series(cls.EMA(ohlc, period), name="KC_MIDDLE")
        else:
            middle = pd.Series(MA, name="KC_MIDDLE")

        up = pd.Series(middle + (2 * cls.ATR(ohlc, atr_period)), name="KC_UPPER")
        down = pd.Series(middle - (2 * cls.ATR(ohlc, atr_period)), name="KC_LOWER")

        return pd.concat([up, down], axis=1)

    @classmethod
    def DMI(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """The directional movement indicator (also known as the directional movement index - DMI) is a valuable tool
         for assessing price direction and strength. This indicator was created in 1978 by J. Welles Wilder, who also created the popular
         relative strength index. DMI tells you when to be long or short.
         It is especially useful for trend trading strategies because it differentiates between strong and weak trends,
         allowing the trader to enter only the strongest trends.
        """

        ohlc["up_move"] = ohlc["high"].diff()
        ohlc["down_move"] = -ohlc["low"].diff()

        DMp = []
        DMm = []

        for row in ohlc.itertuples():
            if row.up_move > row.down_move and row.up_move > 0:
                DMp.append(row.up_move)
            else:
                DMp.append(0)

            if row.down_move > row.up_move and row.down_move > 0:
                DMm.append(row.down_move)
            else:
                DMm.append(0)

        ohlc["DMp"] = DMp
        ohlc["DMm"] = DMm

        diplus = pd.Series(
            100
            * (ohlc["DMp"] / cls.ATR(ohlc, period))
            .ewm(span=period, min_periods=period - 1)
            .mean(),
            name="DI+",
        )
        diminus = pd.Series(
            100
            * (ohlc["DMm"] / cls.ATR(ohlc, period))
            .ewm(span=period, min_periods=period - 1)
            .mean(),
            name="DI-",
        )

        return pd.concat([diplus, diminus], axis=1)

    @classmethod
    def ADX(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """The A.D.X. is 100 * smoothed moving average of absolute value (DMI +/-) divided by (DMI+ + DMI-). ADX does not indicate trend direction or momentum,
        only trend strength. Generally, A.D.X. readings below 20 indicate trend weakness,
        and readings above 40 indicate trend strength. An extremely strong trend is indicated by readings above 50"""

        dmi = cls.DMI(ohlc, period)
        return pd.Series(
            100
            * (abs(dmi["DI+"] - dmi["DI-"]) / (dmi["DI+"] + dmi["DI-"] + 1e-6))
            .ewm(alpha=1 / period)
            .mean(),
            name="{0} period ADX.".format(period),
        )

    @classmethod
    def REGIMESMA(cls, ohlc: DataFrame, period: int = 14) -> Series:
        return pd.Series(cls.SMA(ohlc.shift(1), period), name="REGIMESMA")

    @classmethod
    def REGIMESTOCH(cls, ohlc: DataFrame, period: int = 14) -> Series:
        max_close = pd.Series(ohlc["close"].shift(1).rolling(center=False, window=period).max(), name = "MAXCLOSE")
        min_close = pd.Series(ohlc["close"].shift(1).rolling(center=False, window=period).min(), name = "MINCLOSE")
        return pd.concat([max_close, min_close], axis=1)

       
    @classmethod
    def STOCH(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """Stochastic oscillator %K
         The stochastic oscillator is a momentum indicator comparing the closing price of a security
         to the range of its prices over a certain period of time.
         The sensitivity of the oscillator to market movements is reducible by adjusting that time
         period or by taking a moving average of the result.
        """

        highest_high = ohlc["high"].rolling(center=False, window=period).max()
        lowest_low = ohlc["low"].rolling(center=False, window=period).min()

        STOCH = pd.Series(
            (ohlc["close"] - lowest_low + EPS) / (highest_high - lowest_low + 2* EPS),
            name="{0} period STOCH %K".format(period),
        )

        return 100 * STOCH

    @classmethod
    def STOCHCLOSE(cls, ohlc: DataFrame, period: int = 14) -> Series:
        """Stochastic oscillator %K
         The stochastic oscillator is a momentum indicator comparing the closing price of a security
         to the range of its prices over a certain period of time.
         The sensitivity of the oscillator to market movements is reducible by adjusting that time
         period or by taking a moving average of the result.
        """

        highest_high = ohlc["close"].rolling(center=False, window=period).max()
        lowest_low = ohlc["close"].rolling(center=False, window=period).min()

        STOCH = pd.Series(
            (ohlc["close"] - lowest_low + EPS) / (highest_high - lowest_low + 2* EPS),
            name="{0} period STOCH %K".format(period),
        )

        return 100 * STOCH
    
    @classmethod
    def TP(cls, ohlc: DataFrame) -> Series:
        """Typical Price refers to the arithmetic average of the high, low, and closing prices for a given period."""

        return pd.Series((ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3, name="TP")

    @classmethod
    def CCI(cls, ohlc: DataFrame, period: int = 20, constant: float = 0.015) -> Series:
        """Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new trend or warn of extreme conditions.
        CCI measures the current price level relative to an average price level over a given period of time.
        The CCI typically oscillates above and below a zero line. Normal oscillations will occur within the range of +100 and −100.
        Readings above +100 imply an overbought condition, while readings below −100 imply an oversold condition.
        As with other overbought/oversold indicators, this means that there is a large probability that the price will correct to more representative levels.

        source: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

        :param pd.DataFrame ohlc: 'open, high, low, close' pandas DataFrame
        :period: int - number of periods to take into consideration
        :factor float: the constant at .015 to ensure that approximately 70 to 80 percent of CCI values would fall between -100 and +100.
        :return pd.Series: result is pandas.Series
        """

        tp = cls.TP(ohlc)
        tp_rolling = tp.rolling(window=period, min_periods=0) 
        return pd.Series(
            (tp - tp_rolling.mean()) / (constant * tp_rolling.std()+EPS
            ), name="{0} period CCI".format(period),
        )

if __name__ == "__main__":
    print([k for k in TA.__dict__.keys() if k[0] not in "_"])
