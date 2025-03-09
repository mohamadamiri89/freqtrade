from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SampleStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.05,    # 5% profit target
        "60": 0.03,   # 3% profit target after 60 minutes
        "120": 0.02,  # 2% profit target after 120 minutes
        "240": 0.01   # 1% profit target after 240 minutes
    }

    # Stoploss:
    stoploss = -0.02  # 2% stoploss

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5% trailing stop
    trailing_stop_positive_offset = 0.01  # 1% offset
    trailing_only_offset_is_reached = True

    # Run "populate_indicators" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        # EMAs
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_std'] = dataframe['volume'].rolling(window=20).std()
        dataframe['volume_z'] = (dataframe['volume'] - dataframe['volume_mean']) / dataframe['volume_std']

        # Trend strength
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # Price action
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_ratio'] = dataframe['atr'] / dataframe['close']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Strong trend
                (dataframe['adx'] > 25) &
                (dataframe['plus_di'] > dataframe['minus_di']) &
                
                # Price conditions
                (
                    (dataframe['rsi'] < 40) |  # RSI oversold
                    (dataframe['close'] < dataframe['bb_lowerband'])  # Price below lower BB
                ) &
                
                # Volume confirmation
                (dataframe['volume_z'] > 1.0) &  # Above average volume
                
                # Trend alignment
                (dataframe['ema_9'] > dataframe['ema_21']) &  # Short-term trend
                (dataframe['ema_21'] > dataframe['ema_50']) &  # Medium-term trend
                
                # Volatility check
                (dataframe['atr_ratio'] < 0.02)  # Not too volatile
            ),
            'enter_long'] = 1

        # Log entry signals
        if len(dataframe[dataframe['enter_long'] == 1]) > 0:
            logger.info(f"Entry signal for {metadata['pair']} at {dataframe.index[-1]}")
            logger.info(f"RSI: {dataframe['rsi'].iloc[-1]:.2f}")
            logger.info(f"ADX: {dataframe['adx'].iloc[-1]:.2f}")
            logger.info(f"Volume Z: {dataframe['volume_z'].iloc[-1]:.2f}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Trend reversal
                (dataframe['adx'] > 25) &
                (dataframe['minus_di'] > dataframe['plus_di']) &
                
                # Price conditions
                (
                    (dataframe['rsi'] > 60) |  # RSI overbought
                    (dataframe['close'] > dataframe['bb_upperband'])  # Price above upper BB
                ) &
                
                # Volume confirmation
                (dataframe['volume_z'] > 1.0) &  # Above average volume
                
                # Trend alignment
                (dataframe['ema_9'] < dataframe['ema_21'])  # Short-term trend reversal
            ),
            'exit_long'] = 1

        # Log exit signals
        if len(dataframe[dataframe['exit_long'] == 1]) > 0:
            logger.info(f"Exit signal for {metadata['pair']} at {dataframe.index[-1]}")
            logger.info(f"RSI: {dataframe['rsi'].iloc[-1]:.2f}")
            logger.info(f"ADX: {dataframe['adx'].iloc[-1]:.2f}")
            logger.info(f"Volume Z: {dataframe['volume_z'].iloc[-1]:.2f}")

        return dataframe 