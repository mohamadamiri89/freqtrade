from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce

logger = logging.getLogger(__name__)

class HiddenDivergenceMultiTimeframeStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Can do both long and short
    can_short = True

    # Minimal ROI - More conservative
    minimal_roi = {
        "0": 0.025,    # 2.5% immediate profit target
        "30": 0.02,    # 2% after 30 minutes
        "60": 0.015,   # 1.5% after 60 minutes
        "120": 0.01    # 1% after 120 minutes
    }

    # Stoploss - Tighter for better risk management
    stoploss = -0.015  # 1.5% stoploss

    # Trailing stop settings
    trailing_stop = True
    trailing_stop_positive = 0.008    # 0.8% trailing stop
    trailing_stop_positive_offset = 0.01  # 1% offset
    trailing_only_offset_is_reached = True

    # Stoploss on exchange
    stoploss_on_exchange = True
    stoploss_on_exchange_interval = 60
    stoploss_on_exchange_limit_ratio = 0.99

    # General Settings
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 100  # Need more candles for proper market analysis

    # Hyperopt parameters
    buy_rsi = 30
    sell_rsi = 70
    short_rsi = 70
    exit_short_rsi = 30
    
    # Custom parameters
    max_epa = 1  # Max entries per side
    market_regime_lookback = 100  # Lookback for market regime detection

    def detect_market_regime(self, dataframe: DataFrame) -> DataFrame:
        """Detect if market is trending or ranging"""
        # Calculate directional movement
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe, timeperiod=14)
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe, timeperiod=14)
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Detect regime
        dataframe['is_trending'] = (dataframe['adx'] > 25)
        dataframe['is_ranging'] = (dataframe['adx'] <= 25)
        
        return dataframe

    def detect_market_structure(self, dataframe: DataFrame) -> DataFrame:
        """Detect market structure (higher highs/lows or lower highs/lows)"""
        dataframe['higher_high'] = (dataframe['high'] > dataframe['high'].shift(1)) & (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        dataframe['lower_low'] = (dataframe['low'] < dataframe['low'].shift(1)) & (dataframe['low'].shift(1) < dataframe['low'].shift(2))
        dataframe['higher_low'] = (dataframe['low'] > dataframe['low'].shift(1))
        dataframe['lower_high'] = (dataframe['high'] < dataframe['high'].shift(1))
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if len(dataframe) == 0:
            logger.warning(f"Empty dataframe for {metadata['pair']}")
            return dataframe

        # Volume Indicators
        dataframe['volume_mean'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        dataframe['volume_trend'] = (dataframe['volume'] > dataframe['volume_mean']).astype(int)

        # Trend Indicators
        for window in [8, 13, 21, 34]:
            dataframe[f'ema_{window}'] = ta.EMA(dataframe, timeperiod=window)
        
        # RSI with multiple timeframes
        for window in [6, 14, 24]:
            dataframe[f'rsi_{window}'] = ta.RSI(dataframe, timeperiod=window)
        
        # MACD for trend direction
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger Bands for volatility
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((bollinger['upper'] - bollinger['lower']) / bollinger['mid'])

        # Market regime and structure
        dataframe = self.detect_market_regime(dataframe)
        dataframe = self.detect_market_structure(dataframe)

        # Volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_ratio'] = dataframe['atr'] / dataframe['close']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Volume conditions
        volume_filter = (dataframe['volume_ratio'] > 1.2)

        # LONG CONDITIONS
        long_trend_conditions = [
            dataframe['is_trending'],
            dataframe['ema_8'] > dataframe['ema_13'],
            dataframe['ema_13'] > dataframe['ema_21'],
            dataframe['close'] > dataframe['ema_8'],
            dataframe['macdhist'] > 0,
            dataframe['rsi_14'] < 70,  # Not overbought
            volume_filter
        ]

        long_range_conditions = [
            dataframe['is_ranging'],
            dataframe['close'] < dataframe['bb_lowerband'],
            dataframe['rsi_14'] < self.buy_rsi,
            dataframe['volume_trend'],
            dataframe['higher_low']
        ]

        conditions.append(reduce(lambda x, y: x & y, long_trend_conditions))
        conditions.append(reduce(lambda x, y: x & y, long_range_conditions))
        
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1

        # SHORT CONDITIONS
        short_trend_conditions = [
            dataframe['is_trending'],
            dataframe['ema_8'] < dataframe['ema_13'],
            dataframe['ema_13'] < dataframe['ema_21'],
            dataframe['close'] < dataframe['ema_8'],
            dataframe['macdhist'] < 0,
            dataframe['rsi_14'] > 30,  # Not oversold
            volume_filter
        ]

        short_range_conditions = [
            dataframe['is_ranging'],
            dataframe['close'] > dataframe['bb_upperband'],
            dataframe['rsi_14'] > self.short_rsi,
            dataframe['volume_trend'],
            dataframe['lower_high']
        ]

        conditions = []
        conditions.append(reduce(lambda x, y: x & y, short_trend_conditions))
        conditions.append(reduce(lambda x, y: x & y, short_range_conditions))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG EXIT
        long_exit_conditions = (
            (
                (dataframe['is_trending']) &
                (
                    (dataframe['ema_8'] < dataframe['ema_13']) |
                    (dataframe['close'] < dataframe['ema_8']) |
                    (dataframe['rsi_14'] > self.sell_rsi)
                )
            ) |
            (
                (dataframe['is_ranging']) &
                (
                    (dataframe['close'] > dataframe['bb_upperband']) |
                    (dataframe['rsi_14'] > self.sell_rsi) |
                    (dataframe['lower_low'])
                )
            )
        )
        
        dataframe.loc[long_exit_conditions, 'exit_long'] = 1

        # SHORT EXIT
        short_exit_conditions = (
            (
                (dataframe['is_trending']) &
                (
                    (dataframe['ema_8'] > dataframe['ema_13']) |
                    (dataframe['close'] > dataframe['ema_8']) |
                    (dataframe['rsi_14'] < self.exit_short_rsi)
                )
            ) |
            (
                (dataframe['is_ranging']) &
                (
                    (dataframe['close'] < dataframe['bb_lowerband']) |
                    (dataframe['rsi_14'] < self.exit_short_rsi) |
                    (dataframe['higher_high'])
                )
            )
        )
        
        dataframe.loc[short_exit_conditions, 'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Custom stoploss logic"""
        # Time-based trailing stop
        if current_profit > 0.02:  # If profit > 2%
            return 0.01  # 1% trailing stoploss
        elif current_profit > 0.01:  # If profit > 1%
            return 0.005  # 0.5% trailing stoploss
        
        return self.stoploss  # Otherwise, return the default stoploss