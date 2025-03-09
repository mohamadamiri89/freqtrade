from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

class AITradingStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.05,    # 5% profit target immediately
        "30": 0.04,   # 4% profit target after 30 minutes
        "60": 0.03,   # 3% profit target after 60 minutes
        "120": 0.02   # 2% profit target after 120 minutes
    }

    # Stoploss:
    stoploss = -0.02  # 2% stoploss

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5%
    trailing_stop_positive_offset = 0.01  # 1%
    trailing_only_offset_is_reached = True

    # Run "populate_indicators" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'emergency_exit': 'market',
        'force_exit': 'market',
        'force_entry': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    # Optional order time in force mapping.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC',
        'emergency_exit': 'GTC',
        'force_exit': 'GTC',
        'force_entry': 'GTC',
        'stoploss': 'GTC'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        
        # EMAs
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        
        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_std'] = dataframe['volume'].rolling(window=20).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume_mean']) / dataframe['volume_std']
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_ratio'] = dataframe['atr'] / dataframe['close']
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe, timeperiod=14)
        
        # 50 EMA for longer trend
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Technical conditions
                (dataframe['rsi'] < 40) &  # RSI oversold
                (dataframe['close'] < dataframe['bb_lowerband']) &  # Price below lower BB
                (dataframe['ema_9'] > dataframe['ema_21']) &  # EMA crossover
                (dataframe['volume'] > dataframe['volume_mean']) &  # Above average volume
                (dataframe['adx'] > 25) &  # Strong trend
                (dataframe['di_plus'] > dataframe['di_minus']) &  # Bullish trend
                (dataframe['close'] > dataframe['ema_50']) &  # Price above 50 EMA
                (dataframe['atr_ratio'] < 0.02)  # Not too volatile
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Technical conditions
                (dataframe['rsi'] > 60) |  # RSI overbought
                (dataframe['close'] > dataframe['bb_upperband']) |  # Price above upper BB
                (dataframe['ema_9'] < dataframe['ema_21']) |  # EMA crossover
                (dataframe['volume'] < dataframe['volume_mean']) |  # Below average volume
                (dataframe['di_plus'] < dataframe['di_minus']) |  # Bearish trend
                (dataframe['close'] < dataframe['ema_50'])  # Price below 50 EMA
            ),
            'exit_long'] = 1

        return dataframe 