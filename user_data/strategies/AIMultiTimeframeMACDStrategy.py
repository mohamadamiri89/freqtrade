from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
import json
import re
import time
from functools import reduce

logger = logging.getLogger(__name__)

class AIMultiTimeframeMACDStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Enable both long and short positions
    can_short = True

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.05,     # 5% profit target (adjusted for leverage)
        "30": 0.03,    # 3% profit target after 30 minutes
        "60": 0.02,    # 2% profit target after 60 minutes
        "120": 0.01    # 1% profit target after 120 minutes
    }

    # Stoploss:
    stoploss = -0.02  # 2% stoploss (adjusted for leverage)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1% trailing stop
    trailing_stop_positive_offset = 0.02  # 2% offset
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

    # Settings for how often to check Ollama
    # Only check Ollama every X candles to avoid overloading the API
    ollama_check_interval = 5
    candle_counter = 0
    last_ai_prediction = {}

    def find_peaks(self, data: np.ndarray, lookback: int = 5) -> List[int]:
        """Find peaks in the data using a simple peak detection algorithm."""
        peaks = []
        for i in range(lookback, len(data) - lookback):
            if all(data[i] > data[i-j] for j in range(1, lookback+1)) and \
               all(data[i] > data[i+j] for j in range(1, lookback+1)):
                peaks.append(i)
        return peaks

    def find_hidden_divergence(self, price_data: np.ndarray, macd_data: np.ndarray, 
                             lookback: int = 14) -> Tuple[bool, str]:
        """
        Detect hidden divergence between price and MACD.
        Returns (has_divergence, divergence_type)
        """
        # Get recent peaks
        price_peaks = self.find_peaks(price_data[-lookback:])
        macd_peaks = self.find_peaks(macd_data[-lookback:])

        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return False, ""

        # Get the last two peaks
        price_peak1, price_peak2 = price_peaks[-2:]
        macd_peak1, macd_peak2 = macd_peaks[-2:]

        # Check for hidden bearish divergence
        if (price_data[-lookback:][price_peak2] > price_data[-lookback:][price_peak1] and
            macd_data[-lookback:][macd_peak2] < macd_data[-lookback:][macd_peak1]):
            return True, "bearish"

        # Check for hidden bullish divergence
        if (price_data[-lookback:][price_peak2] < price_data[-lookback:][price_peak1] and
            macd_data[-lookback:][macd_peak2] > macd_data[-lookback:][macd_peak1]):
            return True, "bullish"

        return False, ""

    def get_ai_prediction(self, pair: str, dataframe: DataFrame) -> Optional[Dict]:
        """Get AI prediction for a trading pair."""
        # Temporarily disabled AI predictions
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if len(dataframe) == 0:
            logger.warning(f"Empty dataframe for {metadata['pair']}")
            return dataframe

        # Add RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add EMA for trend confirmation
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # Add Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # 15-minute MACD
        macd_15m = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd_15m'] = macd_15m['macd']
        dataframe['macdsignal_15m'] = macd_15m['macdsignal']
        dataframe['macdhist_15m'] = macd_15m['macdhist']

        try:
            # Get 60-minute data
            informative = self.dp.get_pair_dataframe(metadata['pair'], '1h')
            
            if informative is not None and len(informative) > 0:
                # 60-minute MACD
                macd_60m = ta.MACD(informative, fastperiod=12, slowperiod=26, signalperiod=9)
                dataframe['macd_60m'] = macd_60m['macd'].iloc[-1]
                dataframe['macdsignal_60m'] = macd_60m['macdsignal'].iloc[-1]
                dataframe['macdhist_60m'] = macd_60m['macdhist'].iloc[-1]
                
                # 60-minute RSI
                dataframe['rsi_60m'] = ta.RSI(informative, timeperiod=14).iloc[-1]
                
                # 60-minute EMA
                dataframe['ema_20_60m'] = ta.EMA(informative, timeperiod=20).iloc[-1]
                dataframe['ema_50_60m'] = ta.EMA(informative, timeperiod=50).iloc[-1]
            else:
                logger.warning(f"Empty 1h dataframe for {metadata['pair']}")
                dataframe['macd_60m'] = 0
                dataframe['macdsignal_60m'] = 0
                dataframe['macdhist_60m'] = 0
                dataframe['rsi_60m'] = 50
                dataframe['ema_20_60m'] = dataframe['close'].iloc[-1]
                dataframe['ema_50_60m'] = dataframe['close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error processing 1h timeframe for {metadata['pair']}: {str(e)}")
            dataframe['macd_60m'] = 0
            dataframe['macdsignal_60m'] = 0
            dataframe['macdhist_60m'] = 0
            dataframe['rsi_60m'] = 50
            dataframe['ema_20_60m'] = dataframe['close'].iloc[-1]
            dataframe['ema_50_60m'] = dataframe['close'].iloc[-1]

        # Calculate MACD changes
        dataframe['macdhist_15m_change'] = dataframe['macdhist_15m'] - dataframe['macdhist_15m'].shift(1)
        dataframe['macdhist_60m_change'] = dataframe['macdhist_60m'] - dataframe['macdhist_60m'].shift(1)

        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_std'] = dataframe['volume'].rolling(window=20).std()
        dataframe['volume_z'] = (dataframe['volume'] - dataframe['volume_mean']) / dataframe['volume_std']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Volume requirement
        volume_condition = (
            (dataframe['volume'] > 0) &  # Basic volume check
            (dataframe['volume_z'] > 0)   # Volume above average
        )
        
        # Trend conditions
        trend_long = (
            (dataframe['ema_20'] > dataframe['ema_50']) &  # Short-term trend up
            (dataframe['ema_50'] > dataframe['ema_200']) &  # Long-term trend up
            (dataframe['close'] > dataframe['ema_20'])      # Price above short-term trend
        )
        
        trend_short = (
            (dataframe['ema_20'] < dataframe['ema_50']) &  # Short-term trend down
            (dataframe['ema_50'] < dataframe['ema_200']) &  # Long-term trend down
            (dataframe['close'] < dataframe['ema_20'])      # Price below short-term trend
        )
        
        # RSI conditions
        rsi_long = (
            (dataframe['rsi'] < 40) &      # Not overbought
            (dataframe['rsi'] > 30) &      # Not extremely oversold
            (dataframe['rsi_60m'] < 45)    # Higher timeframe not overbought
        )
        
        rsi_short = (
            (dataframe['rsi'] > 60) &      # Not oversold
            (dataframe['rsi'] < 70) &      # Not extremely overbought
            (dataframe['rsi_60m'] > 55)    # Higher timeframe not oversold
        )
        
        # MACD conditions
        macd_long = (
            (dataframe['macdhist_15m'] > -0.001) &  # MACD histogram positive
            (dataframe['macdhist_15m_change'] > 0) &  # MACD increasing
            (dataframe['macdhist_60m'] > -0.001)     # Higher timeframe MACD positive
        )
        
        macd_short = (
            (dataframe['macdhist_15m'] < 0.001) &   # MACD histogram negative
            (dataframe['macdhist_15m_change'] < 0) &  # MACD decreasing
            (dataframe['macdhist_60m'] < 0.001)      # Higher timeframe MACD negative
        )
        
        # Price conditions
        price_long = (
            (dataframe['close'] > dataframe['bb_middleband']) &  # Price above middle BB
            (dataframe['close'] < dataframe['bb_upperband'])     # Price not too high
        )
        
        price_short = (
            (dataframe['close'] < dataframe['bb_middleband']) &  # Price below middle BB
            (dataframe['close'] > dataframe['bb_lowerband'])     # Price not too low
        )
        
        # Combine conditions for longs
        long_condition = (
            volume_condition &
            trend_long &
            rsi_long &
            macd_long &
            price_long
        )
        
        # Combine conditions for shorts
        short_condition = (
            volume_condition &
            trend_short &
            rsi_short &
            macd_short &
            price_short
        )
        
        # Apply signals and log position direction
        if long_condition.any():
            dataframe.loc[long_condition, 'enter_long'] = 1
            
            # Log long entry signals
            if long_condition.iloc[-1]:
                logger.info(f"LONG Entry Signal for {metadata['pair']}")
                logger.info(f"RSI: {dataframe['rsi'].iloc[-1]:.2f}")
                logger.info(f"MACD: {dataframe['macdhist_15m'].iloc[-1]:.6f}")
                logger.info(f"MACD Change: {dataframe['macdhist_15m_change'].iloc[-1]:.6f}")
                
        if short_condition.any():
            dataframe.loc[short_condition, 'enter_short'] = 1
            
            # Log short entry signals
            if short_condition.iloc[-1]:
                logger.info(f"SHORT Entry Signal for {metadata['pair']}")
                logger.info(f"RSI: {dataframe['rsi'].iloc[-1]:.2f}")
                logger.info(f"MACD: {dataframe['macdhist_15m'].iloc[-1]:.6f}")
                logger.info(f"MACD Change: {dataframe['macdhist_15m_change'].iloc[-1]:.6f}")
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Volume requirement
        volume_condition = (
            (dataframe['volume'] > 0) &  # Basic volume check
            (dataframe['volume_z'] > 0)   # Volume above average
        )
        
        # Exit long conditions
        exit_long = (
            (dataframe['rsi'] > 70) |  # Overbought
            (dataframe['macdhist_15m'] < -0.002) |  # Strong negative MACD
            (dataframe['close'] < dataframe['ema_20']) |  # Price below short-term trend
            (dataframe['close'] < dataframe['bb_lowerband'])  # Price below lower BB
        )
        
        # Exit short conditions
        exit_short = (
            (dataframe['rsi'] < 30) |  # Oversold
            (dataframe['macdhist_15m'] > 0.002) |  # Strong positive MACD
            (dataframe['close'] > dataframe['ema_20']) |  # Price above short-term trend
            (dataframe['close'] > dataframe['bb_upperband'])  # Price above upper BB
        )
        
        # Apply signals and log position direction
        if exit_long.any():
            dataframe.loc[exit_long, 'exit_long'] = 1
            
            # Log long exit signals
            if exit_long.iloc[-1]:
                logger.info(f"LONG Exit Signal for {metadata['pair']}")
                logger.info(f"RSI: {dataframe['rsi'].iloc[-1]:.2f}")
                logger.info(f"MACD: {dataframe['macdhist_15m'].iloc[-1]:.6f}")
                logger.info(f"MACD Change: {dataframe['macdhist_15m_change'].iloc[-1]:.6f}")
                
        if exit_short.any():
            dataframe.loc[exit_short, 'exit_short'] = 1
            
            # Log short exit signals
            if exit_short.iloc[-1]:
                logger.info(f"SHORT Exit Signal for {metadata['pair']}")
                logger.info(f"RSI: {dataframe['rsi'].iloc[-1]:.2f}")
                logger.info(f"MACD: {dataframe['macdhist_15m'].iloc[-1]:.6f}")
                logger.info(f"MACD Change: {dataframe['macdhist_15m_change'].iloc[-1]:.6f}")
            
        return dataframe