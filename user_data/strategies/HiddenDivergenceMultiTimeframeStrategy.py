from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class HiddenDivergenceMultiTimeframeStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Enable both long and short positions
    can_short = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.05,     # 5% profit target
        "20": 0.04,    # 4% profit target after 20 minutes
        "40": 0.03,    # 3% profit target after 40 minutes
        "120": 0.02    # 2% profit target after 120 minutes
    }

    # Stoploss settings - tighter for leverage
    stoploss = -0.015  # 1.5% stoploss

    # Trailing stop settings
    trailing_stop = True
    trailing_stop_positive = 0.01     # 1% trailing stop
    trailing_stop_positive_offset = 0.02  # 2% offset
    trailing_only_offset_is_reached = True

    # Stoploss on exchange settings
    stoploss_on_exchange = True
    stoploss_on_exchange_interval = 60
    stoploss_on_exchange_limit_ratio = 0.99

    # Run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Trading controls
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires
    startup_candle_count: int = 30

    # Divergence detection parameters
    divergence_lookback = 10  # Reduced for more frequent signals
    profit_factor = 1.5      # Risk:reward ratio

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if len(dataframe) == 0:
            logger.warning(f"Empty dataframe for {metadata['pair']}")
            return dataframe

        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)

        # MACD for trend and divergence
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMAs for trend following
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_34'] = ta.EMA(dataframe, timeperiod=34)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((bollinger['upper'] - bollinger['lower']) / bollinger['mid'])

        # ATR for dynamic stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Support and Resistance levels using recent lows/highs
        window = 20
        dataframe['rolling_low'] = dataframe['low'].rolling(window=window).min()
        dataframe['rolling_high'] = dataframe['high'].rolling(window=window).max()
        
        # Calculate dynamic stoploss levels
        dataframe['long_sl'] = dataframe['close'] - (dataframe['atr'] * 1.5)  # 1.5 ATR for long stoploss
        dataframe['short_sl'] = dataframe['close'] + (dataframe['atr'] * 1.5)  # 1.5 ATR for short stoploss
        
        # Use support/resistance to adjust stoploss if they're closer
        dataframe['long_sl'] = dataframe[['long_sl', 'rolling_low']].max(axis=1)  # Use higher of ATR or support
        dataframe['short_sl'] = dataframe[['short_sl', 'rolling_high']].min(axis=1)  # Use lower of ATR or resistance

        # Trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # Price action
        dataframe['higher_high'] = (dataframe['high'] > dataframe['high'].shift(1)) & (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        dataframe['lower_low'] = (dataframe['low'] < dataframe['low'].shift(1)) & (dataframe['low'].shift(1) < dataframe['low'].shift(2))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Log key indicators for debugging
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            logger.info(f"Analyzing {metadata['pair']}")
            logger.info(f"RSI: {last_candle['rsi']:.2f}, MACD: {last_candle['macd']:.6f}, Volume Ratio: {last_candle['volume_ratio']:.2f}")
            logger.info(f"ADX: {last_candle['adx']:.2f}, BB Width: {last_candle['bb_width']:.4f}")

        # Long entry conditions
        dataframe.loc[
            (
                # Trend conditions - Relaxed
                (
                    (dataframe['ema_8'] > dataframe['ema_13']) &
                    (dataframe['close'] > dataframe['ema_8']) &
                    (
                        (dataframe['adx'] > 15) |  # Lower ADX requirement
                        (dataframe['volume_ratio'] > 1.2)  # Higher volume can compensate
                    )
                ) &
                
                # Price action confirmation
                (
                    (dataframe['higher_high']) |
                    (dataframe['close'] > dataframe['bb_upperband'])  # Breakout potential
                ) &
                
                # RSI conditions - Adjusted for ranging market
                (
                    (
                        (dataframe['rsi'] < 40) &  # Oversold condition
                        (dataframe['rsi'] > dataframe['rsi_ma'])  # Starting to recover
                    ) |
                    (
                        (dataframe['rsi'] > 45) &  # Not overbought
                        (dataframe['rsi'] < 65) &
                        (dataframe['volume_ratio'] > 1.0)  # Good volume
                    )
                ) &
                
                # MACD confirmation - More sensitive
                (
                    (dataframe['macdhist'] > 0) |
                    (
                        (dataframe['macdhist'] < 0) &
                        (dataframe['macdhist'] > dataframe['macdhist'].shift(1))  # Rising histogram
                    )
                ) &
                
                # Volatility check - Adjusted for current conditions
                (dataframe['bb_width'] > 0.005)  # Lowered minimum volatility requirement
            ),
            'enter_long'] = 1

        # Short entry conditions
        dataframe.loc[
            (
                # Trend conditions - Relaxed
                (
                    (dataframe['ema_8'] < dataframe['ema_13']) &
                    (dataframe['close'] < dataframe['ema_8']) &
                    (
                        (dataframe['adx'] > 15) |  # Lower ADX requirement
                        (dataframe['volume_ratio'] > 1.2)  # Higher volume can compensate
                    )
                ) &
                
                # Price action confirmation
                (
                    (dataframe['lower_low']) |
                    (dataframe['close'] < dataframe['bb_lowerband'])  # Breakdown potential
                ) &
                
                # RSI conditions - Adjusted for ranging market
                (
                    (
                        (dataframe['rsi'] > 60) &  # Overbought condition
                        (dataframe['rsi'] < dataframe['rsi_ma'])  # Starting to decline
                    ) |
                    (
                        (dataframe['rsi'] < 55) &  # Not oversold
                        (dataframe['rsi'] > 35) &
                        (dataframe['volume_ratio'] > 1.0)  # Good volume
                    )
                ) &
                
                # MACD confirmation - More sensitive
                (
                    (dataframe['macdhist'] < 0) |
                    (
                        (dataframe['macdhist'] > 0) &
                        (dataframe['macdhist'] < dataframe['macdhist'].shift(1))  # Falling histogram
                    )
                ) &
                
                # Volatility check - Adjusted for current conditions
                (dataframe['bb_width'] > 0.005)  # Lowered minimum volatility requirement
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit conditions
        dataframe.loc[
            (
                (
                    # Trend reversal
                    (dataframe['ema_8'] < dataframe['ema_13']) &
                    (dataframe['close'] < dataframe['ema_8'])
                ) |
                # RSI overbought
                (dataframe['rsi'] > 75) |
                # MACD bearish cross
                (
                    (dataframe['macdhist'] < 0) &
                    (dataframe['macdhist'].shift(1) > 0)
                ) |
                # Volatility expansion exit
                (
                    (dataframe['bb_width'] > 0.03) &
                    (dataframe['close'] < dataframe['bb_middleband'])
                )
            ),
            'exit_long'] = 1

        # Short exit conditions
        dataframe.loc[
            (
                (
                    # Trend reversal
                    (dataframe['ema_8'] > dataframe['ema_13']) &
                    (dataframe['close'] > dataframe['ema_8'])
                ) |
                # RSI oversold
                (dataframe['rsi'] < 25) |
                # MACD bullish cross
                (
                    (dataframe['macdhist'] > 0) &
                    (dataframe['macdhist'].shift(1) < 0)
                ) |
                # Volatility expansion exit
                (
                    (dataframe['bb_width'] > 0.03) &
                    (dataframe['close'] > dataframe['bb_middleband'])
                )
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new stoploss value.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return self.stoploss
            
        last_candle = dataframe.iloc[-1]
        
        # Get the appropriate stoploss level based on position type
        if trade.is_short:
            sl_price = last_candle['short_sl']
            sl_percentage = ((sl_price - current_rate) / current_rate)
        else:
            sl_price = last_candle['long_sl']
            sl_percentage = ((sl_price - current_rate) / current_rate)
        
        # Log stoploss adjustments
        logger.info(f"Custom stoploss for {pair}: Current rate: {current_rate:.8f}, SL price: {sl_price:.8f}, SL percentage: {sl_percentage:.3%}")
        
        # If we have a good profit, protect it
        if current_profit > 0.03:  # 3% profit
            sl_percentage = max(sl_percentage, current_profit * 0.5)  # Protect 50% of current profit
            logger.info(f"Protecting profits for {pair}: New SL percentage: {sl_percentage:.3%}")
        
        return sl_percentage