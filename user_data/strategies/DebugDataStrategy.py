from freqtrade.strategy import IStrategy
from pandas import DataFrame
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DebugDataStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI and stoploss (not important for debugging)
    minimal_roi = {"0": 0.10}
    stoploss = -0.10

    # Timeframes to use
    timeframe = '15m'
    informative_timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logs the available data for each pair and timeframe.
        """
        pair = metadata['pair']
        logger.info(f"Processing pair: {pair}")

        # Log 15-minute data
        logger.info(f"15m data for {pair}:")
        logger.info(f"First candle: {dataframe.iloc[0]['date']}")
        logger.info(f"Last candle: {dataframe.iloc[-1]['date']}")
        logger.info(f"Total candles: {len(dataframe)}")

        # Get 1-hour data
        informative = self.dp.get_pair_dataframe(pair, self.informative_timeframe)
        if informative is not None:
            logger.info(f"1h data for {pair}:")
            logger.info(f"First candle: {informative.iloc[0]['date']}")
            logger.info(f"Last candle: {informative.iloc[-1]['date']}")
            logger.info(f"Total candles: {len(informative)}")
        else:
            logger.warning(f"No 1h data found for {pair}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No entry signals for this debug strategy.
        """
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No exit signals for this debug strategy.
        """
        return dataframe