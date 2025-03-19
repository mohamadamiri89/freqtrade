from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time
import logging
from functools import lru_cache
import pytz

logger = logging.getLogger(__name__)

class AITradingStrategy(IStrategy):
    """
    AI-powered trading strategy that combines technical indicators with
    AI model recommendations through Ollama. Includes a robust fallback system.
    """
    INTERFACE_VERSION = 3

    # Strategy parameters
    timeframe = '5m'  # Changed from 15m to 5m for more opportunities
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    startup_candle_count: int = 100
    lookback_candles: int = 100

    # Set timezone
    timezone = pytz.timezone('Asia/Tehran')

    # Risk parameters
    minimal_roi = {
        "0": 0.03,     # 3% immediate profit target
        "15": 0.02,    # 2% after 15 minutes
        "30": 0.015,   # 1.5% after 30 minutes
        "60": 0.01     # 1% after 60 minutes
    }
    stoploss = -0.02  # 2% stoploss - tighter for 5m
    trailing_stop = True
    trailing_stop_positive = 0.005   # 0.5%
    trailing_stop_positive_offset = 0.01  # 1.0%
    trailing_only_offset_is_reached = True

    # AI parameters
    min_ai_confidence = 0.65  # Keep unchanged
    high_confidence_threshold = 0.85  # Keep unchanged
    
    # Indicator parameters (class variables for easy hyperopt tuning)
    rsi_window = 14
    rsi_oversold = 30  # More standard value
    rsi_overbought = 70  # More standard value
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Ollama settings - Optimized for CPU mode
    ollama_api_url = "http://localhost:11434/api/generate"
    ollama_model = "mistral"
    max_retries = 3  # Reduced from 5 to avoid long wait times
    retry_delay = 1  # Reduced initial delay
    api_request_timeout = 45  # Increased from 30 for CPU mode
    initial_load_timeout = 90  # Increased for initial model loading
    max_warmup_attempts = 3  # Reduced from 5
    
    # Add performance tracking
    api_calls = 0
    successful_calls = 0
    failed_calls = 0
    total_response_time = 0
    is_initial_load = True
    
    # Cache expiry settings
    cache_expiry = 600  # Increased to 10 minutes to reduce API load
    
    # Fee parameters
    fee_entry = 0.001
    fee_exit = 0.001
    
    # Add detailed logging for AI status
    ai_status_log_interval = 300
    last_ai_status_log = 0
    
    # New indicator parameters
    adx_window = 14
    adx_threshold = 20  # Reduced from 25 to allow trades in weaker trends
    
    # Market regime detection
    is_trend_market = False
    
    # Add new class variables for hidden divergence settings
    hidden_div_lookback = 20  # Lookback period for hidden divergence
    hidden_div_threshold = 0.02  # Minimum price difference for divergence
    
    # Hidden divergence settings
    minimal_divergence_strength = 0.4  # Reduced from 0.5 for more sensitivity
    max_divergence_age = 15  # Increased from 10 to allow more historical divergence detection
    divergence_lookback = 20  # Increased from 15 to look back further for divergences
    
    # Market regime detection
    trend_strength_threshold = 0.6  # Reduced from 0.7 to be more lenient with trend strength
    
    class AIMetrics:
        """Class to track AI API performance metrics"""
        def __init__(self):
            self.calls = 0
            self.successful_calls = 0
            self.failed_calls = 0
            self.total_response_time = 0
            self.last_log_time = time.time()
            self.log_interval = 600  # log every 10 minutes
            
        def record(self, success: bool, response_time: float):
            """Record metrics for an API call"""
            self.calls += 1
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            self.total_response_time += response_time
            
            # Log metrics periodically
            current_time = time.time()
            if current_time - self.last_log_time > self.log_interval:
                self.log_status()
                self.last_log_time = current_time
                
        def log_status(self):
            """Log current metrics"""
            if self.calls == 0:
                return
                
            avg_response_time = self.total_response_time / self.calls
            success_rate = (self.successful_calls / self.calls) * 100
            logger.info(f"AI API Metrics - Calls: {self.calls}, "
                       f"Success Rate: {success_rate:.1f}%, "
                       f"Avg Response Time: {avg_response_time:.2f}s")
    
    class AIStatus:
        """Class to track AI service status"""
        def __init__(self):
            self.ready = False
            self.last_check_time = 0
            self.check_interval = 300  # 5 minutes between readiness checks
            self.warmup_attempts = 0
            self.max_warmup_attempts = 5
    
    def __init__(self, config: dict) -> None:
        """Initialize strategy with performance tracking"""
        super().__init__(config)
        self.metrics = self.AIMetrics()
        self.ai_status = self.AIStatus()
        self.advice_cache = {}
        self.last_ai_check_time = 0
        self.ai_check_interval = 300  # 5 minutes between checks
        self.ai_ready = False
        self.ai_warmup_attempts = 0  # Initialize warmup attempts counter
        self.ws_pool = {}  # WebSocket connection pool
        self.ws_pool_size = 5  # Maximum number of concurrent connections
        self.ws_timeout = 30  # Connection timeout in seconds

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate technical indicators and consult AI for trading advice"""
        # Initialize AI-related columns with default values
        dataframe['ai_action'] = 'no_trade'
        dataframe['ai_stop_loss'] = self.stoploss
        # Get the immediate ROI value (first key in minimal_roi dict)
        immediate_roi = next(iter(self.minimal_roi.values()))
        dataframe['ai_take_profit'] = immediate_roi
        dataframe['ai_leverage'] = 1.0
        dataframe['ai_confidence'] = 0.0
        dataframe['ai_reasoning'] = ''
        
        # Calculate basic indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['macd'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)[0]
        dataframe['macdsignal'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)[1]
        dataframe['macdhist'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2]
        dataframe['adx'] = ta.ADX(dataframe)
        
        # Calculate Bollinger Bands
        bollinger = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upperband'] = bollinger[0]
        dataframe['bb_middleband'] = bollinger[1]
        dataframe['bb_lowerband'] = bollinger[2]
        
        # Calculate volatility
        dataframe['volatility'] = ta.STDDEV(dataframe['close'], timeperiod=20) / dataframe['close']
        
        # Calculate EMA and slope
        dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema_50_slope'] = dataframe['ema_50'].diff(5)
        
        # Determine trend
        dataframe['trend'] = np.where(dataframe['close'] > dataframe['close'].shift(1), 1, -1)
        
        # Only consult AI for the most recent candle
        if len(dataframe) > 0:
            try:
                market_summary = self.prepare_market_summary(dataframe)
                logger.info(f"Market summary prepared for {metadata['pair']}: {market_summary}")
                
                # Consult AI for trading advice
                if self.check_ai_ready():
                    logger.info(f"AI is ready, consulting for trading advice for {metadata['pair']}...")
                    prompt = self.prepare_ai_prompt(market_summary)
                    logger.info(f"Generated AI prompt for {metadata['pair']}: {prompt}")
                    
                    # Get AI advice
                    advice = self.consult_ai(prompt)
                    if advice:
                        logger.info(f"Received valid AI advice for {metadata['pair']}: {advice}")
                        # Update only the last row with AI advice
                        dataframe.loc[dataframe.index[-1], 'ai_action'] = advice['action']
                        dataframe.loc[dataframe.index[-1], 'ai_stop_loss'] = advice['stop_loss']
                        dataframe.loc[dataframe.index[-1], 'ai_take_profit'] = advice['take_profit']
                        dataframe.loc[dataframe.index[-1], 'ai_leverage'] = advice['leverage']
                        dataframe.loc[dataframe.index[-1], 'ai_confidence'] = advice['confidence']
                        dataframe.loc[dataframe.index[-1], 'ai_reasoning'] = advice['reasoning']
                    else:
                        logger.warning(f"No valid AI advice received for {metadata['pair']}, using technical analysis only")
                else:
                    logger.warning(f"AI is not ready for {metadata['pair']}, using technical analysis only")
                    
            except Exception as e:
                logger.error(f"Error during AI consultation for {metadata['pair']}: {str(e)}")
        
        return dataframe
    
    def check_ai_ready(self) -> bool:
        """Check if the AI model is ready to respond"""
        current_time = time.time()
        
        # Only check periodically to avoid too many requests
        if current_time - self.last_ai_check_time < self.ai_check_interval:
            return self.ai_ready
            
        self.last_ai_check_time = current_time
        
        try:
            # Simple test query
            test_data = {
                "model": self.ollama_model,
                "prompt": "[INST]Reply with a JSON object containing a 'status' field with value 'ready'.[/INST]",
                "stream": False,
                "raw": True  # Added for faster response
            }
            
            # Use longer timeout for initial load
            timeout = self.initial_load_timeout if self.is_initial_load else self.api_request_timeout
            
            response = requests.post(
                self.ollama_api_url,
                headers={"Content-Type": "application/json"},
                json=test_data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    response_text = response_json.get('response', '')
                    logger.info(f"AI readiness check response: {response_text}")
                    
                    # Try to extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > 0:
                        json_str = response_text[json_start:json_end]
                        try:
                            status_json = json.loads(json_str)
                            self.ai_ready = status_json.get('status', '').lower() == 'ready'
                        except json.JSONDecodeError:
                            self.ai_ready = False
                    else:
                        # Fallback to text check if no JSON found
                        self.ai_ready = 'ready' in response_text.lower()
                        
                    if self.ai_ready:
                        logger.info("AI model is ready and responding")
                        self.is_initial_load = False  # Reset initial load flag
                        self.ai_warmup_attempts = 0
                    else:
                        logger.warning("AI model responded but not ready")
                        
                    return self.ai_ready
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing AI readiness response: {str(e)}")
                    self.ai_ready = False
                    
            else:
                logger.error(f"AI readiness check failed with status code: {response.status_code}")
                self.ai_ready = False
                
        except Exception as e:
            logger.error(f"AI readiness check failed: {str(e)}")
            self.ai_ready = False
            
        return False
    
    def generate_market_summary(self, dataframe: DataFrame) -> str:
        """Generate a concise market summary from indicators"""
        if len(dataframe) < 24:  # Minimum required candles
            return "Insufficient data for market summary"
            
        current = dataframe.iloc[-1]
        previous = dataframe.iloc[-2]
        
        try:
            # Calculate recent performance
            price_change_24h = ((current['close'] / dataframe.iloc[-24]['close']) - 1) * 100
            
            # Determine volatility level
            volatility_level = "high" if current['volatility'] > dataframe['volatility'].mean() * 1.5 else "normal"
            
            # Determine overall trend
            if current['trend'] > 0 and current['trend'] == previous['trend']:
                trend = "strong_uptrend"
            elif current['trend'] > 0:
                trend = "uptrend"
            elif current['trend'] < 0 and current['trend'] == previous['trend']:
                trend = "strong_downtrend"
            else:
                trend = "downtrend"
            
            # Identify key support/resistance levels
            support = round(current['bb_lowerband'], 4)
            resistance = round(current['bb_upperband'], 4)
            
            # MACD analysis
            macd_cross = "bullish" if current['macd'] > current['macdsignal'] and previous['macd'] <= previous['macdsignal'] else \
                        "bearish" if current['macd'] < current['macdsignal'] and previous['macd'] >= previous['macdsignal'] else "none"
            
            # RSI analysis
            rsi_condition = "oversold" if current['rsi'] < self.rsi_oversold else \
                          "overbought" if current['rsi'] > self.rsi_overbought else "neutral"
            
            return (
                f"MARKET SUMMARY:\n"
                f"Price: {current['close']:.4f}, 24h Change: {price_change_24h:.2f}%\n"
                f"Trend: {trend}, Volatility: {volatility_level}\n"
                f"Support: {support}, Resistance: {resistance}\n"
                f"RSI ({self.rsi_window}): {current['rsi']:.2f} - {rsi_condition}\n"
                f"MACD Cross: {macd_cross}"
            )
        except Exception as e:
            logger.error(f"Error generating market summary: {str(e)}")
            return "Error generating market summary"
    
    def prepare_candle_data(self, dataframe: DataFrame) -> Dict:
        """Prepare structured candle data for AI analysis"""
        # Take the most recent candles up to lookback_candles
        recent_candles = dataframe.tail(self.lookback_candles).copy()
        
        # Convert to a more efficient dictionary representation
        candles = []
        for idx, row in recent_candles.iterrows():
            candle = {
                "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": round(row['open'], 6),
                "high": round(row['high'], 6),
                "low": round(row['low'], 6),
                "close": round(row['close'], 6),
                "volume": round(row['volume'], 2),
                "rsi": round(row['rsi'], 2) if not pd.isna(row['rsi']) else None,
                "macd": round(row['macd'], 6) if not pd.isna(row['macd']) else None,
                "macdsignal": round(row['macdsignal'], 6) if not pd.isna(row['macdsignal']) else None,
                "macdhist": round(row['macdhist'], 6) if not pd.isna(row['macdhist']) else None,
                "bb_upper": round(row['bb_upperband'], 6) if not pd.isna(row['bb_upperband']) else None,
                "bb_middle": round(row['bb_middleband'], 6) if not pd.isna(row['bb_middleband']) else None,
                "bb_lower": round(row['bb_lowerband'], 6) if not pd.isna(row['bb_lowerband']) else None,
                "trend": int(row['trend']) if not pd.isna(row['trend']) else None
            }
            candles.append(candle)
        
        return {"candles": candles}
    
    def get_cached_ai_advice(self, pair: str, current_time: datetime) -> Optional[Dict]:
        """Retrieve cached AI advice if still valid"""
        if pair in self.advice_cache:
            advice, timestamp = self.advice_cache[pair]
            # Check if the cached advice is still valid
            if (current_time - timestamp).total_seconds() < self.cache_expiry:
                logger.debug(f"Using cached AI advice for {pair} from {timestamp}")
                return advice
        return None
    
    def store_ai_advice(self, pair: str, advice: Dict, current_time: datetime):
        """Store AI advice in cache with timestamp"""
        self.advice_cache[pair] = (advice, current_time)
    
    def cleanup_cache(self, current_time: datetime):
        """Remove expired entries from cache"""
        expired_pairs = []
        for pair, (_, timestamp) in self.advice_cache.items():
            if (current_time - timestamp).total_seconds() >= self.cache_expiry:
                expired_pairs.append(pair)
                
        for pair in expired_pairs:
            del self.advice_cache[pair]
    
    def get_trading_advice(self, dataframe: DataFrame, metadata: dict) -> Optional[Dict]:
        """Get trading advice from AI model with improved error handling and caching"""
        if not self.check_ai_ready():
            logger.warning("AI model not ready, skipping consultation")
            return None
            
        # Prepare market data summary
        current_candle = dataframe.iloc[-1]
        market_summary = self.generate_market_summary(dataframe)
        
        # Check cache first
        cache_key = f"{metadata['pair']}_{current_candle.name}"
        if cache_key in self.advice_cache:
            cached_advice = self.advice_cache[cache_key]
            if time.time() - cached_advice['timestamp'] < self.cache_expiry:
                return cached_advice['advice']
        
        # Prepare prompt with simplified data
        prompt = self.prepare_ai_prompt(market_summary)
        
        # Initialize retry counter and delay
        retry_count = 0
        current_delay = self.retry_delay
        
        while retry_count < self.max_retries:
            try:
                start_time = time.time()
                
                # Make API request with optimized settings
                response = requests.post(
                    self.ollama_api_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "raw": True,  # Added for faster response
                        "temperature": 0.1  # Lower temperature for more consistent responses
                    },
                    timeout=self.api_request_timeout
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    try:
                        # Parse response and validate
                        advice = self.parse_ai_response(response.json().get('response', ''))
                        if advice and self.validate_ai_advice(advice):
                            # Update metrics
                            self.metrics.record(True, response_time)
                            
                            # Cache the valid response
                            self.advice_cache[cache_key] = {
                                'advice': advice,
                                'timestamp': time.time()
                            }
                            
                            return advice
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Failed to parse AI response: {str(e)}")
                        self.metrics.record(False, response_time)
                
                # If we get here, either the response was invalid or couldn't be parsed
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"AI consultation failed (attempt {retry_count}/{self.max_retries})")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                
            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"AI consultation timed out (attempt {retry_count}/{self.max_retries})")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error during AI consultation: {str(e)}")
                self.metrics.record(False, time.time() - start_time)
                break
        
        logger.warning("All AI consultation attempts failed, falling back to technical analysis")
        return None
    
    def generate_fallback_advice(self, dataframe: pd.DataFrame) -> Dict:
        """Generate trading advice using technical indicators when AI is unavailable"""
        current = dataframe.iloc[-1]
        
        # Initialize default advice
        advice = {
            'action': 'hold',
            'confidence': 0.6,  # Lower confidence for fallback
            'stop_loss': self.stoploss,
            'take_profit': self.minimal_roi['0'],
            'leverage': 1.0,
            'reasoning': 'Using fallback strategy based on technical indicators'
        }
        
        # Get technical indicators
        rsi = current['rsi']
        macd = current['macd']
        macd_signal = current['macdsignal']
        adx = current['adx']
        is_trending = current['is_trending']
        
        # Strong trend conditions
        strong_trend = adx > self.adx_threshold
        
        # Long entry conditions
        if (rsi < self.rsi_oversold and 
            macd > macd_signal and 
            strong_trend):
            advice.update({
                'action': 'long',
                'confidence': 0.7 if is_trending else 0.6,
                'reasoning': 'Fallback long signal: Oversold RSI with MACD crossover and strong trend'
            })
            
        # Short entry conditions
        elif (rsi > self.rsi_overbought and 
              macd < macd_signal and 
              strong_trend):
            advice.update({
                'action': 'short',
                'confidence': 0.7 if is_trending else 0.6,
                'reasoning': 'Fallback short signal: Overbought RSI with MACD crossover and strong trend'
            })
            
        return advice
    
    def detect_hidden_divergence(self, dataframe: DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect hidden bullish and bearish divergences"""
        # Initialize divergence signals
        bullish_div = pd.Series(False, index=dataframe.index)
        bearish_div = pd.Series(False, index=dataframe.index)
        
        # Look for hidden divergences within the lookback period
        for i in range(self.hidden_div_lookback, len(dataframe)):
            window = dataframe.iloc[i-self.hidden_div_lookback:i+1]
            
            # Find local price extremes
            price_min_idx = window['close'].idxmin()
            price_max_idx = window['close'].idxmax()
            
            # Find corresponding RSI values
            rsi_at_price_min = window.loc[price_min_idx, 'rsi']
            rsi_at_price_max = window.loc[price_max_idx, 'rsi']
            
            current_price = window['close'].iloc[-1]
            current_rsi = window['rsi'].iloc[-1]
            
            # Hidden Bullish Divergence - more lenient
            if (current_price > window['close'].min() and  # Higher low in price
                current_rsi < rsi_at_price_min and  # Lower low in RSI
                (current_price - window['close'].min()) / window['close'].min() > self.hidden_div_threshold * 0.5):  # Reduced threshold
                bullish_div.iloc[i] = True
            
            # Hidden Bearish Divergence - more lenient
            if (current_price < window['close'].max() and  # Lower high in price
                current_rsi > rsi_at_price_max and  # Higher high in RSI
                (window['close'].max() - current_price) / current_price > self.hidden_div_threshold * 0.5):  # Reduced threshold
                bearish_div.iloc[i] = True
        
        return bullish_div, bearish_div
    
    def consult_ai(self, prompt: str) -> Optional[Dict]:
        """Consult the AI model for trading advice with strict timeout"""
        start_time = time.time()
        
        try:
            # Make API request with strict timeout
            response = requests.post(
                self.ollama_api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,
                    "temperature": 0.1
                },
                timeout=90  # Strict 90-second timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    # Parse response and validate
                    advice = self.parse_ai_response(response.json().get('response', ''))
                    if advice:
                        self.metrics.record(True, response_time)
                        return advice
                except Exception as e:
                    logger.error(f"Failed to parse AI response: {str(e)}")
                    self.metrics.record(False, response_time)
            
            logger.warning("AI consultation failed, falling back to technical analysis")
            return None
            
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            logger.warning(f"AI request timed out or failed: {str(e)}")
            return None
    
    def extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text with improved reliability"""
        # Find the JSON part in the response using balanced braces
        depth = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx != -1:
                    end_idx = i + 1
                    break
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx]
        return None
    
    def validate_ai_advice(self, advice: Dict) -> bool:
        """Validate and sanitize AI advice fields"""
        # Define default values for required fields
        defaults = {
            'action': 'no_trade',
            'leverage': 1.0,
            'take_profit': 0.02,
            'stop_loss': -0.02,
            'confidence': 0.0,
            'reasoning': 'No specific reasoning provided'
        }
        
        # Create a sanitized version of the advice with defaults
        sanitized = defaults.copy()
        
        # Update with provided values
        for key, value in advice.items():
            if key in defaults:
                sanitized[key] = value
        
        # Validate action field
        action = sanitized['action'].lower()
        if action not in ['open_long', 'open_short', 'no_trade']:
            action = 'no_trade'
        sanitized['action'] = action
        
        try:
            # More lenient validation bounds
            # Leverage: between 1 and 5
            leverage = float(sanitized['leverage'])
            sanitized['leverage'] = max(1.0, min(5.0, leverage))
            
            # Take profit: between 0.2% and 10%
            take_profit = float(sanitized['take_profit'])
            sanitized['take_profit'] = max(0.002, min(0.10, take_profit))
            
            # Stop loss: between -0.2% and -5%
            stop_loss = float(sanitized['stop_loss'])
            sanitized['stop_loss'] = max(-0.05, min(-0.002, stop_loss))
            
            # Confidence: between 0 and 1
            confidence = float(sanitized['confidence'])
            sanitized['confidence'] = max(0.0, min(1.0, confidence))
            
            return True
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric values in AI response, using defaults")
            return False
    
    def get_base_entry_conditions(self, dataframe: DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Get base conditions for entry signals"""
        # Volume conditions - less restrictive
        volume_conditions = (
            (dataframe['volume'] > 0) &  
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 0.5)  # Reduced volume requirement
        )
        
        # ADX filter - lower threshold
        trend_filter = (dataframe['adx'] > 15)  # Reduced from original threshold
        
        # Higher timeframe trend alignment - more lenient
        uptrend_confirmed = (
            (dataframe['ema_50_slope'] > -0.0002) |  # Allow slightly negative slope
            (dataframe['close'] > dataframe['ema_50'])  # Or price above EMA
        )
        downtrend_confirmed = (
            (dataframe['ema_50_slope'] < 0.0002) |  # Allow slightly positive slope
            (dataframe['close'] < dataframe['ema_50'])  # Or price below EMA
        )
        
        # Long conditions - less stringent
        long_conditions = (
            volume_conditions &
            (
                trend_filter |  # Either trending
                (dataframe['volatility'] > dataframe['volatility'].rolling(10).mean())  # Or volatile
            ) &
            uptrend_confirmed &
            (
                (dataframe['rsi'] < 40) |  # More lenient RSI
                (dataframe['close'] < dataframe['bb_lowerband'] * 1.05)  # More lenient BB
            ) &
            (dataframe['macd'] > dataframe['macd'].shift(1))  # MACD rising
        )
        
        # Short conditions - less stringent
        short_conditions = (
            volume_conditions &
            (
                trend_filter |  # Either trending
                (dataframe['volatility'] > dataframe['volatility'].rolling(10).mean())  # Or volatile
            ) &
            downtrend_confirmed &
            (
                (dataframe['rsi'] > 60) |  # More lenient RSI
                (dataframe['close'] > dataframe['bb_upperband'] * 0.95)  # More lenient BB
            ) &
            (dataframe['macd'] < dataframe['macd'].shift(1))  # MACD falling
        )
        
        return long_conditions, short_conditions

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate entry signals using technical analysis with AI as optional enhancement"""
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Skip if we don't have enough data
        if len(dataframe) <= self.lookback_candles:
            logger.info(f"Skipping {metadata['pair']} - not enough data")
            return dataframe
            
        # Get basic technical indicator conditions
        long_conditions, short_conditions = self.get_base_entry_conditions(dataframe)
        
        # Log technical conditions for the latest candle
        logger.info(f"Technical conditions for {metadata['pair']}:")
        logger.info(f"Long conditions met: {long_conditions.iloc[-1]}")
        logger.info(f"Short conditions met: {short_conditions.iloc[-1]}")
        
        # Detect hidden divergences
        bullish_div, bearish_div = self.detect_hidden_divergence(dataframe)
        
        # Log divergence conditions
        logger.info(f"Divergence conditions for {metadata['pair']}:")
        logger.info(f"Bullish divergence: {bullish_div.iloc[-1]}")
        logger.info(f"Bearish divergence: {bearish_div.iloc[-1]}")
        
        last_index = dataframe.index[-1]
        
        # Try to get AI advice but don't require it
        try:
            market_summary = self.prepare_market_summary(dataframe)
            prompt = self.prepare_ai_prompt(market_summary)
            ai_advice = self.consult_ai(prompt)
            
            if ai_advice and ai_advice.get('confidence', 0.0) >= self.min_ai_confidence:
                # Use AI advice if available and confident
                action = ai_advice.get('action', 'no_trade')
                if action == 'open_long':
                    dataframe.loc[last_index, 'enter_long'] = 1
                    logger.info(f"Opening LONG position based on AI advice for {metadata['pair']}")
                elif action == 'open_short':
                    dataframe.loc[last_index, 'enter_short'] = 1
                    logger.info(f"Opening SHORT position based on AI advice for {metadata['pair']}")
            else:
                # Use technical analysis if AI is not available or not confident
                # Long entry conditions
                if long_conditions.iloc[-1]:
                    if bullish_div.iloc[-1]:  # Strong signal with divergence
                        dataframe.loc[last_index, 'enter_long'] = 1
                        logger.info(f"Opening LONG position based on technical analysis with divergence for {metadata['pair']}")
                    elif dataframe['adx'].iloc[-1] > 25:  # Strong trend without divergence
                        dataframe.loc[last_index, 'enter_long'] = 1
                        logger.info(f"Opening LONG position based on strong trend for {metadata['pair']}")
                
                # Short entry conditions
                if short_conditions.iloc[-1]:
                    if bearish_div.iloc[-1]:  # Strong signal with divergence
                        dataframe.loc[last_index, 'enter_short'] = 1
                        logger.info(f"Opening SHORT position based on technical analysis with divergence for {metadata['pair']}")
                    elif dataframe['adx'].iloc[-1] > 25:  # Strong trend without divergence
                        dataframe.loc[last_index, 'enter_short'] = 1
                        logger.info(f"Opening SHORT position based on strong trend for {metadata['pair']}")
        
        except Exception as e:
            logger.error(f"Error during AI consultation for {metadata['pair']}: {str(e)}")
            # Use technical analysis as fallback
            if long_conditions.iloc[-1] and (bullish_div.iloc[-1] or dataframe['adx'].iloc[-1] > 25):
                dataframe.loc[last_index, 'enter_long'] = 1
                logger.info(f"Opening LONG position based on technical analysis for {metadata['pair']}")
            
            if short_conditions.iloc[-1] and (bearish_div.iloc[-1] or dataframe['adx'].iloc[-1] > 25):
                dataframe.loc[last_index, 'enter_short'] = 1
                logger.info(f"Opening SHORT position based on technical analysis for {metadata['pair']}")
        
        # Log final entry decisions
        logger.info(f"Final entry decisions for {metadata['pair']}:")
        logger.info(f"Enter long: {dataframe.loc[last_index, 'enter_long']}")
        logger.info(f"Enter short: {dataframe.loc[last_index, 'enter_short']}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals based on technical indicators and AI advice"""
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Basic technical exit conditions
        dataframe['exit_long'] = (
            # RSI moving into overbought
            (dataframe['rsi'] > self.rsi_overbought) &
            # MACD crossed below signal line
            (dataframe['macd'] < dataframe['macdsignal']) &
            # Price at or above upper Bollinger Band
            (dataframe['close'] >= dataframe['bb_upperband'] * 0.99)
        ).astype('int')
        
        dataframe['exit_short'] = (
            # RSI moving into oversold
            (dataframe['rsi'] < self.rsi_oversold) &
            # MACD crossed above signal line
            (dataframe['macd'] > dataframe['macdsignal']) &
            # Price at or below lower Bollinger Band
            (dataframe['close'] <= dataframe['bb_lowerband'] * 1.01)
        ).astype('int')

        return dataframe 
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, 
                      current_rate: float, current_profit: float, **kwargs) -> float:
        """Custom stoploss based on AI recommendations"""
        # Get dataframe from dict
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Use AI-recommended stop loss if available, otherwise use default
        ai_stop_loss = last_candle.get('ai_stop_loss', None)
        
        if ai_stop_loss is not None and not pd.isna(ai_stop_loss):
            # Convert from percentage to negative multiplier as required by freqtrade
            custom_sl = abs(ai_stop_loss)
            logger.debug(f"Using AI stop loss for {pair}: {custom_sl}")
            return custom_sl
        
        # Default to strategy stoploss setting
        return self.stoploss
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                 current_profit: float, **kwargs) -> Optional[str]:
        """Implement custom exit logic based on AI recommendations"""
        # Get dataframe from dict
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Skip if dataframe is empty
        if dataframe.empty:
            return None
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Check if AI has recommended a take profit level
        ai_take_profit = last_candle.get('ai_take_profit', None)
        
        # If we've hit the AI-recommended take profit level, exit
        if (ai_take_profit is not None and not pd.isna(ai_take_profit) and 
                current_profit >= ai_take_profit):
            logger.info(f"AI take profit reached for {pair}: current profit {current_profit:.2%}, " 
                       f"target {ai_take_profit:.2%}")
            return f"ai_take_profit_reached_{ai_take_profit:.2f}"
            
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
               proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Dynamic leverage based on AI recommendations"""
        # Get dataframe from dict
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe.empty:
            return 1.0
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Use AI-recommended leverage if available
        ai_leverage = last_candle.get('ai_leverage', None)
        
        if ai_leverage is not None and not pd.isna(ai_leverage):
            # Apply safety cap
            ai_leverage = min(ai_leverage, max_leverage)
            ai_leverage = max(1.0, ai_leverage)  # Minimum leverage is 1x
            logger.info(f"Using AI-recommended leverage for {pair}: {ai_leverage}x")
            return ai_leverage
            
        # Default to 1x leverage
        return 1.0
        
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Define informative pair/timeframe combinations to be used by the strategy"""
        return []
        
    def bot_loop_start(self, **kwargs) -> None:
        """Called at the start of a bot loop, used for logging and periodic maintenance tasks"""
        # Log AI metrics status periodically
        self.metrics.log_status()
        
        # Clean up WebSocket connections periodically
        self.cleanup_ws_pool()

    def get_ws_connection(self, pair: str) -> Optional[requests.Session]:
        """Get or create a WebSocket connection from the pool"""
        current_time = time.time()
        
        # Clean up expired connections
        expired_pairs = []
        for p, (session, last_used) in self.ws_pool.items():
            if current_time - last_used > self.ws_timeout:
                expired_pairs.append(p)
                session.close()
        
        for p in expired_pairs:
            del self.ws_pool[p]
        
        # Return existing connection if available
        if pair in self.ws_pool:
            session, _ = self.ws_pool[pair]
            self.ws_pool[pair] = (session, current_time)
            return session
            
        # Create new connection if pool isn't full
        if len(self.ws_pool) < self.ws_pool_size:
            session = requests.Session()
            self.ws_pool[pair] = (session, current_time)
            return session
            
        # If pool is full, reuse oldest connection
        oldest_pair = min(self.ws_pool.items(), key=lambda x: x[1][1])[0]
        old_session, _ = self.ws_pool[oldest_pair]
        old_session.close()
        del self.ws_pool[oldest_pair]
        
        session = requests.Session()
        self.ws_pool[pair] = (session, current_time)
        return session
        
    def cleanup_ws_pool(self):
        """Clean up all WebSocket connections"""
        for pair, (session, _) in self.ws_pool.items():
            try:
                session.close()
            except:
                pass
        self.ws_pool.clear()

    def prepare_ai_prompt(self, market_summary: str) -> str:
        """Prepare a concise prompt for the AI model"""
        prompt = (
            "[INST] You are a trading advisor analyzing market data. Provide a JSON response with EXACTLY these fields:\n"
            "{\n"
            "    \"action\": \"open_long\", \"open_short\", or \"no_trade\",\n"
            "    \"leverage\": number between 1-2,\n"
            "    \"take_profit\": percentage as decimal (0.005-0.05),\n"
            "    \"stop_loss\": percentage as decimal (-0.03 to -0.005),\n"
            "    \"confidence\": number between 0-1,\n"
            "    \"reasoning\": \"brief explanation (max 100 chars)\"\n"
            "}\n\n"
            f"Market Summary:\n{market_summary}\n\n"
            "Requirements:\n"
            "1. Only recommend trades with clear signals and high confidence\n"
            "2. Use tighter stops in high volatility markets\n"
            "3. Consider trend strength, volatility, and support/resistance\n"
            "4. Return ONLY valid JSON with ALL required fields\n"
            "5. Do not include any other text or explanations outside the JSON\n"
            "6. Keep reasoning concise and focused on key indicators[/INST]"
        )
        logger.info(f"Generated AI prompt: {prompt}")
        return prompt
        
    def prepare_market_summary(self, dataframe: DataFrame) -> str:
        """Prepare a concise market summary for AI analysis"""
        try:
            current = dataframe.iloc[-1]
            prev = dataframe.iloc[-2]
            
            # Calculate 24h change
            change_24h = ((current['close'] - dataframe.iloc[-96]['close']) / 
                         dataframe.iloc[-96]['close'] * 100) if len(dataframe) >= 96 else 0
            
            # Determine trend strength
            trend_strength = "strong_uptrend" if current['adx'] > 30 and current['trend'] == 1 else \
                            "strong_downtrend" if current['adx'] > 30 and current['trend'] == -1 else \
                            "weak_uptrend" if current['trend'] == 1 else "weak_downtrend"
            
            # Determine volatility level
            volatility = "high" if current['volatility'] > 0.002 else \
                        "low" if current['volatility'] < 0.0005 else "normal"
            
            # Calculate support and resistance
            support = current['bb_lowerband']
            resistance = current['bb_upperband']
            
            # Format market summary
            summary = (
                f"Price: {current['close']:.4f}\n"
                f"24h Change: {change_24h:.2f}%\n"
                f"Trend: {trend_strength}\n"
                f"Volatility: {volatility}\n"
                f"RSI: {current['rsi']:.2f}\n"
                f"ADX: {current['adx']:.2f}\n"
                f"MACD: {current['macd']:.4f}\n"
                f"Support: {support:.4f}\n"
                f"Resistance: {resistance:.4f}"
            )
            
            logger.info(f"Generated market summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error preparing market summary: {str(e)}")
            return "Error preparing market summary"

    def parse_ai_response(self, response_text: str) -> Optional[Dict]:
        """Parse and validate the AI response"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logger.error(f"No JSON found in response: {response_text}")
                return None
                
            json_str = response_text[json_start:json_end]
            logger.info(f"Extracted JSON: {json_str}")
            advice = json.loads(json_str)
            
            # Map the response to our expected format
            action = advice.get('action', '').lower()
            mapped_advice = {
                'action': 'open_long' if 'long' in action else 
                         'open_short' if 'short' in action else 'no_trade',
                'leverage': float(advice.get('leverage', 1.0)),
                'take_profit': float(advice.get('take_profit', 0.02)),
                'stop_loss': float(advice.get('stop_loss', -0.02)),
                'confidence': float(advice.get('confidence', 0.5)),
                'reasoning': str(advice.get('reasoning', 'No reasoning provided'))[:100]
            }
            
            logger.info(f"Mapped advice: {mapped_advice}")
            
            # Validate field values
            if mapped_advice['action'] not in ['open_long', 'open_short', 'no_trade']:
                logger.warning(f"Invalid action in response: {mapped_advice['action']}")
                return None
                
            if not (1 <= mapped_advice['leverage'] <= 2):
                logger.warning(f"Invalid leverage {mapped_advice['leverage']}, using default 1.0")
                mapped_advice['leverage'] = 1.0
                
            if not (0.005 <= mapped_advice['take_profit'] <= 0.05):
                logger.warning(f"Invalid take_profit {mapped_advice['take_profit']}, using default 0.02")
                mapped_advice['take_profit'] = 0.02
                
            if not (-0.03 <= mapped_advice['stop_loss'] <= -0.005):
                logger.warning(f"Invalid stop_loss {mapped_advice['stop_loss']}, using default -0.02")
                mapped_advice['stop_loss'] = -0.02
                
            if not (0 <= mapped_advice['confidence'] <= 1):
                logger.warning(f"Invalid confidence {mapped_advice['confidence']}, using default 0.5")
                mapped_advice['confidence'] = 0.5
                
            logger.info(f"Parsed AI advice: {mapped_advice}")
            return mapped_advice
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing AI response: {str(e)}, Response: {response_text}")
            return None