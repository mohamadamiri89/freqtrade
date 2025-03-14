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

logger = logging.getLogger(__name__)

class AITradingStrategy(IStrategy):
    """
    AI-powered trading strategy that combines technical indicators with
    AI model recommendations through Ollama. Includes a robust fallback system.
    """
    INTERFACE_VERSION = 3
    
    # Strategy settings
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 100
    lookback_candles: int = 100
    timeframe = '5m'
    
    # Risk parameters - Even more conservative settings
    minimal_roi = {
        "0": 0.02,     # 2% immediate profit target
        "30": 0.015,   # 1.5% after 30 minutes
        "60": 0.01,    # 1% after 60 minutes
        "120": 0.008   # 0.8% after 120 minutes
    }
    stoploss = -0.02  # Tighter 2% stoploss
    
    # Trailing stop settings - More conservative
    trailing_stop = True
    trailing_stop_positive = 0.008  # 0.8%
    trailing_stop_positive_offset = 0.015  # 1.5%
    trailing_only_offset_is_reached = True
    
    # Minimum AI confidence required for trades - Slightly reduced
    min_ai_confidence = 0.6  # 60% confidence required (reduced from 70%)
    
    # Indicator parameters (class variables for easy hyperopt tuning)
    rsi_window = 14
    rsi_oversold = 35  # Increased from 30
    rsi_overbought = 65  # Decreased from 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Ollama settings - Optimized for faster responses
    ollama_api_url = "http://ollama:11434/api/generate"
    ollama_model = "mistral"
    max_retries = 3
    retry_delay = 1
    api_request_timeout = 30  # Reduced timeout
    initial_load_timeout = 60  # Reduced initial load timeout
    
    # Add performance tracking
    api_calls = 0
    successful_calls = 0
    failed_calls = 0
    total_response_time = 0
    is_initial_load = True  # Track if this is the first load
    
    # Cache expiry settings
    cache_expiry = 300  # seconds
    
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
        self.ws_pool = {}  # WebSocket connection pool
        self.ws_pool_size = 5  # Maximum number of concurrent connections
        self.ws_timeout = 30  # Connection timeout in seconds
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate technical indicators for decision making"""
        if len(dataframe) < self.startup_candle_count:
            logger.warning(f"Not enough candles ({len(dataframe)}) for {metadata['pair']}, "
                         f"need at least {self.startup_candle_count}")
            return dataframe
            
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_window)
        
        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        
        # Price trends and volatility
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['price_change'].rolling(10).std()
        
        # Trend direction using EMA crossover
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['trend'] = np.where(dataframe['ema_short'] > dataframe['ema_long'], 1, -1)
        
        # Columns for AI recommendations
        dataframe['ai_action'] = ''
        dataframe['ai_stop_loss'] = 0.0
        dataframe['ai_take_profit'] = 0.0
        dataframe['ai_leverage'] = 1.0
        dataframe['ai_confidence'] = 0.0
        dataframe['ai_reasoning'] = ''
        
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
                "prompt": "Reply with 'ready' if you can process this message.",
                "stream": False
            }
            
            # Use longer timeout for initial load
            timeout = self.initial_load_timeout if self.is_initial_load else 10
            
            response = requests.post(
                self.ollama_api_url,
                headers={"Content-Type": "application/json"},
                json=test_data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                response_text = response.json().get('response', '').lower()
                self.ai_ready = 'ready' in response_text
                if self.ai_ready:
                    logger.info("AI model is ready and responding")
                    self.is_initial_load = False  # Reset initial load flag
                    self.ai_warmup_attempts = 0
                return self.ai_ready
                
        except Exception as e:
            logger.warning(f"AI readiness check failed: {str(e)}")
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
    
    def get_trading_advice(self, dataframe: DataFrame, pair: str) -> Dict:
        """Get trading advice either from AI or fallback strategy"""
        current_time = datetime.utcnow()
        
        # Clean up expired cache entries periodically
        self.cleanup_cache(current_time)
        
        # Check for cached advice first
        cached_advice = self.get_cached_ai_advice(pair, current_time)
        if cached_advice:
            return cached_advice
            
        # Check if AI is ready, otherwise use fallback
        if not self.ai_status.ready and not self.check_ai_ready():
            if self.ai_status.warmup_attempts < self.ai_status.max_warmup_attempts:
                self.ai_status.warmup_attempts += 1
                logger.info(f"AI not ready (attempt {self.ai_status.warmup_attempts}/"
                           f"{self.ai_status.max_warmup_attempts}), using fallback")
            else:
                logger.info("Maximum warmup attempts reached, continuing with fallback strategy")
            return self.generate_fallback_advice(dataframe)
        
        # Prepare data for AI
        candle_data = self.prepare_candle_data(dataframe)
        market_summary = self.generate_market_summary(dataframe)
        
        # Try to get AI advice
        ai_advice = self.consult_ai(candle_data, market_summary, pair)
        
        # Use fallback if AI fails
        if not ai_advice and self.use_fallback_strategy:
            logger.info(f"AI advice unavailable for {pair}, using fallback strategy")
            ai_advice = self.generate_fallback_advice(dataframe)
        
        # Cache the advice if available
        if ai_advice:
            self.store_ai_advice(pair, ai_advice, current_time)
            
        return ai_advice or {}
    
    def generate_fallback_advice(self, dataframe: DataFrame) -> Dict:
        """Generate advice using traditional indicators when AI is unavailable"""
        current = dataframe.iloc[-1]
        previous = dataframe.iloc[-2]
        
        # Initialize with default "no trade" action
        action = "no_trade"
        reasoning = "No clear signal from indicators"
        confidence = 0.0
        leverage = 1.0
        take_profit = 0.02
        stop_loss = -0.02
        
        # Calculate confidence factors
        confidence_factors = []
        
        # RSI factor
        if current['rsi'] < self.rsi_oversold:
            confidence_factors.append(0.8)  # Strong oversold
            rsi_signal = "bullish"
        elif current['rsi'] > self.rsi_overbought:
            confidence_factors.append(0.8)  # Strong overbought
            rsi_signal = "bearish"
        else:
            confidence_factors.append(0.4)  # Neutral
            rsi_signal = "neutral"
            
        # MACD factor
        macd_signal = "neutral"
        if current['macd'] > current['macdsignal'] and previous['macd'] <= previous['macdsignal']:
            confidence_factors.append(0.8)  # Fresh bullish crossover
            macd_signal = "bullish"
        elif current['macd'] < current['macdsignal'] and previous['macd'] >= previous['macdsignal']:
            confidence_factors.append(0.8)  # Fresh bearish crossover
            macd_signal = "bearish"
        elif current['macd'] > current['macdsignal']:
            confidence_factors.append(0.6)  # Continuing bullish
            macd_signal = "bullish"
        elif current['macd'] < current['macdsignal']:
            confidence_factors.append(0.6)  # Continuing bearish
            macd_signal = "bearish"
        else:
            confidence_factors.append(0.4)  # Neutral
            
        # Bollinger Bands factor
        bb_signal = "neutral"
        if current['close'] < current['bb_lowerband']:
            confidence_factors.append(0.7)  # Price below lower band - potential bounce
            bb_signal = "bullish"
        elif current['close'] > current['bb_upperband']:
            confidence_factors.append(0.7)  # Price above upper band - potential reversal
            bb_signal = "bearish"
        else:
            confidence_factors.append(0.4)  # Inside bands - neutral
            
        # Trend alignment
        trend_signal = "bullish" if current['trend'] > 0 else "bearish"
        if (macd_signal == "bullish" and trend_signal == "bullish") or (macd_signal == "bearish" and trend_signal == "bearish"):
            confidence_factors.append(0.8)  # Aligned trend
        else:
            confidence_factors.append(0.3)  # Divergent trend
            
        # Calculate overall confidence
        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Determine trade direction based on signals
        bullish_signals = sum(1 for s in [rsi_signal, macd_signal, bb_signal, trend_signal] if s == "bullish")
        bearish_signals = sum(1 for s in [rsi_signal, macd_signal, bb_signal, trend_signal] if s == "bearish")
        
        # Determine action based on signal alignment and confidence
        if bullish_signals >= 3 and confidence >= self.min_ai_confidence:
            action = "open_long"
            reasoning = f"Strong bullish signal: RSI={current['rsi']:.2f}, MACD crossover={macd_signal}, Trend={trend_signal}"
            
            # Adjust take profit based on volatility
            take_profit = max(0.015, min(0.04, current['volatility'] * 10))
            
            # More conservative stop loss for higher volatility
            stop_loss = -max(0.015, min(0.03, current['volatility'] * 8))
            
        elif bearish_signals >= 3 and confidence >= self.min_ai_confidence:
            action = "open_short"
            reasoning = f"Strong bearish signal: RSI={current['rsi']:.2f}, MACD crossover={macd_signal}, Trend={trend_signal}"
            
            # Adjust take profit based on volatility
            take_profit = max(0.015, min(0.04, current['volatility'] * 10))
            
            # More conservative stop loss for higher volatility
            stop_loss = -max(0.015, min(0.03, current['volatility'] * 8))
        
        return {
            "action": action,
            "leverage": leverage,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "confidence": confidence,
            "reasoning": f"Fallback strategy: {reasoning}"
        }
    
    def consult_ai(self, candle_data: Dict, market_summary: str, pair: str) -> Optional[Dict]:
        """Consult the AI model for trading advice with improved error handling"""
        start_time = time.time()
        
        # Create a more structured and strict prompt
        prompt = f"""Analyze {pair} trading data and provide a JSON response with EXACTLY these fields:
{{
    "action": "open_long", "open_short", or "no_trade",
    "leverage": number between 1-3,
    "take_profit": percentage as decimal (0.5-5%),
    "stop_loss": percentage as decimal (-0.5 to -3%),
    "confidence": number between 0-1,
    "reasoning": "brief explanation (max 100 chars)"
}}

Market Summary:
{market_summary}

Candle Data:
{json.dumps(candle_data, indent=2)}

Requirements:
1. Only recommend trades with clear signals
2. Use tighter stops in high volatility
3. Consider current market conditions
4. Return ONLY valid JSON with ALL required fields"""
        
        retry_delay = self.retry_delay
        for attempt in range(self.max_retries):
            try:
                data = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,  # Reduced for more consistent responses
                        "num_predict": 150,  # Reduced for faster responses
                        "top_k": 20,        # Reduced for more focused responses
                        "top_p": 0.8,
                        "repeat_penalty": 1.2
                    }
                }

                logger.debug(f"Sending API request to Ollama (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    self.ollama_api_url,
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=self.api_request_timeout
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    ai_response = response.json()
                    response_text = ai_response.get('response', '')
                    
                    # Extract JSON from response
                    json_match = self.extract_json_from_text(response_text)
                    
                    if json_match:
                        try:
                            advice = json.loads(json_match)
                            validated_advice = self.validate_ai_response(advice)
                            
                            if validated_advice:
                                logger.info(f"AI Advice for {pair} ({elapsed_time:.2f}s):")
                                logger.info(f"Action: {validated_advice['action']}, Confidence: {validated_advice['confidence']}")
                                logger.info(f"Reasoning: {validated_advice['reasoning']}")
                                
                                self.metrics.record(True, elapsed_time)
                                return validated_advice
                            else:
                                logger.warning(f"Invalid fields in AI response, using fallback strategy")
                                self.metrics.record(False, elapsed_time)
                                return self.generate_fallback_advice(pd.DataFrame(candle_data['candles']))
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON: {e}")
                            self.metrics.record(False, elapsed_time)
                            return self.generate_fallback_advice(pd.DataFrame(candle_data['candles']))
                    else:
                        logger.warning(f"Could not extract valid JSON from response, using fallback strategy")
                        self.metrics.record(False, elapsed_time)
                        return self.generate_fallback_advice(pd.DataFrame(candle_data['candles']))
                else:
                    logger.warning(f"Ollama API error (attempt {attempt+1}/{self.max_retries}): "
                                  f"{response.status_code} - {response.text[:100]}...")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
            
            except Exception as e:
                logger.error(f"Error consulting AI (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        elapsed_time = time.time() - start_time
        self.metrics.record(False, elapsed_time)
        logger.warning(f"All attempts to consult AI for {pair} failed, using fallback strategy")
        return self.generate_fallback_advice(pd.DataFrame(candle_data['candles']))
    
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
    
    def validate_ai_response(self, response: Dict) -> Optional[Dict]:
        """Validate and sanitize AI response fields"""
        # Define default values for required fields
        defaults = {
            'action': 'no_trade',
            'leverage': 1.0,
            'take_profit': 0.02,
            'stop_loss': -0.02,
            'confidence': 0.0,
            'reasoning': 'No specific reasoning provided'
        }
        
        # Create a sanitized version of the response with defaults
        sanitized = defaults.copy()
        
        # Update with provided values
        for key, value in response.items():
            if key in defaults:
                sanitized[key] = value
        
        # Validate action field
        action = sanitized['action'].lower()
        if action not in ['open_long', 'open_short', 'no_trade']:
            action = 'no_trade'
        sanitized['action'] = action
        
        # Validate numeric fields with safety bounds
        try:
            # Leverage: between 1 and 3
            leverage = float(sanitized['leverage'])
            sanitized['leverage'] = max(1.0, min(3.0, leverage))
            
            # Take profit: between 0.5% and 5%
            take_profit = float(sanitized['take_profit'])
            sanitized['take_profit'] = max(0.005, min(0.05, take_profit))
            
            # Stop loss: between -0.5% and -3%
            stop_loss = float(sanitized['stop_loss'])
            sanitized['stop_loss'] = max(-0.03, min(-0.005, stop_loss))
            
            # Confidence: between 0 and 1
            confidence = float(sanitized['confidence'])
            sanitized['confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric values in AI response, using defaults")
            return None
            
        # Sanitize reasoning (limit length and remove special chars)
        reasoning = str(sanitized['reasoning'])
        sanitized['reasoning'] = reasoning[:100]  # Limit to 100 chars
        
        return sanitized
    
    def get_base_entry_conditions(self, dataframe: DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate baseline entry conditions using technical indicators"""
        long_conditions = (
            # RSI conditions - More lenient
            (
                (dataframe['rsi'] < self.rsi_oversold * 1.15) |  # Primary RSI condition
                (
                    (dataframe['rsi'] < 45) &  # Secondary RSI condition
                    (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI rising
                    (dataframe['close'] < dataframe['bb_lowerband'] * 1.05)  # Price near BB
                )
            ) &
            # MACD improving
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1)) &
            
            # Volume filter - More lenient
            (dataframe['volume'] > dataframe['volume'].rolling(3).mean() * 0.8)
        )
        
        short_conditions = (
            # RSI conditions - More lenient
            (
                (dataframe['rsi'] > self.rsi_overbought * 0.85) |  # Primary RSI condition
                (
                    (dataframe['rsi'] > 55) &  # Secondary RSI condition
                    (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI falling
                    (dataframe['close'] > dataframe['bb_upperband'] * 0.95)  # Price near BB
                )
            ) &
            # MACD deteriorating
            (dataframe['macdhist'] < dataframe['macdhist'].shift(1)) &
            
            # Volume filter - More lenient
            (dataframe['volume'] > dataframe['volume'].rolling(3).mean() * 0.8)
        )
        
        return long_conditions, short_conditions
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate entry signals using AI advice combined with technical conditions"""
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Skip if we don't have enough data
        if len(dataframe) <= self.lookback_candles:
            return dataframe
            
        # Get basic technical indicator conditions
        long_conditions, short_conditions = self.get_base_entry_conditions(dataframe)
        
        # Get AI trading advice
        ai_advice = self.get_trading_advice(dataframe, metadata['pair'])
        last_index = dataframe.index[-1]
        
        if ai_advice:
            # Store AI recommendations
            dataframe.loc[last_index, 'ai_action'] = ai_advice.get('action', 'no_trade')
            dataframe.loc[last_index, 'ai_stop_loss'] = float(ai_advice.get('stop_loss', self.stoploss))
            dataframe.loc[last_index, 'ai_take_profit'] = float(ai_advice.get('take_profit', 0.02))
            dataframe.loc[last_index, 'ai_leverage'] = float(ai_advice.get('leverage', 1.0))
            dataframe.loc[last_index, 'ai_confidence'] = float(ai_advice.get('confidence', 0.0))
            dataframe.loc[last_index, 'ai_reasoning'] = ai_advice.get('reasoning', '')
            
            # Log detailed analysis for debugging
            logger.info(f"Analysis for {metadata['pair']}:")
            logger.info(f"AI Action: {ai_advice.get('action', 'no_trade')}")
            logger.info(f"AI Confidence: {ai_advice.get('confidence', 0.0)}")
            logger.info(f"Technical Long Conditions: {long_conditions.iloc[-1]}")
            logger.info(f"Technical Short Conditions: {short_conditions.iloc[-1]}")
            logger.info(f"Current RSI: {dataframe['rsi'].iloc[-1]:.2f}")
            logger.info(f"Current MACD Hist: {dataframe['macdhist'].iloc[-1]:.6f}")
            logger.info(f"Current BB Position: {((dataframe['close'].iloc[-1] - dataframe['bb_middleband'].iloc[-1]) / (dataframe['bb_upperband'].iloc[-1] - dataframe['bb_middleband'].iloc[-1])):.2f}")
            
            # Apply AI advice if confidence meets our threshold
            if ai_advice.get('confidence', 0.0) >= self.min_ai_confidence:
                action = ai_advice.get('action', 'no_trade')
                
                if action == 'open_long':
                    # Use either AI confidence or technical conditions
                    dataframe.loc[last_index, 'enter_long'] = 1 if (long_conditions.iloc[-1] or ai_advice.get('confidence', 0.0) > 0.8) else 0
                    if dataframe.loc[last_index, 'enter_long'] == 1:
                        logger.info(f"Generated LONG signal for {metadata['pair']} with confidence {ai_advice.get('confidence', 0.0)}")
                    
                elif action == 'open_short':
                    # Use either AI confidence or technical conditions
                    dataframe.loc[last_index, 'enter_short'] = 1 if (short_conditions.iloc[-1] or ai_advice.get('confidence', 0.0) > 0.8) else 0
                    if dataframe.loc[last_index, 'enter_short'] == 1:
                        logger.info(f"Generated SHORT signal for {metadata['pair']} with confidence {ai_advice.get('confidence', 0.0)}")
            else:
                # Use technical conditions only if AI confidence is low
                dataframe.loc[last_index, 'enter_long'] = 1 if long_conditions.iloc[-1] else 0
                dataframe.loc[last_index, 'enter_short'] = 1 if short_conditions.iloc[-1] else 0
                if dataframe.loc[last_index, 'enter_long'] or dataframe.loc[last_index, 'enter_short']:
                    logger.info(f"Generated signal based on technical analysis for {metadata['pair']}")
                
        else:
            logger.warning(f"No AI advice available for {metadata['pair']}, using technical analysis")
            # Use technical signals as fallback
            dataframe.loc[last_index, 'enter_long'] = 1 if long_conditions.iloc[-1] else 0
            dataframe.loc[last_index, 'enter_short'] = 1 if short_conditions.iloc[-1] else 0
            
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