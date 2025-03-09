import requests
import json

def test_ai_connection():
    print("Testing AI connection...")
    
    # Test prompt
    prompt = """
    You are a professional cryptocurrency trading analyst. Analyze the following market data and provide a trading recommendation.
    
    Current Price: 50000
    RSI: 45.5
    ADX: 30.2
    Volume Z-Score: 1.5
    
    Price Change (last 5 candles):
    0.5%
    -0.3%
    1.2%
    -0.8%
    0.4%
    
    Volume Change (last 5 candles):
    20%
    -10%
    15%
    -5%
    25%
    
    Please provide your analysis in the following format:
    SENTIMENT: [bullish/bearish/neutral]
    CONFIDENCE: [0-100]%
    ACTION: [buy/sell/hold]
    EXPLANATION: [Your detailed analysis]
    """
    
    try:
        # Call Ollama API
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'deepseek-r1:14b',
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nAI Response:")
            print(result['response'])
            print("\nAI connection successful!")
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error connecting to AI: {str(e)}")

if __name__ == "__main__":
    test_ai_connection() 