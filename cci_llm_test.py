#!/usr/bin/env python3
"""
cci-llm-test.py - Concise test for CCI LLM API
Usage: python cci-llm-test.py
"""

from openai import OpenAI
import httpx

# Configuration
BASE_URL = "https://cci-llm.charlotte.edu/api/v1"
API_KEY = "dummy-key"

def test_api():
    """Test CCI LLM API endpoints."""
    print("🚀 Testing CCI LLM API...")
    
    # Initialize client with SSL verification disabled
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(verify=False)
    )
    
    try:
        # Test 1: Get models
        print("📋 Getting models...", end=" ")
        models = client.models.list()
        model_id = models.data[0].id
        print(f"Found: {model_id}")
        
        # Test 2: Simple chat completion
        print("💬 Testing chat...", end=" ")
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say 'Hello from CCI LLM!' in exactly 5 words."}],
            max_tokens=100,
            temperature=0.1
        )
        
        reply = response.choices[0].message.content.strip()
        print(f"Reply: {reply}")
        
        
        print(f"\n🎉 API working at {BASE_URL}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_api() 