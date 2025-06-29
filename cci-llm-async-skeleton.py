#!/usr/bin/env python3
"""
Concurrent CCI-LLM API Request Script

This script demonstrates how to make multiple concurrent requests to CCI-LLM
using asyncio and the AsyncOpenAI client.

Usage:
    1. Modify the configuration section below
    2. Replace SAMPLE_PROMPTS with your own prompts/questions
    3. Run: python cci-llm-async-skeleton.py


Requirements:
    pip install openai httpx asyncio
"""

import asyncio
from openai import AsyncOpenAI
from openai import APITimeoutError, APIConnectionError, APIError
import httpx
import time
from typing import List, Dict, Any

# ================================
# CONFIGURATION - MODIFY AS NEEDED
# ================================

# API Configuration
BASE_URL = "https://cci-llm.charlotte.edu/api/v1"  # Change to your API endpoint
API_KEY = "dummy-key"  # Change to your API key

# Concurrency Settings
MAX_CONCURRENT = 200  # Maximum number of simultaneous requests (start with 2-5, adjust based on how many requests the server can handle without timing out)

# Request Settings
MAX_TOKENS = 4000  # Maximum tokens per response
TEMPERATURE = 0.7  # Response creativity (0.0 = deterministic, 1.0 = creative)
REQUEST_PROMPT_PREFIX = "Provide a detailed answer (max 4000 words): "  # Prefix added to each prompt

# Timeout and Retry Settings
REQUEST_TIMEOUT = 300.0  # Timeout for individual requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for timeout errors
RETRY_DELAY = 30.0 # Delay between retries in seconds

# ========================================
# YOUR PROMPTS - REPLACE WITH YOUR OWN DATA
# ========================================

# Sample prompts/questions to process (replace with your own)
SAMPLE_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What are the benefits of renewable energy sources?",
    "Describe the process of photosynthesis.",
    "How does blockchain technology work?",
    "What is the significance of quantum computing?",
    "Explain the theory of relativity.",
    "What are the main causes of climate change?",
    "How do neural networks function?",
    "What is the importance of biodiversity?",
    "Describe the water cycle process.",
] * 20

# Alternative: Load prompts from a file
# def load_prompts_from_file(filename: str) -> List[str]:
#     """Load prompts from a text file, one per line."""
#     with open(filename, 'r', encoding='utf-8') as f:
#         return [line.strip() for line in f if line.strip()]
# 
# SAMPLE_PROMPTS = load_prompts_from_file('prompts.txt')

# ==========================================
# CORE FUNCTIONS - CUSTOMIZE IF NEEDED
# ==========================================

async def process_single_request(client: AsyncOpenAI, prompt: str, model_id: str, item_id: int) -> Dict[str, Any]:
    """
    Process a single prompt through the API with timeout handling and retry logic.
    
    Args:
        client: AsyncOpenAI client instance
        prompt: The prompt/question to send
        model_id: Model identifier from the API
        item_id: Unique identifier for this request (for tracking)
    
    Returns:
        Dictionary with request results and metadata
    """
    print(f"  Request {item_id}: Processing...")
    
    # Create the full prompt with prefix
    full_prompt = f"{REQUEST_PROMPT_PREFIX}{prompt}"
    
    # Attempt the request with retry logic for timeout errors
    # This loop implements a retry mechanism with exponential backoff for recoverable errors
    last_error = None
    for attempt in range(MAX_RETRIES + 1):  # +1 for initial attempt
        try:
            # Make the API request with asyncio timeout wrapper
            # asyncio.wait_for() adds an additional timeout layer on top of the OpenAI client timeout
            # This provides double protection against hanging requests
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                ),
                timeout=REQUEST_TIMEOUT
            )
            
            reply = response.choices[0].message.content.strip()
            
            if attempt > 0:
                print(f"  Request {item_id}: Complete (succeeded on attempt {attempt + 1})")
            else:
                print(f"  Request {item_id}: Complete")
            
            return {
                "item_id": item_id,
                "prompt": prompt,
                "response": reply,
                "status": "success",
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "attempts": attempt + 1
            }
            
        except (APITimeoutError, asyncio.TimeoutError) as e:
            # TIMEOUT ERRORS - These are recoverable and should be retried
            # 
            # APITimeoutError: OpenAI client timeout (server took too long to respond)
            # asyncio.TimeoutError: Our asyncio.wait_for() timeout (request exceeded REQUEST_TIMEOUT)
            #
            # Why we retry: Timeouts are often temporary network issues or server load
            # The request itself was valid, so retrying often succeeds
            #
            # Customization tips:
            # - Increase REQUEST_TIMEOUT for longer-running requests
            # - Increase RETRY_DELAY for heavily loaded servers
            # - Decrease MAX_RETRIES if you need faster failure detection
            
            last_error = e
            error_type = "API timeout" if isinstance(e, APITimeoutError) else "Request timeout"
            
            if attempt < MAX_RETRIES:
                print(f"  Request {item_id}: {error_type} on attempt {attempt + 1}, retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)  # Wait before retrying to avoid overwhelming the server
                continue
            else:
                print(f"  Request {item_id}: Failed after {MAX_RETRIES + 1} attempts - {error_type}")
                return {
                    "item_id": item_id,
                    "prompt": prompt,
                    "response": None,
                    "status": "timeout_error",
                    "error": f"{error_type} after {MAX_RETRIES + 1} attempts: {str(e)}",
                    "attempts": attempt + 1
                }
                
        except APIConnectionError as e:
            # CONNECTION ERRORS - These are usually recoverable and should be retried
            #
            # APIConnectionError: Network connection issues (DNS, network unreachable, etc.)
            # Common causes: Network hiccups, proxy issues, server temporarily unavailable
            #
            # Why we retry: Connection issues are often temporary
            # The server might be restarting, network might be congested, etc.
            #
            # Customization tips:
            # - For unreliable networks: Increase MAX_RETRIES and RETRY_DELAY
            # - For stable networks: You might want to fail faster with fewer retries
            # - Consider exponential backoff: RETRY_DELAY * (attempt + 1)
            
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  Request {item_id}: Connection error on attempt {attempt + 1}, retrying in {RETRY_DELAY}s...")
                # Simple fixed delay - you can implement exponential backoff like this:
                # delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff  
                # await asyncio.sleep(min(delay, 60))   # Cap at 60 seconds
                await asyncio.sleep(RETRY_DELAY)  # Give network time to recover
                continue
            else:
                print(f"  Request {item_id}: Connection failed after {MAX_RETRIES + 1} attempts")
                return {
                    "item_id": item_id,
                    "prompt": prompt,
                    "response": None,
                    "status": "connection_error",
                    "error": f"Connection error after {MAX_RETRIES + 1} attempts: {str(e)}",
                    "attempts": attempt + 1
                }
                
        except APIError as e:
            # Don't retry on API errors (like rate limits, invalid requests, etc.)
            print(f"  Request {item_id}: API Error - {e}")
            return {
                "item_id": item_id,
                "prompt": prompt,
                "response": None,
                "status": "api_error",
                "error": f"API error: {str(e)}",
                "attempts": attempt + 1
            }
            
        except Exception as e:
            # For other unexpected errors, don't retry
            print(f"  Request {item_id}: Unexpected error - {e}")
            return {
                "item_id": item_id,
                "prompt": prompt,
                "response": None,
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
                "attempts": attempt + 1
            }
    
    # This should never be reached, but just in case
    return {
        "item_id": item_id,
        "prompt": prompt,
        "response": None,
        "status": "error",
        "error": f"Maximum retries exceeded: {str(last_error)}",
        "attempts": MAX_RETRIES + 1
    }

async def process_all_prompts(client: AsyncOpenAI, prompts: List[str], model_id: str) -> List[Dict[str, Any]]:
    """
    Process all prompts concurrently with controlled concurrency.
    
    This function uses asyncio.Semaphore to limit the number of concurrent requests,
    preventing the API from being overwhelmed while still processing multiple requests in parallel.
    
    Args:
        client: AsyncOpenAI client instance
        prompts: List of prompts to process
        model_id: Model identifier from the API
    
    Returns:
        List of dictionaries containing results for each prompt
    """
    print(f"Processing {len(prompts)} prompts with max {MAX_CONCURRENT} concurrent requests...")
    
    # Create a semaphore to limit concurrent requests
    # This ensures we don't overwhelm the API server
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async def process_with_concurrency_limit(prompt: str, item_id: int):
        """Wrapper function to apply concurrency limits to each request."""
        async with semaphore:
            return await process_single_request(client, prompt, model_id, item_id)
    
    # Create tasks for all prompts
    tasks = [
        process_with_concurrency_limit(prompt, i) 
        for i, prompt in enumerate(prompts, 1)
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    print(f"All {len(prompts)} prompts processed!")
    return results

async def run_concurrent_requests():
    """
    Main function to run concurrent API requests.
    
    This function:
    1. Connects to the API and gets available models
    2. Processes all prompts concurrently
    3. Provides detailed results and performance metrics
    """
    print("Starting Concurrent API Request Processing...")
    print(f"Configuration:")
    print(f"  - API Endpoint: {BASE_URL}")
    print(f"  - Max Concurrent Requests: {MAX_CONCURRENT}")
    print(f"  - Max Tokens per Response: {MAX_TOKENS}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Request Timeout: {REQUEST_TIMEOUT}s")
    print(f"  - Max Retries: {MAX_RETRIES}")
    print(f"  - Retry Delay: {RETRY_DELAY}s")
    
    # Initialize async client with timeout configuration
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=REQUEST_TIMEOUT,  # Set default timeout for API calls
        http_client=httpx.AsyncClient(
            verify=False,  # Disable SSL verification
            timeout=httpx.Timeout(REQUEST_TIMEOUT)  # Set timeout for HTTP client
        )
    )
    
    try:
        # Step 1: Get available models
        # This initial call tests API connectivity and gets available models
        print("\nConnecting to API and getting models...", end=" ")
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"Using model: {model_id}")
        
        # Step 2: Process all prompts concurrently
        print(f"\nStarting concurrent processing of {len(SAMPLE_PROMPTS)} prompts...")
        
        start_time = time.time()
        results = await process_all_prompts(client, SAMPLE_PROMPTS, model_id)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Step 3: Generate results summary
        successful = sum(1 for r in results if r['status'] == 'success')
        timeout_errors = sum(1 for r in results if r['status'] == 'timeout_error')
        connection_errors = sum(1 for r in results if r['status'] == 'connection_error')
        api_errors = sum(1 for r in results if r['status'] == 'api_error')
        other_errors = sum(1 for r in results if r['status'] == 'error')
        failed = len(results) - successful
        total_tokens = sum(r.get('tokens_used', 0) for r in results if r.get('tokens_used'))
        total_attempts = sum(r.get('attempts', 1) for r in results)
        retried_requests = sum(1 for r in results if r.get('attempts', 1) > 1)
        
        print(f"\nPROCESSING COMPLETE!")
        print(f"=" * 60)
        print(f"Total prompts processed: {len(results)}")
        print(f"Successful requests: {successful}")
        print(f"Failed requests: {failed}")
        if timeout_errors > 0:
            print(f"  - Timeout errors: {timeout_errors}")
        if connection_errors > 0:
            print(f"  - Connection errors: {connection_errors}")
        if api_errors > 0:
            print(f"  - API errors: {api_errors}")
        if other_errors > 0:
            print(f"  - Other errors: {other_errors}")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Average time per request: {processing_time/len(results):.2f} seconds")
        if retried_requests > 0:
            print(f"Requests that required retries: {retried_requests}")
            print(f"Total API attempts made: {total_attempts}")
        if total_tokens > 0:
            print(f"Total tokens used: {total_tokens}")
        
        # Step 4: Show sample results
        if successful > 0:
            print(f"\nSAMPLE SUCCESSFUL RESULTS:")
            print("-" * 60)
            successful_results = [r for r in results if r['status'] == 'success']
            for i, result in enumerate(successful_results[:2], 1):  # Show first 2 successful results
                print(f"Request {result['item_id']}:")
                print(f"  Prompt: {result['prompt'][:80]}...")
                print(f"  Response: {result['response'][:150]}...")
                if result.get('tokens_used'):
                    print(f"  Tokens used: {result['tokens_used']}")
                print()
        
        # Step 5: Show any errors
        if failed > 0:
            print(f"ERRORS ENCOUNTERED:")
            print("-" * 60)
            failed_results = [r for r in results if r['status'] != 'success']
            for result in failed_results[:3]:  # Show first 3 errors
                error_type = result['status'].replace('_', ' ').title()
                attempts_info = f" (after {result.get('attempts', 1)} attempts)" if result.get('attempts', 1) > 1 else ""
                print(f"Request {result['item_id']} - {error_type}{attempts_info}: {result['error']}")
            if len(failed_results) > 3:
                print(f"... and {len(failed_results) - 3} more errors")
            print()
        
        return results
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        return None
    
    finally:
        # Always close the client connection
        await client.close()

# ============================
# EXAMPLE USAGE FUNCTIONS
# ============================

def save_results_to_file(results: List[Dict[str, Any]], filename: str = "api_results.txt"):
    """
    Save results to a text file for later analysis.
    
    Args:
        results: List of result dictionaries from process_all_prompts
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("API Request Results\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"Request {result['item_id']}:\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Attempts: {result.get('attempts', 1)}\n")
            
            if result['status'] == 'success':
                f.write(f"Response: {result['response']}\n")
                if result.get('tokens_used'):
                    f.write(f"Tokens used: {result['tokens_used']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            
            f.write("-" * 30 + "\n\n")
    
    print(f"Results saved to {filename}")

def main():
    """
    Main entry point for the script.
    
    This function runs the concurrent request processing and handles any errors.
    You can modify this function to customize the behavior.
    """
    print("Concurrent OpenAI API Request Starter Script")
    print("=" * 50)
    
    try:
        # Run the concurrent processing
        results = asyncio.run(run_concurrent_requests())
        
        if results:
            print(f"\nScript completed successfully!")
            
            # Optional: Save results to file
            # save_results_to_file(results, "my_api_results.txt")
            
            return results
        else:
            print(f"\nScript failed to complete.")
            return None
            
    except KeyboardInterrupt:
        print(f"\nScript interrupted by user (Ctrl+C)")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None

# ============================
# SCRIPT EXECUTION
# ============================

if __name__ == "__main__":
    # Run the script
    main()
    
    # Example of how to use this script in your own code:
    # 
    # import asyncio
    # from your_script import process_all_prompts, AsyncOpenAI
    # 
    # async def my_custom_function():
    #     my_prompts = ["Question 1", "Question 2", "Question 3"]
    #     client = AsyncOpenAI(base_url="your-api-url", api_key="your-key")
    #     results = await process_all_prompts(client, my_prompts, "model-id")
    #     await client.close()
    #     return results
    # 
    # results = asyncio.run(my_custom_function())