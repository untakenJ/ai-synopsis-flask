from dotenv import load_dotenv
import asyncio
import os
import json
import re
import aiohttp
import httpx
import time
import threading
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from flask import Flask, jsonify, request

load_dotenv() 

app = Flask(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DIFY_API_URL = 'https://api.dify.ai/v1/chat-messages'
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

# Cache configuration
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "analyze_cache.json")
CACHE_DURATION_HOURS = 24

# Background refresh state - file-based for multi-process support
refresh_lock = threading.Lock()
REFRESH_STATUS_FILE = os.path.join(CACHE_DIR, "refresh_status.json")

def save_refresh_status(in_progress, start_time=None):
    """Save refresh status to file for multi-process access"""
    ensure_cache_dir()
    status_data = {
        "refresh_in_progress": in_progress,
        "start_time": start_time.isoformat() if start_time else None,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(REFRESH_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving refresh status: {e}")

def load_refresh_status():
    """Load refresh status from file"""
    if not os.path.exists(REFRESH_STATUS_FILE):
        return False, None
    
    try:
        with open(REFRESH_STATUS_FILE, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            in_progress = status_data.get("refresh_in_progress", False)
            start_time_str = status_data.get("start_time")
            start_time = datetime.fromisoformat(start_time_str) if start_time_str else None
            return in_progress, start_time
    except Exception as e:
        print(f"Error loading refresh status: {e}")
        return False, None

def cleanup_refresh_status():
    """Clean up refresh status file"""
    try:
        if os.path.exists(REFRESH_STATUS_FILE):
            os.remove(REFRESH_STATUS_FILE)
    except Exception as e:
        print(f"Error cleaning up refresh status: {e}")

query_prefix = '''
    For the news story title, verify whether it happened today or not. 
    "Happened today" means the news story happened in the last 24 hours. 
    News happened earlier than 24 hours ago but come into public on multiple reliable webisites within 24 hours are also considered as "happened today".
    News stories metioned in articles older than 24 hours are NOT considered as "happened today".
    If yes, search it on web and find a link to the original news article from a reliable news website. 
    Also make a summary of about 5 sentences of the news article. 
    Return a JSON-like object with fields "title", "summary", "source_link" and "happened_today" (yes or no). 
    News title to verify:
'''

target_urls = ["https://tmz.com", "https://bbc.com"]
max_titles_per_url = 20

def ensure_cache_dir():
    """Ensure cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def load_cache():
    """Load cache from file - thread-safe version"""
    ensure_cache_dir()
    if os.path.exists(CACHE_FILE):
        try:
            # Use file locking to prevent read/write conflicts
            import fcntl
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                try:
                    cache_data = json.load(f)
                    return cache_data
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
        except (json.JSONDecodeError, FileNotFoundError, ImportError):
            # Fallback for systems without fcntl (like Windows)
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    return cache_data
            except (json.JSONDecodeError, FileNotFoundError):
                return None
    return None

def save_cache(data, execution_time):
    """Save cache to file - thread-safe version"""
    ensure_cache_dir()
    cache_data = {
        "data": data,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat(),
        "cache_duration_hours": CACHE_DURATION_HOURS
    }
    try:
        # Use file locking to prevent read/write conflicts
        import fcntl
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
            try:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                print(f"Cache saved with {len(data)} results, execution time: {execution_time:.2f}s")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
    except (ImportError, Exception) as e:
        # Fallback for systems without fcntl (like Windows)
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"Cache saved with {len(data)} results, execution time: {execution_time:.2f}s")
        except Exception as e2:
            print(f"Error saving cache: {e2}")

def is_cache_valid(cache_data):
    """Check if cache is still valid (within 24 hours)"""
    if not cache_data:
        return False
    
    try:
        cache_timestamp = datetime.fromisoformat(cache_data.get("timestamp", ""))
        current_time = datetime.now()
        time_diff = current_time - cache_timestamp
        
        # Check if cache is within the specified duration
        return time_diff.total_seconds() < (CACHE_DURATION_HOURS * 3600)
    except Exception as e:
        print(f"Error checking cache validity: {e}")
        return False

def get_cache_info(cache_data):
    """Get cache information for debugging"""
    if not cache_data:
        return "No cache found"
    
    try:
        cache_timestamp = datetime.fromisoformat(cache_data.get("timestamp", ""))
        current_time = datetime.now()
        time_diff = current_time - cache_timestamp
        hours_remaining = (CACHE_DURATION_HOURS * 3600 - time_diff.total_seconds()) / 3600
        
        return {
            "cache_age_hours": time_diff.total_seconds() / 3600,
            "hours_remaining": max(0, hours_remaining),
            "execution_time": cache_data.get("execution_time", 0),
            "results_count": len(cache_data.get("data", [])),
            "is_valid": is_cache_valid(cache_data)
        }
    except Exception as e:
        return f"Error getting cache info: {e}"

def get_cache_info_safe():
    """Get cache information without interfering with refresh operations"""
    cache_data = load_cache()
    if not cache_data:
        return "No cache found"
    
    try:
        cache_timestamp = datetime.fromisoformat(cache_data.get("timestamp", ""))
        current_time = datetime.now()
        time_diff = current_time - cache_timestamp
        hours_remaining = (CACHE_DURATION_HOURS * 3600 - time_diff.total_seconds()) / 3600
        
        return {
            "cache_age_hours": time_diff.total_seconds() / 3600,
            "hours_remaining": max(0, hours_remaining),
            "execution_time": cache_data.get("execution_time", 0),
            "results_count": len(cache_data.get("data", [])),
            "is_valid": is_cache_valid(cache_data)
        }
    except Exception as e:
        return f"Error getting cache info: {e}"

def is_cache_valid_safe(cache_data):
    """Check if cache is still valid without interfering with refresh operations"""
    if not cache_data:
        return False
    
    try:
        cache_timestamp = datetime.fromisoformat(cache_data.get("timestamp", ""))
        current_time = datetime.now()
        time_diff = current_time - cache_timestamp
        
        # Check if cache is within the specified duration
        return time_diff.total_seconds() < (CACHE_DURATION_HOURS * 3600)
    except Exception as e:
        print(f"Error checking cache validity: {e}")
        return False

def get_refresh_status():
    """Get current refresh status - file-based version"""
    in_progress, start_time = load_refresh_status()
    
    if in_progress and start_time:
        elapsed = time.time() - start_time.timestamp()
        return {
            "refresh_in_progress": True,
            "elapsed_seconds": elapsed,
            "start_time": start_time.isoformat()
        }
    else:
        return {
            "refresh_in_progress": False,
            "elapsed_seconds": 0,
            "start_time": None
        }

def run_analysis_sync():
    """Synchronous wrapper for analyze_all() to be used in background threads"""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(analyze_all())
        finally:
            loop.close()
    except Exception as e:
        print(f"Error in background analysis: {e}")
        return []

def background_refresh_cache():
    """Background function to refresh cache"""
    print(f"[DEBUG] background_refresh_cache started, thread: {threading.current_thread().name}")
    
    # Set refresh status to started
    start_time = datetime.now()
    save_refresh_status(True, start_time)
    print(f"[DEBUG] Refresh status set to started, thread: {threading.current_thread().name}")
    
    try:
        print("Starting background cache refresh...")
        
        # Create a new event loop for this thread
        print(f"[DEBUG] Creating new event loop, thread: {threading.current_thread().name}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Perform fresh analysis using analyze_all
            print(f"[DEBUG] Starting analyze_all, thread: {threading.current_thread().name}")
            start_time_analysis = time.time()
            results = loop.run_until_complete(analyze_all())
            execution_time = time.time() - start_time_analysis
            
            print(f"[DEBUG] analyze_all completed with {len(results)} results, execution_time: {execution_time:.2f}s")
            
            # Save to cache only if we got some results
            if results:
                print(f"[DEBUG] Saving cache with {len(results)} results, thread: {threading.current_thread().name}")
                save_cache(results, execution_time)
                print(f"Background refresh completed with {len(results)} results")
            else:
                print("Background refresh: No results obtained - not saving to cache")
                
        finally:
            loop.close()
            print(f"[DEBUG] Event loop closed, thread: {threading.current_thread().name}")
            
    except Exception as e:
        print(f"Background refresh failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Set refresh status to completed
        save_refresh_status(False, None)
        print(f"[DEBUG] Refresh status set to completed, thread: {threading.current_thread().name}")
        print("Background refresh finished")

def is_json_object_or_array(s: str) -> bool:
    try:
        obj = json.loads(s)
    except ValueError:
        return False
    return isinstance(obj, (dict, list))

def extract_first_list(text):
    m = re.search(r'(\[.*?\])', text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON list found")
    return json.loads(m.group(1))

async def fetch_raw_html(url):
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            body = soup.body
            return url, body.get_text(separator='\n', strip=True) if body else None
    except Exception as e:
        return url, None

async def analyze_raw_html_with_deepseek(url, html_content):
    prompt = f"""
Analyze the following <body> of the raw HTML content fetched from {url}.
Extract all news stories on the page and return as an array of titles.

[BEGIN HTML]
{html_content[:60000]}
[END HTML]
"""
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
        resp = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return url, resp.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek API error for {url}: {e}")
        # Return a simple fallback response when API fails
        return url, '["API temporarily unavailable - please try again later"]'

def extract_news_fallback(html_content, url):
    """Fallback function to extract basic news titles when AI is unavailable"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        
        # Common selectors for news headlines
        selectors = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            '[class*="headline"]', '[class*="title"]', '[class*="news"]',
            'article h1', 'article h2', 'article h3',
            '.headline', '.title', '.news-title'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) > 10 and len(text) < 200:  # Reasonable headline length
                    news_items.append(text)
        
        # Remove duplicates and limit to first 5 items
        unique_items = list(dict.fromkeys(news_items))[:5]
        
        return unique_items if unique_items else ["No news items found"]
        
    except Exception as e:
        print(f"Fallback parser error: {e}")
        return ["Error parsing HTML"]

async def call_verification_agent(title, max_retries=3):
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "inputs": {},
        "query": f"{query_prefix} {title}",
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "api-request",
    }

    attempt = 0
    last = None  # Keep track of last valid answer only once globally

    while attempt < max_retries:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
                async with session.post(DIFY_API_URL, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        print(f"Dify API error for {title} (attempt {attempt + 1}): HTTP {resp.status}")
                        attempt += 1
                        await asyncio.sleep(1)
                        continue

                    try:
                        async for line_bytes in resp.content:
                            try:
                                line = line_bytes.decode("utf-8").strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                data = json.loads(line[len("data:"):].strip())
                                ans = data.get("answer")
                                if isinstance(ans, str) and len(ans) > 2 and is_json_object_or_array(ans):
                                    last = ans
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"Error parsing line for {title} (attempt {attempt + 1}): {e}")
                                continue
                    except aiohttp.ClientPayloadError as e:
                        print(f"Streaming error for {title} (attempt {attempt + 1}): {e}")

                    if last:
                        print(f"Successfully verified {title} on attempt {attempt + 1}")
                        return title, last
                    else:
                        print(f"Verification failed for {title} (attempt {attempt + 1})")
                        attempt += 1
                        await asyncio.sleep(1)

        except Exception as e:
            print(f"Exception in call_verification_agent for {title} (attempt {attempt + 1}): {e}")
            attempt += 1
            await asyncio.sleep(1)

    print(f"All {max_retries} attempts failed for {title}")
    return title, None


async def analyze_all():
    print("Starting analyze_all...")
    
    # Ensure unique URLs
    unique_urls = list(set(target_urls))
    print("Fetching HTML from URLs:", unique_urls)
    html_results = await asyncio.gather(*(fetch_raw_html(u) for u in unique_urls))
    html_map = dict(html_results)
    print("HTML fetch results:", {url: "Success" if html else "Failed" for url, html in html_map.items()})

    tasks = [
        analyze_raw_html_with_deepseek(url, html)
        for url, html in html_map.items() if html
    ]
    print(f"Starting DeepSeek analysis for {len(tasks)} URLs...")
    deepseek_results = await asyncio.gather(*tasks)

    final = []
    processed_urls = set()  # Track processed URLs to prevent duplicates
    
    for url, result in deepseek_results:
        if url in processed_urls:
            print(f"WARNING: URL {url} already processed, skipping...")
            continue
        processed_urls.add(url)
        
        print(f"DeepSeek result for {url}: {'Success' if result else 'Failed'}")
        if not result:
            continue
        try:
            titles = extract_first_list(result)
            print(f"Extracted {len(titles)} titles from {url}")
        except Exception as e:
            print(f"Failed to extract titles from {url}: {e}")
            # Use fallback parser when AI extraction fails
            if url in html_map and html_map[url]:
                print(f"Using fallback parser for {url}")
                titles = extract_news_fallback(html_map[url], url)
            else:
                continue
        to_verify = titles[:max_titles_per_url]  # limit to first 2
        print(f"Verifying {len(to_verify)} titles from {url}")
        ver_tasks = [call_verification_agent(t) for t in to_verify]
        ver_results = await asyncio.gather(*ver_tasks)
        for title, content in ver_results:
            try:
                obj = json.loads(content) if content else None
                if obj:
                    final.append(obj)
                    print(f"Successfully verified: {title}")
                else:
                    print(f"Failed to verify after all retries: {title}")
            except Exception as e:
                print(f"Failed to parse verification result for {title}: {e}")
    
    print(f"Final results count: {len(final)}")
    return final

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """
    POST /analyze
    Triggers news extraction and verification.
    Returns cached results if available, or fresh results if cache is invalid.
    
    Query parameters:
    - force_refresh: Set to 'true' to bypass cache and force fresh analysis
    """
    # Check for force refresh parameter
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
    
    if force_refresh:
        print("Force refresh requested - bypassing cache")
    else:
        # Try to load from cache first
        cache_data = load_cache()
        if cache_data and is_cache_valid_safe(cache_data):
            cache_info = get_cache_info_safe()
            print(f"Returning cached results: {cache_info}")
            return jsonify({
                "data": cache_data["data"],
                "cached": True,
                "cache_info": cache_info,
                "execution_time": cache_data["execution_time"]
            })
        else:
            print("Cache invalid or expired - performing fresh analysis")
    
    # Perform fresh analysis
    start_time = time.time()
    try:
        results = asyncio.run(analyze_all())
        execution_time = time.time() - start_time
        
        # Save to cache only if we got some results
        if results:
            save_cache(results, execution_time)
        else:
            print("No results obtained - not saving to cache")
        
        return jsonify({
            "data": results,
            "cached": False,
            "execution_time": execution_time,
            "message": f"Analysis completed with {len(results)} results"
        })
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Error during analysis: {e}")
        return jsonify({
            "data": [],
            "cached": False,
            "execution_time": execution_time,
            "error": str(e),
            "message": "Analysis failed - please check API keys and try again"
        }), 500

@app.route('/analyze-force', methods=['POST'])
def analyze_force_endpoint():
    """
    POST /analyze-force
    Forces a fresh analysis, bypassing cache completely.
    Same as /analyze?force_refresh=true but more explicit.
    """
    print("Force refresh requested via /analyze-force endpoint")
    
    # Perform fresh analysis
    start_time = time.time()
    try:
        results = asyncio.run(analyze_all())
        execution_time = time.time() - start_time
        
        # Save to cache only if we got some results
        if results:
            save_cache(results, execution_time)
        else:
            print("No results obtained - not saving to cache")
        
        return jsonify({
            "data": results,
            "cached": False,
            "execution_time": execution_time,
            "message": f"Fresh analysis completed with {len(results)} results"
        })
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Error during force analysis: {e}")
        return jsonify({
            "data": [],
            "cached": False,
            "execution_time": execution_time,
            "error": str(e),
            "message": "Force analysis failed - please check API keys and try again"
        }), 500

@app.route('/refresh-cache', methods=['POST'])
def refresh_cache_endpoint():
    """
    POST /refresh-cache
    Starts background cache refresh. Returns immediately with status.
    """
    print(f"[DEBUG] /refresh-cache called, thread: {threading.current_thread().name}")
    
    # Check if refresh is already in progress
    in_progress, _ = load_refresh_status()
    if in_progress:
        status = get_refresh_status()
        return jsonify({
            "status": "already_running",
            "message": "Cache refresh already in progress",
            "refresh_status": status
        })
    
    # Start background refresh
    print(f"[DEBUG] Starting background thread, thread: {threading.current_thread().name}")
    refresh_task = threading.Thread(target=background_refresh_cache)
    refresh_task.daemon = True
    refresh_task.start()
    print(f"[DEBUG] Background thread started: {refresh_task.name}")
    
    return jsonify({
        "status": "started",
        "message": "Background cache refresh started",
        "refresh_status": get_refresh_status()
    })

@app.route('/refresh-status', methods=['GET'])
def refresh_status_endpoint():
    """
    GET /refresh-status
    Returns the current status of background cache refresh
    """
    return jsonify({
        "refresh_status": get_refresh_status(),
        "cache_info": get_cache_info_safe()
    })

@app.route('/analyze-wait-refresh', methods=['POST'])
def analyze_wait_refresh_endpoint():
    """
    POST /analyze-wait-refresh
    Waits for background refresh to complete and returns updated cache content.
    If no refresh is in progress, returns current cache content.
    """
    import time
    
    # Check if refresh is in progress
    refresh_status = get_refresh_status()
    
    if not refresh_status["refresh_in_progress"]:
        # No refresh in progress, return current cache
        cache_data = load_cache()
        if cache_data and is_cache_valid_safe(cache_data):
            cache_info = get_cache_info_safe()
            return jsonify({
                "data": cache_data["data"],
                "cached": True,
                "cache_info": cache_info,
                "execution_time": cache_data["execution_time"],
                "message": "No refresh in progress, returning current cache"
            })
        else:
            return jsonify({
                "data": [],
                "cached": False,
                "message": "No valid cache available and no refresh in progress"
            })
    
    # Wait for refresh to complete (with timeout)
    timeout = 30000  # 500 minutes timeout
    start_wait = time.time()
    
    while refresh_status["refresh_in_progress"]:
        if time.time() - start_wait > timeout:
            return jsonify({
                "data": [],
                "cached": False,
                "error": "Timeout waiting for refresh to complete",
                "message": "Refresh is still in progress after 5 minutes"
            }), 408
        
        time.sleep(1)  # Wait 1 second before checking again
        refresh_status = get_refresh_status()
    
    # Refresh completed, return updated cache
    cache_data = load_cache()
    if cache_data and is_cache_valid_safe(cache_data):
        cache_info = get_cache_info_safe()
        return jsonify({
            "data": cache_data["data"],
            "cached": True,
            "cache_info": cache_info,
            "execution_time": cache_data["execution_time"],
            "message": "Background refresh completed, returning updated cache"
        })
    else:
        return jsonify({
            "data": [],
            "cached": False,
            "message": "Background refresh completed but no valid cache available"
        })

@app.route('/cache-info', methods=['GET'])
def cache_info_endpoint():
    """
    GET /cache-info
    Returns information about the current cache status
    """
    cache_data = load_cache()
    cache_info = get_cache_info_safe()
    
    return jsonify({
        "cache_exists": cache_data is not None,
        "cache_info": cache_info,
        "cache_file_path": CACHE_FILE,
        "cache_duration_hours": CACHE_DURATION_HOURS
    })

@app.route('/clear-cache', methods=['POST'])
def clear_cache_endpoint():
    """
    POST /clear-cache
    Clears the cache file
    """
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            return jsonify({
                "status": "success",
                "message": "Cache cleared successfully"
            })
        else:
            return jsonify({
                "status": "success", 
                "message": "No cache file found"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error clearing cache: {str(e)}"
        }), 500

@app.route('/analyze-debug', methods=['POST'])
def debug_endpoint():
    """
    POST /analyze-debug
    Returns fixed debug content in the exact same format as /analyze endpoint.
    """
    debug_content = [
        {
            "happened_today": "yes",
            "source_link": "https://www.bbc.com/news/articles/c20nn0p9xg2o",
            "summary": "Jurors in Sean 'Diddy' Combs' trial reached a partial verdict on four charges but deadlocked on the racketeering conspiracy charge, which carries a potential life sentence. After nearly 12 hours of deliberations over two days, the jury informed Judge Arun Subramanian they were unable to agree on the most serious count. The judge ordered deliberations to resume Wednesday despite the deadlock, while the partial verdicts remain undisclosed. Testimony from 34 witnesses, including ex-girlfriend Cassie Ventura and male escort Daniel Phillip, featured prominently in the trial. Combs maintains his not-guilty plea on all charges including sex trafficking and transportation for prostitution.",
            "title": "Diddy jury to keep deliberating after reaching deadlock on most serious charge"
        },
        {
            "happened_today": "no",
            "source_link": "https://www.example.com/old-news",
            "summary": "This is a sample news story that did not happen today. It's used for testing the debug endpoint to ensure the API is working correctly.",
            "title": "Sample Old News Story for Testing"
        },
        {
            "happened_today": "yes",
            "source_link": "https://www.reuters.com/breaking-news",
            "summary": "A major breakthrough in renewable energy technology was announced today. Scientists have developed a new solar panel design that increases efficiency by 25% while reducing manufacturing costs by 30%. The innovation could accelerate the global transition to clean energy and help meet climate change targets. The research team expects commercial production to begin within 18 months.",
            "title": "Breakthrough in Solar Panel Technology Announced"
        }
    ]
    
    return jsonify(debug_content)

@app.route('/api-status', methods=['GET'])
def api_status_endpoint():
    """
    GET /api-status
    Checks the status of external APIs (DeepSeek and Dify)
    """
    status = {
        "deepseek": {"status": "unknown", "error": None},
        "dify": {"status": "unknown", "error": None},
        "timestamp": datetime.now().isoformat()
    }
    
    # Test DeepSeek API
    try:
        async def test_deepseek():
            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
            resp = await client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "Say hello"}],
                stream=False
            )
            return resp.choices[0].message.content
        
        result = asyncio.run(test_deepseek())
        status["deepseek"]["status"] = "working"
        status["deepseek"]["response"] = result[:100] + "..." if len(result) > 100 else result
    except Exception as e:
        status["deepseek"]["status"] = "error"
        status["deepseek"]["error"] = str(e)
    
    # Test Dify API
    try:
        async def test_dify():
            headers = {
                'Authorization': f'Bearer {DIFY_API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                "inputs": {},
                "query": "Say hello",
                "response_mode": "streaming",
                "conversation_id": "",
                "user": "api-test",
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(DIFY_API_URL, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        return "API working"
                    else:
                        return f"HTTP {resp.status}"
        
        result = asyncio.run(test_dify())
        status["dify"]["status"] = "working"
        status["dify"]["response"] = result
    except Exception as e:
        status["dify"]["status"] = "error"
        status["dify"]["error"] = str(e)
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
