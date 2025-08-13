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
import base64
import boto3
from botocore.exceptions import ClientError


load_dotenv() 

app = Flask(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DIFY_API_URL = 'https://api.dify.ai/v1/chat-messages'
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

# Add OpenAI API for image generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# AWS S3 configuration for image storage
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BUCKET_URL = os.getenv("S3_BUCKET_URL")  # Optional: custom domain for S3 bucket

# Cache configuration
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "analyze_cache.json")
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "48"))

target_urls = os.getenv("TARGET_URLS", "https://tmz.com,https://bbc.com,https://cnn.com,https://espn.com,https://pagesix.com/").split(",")
max_titles_per_url = int(os.getenv("MAX_TITLES_PER_URL", "40"))
max_titles_overall = int(os.getenv("MAX_TITLES_OVERALL", "80"))
num_headlines = int(os.getenv("NUM_HEADLINES", "5"))

# Background refresh state - file-based for multi-process support
refresh_lock = threading.Lock()
REFRESH_STATUS_FILE = os.path.join(CACHE_DIR, "refresh_status.json")

def upload_image_to_s3(image_base64, image_key):
    """Upload base64 image to S3 and return a presigned URL that expires after one day"""
    print(f"[DEBUG] Starting S3 upload for key: {image_key}")
    
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
        print("[DEBUG] S3 credentials not configured")
        print(f"[DEBUG] AWS_ACCESS_KEY_ID: {'Set' if AWS_ACCESS_KEY_ID else 'Not set'}")
        print(f"[DEBUG] AWS_SECRET_ACCESS_KEY: {'Set' if AWS_SECRET_ACCESS_KEY else 'Not set'}")
        print(f"[DEBUG] S3_BUCKET_NAME: {S3_BUCKET_NAME}")
        return None
    
    try:
        # Decode base64 image
        print(f"[DEBUG] Decoding base64 image, length: {len(image_base64)}")
        image_bytes = base64.b64decode(image_base64)
        print(f"[DEBUG] Decoded image size: {len(image_bytes)} bytes")
        
        # Create S3 client
        print(f"[DEBUG] Creating S3 client for region: {AWS_REGION}")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            endpoint_url=S3_BUCKET_URL
        )
        
        # Upload to S3 (without public-read ACL)
        print(f"[DEBUG] Uploading to S3 bucket: {S3_BUCKET_NAME}")
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_key,
            Body=image_bytes,
            ContentType='image/png'
        )
        print(f"[DEBUG] Successfully uploaded to S3")
        
        # Generate presigned URL that expires after one day (86400 seconds)
        print(f"[DEBUG] Generating presigned URL with 24-hour expiration")
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': image_key},
            ExpiresIn=86400  # 24 hours in seconds
        )
        
        print(f"[DEBUG] Generated presigned URL: {presigned_url}")
        return presigned_url
            
    except ClientError as e:
        print(f"[DEBUG] S3 ClientError: {e}")
        return None
    except Exception as e:
        print(f"[DEBUG] Unexpected error uploading image to S3: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_image_key(title, timestamp=None):
    """Generate a unique S3 key for an image"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean title for use as filename
    clean_title = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
    clean_title = re.sub(r'\s+', '_', clean_title.strip())
    clean_title = clean_title[:50]  # Limit length
    
    return f"news_images/{timestamp}_{clean_title}.png"

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
    If yes, search it on web and find a link to one of the original news articles from a reliable news website. 
    Also make a summary of the news story. Summarize this news event in neutral English. Start with a 1–2 sentence executive summary. Then write 5-8 short paragraphs covering who/what/when/where/why/how, impact, and what’s next. Target 520 words; keep between 400–600. 
    Return a JSON-like object with fields "title", "summary", "source_link", "happened_today" (yes or no), and "category". Choose the most appropriate category from the following list:
    - World
    - Politics & Society
    - Business & Economy
    - Technology & Science
    - Health & Environment
    - Entertainment & Culture
    - Sports
    - Other
    News title to verify:
'''


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

    # Collect all titles from all URLs
    all_titles = []
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
            # Add titles from this URL to the collection
            all_titles.extend(titles[:max_titles_per_url])  # Limit per URL
        except Exception as e:
            print(f"Failed to extract titles from {url}: {e}")
            # Use fallback parser when AI extraction fails
            if url in html_map and html_map[url]:
                print(f"Using fallback parser for {url}")
                titles = extract_news_fallback(html_map[url], url)
                all_titles.extend(titles[:max_titles_per_url])  # Limit per URL
            else:
                continue
    
    print(f"Total titles collected from all URLs: {len(all_titles)}")
    
    # Deduplicate all collected titles
    if all_titles:
        print("Starting deduplication of all collected titles...")
        deduplicated_titles = await deduplicate_news_titles(all_titles)
        print(f"Deduplication completed: {len(all_titles)} -> {len(deduplicated_titles)} titles")
    else:
        print("No titles collected from any URL")
        return []
    
    # Verify the deduplicated titles
    deduplicated_titles = deduplicated_titles[:max_titles_overall]
    print(f"Starting verification of {len(deduplicated_titles)} deduplicated titles...")
    ver_tasks = [call_verification_agent(title) for title in deduplicated_titles]
    ver_results = await asyncio.gather(*ver_tasks)
    
    final = []
    for title, content in ver_results:
        try:
            obj = json.loads(content) if content else None
            if obj:
                # Generate image for items that happened today
                if obj.get("happened_today") == "yes":
                    print(f"Generating image for: {title}")
                    try:
                        prompt = generate_image_prompt(obj.get("title", ""), obj.get("summary", ""))
                        image_base64 = await generate_image_with_dalle(prompt)
                        if image_base64:
                            # Upload image to S3 and store URL instead of base64
                            image_key = generate_image_key(obj.get("title", ""))
                            image_url = upload_image_to_s3(image_base64, image_key)
                            if image_url:
                                obj["image_url"] = image_url
                                print(f"Successfully uploaded image to S3 for: {title}")
                            else:
                                print(f"Failed to upload image to S3 for: {title}")
                        else:
                            print(f"Failed to generate image for: {title}")
                    except Exception as e:
                        print(f"Error generating image for {title}: {e}")
                
                final.append(obj)
                print(f"Successfully verified: {title}")
            else:
                print(f"Failed to verify after all retries: {title}")
        except Exception as e:
            print(f"Failed to parse verification result for {title}: {e}")
    
    print(f"Final results count: {len(final)}")
    
    # Extract news items that happened today for importance ranking
    today_news = [item for item in final if item.get("happened_today") == "yes"]
    print(f"Found {len(today_news)} news items that happened today")
    
    if today_news:
        # Get importance ranking for today's news
        print("Starting importance ranking for today's news...")
        try:
            today_titles = [item.get("title", "") for item in today_news]
            importance_result = await select_top_headlines(today_titles, num_headlines=num_headlines)
            
            # Create a mapping from title to importance data
            importance_mapping = {}
            if "reasoning" in importance_result and "headline_analysis" in importance_result["reasoning"]:
                for analysis in importance_result["reasoning"]["headline_analysis"]:
                    title_number = analysis.get("title_number", 0)
                    if 1 <= title_number <= len(today_titles):
                        original_title = today_titles[title_number - 1]
                        importance_mapping[original_title] = {
                            "importance_rank": analysis.get("importance_rank", -1),
                            "importance_score": analysis.get("importance_score", -1),
                            "importance_reason": analysis.get("reason", "No reason provided")
                        }
            
            print(f"Importance ranking completed for {len(importance_mapping)} items")
            
        except Exception as e:
            print(f"Error during importance ranking: {e}")
            importance_mapping = {}
    else:
        importance_mapping = {}
    
    # Add importance fields to all final results
    for item in final:
        title = item.get("title", "")
        if title in importance_mapping:
            # Add importance data for selected items
            item["importance_rank"] = importance_mapping[title]["importance_rank"]
            item["importance_score"] = importance_mapping[title]["importance_score"]
            item["importance_reason"] = importance_mapping[title]["importance_reason"]
        else:
            # Set default values for non-selected items
            item["importance_rank"] = -1
            item["importance_score"] = -1
            item["importance_reason"] = "Not ranked"
    
    print(f"Final results with importance ranking: {len(final)}")
    return final

async def deduplicate_news_titles(titles_list):
    """
    Use DeepSeek API to remove semantically duplicate news titles from a list.
    
    Args:
        titles_list (list): List of news titles to deduplicate
        
    Returns:
        list: Deduplicated list of news titles
    """
    if not titles_list or len(titles_list) <= 1:
        return titles_list
    
    prompt = f"""
Given the following list of news titles, identify and remove titles that are semantically similar or refer to the same news event. 
Keep only unique news stories and return a JSON array of deduplicated titles.

Rules:
1. If two titles refer to the same event/story, keep only one (preferably the more descriptive one)
2. If titles are about the same topic but different aspects/events, keep both
3. Consider variations in wording, different sources reporting the same story
4. Return only the deduplicated titles as a JSON array
5. Also remove anything that is not a news title

News titles to deduplicate:
{json.dumps(titles_list, ensure_ascii=False, indent=2)}

Return only a JSON array of deduplicated titles, nothing else.
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
        
        result = resp.choices[0].message.content.strip()
        print(f"[DEBUG] DeepSeek deduplication result: {result}")
        
        # Try to extract JSON array from the response
        try:
            # First try to parse as direct JSON
            deduplicated_titles = json.loads(result)
            if isinstance(deduplicated_titles, list):
                print(f"[DEBUG] Successfully deduplicated {len(titles_list)} titles to {len(deduplicated_titles)} titles")
                return deduplicated_titles
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON array from text
            try:
                # Look for JSON array pattern in the response
                import re
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    deduplicated_titles = json.loads(json_match.group())
                    if isinstance(deduplicated_titles, list):
                        print(f"[DEBUG] Successfully deduplicated {len(titles_list)} titles to {len(deduplicated_titles)} titles")
                        return deduplicated_titles
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # If all parsing attempts fail, return original list
        print(f"[DEBUG] Failed to parse deduplication result, returning original list")
        return titles_list
        
    except Exception as e:
        print(f"[DEBUG] DeepSeek API error during deduplication: {e}")
        # Return original list if API fails
        return titles_list

async def select_top_headlines(titles_list, num_headlines=3):
    """
    Use DeepSeek API to select the most important news titles for headlines.
    
    Args:
        titles_list (list): List of news titles to evaluate
        num_headlines (int): Number of top headlines to select (default: 3)
        
    Returns:
        dict: Dictionary containing selected headlines and reasoning
    """
    if not titles_list:
        return {
            "selected_headlines": [],
            "reasoning": "No titles provided",
            "total_evaluated": 0
        }
    
    if len(titles_list) <= num_headlines:
        return {
            "selected_headlines": titles_list,
            "reasoning": f"All {len(titles_list)} titles selected as there are fewer titles than requested headlines",
            "total_evaluated": len(titles_list)
        }
    
    # Create numbered titles for AI evaluation
    numbered_titles = []
    for i, title in enumerate(titles_list, 1):
        numbered_titles.append(f"{i}. {title}")
    
    prompt = f"""
Given the following numbered list of news titles, select the {num_headlines} most important and newsworthy headlines that would be most suitable for a news website's front page.

Evaluation criteria:
1. **Impact and significance**: How many people are affected or how important the event is
2. **Timeliness**: How current and urgent the news is
3. **Public interest**: How much public attention and curiosity it would generate
4. **Geographic scope**: Global, national, or major regional impact
5. **Diversity**: Try to cover different categories (politics, business, technology, sports, entertainment, etc.)

News titles to evaluate:
{json.dumps(numbered_titles, ensure_ascii=False, indent=2)}

Please return a JSON object with the following structure:
{{
    "selected_headlines": [1, 3, 5],
    "reasoning": {{
        "overall_criteria": "Brief explanation of the selection criteria used",
        "headline_analysis": [
            {{
                "title_number": 1,
                "importance_rank": 1,
                "importance_score": 95,
                "reason": "Why this headline was selected and its significance"
            }},
            {{
                "title_number": 3,
                "importance_rank": 2, 
                "importance_score": 87,
                "reason": "Why this headline was selected and its significance"
            }},
            {{
                "title_number": 5,
                "importance_rank": 3,
                "importance_score": 82, 
                "reason": "Why this headline was selected and its significance"
            }}
        ]
    }}
}}

Important rules:
1. Return ONLY the title numbers (1, 2, 3, etc.) in "selected_headlines", not the actual titles
2. Use numerical importance scores from 1-100 (100 being most important)
3. Assign importance_rank from 1 to {num_headlines} (1 being most important)
4. Do not modify or rewrite the original titles
5. Return only the JSON object, nothing else
"""
    
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
        resp = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a senior news editor with expertise in selecting the most important headlines for news websites. You understand news values, public interest, and editorial priorities. Always return only the requested JSON format with title numbers, never modify the original titles."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        result = resp.choices[0].message.content.strip()
        print(f"[DEBUG] DeepSeek headline selection result: {result}")
        
        # Try to extract JSON object from the response
        try:
            # First try to parse as direct JSON
            selection_result = json.loads(result)
            if isinstance(selection_result, dict) and "selected_headlines" in selection_result:
                # Convert title numbers back to actual titles
                selected_titles = []
                for title_number in selection_result["selected_headlines"]:
                    if isinstance(title_number, int) and 1 <= title_number <= len(titles_list):
                        selected_titles.append(titles_list[title_number - 1])
                    else:
                        print(f"[DEBUG] Invalid title number: {title_number}")
                
                # Update the result with actual titles
                selection_result["selected_headlines"] = selected_titles
                selection_result["total_evaluated"] = len(titles_list)
                
                print(f"[DEBUG] Successfully selected {len(selected_titles)} headlines from {len(titles_list)} titles")
                return selection_result
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON object from text
            try:
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    selection_result = json.loads(json_match.group())
                    if isinstance(selection_result, dict) and "selected_headlines" in selection_result:
                        # Convert title numbers back to actual titles
                        selected_titles = []
                        for title_number in selection_result["selected_headlines"]:
                            if isinstance(title_number, int) and 1 <= title_number <= len(titles_list):
                                selected_titles.append(titles_list[title_number - 1])
                            else:
                                print(f"[DEBUG] Invalid title number: {title_number}")
                        
                        # Update the result with actual titles
                        selection_result["selected_headlines"] = selected_titles
                        selection_result["total_evaluated"] = len(titles_list)
                        
                        print(f"[DEBUG] Successfully selected {len(selected_titles)} headlines from {len(titles_list)} titles")
                        return selection_result
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # If all parsing attempts fail, return a fallback result
        print(f"[DEBUG] Failed to parse headline selection result, returning fallback")
        return {
            "selected_headlines": titles_list[:num_headlines],
            "reasoning": {
                "overall_criteria": "Fallback selection due to parsing error",
                "headline_analysis": [
                    {
                        "title_number": i + 1,
                        "importance_rank": i + 1,
                        "importance_score": 100 - (i * 10),
                        "reason": "Selected as fallback due to API parsing error"
                    } for i in range(min(num_headlines, len(titles_list)))
                ]
            },
            "total_evaluated": len(titles_list)
        }
        
    except Exception as e:
        print(f"[DEBUG] DeepSeek API error during headline selection: {e}")
        # Return fallback result if API fails
        return {
            "selected_headlines": titles_list[:num_headlines],
            "reasoning": {
                "overall_criteria": "Fallback selection due to API error",
                "headline_analysis": [
                    {
                        "title_number": i + 1,
                        "importance_rank": i + 1,
                        "importance_score": 100 - (i * 10),
                        "reason": "Selected as fallback due to API error"
                    } for i in range(min(num_headlines, len(titles_list)))
                ]
            },
            "total_evaluated": len(titles_list)
        }

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
            "title": "Training camp battles to watch for all 32 teams",
            "summary": None,
            "source_link": None,
            "happened_today": "no",
            "category": "Sports",
            "importance_rank": -1,
            "importance_score": -1,
            "importance_reason": "Not ranked"
        },
        {
            "title": "Latest news, buzz from all 32 camps",
            "summary": "NFL training camps featured significant quarterback developments, with the Colts starting Anthony Richardson in their preseason opener while Daniel Jones prepares for Week 2. The Browns will start rookie Shedeur Sanders in their first preseason game due to injuries to Kenny Pickett and Dillon Gabriel. Texans coach DeMeco Ryans praised Nick Chubb's physical running and consistency in camp drills. The Bears held an intense practice with live tackling and multiple player altercations, described as one of their most physical sessions ever. Elsewhere, Packers quarterback Jordan Love impressed with a 55-yard touchdown throw during team drills.",
            "source_link": "https://www.nfl.com/news/inside-training-camp-highlights-buzz-from-tuesday-august-5",
            "happened_today": "yes",
            "category": "Sports",
            "importance_rank": 4,
            "importance_score": 75,
            "importance_reason": "Broad U.S. sports interest: NFL training camp updates engage millions of fans nationally. Timely for preseason but regionally focused, balancing category diversity."
        },
        {
            "title": "Preseason trade tracker",
            "summary": "The Boston Celtics have agreed to trade veteran forward Georges Niang and two future second-round draft picks to the Utah Jazz. The trade helps Boston reduce their luxury tax bill by approximately $50 million while creating roster flexibility. Utah will acquire a solid role player and additional draft assets using a trade exception. This marks Niang's return to Utah where he previously played from 2017-2021. The deal cannot be finalized until the league's moratorium period ends.",
            "source_link": "https://www.hoopsrumors.com/2025/08/celtics-to-trade-niang-two-picks-to-jazz.html",
            "happened_today": "yes",
            "category": "Sports",
            "importance_rank": -1,
            "importance_score": -1,
            "importance_reason": "Not ranked"
        },
        {
            "title": "Despite Trump's peace calls, Russian attacks on Ukraine double since inauguration",
            "summary": "A BBC Verify analysis confirms Russia has more than doubled missile and drone attacks on Ukraine since President Trump's January 2025 inauguration, with 27,158 munitions launched compared to 11,614 during Biden's final six months. The escalation occurred despite Trump's ceasefire efforts and temporary pauses in US military aid to Ukraine. Russian weapons production surged 66% during this period, with attacks peaking at 748 munitions in a single July 2025 barrage. Ukraine's air defenses are now critically strained, unable to intercept all attacks as Russia produces 170 attack drones daily. The White House maintains Trump is pursuing peace through tariffs and sanctions threats against Moscow.",
            "source_link": "https://www.bbc.com/news/articles/c5yl6eegv63o",
            "happened_today": "yes",
            "category": "World",
            "importance_rank": 1,
            "importance_score": 98,
            "importance_reason": "High global impact: Escalating Ukraine-Russia conflict affects international security, energy markets, and geopolitical stability. Extremely timely with recent developments and high public interest."
        },
        {
            "title": "RFK Jr cancels $500m in funding for mRNA vaccines that counter viruses like Covid",
            "summary": "Health Secretary RFK Jr. terminated $500 million in funding for 22 mRNA vaccine projects targeting respiratory viruses like COVID-19 and flu, halting development by companies including Pfizer and Moderna. The decision aligns with Kennedy's longstanding skepticism of mRNA technology, which scientists warn jeopardizes future pandemic preparedness. Experts emphasize mRNA vaccines were crucial during the COVID-19 crisis and adaptable for emerging threats. Kennedy defended the move, stating HHS will prioritize 'safer, broader strategies' like whole-virus vaccines. The announcement coincided with protests in Alaska, where Kennedy promoted a proposed 'universal vaccine' alternative.",
            "source_link": "https://apnews.com/article/kennedy-vaccines-mrna-pfizer-moderna-1fb5b9436f2957075064c18a6cbbe3c9",
            "happened_today": "yes",
            "category": "Health & Environment",
            "importance_rank": 2,
            "importance_score": 92,
            "importance_reason": "Major health policy impact: $500m vaccine funding cancellation affects public health preparedness globally. High relevance amid ongoing COVID concerns and widespread public interest."
        },
        {
            "title": "Hiroshima marks 80 years since atomic bombing",
            "summary": "Hiroshima commemorated the 80th anniversary of the atomic bombing with a ceremony attended by 55,000 people, including representatives from 120 countries. Aging survivors expressed frustration about growing global support for nuclear weapons as deterrence, with their average age now exceeding 86. Mayor Kazumi Matsui warned against military buildups and nuclear escalation amid current global conflicts, calling these policies 'utterly inhumane'. The ceremony featured a minute of silence at 8:15 a.m. – the exact moment the bomb detonated in 1945 – and the release of peace doves. Nearby, protesters gathered near the Atomic Bomb Dome with signs reading 'No Nuke, Stop War' and 'Free Gaza', resulting in two arrests.",
            "source_link": "https://apnews.com/article/japan-us-hiroshima-atomic-bombing-survivors-2a15654cf3689f4f4128098ffbf9a614",
            "happened_today": "yes",
            "category": "World",
            "importance_rank": 3,
            "importance_score": 90,
            "importance_reason": "Global historical significance: 80th anniversary of Hiroshima bombing carries profound lessons on nuclear weapons, with current relevance to global peace efforts. High symbolic impact and international interest."
        },
        {
            "title": "Zara ads banned in UK for 'unhealthily thin' models",
            "summary": "The UK Advertising Standards Authority (ASA) has banned two Zara advertisements for depicting models deemed 'unhealthily thin', citing irresponsible portrayal of body image. The ruling focuses on how clothing choices, poses, and lighting exaggerated the models' thinness in ads released last May. This follows a similar ban against Marks & Spencer last month, reflecting regulators' growing scrutiny of fashion advertising. Zara must remove the banned ads immediately but hasn't publicly responded to the decision. The ASA's crackdown highlights increasing pressure on brands to promote healthier body standards.",
            "source_link": "https://www.retailgazette.co.uk/blog/2025/08/zara-ads-asa/",
            "happened_today": "yes",
            "category": "Health & Environment",
            "importance_rank": 5,
            "importance_score": 70,
            "importance_reason": "Notable business/social impact: Zara ad ban addresses body image concerns, sparking international discourse on advertising ethics. Moderate global relevance but adds category diversity."
        },
        {
            "title": "Kim Kardashian, Help Us Find Amy Bradley!!!",
            "summary": "",
            "source_link": "",
            "happened_today": "no",
            "category": "Entertainment & Culture",
            "importance_rank": -1,
            "importance_score": -1,
            "importance_reason": "Not ranked"
        }
    ]
    
    return jsonify({
        "data": debug_content,
        "cached": False,
        "execution_time": 0.1,
        "message": "Debug content returned successfully"
    })

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

@app.route('/s3-status', methods=['GET'])
def s3_status_endpoint():
    """
    GET /s3-status
    Checks the status of AWS S3 configuration and connectivity
    """
    status = {
        "s3_configured": False,
        "bucket_accessible": False,
        "error": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check if S3 is configured
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
        status["error"] = "S3 credentials not configured"
        return jsonify(status)
    
    status["s3_configured"] = True
    
    # Test S3 connectivity
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            endpoint_url=S3_BUCKET_URL
        )
        
        # Try to list objects in bucket (minimal operation to test access)
        s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=1)
        status["bucket_accessible"] = True
        
    except ClientError as e:
        status["error"] = f"S3 access error: {str(e)}"
    except Exception as e:
        status["error"] = f"Unexpected error: {str(e)}"
    
    return jsonify(status)

@app.route('/test-s3-upload', methods=['POST'])
def test_s3_upload_endpoint():
    """
    POST /test-s3-upload
    Test endpoint to verify S3 configuration and upload functionality
    """
    try:
        data = request.get_json() or {}
        title = data.get('title', 'Test News Title')
        
        print(f"[DEBUG] Testing S3 upload for: {title}")
        
        # Test S3 configuration
        s3_config = {
            "aws_access_key_id": "Set" if AWS_ACCESS_KEY_ID else "Not set",
            "aws_secret_access_key": "Set" if AWS_SECRET_ACCESS_KEY else "Not set", 
            "s3_bucket_name": S3_BUCKET_NAME,
            "aws_region": AWS_REGION
        }
        
        # Create a test base64 image (small placeholder)
        test_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # Test S3 upload
        image_key = generate_image_key(title)
        image_url = upload_image_to_s3(test_base64, image_key)
        
        return jsonify({
            "status": "success" if image_url else "error",
            "s3_config": s3_config,
            "image_uploaded": image_url is not None,
            "image_url": image_url,
            "image_key": image_key
        })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/generate-image', methods=['POST'])
def generate_image_endpoint():
    """
    POST /generate-image
    Generates an image based on news title and summary.
    
    Request body:
    {
        "title": "News title",
        "summary": "News summary",
        "additional_field1": "value1",
        "additional_field2": "value2"
    }
    
    Response:
    {
        "image_url": "https://s3.amazonaws.com/bucket/image.png",
        "title": "News title",
        "summary": "News summary",
        "additional_field1": "value1",
        "additional_field2": "value2",
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        title = data.get('title', '').strip()
        summary = data.get('summary', '').strip()
        
        if not title and not summary:
            return jsonify({
                "status": "error",
                "message": "Either title or summary must be provided"
            }), 400
        
        # Generate image prompt from title and summary
        prompt = generate_image_prompt(title, summary)
        
        # Generate image using OpenAI DALL-E
        image_result = asyncio.run(generate_image_with_dalle(prompt))
        
        if image_result:
            # Upload image to S3 and return URL
            image_key = generate_image_key(title)
            image_url = upload_image_to_s3(image_result, image_key)
            
            if image_url:
                # Create response with original data plus image URL
                response_data = {
                    "status": "success",
                    "image_url": image_url
                }
                
                # Add all original fields from input data
                for key, value in data.items():
                    response_data[key] = value
                
                return jsonify(response_data)
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to upload image to S3"
                }), 500
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to generate image"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error generating image: {str(e)}"
        }), 500

def generate_image_prompt(title, summary):
    """Generate an image prompt from news title and summary"""
    # Combine title and summary for better context
    context = f"Title: {title}"
    if summary:
        context += f"\nSummary: {summary}"
    
    # Create a prompt for image generation
    prompt = f"""
    Generate a cartoon style image based on the following news story:
    {context}
    
    Focus: Visual elements that represent the story's key themes
    """
    
    # Clean up the prompt
    prompt = re.sub(r'\s+', ' ', prompt.strip())
    return prompt[:60000]  # Limit prompt length

async def generate_image_with_dalle(prompt):
    """Generate image using OpenAI DALL-E API"""
    if not OPENAI_API_KEY:
        print("OpenAI API key not configured")
        return None
    
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            tools=[{"type": "image_generation", "quality": "medium", "size":"1024x1024"}],
        )
        
        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]

        if image_data:
            image_base64 = image_data[0]
            return image_base64
            
    except Exception as e:
        print(f"Error generating image with DALL-E: {e}")
        return None

@app.route('/generate-image-debug', methods=['POST'])
def generate_image_debug_endpoint():
    """
    POST /generate-image-debug
    Returns a mock image URL for testing without API calls.
    """
    data = request.get_json() or {}
    title = data.get('title', 'Test News Title')
    summary = data.get('summary', 'Test news summary')
    
    prompt = generate_image_prompt(title, summary)
    
    # Create response with original data plus mock image
    response_data = {
        "status": "success",
        "image_url": "https://via.placeholder.com/1024x1024/0066cc/ffffff?text=Generated+Image",
        "debug": True
    }
    
    # Add all original fields from input data
    for key, value in data.items():
        response_data[key] = value
    
    return jsonify(response_data)

@app.route('/deduplicate-titles', methods=['POST'])
def deduplicate_titles_endpoint():
    """
    POST /deduplicate-titles
    Deduplicates a list of news titles using AI to remove semantically similar titles.
    
    Request body:
    {
        "titles": ["title1", "title2", "title3", ...]
    }
    
    Response:
    {
        "original_count": 5,
        "deduplicated_count": 3,
        "deduplicated_titles": ["title1", "title2", "title3"],
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        if not data or 'titles' not in data:
            return jsonify({
                "status": "error",
                "message": "No titles provided in request body"
            }), 400
        
        titles = data['titles']
        if not isinstance(titles, list):
            return jsonify({
                "status": "error",
                "message": "Titles must be a list"
            }), 400
        
        if not titles:
            return jsonify({
                "status": "success",
                "original_count": 0,
                "deduplicated_count": 0,
                "deduplicated_titles": [],
                "message": "No titles to deduplicate"
            })
        
        # Perform deduplication
        deduplicated_titles = asyncio.run(deduplicate_news_titles(titles))
        
        return jsonify({
            "status": "success",
            "original_count": len(titles),
            "deduplicated_count": len(deduplicated_titles),
            "original_titles": titles,
            "deduplicated_titles": deduplicated_titles,
            "message": f"Successfully deduplicated {len(titles)} titles to {len(deduplicated_titles)} titles"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deduplicating titles: {str(e)}"
        }), 500

@app.route('/select-headlines', methods=['POST'])
def select_headlines_endpoint():
    """
    POST /select-headlines
    Selects the most important news headlines from a list of titles using AI.
    
    Request body:
    {
        "titles": ["title1", "title2", "title3", ...],
        "num_headlines": 3
    }
    
    Response:
    {
        "selected_headlines": ["title1", "title2", "title3"],
        "reasoning": {
            "overall_criteria": "Selection criteria explanation",
            "headline_analysis": [
                {
                    "title": "title1",
                    "importance_score": "High",
                    "reason": "Why this headline was selected"
                }
            ]
        },
        "total_evaluated": 10,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        if not data or 'titles' not in data:
            return jsonify({
                "status": "error",
                "message": "No titles provided in request body"
            }), 400
        
        titles = data['titles']
        num_headlines = data.get('num_headlines', 3)
        
        if not isinstance(titles, list):
            return jsonify({
                "status": "error",
                "message": "Titles must be a list"
            }), 400
        
        if not titles:
            return jsonify({
                "status": "success",
                "selected_headlines": [],
                "reasoning": "No titles provided",
                "total_evaluated": 0,
                "message": "No titles to evaluate"
            })
        
        # Perform headline selection
        selection_result = asyncio.run(select_top_headlines(titles, num_headlines))
        
        return jsonify({
            "status": "success",
            **selection_result,
            "message": f"Successfully selected {len(selection_result['selected_headlines'])} headlines from {selection_result.get('total_evaluated', len(titles))} titles"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error selecting headlines: {str(e)}"
        }), 500

@app.route('/config-info', methods=['GET'])
def config_info_endpoint():
    """
    GET /config-info
    Returns current configuration values from environment variables
    """
    config_info = {
        "cache_config": {
            "cache_dir": CACHE_DIR,
            "cache_duration_hours": CACHE_DURATION_HOURS,
            "cache_file_path": CACHE_FILE
        },
        "news_config": {
            "target_urls": target_urls,
            "max_titles_per_url": max_titles_per_url,
            "max_titles_overall": max_titles_overall,
            "num_headlines": num_headlines
        },
        "api_config": {
            "deepseek_api_key": "Set" if DEEPSEEK_API_KEY else "Not set",
            "dify_api_key": "Set" if DIFY_API_KEY else "Not set",
            "openai_api_key": "Set" if OPENAI_API_KEY else "Not set"
        },
        "s3_config": {
            "aws_access_key_id": "Set" if AWS_ACCESS_KEY_ID else "Not set",
            "aws_secret_access_key": "Set" if AWS_SECRET_ACCESS_KEY else "Not set",
            "s3_bucket_name": S3_BUCKET_NAME,
            "aws_region": AWS_REGION,
            "s3_bucket_url": S3_BUCKET_URL
        },
        "environment_variables": {
            "CACHE_DIR": os.getenv("CACHE_DIR", "Not set (using default: cache)"),
            "CACHE_DURATION_HOURS": os.getenv("CACHE_DURATION_HOURS", "Not set (using default: 48)"),
            "TARGET_URLS": os.getenv("TARGET_URLS", "Not set (using default: https://tmz.com,https://bbc.com,https://cnn.com,https://espn.com,https://pagesix.com/)"),
            "MAX_TITLES_PER_URL": os.getenv("MAX_TITLES_PER_URL", "Not set (using default: 40)"),
            "MAX_TITLES_OVERALL": os.getenv("MAX_TITLES_OVERALL", "Not set (using default: 80)"),
            "NUM_HEADLINES": os.getenv("NUM_HEADLINES", "Not set (using default: 5)")
        }
    }
    
    return jsonify(config_info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
