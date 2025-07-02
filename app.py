from dotenv import load_dotenv
import asyncio
import os
import json
import re
import aiohttp
import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from flask import Flask, jsonify, request

load_dotenv() 

app = Flask(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DIFY_API_URL = 'https://api.dify.ai/v1/chat-messages'
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

query_prefix = (
    'For the news story title, verify whether it happened today or not. If yes, search it on web '
    'and find a link to the original news article from a reliable news website. Also make a summary '
    'of about 5 sentences of the news article. Return a JSON-like object with fields "title", "summary", '
    '"source_link" and "happened_today" (yes or no). News title to verify:'
)

target_urls = ["https://tmz.com", "https://bbc.com"]

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
        import traceback
        print(f"DeepSeek API error for {url}: {e}")
        traceback.print_exc()
        return url, None

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
            continue
        to_verify = titles[:2]  # limit to first 2
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
    Returns a JSON array like:
    [
      { "source": "...", "title": "...", "verification": { ... } },
      ...
    ]
    """
    results = asyncio.run(analyze_all())
    return jsonify(results)


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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
