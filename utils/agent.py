import os, json, time, re, requests, asyncio, aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import tz
from typing import Any, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()


# =============== Tools =================

async def google_search(query, max_results=10, depth="advanced", include_answers=False):
    """
    Google Custom Search API:
      Endpoint: GET https://www.googleapis.com/customsearch/v1
      Query params: {
        "key": "...", "cx": "...", "q": "...",
        "num": 1-10
      }
    Return: JSON string of simplified search results
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    if not api_key or not search_engine_id:
        return f"[google_search] Missing GOOGLE_API_KEY or GOOGLE_SEARCH_ENGINE_ID; fallback echo for '{query}'"
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": min(max_results, 10),  # Google CSE max is 10
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2000)) as session:
            async with session.get(url, params=params) as r:
                r.raise_for_status()
                data = await r.json()
                results = []
                
                # Handle Google CSE response format
                items = data.get("items", [])
                for item in items[:max_results]:
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet")
                    })
                
                # If the model needs structured data directly, keep it as JSON; if readability is needed, it can be converted to text.
                return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"[google_search] error: {e}"

async def crawl_webpage(url, max_chars=20000):
    """
    Simple web crawler: fetch HTML -> extract <title> and readable text, trim to max_chars
    """
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2000)) as session:
            async with session.get(url, headers={"User-Agent": "mini-react-agent/1.0"}) as r:
                r.raise_for_status()
                html_text = await r.text()
                soup = BeautifulSoup(html_text, "html.parser")
                for s in soup(["script", "style", "noscript"]):
                    s.decompose()
                title = (soup.title.string.strip() if soup.title and soup.title.string else "")
                text = " ".join(soup.get_text(" ").split())
                content = (f"TITLE: {title}\n" + text)[:max_chars]
                return content
    except Exception as e:
        return f"[crawl_webpage] error: {e}"

async def get_now(tz_name="America/New_York", fmt="%Y-%m-%d %H:%M:%S %Z"):
    """
    Return current date/time in a timezone
    """
    zone = tz.gettz(tz_name)
    now = datetime.now(zone)
    return now.strftime(fmt)

TOOLS = {
    "google_search": google_search,
    "crawl_webpage": crawl_webpage,
    "get_now": get_now,
}

TOOL_DESCRIPTIONS = """
You have access to the following tools:

1) google_search
   - purpose: web search for queries via Google Custom Search API
   - args: {"query": string, "depth": "basic"|"advanced", "include_answers": optional bool}

2) crawl_webpage
   - purpose: fetch and extract readable text from a URL
   - args: {"url": string}

3) get_now
   - purpose: get current date/time in a timezone
   - args: {"tz_name": optional string, "fmt": optional string}
"""

# =============== LLM（DeepSeek） ===================

DEEPSEEK_BASE  = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")

# =============== ReAct Prompt Template =================

REACT_INSTRUCTION = f'''
You are a helpful research agent that can reason step by step and use tools when needed.

Follow this exact format on each step:
Thought: your reasoning (concise).
Action: one of [google_search, crawl_webpage, get_now] or "finish".
Action Input: a valid JSON object for the chosen tool (or an empty object for finish).
Observation: (will be filled in by the system after tool execution)

Rules:
- Use tools only when they reduce uncertainty.
- Prefer searching before crawling specific URLs.
- Keep each Thought short.
- When you have the final answer, use Action: finish and put your answer text in Action Input as {{"final": "..."}}
- This is production code, not just a similation. When scrape a webpage, put everything in the observation to facilitate future actions.

The output must be in this exact JSON schema (no extra keys, no extra text):
{{
  "thought": "<short string>",
  "action": "<one of google_search | crawl_webpage | get_now | finish>",
  "action_input": {{ ... JSON args for the chosen action... }}
}}
The Observation results will be added to the history automatically after each tool execution. Do not generate observation contents.

Tool specs:
{TOOL_DESCRIPTIONS}
'''

def extract_first_json_object(s: str) -> Dict[str, Any]:
    s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    frag = m.group(0)
    return json.loads(frag)

def validate_step_obj(obj: dict) -> tuple[str, dict, str]:
    """
    Validate JSON object: keys must and are only allowed to be thought/action/action_input;
    action must be in the allowed set; action_input is an object.
    Return (action, action_input, thought)
    """
    if not isinstance(obj, dict):
        raise ValueError("Step output must be a JSON object.")
    required = {"thought", "action", "action_input"}
    extra = set(obj.keys()) - required
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    if extra:
        raise ValueError(f"Unexpected keys: {extra}")
    thought = obj["thought"]
    action = obj["action"]
    action_input = obj["action_input"]

    allowed = set(TOOLS.keys()) | {"finish"}
    if action not in allowed:
        raise ValueError(f"Invalid action '{action}'. Allowed: {sorted(allowed)}")
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    return action, action_input, thought


def build_prompt(user_question, scratchpad):
    return (
        REACT_INSTRUCTION
        + "\n\nUser Question:\n"
        + user_question
        + "\n\nScratchpad (history so far):\n"
        + scratchpad
        + "\n\nNext step, strictly follow the format."
    )

ACTION_RE = re.compile(r"Action:\s*(.+)")
INPUT_RE  = re.compile(r"Action Input:\s*(\{.*\})", re.DOTALL)

def parse_action(text):
    act_match = ACTION_RE.search(text)
    inp_match = INPUT_RE.search(text)
    if not act_match or not inp_match:
        raise ValueError("Failed to parse Action / Action Input")
    action = act_match.group(1).strip()
    try:
        action_input = json.loads(inp_match.group(1).strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for Action Input: {e}")
    return action, action_input

# =============== main loop =======================

async def call_llm_json(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stop": ["```", "\nObservation:", "Observation:"],
        "response_format": {"type": "json_object"}
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6000)) as session:
        async with session.post(f"{DEEPSEEK_BASE}/chat/completions", headers=headers, json=payload) as r:
            r.raise_for_status()
            data = await r.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            return content


async def react_agent_json_only(question: str, max_steps: int = 10) -> Dict[str, Any]:
    """
    Return:
      {
        "steps": <int>,
        "answer": <str>,
        "trace": [ {thought, action, action_input, observation}, ... ]
      }
    """
    trace = []
    trace_jsonl = ""

    for step in range(1, max_steps + 1):
        prompt = build_prompt(question, trace_jsonl)
        try:
            llm_raw = await call_llm_json(prompt)
        except Exception as e:
            return {"steps": step - 1, "answer": f"[error] LLM call failed: {e}", "trace": trace}

        # parse and verify
        try:
            obj = extract_first_json_object(llm_raw)
            action, action_input, thought = validate_step_obj(obj)
        except Exception as e:
            repair_obs = {"error": f"format violation: {e}"}
            trace.append({"thought": "format error; will retry", "action": "none", "action_input": {}, "observation": repair_obs})
            trace_jsonl += json.dumps(trace[-1], ensure_ascii=False) + "\n"
            continue

        # end
        if action == "finish":
            final = action_input.get("final", "")
            if not isinstance(final, str) or not final.strip():
                repair_obs = {"error": "finish requires action_input.final (non-empty string)"}
                trace.append({"thought": thought, "action": action, "action_input": action_input, "observation": repair_obs})
                trace_jsonl += json.dumps(trace[-1], ensure_ascii=False) + "\n"
                continue
            trace.append({"thought": thought, "action": action, "action_input": action_input, "observation": {"note": "finished"}})
            return {"steps": step, "answer": final.strip(), "trace": trace}

        # observation = run_tool(action, action_input)
        if action not in TOOLS:
            observation = f"[system] Unknown tool: {action}"
        else:
            try:
                observation = await TOOLS[action](**action_input)
            except TypeError:
                observation = f"[system] Invalid args for {action}: {action_input}"
            except Exception as e:
                observation = f"[system] Tool {action} error: {e}"

        record = {"thought": thought, "action": action, "action_input": action_input, "observation": observation}
        trace.append(record)
        trace_jsonl += json.dumps(record, ensure_ascii=False) + "\n"

        # Use asyncio.sleep instead of time.sleep for async compatibility
        import asyncio
        await asyncio.sleep(0.25)

    return {"steps": max_steps, "answer": "[fallback] step limit reached without finish", "trace": trace}

query_prefix = """
    For the given news story title, determine whether the event happened today and makwrite a comprehensive summary based on the information from the web.

    Definition of "Happened today":
        - The event took place within the last 24 hours, OR the event occurred earlier but was first reported by multiple reliable news websites within the last 24 hours.
        - Events that only appear in articles published more than 24 hours ago are NOT considered “happened today.”

    If the event qualifies, search the web and provide a link to one of the original articles from a reliable news outlet.
    Then, write a news summary in neutral English:
        - Begin with a 1–2 sentence executive summary.
        - Follow with 5–8 short paragraphs covering more details of the event.
        - Target length: ~520 words (acceptable range: 400–600).
        - Write in the narrative flow of a professional news article: start with a strong lead, expand with details and context in connected paragraphs, and close with implications or what comes next.
        - Do not label sections (‘Executive summary,’ ‘Who/What/When’) — just write as natural prose.
        - Write a clear and easy-to-read summary of the news story. Organize the text into multiple paragraphs instead of one large block. Each paragraph should be between 4–6 sentences (adjust slightly if it makes the flow more natural). Make sure the writing is smooth, logically connected, and easy to follow. Do not compress everything into a single dense paragraph.
        - Use information from at least two independent, reliable news outlets if available. Cross-check details and synthesize them into a unified narrative rather than summarizing each source separately.

    Finally, return a JSON-like object with the following fields:
        - "title"
        - "summary"
        - "source_link"
        - "happened_today" (yes or no)
        - "category" (Choose the single category that best matches the primary subject. Do not invent categories outside this list.):
            - World
            - Politics & Society
            - Business & Economy
            - Technology & Science
            - Health & Environment
            - Entertainment & Culture
            - Sports
            - Other

    News title to verify: """