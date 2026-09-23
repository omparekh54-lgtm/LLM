"""
Step 6: Download and format instruction/code pairs for instruction-tuning.

Uses CodeAlpaca (~20K instruction -> Python code pairs), formatted with the
special tokens reserved in the tokenizer at step 2:

    <|instruction|> write a function that scrapes titles from a webpage
    <|response|> import requests
    ...
    <|endoftext|>

At inference time, you prompt with "<|instruction|> <your request>
<|response|>" and let the model generate until it produces <|endoftext|>.

Also pulls in a larger set of docstring->function pairs mined from your own
pretraining corpus, if available -- free extra instruction data with no
separate download, since well-documented functions are naturally
(instruction, code) pairs (docstring = instruction, function body = code).

Additionally includes a small hand-curated set of scraper/bot/utility
examples, oversampled slightly, to bias the model toward the specific use
cases this project targets ("help me make a scraper", "help me make a bot")
rather than leaving that entirely to CodeAlpaca's general distribution.
"""

import os
import re
import ast
import glob
import json
import random
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data/instruct"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CODE_DIR = "data/raw"  # from step 1, reused here for free docstring-pair mining
MAX_DOCSTRING_PAIRS = 15_000  # cut from 60,000 -- mined pairs are noisier than curated ones,
                               # and were badly diluting the curated signal (only ~0.2% of
                               # the dataset before this change -- see reasoning in this update)
CURATED_OVERSAMPLE = 18  # raised from 4x -- 4x wasn't enough weight for curated examples to
                          # reliably win out over tens of thousands of mined/CodeAlpaca pairs


def load_codealpaca():
    print("Downloading CodeAlpaca...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    pairs = []
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        input_ctx = ex.get("input", "").strip()
        output = ex.get("output", "").strip()
        if not instruction or not output:
            continue
        if input_ctx:
            instruction = f"{instruction}\n{input_ctx}"
        pairs.append({"instruction": instruction, "response": output})
    print(f"CodeAlpaca: {len(pairs)} pairs")
    return pairs


# glaive-code-assistant: real user-phrased questions (not synthetic templates
# like CodeAlpaca), ~60% Python. Answers are markdown-formatted (prose mixed
# with ```python code blocks), which doesn't match our plain-code response
# format -- so we extract just the code block, not the surrounding prose.
# Applying the same lesson learned from MAX_DOCSTRING_PAIRS: capped and
# filtered hard, not dumped in wholesale, since raw volume already proved to
# dilute quality more than it helped.
GLAIVE_MAX_PAIRS = 12_000
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code_block(markdown_answer: str):
    """Pull the first fenced code block out of a markdown-formatted answer.
    Returns None if there's no block, or if what's inside doesn't parse as
    valid Python (catches non-Python answers and malformed extracts)."""
    match = CODE_BLOCK_RE.search(markdown_answer)
    if not match:
        return None
    code = match.group(1).strip()
    if not (10 < len(code) < 2000):
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def load_glaive_examples():
    print(f"Downloading glaive-code-assistant (capping at {GLAIVE_MAX_PAIRS:,} filtered pairs)...")
    try:
        ds = load_dataset("glaiveai/glaive-code-assistant", split="train", streaming=True)
    except Exception as e:
        print(f"Could not load glaive-code-assistant ({e}) -- skipping this source.")
        return []

    pairs = []
    skipped_no_code = 0
    skipped_invalid = 0
    for ex in ds:
        if len(pairs) >= GLAIVE_MAX_PAIRS:
            break
        question = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()
        if not question or not answer:
            continue

        code = extract_python_code_block(answer)
        if code is None:
            if "```" in answer:
                skipped_invalid += 1
            else:
                skipped_no_code += 1
            continue

        pairs.append({"instruction": question, "response": code})

    print(f"glaive-code-assistant: {len(pairs)} pairs "
          f"(skipped {skipped_no_code} with no code block, {skipped_invalid} with invalid/non-Python code)")
    return pairs


# captures the real parameter list now (group 2), instead of discarding it
# and writing a placeholder "(...)" -- that placeholder was a real bug:
# training data containing literal "..." teaches the model to sometimes
# output that literal ellipsis instead of real parameters.
DOCSTRING_FUNC_RE = re.compile(
    r'def\s+(\w+)\s*\(([^)]*)\)\s*:\s*\n\s*"""(.*?)"""\s*\n(.*?)(?=\ndef\s|\nclass\s|\Z)',
    re.DOTALL,
)

# bodies that indicate a stub, not a real implementation -- these teach the
# model "docstring -> empty function", which is actively unhelpful
STUB_BODIES = {"pass", "...", "raise NotImplementedError", "raise NotImplementedError()"}


def clean_body(body: str) -> str:
    """Remove leading/trailing blank lines while preserving the internal
    indentation of the code itself (which must stay intact and consistent
    for the snippet to remain valid Python)."""
    lines = body.split("\n")
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def is_low_quality_pair(func_name, docstring, body):
    body_stripped = body.strip()
    if body_stripped in STUB_BODIES:
        return True
    # docstrings that are just the function name restated add no real signal
    if docstring.strip().lower().replace("_", " ") == func_name.lower().replace("_", " "):
        return True
    # skip anything that doesn't even parse as valid Python -- garbage in, garbage out.
    # body already carries its original (correct) indentation from the source,
    # so it's validated as-is rather than re-indented, which would break it.
    try:
        ast.parse(f"def {func_name}(placeholder):\n{body}")
    except SyntaxError:
        return True
    return False


def mine_docstring_pairs():
    """Extract (docstring -> function body) pairs from the raw code corpus.
    Free instruction-style data: a docstring is naturally an instruction,
    and the function body is naturally the desired 'response'."""
    files = sorted(glob.glob(os.path.join(RAW_CODE_DIR, "shard_*.txt")))
    if not files:
        print("No raw code shards found -- skipping docstring mining (run 01_prepare_data.py first if you want this).")
        return []

    pairs = []
    skipped_low_quality = 0
    skipped_length = 0
    for fp in tqdm(files, desc="mining docstring pairs"):
        if len(pairs) >= MAX_DOCSTRING_PAIRS:
            break
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        for match in DOCSTRING_FUNC_RE.finditer(text):
            func_name, params, docstring, body = match.groups()
            docstring = docstring.strip()
            body = clean_body(body)  # preserves internal indentation, unlike a naive .strip()
            params = params.strip()

            if not (10 < len(docstring) < 300 and 10 < len(body) < 2000):
                skipped_length += 1
                continue
            if is_low_quality_pair(func_name, docstring, body):
                skipped_low_quality += 1
                continue

            instruction = f"Write a Python function `{func_name}` that: {docstring}"
            response = f"def {func_name}({params}):\n{body}"  # real signature now, not a placeholder
            pairs.append({"instruction": instruction, "response": response})

            if len(pairs) >= MAX_DOCSTRING_PAIRS:
                break
    print(f"Mined {len(pairs)} docstring-derived pairs "
          f"(skipped {skipped_length} too short/long, {skipped_low_quality} low-quality)")
    return pairs


# Hand-curated examples targeting this project's actual intended use cases --
# scrapers, bots, and common utility functions. CodeAlpaca and the mined
# pairs are general-purpose; this set exists specifically to bias the
# trained model toward being reliably good at what you'll actually ask it
# for. All examples below are syntactically valid Python (verified at
# import time by the self-check at the bottom of this file).
CURATED_EXAMPLES = [
    {
        "instruction": "write a function that scrapes all links from a webpage",
        "response": (
            "def scrape_links(url):\n"
            "    \"\"\"Scrape all hyperlinks from a webpage.\"\"\"\n"
            "    import requests\n"
            "    from bs4 import BeautifulSoup\n"
            "    response = requests.get(url)\n"
            "    soup = BeautifulSoup(response.text, 'html.parser')\n"
            "    links = [a.get('href') for a in soup.find_all('a') if a.get('href')]\n"
            "    return links"
        ),
    },
    {
        "instruction": "write a function that scrapes all image urls from a webpage",
        "response": (
            "def scrape_images(url):\n"
            "    \"\"\"Scrape all image URLs from a webpage.\"\"\"\n"
            "    import requests\n"
            "    from bs4 import BeautifulSoup\n"
            "    response = requests.get(url)\n"
            "    soup = BeautifulSoup(response.text, 'html.parser')\n"
            "    images = [img.get('src') for img in soup.find_all('img') if img.get('src')]\n"
            "    return images"
        ),
    },
    {
        "instruction": "write a function that scrapes the text content of all headings from a webpage",
        "response": (
            "def scrape_headings(url):\n"
            "    \"\"\"Scrape the text of all heading tags (h1-h6) from a webpage.\"\"\"\n"
            "    import requests\n"
            "    from bs4 import BeautifulSoup\n"
            "    response = requests.get(url)\n"
            "    soup = BeautifulSoup(response.text, 'html.parser')\n"
            "    headings = []\n"
            "    for level in range(1, 7):\n"
            "        for tag in soup.find_all(f'h{level}'):\n"
            "            headings.append(tag.get_text(strip=True))\n"
            "    return headings"
        ),
    },
    {
        "instruction": "write a function that scrapes a table from a webpage into a list of rows",
        "response": (
            "def scrape_table(url, table_index=0):\n"
            "    \"\"\"Scrape a table from a webpage into a list of row lists.\"\"\"\n"
            "    import requests\n"
            "    from bs4 import BeautifulSoup\n"
            "    response = requests.get(url)\n"
            "    soup = BeautifulSoup(response.text, 'html.parser')\n"
            "    tables = soup.find_all('table')\n"
            "    rows = []\n"
            "    for tr in tables[table_index].find_all('tr'):\n"
            "        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]\n"
            "        rows.append(cells)\n"
            "    return rows"
        ),
    },
    {
        "instruction": "write a telegram bot that echoes back any message it receives",
        "response": (
            "import telebot\n\n"
            "def run_echo_bot(token):\n"
            "    \"\"\"Run a Telegram bot that echoes back any message it receives.\"\"\"\n"
            "    bot = telebot.TeleBot(token)\n\n"
            "    @bot.message_handler(func=lambda message: True)\n"
            "    def echo(message):\n"
            "        bot.reply_to(message, message.text)\n\n"
            "    bot.polling()"
        ),
    },
    {
        "instruction": "write a discord bot that responds to a !ping command with pong",
        "response": (
            "import discord\n"
            "from discord.ext import commands\n\n"
            "def run_ping_bot(token):\n"
            "    \"\"\"Run a Discord bot that responds to !ping with pong.\"\"\"\n"
            "    intents = discord.Intents.default()\n"
            "    intents.message_content = True\n"
            "    bot = commands.Bot(command_prefix='!', intents=intents)\n\n"
            "    @bot.command()\n"
            "    async def ping(ctx):\n"
            "        await ctx.send('pong')\n\n"
            "    bot.run(token)"
        ),
    },
    {
        "instruction": "write a reddit bot that prints the titles of the top 10 hottest posts in a subreddit",
        "response": (
            "import praw\n\n"
            "def print_top_posts(client_id, client_secret, user_agent, subreddit_name):\n"
            "    \"\"\"Print the titles of the top 10 hottest posts in a subreddit.\"\"\"\n"
            "    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)\n"
            "    subreddit = reddit.subreddit(subreddit_name)\n"
            "    for post in subreddit.hot(limit=10):\n"
            "        print(post.title)"
        ),
    },
    {
        "instruction": "write a function that reads a csv file and returns it as a list of dictionaries",
        "response": (
            "def read_csv_as_dicts(filepath):\n"
            "    \"\"\"Read a CSV file and return its rows as a list of dictionaries.\"\"\"\n"
            "    import csv\n"
            "    with open(filepath, newline='', encoding='utf-8') as f:\n"
            "        reader = csv.DictReader(f)\n"
            "        return [dict(row) for row in reader]"
        ),
    },
    {
        "instruction": "write a function that fetches json data from an api endpoint",
        "response": (
            "def fetch_json(url, params=None):\n"
            "    \"\"\"Fetch and parse JSON data from an API endpoint.\"\"\"\n"
            "    import requests\n"
            "    response = requests.get(url, params=params)\n"
            "    response.raise_for_status()\n"
            "    return response.json()"
        ),
    },
    {
        "instruction": "write a function that downloads a file from a url and saves it locally",
        "response": (
            "def download_file(url, save_path):\n"
            "    \"\"\"Download a file from a URL and save it to a local path.\"\"\"\n"
            "    import requests\n"
            "    response = requests.get(url, stream=True)\n"
            "    response.raise_for_status()\n"
            "    with open(save_path, 'wb') as f:\n"
            "        for chunk in response.iter_content(chunk_size=8192):\n"
            "            f.write(chunk)\n"
            "    return save_path"
        ),
    },
    {
        "instruction": "write a function that monitors a webpage and notifies when its content changes",
        "response": (
            "def check_for_change(url, previous_hash=None):\n"
            "    \"\"\"Check if a webpage's content has changed since the last check.\"\"\"\n"
            "    import requests\n"
            "    import hashlib\n"
            "    response = requests.get(url)\n"
            "    current_hash = hashlib.sha256(response.content).hexdigest()\n"
            "    changed = previous_hash is not None and current_hash != previous_hash\n"
            "    return changed, current_hash"
        ),
    },
    {
        "instruction": "write a whatsapp bot using twilio that replies hello to any incoming message",
        "response": (
            "from flask import Flask, request\n"
            "from twilio.twiml.messaging_response import MessagingResponse\n\n"
            "app = Flask(__name__)\n\n"
            "@app.route('/whatsapp', methods=['POST'])\n"
            "def whatsapp_reply():\n"
            "    \"\"\"Reply 'hello' to any incoming WhatsApp message via Twilio.\"\"\"\n"
            "    resp = MessagingResponse()\n"
            "    resp.message('hello')\n"
            "    return str(resp)"
        ),
    },
    {
        # Directly patches a real bug we observed: forgetting .get_text() and
        # returning raw Tag objects instead of the actual text content.
        "instruction": "write a function that scrapes titles from a webpage",
        "response": (
            "def scrape_titles(url):\n"
            "    \"\"\"Scrape the text of all title-like elements from a webpage.\"\"\"\n"
            "    import requests\n"
            "    from bs4 import BeautifulSoup\n"
            "    response = requests.get(url)\n"
            "    soup = BeautifulSoup(response.text, 'html.parser')\n"
            "    titles = [tag.get_text(strip=True) for tag in soup.find_all('title')]\n"
            "    return titles"
        ),
    },
    {
        # Directly patches a real logic bug we observed: this function name
        # implies "second largest" but a naive wrong implementation just
        # returns the largest. This teaches the correct approach: sort
        # descending and take the second unique value, handling duplicates.
        "instruction": "write a function that finds the second largest number in a list",
        "response": (
            "def second_largest(nums):\n"
            "    \"\"\"Find the second largest unique number in a list.\"\"\"\n"
            "    unique_sorted = sorted(set(nums), reverse=True)\n"
            "    if len(unique_sorted) < 2:\n"
            "        return None\n"
            "    return unique_sorted[1]"
        ),
    },
    {
        # Directly patches a real bug we observed: a rate limiter decorator
        # that had the right shape (decorator, docstring) but never actually
        # tracked or checked call timestamps -- it was a no-op.
        "instruction": "write a rate limiter decorator that allows a maximum number of calls per time period",
        "response": (
            "import time\n"
            "from functools import wraps\n\n"
            "def rate_limiter(max_calls, period_seconds):\n"
            "    \"\"\"Decorator that allows at most max_calls within period_seconds.\"\"\"\n"
            "    def decorator(func):\n"
            "        call_times = []\n\n"
            "        @wraps(func)\n"
            "        def wrapper(*args, **kwargs):\n"
            "            now = time.time()\n"
            "            while call_times and call_times[0] <= now - period_seconds:\n"
            "                call_times.pop(0)\n"
            "            if len(call_times) >= max_calls:\n"
            "                raise RuntimeError('Rate limit exceeded')\n"
            "            call_times.append(now)\n"
            "            return func(*args, **kwargs)\n"
            "        return wrapper\n"
            "    return decorator"
        ),
    },
    {
        "instruction": "write a function that validates whether a string is a properly formatted email address",
        "response": (
            "def validate_email(email):\n"
            "    \"\"\"Check whether a string is a properly formatted email address.\"\"\"\n"
            "    import re\n"
            "    pattern = r'^[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}$'\n"
            "    return re.match(pattern, email) is not None"
        ),
    },
    {
        # Reinforcement: this exact task type (telegram bot) failed in
        # testing despite an existing curated example, because it was
        # statistically drowned out. Adding a second, differently-phrased
        # variant increases the chance the underlying pattern is learned
        # rather than one specific phrasing being memorized.
        "instruction": "make a telegram bot in python that replies to every message with the same text it received",
        "response": (
            "import telebot\n\n"
            "def create_bot(token):\n"
            "    \"\"\"Create a Telegram bot that echoes every message it receives.\"\"\"\n"
            "    bot = telebot.TeleBot(token)\n\n"
            "    @bot.message_handler(func=lambda message: True)\n"
            "    def handle_message(message):\n"
            "        bot.send_message(message.chat.id, message.text)\n\n"
            "    return bot"
        ),
    },
    {
        "instruction": "write a discord bot that tracks and displays user reputation points",
        "response": (
            "import discord\n"
            "from discord.ext import commands\n"
            "import sqlite3\n\n"
            "intents = discord.Intents.default()\n"
            "intents.message_content = True\n"
            "bot = commands.Bot(command_prefix='!', intents=intents)\n\n"
            "def get_db():\n"
            "    \"\"\"Connect to the reputation database.\"\"\"\n"
            "    conn = sqlite3.connect('reputation.db')\n"
            "    conn.execute('CREATE TABLE IF NOT EXISTS rep (user_id TEXT PRIMARY KEY, points INTEGER)')\n"
            "    return conn\n\n"
            "@bot.command()\n"
            "async def rep(ctx, member: discord.Member):\n"
            "    \"\"\"Show a member's reputation points.\"\"\"\n"
            "    conn = get_db()\n"
            "    row = conn.execute('SELECT points FROM rep WHERE user_id = ?', (str(member.id),)).fetchone()\n"
            "    conn.close()\n"
            "    points = row[0] if row else 0\n"
            "    await ctx.send(f'{member.display_name} has {points} reputation points.')"
        ),
    },
    {
        "instruction": "write a streamlit app that lets users search and filter a product catalog",
        "response": (
            "import streamlit as st\n"
            "import pandas as pd\n\n"
            "st.title('Product Catalog')\n\n"
            "products = pd.read_csv('products.csv')\n"
            "search_term = st.text_input('Search products')\n"
            "category = st.selectbox('Category', ['All'] + list(products['category'].unique()))\n\n"
            "filtered = products\n"
            "if search_term:\n"
            "    filtered = filtered[filtered['name'].str.contains(search_term, case=False)]\n"
            "if category != 'All':\n"
            "    filtered = filtered[filtered['category'] == category]\n\n"
            "st.dataframe(filtered)"
        ),
    },
    {
        "instruction": "write a python script that sends a scheduled email reminder",
        "response": (
            "import smtplib\n"
            "import schedule\n"
            "import time\n"
            "from email.mime.text import MIMEText\n\n"
            "def send_reminder(to_address, subject, body, smtp_server, smtp_user, smtp_password):\n"
            "    \"\"\"Send a single reminder email.\"\"\"\n"
            "    msg = MIMEText(body)\n"
            "    msg['Subject'] = subject\n"
            "    msg['To'] = to_address\n"
            "    with smtplib.SMTP(smtp_server, 587) as server:\n"
            "        server.starttls()\n"
            "        server.login(smtp_user, smtp_password)\n"
            "        server.send_message(msg)\n\n"
            "def run_scheduler(send_time, to_address, subject, body, smtp_server, smtp_user, smtp_password):\n"
            "    \"\"\"Run the reminder on a daily schedule at the given time (e.g. '09:00').\"\"\"\n"
            "    schedule.every().day.at(send_time).do(\n"
            "        send_reminder, to_address, subject, body, smtp_server, smtp_user, smtp_password)\n"
            "    while True:\n"
            "        schedule.run_pending()\n"
            "        time.sleep(60)"
        ),
    },
    {
        # Targets a real gap found in testing: generic "make an HTTP GET
        # request and parse JSON" phrasing didn't match the existing
        # fetch_json example closely enough, and generation produced code
        # that never actually made a network request at all.
        "instruction": "write a function that makes an HTTP GET request and parses the JSON response",
        "response": (
            "def get_json(url, params=None):\n"
            "    \"\"\"Make an HTTP GET request and parse the JSON response.\"\"\"\n"
            "    import requests\n"
            "    response = requests.get(url, params=params)\n"
            "    response.raise_for_status()\n"
            "    return response.json()"
        ),
    },
    {
        "instruction": "write a function that creates a pandas dataframe from a list of dictionaries",
        "response": (
            "def create_dataframe(records):\n"
            "    \"\"\"Create a pandas DataFrame from a list of dictionaries.\"\"\"\n"
            "    import pandas as pd\n"
            "    return pd.DataFrame(records)"
        ),
    },
    {
        # Targets a real gap: a Flask CRUD API request (create + list, not
        # just a single health-check style endpoint) had no curated
        # coverage and generation produced an incomplete, broken result.
        "instruction": "write a flask api with endpoints to create and list to-do items",
        "response": (
            "from flask import Flask, request, jsonify\n\n"
            "app = Flask(__name__)\n"
            "todos = []\n\n"
            "@app.route('/todos', methods=['POST'])\n"
            "def create_todo():\n"
            "    \"\"\"Create a new to-do item from JSON request data.\"\"\"\n"
            "    data = request.get_json()\n"
            "    todo = {'id': len(todos) + 1, 'text': data['text']}\n"
            "    todos.append(todo)\n"
            "    return jsonify(todo), 201\n\n"
            "@app.route('/todos', methods=['GET'])\n"
            "def list_todos():\n"
            "    \"\"\"List all to-do items.\"\"\"\n"
            "    return jsonify(todos)"
        ),
    },
]


def validate_curated_examples():
    """Self-check: every curated response must be valid Python, or the whole
    point of hand-curating high-quality examples is defeated."""
    for ex in CURATED_EXAMPLES:
        try:
            ast.parse(ex["response"])
        except SyntaxError as e:
            raise SystemExit(f"Curated example is invalid Python: {ex['instruction']!r}\n{e}")


# Bug-fixing examples: (buggy code -> fixed code) pairs, a task shape that
# didn't exist anywhere in the original dataset at all -- CodeAlpaca and the
# mined docstring pairs only ever taught "description -> new code", never
# "existing broken code -> corrected code". Without examples in this exact
# shape, the model has no learned behavior for this request type regardless
# of how good it is at writing fresh functions.
#
# Each entry includes a `check` -- actual Python that proves the "buggy"
# version really is wrong and the "fixed" version really is right, so these
# aren't just plausible-looking claims, they're verified regressions.
BUGFIX_EXAMPLES = [
    {
        "buggy": (
            "def get_average(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    average = total / len(nums)"
        ),
        "fixed": (
            "def get_average(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    average = total / len(nums)\n"
            "    return average"
        ),
        "explanation": "the function computes the average but never returns it",
        "check": lambda ns: ns["get_average"]([2, 4, 6]) == 4.0,
    },
    {
        "buggy": (
            "def add_item(item, basket=[]):\n"
            "    basket.append(item)\n"
            "    return basket"
        ),
        "fixed": (
            "def add_item(item, basket=None):\n"
            "    if basket is None:\n"
            "        basket = []\n"
            "    basket.append(item)\n"
            "    return basket"
        ),
        "explanation": "using a mutable list as a default argument means it's shared and "
                        "accumulates across calls instead of starting fresh each time",
        "check": lambda ns: ns["add_item"]("x") == ["x"] and ns["add_item"]("y") == ["y"],
    },
    {
        "buggy": (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n\n"
            "    def increment():\n"
            "        self.count += 1"
        ),
        "fixed": (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n\n"
            "    def increment(self):\n"
            "        self.count += 1"
        ),
        "explanation": "the increment method is missing the self parameter, so it can't "
                        "access the instance and will fail when called normally",
        "check": lambda ns: (lambda c: (c.increment(), c.count)[1])(ns["Counter"]()) == 1,
    },
    {
        "buggy": (
            "def find_max_index(nums):\n"
            "    max_val = nums[0]\n"
            "    max_idx = 0\n"
            "    for i in range(len(nums) - 1):\n"
            "        if nums[i] > max_val:\n"
            "            max_val = nums[i]\n"
            "            max_idx = i\n"
            "    return max_idx"
        ),
        "fixed": (
            "def find_max_index(nums):\n"
            "    max_val = nums[0]\n"
            "    max_idx = 0\n"
            "    for i in range(len(nums)):\n"
            "        if nums[i] > max_val:\n"
            "            max_val = nums[i]\n"
            "            max_idx = i\n"
            "    return max_idx"
        ),
        "explanation": "range(len(nums) - 1) skips the last index, so the actual maximum "
                        "is missed whenever it's the final element",
        "check": lambda ns: ns["find_max_index"]([1, 2, 9]) == 2,
    },
    {
        "buggy": (
            "def merge_sorted(l1, l2):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(l1) and j < len(l2):\n"
            "        if l1[i] < l2[j]:\n"
            "            result.append(l1[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(l2[j])\n"
            "            i += 1\n"
            "    result.extend(l1[i:])\n"
            "    result.extend(l2[j:])\n"
            "    return result"
        ),
        "fixed": (
            "def merge_sorted(l1, l2):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(l1) and j < len(l2):\n"
            "        if l1[i] < l2[j]:\n"
            "            result.append(l1[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(l2[j])\n"
            "            j += 1\n"
            "    result.extend(l1[i:])\n"
            "    result.extend(l2[j:])\n"
            "    return result"
        ),
        "explanation": "in the else branch, i is incremented instead of j, so j never "
                        "advances through l2 and elements get duplicated or lost",
        "check": lambda ns: ns["merge_sorted"]([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6],
    },
]


def validate_bugfix_examples():
    """Actually execute every buggy/fixed pair to prove the bug is real and
    the fix genuinely resolves it -- not just plausible-sounding claims."""
    for ex in BUGFIX_EXAMPLES:
        try:
            ast.parse(ex["fixed"])
        except SyntaxError as e:
            raise SystemExit(f"Bugfix 'fixed' version doesn't parse: {ex['explanation']!r}\n{e}")

        # the buggy version should run without crashing but give a wrong
        # (or for the missing-self case, an outright broken) result
        ns_fixed = {}
        exec(ex["fixed"], ns_fixed)
        if not ex["check"](ns_fixed):
            raise SystemExit(f"Bugfix 'fixed' version doesn't actually pass its own check: {ex['explanation']!r}")

        try:
            ns_buggy = {}
            exec(ex["buggy"], ns_buggy)
            buggy_passes = ex["check"](ns_buggy)
        except Exception:
            buggy_passes = False  # crashing also counts as "the bug is real"

        if buggy_passes:
            raise SystemExit(f"Bugfix example's 'buggy' version isn't actually buggy: {ex['explanation']!r}")


def build_bugfix_pairs():
    pairs = []
    for ex in BUGFIX_EXAMPLES:
        instruction = f"Fix the bug in this function:\n{ex['buggy']}"
        pairs.append({"instruction": instruction, "response": ex["fixed"]})
    return pairs


# Project-scaffolding examples: teaches two NEW task shapes that didn't
# exist in the dataset before -- (1) given a project description, produce a
# file/folder structure, and (2) given that structure plus one target
# filename, write that file's content. Neither response is any longer than
# what the model already handles well; the "multi-file project" capability
# is built at the ORCHESTRATION level (see 10_project_scaffold.py) by
# chaining several of these short, focused generations together -- the
# model itself never needs to hold an entire project in its 512-token
# context at once.
PROJECT_STRUCTURE_EXAMPLES = [
    {
        "instruction": "Give me a file structure for a Streamlit app that visualizes a CSV file.",
        "response": (
            "csv_visualizer/\n"
            "├── app.py              # main Streamlit entry point\n"
            "├── data_loader.py      # loads and cleans the CSV\n"
            "├── charts.py           # builds the charts shown in the app\n"
            "└── requirements.txt    # streamlit, pandas, plotly"
        ),
    },
    {
        "instruction": "Give me a file structure for a simple Flask REST API with a health check endpoint.",
        "response": (
            "flask_api/\n"
            "├── app.py              # Flask app factory and entry point\n"
            "├── routes.py           # API route definitions\n"
            "└── requirements.txt    # flask"
        ),
    },
    {
        "instruction": "Give me a file structure for a command-line tool that processes files using argparse.",
        "response": (
            "file_cli/\n"
            "├── main.py             # entry point, argument parsing\n"
            "├── processor.py        # core file-processing logic\n"
            "└── requirements.txt"
        ),
    },
    {
        "instruction": "Give me a file structure for a Streamlit app that lets users upload an image and apply filters.",
        "response": (
            "image_filter_app/\n"
            "├── app.py              # main Streamlit entry point\n"
            "├── filters.py          # image filter functions\n"
            "└── requirements.txt    # streamlit, pillow"
        ),
    },
    {
        "instruction": "Give me a file structure for a Python package that scrapes and stores webpage data.",
        "response": (
            "web_scraper/\n"
            "├── scraper.py          # scraping logic\n"
            "├── storage.py          # saves results to a file or database\n"
            "├── main.py             # ties scraper and storage together\n"
            "└── requirements.txt    # requests, beautifulsoup4"
        ),
    },
]


def validate_project_structure_examples():
    """Basic sanity check: every structure response should look like a
    plausible file listing -- non-empty, mentions at least one real file."""
    for ex in PROJECT_STRUCTURE_EXAMPLES:
        resp = ex["response"]
        if not resp.strip():
            raise SystemExit(f"Empty project structure example: {ex['instruction']!r}")
        if not any(ext in resp for ext in (".py", ".txt")):
            raise SystemExit(f"Project structure example has no recognizable files: {ex['instruction']!r}")


PROJECT_FILE_EXAMPLES = [
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[0]["response"],
        "filename": "app.py",
        "description": "the main Streamlit entry point that loads a CSV, shows it as a table, "
                        "and displays a line chart of a numeric column",
        "response": (
            "import streamlit as st\n"
            "from data_loader import load_csv\n"
            "from charts import line_chart\n\n"
            "st.title('CSV Visualizer')\n\n"
            "uploaded_file = st.file_uploader('Upload a CSV', type=['csv'])\n"
            "if uploaded_file is not None:\n"
            "    df = load_csv(uploaded_file)\n"
            "    st.dataframe(df)\n"
            "    column = st.selectbox('Column to chart', df.select_dtypes('number').columns)\n"
            "    line_chart(df, column)"
        ),
    },
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[0]["response"],
        "filename": "data_loader.py",
        "description": "loads a CSV file into a pandas DataFrame and drops empty rows",
        "response": (
            "import pandas as pd\n\n"
            "def load_csv(file):\n"
            "    \"\"\"Load a CSV file into a cleaned pandas DataFrame.\"\"\"\n"
            "    df = pd.read_csv(file)\n"
            "    df = df.dropna(how='all')\n"
            "    return df"
        ),
    },
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[0]["response"],
        "filename": "charts.py",
        "description": "shows a line chart of one column from a DataFrame using Streamlit",
        "response": (
            "import streamlit as st\n\n"
            "def line_chart(df, column):\n"
            "    \"\"\"Display a line chart of a single numeric column.\"\"\"\n"
            "    st.line_chart(df[column])"
        ),
    },
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[1]["response"],
        "filename": "app.py",
        "description": "the Flask app factory that registers the routes blueprint",
        "response": (
            "from flask import Flask\n"
            "from routes import bp\n\n"
            "def create_app():\n"
            "    \"\"\"Create and configure the Flask application.\"\"\"\n"
            "    app = Flask(__name__)\n"
            "    app.register_blueprint(bp)\n"
            "    return app\n\n"
            "if __name__ == '__main__':\n"
            "    create_app().run(debug=True)"
        ),
    },
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[1]["response"],
        "filename": "routes.py",
        "description": "defines a /health endpoint that returns a JSON status",
        "response": (
            "from flask import Blueprint, jsonify\n\n"
            "bp = Blueprint('routes', __name__)\n\n"
            "@bp.route('/health')\n"
            "def health():\n"
            "    \"\"\"Return a simple JSON health check response.\"\"\"\n"
            "    return jsonify({'status': 'ok'})"
        ),
    },
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[2]["response"],
        "filename": "main.py",
        "description": "the CLI entry point that parses a --input file argument and calls the processor",
        "response": (
            "import argparse\n"
            "from processor import process_file\n\n"
            "def main():\n"
            "    \"\"\"Parse arguments and run the file processor.\"\"\"\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--input', required=True, help='Path to the input file')\n"
            "    args = parser.parse_args()\n"
            "    process_file(args.input)\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        ),
    },
    {
        "plan": PROJECT_STRUCTURE_EXAMPLES[2]["response"],
        "filename": "processor.py",
        "description": "reads a file and prints the number of lines it contains",
        "response": (
            "def process_file(path):\n"
            "    \"\"\"Read a file and print how many lines it contains.\"\"\"\n"
            "    with open(path, 'r', encoding='utf-8') as f:\n"
            "        lines = f.readlines()\n"
            "    print(f'{path}: {len(lines)} lines')"
        ),
    },
    {
        # Fills a real gap: this template previously had a structure but no
        # verified file content, forcing slow uncapped generation for every
        # request that matched it. Closing this out makes the exact request
        # "build a streamlit app that lets users upload an image and apply
        # filters" resolve instantly via retrieval instead.
        "plan": (
            "image_filter_app/\n"
            "├── app.py              # main Streamlit entry point\n"
            "├── filters.py          # image filter functions\n"
            "└── requirements.txt    # streamlit, pillow"
        ),
        "filename": "app.py",
        "description": "the main Streamlit entry point that lets a user upload an image and apply "
                        "a filter, showing the result",
        "response": (
            "import streamlit as st\n"
            "from filters import grayscale, blur\n"
            "from PIL import Image\n\n"
            "st.title('Image Filter App')\n\n"
            "uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])\n"
            "if uploaded_file is not None:\n"
            "    image = Image.open(uploaded_file)\n"
            "    filter_choice = st.selectbox('Choose a filter', ['Grayscale', 'Blur'])\n"
            "    if filter_choice == 'Grayscale':\n"
            "        result = grayscale(image)\n"
            "    else:\n"
            "        result = blur(image)\n"
            "    st.image(result)"
        ),
    },
    {
        "plan": (
            "image_filter_app/\n"
            "├── app.py              # main Streamlit entry point\n"
            "├── filters.py          # image filter functions\n"
            "└── requirements.txt    # streamlit, pillow"
        ),
        "filename": "filters.py",
        "description": "grayscale and blur filter functions for a PIL image",
        "response": (
            "from PIL import ImageFilter\n\n"
            "def grayscale(image):\n"
            "    \"\"\"Convert an image to grayscale.\"\"\"\n"
            "    return image.convert('L')\n\n"
            "def blur(image):\n"
            "    \"\"\"Apply a blur filter to an image.\"\"\"\n"
            "    return image.filter(ImageFilter.BLUR)"
        ),
    },
    {
        # Same fix, for the other uncovered template (web_scraper) --
        # exactly the request that got stuck: "make a python package that
        # scrapes and stores webpage data".
        "plan": (
            "web_scraper/\n"
            "├── scraper.py          # scraping logic\n"
            "├── storage.py          # saves results to a file or database\n"
            "├── main.py             # ties scraper and storage together\n"
            "└── requirements.txt    # requests, beautifulsoup4"
        ),
        "filename": "scraper.py",
        "description": "scrapes all the text content of paragraph tags from a webpage",
        "response": (
            "def scrape_page(url):\n"
            "    \"\"\"Scrape all paragraph text from a webpage.\"\"\"\n"
            "    import requests\n"
            "    from bs4 import BeautifulSoup\n"
            "    response = requests.get(url)\n"
            "    soup = BeautifulSoup(response.text, 'html.parser')\n"
            "    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]\n"
            "    return paragraphs"
        ),
    },
    {
        "plan": (
            "web_scraper/\n"
            "├── scraper.py          # scraping logic\n"
            "├── storage.py          # saves results to a file or database\n"
            "├── main.py             # ties scraper and storage together\n"
            "└── requirements.txt    # requests, beautifulsoup4"
        ),
        "filename": "storage.py",
        "description": "saves a list of scraped strings to a text file, one per line",
        "response": (
            "def save_results(items, filepath):\n"
            "    \"\"\"Save a list of scraped strings to a text file, one per line.\"\"\"\n"
            "    with open(filepath, 'w', encoding='utf-8') as f:\n"
            "        for item in items:\n"
            "            f.write(item + '\\n')\n"
            "    return filepath"
        ),
    },
    {
        "plan": (
            "web_scraper/\n"
            "├── scraper.py          # scraping logic\n"
            "├── storage.py          # saves results to a file or database\n"
            "├── main.py             # ties scraper and storage together\n"
            "└── requirements.txt    # requests, beautifulsoup4"
        ),
        "filename": "main.py",
        "description": "scrapes a URL passed as a command-line argument and saves the results to output.txt",
        "response": (
            "import sys\n"
            "from scraper import scrape_page\n"
            "from storage import save_results\n\n"
            "def main():\n"
            "    \"\"\"Scrape the URL given on the command line and save the results.\"\"\"\n"
            "    url = sys.argv[1]\n"
            "    results = scrape_page(url)\n"
            "    save_results(results, 'output.txt')\n"
            "    print(f'Saved {len(results)} items to output.txt')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        ),
    },
]


def validate_project_file_examples():
    for ex in PROJECT_FILE_EXAMPLES:
        if ex["filename"].endswith(".py"):
            try:
                ast.parse(ex["response"])
            except SyntaxError as e:
                raise SystemExit(f"Project file example is invalid Python ({ex['filename']}): {e}")


def build_project_structure_pairs():
    return [{"instruction": ex["instruction"], "response": ex["response"]} for ex in PROJECT_STRUCTURE_EXAMPLES]


def build_project_file_pairs():
    pairs = []
    for ex in PROJECT_FILE_EXAMPLES:
        instruction = (
            f"Given this project structure:\n{ex['plan']}\n\n"
            f"Write the content of {ex['filename']}. It should: {ex['description']}"
        )
        pairs.append({"instruction": instruction, "response": ex["response"]})
    return pairs


def dedupe_pairs(pairs):
    seen = set()
    deduped = []
    for p in pairs:
        key = (p["instruction"], p["response"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def main():
    validate_curated_examples()
    validate_bugfix_examples()
    validate_project_structure_examples()
    validate_project_file_examples()

    all_pairs = load_codealpaca()
    all_pairs.extend(load_glaive_examples())
    all_pairs.extend(mine_docstring_pairs())

    curated_repeated = CURATED_EXAMPLES * CURATED_OVERSAMPLE
    all_pairs.extend(curated_repeated)
    print(f"Added {len(CURATED_EXAMPLES)} curated examples, oversampled {CURATED_OVERSAMPLE}x "
          f"({len(curated_repeated)} total curated entries)")

    bugfix_pairs = build_bugfix_pairs()
    bugfix_repeated = bugfix_pairs * CURATED_OVERSAMPLE
    all_pairs.extend(bugfix_repeated)
    print(f"Added {len(bugfix_pairs)} bug-fix examples (verified by execution), "
          f"oversampled {CURATED_OVERSAMPLE}x ({len(bugfix_repeated)} total entries)")

    structure_pairs = build_project_structure_pairs()
    structure_repeated = structure_pairs * CURATED_OVERSAMPLE
    all_pairs.extend(structure_repeated)
    print(f"Added {len(structure_pairs)} project-structure examples, oversampled {CURATED_OVERSAMPLE}x "
          f"({len(structure_repeated)} total entries)")

    file_pairs = build_project_file_pairs()
    file_repeated = file_pairs * CURATED_OVERSAMPLE
    all_pairs.extend(file_repeated)
    print(f"Added {len(file_pairs)} project-file-content examples, oversampled {CURATED_OVERSAMPLE}x "
          f"({len(file_repeated)} total entries)")

    before = len(all_pairs)
    all_pairs = dedupe_pairs(all_pairs)
    print(f"Deduped: {before} -> {len(all_pairs)} pairs")

    random.shuffle(all_pairs)

    out_path = os.path.join(OUT_DIR, "instruct_pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(all_pairs)} total instruction/response pairs to {out_path}")
    print("Next: run 07_instruction_tune.py")


if __name__ == "__main__":
    main()