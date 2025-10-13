import sqlite3
import os
import re
import json
import time
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
import google.generativeai as genai

# ================================
# CONFIGURATION
# ================================
load_dotenv()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    client = genai.GenerativeModel(GEMINI_MODEL)
else:
    client = None
    print("🔍 Gemini mode: MOCK/OFFLINE (no API key found)")

# ================================
# CACHE HELPERS
# ================================
def _cache_path(user_id: str, product_id: str) -> str:
    key = f"{user_id}_{product_id}".encode()
    return os.path.join(CACHE_DIR, hashlib.sha256(key).hexdigest()[:16] + ".json")

CACHE_TTL = 7 * 24 * 60 * 60  # 7 days

def load_from_cache(uid, pid):
    path = _cache_path(uid, pid)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > CACHE_TTL:
        os.remove(path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_to_cache(uid, pid, data):
    path = _cache_path(uid, pid)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ================================
# PRODUCT NAME LOOKUP
# ================================
def get_product_name(product_id: str) -> str:
    """Fetch product_name from SQLite db. Returns product_id if not found."""
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "product_catalog.db")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT product_name FROM products WHERE product_id = ?", (product_id,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
    except Exception as e:
        print(f"⚠️ DB lookup failed for {product_id}: {e}")
    return product_id  # fallback

# ================================
# MAIN FUNCTION
# ================================
def generate_llm_explanation(user_id, product, interactions_df, products_df, use_cache=True):
    # 1️⃣ Cache
    if use_cache:
        cached = load_from_cache(user_id, product["product_id"])
        if cached:
            return cached

    # 2️⃣ Summarize user activity
    user_data = interactions_df[interactions_df["user_id"] == user_id].copy()
    if user_data.empty:
        user_summary = "No recent user behavior found."
    else:
        user_data = user_data.sort_values("timestamp", ascending=False).head(5)
        actions = [f"{r.event_type} product {r.product_id}" for r in user_data.itertuples()]
        user_summary = "Recent actions: " + ", ".join(actions)

    # 3️⃣ Product info
    row = products_df.loc[products_df["product_id"] == product["product_id"]]
    if row.empty:
        product_info = {"title": "Unknown Product"}
    else:
        row = row.iloc[0]
        product_info = {
            "title": row.get("title", "Unknown"),
            "category": row.get("category", "Unknown"),
            "brand": row.get("brand", "Unknown"),
            "price": row.get("price", "N/A"),
        }

    # 4️⃣ Prompt
    prompt = f"""
        Generate a 2–3 sentence JSON explanation for why this product is recommended.

        User behavior:
        {user_summary}

        Product:
        Title: {product_info.get('title','N/A')}
        Category: {product_info.get('category','N/A')}
        Brand: {product_info.get('brand','N/A')}
        Price: {product_info.get('price','N/A')}

        Return JSON only:
        {{"explanation": "...", "evidence": ["...", "..."]}}
        """

    # 5️⃣ Call Gemini or fallback
    if client:
        try:
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 0.7},
                request_options={"timeout": 30}
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z]*", "", text)
                text = re.sub(r"```$", "", text)
                text = text.strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"explanation": text, "evidence": []}
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            parsed = {
                "explanation": "Explanation temporarily unavailable.",
                "evidence": []
            }
    else:
        # Mock/offline mode
        recent_action = user_data.iloc[0]["event_type"] if not user_data.empty else "view"
        parsed = {
            "explanation": f"This product is recommended because you recently {recent_action} similar items.",
            "evidence": [recent_action],
        }

    # 6️⃣ Replace product IDs with product names
    def _replace_ids_with_names(obj):
        if isinstance(obj, str):
            return re.sub(
                r"P\d{5,}",
                lambda m: get_product_name(m.group(0)),
                obj
            )
        elif isinstance(obj, list):
            return [_replace_ids_with_names(item) for item in obj]
        return obj

    parsed["explanation"] = _replace_ids_with_names(parsed.get("explanation", ""))
    parsed["evidence"] = _replace_ids_with_names(parsed.get("evidence", []))

    # 7️⃣ Metadata + cache
    parsed.update({
        "product_id": product["product_id"],
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
    })

    if use_cache:
        save_to_cache(user_id, product["product_id"], parsed)

    return parsed
