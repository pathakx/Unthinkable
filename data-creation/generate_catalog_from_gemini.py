"""
generate_catalog_from_gemini.py
---------------------------------------------------------
End-to-end product catalog generator using Gemini + SQLite.

Features:
✅ Uses Gemini to generate synthetic e-commerce products
✅ Stores data in a local JSON cache (avoids repeat API calls)
✅ Inserts all products into the SQLite database automatically
✅ Supports multi-domain categories and subcategories
✅ Generates globally unique product IDs
---------------------------------------------------------
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Import your DB helper
from create_product_catalog_db import create_tables, insert_product

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
CACHE_FILE = DATA_DIR / "generated_cache.json"

PRODUCTS_PER_CATEGORY = 10      # adjust for testing
DELAY_SECONDS = 3               # delay between API calls

# ---------------------------------------------------------------------
# 2. CATEGORY LIST
# ---------------------------------------------------------------------
CATEGORIES = [
    # 🎧 Electronics
    "Electronics > Headphones",
    "Electronics > Smartphones",
    "Electronics > Laptops",
    "Electronics > Smartwatches",

    # 👕 Fashion
    "Fashion > Men’s T-Shirts",
    "Fashion > Women’s Dresses",
    "Fashion > Footwear",
    "Fashion > Watches",

    # 🏠 Home & Kitchen
    "Home Appliances > Kitchen Appliances",
    "Home Appliances > Cleaning Devices",
    "Home Decor > Lighting",
    "Home Decor > Furniture",

    # 💄 Beauty & Personal Care
    "Beauty & Personal Care > Skincare",
    "Beauty & Personal Care > Haircare",
    "Beauty & Personal Care > Makeup",

    # 📚 Books
    "Books > Self-help",
    "Books > Fiction",
    "Books > Academic",

    # 🏋️ Sports & Fitness
    "Sports & Fitness > Gym Equipment",
    "Sports & Fitness > Yoga Accessories",
    "Sports & Fitness > Sportswear",

    # 🎲 Toys & Games
    "Toys & Games > Educational Toys",
    "Toys & Games > Board Games",
    "Toys & Games > Outdoor Toys",

    # 🛒 Groceries
    "Groceries > Beverages",
    "Groceries > Snacks",
    "Groceries > Organic Foods"
]

# ---------------------------------------------------------------------
# 3. LOAD / SAVE CACHE
# ---------------------------------------------------------------------
def load_cache() -> dict:
    """Load existing product cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
                print(f"♻️  Loaded cached data for {len(cache)} categories.")
                return cache
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
    return {}

def save_cache(cache: dict):
    """Save the current cache back to disk."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print("💾 Cache saved to disk.")

generated_cache = load_cache()

# ---------------------------------------------------------------------
# 4. GEMINI PRODUCT GENERATION
# ---------------------------------------------------------------------
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_products_for_category(category: str, n: int = 10):
    """Generate dummy product data for a category using Gemini."""
    if category in generated_cache:
        print(f"♻️  Using cached products for {category}")
        return generated_cache[category]

    print(f"🚀 Generating {n} products for {category} ...")

    prompt = f"""
    You are a product data generator for an e-commerce site.

    Generate {n} realistic, diverse products for the category "{category}".
    Each product must be an object inside a JSON array with fields:
    - product_name
    - brand
    - about_product (1–2 lines)
    - actual_price (integer)
    - discounted_price (integer)
    - discount_percentage (float)
    - rating (float 3.5–5.0)
    - rating_count (integer)
    - features (3–5 key:value pairs relevant to this category)
    - img_link (dummy URL)
    - product_link (dummy URL)
    Output only valid JSON.
    """

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Extract JSON block
    start = text.find("[")
    end = text.rfind("]") + 1
    text = text[start:end]

    try:
        products = json.loads(text)
        generated_cache[category] = products
        save_cache(generated_cache)
        time.sleep(DELAY_SECONDS)
        return products
    except json.JSONDecodeError:
        print(f"⚠️  JSON parsing failed for {category}. Skipping...")
        return []

# ---------------------------------------------------------------------
# 5. DATABASE INSERTION LOGIC
# ---------------------------------------------------------------------
global_product_counter = 1

def save_products_to_db(category: str, products: list):
    """Insert generated products into the SQLite database."""
    global global_product_counter
    main, sub = [p.strip() for p in category.split(">")]

    for prod in products:
        try:
            product = {
                "product_id": f"P{global_product_counter:05d}",
                "product_name": prod.get("product_name", "Unnamed Product"),
                "category": f"{main} > {sub}",
                "main_category": main,
                "sub_category": sub,
                "brand": prod.get("brand", ""),
                "about_product": prod.get("about_product", ""),
                "actual_price": prod.get("actual_price", 0),
                "discounted_price": prod.get("discounted_price", 0),
                "discount_percentage": prod.get("discount_percentage", 0),
                "rating": prod.get("rating", 0),
                "rating_count": prod.get("rating_count", 0),
                "img_link": prod.get("img_link", ""),
                "product_link": prod.get("product_link", ""),
                "features": prod.get("features", {}),
                "tags": sub
            }
            insert_product(product)
            global_product_counter += 1
        except Exception as e:
            print(f"❌ Failed to insert product in {category}: {e}")

# ---------------------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🧠 Starting Product Catalog Generation...\n")
    start_time = datetime.now()

    # Ensure database exists
    create_tables()

    total_products = 0

    for category in CATEGORIES:
        products = generate_products_for_category(category, n=PRODUCTS_PER_CATEGORY)
        if products:
            save_products_to_db(category, products)
            total_products += len(products)
            print(f"✅ {len(products)} products inserted for {category}\n")
        else:
            print(f"⚠️ No products generated for {category}\n")

    print("🎉 Catalog generation complete!")
    print(f"🛒 Total products added: {total_products}")
    print(f"🕒 Duration: {datetime.now() - start_time}")
