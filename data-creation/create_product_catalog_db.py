"""
create_product_catalog_db.py
---------------------------------------------------------
Industry-level SQLite schema for an e-commerce Product Catalog.

Features:
- Creates a robust 'products' table for multi-category catalogs
- Handles flexible JSON-based 'features' for category-specific attributes
- Provides reusable helper functions for connecting, inserting, and querying
- Compatible with Gemini-based product data generation pipeline
---------------------------------------------------------
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------
# 1. DATABASE CONFIGURATION
# ---------------------------------------------------------------------

DB_PATH = Path("data/product_catalog.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Create or return a SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


# ---------------------------------------------------------------------
# 2. TABLE CREATION
# ---------------------------------------------------------------------

def create_tables():
    """Create the products table with category-flexible fields."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        brand TEXT,
        about_product TEXT,
        actual_price REAL,
        discounted_price REAL,
        discount_percentage REAL,
        rating REAL,
        rating_count INTEGER,
        img_link TEXT,
        product_link TEXT,
        features TEXT,              -- JSON string for category-specific attributes
        tags TEXT,                  -- comma-separated or keyword tags
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("✅ Products table created successfully.")


# ---------------------------------------------------------------------
# 3. INSERTION LOGIC
# ---------------------------------------------------------------------

def insert_product(product: dict):
    """
    Insert or replace a single product record into the database.

    Args:
        product (dict): Dictionary containing all product fields.
    """
    required_fields = ["product_id", "product_name", "category"]
    for field in required_fields:
        if field not in product or not product[field]:
            raise ValueError(f"Missing required field: {field}")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO products (
        product_id, product_name, category, brand, about_product,
        actual_price, discounted_price, discount_percentage,
        rating, rating_count, img_link, product_link,
        features, tags, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        product["product_id"],
        product["product_name"],
        product["category"],
        product.get("brand"),
        product.get("about_product"),
        product.get("actual_price"),
        product.get("discounted_price"),
        product.get("discount_percentage"),
        product.get("rating"),
        product.get("rating_count"),
        product.get("img_link"),
        product.get("product_link"),
        json.dumps(product.get("features", {})),
        product.get("tags", ""),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# 4. QUERY UTILITIES
# ---------------------------------------------------------------------

def get_all_products(limit=10):
    """Return a list of all products (limited by `limit`)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products LIMIT ?;", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def search_by_category(category_keyword: str):
    """Search products by partial category match."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE category LIKE ?;", (f"%{category_keyword}%",))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_product_by_id(product_id: str):
    """Fetch a single product by its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE product_id = ?;", (product_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------
# 5. SAMPLE DATA (Optional for testing)
# ---------------------------------------------------------------------

SAMPLE_PRODUCTS = [
    {
        "product_id": "P101",
        "product_name": "Sony WH-1000XM5",
        "category": "Electronics > Headphones",
        "brand": "Sony",
        "about_product": "Wireless noise-cancelling headphones with 30-hour battery life.",
        "actual_price": 29999,
        "discounted_price": 24999,
        "discount_percentage": 16.7,
        "rating": 4.6,
        "rating_count": 1200,
        "img_link": "https://example.com/images/sony.jpg",
        "product_link": "https://example.com/products/sonywh1000xm5",
        "features": {
            "battery_life": "30 hours",
            "connectivity": "Bluetooth 5.2",
            "noise_cancellation": "Active",
            "color": "Black",
            "warranty": "1 year"
        },
        "tags": "wireless,noise-cancelling,bluetooth,audio"
    },
    {
        "product_id": "P202",
        "product_name": "Levi’s Slim Fit Shirt",
        "category": "Fashion > Men’s T-Shirts",
        "brand": "Levi’s",
        "about_product": "Slim fit cotton casual shirt for men.",
        "actual_price": 1999,
        "discounted_price": 1499,
        "discount_percentage": 25.0,
        "rating": 4.2,
        "rating_count": 450,
        "img_link": "https://example.com/images/levis.jpg",
        "product_link": "https://example.com/products/levis-shirt",
        "features": {
            "fabric": "100% Cotton",
            "fit_type": "Slim",
            "available_sizes": ["M", "L", "XL"],
            "color": "Blue",
            "pattern": "Solid"
        },
        "tags": "clothing,shirt,casual,cotton"
    }
]


def insert_sample_data():
    """Insert example sample products for quick testing."""
    for product in SAMPLE_PRODUCTS:
        insert_product(product)
    print(f"✅ Inserted {len(SAMPLE_PRODUCTS)} sample products.")


# ---------------------------------------------------------------------
# 6. MAIN EXECUTION (for standalone use)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("🧱 Setting up product catalog database...")
    create_tables()

    # Uncomment the next line to insert sample data for testing
    # insert_sample_data()

    print("✅ Database ready at:", DB_PATH)
