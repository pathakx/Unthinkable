import os
import sqlite3
import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta
from contextlib import closing
from pathlib import Path


# -------------------------------
# CONFIGURATION
# -------------------------------

DB_PATH = Path("data/user_interactions.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
TABLE_NAME = "interactions"

NUM_CUSTOMERS = 100
NUM_PRODUCTS = 270
NUM_INTERACTIONS = 2000

EVENT_TYPES = ["view", "add_to_cart", "purchase"]
EVENT_PROBABILITIES = [0.6, 0.25, 0.15]

# -------------------------------
# SETUP LOGGING
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------
# STEP 1: GENERATE SYNTHETIC DATA
# -------------------------------
def generate_interaction_data(num_customers, num_products, num_interactions):
    """Generate realistic user–product interaction data."""
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
    product_ids = [f"P{str(i).zfill(5)}" for i in range(1, num_products + 1)]

    def random_timestamp():
        days_ago = random.randint(0, 90)
        random_time = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        return random_time.strftime("%Y-%m-%d %H:%M:%S")

    data = [
        {
            "user_id": random.choice(customer_ids),
            "product_id": random.choice(product_ids),
            "event_type": random.choices(EVENT_TYPES, weights=EVENT_PROBABILITIES, k=1)[0],
            "timestamp": random_timestamp()
        }
        for _ in range(num_interactions)
    ]

    df = pd.DataFrame(data)
    df.sort_values(by=["user_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Generated {len(df)} interaction records for {num_customers} users and {num_products} products.")
    return df


# -------------------------------
# STEP 2: INITIALIZE DATABASE
# -------------------------------
def initialize_database(db_path):
    """Create SQLite DB file and interaction table with proper schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            event_type TEXT CHECK(event_type IN ('view', 'add_to_cart', 'purchase')) NOT NULL,
            timestamp DATETIME NOT NULL
        );
        """)
        conn.commit()
    logging.info(f"Database initialized at {db_path} with table '{TABLE_NAME}'.")


# -------------------------------
# STEP 3: STORE INTERACTIONS
# -------------------------------
def store_interactions(df, db_path):
    """Insert interaction data into SQLite table."""
    with closing(sqlite3.connect(db_path)) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
    logging.info(f"Inserted {len(df)} new interaction records into {TABLE_NAME}.")


# -------------------------------
# STEP 4: LOAD INTERACTIONS (optional)
# -------------------------------
def load_interactions(db_path, user_id=None):
    """Load interactions for all users or a specific user."""
    with closing(sqlite3.connect(db_path)) as conn:
        if user_id:
            query = f"SELECT * FROM {TABLE_NAME} WHERE user_id = ? ORDER BY timestamp"
            df = pd.read_sql(query, conn, params=(user_id,))
            logging.info(f"Loaded {len(df)} interactions for user {user_id}.")
        else:
            query = f"SELECT * FROM {TABLE_NAME}"
            df = pd.read_sql(query, conn)
            logging.info(f"Loaded all {len(df)} interaction records from database.")
    return df


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    logging.info("=== Starting User Interaction Pipeline ===")

    # 1️⃣ Generate synthetic user–product interaction data
    interactions_df = generate_interaction_data(NUM_CUSTOMERS, NUM_PRODUCTS, NUM_INTERACTIONS)

    # 2️⃣ Initialize SQLite database (create folder + table)
    initialize_database(DB_PATH)

    # 3️⃣ Store the generated interactions
    store_interactions(interactions_df, DB_PATH)

    # 4️⃣ (Optional) Load one user’s interactions for verification
    sample_user = random.choice(interactions_df["user_id"].unique())
    user_df = load_interactions(DB_PATH, sample_user)
    print(user_df.head())

    logging.info("✅ Pipeline completed successfully.")
