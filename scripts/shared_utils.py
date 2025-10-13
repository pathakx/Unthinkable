import os
import sqlite3
import numpy as np
import pandas as pd
import faiss
from datetime import datetime

# ================================
# CONFIG
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings", "products.parquet")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "user_interactions.parquet")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "user_interactions.db")
USER_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "user_embeddings.parquet")

EVENT_WEIGHTS = {"purchase": 5, "add_to_cart": 3, "view": 1}

# ================================
# ENSURE INTERACTION FILE
# ================================
def ensure_user_interactions_parquet():
    """
    Ensures that the Parquet file for user interactions exists.
    If not, it loads from the SQLite database and exports it.
    """
    if os.path.exists(INTERACTIONS_PATH):
        return
    if not os.path.exists(SQLITE_DB_PATH):
        raise FileNotFoundError(f"❌ SQLite database not found at {SQLITE_DB_PATH}")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    query = "SELECT user_id, product_id, event_type, timestamp FROM interactions"
    df = pd.read_sql(query, conn)
    conn.close()
    os.makedirs(os.path.dirname(INTERACTIONS_PATH), exist_ok=True)
    df.to_parquet(INTERACTIONS_PATH, index=False)
    print(f"✅ Exported {len(df)} interactions to {INTERACTIONS_PATH}")

# ================================
# LOAD PRODUCT EMBEDDINGS
# ================================
def load_product_embeddings(path=EMBEDDINGS_PATH):
    df = pd.read_parquet(path)
    if "embedding" not in df.columns:
        raise ValueError("❌ Missing 'embedding' column in product embeddings file.")
    df["embedding"] = df["embedding"].apply(np.array)
    embeddings = np.vstack(df["embedding"].to_numpy())
    product_ids = df["product_id"].astype(str).tolist()
    return product_ids, embeddings

# ================================
# EVENT PROBABILITY (optional use)
# ================================
def compute_event_probabilities(df, decay_rate=0.1):
    df = df.copy()
    now = pd.Timestamp.now()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["delta_days"] = (now - df["timestamp"]).dt.total_seconds() / (3600 * 24)
    df["base_weight"] = df["event_type"].map(EVENT_WEIGHTS).fillna(1)
    df["decay_weight"] = df["base_weight"] * np.exp(-decay_rate * df["delta_days"])
    df["probability"] = df["decay_weight"] / df["decay_weight"].sum()
    return df

# ================================
# USER EMBEDDING COMPUTATION
# ================================
def compute_user_embedding(user_id, interactions_df, product_embeddings, product_id_to_index, apply_time_decay=True):
    user_data = interactions_df[interactions_df["user_id"] == user_id]
    if user_data.empty:
        raise ValueError(f"No interactions found for user {user_id}")
    weights = [EVENT_WEIGHTS.get(e, 1) for e in user_data["event_type"]]
    if apply_time_decay:
        timestamps = pd.to_datetime(user_data["timestamp"]).astype("int64") / 1e9
        now = timestamps.max()
        decay_rate = 0.05
        weights = [w * np.exp(-decay_rate * (now - t)) for w, t in zip(weights, timestamps)]
    vectors = [product_embeddings[product_id_to_index[pid]] for pid in user_data["product_id"]]
    weighted_vectors = [v * w for v, w in zip(vectors, weights)]
    return np.sum(weighted_vectors, axis=0) / np.sum(weights)

# ================================
# USER EMBEDDING CACHE HELPERS
# ================================
def _read_user_cache():
    if os.path.exists(USER_EMBEDDINGS_PATH):
        df = pd.read_parquet(USER_EMBEDDINGS_PATH)
        df["user_id"] = df["user_id"].astype(str)
        return df
    return pd.DataFrame(columns=["user_id", "embedding", "last_event_ts"])

def _write_user_cache(df):
    os.makedirs(os.path.dirname(USER_EMBEDDINGS_PATH), exist_ok=True)
    df.to_parquet(USER_EMBEDDINGS_PATH, index=False)

def get_or_build_user_embedding(user_id, apply_time_decay=True, decay_rate=0.05, force=False):
    """
    Returns a cached user embedding if fresh, otherwise recomputes it.
    Automatically stores embeddings in data/user_embeddings.parquet.
    """
    ensure_user_interactions_parquet()
    interactions_df = pd.read_parquet(INTERACTIONS_PATH)
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    user_df = interactions_df[interactions_df["user_id"] == user_id].copy()
    if user_df.empty:
        raise ValueError(f"No interactions found for user {user_id}")
    last_event_ts = user_df["timestamp"].max()

    # Load cache
    cache_df = _read_user_cache()
    cached_row = cache_df[cache_df["user_id"] == str(user_id)]
    if not force and not cached_row.empty:
        cached_ts = pd.to_datetime(cached_row.iloc[0]["last_event_ts"])
        if cached_ts >= last_event_ts:
            emb = np.array(cached_row.iloc[0]["embedding"], dtype=np.float32)
            return emb

    # Recompute embedding
    product_ids, product_embeddings = load_product_embeddings()
    product_id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}
    user_vector = compute_user_embedding(user_id, interactions_df, product_embeddings, product_id_to_index, apply_time_decay)
    rec = {
        "user_id": str(user_id),
        "embedding": user_vector.tolist(),
        "last_event_ts": last_event_ts.isoformat()
    }

    # Update cache
    if cache_df.empty:
        cache_df = pd.DataFrame([rec])
    else:
        mask = cache_df["user_id"] == str(user_id)
        if mask.any():
            cache_df.loc[mask, ["embedding", "last_event_ts"]] = [rec["embedding"], rec["last_event_ts"]]
        else:
            cache_df = pd.concat([cache_df, pd.DataFrame([rec])], ignore_index=True)
    _write_user_cache(cache_df)

    print(f"✅ User embedding updated for {user_id}")
    return user_vector

# ================================
# FAISS INDEX BUILDING
# ================================
def build_faiss_index(embeddings):
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
