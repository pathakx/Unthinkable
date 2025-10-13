#!/usr/bin/env python3
"""
Generate dense embeddings for products to power content-based recommendations.

Reads from a SQLite DB (products table must include the columns from your schema),
builds descriptive text from all attributes, encodes with sentence-transformers,
and writes a Parquet dataset plus a manifest file.

Usage:
  python scripts/generate_embeddings.py \
    --db data/product_catalog.db \
    --table products \
    --out data/embeddings/products.parquet \
    --model all-MiniLM-L6-v2 \
    --batch-size 256 \
    --chunksize 5000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ----------------------------- Logging --------------------------------------- #
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ----------------------------- Config ---------------------------------------- #
@dataclass(frozen=True)
class RunConfig:
    db_path: str
    table: str
    out_parquet: str
    model_name: str
    batch_size: int
    chunksize: int
    device: Optional[str]
    text_max_len: int
    fail_on_json_error: bool


# ----------------------------- Utils ----------------------------------------- #
def md5_of_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def safe_parse_features(raw: Optional[str], fail_on_error: bool = False) -> Dict[str, str]:
    """
    Parse JSON features column. Returns {} on empty/invalid unless fail_on_error=True.
    """
    if not raw or str(raw).strip() in ("", "null", "None"):
        return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {
                str(k): normalize_space(json.dumps(v)) if not isinstance(v, (str, int, float, bool)) else str(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            flattened = {}
            for item in obj:
                if isinstance(item, dict):
                    for k, v in item.items():
                        flattened[str(k)] = str(v)
            return flattened
        if fail_on_error:
            raise ValueError("features JSON is neither dict nor list")
        return {}
    except Exception as e:
        if fail_on_error:
            raise
        logging.debug("Failed to parse features JSON: %s | raw=%s", e, raw)
        return {}


# ----------------------------- Text Builder ---------------------------------- #
def build_text(row: pd.Series, text_max_len: int) -> str:
    """
    Compose a rich descriptive text for embedding using all relevant product fields.
    """
    parts: List[str] = []

    # Core identifying fields
    if pd.notna(row.get("product_name")):
        parts.append(f"Product Name: {row['product_name']}")
    if pd.notna(row.get("brand")) and str(row["brand"]).strip():
        parts.append(f"Brand: {row['brand']}")
    if pd.notna(row.get("category")):
        parts.append(f"Category: {row['category']}")
    # if pd.notna(row.get("main_category")):
    #     parts.append(f"Main Category: {row['main_category']}")
    # if pd.notna(row.get("sub_category")):
    #     parts.append(f"Sub Category: {row['sub_category']}")

    # Description
    if pd.notna(row.get("about_product")):
        parts.append(f"Description: {normalize_space(row['about_product'])}")

    # Pricing information
    if pd.notna(row.get("actual_price")) and float(row["actual_price"]) > 0:
        parts.append(f"Actual Price: {row['actual_price']}")
    if pd.notna(row.get("discounted_price")) and float(row["discounted_price"]) > 0:
        parts.append(f"Discounted Price: {row['discounted_price']}")
    if pd.notna(row.get("discount_percentage")) and float(row["discount_percentage"]) > 0:
        parts.append(f"Discount: {row['discount_percentage']}%")

    # Ratings
    if pd.notna(row.get("rating")) and float(row["rating"]) > 0:
        parts.append(f"Rating: {row['rating']}/5")
    if pd.notna(row.get("rating_count")) and int(row["rating_count"]) > 0:
        parts.append(f"Rated by {int(row['rating_count'])} customers")

    # Tags and features
    if pd.notna(row.get("tags")) and str(row["tags"]).strip():
        parts.append(f"Tags: {row['tags']}")

    feats = safe_parse_features(row.get("features"))
    if feats:
        feat_str = " ".join(f"{k}: {v}" for k, v in feats.items())
        parts.append(f"Features: {feat_str}")

    # Optional: URLs
    if pd.notna(row.get("product_link")) and str(row["product_link"]).strip():
        parts.append(f"Product Link: {row['product_link']}")
    if pd.notna(row.get("img_link")) and str(row["img_link"]).strip():
        parts.append(f"Image Link: {row['img_link']}")

    # Final join and truncation
    text = normalize_space(" ".join(parts))
    if text_max_len and len(text) > text_max_len:
        text = text[:text_max_len]

    return text


# ----------------------------- Database Reader ------------------------------- #
def read_in_chunks(conn: sqlite3.Connection, table: str, chunksize: int) -> Iterable[pd.DataFrame]:
    """
    Read the products table in chunks.
    Includes all expected columns for embedding generation.
    """
    sql = f"""
        SELECT
            product_id, product_name, brand, about_product,
            category,
            actual_price, discounted_price, discount_percentage,
            rating, rating_count, tags, features,
            img_link, product_link
        FROM {table}
    """
    if chunksize > 0:
        for chunk in pd.read_sql_query(sql, conn, chunksize=chunksize):
            yield chunk
    else:
        yield pd.read_sql_query(sql, conn)


# ----------------------------- Model & Embedding ----------------------------- #
def load_model(name: str, device: Optional[str]) -> SentenceTransformer:
    logging.info("Loading model %s ...", name)
    model = SentenceTransformer(name)
    if device:
        model = model.to(device)
        logging.info("Moved model to device: %s", device)
    return model


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


# ----------------------------- Output Writers -------------------------------- #
def write_parquet(df: pd.DataFrame, out_path: str) -> None:
    ensure_parent_dir(out_path)
    df.to_parquet(out_path, index=False)
    logging.info("Wrote Parquet: %s (%d rows)", out_path, len(df))


def write_manifest(out_parquet: str, cfg: RunConfig, dim: int, count: int, sample_md5: str) -> None:
    manifest = {
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_db": os.path.abspath(cfg.db_path),
        "table": cfg.table,
        "parquet_path": os.path.abspath(out_parquet),
        "model": cfg.model_name,
        "device": cfg.device,
        "embedding_dim": dim,
        "row_count": count,
        "batch_size": cfg.batch_size,
        "chunksize": cfg.chunksize,
        "text_max_len": cfg.text_max_len,
        "sample_embedding_md5": sample_md5,
        "version": 2,
    }
    manifest_path = os.path.splitext(out_parquet)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Wrote manifest: %s", manifest_path)


# ----------------------------- Main ------------------------------------------ #
def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(description="Generate product embeddings from SQLite product catalog.")
    p.add_argument("--db", dest="db_path", required=True, help="Path to SQLite DB (e.g., data/product_catalog.db)")
    p.add_argument("--table", default="products", help="Source table name (default: products)")
    p.add_argument("--out", dest="out_parquet", required=True, help="Output Parquet path (e.g., data/embeddings/products.parquet)")
    p.add_argument("--model", dest="model_name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    p.add_argument("--chunksize", type=int, default=0, help="Rows per DB chunk (0 = load all)")
    p.add_argument("--device", type=str, default=None, help='Force device (e.g., "cuda", "cpu"). Default: auto')
    p.add_argument("--text-max-len", type=int, default=2000, help="Truncate composed text to N chars")
    p.add_argument("--fail-on-json-error", action="store_true", help="Raise if features JSON is malformed")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    return RunConfig(
        db_path=args.db_path,
        table=args.table,
        out_parquet=args.out_parquet,
        model_name=args.model_name,
        batch_size=args.batch_size,
        chunksize=args.chunksize,
        device=args.device,
        text_max_len=args.text_max_len,
        fail_on_json_error=args.fail_on_json_error,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    logging.info("Config: %s", cfg)

    if not os.path.exists(cfg.db_path):
        logging.error("SQLite DB not found: %s", cfg.db_path)
        return 2

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row

    model = load_model(cfg.model_name, cfg.device)

    frames: List[pd.DataFrame] = []
    total_rows = 0

    with conn:
        for chunk in read_in_chunks(conn, cfg.table, cfg.chunksize):
            if chunk.empty:
                continue

            texts = [build_text(row, cfg.text_max_len) for _, row in chunk.iterrows()]

            slim = chunk[["product_id", "product_name", "category"]].copy()
            slim["text"] = texts

            embeddings_list: List[np.ndarray] = []
            for i in tqdm(range(0, len(texts), cfg.batch_size), disable=False, desc="Encoding"):
                batch = texts[i : i + cfg.batch_size]
                emb = encode_texts(model, batch, batch_size=cfg.batch_size)
                embeddings_list.append(emb)

            embeddings = np.vstack(embeddings_list)
            slim["embedding"] = [embeddings[i].astype(np.float32).tolist() for i in range(len(slim))]

            frames.append(slim)
            total_rows += len(slim)
            logging.info("Processed chunk: %d rows (total so far: %d)", len(slim), total_rows)

    if not frames:
        logging.error("No rows read from table=%s. Aborting.", cfg.table)
        return 3

    df = pd.concat(frames, ignore_index=True)

    if df["embedding"].empty or not isinstance(df["embedding"].iloc[0], list):
        logging.error("Embeddings missing or malformed.")
        return 4

    write_parquet(df, cfg.out_parquet)

    first_vec = np.array(df["embedding"].iloc[0], dtype=np.float32)
    dim = int(first_vec.shape[0])
    sample_md5 = md5_of_bytes(first_vec.tobytes())
    write_manifest(cfg.out_parquet, cfg, dim=dim, count=len(df), sample_md5=sample_md5)

    logging.info("✅ Saved embeddings for %d products. dim=%d", len(df), dim)
    print(f"✅ Saved embeddings for {len(df)} products. dim={dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
