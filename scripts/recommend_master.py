import os
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

from .recommend_view_based import recommend_view_based
from .recommend_cart_based import recommend_cart_based
from .recommend_purchase_based import recommend_purchase_based
from .recommend_profile_based import recommend_profile_based
from .shared_utils import (
    ensure_user_interactions_parquet,
    load_product_embeddings,
)
from .recommend_explain_llm import generate_llm_explanation


# ================================
# MAIN PIPELINE
# ================================
def recommend_for_user(user_id):
    ensure_user_interactions_parquet()
    product_ids, product_embeddings = load_product_embeddings()



    INTERACTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "user_interactions.parquet"
    PRODUCTS_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "products.parquet"
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "product_catalog.db"

    interactions_df = pd.read_parquet(INTERACTIONS_PATH)
    products_df = pd.read_parquet(PRODUCTS_PATH)

    product_id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

    rec_dict = {}

    def add_recs(new_recs):
        for r in new_recs:
            pid = r["product_id"]
            if pid not in rec_dict or r["score"] > rec_dict[pid]["score"]:
                rec_dict[pid] = r

    # 1️⃣ View-based
    view_recs = recommend_view_based(user_id, interactions_df, product_embeddings, product_ids, product_id_to_index)
    add_recs(view_recs)

    # 2️⃣ Cart-based
    cart_recs = recommend_cart_based(user_id, interactions_df, product_embeddings, product_ids, product_id_to_index)
    add_recs(cart_recs)

    # 3️⃣ Purchase-based
    purchase_recs = recommend_purchase_based(user_id, interactions_df, product_embeddings, product_ids, product_id_to_index)
    add_recs(purchase_recs)

    # 4️⃣ Profile-based
    seen_products = set(rec_dict.keys())
    profile_recs = recommend_profile_based(user_id, interactions_df, product_embeddings, product_ids, product_id_to_index, seen_products=seen_products)
    add_recs(profile_recs)

    recommendations = sorted(rec_dict.values(), key=lambda x: x["score"], reverse=True)[:5]

    # 5️⃣ Generate LLM explanations
    final_output = []
    for rec in recommendations:
        explanation = generate_llm_explanation(
            user_id=user_id,
            product=rec,
            interactions_df=interactions_df,
            products_df=products_df,
        )
        rec["explanation"] = explanation.get("explanation", "")
        rec["evidence"] = explanation.get("evidence", [])
        final_output.append(rec)

    # 6️⃣ Attach product names from SQLite
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            for rec in final_output:
                cur.execute("SELECT product_name FROM products WHERE product_id = ?", (rec["product_id"],))
                row = cur.fetchone()
                rec["product_name"] = row[0] if row and row[0] else "Unknown Product"
    except Exception as e:
        print(f"⚠️ Could not fetch product names: {e}")
        for rec in final_output:
            rec["product_name"] = "Unknown Product"

    return final_output




# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    user_id = "C00078"
    recs = recommend_for_user(user_id)
    print(f"\n🎯 Final recommendations for user {user_id}:")

    for r in recs:
        explanation = r.get('explanation', 'N/A')
        explanation = (
            explanation.replace("```json", "")
                      .replace("```", "")
                      .strip()
        )

        print(f"- {r['product_id']} ({r['product_name']}) | Score={r['score']:.4f} | Source={r['source_event']}")
        print(f"  💬 Explanation: {explanation}")
        if r.get('evidence'):
            print(f"  🔗 Evidence: {', '.join(r['evidence'])}")
        print()


