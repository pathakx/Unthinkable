import numpy as np
import faiss
import pandas as pd
from .shared_utils import build_faiss_index, compute_event_probabilities

def recommend_cart_based(user_id, interactions_df, product_embeddings, product_ids, product_id_to_index, top_k=3):
    """Recommend products similar to recently added-to-cart items."""
    user_df = interactions_df[interactions_df["user_id"] == user_id]
    user_df = compute_event_probabilities(user_df)
    user_df = user_df[user_df["event_type"] == "add_to_cart"]

    if user_df.empty:
        return []

    faiss_index = build_faiss_index(product_embeddings)
    recs, seen = [], set()

    for pid in user_df.sort_values("probability", ascending=False)["product_id"].head(3):
        if pid not in product_id_to_index:
            continue
        vec = product_embeddings[product_id_to_index[pid]].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        D, I = faiss_index.search(vec, top_k * 3)

        for i, s in zip(I[0], D[0]):
            candidate_id = product_ids[i]
            if candidate_id != pid and candidate_id not in seen:
                recs.append({"product_id": candidate_id, "score": float(s), "source_event": "add_to_cart"})
                seen.add(candidate_id)
                if len(recs) >= top_k:
                    break

    return recs
