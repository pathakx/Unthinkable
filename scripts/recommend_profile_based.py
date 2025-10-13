import numpy as np
import faiss
from .shared_utils import get_or_build_user_embedding, build_faiss_index

def recommend_profile_based(
    user_id,
    interactions_df,
    product_embeddings,
    product_ids,
    product_id_to_index,
    seen_products=None,
    top_k=10
):
    seen_products = seen_products or set()
    
    # 🧠 Fetch cached embedding (auto rebuild if stale)
    user_vector = get_or_build_user_embedding(user_id)

    index = build_faiss_index(product_embeddings)
    user_vector = np.ascontiguousarray(user_vector.astype(np.float32)).reshape(1, -1)
    faiss.normalize_L2(user_vector)
    D, I = index.search(user_vector, top_k * 2)

    recs = []
    for i, s in zip(I[0], D[0]):
        pid = product_ids[i]
        if pid not in seen_products:
            recs.append({"product_id": pid, "score": float(s), "source_event": "profile"})
        if len(recs) >= top_k:
            break
    return recs
