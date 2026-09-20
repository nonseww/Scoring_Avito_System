import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT = "all_items_for_encoding.parquet"

df = pd.read_parquet(INPUT)
print(f"Объявлений: {len(df)}")
texts = df["item_text"].tolist()
ids = df["item_id"].to_numpy()


def encode_and_save(model_name, doc_prefix, out_path, batch_size, dtype):
    """Кодирует объединение объявлений двумя моделями"""
    print(f"\n>> {model_name}")
    model = SentenceTransformer(model_name, device="cuda")
    print(f"размерность: {model.get_embedding_dimension()}")

    vecs = model.encode(
        [f"{doc_prefix}{t}" for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(dtype)

    np.savez(out_path, vectors=vecs, ids=ids)
    print(f"сохранено: {out_path}, форма {vecs.shape}, dtype {vecs.dtype}")

    del model
    import torch
    torch.cuda.empty_cache()


encode_and_save("intfloat/multilingual-e5-small", "passage: ",
                "e5_union_embeddings.npz", batch_size=128, dtype=np.float32)

encode_and_save("ai-forever/FRIDA", "search_document: ",
                "frida_union_embeddings.npz", batch_size=64, dtype=np.float16)

print("\n>> Готово")