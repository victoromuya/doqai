# from .embedding_service import generate_embedding
# from .vectordb import collection

# def retrieve_chunks(query, top_k=3):
#     query_embedding = generate_embedding(query)

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )

#     return results["documents"][0]

from .embedding_service import generate_query_embedding
from .vectordb import collection
from .rag_evaluation import retrieval_quality_score


def retrieve_chunks(query, top_k=3):

    query_embedding = generate_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    chunks = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    score = retrieval_quality_score(results)

    return {
        "chunks": chunks,
        "ids": ids,
        "metadatas": metadatas,
        "retrieval_score": float(score)
    }