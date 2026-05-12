# -----------------------------------------
#Use locally...
# -----------------------------------------
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("BAAI/bge-small-en")
# def generate_embedding(text):
#     return model.encode(text).tolist()

# from sentence_transformers import SentenceTransformer

# Load once (important for performance)
# model = SentenceTransformer("intfloat/e5-base-v2")


# def generate_query_embedding(text: str):
#     """
#     Embedding for search queries (IMPORTANT for E5)
#     """
#     text = text.strip()
#     return model.encode(f"query: {text}").tolist()


# def generate_passage_embedding(text: str):
#     """
#     Embedding for document chunks (IMPORTANT for E5)
#     """
#     text = text.strip()
#     return model.encode(f"passage: {text}").tolist()


# -----------------------------------------
# use API
# -----------------------------------------
# from huggingface_hub import InferenceClient
# from django.conf import settings

# HF_TOKEN = settings.DOC_HF_TOKEN

# # Initialize Hugging Face Inference Client
# client = InferenceClient(
#     token=HF_TOKEN,
#     timeout=120,
#     headers={"x-wait-for-model": "true"}
# )


# def generate_embedding(text):
#     """
#     Generate semantic embeddings using Hugging Face Hosted Inference API.

#     Uses a lightweight sentence-transformer model suitable
#     for Retrieval-Augmented Generation (RAG) pipelines.
#     """

#     try:
#         # Clean and truncate text
#         cleaned_text = text.strip()[:2000]

#         if not cleaned_text:
#             return None

#         embedding = client.feature_extraction(
#             cleaned_text,
#             model="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         return embedding

#     except Exception as e:
#         print(f"Embedding Generation Error: {e}")
#         return None


from huggingface_hub import InferenceClient
from django.conf import settings

HF_TOKEN = settings.DOC_HF_TOKEN

client = InferenceClient(
    token=HF_TOKEN,
    timeout=120,
    headers={"x-wait-for-model": "true"}
)


def _embed(text: str):
    """
    Internal API call to HF embedding model
    """
    text = text.strip()

    if not text:
        return None

    return client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


def generate_query_embedding(text: str):
    """
    Query embedding for RAG search
    """
    return _embed(f"query: {text}")


def generate_passage_embedding(text: str):
    """
    Document chunk embedding for indexing
    """
    return _embed(f"passage: {text}")