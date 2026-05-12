from .retrieval import retrieve_chunks
from huggingface_hub import InferenceClient
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

client = InferenceClient(
    token=settings.DOC_HF_TOKEN,
    timeout=120
)


def answer_query(query: str):
    try:
        # -----------------------------
        # 1. Retrieve context
        # -----------------------------
        retrieval = retrieve_chunks(query)

        chunks = retrieval.get("chunks", [])
        score = retrieval.get("retrieval_score", 0.0)

        if not chunks:
            return {
                "answer": "No relevant information found in documents.",
                "sources": [],
                "retrieval_score": score
            }

        context = "\n\n".join(chunks)

        # -----------------------------
        # 2. Build strong RAG prompt
        # -----------------------------
        prompt = f"""
You are a strict RAG assistant.

Rules:
- Use ONLY the provided context
- If the answer is not in context, say "Not found in documents"
- Do not guess or hallucinate

Context:
{context}

Question:
{query}

Answer:
"""

        # -----------------------------
        # 3. LLM call
        # -----------------------------
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful and strict document QA assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()

        # -----------------------------
        # 4. Return structured response
        # -----------------------------
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_score": round(float(score), 4)
        }

    except Exception as e:
        logger.exception(f"RAG error: {e}")

        return {
            "answer": "System error occurred while processing query.",
            "sources": [],
            "retrieval_score": 0.0
        }