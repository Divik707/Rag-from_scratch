from document import DOCUMENTS
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import os
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
import re

load_dotenv()

def chunking(text: str, max_size: int = 200) -> List[str]:
    chunks = []

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    for para in paragraphs:

        if len(para) <= max_size:
            chunks.append(para)

        else:
            sentences = para.replace(".", ".\n").split("\n")

            current = []
            current_len = 0

            for sent in sentences:

                sent = sent.strip()

                if not sent:
                    continue

                if current_len + len(sent) <= max_size:
                    current.append(sent)
                    current_len += len(sent)

                else:

                    if current:
                        chunks.append(" ".join(current))

                    current = [sent]
                    current_len = len(sent)

            if current:
                chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) > 3]

all_chunks = []

for doc in DOCUMENTS:
    doc_chunks = chunking(doc["content"])
    all_chunks.extend(doc_chunks)

print(f"Created {len(all_chunks)} chunks")


OPENROUTER_API = os.getenv("OPENROUTER_API")

client = OpenAI(
    api_key=OPENROUTER_API,
    base_url="https://openrouter.ai/api/v1/"
)

import traceback

def embeddings(
    texts: List[str],
    model: str = "openai/text-embedding-3-small"
) -> List[List[float]]:

    cleaned = [
        t.replace("\n", " ").strip()
        for t in texts
    ]

    try:

        response = client.embeddings.create(
            input=cleaned,
            model=model
        )

        return [x.embedding for x in response.data]

    except Exception:
        print("\n===== EMBEDDING ERROR =====")
        traceback.print_exc()
        print("===========================\n")
        return []


chroma = chromadb.Client()

try:
    chroma.delete_collection("naive_rag")
except:
    pass

collection = chroma.create_collection(
    name="naive_rag",
    metadata={"hnsw:space": "cosine"}
)

chunk_meta_data = [
    {"source": "AetherSoft_Handbook"}
    for _ in all_chunks
]

chunk_embeddings = embeddings(all_chunks)

if chunk_embeddings:

    collection.add(
        ids=[f"chunk {i}" for i in range(len(all_chunks))],
        embeddings=chunk_embeddings,
        documents=all_chunks,
        metadatas=chunk_meta_data
    )

    print(f"Stored {len(chunk_embeddings)} vectors")

else:
    print("Storage failed")


def tokenizer(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi(
    [tokenizer(chunk) for chunk in all_chunks]
)

print(f"Built BM25 index over {len(all_chunks)} chunks")


def rank_reciprocal_fusion(
    semantics: List[Tuple[int, float]],
    keyword: List[Tuple[int, float]],
    k: int = 60
) -> List[Tuple[int, float]]:

    scores = {}

    for rank, (idx, _) in enumerate(semantics):

        scores[idx] = (
            scores.get(idx, 0)
            + 1 / (rank + k + 1)
        )

    for rank, (idx, _) in enumerate(keyword):

        scores[idx] = (
            scores.get(idx, 0)
            + 1 / (rank + k + 1)
        )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


def hybrid_search(
    question: str,
    k: int = 5
) -> List[Dict]:

    question_emb = embeddings([question])

    sem = collection.query(
        query_embeddings=question_emb,
        n_results=20
    )

    sem_ranked = [

        (
            int(chunk_id.split(" ")[1]),
            dist
        )

        for chunk_id, dist in zip(
            sem["ids"][0],
            sem["distances"][0]
        )
    ]

    bm25_scores = bm25.get_scores(
        tokenizer(question)
    )

    bm25_ranked = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    fused = rank_reciprocal_fusion(
        sem_ranked,
        bm25_ranked
    )

    return [

        {
            "chunk": all_chunks[idx],
            "meta": chunk_meta_data[idx],
            "score": score
        }

        for idx, score in fused[:k]
    ]

results = hybrid_search(
    "tell me about gupta empire"
)

for r in results:

    print(
        f"Score: {r['score']:.4f} | "
        f"{r['chunk'][:80]}..."
    )


