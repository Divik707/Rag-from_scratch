import os
import re
import math
import traceback
import json
from typing import List, Dict
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from document import DOCUMENTS
import numpy as np

load_dotenv()

OPENROUTER_API = os.getenv("OPENROUTER_API")

client = OpenAI(
    api_key=OPENROUTER_API,
    base_url="https://openrouter.ai/api/v1/"
)

# -----------------------------
# Helpers
# -----------------------------
def tokenizer(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

def dedupe_keep_order(items: List[Dict], key_fn):
    seen = set()
    out = []
    for item in items:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out

# -----------------------------
# Cosine Similarity (pure numpy)
# -----------------------------
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# -----------------------------
# Simple Chunking (Fallback)
# -----------------------------
def chunking(text: str, max_words: int = 120, overlap_words: int = 30) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if current_len + len(words) <= max_words:
            current.extend(words)
            current_len += len(words)
        else:
            if current:
                chunks.append(" ".join(current))
                overlap = current[-overlap_words:] if overlap_words < len(current) else current
                current = overlap + words
                current_len = len(current)
            else:
                chunks.append(" ".join(words[:max_words]))
                current = words[max_words-overlap_words:max_words] if len(words) > max_words else words
                current_len = len(current)
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if len(c.split()) > 8]

# -----------------------------
# Embeddings
# -----------------------------
def embeddings(texts: List[str], model: str = "openai/text-embedding-3-small") -> List[List[float]]:
    cleaned = [t.replace("\n", " ").strip() for t in texts if t.strip()]
    if not cleaned:
        return []
    try:
        resp = client.embeddings.create(input=cleaned, model=model)
        return [x.embedding for x in resp.data]
    except Exception as e:
        print("EMBEDDING ERROR:", e)
        traceback.print_exc()
        return []

# -----------------------------
# HyDE (Query Transformation)
# -----------------------------
def generate_hyde_document(question: str) -> str:
    prompt = f"""Given the question, write a short, factual, hypothetical answer passage:

Question: {question}

Answer:"""
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except:
        return question

# -----------------------------
# Contextual Chunk Enrichment
# -----------------------------
def contextualize_chunk(chunk: str, full_doc: str, title: str) -> str:
    prompt = f"""
Document Title: {title}
Full Document (excerpt):
{full_doc[:6000]}

Chunk:
{chunk}

Provide 1-2 sentences of context, then append the original chunk."""
    try:
        resp = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except:
        return chunk

# -----------------------------
# Build Vector + BM25 Index
# -----------------------------
all_chunks = []      # for BM25
ctx_chunks = []      # for semantic search
chunk_meta = []

print("Creating chunks and indexes...")

for doc_id, doc in enumerate(DOCUMENTS):
    raw_chunks = chunking(doc["content"])  # Using simple for stability now
    
    for chunk_id, chunk in enumerate(raw_chunks):
        enriched = contextualize_chunk(chunk, doc["content"], doc["title"])
        
        all_chunks.append(chunk)
        ctx_chunks.append(enriched)
        
        chunk_meta.append({
            "doc_id": str(doc_id),
            "chunk_id": str(chunk_id),
            "source": doc["title"],
            "raw_chunk": chunk
        })

print(f"✅ Prepared {len(ctx_chunks)} chunks")

# Chroma DB
chroma = chromadb.PersistentClient(path="./chroma_db")  # Better: use persistent storage
try:
    chroma.delete_collection("ctx_rag")
except:
    pass

ctx_collection = chroma.create_collection(
    name="ctx_rag",
    metadata={"hnsw:space": "cosine"}
)

ctx_embeddings = embeddings(ctx_chunks)
ctx_collection.add(
    ids=[f"ctx_{i}" for i in range(len(ctx_chunks))],
    embeddings=ctx_embeddings,
    documents=ctx_chunks,
    metadatas=chunk_meta
)

ctx_bm25 = BM25Okapi([tokenizer(c) for c in all_chunks])
print("✅ Indexes built successfully!")

# -----------------------------
# Hybrid Search
# -----------------------------
def ctx_hybrid_search(question: str, candidate_k: int = 30, fused_k: int = 12) -> List[Dict]:
    # HyDE
    hyde_doc = generate_hyde_document(question)
    hyde_emb = embeddings([hyde_doc])
    q_emb = embeddings([question])
    
    final_emb = hyde_emb[0] if hyde_emb else (q_emb[0] if q_emb else None)
    
    if not final_emb:
        return []
    
    sem = ctx_collection.query(query_embeddings=[final_emb], n_results=candidate_k)
    sem_ranked = [(int(i.split("_")[1]), d) for i, d in zip(sem["ids"][0], sem["distances"][0])]
    
    bm25_scores = ctx_bm25.get_scores(tokenizer(question))
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:candidate_k]
    
    fused = reciprocal_rank_fusion(sem_ranked, bm25_ranked)
    
    results = []
    for idx, score in fused[:fused_k]:
        results.append({
            "idx": idx,
            "chunk": ctx_chunks[idx],
            "raw_chunk": chunk_meta[idx]["raw_chunk"],
            "meta": chunk_meta[idx],
            "rrf_score": score
        })
    
    return dedupe_keep_order(results, lambda x: (x["meta"]["doc_id"], x["meta"]["chunk_id"]))

def reciprocal_rank_fusion(sem_ranked, bm25_ranked, k: int = 70):
    scores = {}
    for rank, (idx, _) in enumerate(sem_ranked):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -----------------------------
# Final RAG Function
# -----------------------------
def final_rag_answer(question: str) -> str:
    retrieved = ctx_hybrid_search(question)
    
    # Simple rerank (take top)
    top_results = retrieved[:6]
    
    context_blocks = []
    for r in top_results:
        block = f"[Source: {r['meta']['source']} | Chunk {r['meta']['chunk_id']}]\n{r['chunk']}"
        context_blocks.append(block)
    
    context = "\n\n---\n\n".join(context_blocks)
    
    prompt = f"""
Answer the question using only the provided context.
Cite sources as [Source Name | Chunk X].
If answer not found, say so clearly.

Context:
{context}

Question: {question}
"""
    
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    q = "What is the Mauryan Empire?"
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(final_rag_answer(q))