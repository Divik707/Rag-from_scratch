import os
import re
import math
import time
import traceback
import json
from typing import List, Dict, Optional, Tuple
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

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

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

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunking(text: str, max_words: int = 120, overlap_words: int = 30) -> List[str]:
    textt = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', textt)
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
                current = words[max_words - overlap_words:max_words] if len(words) > max_words else words
                current_len = len(current)
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if len(c.split()) > 8]

# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# HyDE — HYPOTHETICAL DOCUMENT EMBEDDING
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# CONTEXTUAL CHUNK ENRICHMENT
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# ① CROSS-ENCODER RERANKING
# ─────────────────────────────────────────────

def cross_encoder_rerank(question: str, candidates: List[Dict], top_n: int = 6) -> List[Dict]:
    """
    Uses an LLM as a cross-encoder to score each (query, chunk) pair.
    Returns candidates sorted by relevance score descending.
    """
    if not candidates:
        return []

    scored_candidates = []
    batch_prompt_parts = []

    for i, cand in enumerate(candidates):
        batch_prompt_parts.append(
            f"[Passage {i+1}]\n{cand['chunk'][:600]}"
        )

    batch_text = "\n\n".join(batch_prompt_parts)

    prompt = f"""You are a relevance scorer. Given a question and a list of passages, 
score each passage from 0 to 10 for how directly and accurately it answers the question.
Return ONLY a JSON array of numbers (one per passage, in order), no explanation.
Example for 3 passages: [7, 2, 9]

Question: {question}

Passages:
{batch_text}

Scores (JSON array only):"""

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON array safely
        match = re.search(r'\[[\d\s.,]+\]', raw)
        if match:
            scores = json.loads(match.group())
        else:
            scores = [5.0] * len(candidates)
    except Exception as e:
        print(f"Cross-encoder error: {e}")
        scores = [5.0] * len(candidates)

    # Pad/trim scores to match candidates
    scores = (scores + [5.0] * len(candidates))[:len(candidates)]

    for cand, score in zip(candidates, scores):
        enriched = dict(cand)
        enriched["ce_score"] = float(score)
        scored_candidates.append(enriched)

    scored_candidates.sort(key=lambda x: x["ce_score"], reverse=True)
    return scored_candidates[:top_n]

# ─────────────────────────────────────────────
# ② METADATA FILTERING
# ─────────────────────────────────────────────

def build_metadata_filter(
    source: Optional[str] = None,
    doc_type: Optional[str] = None,
    date_after: Optional[str] = None,   # ISO format "YYYY-MM-DD"
    version: Optional[str] = None,
    access_scope: Optional[str] = None
) -> Dict:
    """
    Returns a structured filter dict. Each field is optional.
    Docs must include these keys in their metadata to be filtered.
    """
    filters = {}
    if source:
        filters["source"] = source
    if doc_type:
        filters["doc_type"] = doc_type
    if date_after:
        filters["date_after"] = date_after
    if version:
        filters["version"] = version
    if access_scope:
        filters["access_scope"] = access_scope
    return filters

def apply_metadata_filter(results: List[Dict], filters: Dict) -> List[Dict]:
    """
    Filters retrieved chunks by metadata fields.
    Supports: source (exact), doc_type (exact), date_after (ISO string compare),
              version (exact), access_scope (exact).
    """
    if not filters:
        return results

    filtered = []
    for r in results:
        meta = r.get("meta", {})
        pass_filter = True

        if "source" in filters:
            if meta.get("source", "") != filters["source"]:
                pass_filter = False

        if "doc_type" in filters:
            if meta.get("doc_type", "") != filters["doc_type"]:
                pass_filter = False

        if "date_after" in filters:
            doc_date = meta.get("date", "")
            if doc_date and doc_date < filters["date_after"]:
                pass_filter = False
            elif not doc_date:
                # If no date in metadata, don't filter it out — be permissive
                pass

        if "version" in filters:
            if meta.get("version", "") != filters["version"]:
                pass_filter = False

        if "access_scope" in filters:
            chunk_scope = meta.get("access_scope", "public")
            if chunk_scope != filters["access_scope"] and chunk_scope != "public":
                pass_filter = False

        if pass_filter:
            filtered.append(r)

    return filtered

# ─────────────────────────────────────────────
# ③ CONTEXT COMPRESSION
# ─────────────────────────────────────────────

def compress_context(question: str, chunks: List[Dict], max_tokens: int = 1200) -> str:
    """
    Extracts only the query-relevant sentences from each chunk,
    then assembles a tight context string under max_tokens.
    Reduces hallucination and token waste.
    """
    if not chunks:
        return ""

    compressed_parts = []
    current_tokens = 0

    for r in chunks:
        chunk_text = r.get("chunk", r.get("raw_chunk", ""))
        source_label = f"[{r['meta']['source']} | Chunk {r['meta']['chunk_id']}]"

        prompt = f"""Extract only the sentences from the passage below that are directly relevant 
to answering the question. Return only the extracted sentences, nothing else.
If nothing is relevant, return: IRRELEVANT

Question: {question}

Passage:
{chunk_text}

Relevant sentences:"""

        try:
            resp = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300
            )
            extracted = resp.choices[0].message.content.strip()
        except:
            extracted = chunk_text  # fallback: use full chunk

        if extracted.upper() == "IRRELEVANT" or not extracted:
            continue

        block = f"{source_label}\n{extracted}"
        block_tokens = approx_token_count(block)

        if current_tokens + block_tokens > max_tokens:
            # Add as much as we can
            remaining = max_tokens - current_tokens
            if remaining > 50:
                truncated_words = extracted.split()[:remaining * 3 // 4]
                block = f"{source_label}\n{' '.join(truncated_words)}"
                compressed_parts.append(block)
            break

        compressed_parts.append(block)
        current_tokens += block_tokens

    return "\n\n---\n\n".join(compressed_parts)

# ─────────────────────────────────────────────
# INDEX BUILDING
# ─────────────────────────────────────────────

all_chunks = []
ctx_chunks = []
chunk_meta = []

print("Creating chunks and indexes...")

for doc_id, doc in enumerate(DOCUMENTS):
    raw_chunks = chunking(doc["content"])

    for chunk_id, chunk in enumerate(raw_chunks):
        enriched = contextualize_chunk(chunk, doc["content"], doc["title"])

        all_chunks.append(chunk)
        ctx_chunks.append(enriched)

        # Extend metadata from doc if present (supports doc_type, date, version, access_scope)
        meta_entry = {
            "doc_id": str(doc_id),
            "chunk_id": str(chunk_id),
            "source": doc["title"],
            "raw_chunk": chunk,
            # Optional fields — populated from doc if available
            "doc_type": doc.get("doc_type", "general"),
            "date": doc.get("date", ""),
            "version": doc.get("version", ""),
            "access_scope": doc.get("access_scope", "public"),
        }
        chunk_meta.append(meta_entry)

print(f"✅ Prepared {len(ctx_chunks)} chunks")

chroma = chromadb.PersistentClient(path="./chroma_db")
try:
    chroma.delete_collection("ctx_rag")
except:
    pass

ctx_collection = chroma.create_collection(
    name="ctx_rag",
    metadata={"hnsw:space": "cosine"}
)

ctx_embeddings_list = embeddings(ctx_chunks)
ctx_collection.add(
    ids=[f"ctx_{i}" for i in range(len(ctx_chunks))],
    embeddings=ctx_embeddings_list,
    documents=ctx_chunks,
    metadatas=chunk_meta
)

ctx_bm25 = BM25Okapi([tokenizer(c) for c in all_chunks])
print("✅ Indexes built successfully!")

# ─────────────────────────────────────────────
# RETRIEVAL — HYBRID SEARCH
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(sem_ranked, bm25_ranked, k: int = 70):
    scores = {}
    for rank, (idx, _) in enumerate(sem_ranked):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def ctx_hybrid_search(question: str, candidate_k: int = 30, fused_k: int = 12) -> List[Dict]:
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

# ─────────────────────────────────────────────
# FULL PIPELINE — ANSWER GENERATION
# ─────────────────────────────────────────────

def final_rag_answer(
    question: str,
    metadata_filters: Optional[Dict] = None,
    use_compression: bool = True,
    use_reranking: bool = True,
    return_diagnostics: bool = False
) -> str | Dict:
    """
    Full enhanced RAG pipeline:
      1. Hybrid retrieval (HyDE + BM25 + dense + RRF)
      2. Metadata filtering
      3. Cross-encoder reranking
      4. Context compression
      5. LLM answer generation

    Args:
        question: User's question
        metadata_filters: Optional dict from build_metadata_filter()
        use_compression: Toggle context compression
        use_reranking: Toggle cross-encoder reranking
        return_diagnostics: If True, returns dict with answer + pipeline details

    Returns:
        Answer string, or diagnostic dict if return_diagnostics=True
    """
    t0 = time.time()

    # Step 1 — Hybrid retrieval
    retrieved = ctx_hybrid_search(question, candidate_k=30, fused_k=15)
    t_retrieval = time.time() - t0

    # Step 2 — Metadata filtering
    if metadata_filters:
        retrieved = apply_metadata_filter(retrieved, metadata_filters)

    if not retrieved:
        answer = "No relevant documents found matching the query and filters."
        if return_diagnostics:
            return {"answer": answer, "retrieved_count": 0, "latency_s": time.time() - t0}
        return answer

    # Step 3 — Cross-encoder reranking
    if use_reranking:
        t_rerank_start = time.time()
        reranked = cross_encoder_rerank(question, retrieved, top_n=6)
        t_rerank = time.time() - t_rerank_start
    else:
        reranked = retrieved[:6]
        t_rerank = 0.0

    # Step 4 — Context compression
    if use_compression:
        context = compress_context(question, reranked, max_tokens=1200)
    else:
        context_blocks = []
        for r in reranked:
            block = f"[Source: {r['meta']['source']} | Chunk {r['meta']['chunk_id']}]\n{r['chunk']}"
            context_blocks.append(block)
        context = "\n\n---\n\n".join(context_blocks)

    # Step 5 — Answer generation
    prompt = f"""Answer the question using only the provided context.
Cite sources as [Source Name | Chunk X].
If the answer is not found in context, say so clearly — do not hallucinate.

Context:
{context}

Question: {question}
"""

    t_gen_start = time.time()
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"
    t_gen = time.time() - t_gen_start

    total_latency = time.time() - t0

    if return_diagnostics:
        return {
            "answer": answer,
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "context_tokens": approx_token_count(context),
            "latency_retrieval_s": round(t_retrieval, 3),
            "latency_rerank_s": round(t_rerank, 3),
            "latency_generation_s": round(t_gen, 3),
            "latency_total_s": round(total_latency, 3),
            "top_sources": [
                {"source": r["meta"]["source"], "chunk_id": r["meta"]["chunk_id"],
                 "ce_score": r.get("ce_score", None), "rrf_score": round(r.get("rrf_score", 0), 4)}
                for r in reranked
            ],
            "compressed_context": context
        }

    return answer

# ─────────────────────────────────────────────
# ④ EVALUATION PIPELINE
# ─────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates the RAG pipeline across:
    - Retrieval metrics: Precision@k, Recall@k, MRR, NDCG
    - Answer metrics: Faithfulness, Answer Relevance, Groundedness, Latency
    """

    def __init__(self, pipeline_fn=None):
        self.pipeline_fn = pipeline_fn or final_rag_answer

    # ── Retrieval Metrics ──────────────────────────────────────

    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Fraction of top-k retrieved docs that are relevant."""
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / k if k > 0 else 0.0

    def recall_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Fraction of relevant docs found in top-k."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / len(relevant_ids)

    def mean_reciprocal_rank(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """MRR: reciprocal rank of the first relevant result."""
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain @k."""
        def dcg(ids, rel_set, k):
            score = 0.0
            for i, doc_id in enumerate(ids[:k]):
                if doc_id in rel_set:
                    score += 1.0 / math.log2(i + 2)
            return score

        actual_dcg = dcg(retrieved_ids, set(relevant_ids), k)
        ideal_ids = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids] + \
                    [doc_id for doc_id in retrieved_ids if doc_id not in relevant_ids]
        ideal_dcg = dcg(ideal_ids, set(relevant_ids), k)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # ── Answer Quality Metrics (LLM-as-judge) ─────────────────

    def _llm_score(self, prompt: str) -> float:
        """Calls LLM for a score between 0 and 1."""
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'\d+(\.\d+)?', raw)
            if match:
                val = float(match.group())
                return min(val, 1.0) if val <= 1.0 else val / 10.0
            return 0.5
        except:
            return 0.5

    def faithfulness(self, answer: str, context: str) -> float:
        """
        Measures whether the answer is supported by the context.
        Score: 0 (hallucinated) → 1 (fully grounded).
        """
        prompt = f"""Rate how faithfully the answer is supported by the context on a scale 0 to 1.
1 = every claim in the answer is directly supported by the context.
0 = the answer contains facts not in the context.
Return only a number between 0 and 1.

Context:
{context[:2000]}

Answer:
{answer}

Faithfulness score (0-1):"""
        return self._llm_score(prompt)

    def answer_relevance(self, question: str, answer: str) -> float:
        """
        Measures whether the answer actually addresses the question.
        Score: 0 (irrelevant) → 1 (fully relevant).
        """
        prompt = f"""Rate how relevant the answer is to the question on a scale 0 to 1.
1 = directly and completely answers the question.
0 = does not address the question.
Return only a number between 0 and 1.

Question: {question}
Answer: {answer}

Relevance score (0-1):"""
        return self._llm_score(prompt)

    def groundedness(self, answer: str, context: str) -> float:
        """
        Measures citation accuracy and factual consistency with context.
        Stricter than faithfulness — checks specific claims.
        """
        prompt = f"""For each factual claim in the answer, check whether it can be verified in the context.
Rate groundedness 0 to 1.
1 = all claims verifiable in context.
0 = claims contradict or are absent from context.
Return only a number between 0 and 1.

Context:
{context[:2000]}

Answer:
{answer}

Groundedness score (0-1):"""
        return self._llm_score(prompt)

    # ── Full Evaluation Run ────────────────────────────────────

    def evaluate(
        self,
        test_cases: List[Dict],
        k_values: List[int] = [1, 3, 5],
        verbose: bool = True
    ) -> Dict:
        """
        Runs full evaluation on a list of test cases.

        Each test case must have:
            "question": str
            "relevant_doc_ids": List[str]  — ground truth doc IDs (source titles or doc_ids)
            "reference_answer": str (optional) — for answer quality metrics

        Returns aggregated metrics dict.
        """
        all_results = []
        agg = {f"precision@{k}": [] for k in k_values}
        agg.update({f"recall@{k}": [] for k in k_values})
        agg.update({f"ndcg@{k}": [] for k in k_values})
        agg["mrr"] = []
        agg["faithfulness"] = []
        agg["answer_relevance"] = []
        agg["groundedness"] = []
        agg["latency_s"] = []

        for i, tc in enumerate(test_cases):
            question = tc["question"]
            relevant_ids = tc.get("relevant_doc_ids", [])
            reference_answer = tc.get("reference_answer", "")

            if verbose:
                print(f"\n[{i+1}/{len(test_cases)}] Evaluating: {question[:60]}...")

            t0 = time.time()
            diag = self.pipeline_fn(
                question,
                return_diagnostics=True,
                use_reranking=True,
                use_compression=True
            )
            latency = time.time() - t0

            answer = diag["answer"]
            context = diag.get("compressed_context", "")
            top_sources = [r["source"] for r in diag.get("top_sources", [])]

            # Retrieval metrics
            for k in k_values:
                agg[f"precision@{k}"].append(self.precision_at_k(top_sources, relevant_ids, k))
                agg[f"recall@{k}"].append(self.recall_at_k(top_sources, relevant_ids, k))
                agg[f"ndcg@{k}"].append(self.ndcg_at_k(top_sources, relevant_ids, k))

            agg["mrr"].append(self.mean_reciprocal_rank(top_sources, relevant_ids))

            # Answer quality metrics
            faith = self.faithfulness(answer, context)
            rel = self.answer_relevance(question, answer)
            ground = self.groundedness(answer, context)

            agg["faithfulness"].append(faith)
            agg["answer_relevance"].append(rel)
            agg["groundedness"].append(ground)
            agg["latency_s"].append(round(latency, 3))

            case_result = {
                "question": question,
                "answer": answer,
                "latency_s": round(latency, 3),
                "faithfulness": round(faith, 3),
                "answer_relevance": round(rel, 3),
                "groundedness": round(ground, 3),
                "top_sources": top_sources,
            }
            for k in k_values:
                case_result[f"precision@{k}"] = round(agg[f"precision@{k}"][-1], 3)
                case_result[f"ndcg@{k}"] = round(agg[f"ndcg@{k}"][-1], 3)

            all_results.append(case_result)

            if verbose:
                print(f"  ✓ Faithfulness={faith:.2f} | Relevance={rel:.2f} | "
                      f"Groundedness={ground:.2f} | Latency={latency:.2f}s")

        # Aggregate
        summary = {}
        for key, vals in agg.items():
            if vals:
                summary[key] = round(float(np.mean(vals)), 4)

        if verbose:
            print("\n" + "=" * 55)
            print("EVALUATION SUMMARY")
            print("=" * 55)
            for key, val in summary.items():
                print(f"  {key:<25}: {val:.4f}")

        return {
            "summary": summary,
            "per_case": all_results
        }

# ─────────────────────────────────────────────
# MAIN — DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("DEMO: Enhanced RAG Pipeline")
    print("=" * 55)

    # ── Basic usage ──────────────────────────────
    question = "What is the Mauryan Empire?"

    print(f"\nQuestion: {question}\n")
    result = final_rag_answer(
        question,
        metadata_filters=None,       # No filter — query all docs
        use_compression=True,
        use_reranking=True,
        return_diagnostics=True
    )

    print("ANSWER:")
    print(result["answer"])
    print(f"\nLatency     : {result['latency_total_s']}s total")
    print(f"Context size: ~{result['context_tokens']} tokens")
    print(f"Sources used: {[s['source'] for s in result['top_sources']]}")

    # ── Metadata filter example ───────────────────
    print("\n" + "-" * 55)
    print("DEMO: With Metadata Filter (source filter)")
    filters = build_metadata_filter(source=DOCUMENTS[0]["title"])
    filtered_answer = final_rag_answer(
        question,
        metadata_filters=filters,
        return_diagnostics=False
    )
    print(filtered_answer)

    # ── Evaluation example ────────────────────────
    print("\n" + "-" * 55)
    print("DEMO: Evaluation Pipeline")

    test_cases = [
        {
            "question": "What is the Mauryan Empire?",
            "relevant_doc_ids": [DOCUMENTS[0]["title"]],
        },
        {
            "question": "Who founded the Gupta dynasty?",
            "relevant_doc_ids": [DOCUMENTS[0]["title"]],
        },
    ]

    evaluator = RAGEvaluator(pipeline_fn=final_rag_answer)
    eval_results = evaluator.evaluate(test_cases, k_values=[1, 3, 5], verbose=True)

    print("\nPer-case results saved in eval_results['per_case']")
    print("Aggregate summary:")
    for k, v in eval_results["summary"].items():
        print(f"  {k}: {v}")