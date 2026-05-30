import keyword
from document import DOCUMENTS
from typing import List, Tuple
from dotenv import load_dotenv
import os
import chromadb
from openai import OpenAI

load_dotenv()

def chunking(text: str, max_size: int=200) -> List[str]:
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
        else :
            sentences = para.replace('.', '.\n').split('\n');
            current, current_len = [], 0

            for sent in sentences:
                sent = sent.strip()

                if not sent:
                    continue
                if current_len + len(sent) <= max_size:
                    current.append(sent)
                    current_len += len(sent)
                else :
                    if current :
                        chunks.append(" ".join(current))
                    current_len = len(sent) 
                    current = [sent]   
            if current:
                chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) > 3]                   

all_chunks = []
for doc in DOCUMENTS:
    doc_chunk = chunking(doc["content"])
    all_chunks.extend(doc_chunk)
print(f"created {len(all_chunks)} chunk from the documents")

Openrouter_api = os.getenv("OPENROUTER_API")
client = OpenAI(
    api_key=Openrouter_api,
    base_url="https://openrouter.ai/api/v1/"
)

def embeddings(text: List[str], model: str= "openai/text-embedding-3-small") -> List[List[float]] :
    cleaned = [t.replace("\n", " ").strip() for t in text] 

    try:
        response = client.embeddings.create(input=cleaned, model=model)
        vectors = [i.embedding for i in response.data]
        return vectors
    except Exception as e : 
        print(f" exception : {e}")
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
chunk_meta_data = [{"source": "AetherSoft_Handbook"} for _ in all_chunks]

emb = embeddings(all_chunks)

if(emb):
        collection.add(
            ids=[f"chunk {i}" for i in range(len(all_chunks))],
            embeddings=emb,
            documents=all_chunks,
            metadatas=chunk_meta_data
        )
        print(f"stored vector: {len(emb)}" )
else :
        print("storage failed")    

def naive_rag(question: str,  k: int = 10, verbose : bool = True) -> str:
    question_emb = embeddings([question])

    result = collection.query(
        query_embeddings=question_emb,
        n_results=k
    )

    docs = result["documents"][0]
    meta = result["metadatas"][0] if result["metadatas"] else [{"source": "AetherSoft_Handbook"}] * len(docs)
    dis = result["distances"][0]

    if verbose:
        print(f"\n Query: {question}")
        for i, (d, m, di) in enumerate(zip(docs, meta, dis)):
            print(f"index: [ {i + 1} ] | Similarity: {1-di:.4f} | Sources: {m.get('source')}")
    context = '\n\n'.join([f"Source: {m.get('source')} \n {d}" for d, m in zip(docs, meta)]) 


    resp = client.chat.completions.create(
        model = "openai/gpt-4o-mini",
        messages= [
            {"role": "system", "content": "Answer based ONLY on context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

print(naive_rag("how much is my office stipend"))



from rank_bm25 import BM25Okapi
import re

def tokenizer(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

bm25 =  BM25Okapi([tokenizer(c) for c in all_chunks])
print(f"build bm25 indexes over {len(all_chunks)}")    


def rank_reciprocal_fusion(semantics: List[Tuple[int, float]], keyoword: List[Tuple[int, float]], k:int = 60) -> List[Tuple[int, float]] :

    scores = {}

    for rank, (idx, _ ) in enumerate(semantics):
        scores[idx] = scores.get(idx, 0) + 1 / ( rank + 1 + k )


    for rank, (idx, _ ) in enumerate(keyword):
        scores[idx] = scores.get(idx, 0) + 1 / ( rank + 1 + k)

    return sorted(scores.item(), key=lambda x:x[1], reverse = True)


