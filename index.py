from document import DOCUMENTS
from typing import List
from dotenv import load_dotenv
import os
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
'''
all_chunks = []
for doc in DOCUMENTS:
    doc_chunk = chunking(doc["content"])
    all_chunks.extend(doc_chunk)
'''

print(f"created {len(all_chunks)} from the documents")

Openrouter_api = os.getenv("OPENROUTER_API")

client = OpenAI(
    api_key=Openrouter_api,
    base_url="https://openrouter.ai/api/v1/"
)


