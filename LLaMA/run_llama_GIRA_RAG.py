import os
import json
import sys
from typing import List
from tqdm import tqdm
import csv
import torch

from llama_index.core import (
    Document,
    ServiceContext,
    VectorStoreIndex,
    Settings
)

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
)

from llama_index.core import load_index_from_storage

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

import logging
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig



# ----------------------------
# CONFIG
# ----------------------------
GUIDELINE_TXT_PATH = "GIRA_Reference.txt"   
NOTES_CSV_PATH = "note_data.csv"           
OUTPUT_JSONL = "llama_GIRA_rag.json"

PERSIST_DIR = "rag_index_store/"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LLM_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

RAG_TOP_K = 2

GEN_MAX_NEW_TOKENS = 1024
GEN_TEMPERATURE = 0.0

NOTE_TEXT_COLUMN = "note_text"
NOTE_ID_COLUMN = "note_id"   

# ----------------------------
# TXT loader + chunking
# ----------------------------
def load_txt_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TXT file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size_words: int = 800, overlap_words: int = 80) -> List[Document]:
    """
    Split `text` into word-based chunks and return a list of llama_index Documents.
    Use keyword args for Document constructor (text= or page_content=) to match
    llama_index >= 0.14.x constructor signatures.
    """
    words = text.split()
    docs: List[Document] = []
    start = 0
    i = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end])

        docs.append(Document(text=chunk, metadata={"chunk": i}))

        i += 1
        if end == len(words):
            break
        start = end - overlap_words
    return docs


# ----------------------------
# Build / load index
# ----------------------------


def build_or_load_rag_index(guideline_txt_path: str):
    required_files = ['docstore.json', 'index_store.json', 'vector_store.json']
    index_exists = (
        os.path.exists(PERSIST_DIR) and 
        all(os.path.exists(os.path.join(PERSIST_DIR, f)) for f in required_files)
    )
    
    if index_exists:
        print(f"Loading existing index from {PERSIST_DIR}")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index

    print(f"Building new index (persist dir incomplete or missing)")
    with open(guideline_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    docs = chunk_text(raw_text)

    index = VectorStoreIndex.from_documents(docs)
    
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index persisted to {PERSIST_DIR}")
    return index
# ----------------------------
# Retrieval
# ----------------------------
def retrieve_guideline_passages(index, query: str, top_k: int):
    """Retrieve passages from guideline index only"""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]

# ----------------------------
# Prompt (STRICT grounding)
# ----------------------------
def format_prompt(note):
    return f""" <|begin_of_text|>
                <|start_header_id|>system<|end_header_id|>
               You are a medical expert. Your task is to analyze a medical note and extract genome-informed clinical actions in a structured JSON format. Do not include any explanations, disclaimers, or text outside of the JSON object.
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                TASK:
                Using ONLY the MEDICAL NOTE {note}, identify clinical actions that are explicitly documented as being initiated, ordered, or performed **in response to genetic findings or a Genome-informed Risk Assessment (GIRA)**.
                Use the GIRA-RELATED RECOMMENDATION CONTEXT provided below ONLY to determine whether an action qualifies as GIRA-triggered and to classify the action type.

                DEFINITION:
                A GIRA-triggered clinical action is one that the MEDICAL NOTE explicitly links to:
                - a genetic result, variant, mutation, polygenic risk, or
                - a documented genetic-informed risk assessment or interpretation.

                RULES:
                - The MEDICAL NOTE is the ONLY evidence that an action occurred.
                - The MEDICAL NOTE must explicitly indicate a genetic or GIRA-related reason for the action.
                - Use the GIRA guideline ONLY to classify the action type, NOT to infer missing actions.
                - Do NOT infer causality, assume intent, or recommend actions.
                - If a genetic or GIRA-related trigger is not explicitly documented, do NOT extract the action.
                - Routine, screening, or chronic care actions without an explicit genetic/GIRA link must be excluded.
                - Only consider genetic results related to the following conditions:
                    - Asthma
                    - Atrial fibrillation
                    - Breast cancer
                    - Chronic kidney disease
                    - Coronary heart disease
                    - Hypercholesterolemia
                    - Obesity / BMI
                    - Prostate cancer
                    - Type 1 diabetes
                    - Type 2 diabetes

                ACTION TYPES (GIRA-ALIGNED):
                - referral
                - lab_test
                - imaging_or_monitoring

                GENETIC-RELATED SPECIALIST REFERRALS:
                Only classify an action as a referral if the MEDICAL NOTE explicitly documents referral to one of the following AND links it to a genetic or GIRA-related reason:
                - Oncologist
                - Surgeon (oncology-related)
                - Cardiologist
                - Electrophysiologist (cardiology)
                - Lipids Specialist (cardiology)
                - Nephrologist
                - Endocrinologist
                - Gynecologist
                - Urologist
                - Dietician
                - Nutritionist
                - Genetic Counselor

                EXTRACTION:
                For each GIRA-triggered action explicitly documented, extract:
                - action_type
                - action_name (e.g., cardiology referral, lipid panel, ECG, renal ultrasound)
                - evidence_citation (verbatim text demonstrating both the action and its genetic/GIRA trigger)

                OUTPUT (JSON ONLY):
                {{
                "actions_present": true or false,
                "actions": [
                    {{
                    "action_type": string,
                    "action_name": string,
                    "evidence_citation": string
                    }}
                    ...
                    /* repeat for additional clinical actions if applicable */
                ]
                }}
                IMPORTANT: The guideline context is provided BELOW. Do not answer until you have read it."""


def build_context_block(passages: List[str]) -> str:
    header = "===== BEGIN GUIDELINE CONTEXT =====\n"
    footer = "\n===== END GUIDELINE CONTEXT ====="

    parts = []
    for i, p in enumerate(passages):
        parts.append(f"[passage_{i+1}]\n{p}")
    return header + "\n\n".join(parts) + footer

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    if not os.path.exists(GUIDELINE_TXT_PATH):
        print("Guideline TXT not found", file=sys.stderr)
        sys.exit(1)

    print("Loading embedding model from cache...")
    Settings.embed_model = HuggingFaceEmbedding(
        EMBED_MODEL_NAME,
        cache_folder="./models/"  
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
    
    print(f"Loading tokenizer from cache...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir="./models/"
    )
    
    print(f"Loading model from cache...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        cache_dir="./models/",
        trust_remote_code=True
    )

    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        context_window=128000, 
        generate_kwargs={
            "temperature": 0.0,
            "top_k": 50,
            "do_sample": False,
        }
    )
    Settings.llm = llm

    rag_index = build_or_load_rag_index(GUIDELINE_TXT_PATH)

    notes = []
    with open(NOTES_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        notes = list(reader)

    all_outputs = []
    for i, row in enumerate(tqdm(notes)):
        note_id = row.get(NOTE_ID_COLUMN, i)
        note_text = row[NOTE_TEXT_COLUMN]

        passages = retrieve_guideline_passages(rag_index, note_text, RAG_TOP_K)
        context_block = build_context_block(passages)

        prompt = format_prompt(note_text) + "\n\n" + context_block + "\n\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

        try:
            response = llm.complete(prompt)  
            content = response.text.strip()
            
            content_clean = re.sub(r"^```(?:json)?|```$", "", content, flags=re.DOTALL).strip()

            match = re.search(r"\{.*\}", content_clean, re.DOTALL)
            if match:
                content_clean = match.group()

            parsed = json.loads(content_clean)
            parsed["person"] = int(row["MRN"])
            parsed["note_date"] = str(row["note_date"])
            parsed["note_id"] = int(row["note_id"])
            parsed["note_title"] = str(row["note_title"])
            parsed["GIRA_date"] = str(row["GIRA_date"])
            all_outputs.append(parsed)

        except json.JSONDecodeError:
            logging.warning(f"Warning: Could not parse JSON for index {i}. Content was:\n{content_clean}")
            all_outputs.append({
                "error": "Invalid JSON",
                "index": i,
                "raw": content_clean
            })
        except Exception as e:
            logging.error(f"Error during inference at index {i}: {e}")
            all_outputs.append({
                "error": "Model error",
                "index": i,
                "exception": str(e)
            })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)     
    print(f"Done. Output written to {OUTPUT_JSONL}")

# ----------------------------
if __name__ == "__main__":
    main()
