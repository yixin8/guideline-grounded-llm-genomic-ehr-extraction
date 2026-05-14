import logging
import torch
import pandas as pd
from tqdm import tqdm
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

def load_full_model(model_id, model_basename, device_type):
    logging.info(f"Loading model: {model_id} on {device_type}")

    if device_type.lower() in ["mps", "cpu"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="./models/"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir="./models",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
        model.tie_weights()

    return model, tokenizer

def load_model(device_type, model_id, model_basename=None):
    model, tokenizer = load_full_model(model_id, model_basename, device_type)

    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        generation_config=generation_config,
        top_k=50,
        temperature=0.0,
        truncation=True,
        return_full_text=False,
        do_sample=False    
     )
    return HuggingFacePipeline(pipeline=pipe)

def format_prompt(note, GIRA_guideline):
    return f""" <|begin_of_text|>
                <|start_header_id|>system<|end_header_id|>
               You are a medical expert. Your task is to analyze a medical note and extract genome-informed clinical actions in a structured JSON format. Do not include any explanations, disclaimers, or text outside of the JSON object.
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                TASK:
                Using ONLY the MEDICAL NOTE {note}, identify clinical actions that are explicitly documented as being initiated, ordered, or performed **in response to genetic findings or a Genome-informed Risk Assessment (GIRA)**.
                Use the GIRA-RELATED RECOMMENDATION CONTEXT {GIRA_guideline} ONLY to determine whether an action qualifies as GIRA-triggered and to classify the action type.

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
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|> """

def main():
    logging.basicConfig(level=logging.INFO)
    MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
    MODEL_BASENAME = None
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    df_note = pd.read_csv("note_data.csv")
    with open("GIRA_Reference.txt", "r", encoding="utf-8") as f:
        GIRA_guideline = f.read()

    llm = load_model(device_type, MODEL_ID, MODEL_BASENAME)

    all_outputs = []

    for i, row in tqdm(df_note.iterrows(), total=len(df_note)):
  
        prompt = format_prompt(row["note_text"], GIRA_guideline)

        try:
            response = llm.invoke(prompt)
            content = (
                response[0]["generated_text"]
                if isinstance(response, list) and "generated_text" in response[0]
                else response if isinstance(response, str)
                else ""
            )
            content = content.strip()
            
            # Remove enclosing ``` and optional `json`
            content_clean = re.sub(r"^```(?:json)?|```$", "", content, flags=re.DOTALL).strip()

            # Try to extract JSON from any messy content using a greedy match
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

    with open( "llama_GIRA_direct.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()