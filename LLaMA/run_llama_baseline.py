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

def format_prompt(note):
    return f""" <|begin_of_text|>
                <|start_header_id|>system<|end_header_id|>
                You are a medical expert. Your task is to analyze a medical note and extract referral information in a structured JSON format. Do not include any explanations, disclaimers, or text outside of the JSON object.
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                Follow these steps carefully:
                
                Step 1 – Determine Specialist Referral
                Check if the medical note indicates that the patient was referred to any of the following specialists:
                
                Oncologist, Cardiologist, Gynecologist, Electrophysiologist (cardiology), Lipids Specialist (cardiology), Surgeon (for oncology-related findings), Nephrologist, Endocrinologist, Urologist, Dietician, Nutritionist, Genetic Counselor.
                
                - If a referral to any of these specialists is found, extract and quote the relevant portion of the note and place it in the `referral_confirmation_citation` field.  
                - If no referral to any of the listed specialists is mentioned, set `referral_confirmation_citation` to `null`.
                
                Step 2 – Identify Specialist Type
                If a referral is confirmed, identify the type of specialist (limited to the list above).
                - Quote the part of the note that names the specialist type and place it in `specialist_type_citation`.  
                - If no specific specialist type from the list is provided, set both `specialist_type` and `specialist_type_citation` to `null`.
                
                Step 3 – Output Format
                Return the output in the following JSON structure:
                
                {{
                  "referred": true or false,
                  "specialists": [
                    {{
                      "specialist_type": "type_of_specialist" or null,
                      "referral_confirmation_citation": "Text from the medical note confirming referral" or null,
                      "specialist_type_citation": "Text from the medical note specifying the specialist type" or null
                    }}
                    ...
                    /* repeat for additional specialists if applicable */
                  ]
                }}
    
                Now analyze the following medical note: {note}
                
                Please only output the JSON structure.
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|> """

def main():
    logging.basicConfig(level=logging.INFO)
    MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
    MODEL_BASENAME = None
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    df_note = pd.read_csv("note_data.csv")
    llm = load_model(device_type, MODEL_ID, MODEL_BASENAME)

    all_outputs = []

    for i, row in tqdm(df_note.iterrows(), total=len(df_note)):
        prompt = format_prompt(row["note_text"])

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

    with open( "llama_baseline.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()