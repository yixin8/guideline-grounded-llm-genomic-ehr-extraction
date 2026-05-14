# Guideline-Grounded LLM Pipeline for Extracting Genomics-Informed Clinical Recommendations from EHR Notes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code for the paper:

> **Guideline-Grounded Large Language Models for Extracting Genome-Informed Clinical Recommendations from Electronic Health Records**

Precision medicine increasingly requires delivery and tracking of genetics-informed risk assessments (GIRAs), which are frequently documented in unstructured electronic health record (EHR) notes. This project evaluates guideline-grounded large language model (LLM) approaches for automatically extracting genomics-informed clinical recommendations (e.g., specialist referrals, laboratory tests, imaging) from EHR notes in the eMERGE study.

We benchmark three LLMs — **GPT-4o**, **LLaMA-3.1-8B-Instruct**, and **LLaMA-3.3-70B-Instruct** — across three prompting strategies:
- **Baseline prompting**
- **GIRA-guideline-aware prompting** (direct guideline inclusion)
- **Retrieval-Augmented Generation (RAG)**

---

## Repository Structure

```
├── GPT-4o/
│   ├── GPT-4o_baseline.ipynb             # Baseline prompting with GPT-4o
│   └── GPT-4o_guideline_aware.ipynb      # Guideline-aware prompting + RAG with GPT-4o
│
├── LLaMA/
│   ├── run_llama_baseline.py             # Baseline prompting with LLaMA models
│   ├── run_llama_GIRA_direct.py          # Direct guideline inclusion with LLaMA models
│   └── run_llama_GIRA_RAG.py             # RAG-based extraction with LLaMA models
│
└── README.md
```

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| **Baseline** | Standard zero-shot or few-shot prompting without external knowledge |
| **Guideline-Aware (Direct)** | Clinical guidelines are directly included in the prompt context |
| **RAG** | Relevant guideline chunks are dynamically retrieved and injected at inference time |

---

## Models

| Model | Type | Deployment |
|---|---|---|
| GPT-4o | Closed-source | HIPAA-compliant Azure OpenAI |
| LLaMA-3.1-8B-Instruct | Open-source (small) | Local H100 GPU |
| LLaMA-3.3-70B-Instruct | Open-source (large) | Local H100 GPU |

---

## Requirements

### GPT-4o (Azure OpenAI)
- Azure OpenAI API access (HIPAA-compliant deployment)
- Python 3.9+
- `openai`, `pandas`, `jupyter`

### LLaMA Models (Local)
- NVIDIA H100 GPU (or equivalent)
- Python 3.9+
- `transformers`, `torch`, `llama-index`, `pandas`

Install dependencies:
```bash
pip install openai transformers torch llama-index pandas jupyter
```

---

## Usage

### GPT-4o
Open and run the Jupyter notebooks in the `GPT-4o/` folder:
```bash
cd GPT-4o
jupyter notebook GPT-4o_baseline.ipynb
jupyter notebook GPT-4o_guideline_aware.ipynb
```

### LLaMA Models
Run the Python scripts in the `LLaMA/` folder:
```bash
cd LLaMA

# Baseline prompting
python run_llama_baseline.py

# Direct guideline inclusion
python run_llama_GIRA_direct.py

# RAG-based extraction
python run_llama_GIRA_RAG.py
```

---

## Evaluation

Model outputs were evaluated against manual chart review (N=18 patients; N=34 documents) using:
- **Accuracy** (TP/34)
- **Precision** (TP/[TP+FP])
- **Recall** (TP/[TP+FN])
- **F1 Score**
- **Computational efficiency** (runtime/throughput)
- **Resource utilization** (API and electricity costs)

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourname2025,
  title={Guideline-Grounded Large Language Models for Extracting Genome-Informed Clinical Recommendations from Electronic Health Records},
  author={Your Name et al.},
  journal={Journal of the American Medical Informatics Association},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please open an issue or contact [your email].
