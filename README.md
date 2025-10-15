# Agentic Multimodal RAG  
### An Intelligent Framework for Scientific Concept Discovery from Text and Visuals  

---

## 🧠 Overview
**Agentic Multimodal RAG** is a lightweight, open-source framework designed to enable **multimodal question answering** (DocQA) over scientific papers.  
Unlike traditional RAG systems that rely solely on textual content, our framework jointly leverages **text**, **figures**, and **tables**, combining them through a **dual-index retrieval and evidence fusion** pipeline enhanced with a **reflection mechanism** for factual grounding.

This project was developed as part of the **COMP5703 Capstone** at the **University of Sydney (Group CS55)**.

---

## 🎯 Objectives
- Build an **agentic multimodal RAG pipeline** integrating text, figure, and table understanding.  
- Achieve **evidence-grounded** and **traceable** answers to scientific questions.  
- Use **open-source models only** (no paid APIs) to ensure full reproducibility.  
- Evaluate on **three benchmark datasets**: SPIQA, ScienceQA, and DocVQA.  

---

## 🧩 System Architecture
Our architecture follows a **dual-level key-value graph RAG** design, integrating:
1. **Document Ingestion & Indexing** – parses and embeds text (SciBERT) and images (CLIP) into FAISS indexes.  
2. **Query-side Compiler** – normalizes and interprets queries to guide retrieval strategy.  
3. **Micro Planner** – orchestrates text, table, and figure retrieval dynamically.  
4. **Evidence Fusion & Re-ranking (EFR)** – merges multi-modal candidates using weighted reciprocal rank fusion and reranking.  
5. **Reflection Layer** – reviews draft answers, detects hallucination, and triggers targeted re-retrieval for correction.  

> **Goal:** deliver concise, evidence-supported answers with page/figure citations and robust multimodal grounding.

---

## 📊 Datasets
| Dataset | Year | Samples | Modality | Domain |  
|:--|:--:|:--:|:--|:--|  
| [SPIQA](https://huggingface.co/datasets/google/spiqa) | 2024 | ~50 K | Text + Image | Scientific QA |  
| [ScienceQA](https://github.com/lupantech/ScienceQA) | 2022 | ~21 K | Text + Image | Science Education QA |  
| [DocVQA](https://www.docvqa.org/datasets/docvqa) | 2020 | ~12 K | Document + Text | Document Understanding |  

All datasets are public, version-pinned, and stored with checksums to ensure reproducibility.

---

## ⚙️ Core Technologies
| Component | Tool / Framework | Purpose |  
|:--|:--|:--|  
| Language | Python 3.10 | Development environment |  
| Deep Learning | PyTorch | Model training & inference |  
| Vector Index | FAISS | Efficient text & image retrieval |  
| Text Embedding | BGE-M3 | Semantic representation |  
| Image Embedding | CLIP | Visual-text alignment |  
| Knowledge Graph | Neo4j | Cross-modal entity linking |  
| LLM Engines | LLaMA / Mistral / Qwen (3B-7B) | Evidence-grounded generation |  
| Frontend Demo | Streamlit / Gradio | Interactive interface |  

---

## 🧪 Evaluation Metrics
- **Retrieval:** Hit@5 | nDCG@5 | Evidence-Completeness  
- **Answer Generation:** Acc@1 | Quote-F1 | Attribution Rate  
- **Efficiency:** ≤ 16 GB VRAM | Index Build ≤ 2 h | Latency P95 < 5 s  

A +5 – 10 pp improvement on Acc@1 vs text-only RAG is targeted.

---

## 📦 Deliverables
1. **Multimodal QA Prototype** – end-to-end DocQA system.  
2. **Dual Index Module** – cross-modal text + image retrieval fusion.  
3. **Agentic Planner** – tool-calling framework with query-aware routing.  
4. **Evaluation Reports** – results on SPIQA, ScienceQA, DocVQA + ablation studies.  
5. **Reproducible Artifacts** – scripts, configs, demo dataset, and user guide.

---

## 👥 Team & Roles
| Member | Role | Responsibilities |  
|:--|:--|:--|  
| **Hanyu Wang (540063972)** | Project Manager | Overall coordination & client communication |  
| **Junbo Liu (540107070)** | Research Lead | Literature review & method design |  
| **Xiaoran Wang (510480190)** | Data Engineer | Data collection & preprocessing |  
| **Jinlin Zhong (540064511)** | Model Developer | Multimodal QA prototype implementation |  
| **Zhencheng Huang (530387996)** | System Architect | Architecture & deployment |  
| **Kunming Lyu (530355135)** | Testing & Docs | Evaluation & final report preparation |  

---

## 📅 Milestones
| Week | Milestone | Key Tasks |  
|:--:|:--|:--|  
| 1–2 | Project Planning & Architecture | Define objectives, allocate roles |  
| 3–4 | Workflow & Data Setup | Pipeline design, dataset preparation |  
| 5 | Proposal Submission | Client review |  
| 6–7 | Module Development & Integration | Retrieval, fusion, reflection layers |  
| 8–9 | Experimentation & Testing | Benchmark on SPIQA, ScienceQA, DocVQA |  
| 10–11 | Deployment & Documentation | Demo setup + user guide |  
| 12–13 | Final Presentation & Report | Results summary & submission |  

---

## 🧩 Risk Management
- **Hardware Limitations:** fallback to smaller open-source models or Google Gemini API credits.  
- **Scheduling Delays:** parallelize documentation & development to mitigate impact.  
- **Dataset Integrity:** all datasets archived with SHA-256 checksums.  

---

## 📚 References
Key works include **RA-RAG (2024)**, **Self-RAG (2023)**, **VisDoMRAG (2025)**, **MDocAgent (2025)**, and **HM-RAG (2025)**, among others.  
See the [report reference list](./report.pdf) for full citations.

---

## 🤝 Acknowledgements
Developed by **Group CS55**, University of Sydney, under the supervision of Dr. Weibao.  
We gratefully acknowledge the use of open-source tools, datasets, and LLMs that made this research possible.

---

## 🪪 License
This project is released under the **MIT License**.  
See [LICENSE](./LICENSE) for details.
