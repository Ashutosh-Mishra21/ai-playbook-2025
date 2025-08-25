# 2025 AI Playbook — Roadmaps, Resources & Links

A practical guide to the current wave of AI (2025): what’s changed, what matters, and how to skill up fast.

> **Who this is for:** builders, researchers, product leaders, and students who want a clear, up-to-date path through agents, RAG, multimodal models, on-device AI, evaluation, safety, and deployment.

---

## 🚀 What’s new in the 2025 wave (in one page)

- **Agentic systems go production**: frameworks like LangGraph and AutoGen have matured for **stateful, tool-using, multi-agent** workflows, with human-in-the-loop and replay/checkpointing.  
- **RAG 2.0**: retrieval quality + **chunking, routing, re-ranking, verification**; evals (RAGAS, OpenAI Evals) standardize quality gates before ship.  
- **Multimodal & on-device**: consumer UX shifts with Apple Intelligence; mobile/offline use grows with **small language models** (SLMs) like Gemma 2 and Phi-3.  
- **Reasoning + low-latency infra**: spec decoding, FlashAttention/vLLM and next-gen GPUs (Blackwell) make **faster, cheaper inference** a must-know.  
- **Compliance becomes product work**: EU AI Act key dates land in 2025; NIST GenAI profiles and UK/US safety evals push risk controls and documentation into delivery pipelines.  
- **Open model ecosystems diversify**: Meta Llama 3.1, Google Gemma 2, Qwen 2.5 broaden choices for cost, privacy, and customization.  

---

## 🧭 Roadmaps (choose your path)

### 1) Builder / SWE (12–16 weeks)
1. **Foundations (2–3 wks)**  
   - Python, vector search basics (FAISS/Milvus), prompt engineering, evaluation mindset.  
2. **RAG Systems (3–4 wks)**  
   - Build doc ingestion → chunking → embeddings → retrieval → re-ranking → answer verification; add RAG evals (RAGAS/OpenAI Evals).  
3. **Agents (3–4 wks)**  
   - Single-agent → tool use → multi-agent orchestration with LangGraph/AutoGen; add human-in-the-loop & guardrails.  
4. **Performance (2–3 wks)**  
   - Speed + cost: vLLM, FlashAttention, speculative decoding; memory/KV cache efficiency.  
5. **Ship & observe (ongoing)**  
   - CI for prompts/tools, offline evals (Evals/HELM), tracing & feedback loops; align with org compliance (NIST profile / EU AI Act).  

### 2) Research-minded (12–20 weeks)
- Study **reasoning**, tool-use, planning; reproduce agent papers; measure with rigorous eval suites (HELM/HumanEval).  
- Explore low-rank adaptation (LoRA/QLoRA), small models finetuning, and retrieval-augmented finetuning. *(Supplement with your lab’s infra.)*

### 3) Product/Startup (6–10 weeks to MVP)
- Narrow use case → data contracts → RAG baseline → agent wrapper for task automation → **obs/evals + safety** → pilot → pricing.  
- Prioritize **on-device or private open-weights** when data residence or cost dictates (Gemma 2 / Llama family + vector DB).  

---

## 🧱 System blueprints (copy these patterns)

- **RAG 2.0 baseline**: Ingest → semantic chunking → hybrid retrieval (BM25+dense) → re-rank → synthesis → citation + self-check → RAGAS/Evals.  
- **Task agent**: Planner (LLM) → Tools (search/db/code) via orchestrator (LangGraph) → Human gate → Retry/rollback → Telemetry.  
- **Multimodal assistant**: Text+image input, OCR, table extractor, RAG over docs, vision-grounded actions, safety layer → transcript & audits.  
- **On-device SLM**: Quantized Gemma/Phi with local vector DB; sync embeddings to edge cache; private mode first.  

---

## 📚 Learn — curated links

### Official model stacks & docs
- [Google AI / Gemini & Gemma](https://ai.google/discover/gemma)  
- [Meta Llama 3.1](https://ai.meta.com/llama)  
- [Qwen 2.5](https://huggingface.co/Qwen)  

### Courses & YouTube
- [Hugging Face Transformers Course (free)](https://huggingface.co/course)  
- [Microsoft “AI Agents for Beginners” (free, 11 lessons)](https://microsoft.github.io/autogen/stable/)  
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)  

### Agent frameworks & orchestration
- [LangGraph](https://www.langchain.com/langgraph)  
- [AutoGen / AG2](https://microsoft.github.io/autogen/stable/)  

### Retrieval & Vector DBs
- [FAISS](https://github.com/facebookresearch/faiss)  
- [Milvus](https://milvus.io)  
- [RAG Surveys/Guides](https://arxiv.org/abs/2312.10997)  

### Evaluation & Benchmarks
- [OpenAI Evals](https://github.com/openai/evals)  
- [HELM (Stanford CRFM)](https://crfm.stanford.edu/helm/latest/)  
- [HumanEval](https://github.com/openai/human-eval)  
- [RAGAS](https://github.com/explodinggradients/ragas)  

### Performance & Inference
- [vLLM](https://github.com/vllm-project/vllm)  
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)  
- [Speculative decoding](https://arxiv.org/abs/2302.01318)  
- [NVIDIA Blackwell architecture](https://developer.nvidia.com)  

### Multimodal & On-device
- [Apple Intelligence](https://www.apple.com/apple-intelligence/)  
- [Gemma 2](https://ai.google/discover/gemma)  
- [Phi-3](https://huggingface.co/microsoft/phi-3)  

### Policy, Risk & Safety
- [EU AI Act Summary](https://artificialintelligenceact.eu/)  
- [NIST Generative AI Profile](https://www.nist.gov/news-events/news/2024/01/nist-releases-draft-profile-manage-risks-generative-ai)  
- [UK AI Safety Institute](https://www.aisafety.gov.uk/)  

---

## 🗂️ Project templates (start here)

- **RAG Starter**: ingestion + BM25+dense + re-rank + answer verify + RAGAS + Evals.  
- **Agent Starter**: LangGraph task agent with tools (search/db/code), HIL gates, retries, tracing.  
- **Multimodal QA**: OCR → table extraction → doc-grounded chat → cite sources (Gemini/Gemma/Llama family).  
- **On-device private assistant**: Gemma/Phi quantized + local vector DB (FAISS/Milvus).  

---

## 🛡️ Shipping checklists (2025-ready)

**Quality & Evaluation**
- Automated offline evals per PR (HELM/Evals); RAG evals (faithfulness, answer correctness) per dataset slice.  

**Safety & Compliance**
- Risk register mapped to **NIST GenAI profile** controls; document model cards, data lineage, red-teaming; tag use-case risk vs EU AI Act categories.  

**Performance**
- Budget latency/throughput; prototype with **vLLM/FlashAttention/spec decoding**; regression tests for context length & cost per 1k requests.  

**Operations**
- Prompt & tool versioning; telemetry + feedback loops; incident playbooks for drift, jailbreaks, and data leakage.  

---

## 🧪 Reading list (hand-picked)

- [AutoGen: Enabling Next-Gen Multi-Agent Systems](https://arxiv.org/abs/2308.08155)  
- [RAG Surveys](https://arxiv.org/abs/2312.10997)  
- [Speculative Decoding Paper](https://arxiv.org/abs/2302.01318)  
- [HELM Benchmark](https://crfm.stanford.edu/helm/latest/)  
- [HumanEval](https://github.com/openai/human-eval)  

---

## 🧰 Toolbelt (shortlist)

- **Vector**: [FAISS](https://github.com/facebookresearch/faiss), [Milvus](https://milvus.io)  
- **Orchestration**: [LangGraph](https://www.langchain.com/langgraph), [AutoGen](https://microsoft.github.io/autogen/stable/)  
- **Serving**: [vLLM](https://github.com/vllm-project/vllm), [ONNX/Triton](https://onnx.ai/)  
- **Eval**: [OpenAI Evals](https://github.com/openai/evals), [RAGAS](https://github.com/explodinggradients/ragas), [HELM](https://crfm.stanford.edu/helm/latest/)  
- **Models**: [Llama 3.1](https://ai.meta.com/llama), [Gemma 2](https://ai.google/discover/gemma), [Qwen 2.5](https://huggingface.co/Qwen), [Phi-3](https://huggingface.co/microsoft/phi-3)  

---

## 📅 2025 awareness (dates that matter)

- **EU AI Act** obligations phase-in across 2025; align product gating and documentation early.  
- **NVIDIA Blackwell** hardware cycle drives training/inference cost curves; plan migrations.  
- **Apple Intelligence** expands on-device experiences; expect rising customer expectations for **privacy + offline** features.  

---

## 🙋 FAQ

**Which stack do I start with?**  
RAG + single agent (LangGraph) + vLLM. Add evals from day 1. Scale up only when metrics demand it.  

**Open or closed models?**  
Pick by **data sensitivity, latency, and cost**. Open models (Gemma/Llama/Qwen) shine for privacy & control; paid APIs for top-tier reasoning or uptime SLAs.  

**How do I keep costs down?**  
Smaller models + retrieval + serving tricks (**spec decoding, FlashAttention, vLLM**) before fine-tuning.  

---

## 🧾 Attribution & further exploration
This README aggregates official docs, standards, and community tooling. Key references: Google AI (Gemini/Gemma), Meta Llama 3.1, Microsoft Phi-3, Qwen 2.5, LangGraph, AutoGen, FAISS, Milvus, vLLM/FlashAttention/speculative decoding, OpenAI/Stanford eval suites, EU/NIST/UK safety frameworks.  

---

### How to use this README
- Clone this repo section into your project root as `README.md`.  
- Create `/evals` with your task evals (RAGAS/OpenAI Evals).  
- Add `/ops` for checklists (risk, prompts, tools, incidents).  
- Wire CI to run evals on each PR before deploy.
