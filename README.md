# Automatic Test Plan Generation from Technical Standards

A research pipeline that reads a technical standard (PDF) and outputs a
structured test plan conforming to ISO/IEC/IEEE 29119-3:2021.

## Architecture

Three sequential axes, each building on the previous:

```
PDF norm
  └─ Axis A: extract requirements → FAISS knowledge base + JSONL chunks
       └─ Axis B: multi-agent generation → ISO 29119-3 TestPlan JSON
            └─ Axis C (optional): agentic validation → audit report JSON
```

**Axis A** — extracts normative clauses (SHALL/MUST/SHOULD/MAY) from the
source PDF using pdfplumber + spaCy, chunks them with a section-aware
`RecursiveCharacterTextSplitter`, and embeds them into a FAISS index with
`nomic-embed-text`.

**Axis B** — runs three agents per chunk (Analyst → RAG Router → Designer)
using `llama3.2` via Ollama to produce fully structured `TestCase` objects
that are assembled into the ISO 29119-3 hierarchy
(`TestPlan → FeatureSet → TestCondition → TestCoverageItem → TestCase`).

**Axis C** — validates each generated test case for hallucinations,
contradictions and omissions; verdict logic is deterministic Python
(not delegated to the LLM).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally on `localhost:11434`
- Required Ollama models:

```bash
ollama pull nomic-embed-text   # embeddings (Axis A)
ollama pull llama3.2           # chat model (Axis B / C)
```

- spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Place the source norm PDF at `data/input/iso_18013_5.pdf`, then run the
axes in order.

### Axis A — extract and index requirements

```bash
python scripts/run_axis_a.py \
  --pdf data/input/iso_18013_5.pdf \
  --norm "ISO/IEC 18013-5" \
  --output-chunks data/output/chunks/iso_18013_5_chunks.jsonl \
  --output-index data/output/index/faiss_index
```

### Axis B — generate test plan

```bash
python scripts/run_axis_b.py \
  --chunks data/output/chunks/iso_18013_5_chunks.jsonl \
  --index data/output/index/faiss_index \
  --norm "ISO/IEC 18013-5" \
  --output data/output/test_plans/TP-ISO-IEC-18013-5-001.json
```

### Axis C — validate generated plan

```bash
python scripts/run_axis_c.py \
  --plan data/output/test_plans/TP-ISO-IEC-18013-5-001.json \
  --chunks data/output/chunks/iso_18013_5_chunks.jsonl \
  --index data/output/index/faiss_index
```

Each script supports `--help` for full option documentation.

## Tests

```bash
pytest
```

All tests mock Ollama and FAISS — no external services required.

## Project structure

```
axis_a/          PDF extraction, NLP annotation, chunking, indexing, evaluation
axis_b/          Pydantic models, LLM setup, three agents, pipeline orchestrator
axis_c/          Auditor agent, structural (Guardrails) validator
scripts/         CLI entry points for each axis
tests/           Unit tests (mocked)
data/
  input/         Place the source PDF here
  ground_truth/  Manually annotated clauses for Axis A evaluation
  output/        Generated chunks, FAISS index, test plans (git-ignored)
```

## Sensitivity testing (Axis B)

Swap the chat model via the `--plan-id` argument or by setting
`DEFAULT_CHAT_MODEL` in `axis_b/llm_setup.py`:

```
llama3.2      default
mistral:7b    alternative
llama3.1:8b   alternative
```
