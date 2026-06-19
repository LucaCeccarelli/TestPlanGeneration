# Automatic Test Plan Generation from Technical Standards

A research pipeline that reads a technical standard (PDF) and outputs a
structured test plan conforming to ISO/IEC/IEEE 29119-3:2021.

## Architecture

Three sequential axes, each building on the previous:

```
PDF norm
  └─ Axis A: extract requirements → FAISS + BM25 knowledge base + JSONL chunks
       └─ Axis B: multi-agent generation → ISO 29119-3 TestPlan JSON
            └─ Axis C (optional): agentic validation → audit report JSON
```

**Axis A** — extracts normative clauses (SHALL/MUST/SHOULD/MAY) from the
source PDF using pdfplumber + spaCy.  A two-pass section-aware chunker keeps
each numbered clause intact when it fits within the size limit, falling back
to `RecursiveCharacterTextSplitter` only for oversized clauses.  Cross-
references between clauses are resolved automatically and defined terms are
injected into each chunk's metadata.  Chunks are embedded with
`nomic-embed-text` into a FAISS vector index; a BM25 sparse index is built
alongside for hybrid retrieval.

**Axis B** — a Planner agent runs once to cluster all normative chunks into
feature-set groups using FAISS k-means, then three agents are run per chunk
(Analyst → RAG Router → Designer) using `llama3.2` via Ollama.  The
Analyst's prompt is built dynamically per chunk (TraceLLM-style traceability
anchoring).  The RAG Router uses hybrid FAISS + BM25 retrieval with
Reciprocal Rank Fusion and a minimum-evidence guard.  The Designer generates
two candidate test cases at different temperatures and selects the better one
via BERTScore self-evaluation; low-confidence outputs are flagged
automatically.  Outputs are assembled into the full ISO 29119-3 hierarchy
(`TestPlan → FeatureSet → TestCondition → TestCoverageItem → TestCase`).

**Axis C** — validates each generated test case against its source chunk: an
embedding-distance pre-audit flags semantic drift before the LLM runs;
hallucination, contradiction, and omission detection are performed by the
Auditor agent; the final verdict is overridden by deterministic Python logic.
Each audit report includes `traceability_confidence` (cosine similarity
between the test case objective and its source clause).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally on `localhost:11434`
- Required Ollama models:

```bash
ollama pull nomic-embed-text   # embeddings (Axis A / C)
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
  --output-index data/output/index/faiss_index \
  --skip-pages 1 2 3 4 5 6 7 8
```

`--skip-pages` accepts any number of 1-based page numbers to exclude from
extraction (cover page, table of contents, etc.).

Axis A now builds **both** a FAISS dense index and a BM25 sparse index
(`bm25_index.pkl`) in the output directory.  If you have an existing index
from a previous run, a full re-run is required to generate the BM25 file.

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

### Cohen's kappa — compare Auditor verdicts against human annotations

```bash
python scripts/compute_kappa.py \
  --audit-json data/output/axis_c/audit_report.json \
  --human-csv  data/ground_truth/human_verdicts.csv \
  --output     data/output/axis_c/kappa_result.json
```

The human CSV must have columns `tc_id,human_verdict` with verdicts in
`{PASS, FAIL, WARNING}`.

Each script supports `--help` for full option documentation.

## Tests

```bash
pytest
```

All tests mock Ollama and FAISS — no external services required.

**38 tests** across three files:

| File | Tests | Coverage |
|---|---|---|
| `tests/test_axis_a_chunker.py` | 9 | Two-pass chunking, cross-ref resolution, defined-terms injection |
| `tests/test_axis_b_schema.py` | 14 | Schema round-trips, Planner clustering, Designer schema injection, BERTScore gate |
| `tests/test_axis_c_auditor.py` | 15 | Verdict override, embedding pre-audit, traceability confidence, adversarial injection suite |

## Project structure

```
axis_a/          PDF extraction, NLP annotation, chunking, indexing, evaluation
axis_b/          Pydantic models, LLM setup, four agents (Planner + 3), pipeline
axis_c/          Auditor agent, structural (Guardrails) validator
scripts/         CLI entry points for each axis + compute_kappa.py
tests/           Unit tests (mocked)
docs/            INITIAL_STATE.md — baseline description for publication
                 IMPROVEMENTS.md  — SoTA improvement propositions with bibliography
data/
  input/         Place the source PDF here
  ground_truth/  Manually annotated clauses for Axis A evaluation;
                 human_verdicts.csv for Cohen's kappa
  output/        Generated chunks, FAISS index, BM25 index, test plans (git-ignored)
```

## Key data model fields added by SoTA improvements

`RequirementChunk` carries three new fields populated by Axis A:

| Field | Type | Description |
|---|---|---|
| `is_full_clause` | `bool` | True when the chunk survived the two-pass splitter intact |
| `context_refs` | `list[str]` | Excerpts of cross-referenced clauses (§ / clause / table) |
| `defined_terms` | `dict[str, str]` | Terms-and-Definitions entries whose keys appear in this chunk |

Axis C audit reports carry three new fields:

| Field | Type | Description |
|---|---|---|
| `pre_audit_flag` | `bool` | True when embedding distance exceeds the semantic-drift threshold |
| `embedding_distance` | `float` | Cosine distance between the test case objective and its source clause |
| `traceability_confidence` | `float` | Cosine similarity (1 − distance); measurable traceability quality score |

## Sensitivity testing (Axis B)

Swap the chat model via `DEFAULT_CHAT_MODEL` in `axis_b/llm_setup.py`:

```
llama3.2      default
mistral:7b    alternative
llama3.1:8b   alternative
```

BERTScore F1 comparisons across models are computed by `axis_a/evaluate.py`
against the ground-truth JSONL once `data/ground_truth/annotated_requirements.jsonl`
is populated.
