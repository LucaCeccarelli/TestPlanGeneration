# Baseline System — Initial Architecture and Design Decisions

## Automatic Test Plan Generation from Technical Standards

*This document describes the state of the system at the start of the research project, before any improvement informed by the state of the art was applied. It is intended to serve as the baseline description section of the associated publication, against which all subsequent improvements are measured.*

---

## 1. Motivation and Research Gap

No existing automated approach generates a complete, structured test plan directly from a normative technical document — upstream of any software implementation. The work available in the literature targets either the derivation of test cases from user stories and informal SRS documents, or the generation of executable unit tests from already-written source code. The present project addresses a distinct and earlier stage in the verification lifecycle: given a technical standard published as a PDF (here, ISO/IEC 18013-5, the international standard for mobile driving licences), produce a formal test plan conforming to ISO/IEC/IEEE 29119-3:2021 before any implementation exists.

The output is therefore not executable test code but a structured, traceable, human-readable test plan in the ISO 29119-3 document hierarchy, expressed as a JSON artefact.

---

## 2. System Overview

The system is organised into three sequential axes. Axis A and Axis B form the minimal viable pipeline; Axis C is an optional validation layer.

```
PDF (ISO/IEC 18013-5)
        │
        ▼
┌───────────────────┐
│     AXIS A        │  Requirements Extraction
│  pdf_extractor    │  pdfplumber → raw pages
│  nlp_processor    │  spaCy → sentence boundaries, modal detection
│  chunker          │  RecursiveCharacterTextSplitter → RequirementChunk list
│  indexer          │  OllamaEmbeddings + FAISS → searchable index on disk
└────────┬──────────┘
         │  list[RequirementChunk] + FAISS index path
         ▼
┌───────────────────┐
│     AXIS B        │  Agentic Test Plan Generation
│  Analyst Agent    │  interprets one normative chunk
│  RAG Router Agent │  enriches with retrieved context
│  Designer Agent   │  produces one ISO 29119-3 TestCase
│  pipeline         │  orchestrates all agents, builds TestPlan
└────────┬──────────┘
         │  TestPlan (JSON)
         ▼
┌───────────────────┐
│     AXIS C        │  Agentic Validation  (optional)
│  guardrails       │  structural pre-check
│  Auditor Agent    │  hallucination / contradiction / omission detection
└───────────────────┘
         │  per-TestCase audit reports + global metrics
```

The three axes are fully decoupled at the file-system level: Axis A writes a JSONL chunk file and a FAISS index to disk; Axis B reads those files; Axis C reads the Axis B JSON output. Each axis can be re-run independently.

---

## 3. Technology Stack

| Component | Library | Version | Role |
|---|---|---|---|
| PDF extraction | `pdfplumber` | 0.11.9 | Text and table extraction per page |
| NLP annotation | `spacy` + `en_core_web_sm` | 3.8.14 | Tokenisation, POS tagging, sentence boundaries |
| Chunking | `langchain-text-splitters` | 1.1.2 | `RecursiveCharacterTextSplitter` |
| Embeddings | `langchain-ollama` | 1.1.0 | `OllamaEmbeddings(model="nomic-embed-text")` |
| Vector store | `faiss-cpu` + `langchain-community` | 1.14.2 / 0.4.2 | FAISS index build, persist, load |
| LLM | `langchain-ollama` | 1.1.0 | `ChatOllama(model="llama3.2", temperature=0.2, num_ctx=4096)` |
| Agent orchestration | `deepagents` | 0.6.8 | `Agent(name, llm, system_prompt, tools)` |
| Data models | `pydantic` | ≥2.0, <3.0 | All schema models with v2 API |
| Structural validation | `guardrails-ai` | 0.10.2 | String-length / format checks in Axis C |
| Evaluation | `bert-score` | 0.3.13 | Sensitivity tests across LLMs |
| Testing | `pytest` | ≥8, <9 | Unit tests, all Ollama/FAISS calls mocked |

Ollama runs locally on `localhost:11434`. Two models are required: `nomic-embed-text` for embeddings and `llama3.2` as the default chat model. Alternative chat models tested for sensitivity analysis are `mistral:7b` and `llama3.1:8b`.

The project runs on Python 3.11+. All file paths use `pathlib.Path`. No internet calls occur at runtime.

---

## 4. Axis A — Requirements Extraction

### 4.1 PDF Extraction (`axis_a/pdf_extractor.py`, 175 lines)

`pdfplumber` is used to extract content page by page. Tables are extracted first via `page.extract_tables()`, and their bounding boxes are used to filter them out of the main text flow, preventing duplicate content. The module exposes a `extract_pages()` generator that yields one `RawPage(page_number, text, tables)` namedtuple per page, keeping memory usage bounded regardless of document size.

The full document text is assembled by the caller (`axis_a/chunker.py`) by concatenating page texts with page-marker tokens (`\n\n<!-- PAGE {n} -->\n\n`), which are later parsed back to assign `page_start` values to chunks.

### 4.2 NLP Annotation (`axis_a/nlp_processor.py`, 177 lines)

The spaCy model `en_core_web_sm` is loaded once at module level. The module provides two functions consumed by the chunker:

- **`detect_section_header(line)`** — matches a line against the compiled regex `r'^(\d+(?:\.\d+)+)\s+\S'`. The regex is compiled once at module level.
- **`scan_modals(text)`** — tokenises the text with the spaCy pipeline and returns a deduplicated list of modal verb lemmas found among the token set `{shall, must, should, may, shall not, must not, need not}`, requiring the token's POS to be `AUX` or `VERB`. This is the normative-clause detection mechanism used to set `RequirementChunk.is_normative`.

### 4.3 Semantic Chunking (`axis_a/chunker.py`, 153 lines)

The core chunking function `chunk_document(text, source_norm)` uses `RecursiveCharacterTextSplitter` configured as follows:

```python
separators        = [r"\n(?=\d+(?:\.\d+)+\s)", "\n\n", "\n", ". ", " "]
chunk_size        = 512
chunk_overlap     = 64
is_separator_regex = True
keep_separator    = True
add_start_index   = True
```

The splitter is called in a single pass on the full concatenated document text. After splitting, the function post-processes each LangChain `Document` object:

1. Strips page-marker tokens from the chunk text.
2. Determines `page_start` by finding the last page-marker offset at or before the chunk's `start_index` character position.
3. Determines `section` by preferring a section header found in the first half of the chunk text, falling back to the last header seen before the chunk start.
4. Calls `scan_modals()` on the clean text; sets `is_normative = bool(modals)`.
5. Assigns `chunk_id` as `{norm-slug}_{index:04d}`.

The first complete run on ISO/IEC 18013-5 produced **843 chunks** (410 normative, 48.6%). A second run with adjusted skip-page parameters produced **1,175 chunks** (466 normative, 39.7%). The FAISS index in use corresponds to the 1,175-chunk run.

### 4.4 Embedding and Indexing (`axis_a/indexer.py`, 112 lines)

Each `RequirementChunk` is converted to a LangChain `Document` with `page_content = chunk.text` and metadata fields `{chunk_id, section, page_start, is_normative, modals, source_norm}`. The FAISS index is built with `FAISS.from_documents(docs, OllamaEmbeddings(model="nomic-embed-text"))` and persisted with `vectorstore.save_local(index_path)`. Alongside the index, all chunks are serialised as a JSONL file (`chunks.jsonl`) for later reload without reprocessing. The resulting index files are:

- `index.faiss` — 3.5 MB binary FAISS index
- `index.pkl` — 583 KB LangChain docstore pickle
- `chunks.jsonl` — 590 KB plain JSONL chunk file

### 4.5 Evaluation (`axis_a/evaluate.py`, 175 lines)

The evaluation module implements a retrieval quality assessment against a manually annotated ground truth (`data/ground_truth/annotated_requirements.jsonl`). For each ground-truth entry the top-5 FAISS results are retrieved; a hit is recorded if any result contains the annotated key phrase. Precision, Recall, and F1 are reported. The module also supports a baseline comparison against `CharacterTextSplitter(chunk_size=512, separator="\n")`. At the time of this baseline description, `annotated_requirements.jsonl` has not yet been created; evaluation therefore cannot be executed end-to-end.

---

## 5. Axis B — Agentic Test Plan Generation

### 5.1 Data Models (`axis_b/schema.py`, 164 lines)

All Pydantic v2 models are defined in a single module. Every model carries `model_config = ConfigDict(extra="forbid")` to catch schema drift at validation time. The full hierarchy follows the ISO 29119-3 document structure:

```
TestPlan
  └── FeatureSet           (§7.2.4 — logical subset of the test item)
        └── TestCondition  (§7.2.5 — one verifiable item from the norm)
              ├── TestCoverageItem  (§7.3.4 — intermediate layer)
              └── TestCase          (§7.3.5 — core test case)
                    └── TestInput   (§7.3.5.7–8 — one numbered input+result pair)
```

`RequirementChunk` also lives in this module so that all axes import their models from a single location.

The conformance mapping between model fields and ISO 29119-3 Annex B.1.7 is:

| Field | ISO ref | Conformance |
|---|---|---|
| `TestCase.tc_id` | §7.3.5.2 | SHALL |
| `TestCase.objective` | §7.3.5.3 | SHOULD |
| `TestCase.priority` | §7.3.5.4 | SHALL |
| `TestCase.traceability` | §7.3.5.5 | SHALL |
| `TestCase.preconditions` | §7.3.5.6 | SHALL |
| `TestCase.inputs` | §7.3.5.7 | SHALL — minimum 2 entries |
| `TestCase.inputs[].expected_result` | §7.3.5.8 | SHALL |
| `TestCase.actual_results` | §7.3.5.9 | SHALL — placeholder `""` |
| `TestCase.coverage_item_id` | §7.3.4 / B.1.7.a | SHALL |

### 5.2 LLM Setup and Tools (`axis_b/llm_setup.py`, 94 lines)

The LLM is instantiated via a `get_llm()` factory function (not at module level) to allow test mocking:

```python
ChatOllama(model="llama3.2", temperature=0.2, num_ctx=4096)
```

Two LangChain `@tool` functions are defined:

- **`search_norm_knowledge_base(query)`** — calls `vectorstore.similarity_search(query, k=3)` on the loaded FAISS index; returns a formatted string of the top-3 results with their `chunk_id`, `section`, and text.
- **`get_chunks_for_section(section_number)`** — iterates `vectorstore.docstore._dict.values()` and returns all chunks whose `metadata["section"]` starts with the given section number string.

Both tools hold a module-level `_vectorstore` reference initialised by `init_tools(index_path)`, which is called once at the start of `axis_b/pipeline.py` before any agent runs.

### 5.3 Agent 1 — Analyst (`axis_b/agents/analyst.py`, 132 lines)

**Role:** Interpret a raw `RequirementChunk` and identify what the system must do, under what conditions, and what the testable assertion is.

**Tools available:** `search_norm_knowledge_base`

**Input to the agent:** the full chunk text, its `chunk_id`, `section`, `source_norm`, and `modals` list.

**Expected JSON output:**
```json
{
  "requirement_summary": "...",
  "testable_assertion": "...",
  "preconditions": ["..."],
  "related_sections": ["..."],
  "requirement_type": "SHALL|SHOULD|MAY",
  "chunk_id": "...",
  "section": "..."
}
```

The agent is instantiated as `Agent(name="Analyst", llm=get_llm(), system_prompt=..., tools=[search_norm_knowledge_base])` and run synchronously with `agent.run(prompt)`. The raw output is stripped of markdown fences with `re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())` before `json.loads()`. On `json.JSONDecodeError`, the agent is retried once with the instruction `"Your previous response was not valid JSON. Return only a raw JSON object."`. After two failures a `ValueError` is raised.

### 5.4 Agent 2 — RAG Router (`axis_b/agents/rag_router.py`, 109 lines)

**Role:** Enrich the Analyst's output with supporting context retrieved from the knowledge base.

**Tools available:** `search_norm_knowledge_base`, `get_chunks_for_section`

**Input:** the Analyst's JSON output dict.

**Expected JSON output:**
```json
{
  "supporting_clauses": ["..."],
  "cross_norm_refs": ["..."],
  "definitions": {"term": "definition"},
  "test_method_hints": "...",
  "full_context_summary": "...",
  "analyst_output": { ... }
}
```

The same fence-stripping and retry logic applies as in the Analyst.

### 5.5 Agent 3 — Designer (`axis_b/agents/designer.py`, 133 lines)

**Role:** Write a complete ISO 29119-3 test case from the RAG Router's context package.

**Tools available:** none (context-only agent).

**Input:** the RAG Router's full context package JSON.

**Output:** JSON matching the `TestCase` schema exactly. After `json.loads()`, the output is validated with `TestCase.model_validate(parsed)`. If validation fails, a `ValueError` is raised. The `actual_results` field is always `""`.

The same fence-stripping and retry logic applies. The Designer's retry prompt appends `"Your previous response was not valid JSON. Return only a raw JSON object."` — it does not identify which specific field failed validation.

### 5.6 Pipeline Orchestrator (`axis_b/pipeline.py`, 212 lines)

`run_pipeline(chunks, index_path, norm, output_path, plan_id)` is the public entry point:

1. Calls `init_tools(index_path)`.
2. Filters chunks to `is_normative=True` only.
3. For each normative chunk, runs `run_analyst → run_rag_router → run_designer` sequentially. Any exception is caught, logged, and skipped; the pipeline never aborts on a single bad chunk.
4. Collects all `TestCase` objects and passes them to `build_test_plan()`.
5. Groups test cases by `TestCase.feature_set`. For each group, builds one `TestCoverageItem` per test case, wraps them in a single `TestCondition`, and wraps that in a `FeatureSet`.
6. Validates the full plan with `TestPlan.model_validate(plan.model_dump())`. On failure, falls back to `_salvage_partial_plan()`, which re-validates each test case individually and drops invalid ones.
7. Writes the final plan as `plan.model_dump_json(indent=2)` to the output path.

At the time of this baseline description, no `TestPlan` JSON output has been generated because an Ollama instance with the required models was not yet available during the documented runs. The Axis A output (FAISS index + JSONL chunks) is fully produced and present on disk.

---

## 6. Axis C — Agentic Validation

### 6.1 Structural Validation (`axis_c/guardrails_validator.py`, 123 lines)

`validate_test_case(tc: TestCase) -> tuple[bool, list[str]]` runs before any LLM call.

The following checks are implemented as explicit Python conditionals:

1. `len(tc.inputs) >= 2` — minimum two `TestInput` entries.
2. `tc.priority in {"High", "Medium", "Low"}` — valid priority value.
3. `tc.requirement_type in {"SHALL", "SHOULD", "MAY"}` — valid modal type.
4. `tc.tc_id`, `tc.traceability`, and `tc.coverage_item_id` are all non-empty strings.
5. `tc.preconditions` is non-empty (one additional check beyond the SPEC).

String-length and format checks (objective length 10–500 chars, `tc_id` length 1–100 chars) are delegated to Guardrails AI when the library is available; the module falls back to equivalent plain-Python checks on `ImportError`.

### 6.2 Auditor Agent (`axis_c/agents/auditor.py`, 144 lines)

**Role:** For each `TestCase`, verify it against its source `RequirementChunk`; detect hallucinations, contradictions, and omissions.

**Pre-retrieval:** Before invoking the LLM, the auditor calls `search_norm_knowledge_base.invoke(tc.objective)` and `search_norm_knowledge_base.invoke(tc.inputs[0].expected_result)` and injects both results verbatim into the agent's prompt under labelled sections.

**LLM output fields:**
```json
{
  "hallucinations": ["..."],
  "contradictions": ["..."],
  "omissions": ["..."],
  "verdict": "PASS|FAIL|WARNING",
  "confidence": 0.0,
  "corrected_objective": "..." | null
}
```

**Verdict override (`compute_verdict`):** The LLM's `verdict` field is unconditionally overwritten by a deterministic Python function after parsing:
```python
if audit["hallucinations"]:  verdict = "FAIL"
elif audit["contradictions"]: verdict = "FAIL"
elif audit["omissions"]:      verdict = "WARNING"
else:                          verdict = "PASS"
```

The same fence-stripping and two-attempt retry logic applies as in Axis B agents.

---

## 7. Test Suite

The test suite contains **23 tests** across three files, all passing. All calls to `ChatOllama`, `OllamaEmbeddings`, and `FAISS` are mocked with `unittest.mock.patch` — no external services are required to run the suite.

| File | Tests | Coverage |
|---|---|---|
| `tests/test_axis_a_chunker.py` | 8 | `chunk_document()` on synthetic text: `section`, `is_normative`, `modals`, `page_start` |
| `tests/test_axis_b_schema.py` | 9 | Full `TestCase` round-trip; missing SHALL field raises `ValidationError`; `TestPlan` hierarchy |
| `tests/test_axis_c_auditor.py` | 6 | Verdict override: hallucination → FAIL; omission → WARNING; clean → PASS; `validate_test_case` boundary cases |

---

## 8. Known Limitations at Baseline

The following limitations exist in the baseline system and motivate the improvements described in `IMPROVEMENTS.md`:

1. **Single-pass flat chunking.** `RecursiveCharacterTextSplitter` is applied in one pass on the full document text. Normative clauses that exceed 512 tokens are split across two chunks, breaking the atomic unit. No mechanism exists to detect or prevent mid-clause splits.

2. **Dense-only retrieval.** The `search_norm_knowledge_base` tool uses FAISS cosine similarity exclusively. Short, keyword-specific normative clauses (e.g. field names, section identifiers) that do not survive embedding compression may not appear in top-k results.

3. **No cross-reference awareness.** Chunks are self-contained text windows. When a clause references another section (`"as specified in §7.1"`), the Analyst agent receives no context for that reference. The defined-terms section (§3 of ISO 18013-5) is not separately extracted.

4. **Independent per-chunk generation.** The three-agent chain runs per chunk with no shared state. Adjacent chunks covering the same clause from different angles may produce duplicate or near-duplicate test cases that are not detected until Axis C (if run).

5. **Non-reflective retry on validation failure.** When the Designer's output fails `TestCase.model_validate()`, the pipeline raises `ValueError` and skips the chunk. The retry prompt for `json.JSONDecodeError` does not identify which field failed or why, giving the model no targeted signal for correction.

6. **No generation-time quality gate.** Every test case that passes schema validation is accepted into the `TestPlan` regardless of its semantic distance from the source requirement. Low-confidence outputs are indistinguishable from high-confidence ones in the output artefact.

7. **LLM-only hallucination detection.** Axis C's Auditor relies solely on LLM judgment to detect hallucinations. No independent, non-LLM signal (e.g. embedding distance between the test case objective and its source chunk) is computed to calibrate or pre-screen the LLM's verdict.

8. **Manual adversarial protocol.** The adversarial test protocol specified in SPEC.md §6 is described as a one-time manual experiment. No automated, reproducible test implementation exists, making it impossible to track the `adversarial_detection_rate` metric across pipeline versions.

9. **No ground truth for Axis A evaluation.** `data/ground_truth/annotated_requirements.jsonl` does not exist. The `axis_a/evaluate.py` module is fully implemented but cannot be exercised end-to-end.

---

## 9. Codebase Metrics at Baseline

| Metric | Value |
|---|---|
| Total Python source files | 28 |
| Total source lines (excl. `.venv`) | 2,678 |
| Fully implemented files | 28 / 28 |
| Stubs / TODOs / `pass` bodies | 0 |
| Test count | 23 |
| Test pass rate | 100% |
| Chunk files produced | 2 (843 and 1,175 chunks) |
| FAISS index size | 3.5 MB (index) + 583 KB (docstore) |
| Source PDF | ISO/IEC 18013-5, 7.3 MB |
| TestPlan JSON outputs | 0 (requires Ollama runtime) |

---

*For the list of improvements proposed from the state of the art, see `docs/IMPROVEMENTS.md`.*
