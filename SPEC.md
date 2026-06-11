# Project Specification — Automatic Test Plan Generation from Technical Standards

## 1. Project Definition

- **Goal:** Pipeline that reads a technical standard (PDF) and outputs a structured test plan conforming to ISO/IEC/IEEE 29119-3:2021.
- **Input:** ISO/IEC 18013-5 PDF (target norm for prototype)
- **Output:** JSON + exportable PDF test plan (no executable code — high-level plan only)
- **Research gap:** No existing approach generates a test plan directly from a normative document upstream of any development.
- **3 axes, sequential dependency:** A → B → (C optional)
  - A: extract requirements from the norm into a searchable knowledge base
  - B: multi-agent system generates ISO 29119-3 test cases from Axis A output
  - C: agentic validation of the generated plan (hallucination, contradiction, omission detection)
- **Minimal viable scope:** Axis A + Axis B only.

---

## 2. Tech Stack

```
pdfplumber>=0.11          # PDF text + table extraction
spacy>=3.7                # NLP: tokenisation, POS tagging, sentence boundaries
en_core_web_sm            # spaCy model (python -m spacy download en_core_web_sm)
langchain>=0.2
langchain-text-splitters>=0.2
langchain-ollama>=0.3     # OllamaEmbeddings + ChatOllama
langchain-community>=0.3  # FAISS wrapper
faiss-cpu
deepagents                # multi-agent orchestration
pydantic>=2.0
pytest>=8
```

Ollama models required (local):
- `nomic-embed-text` — embeddings
- `llama3.2` — default chat model (swappable for sensitivity tests with `mistral`, `llama3.1:8b`)

---

## 3. Data Models

All models in `axis_b/schema.py`.

### 3.1 ISO 29119-3 Document Hierarchy (§7.3, Annex A.2.8, Annex B.1.7)

The standard defines this nesting: `TestPlan → FeatureSet → TestCondition → TestCoverageItem → TestCase`

This project implements the full hierarchy. Each level maps to a section of the source norm.

### 3.2 Models

```python
# --- Axis A (axis_a/chunker.py) ---

class RequirementChunk(BaseModel):
    chunk_id: str           # "<norm-slug>_<index:04d>"
    text: str
    section: str            # e.g. "7.2.1"
    page_start: int
    is_normative: bool      # True if contains SHALL/MUST/SHOULD/MAY
    modals: list[str]       # e.g. ["shall", "must not"]
    source_norm: str        # e.g. "ISO/IEC 18013-5"

# --- Axis B (axis_b/schema.py) ---

# ISO 29119-3 §7.3.5 — one numbered input+result pair within a test case
# Maps to §7.3.5.7 "Inputs" and §7.3.5.8 "Expected results" (parallel lists, numbered)
class TestInput(BaseModel):
    input_number: int           # sequential number within the test case
    action: str                 # §7.3.5.7 — action to bring item to testable state
    expected_result: str        # §7.3.5.8 — observable output for this specific action

# ISO 29119-3 §7.3.4 — intermediate layer between test condition and test case
# SHALL per Annex B §B.1.7.a
class TestCoverageItem(BaseModel):
    tci_id: str                 # e.g. "TCI-18013-5-007-001"
    description: str            # §7.3.4.3 — what is expected to be covered
    priority: Literal["High", "Medium", "Low"]   # §7.3.4.4
    traceability: str           # §7.3.4.5 — ref to test condition / feature set

# ISO 29119-3 §7.3.5 — core test case
# All fields marked SHALL in Annex B §B.1.7.b unless noted
class TestCase(BaseModel):
    tc_id: str                              # §7.3.5.2 SHALL — e.g. "TC-18013-5-<section>-<seq>"
    objective: str                          # §7.3.5.3 SHOULD — brief focus/title of the test case
    priority: Literal["High", "Medium", "Low"]  # §7.3.5.4 SHALL
    traceability: str                       # §7.3.5.5 SHALL — chunk_id + section ref from Axis A
    preconditions: list[str]                # §7.3.5.6 SHALL — required env state before execution
    inputs: list[TestInput]                 # §7.3.5.7 SHALL — min 2 inputs
    actual_results: str = ""               # §7.3.5.9 SHALL — placeholder filled during execution
    requirement_type: Literal["SHALL", "SHOULD", "MAY"]  # derived from source norm modal
    coverage_item_id: str                   # link to TestCoverageItem (traceability chain)
    feature_set: str                        # functional area of the norm (e.g. "Authentication")
    notes: str = ""

# ISO 29119-3 §7.2.5 — test condition (one verifiable item from the norm)
class TestCondition(BaseModel):
    tc_condition_id: str        # e.g. "TCOND-18013-5-007"
    description: str            # §7.2.5.3 — what can be tested
    priority: Literal["High", "Medium", "Low"]
    traceability: str           # §7.2.5.5 — ref to feature set and source norm section
    coverage_items: list[TestCoverageItem]
    test_cases: list[TestCase]

# ISO 29119-3 §7.2.4 — feature set (logical subset of test item)
class FeatureSet(BaseModel):
    fs_id: str                  # e.g. "FS-18013-5-007"
    objective: str              # §7.2.4.3
    priority: Literal["High", "Medium", "Low"]
    traceability: str           # §7.2.4.6 — ref to norm sections
    test_conditions: list[TestCondition]

# ISO 29119-3 §6.2 — top-level test plan document
class TestPlan(BaseModel):
    plan_id: str                # e.g. "TP-ISO-IEC-18013-5-001"
    title: str
    norm_reference: str         # e.g. "ISO/IEC 18013-5:2021"
    generation_date: str
    test_scope: str             # §6.2.4.3 — features in/out of scope
    assumptions: list[str]      # §6.2.4.4
    feature_sets: list[FeatureSet]
    coverage_sections: list[str]  # norm sections covered by at least one TC
```

### 3.3 Conformance Notes (from ISO 29119-3 Annex B.1.7)

| Field | ISO ref | Conformance |
|---|---|---|
| `tc_id` | §7.3.5.2 | SHALL |
| `objective` | §7.3.5.3 | SHOULD |
| `priority` | §7.3.5.4 | SHALL |
| `traceability` | §7.3.5.5 | SHALL |
| `preconditions` | §7.3.5.6 | SHALL |
| `inputs` (list of `TestInput`) | §7.3.5.7 | SHALL |
| `inputs[].expected_result` | §7.3.5.8 | SHALL |
| `actual_results` (placeholder) | §7.3.5.9 | SHALL |
| `coverage_item_id` → `TestCoverageItem` | §7.3.4, B.1.7.a | SHALL |

---

## 4. Axis A — Technical Requirements Extraction

**Input:** PDF file path, norm name string  
**Output:** `list[RequirementChunk]` + FAISS index on disk

### Pipeline (in order)

1. **PDF extraction** (`axis_a/pdf_extractor.py`)
   - Use `pdfplumber`; extract tables separately to avoid duplicate content in text flow
   - Yield one `RawPage(page_number, text, tables)` per page

2. **NLP annotation** (`axis_a/nlp_processor.py`)
   - Run `spacy` pipeline on each page text
   - Detect sentence boundaries
   - Flag sentences as normative if they contain modal verbs: `{shall, must, should, may, shall not, must not, need not}` with POS `AUX` or `VERB`
   - Detect section headers via regex `r'^(\d+(?:\.\d+)+)\s+\S'`

3. **Semantic chunking** (`axis_a/chunker.py`)
   - Use `RecursiveCharacterTextSplitter` with:
     ```
     separators = [
         r"\n(?=\d+(?:\.\d+)+\s)",  # section heading boundary
         "\n\n",
         "\n",
         ". ",
         " ",
     ]
     chunk_size    = 512
     chunk_overlap = 64
     is_separator_regex = True
     keep_separator     = True
     ```
   - Each chunk → `RequirementChunk`; set `is_normative` and `modals` from regex scan

4. **Embedding + indexing** (`axis_a/indexer.py`)
   - Embed with `OllamaEmbeddings(model="nomic-embed-text")`
   - Build `FAISS.from_documents(docs, embeddings)`
   - Persist index to `data/output/index/faiss_index/`
   - Metadata per doc: `chunk_id, section, page_start, is_normative, modals, source_norm`

### Evaluation (`axis_a/evaluate.py`)

- Ground truth: manually annotate ~200 clauses from sections 7–8 of ISO 18013-5 → `data/ground_truth/annotated_requirements.jsonl`
- For each ground-truth entry, query FAISS (top-k=5); hit = any result contains the annotated key phrase
- Metrics: `Precision`, `Recall`, `F1`
- Experiment: run same evaluation with naive `CharacterTextSplitter(chunk_size=512, separator="\n")` as baseline; report delta

---

## 5. Axis B — Agentic Test Plan Generation

**Input:** `list[RequirementChunk]` (normative only), FAISS index path, norm name  
**Output:** `TestPlan`

### Pipeline

```
RequirementChunk → Analyst Agent → RAG Router Agent → Designer Agent → TestCase
(repeat for all normative chunks, collect into TestPlan)
```

### LangChain Tools (defined in `axis_b/llm_setup.py`, used by agents)

```python
@tool
def search_norm_knowledge_base(query: str) -> str:
    # similarity_search(query, k=3) on FAISS; return top chunks with section metadata

@tool
def get_chunks_for_section(section_number: str) -> str:
    # iterate FAISS docstore, return all chunks where metadata.section starts with section_number
```

LLM: `ChatOllama(model="llama3.2", temperature=0.2, num_ctx=4096)`

### Agent 1 — Analyst (`axis_b/agents/analyst.py`)

- **Role:** Interpret a raw `RequirementChunk`; identify what the system must do, under what conditions, and what the testable assertion is
- **Tools:** `search_norm_knowledge_base`
- **Output (JSON):**
  ```
  requirement_summary, testable_assertion, preconditions: list,
  related_sections: list, requirement_type: SHALL|SHOULD|MAY, chunk_id, section
  ```

### Agent 2 — RAG Router (`axis_b/agents/rag_router.py`)

- **Role:** Enrich the Analyst's output with supporting context from the knowledge base
- **Tools:** `search_norm_knowledge_base`, `get_chunks_for_section`
- **Output (JSON):**
  ```
  supporting_clauses: list, cross_norm_refs: list, definitions: dict,
  test_method_hints: str, full_context_summary: str, analyst_output: dict
  ```

### Agent 3 — Designer (`axis_b/agents/designer.py`)

- **Role:** Write a complete ISO 29119-3 test case from the RAG Router's context package
- **Tools:** none (context-only)
- **Output:** JSON matching `TestCase` schema exactly (see §3.2); validated with `TestCase(**raw)`
- **Required output keys:** `tc_id, objective, priority, traceability, preconditions, inputs (list of {input_number, action, expected_result}), actual_results, requirement_type, coverage_item_id, feature_set`

### Orchestrator (`axis_b/pipeline.py`)

- Filter: process only chunks where `is_normative=True`
- For each chunk: run Analyst → RAG Router → Designer; catch and log exceptions per chunk without stopping the pipeline
- Collect all `TestCase` objects into a single `TestPlan`

### Evaluation

- **Sensitivity test:** run pipeline on same 20 chunks with `llama3.2`, `mistral:7b`, `llama3.1:8b`; compare BERTScore F1
- **Human evaluation:** score sample of 30 test cases on 5 dimensions (Completeness, Correctness, Clarity, Traceability, Executability) scale 1–5; target avg >= 3.5
- **ISO coverage:** `|sections with >= 1 test case| / |total normative sections|`; target >= 80%
- **Reference plan comparison:** ISO/IEC 18013-6 references an existing test plan for ISO 18013-5; use FAISS similarity to match generated TCs against existing ones; target >= 60% match

---

## 6. Axis C — Agentic Validation

**Input:** `TestPlan`, `list[RequirementChunk]`, FAISS index path  
**Output:** per-`TestCase` audit report JSON + global metrics

### Agent — Auditor (`axis_c/agents/auditor.py`)

- **Role:** For each `TestCase`, verify it against its source `RequirementChunk`; detect hallucinations, contradictions, omissions
- **Tools:** `search_norm_knowledge_base`
- **Output (JSON) per test case:**
  ```
  hallucinations: list[str], contradictions: list[str], omissions: list[str],
  verdict: PASS|FAIL|WARNING, confidence: float, corrected_objective: str|null
  ```

### Structural Validation (`axis_c/guardrails_validator.py`)

- Run before Auditor agent (cheap check first)
- Checks: `len(inputs) >= 2`, `priority in {High, Medium, Low}`, `requirement_type in {SHALL, SHOULD, MAY}`, all SHALL fields non-empty (`tc_id, traceability, preconditions, inputs, coverage_item_id`)
- Use Guardrails AI; return `(is_valid: bool, violations: list[str])`

### Metrics

```
hallucination_rate  = |TCs with >= 1 hallucination| / |total TCs|   target < 10%
contradiction_rate  = |TCs with >= 1 contradiction| / |total TCs|   target < 5%
omission_rate       = |TCs with >= 1 omission|      / |total TCs|   target < 20%
adversarial_detection_rate = |correctly flagged injections| / |total injections|  target >= 80%
```

### Adversarial Test Protocol

- Inject errors into 30 generated test cases: replace a factual claim (hallucination), flip an `expected_result` inside an `inputs` entry (contradiction), remove one `TestInput` entry (omission)
- Run Auditor; measure detection rate per error type
- Baseline: same 30 TCs reviewed by 2 human experts; compute Cohen's kappa vs. Auditor

---

## 7. Directory Structure

```
TestPlanGeneration/
├── data/
│   ├── input/iso_18013_5.pdf
│   ├── ground_truth/annotated_requirements.jsonl
│   └── output/
│       ├── chunks/iso_18013_5_chunks.jsonl
│       ├── index/faiss_index/
│       └── test_plans/TP-ISO-IEC-18013-5-001.json
├── axis_a/
│   ├── pdf_extractor.py
│   ├── nlp_processor.py
│   ├── chunker.py
│   ├── indexer.py
│   └── evaluate.py
├── axis_b/
│   ├── llm_setup.py
│   ├── schema.py
│   ├── pipeline.py
│   └── agents/
│       ├── analyst.py
│       ├── rag_router.py
│       └── designer.py
├── axis_c/
│   ├── agents/auditor.py
│   └── guardrails_validator.py
├── scripts/
│   ├── run_axis_a.py
│   ├── run_axis_b.py
│   └── run_axis_c.py
├── tests/
│   ├── test_axis_a_chunker.py
│   ├── test_axis_b_schema.py
│   └── test_axis_c_auditor.py
└── requirements.txt
```

---

## 8. CLI Commands

```bash
# Axis A — extract and index requirements
python scripts/run_axis_a.py \
  --pdf data/input/iso_18013_5.pdf \
  --norm "ISO/IEC 18013-5" \
  --output-chunks data/output/chunks/iso_18013_5_chunks.jsonl \
  --output-index data/output/index/faiss_index

# Axis B — generate test plan
python scripts/run_axis_b.py \
  --chunks data/output/chunks/iso_18013_5_chunks.jsonl \
  --index data/output/index/faiss_index \
  --norm "ISO/IEC 18013-5" \
  --output data/output/test_plans/TP-ISO-IEC-18013-5-001.json

# Axis C — validate generated plan
python scripts/run_axis_c.py \
  --plan data/output/test_plans/TP-ISO-IEC-18013-5-001.json \
  --chunks data/output/chunks/iso_18013_5_chunks.jsonl \
  --index data/output/index/faiss_index
```
