# Implementation Guide for Claude

You are generating a full, working Python research project from `SPEC.md`.
Read `SPEC.md` completely before writing a single line of code.
This file tells you **how** to implement it.

---

## Absolute Rules

- Generate **every file** listed in the directory structure in `SPEC.md §7`. No stubs, no `pass`, no `TODO`.
- Every function must be fully implemented and runnable.
- All Pydantic models use v2 API: `model_validate()`, `model_dump()`, never `.dict()` or `.parse_obj()`.
- All file paths use `pathlib.Path`, never raw strings.
- Never use `print()` — use `logging.getLogger(__name__)` everywhere.
- Python 3.11+. Type hints on every function signature.
- No internet calls at runtime except Ollama on `localhost:11434`.

---

## Implementation Order

Generate files in this exact order. Each step depends on the previous.

```
1.  requirements.txt
2.  axis_b/schema.py              ← all Pydantic models; everything imports from here
3.  axis_a/pdf_extractor.py
4.  axis_a/nlp_processor.py
5.  axis_a/chunker.py
6.  axis_a/indexer.py
7.  axis_a/evaluate.py
8.  axis_b/llm_setup.py           ← LLM instance + LangChain tools
9.  axis_b/agents/analyst.py
10. axis_b/agents/rag_router.py
11. axis_b/agents/designer.py
12. axis_b/pipeline.py
13. axis_c/agents/auditor.py
14. axis_c/guardrails_validator.py
15. scripts/run_axis_a.py
16. scripts/run_axis_b.py
17. scripts/run_axis_c.py
18. tests/test_axis_a_chunker.py
19. tests/test_axis_b_schema.py
20. tests/test_axis_c_auditor.py
```

Do not skip steps. Do not reorder steps.

---

## Per-File Implementation Notes

### `requirements.txt`
Pin major versions. Include: `pdfplumber`, `spacy`, `langchain`, `langchain-text-splitters`,
`langchain-ollama`, `langchain-community`, `faiss-cpu`, `deepagents`, `pydantic>=2.0`,
`guardrails-ai`, `pytest`, `bert-score`.

### `axis_b/schema.py`
- Implement all 6 models from `SPEC.md §3.2` in dependency order (innermost first):
  `TestInput → TestCase → TestCoverageItem → TestCondition → FeatureSet → TestPlan`
- `RequirementChunk` also lives here (not in `axis_a/`) so all modules import from one place.
- Add a `model_config = ConfigDict(extra="forbid")` to every model to catch schema drift early.

### `axis_a/pdf_extractor.py`
- Use `pdfplumber`. Extract tables first, then text with table bounding boxes filtered out.
- The `extract_pages()` function is a generator — do not load the entire PDF into memory.

### `axis_a/nlp_processor.py`
- Load spaCy model once at module level: `nlp = spacy.load("en_core_web_sm")`.
- Modal detection: match token text against the set in `SPEC.md §4` AND check POS in `{AUX, VERB}`.
- Section header regex must compile once at module level, not inside the loop.

### `axis_a/chunker.py`
- `RecursiveCharacterTextSplitter` config: use exact values from `SPEC.md §4.3`.
- `chunk_document()` must assign `page_start` by tracking page markers injected by `pdf_extractor`.
  Inject page markers as `\n\n<!-- PAGE {n} -->\n\n` in the concatenated text; parse them back in `chunk_document`.

### `axis_a/indexer.py`
- `build_index()` creates the FAISS index and calls `vectorstore.save_local(index_path)`.
- `load_index()` calls `FAISS.load_local(..., allow_dangerous_deserialization=True)` — this flag is required, do not omit it.
- Serialize all chunks to `jsonl` alongside the index for later reload without re-processing.

### `axis_b/llm_setup.py`
- Instantiate `ChatOllama(model="llama3.2", temperature=0.2, num_ctx=4096)` in a factory function `get_llm()`, not at module level, so tests can mock it.
- The two `@tool` functions hold a module-level `_vectorstore` reference initialized by `init_tools(index_path)`. Call `init_tools` once in `axis_b/pipeline.py` before running any agent.
- `get_chunks_for_section` iterates `vectorstore.docstore._dict.values()` — this is the correct internal API for `langchain_community.vectorstores.FAISS`.

### `axis_b/agents/analyst.py`, `rag_router.py`, `designer.py`
- Each agent is a function (`run_analyst`, `run_rag_router`, `run_designer`), not a class.
- DeepAgents `Agent` usage pattern:
  ```python
  from deepagents import Agent
  agent = Agent(name="...", llm=get_llm(), system_prompt="...", tools=[...])
  raw = agent.run(prompt)
  ```
- Agent output may be wrapped in markdown fences (` ```json ... ``` `). Strip them before `json.loads()`:
  ```python
  raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
  ```
- Retry loop: on `json.JSONDecodeError`, retry the agent call once with an appended instruction
  `"Your previous response was not valid JSON. Return only a raw JSON object."`.
  After 2 failures, raise `ValueError` with the chunk_id for the pipeline to catch and log.
- `run_designer` must call `TestCase.model_validate(json.loads(raw))` as the final step — if validation fails, raise `ValueError`.

### `axis_b/agents/designer.py` — Designer output contract
The Designer must produce JSON with exactly these keys (matching `SPEC.md §3.2 TestCase`):
```
tc_id, objective, priority, traceability, preconditions,
inputs, actual_results, requirement_type, coverage_item_id, feature_set, notes
```
`inputs` is a list of objects with keys: `input_number, action, expected_result`.
`actual_results` must always be `""` (empty string — placeholder for execution phase).

### `axis_b/pipeline.py`
- Call `init_tools(index_path)` once at the start.
- Filter chunks: `[c for c in chunks if c.is_normative]`.
- Wrap each chunk's agent chain in `try/except Exception as e: logger.error(...)` — the pipeline must never crash on a single bad chunk.
- Group `TestCase` objects by `feature_set` to build `FeatureSet` → `TestCondition` → `TestCoverageItem` hierarchy before constructing `TestPlan`.
  Use `feature_set` field from each `TestCase` as the grouping key.
- Write the final `TestPlan` as JSON to the output path with `plan.model_dump_json(indent=2)`.

### `axis_c/agents/auditor.py`
- Input: `TestCase` object + source `RequirementChunk` text.
- Must call `search_norm_knowledge_base` to verify at least the `objective` and first `inputs[0].expected_result` against the norm.
- Verdict logic (implement explicitly, do not leave to the LLM):
  ```python
  if audit["hallucinations"]: verdict = "FAIL"
  elif audit["contradictions"]: verdict = "FAIL"
  elif audit["omissions"]: verdict = "WARNING"
  else: verdict = "PASS"
  ```
  Override the LLM's `verdict` field with this computed value after parsing.

### `axis_c/guardrails_validator.py`
- Validate before the Auditor agent runs.
- Required checks (implement as explicit Python, not Guardrails for these):
  - `len(tc.inputs) >= 2`
  - `tc.priority in {"High", "Medium", "Low"}`
  - `tc.requirement_type in {"SHALL", "SHOULD", "MAY"}`
  - All of `tc.tc_id, tc.traceability, tc.coverage_item_id` are non-empty strings
- Use Guardrails AI only for string length / format checks beyond these basics.
- Return `(True, [])` on pass; `(False, list_of_violation_strings)` on failure.

### `scripts/run_axis_a.py`, `run_axis_b.py`, `run_axis_c.py`
- Use `argparse`. Every script must support `--help`.
- Scripts must exit with code `1` on error, `0` on success.
- Print a one-line summary to stdout on completion: e.g. `"[Axis A] Indexed 342 chunks."`.

### `tests/`
- Use `pytest`. No external services — mock `ChatOllama` and `OllamaEmbeddings` with `unittest.mock.patch`.
- `test_axis_a_chunker.py`: test that `chunk_document()` on a short synthetic text produces chunks with correct `section`, `is_normative`, and `modals` values.
- `test_axis_b_schema.py`: test that a fully populated `TestCase` serializes and deserializes to JSON without loss; test that a `TestCase` with missing SHALL field raises `ValidationError`.
- `test_axis_c_auditor.py`: test verdict override logic — inject a mock audit result with a hallucination and assert verdict is `"FAIL"` regardless of LLM output.

---

## Critical Integration Points

These are the most likely sources of bugs. Implement them carefully.

1. **`RequirementChunk.chunk_id` is the traceability key across the entire system.**
   The `TestCase.traceability` field must contain the `chunk_id` of the source chunk. Never generate a `tc_id` that doesn't correspond to a real `chunk_id`.

2. **FAISS index must be built before any agent runs.**
   `axis_b/pipeline.py` calls `init_tools(index_path)` which loads the FAISS index built by `axis_a/indexer.py`. If the index doesn't exist on disk, raise `FileNotFoundError` with a clear message.

3. **DeepAgents `Agent.run()` is synchronous.**
   Do not use `asyncio` or `await` with it. The pipeline loop is a standard `for` loop.

4. **`TestPlan` JSON output is the research artifact.**
   Before writing to disk, validate the full plan: `TestPlan.model_validate(plan.model_dump())`.
   If this raises, log the error and write a partial plan with only the valid test cases.

5. **spaCy model must be downloaded before running.**
   In `scripts/run_axis_a.py`, check `spacy.util.is_package("en_core_web_sm")` at startup and raise a clear error if missing: `"Run: python -m spacy download en_core_web_sm"`.

---

## What a Correct Implementation Looks Like

- `python scripts/run_axis_a.py --pdf data/input/iso_18013_5.pdf --norm "ISO/IEC 18013-5" --output-chunks data/output/chunks/iso_18013_5_chunks.jsonl --output-index data/output/index/faiss_index` runs without error and prints the chunk count.
- `python scripts/run_axis_b.py --chunks data/output/chunks/iso_18013_5_chunks.jsonl --index data/output/index/faiss_index --norm "ISO/IEC 18013-5" --output data/output/test_plans/TP-ISO-IEC-18013-5-001.json` produces a valid `TestPlan` JSON file.
- `pytest` passes all tests with no errors.
- `python -c "from axis_b.schema import TestPlan; print('ok')"` prints `ok` with no import errors.
