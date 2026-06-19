# State-of-the-Art Improvement Propositions
## Automatic Test Plan Generation from Technical Standards

---

## 1. Introduction

This document proposes concrete improvements to each axis of the `TestPlanGeneration` pipeline, grounded in the 2020–2026 academic literature. Each proposition follows a structured three-part format:

- **Gap** — what the current SPEC (`SPEC.md`) leaves unaddressed relative to the research frontier.
- **SoTA Evidence** — the peer-reviewed or preprint works that motivate the change, with inline citations referencing the full bibliography in §6.
- **Concrete Proposition** — a focused description of what to add or modify, at the level of modules and data-model fields.

Papers were retrieved across eight search queries spanning: automatic test case generation from natural-language requirements, LLM-based test generation from SRS documents, RAG for requirements engineering, hallucination detection, ISO 29119 automation, NLP extraction from normative documents, multi-agent LLM testing, and semantic chunking for traceability. Sources queried: arXiv, Semantic Scholar, CrossRef.

---

## 2. Axis A — Requirements Extraction

### A1 — Structure-Aware Two-Pass Chunking

**Gap.**
`SPEC.md §4.3` configures `RecursiveCharacterTextSplitter` with a flat list of separators including a section-heading regex. This single-pass strategy treats all content uniformly: a 512-token window may split a normative clause ("The system SHALL...") mid-sentence if it falls near a chunk boundary, breaking the atomic unit that downstream agents must reason over. The current design has no mechanism to detect when a boundary falls inside a normative statement rather than between two independent clauses.

**SoTA Evidence.**
A systematic benchmark of 36 chunking strategies across six knowledge domains found that structure-aware chunking — where document hierarchy is used to define primary split boundaries before any size-based splitting is applied — achieves the highest nDCG@5 for legal and technical documents, outperforming naive fixed-size character splitting by over 85% [14]. The same study identifies that domain-specific structure (numbered headings, annexes, notes) is the single strongest predictor of retrieval quality for regulatory corpora. An independent adaptive-chunking study using five intrinsic document metrics (References Completeness, Intra-chunk Cohesion, Document Contextual Coherence, Block Integrity, Size Compliance) demonstrated that selecting the splitting strategy based on the document's own structural signals raised RAG answer correctness from 62–64% to 72% and increased the proportion of successfully answered questions by 30%, without changing the embedding model or the generation prompt [13]. For long-form structured documents, LumberChunker showed that dynamically identifying content-shift points before applying any size constraint outperforms all static baselines by 7.37% on retrieval DCG@20 [16]. Finally, a max–min semantic chunking algorithm that explicitly maximises within-chunk cohesion while minimising between-chunk similarity provided a principled, tunable alternative to heuristic separators for dense normative corpora [15].

**Concrete Proposition.**
Replace the single-pass splitter in `axis_a/chunker.py` with a two-pass strategy. In the first pass, split the concatenated document text exclusively on clause boundaries — defined as lines matching the existing section-header regex — so that each numbered clause (e.g. §7.2.1) is kept intact as a primary unit. In the second pass, apply `RecursiveCharacterTextSplitter` only to units that exceed the 512-token limit, preserving the full clause for shorter units. This guarantees that every normative SHALL/SHOULD clause is contained within a single chunk rather than straddling two. Add a boolean field `is_full_clause: bool` to `RequirementChunk` to record whether the chunk survived the first pass intact; this signal can be used downstream by the Analyst agent as a confidence indicator for its `testable_assertion` output.

---

### A2 — Hybrid Dense + Sparse Retrieval (FAISS + BM25)

**Gap.**
`SPEC.md §4.4` builds a FAISS index with `OllamaEmbeddings(model="nomic-embed-text")` and exposes a single `search_norm_knowledge_base` tool that calls `similarity_search(query, k=3)`. Dense vector retrieval excels at semantic similarity but systematically under-retrieves short, highly specific normative clauses (e.g. "The mDL SHALL include field family_name") whose exact terminology does not survive the embedding compression. Keyword-critical normative language — proper nouns, section identifiers, defined abbreviations — is precisely the content the Analyst and Designer agents need most.

**SoTA Evidence.**
Blended RAG, combining a dense FAISS index with a sparse BM25 encoder under a reciprocal-rank fusion (RRF) combiner, outperformed pure dense retrieval on SQUAD Q&A and TREC-COVID benchmarks, setting new state-of-the-art scores and exceeding fine-tuned model baselines [7]. The complementarity of dense and sparse signals is most pronounced for queries containing rare proper nouns and identifiers — exactly the vocabulary of ISO normative clauses. An iterative refinement layer (FAIR-RAG) that detects evidence gaps in the retrieved context before generation and re-issues a reformulated query achieved +8.3 F1 points on multi-hop reasoning benchmarks over the strongest non-iterative baseline [8]; the same gap-detection mechanism applies directly to the RAG Router agent's task of assembling a sufficient context package for the Designer. The systematic mapping of test case generation techniques confirms that retrieval quality is the primary bottleneck in NLP-to-test pipelines when the source documents are dense technical specifications [1].

**Concrete Proposition.**
In `axis_a/indexer.py`, build a BM25 index in parallel with the FAISS index, serialising it alongside the JSONL chunk file so both are available without reprocessing. In `axis_b/llm_setup.py`, rewrite `search_norm_knowledge_base` to query both indexes, fuse the ranked lists using reciprocal rank fusion (RRF with `k=60`), and return the top-3 fused results. Add a minimum-evidence guard to `axis_b/agents/rag_router.py`: if the fused result set contains zero chunks where `is_normative=True`, the RAG Router automatically issues a second query using `get_chunks_for_section` with the source chunk's `section` field before forwarding context to the Designer. This guard is a direct implementation of the FAIR-RAG evidence-gap detection principle [8] and ensures the Designer never operates on a context package that contains only non-normative definitional text.

---

### A3 — Cross-Reference Resolution and Defined-Terms Injection

**Gap.**
`SPEC.md §4.2` detects section headers and modal verbs within a page's text in isolation. ISO standards are heavily self-referential: a requirement in §7.2 may depend on a definition in §3, a precondition specified in §6, or a table in an annex. The current chunker produces self-contained text windows with no awareness of these dependencies. When the Analyst agent reads a chunk containing "as defined in §3.1" or "shall conform to the requirements of Table 1", it receives no context for those references and is forced to hallucinate or omit the referenced constraint.

**SoTA Evidence.**
AgenticIE, designed for key information extraction from multi-page EU regulatory documents, demonstrated that a planner–executor–corresponder architecture that explicitly resolves cross-document references achieves +16–26% Exact Match over static LLM baselines on a 15K-entity annotated regulatory dataset [12]. The study on NLP for requirements traceability found that embedding-based and transformer-based approaches are outperformed by hybrid approaches that explicitly inject definitional context, particularly when requirements span multiple abstraction levels of the source document [22]. The study extracting structured requirements from building technical specifications used Named Entity Recognition to identify defined terms and relation extraction to link them to their definitions, achieving over 90% F1 on normative entity extraction [24]. The work on extraction from domain normative documents validated that requirement-type classifiers trained on regulatory corpora significantly outperform general-domain classifiers, motivating domain-specific term handling [23].

**Concrete Proposition.**
Add a post-processing pass in `axis_a/chunker.py` that, after all chunks are produced, scans each chunk for cross-references matching the pattern `r'(?:§|clause|section|table|annex)\s*(\d[\d.]*)'`. For each matched reference, retrieve the chunk whose `section` field matches and append its first 150 characters as a `context_refs: list[str]` field in `RequirementChunk`. Additionally, extract the Terms and Definitions section (typically §3 in ISO standards) during PDF extraction and store it as a separate dictionary `defined_terms: dict[str, str]` in `RequirementChunk` metadata: for each term appearing in the chunk text, inject its short definition. These two additions require no changes to the FAISS index schema (both fields can be stored in LangChain document metadata) and give the Analyst agent grounded definitions without a retrieval call.

---

## 3. Axis B — Multi-Agent Test Generation

### B1 — Planner Agent for Cross-Chunk Deduplication and Feature-Set Skeleton

**Gap.**
`SPEC.md §5` runs the three-agent chain (Analyst → RAG Router → Designer) independently per normative chunk. Two problems follow directly from this independence. First, semantically overlapping chunks from adjacent sub-clauses (e.g. §7.2.1 and §7.2.2 both specifying behaviour of the same mDL field) will produce duplicate or near-duplicate test cases with different `tc_id` values and no cross-reference between them. Second, shared preconditions that apply to an entire section (e.g. "the mDL reader SHALL have established a secure channel") are regenerated from scratch by every Designer call rather than defined once and referenced. Neither problem is catchable by Axis C after the fact without significant rework of the generated plan.

**SoTA Evidence.**
ALMAS, a role-decomposed multi-agent framework aligned with software development lifecycle roles, showed that a dedicated planning agent that reads all work items before dispatching to specialist agents significantly reduces duplication and improves coherence of the resulting artefacts [20]. AutoMT, the closest single analogue to the present project — a multi-agent system that extracts metamorphic relations from traffic rule documents using RAG and generates test cases from them — uses an upfront relation-extraction pass over all source clauses before any test generation begins, achieving five times higher test diversity and detecting 20.55% more behavioural violations compared to per-clause-independent generation [18]. The multi-agent committee framework demonstrated that agent diversity combined with a pre-generation consensus protocol reduced hallucination-driven test failures from 22% to 10.5% in beta-testing workflows [19].

**Concrete Proposition.**
Add `axis_b/agents/planner.py` as a new module containing a `run_planner` function that executes once before the per-chunk loop in `axis_b/pipeline.py`. The Planner reads all normative chunks, groups them by cosine similarity of their embeddings into `feature_set` clusters (using the FAISS index already built in Axis A), and outputs a plan skeleton as a dictionary: `{feature_set_name: {"chunk_ids": [...], "shared_preconditions": [...], "section_range": "7.x–7.y"}}`. The pipeline passes the relevant skeleton entry to each Designer call, so the Designer populates `TestCase.preconditions` from the shared list rather than generating it independently. The `feature_set` field of every generated `TestCase` is taken from the Planner's cluster assignment, guaranteeing consistent grouping across the final `TestPlan` hierarchy.

---

### B2 — Schema-Constrained Generation with Error-Reflective Retry

**Gap.**
`SPEC.md §5` (AGENTS.md implementation notes) strips markdown fences and retries once on `json.JSONDecodeError`. This catches syntactically invalid JSON but not semantically invalid output: the LLM may produce well-formed JSON with the wrong types (`priority: "CRITICAL"` instead of `"High"`), missing nested objects (`inputs: []` with zero entries), or empty required strings (`tc_id: ""`). All of these pass `json.loads()` and then raise a Pydantic `ValidationError` that the current retry logic cannot handle — the retry prompt only says "return raw JSON", giving the model no information about what specifically failed.

**SoTA Evidence.**
A comparison of LLM approaches to test case generation from natural-language requirements found that schema-constrained generation — where the model is shown the exact JSON Schema before generating — reduced structurally invalid outputs by approximately 40% compared to unconstrained prompting [3]. The hybrid IoT test generation study showed that injecting domain-specific templates alongside LLM generation significantly improved traceability and format compliance over pure LLM approaches [4]. The IJIES 2026 study on RAG-LLM test case generation from SRS explicitly recommends structured-output prompting as a prerequisite for downstream traceability [5]. The LLM-based RTM generator demonstrated that multi-pass validation — where a validator agent returns the exact field name and constraint that was violated, not just a generic error signal — raised trace-link accuracy to 92.4% [21].

**Concrete Proposition.**
In `axis_b/agents/designer.py`, inject the full JSON Schema of `TestCase` (produced by `TestCase.model_json_schema()`) into the Designer's system prompt, formatted as a JSON block with inline comments explaining each field's allowed values and constraints. Replace the current single-retry logic with a two-step error-reflective retry: on `ValidationError`, parse the Pydantic error list and construct a correction prompt that names the specific failing field and its constraint (e.g. `"Field 'inputs' must contain at least 2 items; you returned 1. Add a second TestInput entry."`) before re-invoking the agent. This targeted feedback mirrors the multi-pass validation approach of [21] and requires no changes to the agent architecture or the LLM call interface.

---

### B3 — BERTScore Self-Evaluation Gate

**Gap.**
`SPEC.md §5` (Evaluation) plans BERTScore comparisons across models as an offline sensitivity test. No quality gate exists at generation time: every test case produced by the Designer, regardless of its semantic distance from the source requirement, is accepted into the `TestPlan`. The human evaluation target (average >= 3.5/5 on five dimensions) is therefore entirely dependent on LLM temperature and prompt quality, with no runtime signal that a particular test case is low-confidence.

**SoTA Evidence.**
The LLM comparison study for test case generation found that a self-consistency check — generating multiple candidate outputs and selecting the one most similar to the others by BERTScore — improved correctness by approximately 15% without any change to the base model [3]. Sampling-based consensus verification for hallucination detection in LLM code generation showed that comparing multiple sampled outputs provides a reliable hallucination signal: when consensus across samples is low (high variance), the probability of a factual error in the output is significantly higher [9]. The adaptive Bayesian semantic entropy method reduces the number of samples required for a reliable entropy estimate by approximately 50% versus fixed-budget sampling, making self-consistency evaluation practical in production pipelines [11].

**Concrete Proposition.**
In `axis_b/agents/designer.py`, generate two candidate `TestCase` objects per normative chunk by calling the Designer agent twice with slightly different temperature values (0.2 and 0.5). Compute BERTScore F1 between the two candidates' `objective` and `inputs[0].expected_result` fields. If the F1 score exceeds a configurable threshold (default: 0.80), select the candidate with the higher individual score against the source `RequirementChunk.text` and emit it. If the F1 falls below the threshold, emit the better candidate but set `notes: "low-confidence — BERTScore F1={value:.2f} — manual review recommended"`. This adds one LLM call per normative chunk and requires no external service, since `bert-score` is already listed in `requirements.txt`.

---

## 4. Axis C — Agentic Validation

### C1 — Embedding-Distance Pre-Audit

**Gap.**
`SPEC.md §6` has the Auditor agent use an LLM to detect hallucinations, contradictions, and omissions. The LLM judgment step is itself susceptible to the failure modes it is meant to detect: a model may fail to notice a hallucinated claim it would also produce. The verdict override logic in `AGENTS.md` is rule-based over the LLM's parsed output fields, which is sound — but the quality of those fields still depends entirely on a single LLM pass with no independent verification signal.

**SoTA Evidence.**
Probabilistic distance-based hallucination detection computes distances between the token embeddings of the prompt and those of the response in a RAG context, achieving state-of-the-art performance on hallucination detection without any task-specific training and with direct transferability from NLI tasks [10]. The sampling-based consensus approach demonstrated that embedding-level agreement across multiple outputs is a reliable proxy for factual accuracy in LLM-generated technical content [9]. The adaptive Bayesian entropy framework showed that a lightweight pre-screening pass that flags high-uncertainty outputs before invoking expensive validation reduces total validation cost by approximately 50% [11].

**Concrete Proposition.**
In `axis_c/agents/auditor.py`, add a pre-audit function `compute_embedding_distance(tc: TestCase, source_chunk: RequirementChunk) -> float` that embeds `tc.objective` and `source_chunk.text` using the same `OllamaEmbeddings` model used in Axis A, and returns their cosine distance. If the distance exceeds a configurable threshold (default: 0.55), set a `pre_audit_flag: bool = True` field in the audit report and inject the string `"WARNING: embedding distance between objective and source chunk is {value:.3f}, indicating possible semantic drift."` into the Auditor agent's prompt context before it runs. The Auditor's LLM prompt then has an independent signal to calibrate its hallucination judgment. The `pre_audit_flag` is recorded in the per-test-case audit report regardless of the final verdict, allowing quantitative analysis of how often the embedding distance correlates with the LLM-detected hallucination verdict.

---

### C2 — Automated Adversarial Test Protocol as a Reproducible pytest Suite

**Gap.**
`SPEC.md §6` specifies an adversarial test protocol (inject 30 errors of three types; run Auditor; measure detection rate; compute Cohen's kappa against human reviewers) as a manual, one-time experiment. This design makes the protocol non-reproducible: there is no guarantee that the same 30 injections are used across different LLM sensitivity tests, and there is no way to run it as part of a CI pipeline. The measurement of `adversarial_detection_rate` therefore cannot be automatically verified to remain above the 80% target as the pipeline evolves.

**SoTA Evidence.**
The OpenHalDet benchmark standardises hallucination detection evaluation across black-box, grey-box, and white-box detector families by encoding the full pipeline — from prompt construction to truthfulness annotation to metric computation — as a reproducible, parametrised test suite [OpenHalDet, arXiv:2606.06959, 2026]. Reference kappa values from that benchmark for good inter-rater agreement on hallucination detection tasks range from κ = 0.60 to κ = 0.75, providing a concrete calibration target for the Cohen's kappa comparison against human experts specified in `SPEC.md §6`.

**Concrete Proposition.**
Implement the adversarial protocol in `tests/test_axis_c_auditor.py` as a `pytest.mark.parametrize` suite. Encode the three injection types as pure Python transformation functions applied to `TestCase` objects: `inject_hallucination(tc, fake_claim)` replaces one sentence in `tc.objective` with a fabricated technical claim; `inject_contradiction(tc)` flips the `expected_result` of `tc.inputs[0]`; `inject_omission(tc)` removes `tc.inputs[-1]`. Generate the 30 adversarial cases deterministically from a fixed seed applied to a fixed slice of the generated test plan, so the suite is fully reproducible across runs. Assert that the Auditor returns verdict `"FAIL"` for all hallucination and contradiction injections and at minimum `"WARNING"` for omission injections. The Cohen's kappa calculation against human annotations is added as a separate script `scripts/compute_kappa.py` that reads a human-annotation CSV and the Auditor's JSON output, keeping it out of the automated test suite while remaining reproducible.

---

## 5. Cross-Cutting — TraceLLM-Style Traceability Enrichment

**Gap.**
`SPEC.md §3.2` defines `TestCase.traceability` as a string containing the source `chunk_id`. This field is populated by the Designer agent based on the chunk passed to it, but neither the Analyst nor the Designer's system prompt explicitly frames their role in the traceability chain. As a result, the LLM may produce objectives or expected results that are accurate paraphrases of the source norm but are not grounded in the specific clause identified by `chunk_id` — a subtle form of traceability drift that Axis C hallucination detection may not catch because the content is technically correct but sourced from general knowledge rather than the cited clause.

**SoTA Evidence.**
TraceLLM demonstrated that enriching each LLM prompt with a contextual role description — explicitly stating the agent's position in the traceability chain and the domain knowledge it is operating over — achieved state-of-the-art requirements traceability F2 scores across four datasets (aerospace, healthcare, requirements, and test cases), outperforming fine-tuned baselines using prompt engineering alone [6]. The LLM-based RTM generator achieved 92.4% trace-link accuracy using semantic-context block segmentation combined with explicit traceability-chain prompting [21]. The NLP survey on requirements traceability confirmed that embedding-based methods combined with contextual role injection outperform pure IR approaches for technical specification corpora [22].

**Concrete Proposition.**
Extend the system prompts of all three agents in `axis_b/agents/` with a traceability role header. The Analyst's prompt should open with: `"You are the first agent in a three-stage traceability chain. Your output will be cited in a formal test plan under ISO 29119-3. Every claim you make must be directly derivable from the provided chunk (chunk_id: {chunk_id}, section: {section}, norm: {source_norm}). Do not introduce information from general knowledge."` The RAG Router's prompt should identify itself as enriching context only from retrieved norm chunks, with retrieved chunks cited by their `chunk_id`. The Designer's prompt should close with: `"Set the 'traceability' field to exactly '{chunk_id}'. Do not generate a tc_id that does not correspond to this chunk."` These additions require no code changes to agent architecture and no new dependencies; they are pure prompt engineering changes to three string constants.

Additionally, add a `traceability_confidence: float` field to the Auditor's per-test-case output in `axis_c/agents/auditor.py`, computed as the cosine similarity between the embedding of `tc.objective` and the embedding of the source `RequirementChunk.text`. This makes traceability quality quantitatively measurable across the full test plan and directly comparable across LLM sensitivity runs.

---

## 6. Bibliography

[1] A. Rodrigues, J. Vilela, and C. Silva, "A Systematic Mapping Study on Techniques for Generating Test Cases from Requirements," in *Proc. 9th Int. Conf. Evaluation of Novel Approaches to Software Engineering (ENASE)*, SCITEPRESS, 2024, pp. 141–148. DOI: 10.5220/0012551900003705.

[2] M. Krishna, B. Gaur, A. Verma, and P. Jalote, "Using LLMs in Software Requirements Specifications: An Empirical Evaluation," arXiv preprint arXiv:2404.17842, 2024. *(Presented at IEEE Requirements Engineering Conf. — RE 2024.)*

[3] B. R. Korraprolu, P. Pinninti, and Y. R. Reddy, "Test Case Generation for Requirements in Natural Language — An LLM Comparison Study," in *Proc. 18th Innovations in Software Engineering Conf. (ISEC)*, ACM, 2025, pp. 1–5. DOI: 10.1145/3717383.3717389.

[4] Z. Chenail-Larcher, J. B. Minani, and N. Moha, "Test Generation from Use Case Specifications for IoT Systems: Custom, LLM-Based, and Hybrid Approaches," in *2025 IEEE Conf. Software Testing, Verification and Validation (ICST)*, IEEE, 2025, pp. 597–602. DOI: 10.1109/icst62969.2025.10988996.

[5] *(Authors unlisted in CrossRef record)*, "Leveraging Retrieval-Augmented LLMs for Automated Test Case Generation from Software Requirements Specification," *Int. J. Intelligent Engineering and Systems (IJIES)*, vol. 19, no. 1, pp. 52–66, 2026. DOI: 10.22266/ijies2026.0131.04. *(Full author list available at: https://inass.org/wp-content/uploads/2025/09/2026013104-3.pdf)*

[6] N. Alturayeif, I. Ahmad, and J. Hassine, "TraceLLM: Leveraging Large Language Models with Prompt Engineering for Enhanced Requirements Traceability," arXiv preprint arXiv:2602.01253, 2026. *(Targeting Requirements Engineering journal.)*

[7] K. Sawarkar, A. Mangal, and S. R. Solanki, "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers," in *2024 IEEE 7th Int. Conf. Multimedia Information Processing and Retrieval (MIPR)*, IEEE, 2024, pp. 155–161. DOI: 10.1109/MIPR62202.2024.00031. arXiv: 2404.07220.

[8] M. Aghajani Asl, M. Asgari-Bidhendi, and B. Minaei-Bidgoli, "FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation," arXiv preprint arXiv:2510.22344, 2025.

[9] T. Huang, Z. Ren, Y. Huang, X. Chen, Y. Liu, and Z. Zheng, "Hallucination Detection in LLM Code Generation: A Sampling-Based Consensus Verification Approach," *Automated Software Engineering*, vol. 33, no. 2, 2026. DOI: 10.1007/s10515-026-00605-0.

[10] R. Oblovatny, A. Kuleshova, K. Polev, and A. Zaytsev, "Probabilistic Distances-Based Hallucination Detection in LLMs with RAG," arXiv preprint arXiv:2506.09886, 2025. *(Published in Zapiski Nauchnykh Seminarov POMI, vol. 541, 2025.)*

[11] Q. Sun, X. Li, X. He, A. Cheng, X. Ji, H. Lu, R. Huang, and Q. Hu, "Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration," arXiv preprint arXiv:2603.22812, 2026. *(Accepted at AAAI 2026.)*

[12] G. Colakoglu, G. Solmaz, and J. Fürst, "AgenticIE: An Adaptive Agent for Information Extraction from Complex Regulatory Documents," arXiv preprint arXiv:2509.11773, 2025. *(v3 dated January 2026.)*

[13] P. R. de Moura Júnior, J. Lelong, and A. Blangero, "Adaptive Chunking: Optimizing Chunking-Method Selection for RAG," arXiv preprint arXiv:2603.25333, 2026.

[14] M. A. Shaukat, M. Adnan, and C. C. N. Kuhn, "A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity," arXiv preprint arXiv:2603.06976, 2026. DOI: 10.48550/arXiv.2603.06976.

[15] C. Kiss, M. Nagy, and P. Szilágyi, "Max–Min Semantic Chunking of Documents for RAG Application," *Discover Computing*, vol. 28, no. 1, 2025. DOI: 10.1007/s10791-025-09638-7.

[16] A. V. Duarte, J. Marques, M. Graça, M. Freire, L. Li, and A. Oliveira, "LumberChunker: Long-Form Narrative Document Segmentation," in *Findings of the Association for Computational Linguistics: EMNLP 2024*, ACL, 2024, pp. 6473–6486. arXiv: 2406.17526.

[17] *(See [7].)*

[18] L. Liang, C. Tan, Y. Deng, Y. Cai, T. Y. Chen, and X. Zheng, "AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems," arXiv preprint arXiv:2510.19438, 2025. DOI: 10.48550/arXiv.2510.19438.

[19] S. B. H. Karanam and D. A. Kennady, "Multi-Agent LLM Committees for Autonomous Software Beta Testing," arXiv preprint arXiv:2512.21352, 2025. DOI: 10.48550/arXiv.2512.21352.

[20] V. Tawosi, K. Ramani, S. Alamir, and X. Liu, "ALMAS: An Autonomous LLM-Based Multi-Agent Software Engineering Framework," in *2025 40th IEEE/ACM Int. Conf. Automated Software Engineering Workshops (ASEW)*, IEEE, 2025, pp. 287–290. DOI: 10.1109/ASEW67777.2025.00059.

[21] M. S. Thangakrishnan, K. Somasundaram, M. G. Nayagam, M. Amanullah, and S. E. V. Dinesh, "LLM-Based Requirement Traceability Matrix Generator from SRS Documents," in *2025 IEEE 3rd Global Conf. Wireless Computing and Networking (GCWCN)*, IEEE, 2025, pp. 1–8. DOI: 10.1109/GCWCN66157.2025.11448459.

[22] J. L. C. Guo, J.-P. Steghöfer, A. Vogelsang, and J. Cleland-Huang, "Natural Language Processing for Requirements Traceability," arXiv preprint arXiv:2405.10845, 2024.

[23] I. Baimuratov, D. Turygin, I. Shilin, D. Pliukhin, and D. Mouromtsev, "Extraction of Requirement Bases from Domain Normative Documents and Classifiers with Application to the Russian Building Code," *Lobachevskii Journal of Mathematics*, vol. 44, no. 1, pp. 97–110, 2023. DOI: 10.1134/S1995080223010031.

[24] I. Nahri, R. Pinquié, P. Véron, N. Bus, and M. Thorel, "Extracting Structured Requirements from Unstructured Building Technical Specifications for Building Information Modeling," arXiv preprint arXiv:2508.13833, 2025.

[25] A. Pereira, B. Lima, and J. P. Faria, "APITestGenie: Automated API Test Generation through Generative AI," arXiv preprint arXiv:2409.03838, 2024. *(Published in IEEE Software special issue on next-generation software testing; DOI pending final assignment.)*

[OpenHalDet] X. Li et al., "OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios," arXiv preprint arXiv:2606.06959, 2026.
