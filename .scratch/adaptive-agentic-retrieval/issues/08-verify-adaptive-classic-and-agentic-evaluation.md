# 08: Verify adaptive Classic and Agentic evaluation end to end

**What to build:** Complete an end-to-end acceptance pass across the integrated Classic and Agentic workflows, their configuration and architecture documentation, and their evaluation artifacts. Demonstrate with the same reviewed sample and seed that runtime outcomes are always non-empty, adaptive searches remain bounded and auditable, workflow errors remain visible, and generated reports agree with machine-readable results without touching runtime notes or persistent indexes.

**Blocked by:** 07: Judge applicable outcomes and report workflow reliability

**Status:** resolved

- [x] All focused runtime, Agentic workflow, Hybrid retrieval, evaluation-runner, stage-report, and renderer tests pass with fake providers.
- [x] The full offline test suite passes after all slices are integrated.
- [x] Repository-configured lint, compilation, lockfile, and whitespace checks relevant to the changed code pass.
- [x] Configuration examples and user documentation describe starting thresholds, fixed top-k ownership, bounded attempts, clarification, reason codes, and Classic-versus-Agentic behavior without calling Classic RAG LangChain.
- [x] Accepted ADRs and the domain glossary are consistent with Retrieval reopening, adaptive thresholds, structured outcomes, and revised Ragas applicability.
- [x] A Classic and Agentic evaluation use the same selected sample and seed against isolated evaluation assets.
- [x] Both evaluation runs leave the runtime note corpus and persistent index unchanged.
- [x] Every selected query has a non-empty response and an auditable normal or degraded outcome.
- [x] Adaptive Agentic searches show their rewritten queries, thresholds, evidence modes, budgets, and complete stage provenance.
- [x] Workflow reliability, workflow errors, Ragas applicability, stage metrics, and Final generation context agree between results and README artifacts.
- [x] Live-provider Ragas scores are recorded as observations only and are not asserted as stable pass/fail thresholds.
- [x] Executed commands, passing checks, failures, and any behavior that remains unverified are recorded in the ticket handoff.

## Comments

- Documentation and domain reconciliation were implemented in `d4172ed`. No runtime behavior changed in this slice, so no additional TDD production-code cycle was required.
- Fake-provider verification: focused runtime/query, Agentic, Hybrid retrieval, evaluation-runner, stage-report, renderer, and Ragas tests passed (134); the full offline pytest suite passed (237).
- Static verification: `python -m compileall -q src tests`, `UV_CACHE_DIR=/private/tmp/rag-adaptive-08-uv-cache uv lock --check --offline`, and `git diff --check` passed. The first lock check failed only because sandbox policy denied access to `~/.cache/uv`; the task-specific cache rerun passed. This repository does not configure Ruff, Mypy, or another standalone lint command beyond CI pytest.
- README, `.env.example`, GLOSSARY, and ADR status now agree on human-owned fixed top-k values, configurable starting thresholds, validated adaptive searches, replace/extend evidence modes, independent three-Search/three-Answer budgets, clarification, stable reason codes, Retrieval reopening, structured outcomes, and outcome/input-based Ragas applicability. Classic RAG remains a fixed pipeline and is not described as LangChain.
- Manual acceptance used the existing local Ollama generation/judge configuration (`gemma4:e2b`) without editing or copying `.env`. Both workflows used `--sample 2 --seed 42` and selected `direct-08` plus `ambiguous-12` from the isolated evaluation assets.
- Classic artifacts: `evals/results/2026-11-03-235153`; schema 8, run `completed`, workflow `completed`, zero workflow errors, reliability 2/2. Outcomes were non-empty `answered` and `insufficient_evidence` responses. All four Ragas diagnostics were applicable because both cases had Final generation context.
- Agentic artifacts: `evals/results/2026-11-03-235412`; schema 8, run `completed`, workflow `completed_with_errors`, two workflow errors, reliability 0/2. Both outcomes were non-empty deterministic `controller_failure` responses after the live controller remained invalid following its one validation retry (`AgenticWorkflowError`). Answer Relevancy and Answer Correctness were still scored; Faithfulness and Context Utilization correctly remained unavailable with explicit no-Final-context reasons.
- The two machine results and README reports agree on selected IDs, run/workflow statuses, reliability, errors, Search registries, thresholds, Vector/BM25/Fusion/Rerank/Generation rows, Ragas applicability, and Final generation context. Every live Search preserved effective thresholds and `replace` evidence mode; run metadata reported the 3/3 Agentic budgets.
- The live Agentic sample did not reach a rewritten second search because both controller calls degraded before a valid action. Rewritten-query adaptation, raised/lowered thresholds, replace/extend composition, budget enforcement, and multi-search provenance remain verified by the passing fake-provider public-seam tests rather than this live observation.
- Live Ragas values are retained only in the generated artifacts as provider-dependent observations; no score was used as a pass/fail threshold.
- Runtime `data/` and `storage/` were unchanged: the combined content hash remained `75f0dff11bb26eec30fc0b3f489642c366f5693b44c487b005ad8b4be5acfe12`, and the path-set hash remained `2a72489edfe807f1dd45ba33a21b1887054eb21239bb6b766dac39337a3bfcca` before and after both evaluations.
