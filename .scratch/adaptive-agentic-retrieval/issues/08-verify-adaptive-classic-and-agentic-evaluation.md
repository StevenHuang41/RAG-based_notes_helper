# 08: Verify adaptive Classic and Agentic evaluation end to end

**What to build:** Complete an end-to-end acceptance pass across the integrated Classic and Agentic workflows, their configuration and architecture documentation, and their evaluation artifacts. Demonstrate with the same reviewed sample and seed that runtime outcomes are always non-empty, adaptive searches remain bounded and auditable, workflow errors remain visible, and generated reports agree with machine-readable results without touching runtime notes or persistent indexes.

**Blocked by:** 07: Judge applicable outcomes and report workflow reliability

**Status:** ready-for-agent

- [ ] All focused runtime, Agentic workflow, Hybrid retrieval, evaluation-runner, stage-report, and renderer tests pass with fake providers.
- [ ] The full offline test suite passes after all slices are integrated.
- [ ] Repository-configured lint, compilation, lockfile, and whitespace checks relevant to the changed code pass.
- [ ] Configuration examples and user documentation describe starting thresholds, fixed top-k ownership, bounded attempts, clarification, reason codes, and Classic-versus-Agentic behavior without calling Classic RAG LangChain.
- [ ] Accepted ADRs and the domain glossary are consistent with Retrieval reopening, adaptive thresholds, structured outcomes, and revised Ragas applicability.
- [ ] A Classic and Agentic evaluation use the same selected sample and seed against isolated evaluation assets.
- [ ] Both evaluation runs leave the runtime note corpus and persistent index unchanged.
- [ ] Every selected query has a non-empty response and an auditable normal or degraded outcome.
- [ ] Adaptive Agentic searches show their rewritten queries, thresholds, evidence modes, budgets, and complete stage provenance.
- [ ] Workflow reliability, workflow errors, Ragas applicability, stage metrics, and Final generation context agree between results and README artifacts.
- [ ] Live-provider Ragas scores are recorded as observations only and are not asserted as stable pass/fail thresholds.
- [ ] Executed commands, passing checks, failures, and any behavior that remains unverified are recorded in the ticket handoff.
