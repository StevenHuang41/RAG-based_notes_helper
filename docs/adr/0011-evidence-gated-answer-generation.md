# Gate Agentic Answer generation on Evidence coverage

Status: Accepted

Agentic Retrieval continues bounded Adaptive search for partial or zero pool coverage. If the final pool is assessed as zero evidence, return a deterministic no-related-information outcome and do not generate an answer. Select the generation top-k only for partial or sufficient pool coverage. When selection removes evidence, independently assess the exact ordered selected context; only partial or sufficient coverage permits Answer generation. This final review does not reopen Search. Pool and generation-context assessments remain distinct audit records. Controller judgments are not ground truth.

When Retrieval or generation-context review fails after its bounded validation retries, report a controller workflow error instead of generating with unknown coverage or claiming no evidence. Preserve retrieved evidence and failure category. Answer review and revision remain bounded and use the same context.

This amends ADR-0008's zero-evidence generation and degraded-review fallback and ADR-0009's degraded-review fallback. Their complete-pool review, context selection, independent budgets, and removal of the local context-window guard remain in effect. Initial empty-search handling is described by the current workflow documentation.
