Type: task
Status: ready-for-agent
Spec: ../spec.md

# Implement optional Agentic RAG and paired notes evaluation

Implement the behavior and decisions in the linked spec.

## Acceptance criteria

- Classic remains the default; --agentic selects the new workflow independently of --repl.
- Agentic RAG searches with the original question first, lets a Pydantic-validated JSON controller choose follow-up search, selected-evidence answer, or abstain, and never searches more than three times per query.
- Invalid actions get one retry and then return a visible workflow error without Classic fallback.
- Both workflows abstain without an LLM call when retrieval has no chunks; Agentic citations use only validated, selected evidence.
- The synthetic 16-case benchmark is isolated from data/, compares both workflows, and is kept outside the runtime wheel.
- --eval writes a UTC-stamped root eval/ artifact directory with a run log, per-case results, summary metrics, and available provider token usage; judge usage and character counts remain separate.
- Tests cover CLI dispatch, bounded Agentic behavior, evidence validation, empty retrieval, and fixture-backed evaluation using isolated temporary notes and storage.

The 16-case benchmark is a regression seed, not a statistical quality claim. See the spec for categories, metrics, artifact details, and additional failure behavior.
