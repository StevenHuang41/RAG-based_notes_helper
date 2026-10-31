# RAG Notes Helper — Agent Guide

## Repository orientation

- Run `git status --short` before modifying files. Preserve unrelated changes.
- Read `GLOSSARY.md` before modifying RAG, retrieval, or evaluation behavior.
- Consult relevant ADRs in `docs/adr/` before changing architecture governed by an accepted decision.
- Verify project-specific commands, paths, and behavior against the current checkout.

## Source of truth

- Treat the current codebase as the source of truth for existing behavior and approved specifications as the source of truth for intended changes.
- Preserve existing interfaces and architectural constraints unless an approved change explicitly requires otherwise.
- Identify conflicts between specifications, architectural decisions, and existing implementation before proceeding.

## Architecture invariants

- **Classic RAG** is the default single-retrieval workflow.
- **Agentic RAG** is a bounded, read-only retrieval workflow. It may search repeatedly, select evidence, and answer or abstain; do not expand it into a general-purpose autonomous agent without approval.
- Ground answers in retrieved evidence. Abstain when evidence is insufficient.
- Preserve per-query isolation. Do not introduce persistent cross-query memory or external write side effects without an approved design change.

## Evaluation invariants

- `evals/notes/` contains the version-controlled evaluation corpus; `evals/qa/notes_qa.json` contains the reviewed ground-truth QA set. Source paths are relative to the evaluation corpus root.
- `rag-app --eval` evaluates Classic RAG; `rag-app --agentic --eval` evaluates Agentic RAG.
- Both evaluation modes must use isolated temporary indexes and must not read or mutate `data/` or `storage/`.
- `evals/results/` contains generated, Git-ignored reports; preserve the existing report contract.
- Keep retrieval candidates separate from answer-generation context. Line-grounded Recall, Precision, and NDCG measure evidence; Ragas metrics are model-based diagnostics.
- Do not modify evaluation ground truth solely to improve scores.

## Development and verification

- Follow existing conventions and avoid unrelated refactoring.
- Keep Ragas and Torch imports lazy where evaluation runtime isolation requires it.
- Keep development-only evaluation dependencies out of production runtime dependencies unless explicitly approved.
- Run focused offline tests for local changes; run the full pytest suite for changes affecting shared interfaces or multiple subsystems.
- Verify relevant acceptance criteria and report executed tests, failures, and unverified behavior. Do not claim checks passed unless executed.

## Safety boundaries

- Do not edit `.env`, push, rewrite Git history, delete user data, or merge into `main` without explicit authorization.
- Do not remove existing worktrees or branches without authorization.
- Preserve unrelated user changes and request approval before destructive operations.

## Agent skills

### Issue tracker

Issues and specs live as files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. See `docs/agents/domain.md`.
