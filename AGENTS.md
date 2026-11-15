# RAG Notes Helper — Agent Guide

## Project Overview

A CLI tool for answering questions about local notes using RAG. Users can choose between Classic and Agentic workflows.

## Repository Structure

- `src/rag_notes_helper/`: application source code
- `tests/`: unit and integration tests
- `docs/`: maintained architecture docs, ADRs, and demo documentation
- `data/`: local notes
- `storage/`: embedding artifacts
- `evals/`: ground-truth dataset, QA and evaluation report results
- `logs/`: system logs

Follow the existing modular structure before introducing new directories.

## Environment

- Language: Python >=3.11
- Run Python and pytest commands through `uv run`.

## Coding Conventions

- Run `git status --short` before modifying files. Preserve unrelated changes
- Read `GLOSSARY.md` before modifying codebase
- Consult relevant ADRs in `docs/adr/` before changing architecture governed by an accepted decision
- Verify existing behavior, commands, and paths against the current checkout. Use the approved spec or ticket for intended changes.
- Surface conflicts among the request, spec, relevant ADRs, and existing behavior.
- Prefer small, modular functions with clear responsibilities
- Reuse existing abstractions before introducing new ones
- Avoid unnecessary dependencies and refactoring

## Testing & Validation

- Add or update tests for changed behavior.
- Run focused tests with `uv run pytest`; consider the full suite for changes affecting shared interfaces or multiple subsystems.
- Report commands run, failures, and unverified behavior. Do not claim checks passed unless they were executed.

## Evaluation

- Preserve isolation between the evaluation corpus (`evals/notes/`), reviewed QA set (`evals/qa/notes_qa.json`), and runtime data (`data/`, `storage/`). Evaluation uses temporary indexes and must not read or modify runtime data.
- When changing the evaluation corpus, review affected line-anchored QA evidence in the same change. See the Evaluation section in `README.md`.

## Boundaries & Constraints

- Do not modify unrelated files
- Do not edit `.env`, push, rewrite Git history, delete user data, or merge into `main` without explicit authorization
- Do not commit unless explicitly requested.
- Preserve unrelated user changes and request approval before destructive operations
- Do not silently suppress exceptions or test failures

## Agent workflows

- When creating or updating specs or tickets under `.scratch/`, follow `docs/agents/issue-tracker.md`.
- When assigning a triage status, use `docs/agents/triage-labels.md`.
- When exploring domain terminology or architectural decisions, consult `docs/agents/domain.md`.
