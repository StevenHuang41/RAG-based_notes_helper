# OpenRouter Provider, Agentic Progress, and Evaluation

Status: ready-for-agent

## Problem Statement

Notes Helper has an abstract generation-provider interface, but OpenRouter is not an available provider. Users who choose OpenRouter cannot use it for Classic RAG or Agentic RAG. Agentic RAG can also make several sequential model and retrieval calls without telling a terminal user which stage is running.

The repository has a fixed synthetic notes benchmark, but its current Ragas metrics use reference answers. The maintainer wants to preserve reviewed ground truth for repeatable checks while using reference-free Ragas diagnostics. The maintainer also wants Ragas-generated QA candidates to improve the fixed fixture without generating a new benchmark during every evaluation run.

## Solution

Add OpenRouter as a provider behind the existing LLM abstraction, and surface concise Agentic RAG stage messages in terminal sessions. Extend the evaluation setup so users can configure a Ragas judge provider and model, with safe fallback to the generation provider when omitted.

Keep the ground-truth QA fixture as the stable benchmark. Provide a one-time development workflow that generates QA candidates from the synthetic notes using Ragas, requires human review and source/evidence annotation, and adds approved cases to the fixture. During evaluation, compare Classic RAG and Agentic RAG on that fixed corpus, use deterministic ground-truth checks, and use reference-free Ragas metrics as additional diagnostics.

## User Stories

1. As a Notes Helper user, I want to select OpenRouter through the existing provider configuration, so that I can use OpenRouter models without changing the RAG workflow.
2. As a Notes Helper user, I want OpenRouter to use the same API key and model configuration pattern as the other providers, so that provider setup remains familiar.
3. As a Notes Helper user, I want to select an OpenRouter model by its provider/model identifier, so that I can choose a model from the OpenRouter catalog.
4. As a Notes Helper user, I want OpenRouter to support both normal generation and text streaming, so that Classic and Agentic answers retain the existing output behavior.
5. As a Notes Helper user, I want OpenRouter usage fields to be recorded when its API response provides them, so that evaluation can report provider-reported token costs consistently.
6. As a Notes Helper user, I want OpenRouter API failures to be surfaced through the existing error path, so that the system does not silently answer through a different provider.
7. As a Notes Helper user, I want the same OpenRouter configuration to work for Classic and Agentic RAG, so that enabling an agentic workflow does not require a separate provider setup.
8. As an Agentic RAG user, I want to see a progress stage while notes are searched, so that I know a multi-step query is still running.
9. As an Agentic RAG user, I want to see when the system is reviewing retrieved evidence and generating an answer, so that I can distinguish the workflow stages.
10. As an Agentic RAG user, I want an explicit stage message when the system abstains for insufficient evidence, so that an abstention is understandable.
11. As a terminal user, I want progress messages written to stderr only when stderr is a terminal, so that redirected output remains suitable for scripts and files.
12. As a terminal user, I want progress messages to contain stage names and search counts but no note text or chunks, so that private note content is not echoed as progress.
13. As a CLI user, I want stdout to remain reserved for the answer, so that progress reporting does not break piping or answer capture.
14. As an evaluator, I want to configure the Ragas judge provider and model independently from answer generation, so that I can control evaluation quality and cost.
15. As an evaluator, I want the judge provider and model to default together to the configured generation provider and model, so that evaluation works without duplicate configuration.
16. As an evaluator, I want a partial judge configuration to fail clearly, so that an evaluation provider is never paired with an unintended model.
17. As an evaluator, I want the generation API key reused only when the judge and generation providers match, so that credentials are not sent to the wrong provider.
18. As an evaluator, I want a separate judge API key available when the judge provider differs, so that generation and evaluation can use different services.
19. As an evaluator, I want the judge provider choices to match the generation provider choices, so that the configured generation provider can also serve as the default judge.
20. As an evaluator, I want reference-free Ragas diagnostics for answer grounding, answer relevance, and context utilization, so that Ragas scores do not consume the reference answers.
21. As an evaluator, I want expected source paths and evidence snippets checked deterministically, so that retrieval can be compared with ground truth without relying on chunk IDs.
22. As an evaluator, I want expected answerability and abstention checked deterministically, so that unanswerable and ambiguous cases remain measurable without asking Ragas to judge the abstention text.
23. As an evaluator, I want the reviewed reference answer to remain in each fixture case for human inspection, so that generated answers can be compared with the intended answer during review.
24. As a maintainer, I want Ragas test-set generation to produce candidate cases from the fictional notes corpus, so that I can expand QA coverage without using personal notes.
25. As a maintainer, I want generated QA candidates manually reviewed and annotated with source paths and evidence snippets before they enter the benchmark, so that generated content is not treated as ground truth automatically.
26. As a maintainer, I want approved Ragas-generated cases to augment the existing reviewed cases, so that established direct, multi-hop, unanswerable, and ambiguous regression coverage is preserved.
27. As a maintainer, I want test-set generation to run as a one-time development workflow, so that normal evaluation remains repeatable and does not create new questions on each run.
28. As a maintainer, I want Ragas test-set generation to operate on source documents without requiring the Notes Helper FAISS index, so that QA authoring and application retrieval remain separate concerns.
29. As an evaluator, I want each evaluation run to rebuild an isolated index from the fixed synthetic notes corpus, so that current retrieval behavior is tested without reading or modifying user notes.
30. As an evaluator, I want generation usage and Ragas judge usage reported separately, so that evaluation cost is not attributed to either RAG workflow.
31. As a maintainer, I want generated test-set drafts excluded from the fixed benchmark until they are reviewed, so that automated generation cannot silently change regression results.
32. As a maintainer, I want the benchmark and Ragas generation workflow to remain development-checkout assets, so that runtime installations do not gain test fixtures or a new runtime provider dependency.

## Implementation Decisions

### OpenRouter generation provider

- Register OpenRouter as an additional LLM provider using the existing provider configuration and factory.
- Implement an independent adapter over the existing OpenAI SDK and OpenRouter Chat Completions API. Do not refactor the existing OpenAI adapter, which uses a different API surface.
- Reuse the generic LLM API key setting and the existing model string. OpenRouter model identifiers use the provider/model form.
- Implement the existing generation and streaming contract, including prompt messages, maximum output tokens, temperature, text extraction, and provider-reported usage where available.
- Do not add a new SDK or a configurable OpenRouter endpoint in this change.
- Do not send OpenRouter app-attribution headers or add environment settings for them.
- Route errors through the existing provider error behavior. Do not silently fall back to another provider.

### Agentic terminal progress

- Emit progress from the Agentic workflow through an optional progress callback that the CLI can render.
- Show concise stages such as searching notes with the current search count, reviewing evidence, generating an answer, and abstaining for insufficient evidence.
- Write progress to stderr, flush it, and render it only when stderr is attached to a terminal.
- Keep answer text and answer streaming on stdout. Do not include retrieved note content, search queries, or evidence chunks in progress messages.
- Apply progress reporting to Agentic one-shot and Agentic REPL queries. Classic RAG output is unchanged.

### Ragas judge configuration and provider adapters

- Make the evaluation provider and model optional settings. If both are omitted, resolve both from the effective generation provider and model. If exactly one is supplied, fail configuration validation with a clear message.
- Use an evaluation API key when provided. Otherwise, reuse the generation API key only when the effective judge provider equals the generation provider. Require an evaluation key for a different non-Ollama provider. Ollama does not require an API key.
- Support the same provider choices for generation and judge configuration: Hugging Face, OpenAI, Ollama, Gemini, and OpenRouter. Each judge provider must be adapted to the Ragas LLM interface; sharing a provider name does not imply sharing the generation adapter implementation.
- Use the current Ragas LLM and metric interfaces rather than extending deprecated LangChain wrapper usage.
- Configure OpenRouter as an OpenAI-compatible Ragas client. Use provider-specific judge adapters for other providers, including a Ragas-compatible Hugging Face adapter.
- If a configured judge model cannot satisfy the structured-output requirements of a selected Ragas metric, report the failure. Do not silently switch models or providers.
- Keep evaluation embeddings as the existing local Hugging Face embedding model. The judge provider/model settings do not change the embedding model.

### Ground-truth fixture and reference-free evaluation

- Keep the existing synthetic notes corpus and reviewed QA cases as the benchmark seed. Add newly approved cases without removing the current direct, multi-hop, unanswerable, or ambiguous cases.
- Allow the fixed QA collection to grow beyond the initial 16 cases while retaining the current baseline coverage. Require unique case identifiers and the existing answerability, reference-answer, expected-source, and optional evidence fields.
- Keep reference answers in the fixture and per-case artifacts for human inspection. Do not pass reference answers to the Ragas judge.
- Use reference-free Ragas diagnostics for faithfulness, answer relevance, and context utilization. Do not run reference-dependent context recall, reference-based context precision, or AnswerCorrectness in this evaluation.
- Continue to use deterministic expected-source coverage, expected-evidence coverage, answerability, and abstention checks as ground-truth comparisons.
- Run Ragas answer/context diagnostics only for cases expected to be answerable. Evaluate unanswerable and ambiguous cases through deterministic evidence and abstention behavior.
- Preserve separate provider-reported generation and judge token usage in the timestamped evaluation artifacts. Missing provider usage remains unavailable and is never estimated from characters.

### Ragas test-set authoring

- Provide a development-only workflow that uses Ragas to generate candidate QA from the version-controlled fictional notes documents.
- Reuse the effective Ragas provider/model configuration and the local evaluation embedding model; do not introduce another provider/model configuration solely for one-time authoring.
- Use Ragas-generated questions, reference answers, and reference contexts as candidates. A maintainer reviews each candidate, verifies the answer against the source notes, and annotates expected source paths and evidence snippets before adding it to the fixed fixture.
- Store only reviewed cases in the benchmark fixture. Draft generations are not consumed by evaluation.
- Preserve the manually authored unanswerable and ambiguous cases because default Ragas generation primarily creates answerable single-hop and multi-hop candidates.
- Do not require or rebuild the Notes Helper FAISS index for candidate generation. Ragas builds its own knowledge graph and uses an LLM and embeddings over the supplied documents.
- Keep the existing evaluation behavior that copies the synthetic notes into temporary storage and builds an isolated Notes Helper index for each evaluation run.

## Testing Decisions

- Tests should assert observable provider, CLI, evaluation, and fixture behavior rather than private implementation details.
- Use fake OpenAI SDK clients to test OpenRouter request configuration, Chat Completions response parsing, streaming text, and usage normalization without network access. Follow the existing provider-usage tests as prior art.
- Exercise provider routing through the existing provider factory. Verify that both Classic and Agentic generation can use the configured OpenRouter adapter.
- Test evaluation settings for both omitted, fully specified, and partially specified provider/model pairs; same-provider key reuse; different-provider key requirements; explicit evaluation-key override; and Ollama's key exemption.
- Test Agentic progress at the CLI boundary with captured stdout/stderr and terminal/non-terminal streams. Assert stage messages appear only on terminal stderr, answers remain on stdout, and note content is absent from progress output.
- Test reference-free Ragas inputs using a fake worker. Assert that the judge receives question, response, and contexts but no reference answer; expected source/evidence and abstention checks still use the fixed ground truth.
- Test metric applicability: answerable cases receive reference-free Ragas scores; unanswerable and ambiguous cases retain deterministic scoring and abstention behavior without reference-based Ragas metrics.
- Test that Ragas-generated candidate records can be validated and mapped into the reviewed fixture schema using an offline sample result. Do not call live generation models in automated tests.
- Exercise the public evaluation runner with temporary fixture notes and storage. Verify paired Classic and Agentic runs use an isolated index and preserve separate generation and judge usage.
- Reuse the existing temporary notes/index isolation and Ragas-worker test seams. No automated test may read or modify user notes or require live credentials.

## Out of Scope

- LangGraph, MCP, LangChain-based Agentic orchestration, or changes to the bounded Python Agentic loop.
- Conversation memory shared between separate queries.
- OpenRouter attribution headers, application ranking setup, or attribution environment variables.
- External web retrieval, note mutation, or evaluation over a user's personal notes.
- Automatically trusting Ragas-generated answers or running test-set generation during --eval.
- Passing ground-truth reference answers to Ragas, reference-based answer correctness, or reference-based context recall in the benchmark.
- Changing the project's local FAISS index format or adding an application-index prerequisite to test-set generation.
- Shipping the synthetic notes, QA fixture, or test-set generation tooling in the runtime wheel.
- A formal guarantee that answers cannot contain unsupported claims or statistical benchmark claims from the seed dataset.

## Further Notes

- The existing Agentic RAG architecture remains governed by the bounded Python-loop decision: no LangGraph, LangChain agent framework, or MCP transport is introduced.
- OpenRouter uses OpenAI-compatible Chat Completions; the existing OpenAI generation adapter is not interchangeable because it calls the Responses API.
- Reference-free Ragas metrics are diagnostic signals, not ground truth. They do not establish answer correctness or retrieval completeness. The fixed source/evidence annotations and human review remain necessary.
- The initial hand-authored cases remain the regression seed. Ragas-generated cases are additions only after review, and each run records a fixture digest so results can be tied to the exact corpus and QA set.
- Test-set generation and evaluation are separate Ragas activities: generation creates candidate queries and references from documents; evaluation runs reference-free metrics over application outputs.
- The current evaluation runner rebuilds the Notes Helper index from the isolated fixture corpus during evaluation. Ragas test-set generation instead builds its own knowledge graph from documents and does not depend on that application index.
- References: [OpenRouter Quickstart](https://openrouter.ai/docs/quickstart), [Ragas RAG test-set generation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/), [Ragas Context Utilization](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/), [Ragas Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/).
