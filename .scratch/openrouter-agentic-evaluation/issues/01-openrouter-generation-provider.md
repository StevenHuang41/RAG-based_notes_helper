# 01: OpenRouter generation provider

**What to build:** Let users configure OpenRouter as the Notes Helper generation provider and use it end to end for Classic and Agentic RAG, including normal and streamed answers.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Users can select OpenRouter through the existing LLM provider setting, reuse the existing API key setting, and provide a model identifier in provider/model form.
- [x] The OpenRouter adapter uses the OpenAI SDK with the OpenRouter Chat Completions API, while leaving the existing OpenAI Responses adapter unchanged.
- [x] Both non-streaming and streaming calls preserve the existing prompt, output, temperature, and token-limit behavior.
- [x] Provider-reported input, output, and total usage are captured when available; missing usage remains explicitly unavailable.
- [x] Classic and Agentic RAG both route through the selected OpenRouter provider without provider-specific changes to either workflow.
- [x] API errors remain visible through the existing error path and never silently fall back to another provider.
- [x] No OpenRouter attribution headers or additional runtime SDK are introduced.
- [x] Provider documentation and configuration examples describe OpenRouter setup.
- [x] Automated tests use a fake SDK client and cover request configuration, response parsing, streaming, usage normalization, and provider routing without network access.

## Comments

- Resolved in commit 89d4716. OpenRouter generation and streaming use the provider abstraction and are covered by offline provider, usage, Classic, Agentic, and CLI tests (41 passed).
