# 03: Configurable Ragas judge providers

**What to build:** Let evaluators select a Ragas judge provider and model independently from answer generation, with safe defaults and credentials for every supported generation provider.

**Blocked by:** 01: OpenRouter generation provider.

**Status:** ready-for-agent

- [ ] The evaluation provider and model are optional settings and support the same provider choices as generation: Hugging Face, OpenAI, Ollama, Gemini, and OpenRouter.
- [ ] When both evaluation settings are absent, the effective judge provider and model resolve together to the generation provider and model.
- [ ] If exactly one of the evaluation provider or model is set, configuration fails with a clear validation message.
- [ ] An explicit evaluation API key takes precedence. Without one, the generation API key is reused only when the judge provider matches the generation provider.
- [ ] A different non-Ollama judge provider requires an evaluation API key; Ollama remains keyless.
- [ ] Ragas uses its current LLM interface and provider-specific adapters, including an OpenRouter-compatible client and a Hugging Face judge adapter.
- [ ] A judge model that cannot satisfy a metric's structured-output requirements fails visibly; the system does not switch models or providers silently.
- [ ] Evaluation artifacts identify the effective non-secret judge provider and model and keep judge usage separate from generation usage.
- [ ] Automated tests cover provider/model fallback, partial configuration, API-key rules, all supported provider routes, and unavailable usage without live credentials.
