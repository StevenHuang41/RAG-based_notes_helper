# Agentic RAG for Notes Helper

## Problem Statement

目前 Notes Helper 只有 Classic RAG：對問題做一次 retrieval，然後把結果交給 LLM 回答。使用者無法讓系統根據第一輪 evidence 再提出搜尋、補足跨筆記資訊，或在 evidence 不足時明確結束。

這項變更要新增可選的 Agentic RAG，同時保留 Classic RAG 作為預設。它必須只讀取已索引的本機 notes，以 notes evidence 為回答依據，保留現有 citations，並在 evidence 不足時 abstain。產品目標是減少無 evidence 支持的回答；任何 prompt、schema 或 evaluation 都不能形式化保證 LLM 永不產生錯誤主張，因此 faithfulness 必須持續量測，且介面不得宣稱「零 hallucination」。

目前沒有可重複使用的 notes QA ground truth。既有 eval 資料集沒有固定 notes corpus 與預期 evidence，無法可靠比較兩種 workflow，也不應把使用者的 data/ 當成公開或固定的 benchmark。

## Solution

增加一條 opt-in 的 Agentic RAG 路徑，使用 bounded Python loop、Pydantic 驗證的 JSON actions，以及既有 retrieval 與 answer generation 能力。Classic 與 Agentic 是 RAG workflow；one-shot 與 REPL 是互動方式，兩者獨立選擇。

Agentic query 流程：

1. 使用原始問題呼叫唯讀的 Note Search Tool；不先多做一次 planning LLM call。
2. 若該次搜尋沒有 chunks，立即回傳 deterministic abstention，不呼叫 controller 或 answer LLM。
3. 有 evidence 時，controller 收到原始問題、已去重的累積 chunks、目前搜尋次數與剩餘搜尋額度。controller 會閱讀 chunks 並推理 evidence 是否足夠，再輸出一個 JSON action：
   - search：提出精煉後的新 query。
   - answer：選出要交給最終回答的 evidence references。
   - abstain：表示現有 evidence 不足以回答。
4. Pydantic 驗證 action 的 JSON 結構、欄位與型別；應用程式另外驗證 answer 所選的 references 確實存在於本次累積搜尋結果。Pydantic 不替模型判斷 chunks 的語意相關性或答案是否正確。
5. search 會再次呼叫 Note Search Tool，將新結果去重後交由 controller 判斷。整個 query 最多搜尋三次，包含第一次原始問題搜尋。搜尋額度用完後只接受 answer 或 abstain。
6. answer 會把有效、去重且由 controller 選定的 chunks 傳給既有 rag_answer() 做最後生成；citations 由這組 chunks 產生。abstain 不呼叫最終回答生成。
7. JSON parse 或 validation 失敗時，最多重試一次。再次失敗時回傳清楚的 Agentic workflow error，不 fallback 到 Classic，也不把程式錯誤冒充 evidence不足。

Evaluation 使用版控中的 synthetic notes corpus 與 16 筆人工審核 QA fixture。每筆 QA 同時跑 Classic 與 Agentic，記錄逐題結果、aggregate metrics、provider usage 與本機 artifact。benchmark 不讀取或修改使用者的 data/。

## User Stories

1. As a CLI user, I want a one-shot query to use Classic RAG by default, so that existing invocations keep their established behavior.
2. As a CLI user, I want to opt into Agentic RAG for one query with --agentic, so that I can compare workflows without changing configuration files.
3. As a CLI user, I want --repl to start a Classic REPL by default, so that the existing interactive workflow remains familiar.
4. As a CLI user, I want to combine --repl and --agentic, so that I can use Agentic RAG for each query in an interactive session.
5. As a CLI user, I want --repl followed by a question to answer that question before entering the REPL, so that the current startup behavior is preserved for either workflow.
6. As a CLI user, I want a REPL session to keep the workflow selected at startup, so that its behavior is predictable for the whole session.
7. As a notes owner, I want the first Agentic search to use my original question, so that the system establishes baseline evidence without spending an extra planning call.
8. As a notes owner, I want the controller to inspect retrieved chunks and decide whether they are sufficient, so that follow-up retrieval is based on evidence rather than a fixed query sequence.
9. As a notes owner, I want the controller to propose a refined search query when evidence is incomplete, so that related notes can be found through bounded iterative retrieval.
10. As a notes owner, I want the controller to select the chunks used for the final answer, so that unrelated retrieved material is excluded from answer context.
11. As a notes owner, I want citations to identify the chunks used for the final answer, so that I can verify claims against their source notes.
12. As a notes owner, I want Agentic RAG to abstain when evidence is insufficient, so that it does not present unsupported information as if my notes contained it.
13. As a notes owner, I want both workflows to abstain deterministically when retrieval returns no chunks, so that an empty context never triggers answer generation.
14. As a notes owner, I want each query to perform at most three searches including the initial search, so that retrieval cost and latency remain bounded.
15. As a maintainer, I want action JSON to be validated by Pydantic and selected evidence references to be checked against retrieved chunks, so that malformed actions and invented chunk references cannot silently control the workflow.
16. As a maintainer, I want one retry for malformed or invalid actions and a clear workflow error after a second failure, so that control-flow failures are observable and never masquerade as abstentions.
17. As a user of any currently supported LLM provider, I want Agentic RAG to use the configured provider through the existing text-generation interface, so that enabling the workflow does not require provider-specific native tool calling.
18. As a notes owner, I want each query to start without prior query state, so that Agentic retrieval does not imply conversation memory.
19. As a notes owner, I want Agentic RAG to search only indexed local notes through a read-only interface, so that the workflow cannot modify notes or introduce outside sources.
20. As a maintainer, I want the evaluation corpus stored separately from data/, so that tests never depend on, expose, or overwrite a user's private notes.
21. As a maintainer, I want a fixed synthetic corpus and 16 reviewed QA cases covering direct lookup, multi-hop integration, unanswerable, and ambiguous questions, so that each change can be compared against the same regression seed.
22. As an evaluator, I want each QA case to record a reference answer, expected answerability, expected source paths, and optional evidence snippets, so that answer and retrieval quality can be checked without relying on unstable chunk IDs.
23. As an evaluator, I want each QA case to run through both Classic and Agentic RAG against the same indexed fixture corpus, so that differences reflect workflow behavior rather than different test data.
24. As an evaluator, I want per-case and aggregate metrics for faithfulness, answer relevance, context precision, and context recall, so that I can inspect both individual failures and overall trends.
25. As a maintainer, I want the legacy base.json system behavior smoke data to remain independently available and excluded from the paired notes RAG score, so that unrelated system checks do not distort notes-quality comparisons.
26. As an evaluator, I want each --eval run to create a UTC timestamped directory directly under the repository-root eval/ directory, so that each benchmark run is preserved for inspection.
27. As an evaluator, I want run.log, per-case results, and a summary in each artifact directory, so that I can diagnose individual answers and compare aggregate outcomes.
28. As an evaluator, I want each generation call to record provider-reported input, output, and total tokens when available, so that I can quantify Agentic controller overhead.
29. As an evaluator, I want token usage reported separately for each query, workflow, and Ragas judge, so that judge cost is not confused with answer-generation cost.
30. As an evaluator, I want unavailable provider token usage to be explicit and character counts kept in separate fields, so that characters are never mislabeled as tokens.
31. As a maintainer, I want evaluation fixtures and evaluator code to remain development-checkout assets, so that the runtime wheel does not gain benchmark data or a new runtime dependency.
32. As an evaluator, I want each run to record non-secret model and retrieval configuration, so that results can be interpreted and reproduced without writing credentials to artifacts.

## Implementation Decisions

### Workflow and CLI boundary

- Classic RAG remains the default workflow. Add a boolean --agentic workflow selector and keep --repl as an independent interaction selector.
- Supported combinations are Classic one-shot, Agentic one-shot, Classic REPL, and Agentic REPL. No query with no REPL continues to display help. A query supplied with --repl is answered first, then the selected REPL starts.
- The selected workflow is fixed for one REPL session. Individual queries do not share agent state.
- Agent orchestration is a small Python loop. Do not add LangChain, LangGraph, or MCP to the first implementation.

### Agent action and evidence contract

- Use strict JSON actions validated as a discriminated Pydantic model:
  - search requires a non-empty query.
  - answer requires one or more evidence references, each identified by source and chunk_id.
  - abstain ends the query without final answer generation.
- Controller input includes the original user query, accumulated unique evidence, search count, and remaining search budget. The model reasons about evidence sufficiency; JSON is the machine-readable control protocol for that decision.
- Pydantic checks action shape and field types. Application logic checks that every selected reference belongs to the current query's retrieved evidence. Invalid references count as invalid actions.
- Deduplicate chunks using the composite runtime identity (source, chunk_id), because chunk_id is only unique within a source. Do not use chunk_id as a persistent evaluation ground-truth key.
- Search count includes the first retrieval and is capped at three. After the third retrieval, the controller can only answer or abstain. An out-of-budget search action is invalid.
- Retry an invalid controller response once. If the retry is invalid, surface a clear workflow error; do not run Classic RAG as a fallback.
- An answer action sends only its selected evidence to the existing final answer generation path and reuses its citations and streaming behavior. The final answer is a separate LLM call.
- Use one consistent abstention response for deterministic empty retrieval and model-selected abstention. Empty cumulative evidence ends without an LLM call.

### Retrieval and answer grounding

- Expose existing indexed-note retrieval through a small read-only Note Search Tool interface. It returns candidate chunks with source, chunk ID, text, and relevance score; the initial implementation calls Python directly.
- Apply a deterministic no-hit guard to Classic and Agentic RAG before answer generation.
- Keep answer generation constrained to supplied note context and preserve the existing citations output contract.
- Treat evidence grounding as a product goal measured by tests and evaluation, not as a formal guarantee that an LLM can never make an unsupported claim.

### Model usage reporting

- Normalize provider usage metadata for all existing adapters without breaking existing answer text and streaming callers.
- Record input, output, and total tokens for each controller and final-answer call when a provider reports them. A missing value is represented as unavailable, not estimated from characters.
- Record input and output character counts in separate fields as a diagnostic fallback. Never convert or label character counts as tokens.
- Keep Ragas judge usage separate from retrieval and answer workflow usage. Preserve unavailable values when the judge provider does not report usage.

### Evaluation and artifacts

- Keep the synthetic note corpus and QA benchmark under the test fixtures, outside the user's notes directory and outside the runtime package.
- Use 16 human-reviewed QA cases: four direct lookup, four multi-hop integration, four unanswerable, and four ambiguous. Each row records expected answerability, reference answer, source paths, and optional evidence snippets.
- Evaluate both workflows over the same fixture corpus and question set. Include per-case workflow outputs and separate aggregate results. Do not combine the legacy base.json system smoke data with notes RAG scores.
- Report the existing Ragas dimensions: faithfulness, answer relevancy, context precision, and context recall. Also report expected-evidence/source coverage and answerability behavior per case so that abstention and retrieval failures are visible.
- Run Ragas answer-quality and context metrics only for QA cases expected to be answerable. For unanswerable and ambiguous cases, leave those metrics not applicable and evaluate them through expected-answerability, abstention, and evidence-coverage fields instead of treating the abstention text as a factual ground truth.
- Store each run under a unique UTC timestamp directory directly below the repository-root evaluation output directory. Include a run log, per-case results, and a summary. Keep generated run directories out of Git while keeping fixture data and evaluation code tracked.
- Record safe run metadata such as provider/model identifiers, timestamp, retrieval settings, and fixture version or digest. Never write API keys or other credentials to evaluation artifacts.
- Treat the 16-case set as an initial regression seed, not a statistically representative benchmark. The target is improved multi-hop performance without regressions in faithfulness and correct abstention.

## Testing Decisions

### Test seam and prior art

Use one primary behavioral seam at the query application boundary, reached through CLI dispatch. Run one-shot and REPL interactions with a temporary fixture index and replaceable model provider; assert user-visible answers, citations, abstentions, and workflow errors without inspecting private loop steps. Exercise --eval through its public runner behavior using the same isolated fixture setup and assert the generated paired results and artifacts.

The modules under test are CLI workflow dispatch, query orchestration, retrieval/answer integration, and the evaluation runner. Fixture-backed evaluation should use the real ingestion, chunking, indexing, and retrieval path. Existing tests for ingestion, indexing, retrieval, and answer generation provide prior art, and the pytest setup already isolates NOTES_DIR and STORAGE_DIR with tmp_path. Reuse that isolation; never point tests at the user's data/ or persistent index.

### Required behavioral coverage

- Classic remains default; --agentic and --repl combine independently for one-shot and REPL usage.
- A query passed with --repl is answered before the REPL begins, using the selected workflow.
- Initial search uses the original question; follow-up searches use controller-proposed queries.
- The controller receives accumulated deduplicated evidence and its remaining search budget.
- Search calls never exceed three, including the initial retrieval; the exhausted-budget decision can only answer or abstain.
- Answer actions with unknown or empty evidence references are rejected; valid selections alone reach final answer generation and citations.
- Empty retrieval in either workflow returns the deterministic abstention without an LLM call.
- Model-selected abstention does not call final answer generation.
- Invalid action JSON/schema gets exactly one retry; a second invalid action surfaces a workflow error and never falls back to Classic.
- Independent queries and REPL prompts do not carry forward agent state.
- Evaluation fixtures are isolated from data/ and produce paired per-case results, aggregate metrics, token usage, and run artifacts.
- Provider usage is reported as unavailable when absent; character counts remain a separate metric.

Deterministic tests should use a fake model provider and fixture notes and must not require live LLM credentials. The developer --eval workflow may use the configured answer and judge providers; its generated artifacts make those live results auditable. Do not treat live-provider scores as stable unit-test assertions.

## Out of Scope

- MCP server/client transport or exposing note search to external MCP hosts.
- LangChain, LangGraph, or another orchestration framework in the first implementation.
- Conversation memory shared across separate questions.
- A separate reranker, relevance-judge model, or embedding-model replacement.
- External web search, filesystem mutation tools, or any non-notes evidence source.
- Shipping the benchmark fixtures in the runtime wheel.
- Claiming a formal zero-hallucination guarantee.
- Large-scale benchmark claims from the initial 16 examples.

## Further Notes

- The JSON action is the model's decision expressed in a form the Python loop can safely route. The model examines the chunks to decide sufficiency; Pydantic validates that the requested next step is well-formed. These are separate responsibilities.
- A valid action and valid chunk references do not prove semantic relevance. Faithfulness metrics and human review remain necessary; Ragas scores are diagnostic signals, not proof of correctness.
- In the normal successful path, Agentic RAG can make up to three controller calls and one final answer call. One retry per invalid action can increase controller calls, while the three-search limit remains unchanged.
- Ragas judge calls add evaluation cost. Their token usage must be shown separately from both retrieval workflows.
- The benchmark is intended for repository development and regression review. Changes to chunking may change chunk IDs, so expected evidence is anchored to source paths and optional text snippets.
