# 01: 初次 Retrieval 為空時直接回覆

**What to build:** Classic RAG 與 Agentic RAG 第一次 Retrieval search 沒有 chunks 時，直接以固定訊息說明搜尋沒有找到相關資訊並建議調整問題，讓使用者不必等待不必要的 model 呼叫。

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Classic 和 Agentic 的第一次 Search 為空時回傳固定的 retrieval-empty outcome；訊息不聲稱 corpus 沒有答案，也不判定問題無法理解。
- [x] 兩種 workflow 均不呼叫 Answer model；Agentic 不呼叫 controller，也不執行後續 Search。
- [x] 保留 Search audit、空的 Answer generation context，以及 evaluation 中與 workflow error 和生成答案可區分的 retrieval-empty outcome；不將此 outcome 納入生成答案的 Ragas averages。
- [x] 在既有 query workflow 和 evaluation runner seam 加入行為測試，包含 provider 未被呼叫的驗證，並更新受影響的說明與架構決策。
- [x] 執行相關 `uv run pytest` 測試；保留既有未提交變更與 evaluation isolation。

## Comments

Implemented on the isolated integration branch. Full offline suite passed after integration and review fixes; live-provider acceptance was not run.
