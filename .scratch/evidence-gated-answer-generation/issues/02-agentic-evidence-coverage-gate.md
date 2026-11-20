# 02: Agentic Evidence coverage gate

**What to build:** Agentic RAG 僅在 controller 評估實際 Answer generation context 至少有部分 evidence 時生成答案；搜尋與審查失敗保留清楚、可稽核的結果。

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] 初次 Search 有 chunks 時，controller 針對原始問題評估完整 accumulated chunks pool；`zero_evidence` 或 `partial_evidence` 且 Search budget 尚未用盡時，依有效計畫進行 bounded Adaptive search，保留去重、provenance、Search audit 與獨立的 coverage／stop reason。
- [x] Search 結束後 pool 仍為 `zero_evidence` 時，回傳固定的無相關資訊 outcome，不呼叫 Answer；保留 retrieved evidence 和 review trace，Answer generation context 為空。
- [x] 保留 final reranking 和 Generation top-k；若選取可能移除 evidence，controller 在 Answer 前評估實際選取的有序 context；context 與已評估 pool 相同時可重用評估。選取後的 `zero_evidence` 不呼叫 Answer，`partial_evidence` 或 `sufficient_evidence` 才使用相同 context 生成；最終評估不重啟 Search。
- [x] Pool 和 selected-context coverage 分別可稽核；controller provider 失敗或輸出經 bounded validation retry 仍無效時，回報獨立的 workflow error，保留錯誤類別與 retrieved evidence，不宣稱 `zero_evidence`，也不呼叫 Answer。
- [x] 保留獨立的 Answer attempt budget、同一 context 的 revision、受支持且揭露缺口的 partial answer，以及未通過 review 的候選答案不對使用者顯示的行為。
- [x] Evaluation 區分 controller 決定的 evidence 不足、retrieval-empty 與 workflow error，正確處理 reliability、Ragas applicability、usage 和 evidence audit；保留歷史結果相容性及 evaluation isolation。
- [x] 更新受影響的維護文件及 ADR-0008、ADR-0009 的衝突決策，保留其他適用約束。以既有 query workflow 和 evaluation runner seam 測試成功搜尋、budget 用盡、context selection 改變 coverage、validation 和 provider failure、partial answer 與 revision。
- [x] 執行相關 `uv run pytest` 測試；保留既有未提交變更。

## Comments

Implemented on the isolated integration branch. Full offline suite passed after integration and review fixes; live-provider acceptance was not run.
