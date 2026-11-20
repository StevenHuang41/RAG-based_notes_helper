# 03: 移除固定 Answer abstention 文字

**What to build:** Answer model 根據 Context 自然說明缺少的資訊，Agentic 以 Answer review 的結構化決策辨識 abstention，Classic evaluation 對無法可靠分類的自由回答標記未知。

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] 移除 `ABSTENTION_MESSAGE`、相關 import 和依賴回答文字完全相等的分類；Answer prompt 保持 evidence grounding，並要求在資訊不足時自然說明，而非指定固定句子。
- [x] Classic 對非空 Retrieval 結果直接呼叫 Answer，保留 streaming 和空白或失敗回應的既有安全處理；不新增 controller、分類器或 retry budget。
- [x] Classic 自由生成回答的 answerability 和 abstention 在無可靠分類依據時為未知，不猜測；未知分類不阻止適用的生成品質指標評估。
- [x] Agentic 使用 Answer review 決策辨識自然措辭的 abstention，保留 partial answer 缺口揭露、同 context revision 和已接受答案的各指標適用性。
- [x] Evaluation 與報告使用結構化 outcome 區分模型回答和 deterministic outcome，不依賴固定文字；保留歷史結果相容性及 evaluation isolation。
- [x] 透過既有 query workflow 和 evaluation runner seam 測試不同 abstention 措辭、Classic 未知分類、Agentic review 決策和 streaming；更新受影響的文件。
- [x] 執行相關 `uv run pytest` 測試；在三張 ticket 整合後執行完整 suite，並區分 fake-provider 驗證與 live-provider acceptance。

## Comments

Implemented on the isolated integration branch. Full offline suite passed after integration and review fixes; live-provider acceptance was not run.
