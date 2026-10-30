# 02: Remove candidate benchmark authoring

**What to build:** Remove the unreviewed Ragas candidate-authoring workflow so the fixed evaluation corpus and reviewed QA set are the only benchmark-maintenance inputs. Users no longer have a generator command, draft artifact lifecycle, or documentation implying automatic candidate promotion.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] The candidate-generator implementation, module CLI, draft-output behavior, and related tests are removed.
- [ ] No supported command or repository guidance describes a candidate-draft directory, automatic promotion, or live-provider candidate generation.
- [ ] The remaining evaluation workflow has no dependency on candidate-generation configuration, paths, or ignore rules.
- [ ] The standard test suite confirms the fixed reviewed benchmark remains runnable without the removed workflow.

