# Aurora Planner release test plan

Test plan version 2.3 was approved by Priya Shah on 2026-05-14. It uses the separately approved $4,000 QA and validation allocation from budget version 2.1.
The calendar-export suite contains five recurring-event scenarios: duplicate events, missing events, time-zone changes, recurring edits, and cancellation.
Two QA testers are assigned for five full days each. The plan therefore provides ten tester-days; it does not promise ten calendar days or five days per scenario.
The test budget ceiling is $4,000, matching the QA allocation in the approved budget. The plan does not authorize spending the $2,000 contingency reserve.
The 2026-05-20 run summary marks all five scenarios executed. Four passed on release candidate RC-17. The cancellation case needs a retest because the first result used a build that was replaced after code freeze.
The cancellation retest is assigned to Priya Shah and depends on the approved hotfix build. No passing retest is recorded in this plan as of 2026-05-20.
Evidence for each scenario must include the build identifier, input calendar, time-zone settings when relevant, expected event count, observed event count, and tester initials.
A test being executed does not establish a pass. Only a completed result with the required evidence and QA sign-off can satisfy the launch checklist.
Test execution and launch approval are distinct: passing the tests does not close the open marketing review, and completing marketing review does not replace the cancellation retest.
