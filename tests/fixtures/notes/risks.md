# Aurora Planner risks and owners

Risk register version 2.0 was reviewed on 2026-05-20. Owners below are accountable for the listed follow-up, not necessarily the product or release as a whole.
Primary launch risk: calendar export can duplicate recurring events when an event is edited across daylight-saving boundaries.
QA lead Priya Shah owns verification of the five recurring-event cases before launch. She must attach the cancellation retest result to the checklist; a run without the approved build identifier is not final evidence.
Secondary risk: pilot support requests may wait too long when the shared inbox receives reports outside Northstar Studio’s business hours.
Launch coordinator Marco Bell checks support@northstar.example each business day and routes product defects to Priya. That operational task does not make him the product owner.
Product owner Lena Ortiz decides whether a requested behavior is inside the first-release scope. She does not sign off QA evidence or approve legal review.
The risk register uses “owner” to mean the person accountable for a risk action. Meeting minutes sometimes use the word for the person presenting a topic, which is not an ownership change.
A third, lower-severity risk is that time-zone examples in pilot feedback use local abbreviations without a UTC offset. Priya’s test records must include the zone name and offset when documenting a mismatch.
The duplicate-event risk is not closed by the code freeze. It remains open until the cancellation retest is attached and the QA lead records the release-candidate identifier.
