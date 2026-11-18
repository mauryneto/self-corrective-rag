# Reading the 32 ceiling failures

The ceiling run (`ceiling_repro`: gold passages first, V1's other passages around them) gets 88
of the 120 answerable questions right. This is a reading of the other 32, one by one: the
question, the reference, the gold passages, the answer, the judge's reasoning, and the answer
the same model gave with only the gold passages (`ceiling_gold_repro`).

**What this is and is not.** One reader, one pass, not blind to the verdicts. It changes no
number in the results table. The strictness described below applies to every row of that table
alike, so the comparisons between versions stand.

## Result

| Cause | Direct | Multi-hop | Total |
|---|---|---|---|
| Wrong reading: the passage is there and the answer contradicts it or adds a rule | 3 | 5 | 8 |
| Distracted: answers from another passage in the prompt | 3 | 3 | 6 |
| Leaves out a condition or exception that is in the gold passage | 2 | 6 | 8 |
| Strict reference or judge: the practical answer is there | 4 | 6 | 10 |
| **Total** | **12** | **20** | **32** |

- **About 22 of the 32 are the model's, not 32.** In 10 the answer gives what the question
  asked. Seven of those lose the point for something the reference adds beyond the question (a
  second exception, a substitute, a deadline that was not asked). The other three got "correct"
  in the gold-only run for an answer that says the same thing: judge inconsistency, expected
  from a judge that agrees with human labels at kappa 0.63.
- **Completeness is the largest single pattern: 15 questions** (the 8 omissions and the 7
  strict references). In all of them the missing piece is a condition, an exception or a second
  item that sits in a passage the model was given. The answering prompt asks for a direct answer
  in up to three short paragraphs and says nothing about exceptions.
- **Distraction is real but narrow: 6 questions**, 5 of them correct with only the gold
  passages. The model picks the neighbouring article (the rule for dismissed employees instead
  of retirees, the deadline for care requests instead of non-care requests).
- **Wrong reading is 8 questions**, and it is the only group where a stronger or fine-tuned
  model is the obvious fix. The model quotes both articles correctly and then combines them
  into a conclusion neither supports, or invents a detail (a year, a 60-day notice).

## What it changes

- **Fine-tuning is not justified yet.** The group it would address is 8 questions, too few to
  measure a gain on (the V0 → V1 difference of 9 questions has p = 0.12).
- **The cheap experiment is a completeness instruction in the answering prompt**: state the
  conditions, exceptions and deadlines that the cited passage attaches to the rule. It targets
  up to 15 questions. It can also cost: longer answers leave more room for the wrong-reading
  kind of error.
- **Caveat on that experiment:** the instruction comes from reading failures on the same 180
  questions it would be measured on. There is no held-out set. It must stay a general
  instruction, never mention a norm or a question, and the README must say how it was arrived
  at.
- **Fewer passages is not a free fix for distraction.** The gold-only run recovers 5 of these 6
  and still ends at the same 88, because 11 other questions go the opposite way.

## Follow-up: the completeness instruction was tried

Same day, three rows re-run with one general rule added to the answering prompt (README,
"Experiment: a completeness rule in the answering prompt"). Correct of 120: ceiling 88 -> 94,
gold-only ceiling 88 -> 96, V1 78 -> 79.

| Cause | Correct with the rule (ceiling) |
|---|---|
| Wrong reading | 4 of 8 (q008, q032, q054, q103) |
| Distracted | 1 of 6 (q115) |
| Leaves out a condition or exception | 4 of 8 (q031, q051, q105, q112) |
| Strict reference or judge | 6 of 10 (q048, q093, q099, q108, q113, q121) |

Fifteen of the 32 become correct and 9 questions that were correct are lost. In V1 the rule
changes nothing: the gain sits in questions whose gold passages V1 does not retrieve.

## The 32, one line each

Verdicts: ceiling / gold-only. C correct, P partial, I incorrect.

### Wrong reading (8)

| Id | Type | Verdicts | What happened |
|---|---|---|---|
| q008 | multi-hop | I / I | Quotes the 3-day and the 5-day deadlines, ignores the paragraph that says the shorter one governs the reply |
| q022 | direct | I / I | Places June between January and March; with gold only, gets the rule right and invents the year 2022 |
| q032 | direct | I / P | Final "no" is right, but says the exemption covers professional associations (it covers councils and unions) |
| q054 | direct | I / P | Third hypothesis turns the operator's duty to offer individual plans into a condition for losing the right |
| q101 | multi-hop | I / P | Both discounts right (40%, 20%); adds that the 20% holds while a second-instance decision is pending |
| q103 | multi-hop | I / I | States both delivery rules correctly, then concludes the two documents are delivered at the same moment |
| q104 | multi-hop | I / I | Drops the general rule (anniversary, 60 days, reasons) and invents a 60-day notice for non-payment |
| q109 | multi-hop | I / I | "30 days from birth" becomes "until the end of this week"; with gold only, says the newborn must serve the minimum stay |

### Distracted by another passage (6)

| Id | Type | Verdicts | What happened |
|---|---|---|---|
| q004 | direct | I / C | Answers 10 working days from the article on care requests; the gold article (7 days) was passage 1 |
| q013 | direct | P / C | Answers about the operator cancelling for non-payment; the question is the user's own cancellation |
| q055 | direct | I / C | Applies the dismissed-employee rule (a third of the time, 6 to 24 months) to a retiree |
| q085 | multi-hop | I / C | Counts the 60 days from the end of the agreement, not from the user learning of it |
| q091 | multi-hop | I / C | Reasons from the article on transfers inside the network, misses the contract exception |
| q115 | multi-hop | P / P | Says deadlines are suspended while the board is formed, from an article that allows one 3-day suspension |

### Leaves out a condition or exception (8)

| Id | Type | Verdicts | Missing |
|---|---|---|---|
| q031 | direct | P / P | The proposal must be filed within 30 days of the contract anniversary |
| q051 | direct | P / C | The operator must tell beneficiaries about the suspension |
| q098 | multi-hop | P / P | Age-band change, migration and adaptation are outside the 12-month rule |
| q100 | multi-hop | P / P | 24 hours for urgent care and 300 days for childbirth, next to the 180 days it gives |
| q105 | multi-hop | P / I | 180 days from the hospital's removal to ask for portability |
| q112 | multi-hop | P / C | What is charged if the plan was used within the 7 days |
| q114 | multi-hop | P / C | The requirements themselves: ANS authorisation and notice to beneficiaries |
| q120 | multi-hop | P / P | The single 15-day extension of the 30-day deadline |

### Strict reference or judge (10)

| Id | Type | Verdicts | Why the answer is acceptable |
|---|---|---|---|
| q002 | direct | P / P | Gives the 3 working days asked; the reference adds the high-complexity exception |
| q003 | direct | P / P | Gives the whole sequence, transport included; the reference adds who pays the provider |
| q016 | direct | P / C | Same content in both runs: rule, exceptions named. Judge inconsistency |
| q048 | direct | I / C | Same content in both runs, same imprecision about when the 10 days start. Judge inconsistency |
| q093 | multi-hop | I / C | Same content in both runs (yes, and the class changes the fine), no factor values in either. Judge inconsistency |
| q099 | multi-hop | P / P | Covers the clinic's report and the operator's duty; the reference adds the notice to the patient |
| q108 | multi-hop | P / P | Yes to transport and companion costs for a 72-year-old; the reference adds the medical contraindication |
| q113 | multi-hop | P / P | IPCA and the 115/110/105% factors; the reference adds the 100% tier and the anniversary date |
| q121 | multi-hop | P / P | Yes, a TISS Coordinator, and the sanction; the reference adds the substitute |
| q122 | multi-hop | P / P | The infraction and its sanction; the reference adds the denial-of-coverage exception |

## Found on the way

- **q032's gold source is not enough on its own.** The gold article refers to "items I and II
  of art. 15" without their text, so the gold-only run cannot answer. Art. 15 should be a
  second gold source.
- **The corpus holds some text twice.** An amending norm (RN 649) carries the new wording of
  another norm's article (RN 137, art. 20), and the consolidated RN 137 has it too. Both get
  retrieved and either citation is valid; retrieval metrics count only one as gold.
