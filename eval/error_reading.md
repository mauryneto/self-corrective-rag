# Reading V4's errors, with the 14B and with the 7B

V4 gets 78 of the 120 answerable questions right with `qwen2.5:14b` (run `v4_repro`) and 58
with `qwen2.5:7b` (run `v4_7b`), both on the same retrieved passages. This is a reading of the
other 42 and 62: why each one is not correct, and how far from correct it is.

**What this is and is not.** One reader, one pass, from the judge's written reasoning for each
answer, not blind to the verdicts. Correct / partial / incorrect and the declines are the
judge's. The cause of each error and the split of the "incorrect" answers in two are the
reader's. It changes no number in the results table.

## Why the answer is not correct

| Cause | 14B | 7B |
|---|---|---|
| A gold article is not among the six passages | 22 | 27 |
| Leaves out a condition, an exception or a part that is in the prompt | 10 | 14 |
| Misreads or misapplies a passage that is in the prompt | 9 | 11 |
| Adds a condition or a qualifier the norm does not have | 1 | 9 |
| Declines with the answer in the prompt | 0 | 1 |
| **Total** | **42** | **62** |

- **The first row is counted, not read.** 28 of the 120 questions do not have every gold
  article in the prompt. The 14B still gets 6 of them right, the 7B 1. Those answers were not
  read for a second cause.
- **With every gold article in the prompt (92 questions) the 14B gets 72 right and the 7B 57.**
  The rows below the first are those 20 and 35 answers.
- **Adding to the norm is the 7B's own error: 9 answers against 1.** Eight of the nine are on
  questions the 14B answers correctly. Three add "úteis" (working days) to a deadline the norm
  counts in days (q048, q078, q083); the others add a condition or an exception (q017, q019,
  q042, q050, q087) or a calculation base (q041).
- **Omissions and misreadings are close in the two models** (19 against 25), and often on the
  same questions: 37 questions are not correct with either model. 25 are wrong only with the
  7B, 20 of them with every gold article in the prompt. 5 are wrong only with the 14B.

## How far from correct

| Outcome (of 120 answerable) | 14B | 7B |
|---|---|---|
| Correct | 78 | 58 |
| Partial: what it says is right, something that changes the practical answer is missing | 22 | 23 |
| Incorrect, the core is right: answers what was asked, or one of the two parts asked, and states something wrong with it | 11 | 23 |
| Incorrect, the main conclusion is wrong: the yes or no, the deadline or the amount asked for | 5 | 13 |
| Declined a question the norms answer | 4 | 3 |

The first, second and last rows are the judge's verdicts. The two "incorrect" rows are the
judge's 16 and 36 incorrect answers, split by the reader.

- **Most errors are near misses in both models.** The main conclusion is wrong in 5 of the 120
  answers of the 14B and in 13 of the 7B.
- **The smaller model does not produce more partial answers, it produces more wrong ones.**
  Partial answers are the same (22 and 23); incorrect ones more than double (16 → 36).

## Questions it should decline

Of the 60 questions the norms do not answer, the 14B answers 4 (q010, out of scope; q140,
q153, q154, unanswerable) and the 7B the same four and q143. Each of these answers states a
rule the corpus does not have, which is the worst error this system can make. The 7B also
declines q150 after stating a rule the cited passage does not contain; the judge grades it
incorrect, so 54 of its 55 declines are correct.

## A score with half points

One point for a correct verdict (a correct answer, or a correct decline), over the 180
questions, as a score of 100:

| Scoring | 14B | 7B | Distance |
|---|---|---|---|
| Correct only | 74.4 | 62.2 | 12.2 |
| Half a point for a partial answer | 80.6 | 68.6 | 12.0 |
| Half a point also for an incorrect answer with the core right | 83.6 | 75.0 | 8.6 |

- **The results table keeps "correct only".** Half a point for a partial answer is a fair
  second reading: nothing in a partial answer is wrong. It moves both models up six points and
  leaves the distance between them where it was.
- **The third row is generous, and it is the reader's split, not the judge's.** An answer with
  "ten working days" where the norm says "ten days" has the core right and would make the
  person miss a deadline. In this domain that is why the judge calls it incorrect.
- **The split between partial and incorrect is the least reliable part of the judge.** It
  agrees with human labels at kappa 0.63 over the three classes (README, "Judge validation"),
  so a half-point score carries more judge noise than the count of correct answers.

## The questions

**14B (`v4_repro`), 42 not correct**

- Gold article missing (22): q013 q036 q039 q045 q046 q085 q086 q091 q093 q094 q096 q098 q099
  q103 q106 q112 q113 q117 q118 q120 q121 q122
- Leaves out (10): q002 q005 q012 q016 q031 q051 q100 q104 q107 q114
- Misreads (9): q003 q008 q022 q048 q055 q058 q095 q101 q115
- Adds (1): q054
- Incorrect, main conclusion wrong (5): q013 q022 q055 q086 q094
- Incorrect, core right (11): q003 q008 q048 q054 q058 q085 q093 q101 q103 q112 q118
- Declined (4): q036 q039 q045 q106

**7B (`v4_7b`), 62 not correct**

- Gold article missing (27): q013 q027 q036 q039 q045 q046 q085 q086 q089 q091 q092 q093 q094
  q096 q098 q099 q103 q105 q106 q112 q113 q117 q118 q119 q120 q121 q122
- Leaves out (14): q002 q003 q005 q012 q035 q043 q051 q057 q075 q100 q102 q108 q109 q114
- Misreads (11): q008 q015 q022 q055 q058 q069 q095 q097 q101 q104 q110
- Adds (9): q017 q019 q041 q042 q048 q050 q078 q083 q087
- Declines with the answer in the prompt (1): q067
- Incorrect, main conclusion wrong (13): q008 q013 q022 q036 q045 q055 q058 q069 q086 q092
  q094 q097 q122
- Incorrect, core right (23): q015 q017 q019 q042 q048 q050 q078 q083 q087 q091 q095 q096 q099
  q101 q104 q105 q109 q110 q113 q114 q118 q120 q121
- Declined (3): q039 q067 q098

Partial answers are the non-correct ones in neither "incorrect" list nor "declined".
