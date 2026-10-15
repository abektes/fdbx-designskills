# Eval: Answer-first, one sample per scenario

Run after commit "Open every output with a direct answer to the question asked"
(2026-10-12), which added an Answer section to every skill. Only the with-skill
arm was regenerated (18 outputs, DeepSeek V4 Pro via OpenRouter, $0.37);
the baseline arm, told the published method by name, came from cache. Pairs were
judged blind by Claude Sonnet subagents (in session, no API cost) with the
seven-dimension rubric and the quote check against label swaps
(`scripts/judge_pairs.py`, ported from the family harness).

## Result

- Skill preferred in **17/18** pairs, ties 0, label swaps 0.
- Mean score gap per dimension (skill minus baseline, 1-5 scale): structure_format +0.78 · depth_specificity +0.56 · actionability +0.78 · methodology_fidelity +2.17 · coverage_completeness +1.33 · ethical_nuance +0.89 · evidence_honesty +1.83.
- Structural conformance (code checks, semantic checks not run): skill 262/277 (95%), baseline 21/273 (8%).
- New checks on the skill arm: answer_first 18/18, Evidence Ledger present 15/15.

## Notes

- **This is fdbx's first independent blind A/B.** Until now the box had only the owner's partially blind review (12/12).
- The one loss is Horizon Scanning (food-delivery-scan, delta -1.0). The run had no web search, so the skill honestly left the signal list mostly as placeholders, and the judge preferred the baseline's usable but overstated scan. With live search (baseline-probe-batch-2) the same skill cited 54 real links. Horizon Scanning should be judged with search on; this run measures it without.
- Evidence Ledger presence is checked only where the rubric has the check (15 of 18 outputs).
- One sample per scenario: read the box-level rate, not per-skill counts.

Data: `judging/` (Sonnet verdicts and tally).
