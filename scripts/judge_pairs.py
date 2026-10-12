#!/usr/bin/env python3
"""Blind pairwise judging of a run_generation.py run, the RESULTS.md way.

For every (skill, scenario, rep) the run produced in both arms, each judge sees
the task, the published method's name, and the two responses as "Response A" and
"Response B", in an order randomized per judge and per pair with a fixed seed. It
scores both on edbx's six dimensions (1-5) under the length-blind instruction and
picks a winner. Delta = (skill total - baseline total) / number of dimensions.

v0.3 adds a seventh dimension, evidence_honesty: in v0.2 one judge preferred a
baseline for "realistic, illustrative data" it had invented, and the six-
dimension rubric had no way to count that against it.

Judging by Claude subagents instead of paid API calls (the default since
v0.3): `--emit DIR` writes one blind prompt per judgment plus jobs.json; a
subagent reads each prompt and writes its JSON verdict next to it;
`--ingest DIR` checks and tallies them exactly as API verdicts are.

One addition, from cdbx v0.1: a judge there described the skill output's content
under the wrong label, so its verdict pointed at the baseline. Each judge now
quotes 6-15 words from the response it prefers, and this script checks which
response the quote actually comes from. A quote found only in the other response
is flagged `label_swap` and reported, never silently corrected.

    python3 scripts/judge_pairs.py --run 20260930T120000Z --dry-run
    python3 scripts/judge_pairs.py --run 20260930T120000Z
    python3 scripts/judge_pairs.py --run 20260930T120000Z --tally docs/eval-v0.2

Judgments are cached in eval-framework/judgments/ by (judge, pair, order), so a
rerun costs nothing for pairs already judged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_generation as rg  # noqa: E402
from run_generation import CACHE_DIR, ENDPOINT, EVAL_DIR, REPO, RUNS_DIR, SKILLS_DIR, load_env, method_name  # noqa: E402

# Ported from cdbx for edbx, fdbx and sdbx (2026-10-12). Those harnesses name the
# method from each SKILL.md's H1 (what the baseline arm was told) rather than a
# PUBLISHED_METHOD table, and edbx keeps its scenarios in eval-framework/.
BOX = getattr(rg, "BOX", SKILLS_DIR.name)
JUDGE_ROLE = {"edbx": "ethical design work for digital products",
              "fdbx": "futures and foresight work for design and strategy",
              "sdbx": "systemic design work (systems mapping, soft systems, transitions)"}[BOX]


def evals_file(skill: str) -> Path:
    if hasattr(rg, "evals_path"):
        return rg.evals_path(skill)
    return EVAL_DIR / skill / "evals.json"


def published_method(skill: str) -> str:
    return method_name((SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8"), skill)

JUDGE_DIR = EVAL_DIR / "judgments"
DEFAULT_JUDGES = ["anthropic/claude-sonnet-5.5", "google/gemini-3.1-pro-preview"]
DIMENSIONS = [
    "structure_format", "depth_specificity", "actionability",
    "methodology_fidelity", "coverage_completeness", "ethical_nuance",
    "evidence_honesty",
]

JUDGE_SYSTEM = """You are an expert evaluator of {role}. You will see a design task, the name of the method the work was asked to apply, and two responses, "Response A" and "Response B". You do not know how either was produced.

Score EACH response from 1 (poor) to 5 (excellent) on seven dimensions:
1. structure_format: organization, clarity, scannability.
2. depth_specificity: concrete, specific, checkable content rather than generic advice.
3. actionability: a team could act on it directly.
4. methodology_fidelity: faithful application of the named method as published.
5. coverage_completeness: covers what the task and the method require.
6. ethical_nuance: attention to affected people, risks, and accountability.
7. evidence_honesty: does not invent facts about the user's situation (figures, results, incidents, sources) or present illustrative numbers as real; says "unknown" or marks an assumption where the task gives no data. Invented specifics that look real score low, however useful they seem.

Length is not a quality signal. A concise, complete response can score 5 on Coverage and Structure. Do not reward verbosity, padding, or repeated framings. Penalize bloat that does not add substance.

Then pick the better response overall, or "tie".

Finally, copy a verbatim quote of 6 to 15 consecutive words from the response you picked (any response if you pick "tie"), so the choice can be checked.

Reply with JSON only, in exactly this shape:
{"A": {"structure_format": {"score": n, "reason": "..."}, ... all seven ..., "total": n, "overall_reason": "..."},
 "B": { same shape, all seven },
 "winner": "A" | "B" | "tie",
 "winner_reason": "...",
 "winner_quote": "..."}"""

JUDGE_SYSTEM = JUDGE_SYSTEM.replace("{role}", JUDGE_ROLE)

JUDGE_USER = """## Task given to both responses
{prompt}

## Method the task asked for
{method}

## Response A
{a}

## Response B
{b}"""


def load_run(run_id: str) -> dict:
    return json.loads((RUNS_DIR / f"{run_id}.json").read_text())


def scenario_prompts() -> dict[tuple[str, str], str]:
    prompts = {}
    for skill_dir in SKILLS_DIR.iterdir():
        path = evals_file(skill_dir.name)
        if skill_dir.is_dir() and path.is_file():
            for e in json.loads(path.read_text())["evals"]:
                prompts[(skill_dir.name, e.get("name") or f"eval-{e.get('id')}")] = e["prompt"]
    return prompts


def pairs_of(run: dict) -> list[dict]:
    by = {}
    for t in run["tasks"]:
        by.setdefault((t["skill"], t["scenario"], t["rep"]), {})[t["arm"]] = t["key"]
    out = []
    for (skill, scenario, rep), arms in sorted(by.items()):
        if {"with_skill", "without_skill"} <= arms.keys():
            out.append({"skill": skill, "scenario": scenario, "rep": rep,
                        "skill_key": arms["with_skill"], "raw_key": arms["without_skill"]})
    return out


def output_of(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    return json.loads(path.read_text())["output"] if path.is_file() else None


def skill_is_a(seed: int, judge: str, pair: dict) -> bool:
    h = hashlib.sha256(f"{seed}|{judge}|{pair['skill_key']}|{pair['raw_key']}".encode()).digest()
    return random.Random(h).random() < 0.5


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def call_judge(model: str, system: str, user: str, api_key: str, attempts: int = 5) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 20000,
        "temperature": 0,
    }).encode()
    last = ""
    for attempt in range(attempts):
        req = urllib.request.Request(ENDPOINT, data=payload, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                body = json.loads(resp.read())
            text = body["choices"][0]["message"].get("content") or ""
            verdict = extract_json(text)
            if verdict.get("winner") not in ("A", "B", "tie"):
                raise ValueError(f"bad winner {verdict.get('winner')!r}")
            usage = body.get("usage", {})
            return {"verdict": verdict, "served_model": body.get("model"),
                    "cost_usd": usage.get("cost", 0.0), "attempts": attempt + 1}
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read().decode()[:200]}"
            if exc.code not in (408, 429, 500, 502, 503, 504, 529):
                raise RuntimeError(last) from exc
        except Exception as exc:  # noqa: BLE001 - malformed JSON, dropped connection: retry
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{model}: gave up after {attempts} attempts ({last})")


def judge_one(pair: dict, judge: str, seed: int, prompts: dict, api_key: str) -> dict:
    skill_first = skill_is_a(seed, judge, pair)
    skill_out, raw_out = output_of(pair["skill_key"]), output_of(pair["raw_key"])
    a, b = (skill_out, raw_out) if skill_first else (raw_out, skill_out)
    tag = hashlib.sha256(f"{judge}|{pair['skill_key']}|{pair['raw_key']}|{skill_first}".encode()).hexdigest()[:16]
    cache = JUDGE_DIR / f"{tag}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    user = JUDGE_USER.format(prompt=prompts[(pair["skill"], pair["scenario"])],
                             method=published_method(pair["skill"]), a=a, b=b)
    got = call_judge(judge, JUDGE_SYSTEM, user, api_key)
    record = build_record(pair, judge, skill_first, skill_out, raw_out, got)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, indent=2))
    return record


def build_record(pair: dict, judge: str, skill_first: bool, skill_out: str, raw_out: str, got: dict) -> dict:
    a, b = (skill_out, raw_out) if skill_first else (raw_out, skill_out)
    v = got["verdict"]
    skill_label, raw_label = ("A", "B") if skill_first else ("B", "A")
    quote = norm(v.get("winner_quote", ""))
    in_skill, in_raw = bool(quote) and quote in norm(skill_out), bool(quote) and quote in norm(raw_out)
    picked = {"A": a, "B": b}.get(v["winner"])
    label_swap = bool(quote) and picked is not None and quote not in norm(picked) and (in_skill or in_raw)
    total = lambda side: sum(int(v[side][d]["score"]) for d in DIMENSIONS)  # noqa: E731
    record = {
        **pair, "judge": judge, "served_model": got["served_model"], "skill_is": skill_label,
        "winner_arm": {skill_label: "skill", raw_label: "raw"}.get(v["winner"], "tie"),
        "skill_total": total(skill_label), "raw_total": total(raw_label),
        "delta": round((total(skill_label) - total(raw_label)) / len(DIMENSIONS), 3),
        "dims": {d: {"skill": int(v[skill_label][d]["score"]), "raw": int(v[raw_label][d]["score"])} for d in DIMENSIONS},
        "quote_found_in": "skill" if in_skill and not in_raw else "raw" if in_raw and not in_skill else "both" if in_skill else "neither",
        "label_swap": label_swap, "verdict": v, "cost_usd": got.get("cost_usd", 0.0),
    }
    return record


def emit(pairs: list[dict], out_dir: Path, seed: int, prompts: dict, judge: str) -> None:
    """Blind prompt files for subagent judges; nothing in them names the arms."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pair in pairs:
        skill_first = skill_is_a(seed, judge, pair)
        skill_out, raw_out = output_of(pair["skill_key"]), output_of(pair["raw_key"])
        a, b = (skill_out, raw_out) if skill_first else (raw_out, skill_out)
        tag = hashlib.sha256(f"{judge}|{pair['skill_key']}|{pair['raw_key']}|{skill_first}".encode()).hexdigest()[:12]
        (out_dir / f"{tag}.prompt.md").write_text(
            "# Instructions\n\n" + JUDGE_SYSTEM + "\n\n" +
            JUDGE_USER.format(prompt=prompts[(pair["skill"], pair["scenario"])],
                              method=published_method(pair["skill"]), a=a, b=b))
        jobs.append({**pair, "tag": tag, "skill_first": skill_first, "judge": judge})
    (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=2))
    print(f"wrote {len(jobs)} prompts to {out_dir}")


def ingest(out_dir: Path) -> list[dict]:
    records = []
    for job in json.loads((out_dir / "jobs.json").read_text()):
        path = out_dir / f"{job['tag']}.verdict.json"
        if not path.is_file():
            print(f"  missing verdict {path.name} ({job['skill']}/{job['scenario']}#{job['rep']})")
            continue
        got = {"verdict": extract_json(path.read_text()), "served_model": job["judge"]}
        pair = {k: job[k] for k in ("skill", "scenario", "rep", "skill_key", "raw_key")}
        records.append(build_record(pair, job["judge"], job["skill_first"],
                                    output_of(job["skill_key"]), output_of(job["raw_key"]), got)
                       | {"batch": out_dir.name, "judge": out_dir.name})
    return records


def tally(records: list[dict], out_dir: Path, run_id: str, judges: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_skill: dict[str, dict] = {}
    for r in records:
        s = per_skill.setdefault(r["skill"].removeprefix(f"{BOX}-"), {j: {"skill": 0, "raw": 0, "tie": 0, "deltas": []} for j in judges})
        s[r["judge"]][r["winner_arm"]] += 1
        s[r["judge"]]["deltas"].append(r["delta"])
    summary = {"run_id": run_id, "judges": judges, "pairs_per_judge": len(records) // max(len(judges), 1),
               "label_swaps": [f"{r['skill']}/{r['scenario']}#{r['rep']} ({r['judge']})" for r in records if r["label_swap"]],
               "skills": {}}
    agree = total = 0
    by_pair: dict[tuple, dict] = {}
    for r in records:
        by_pair.setdefault((r["skill"], r["scenario"], r["rep"]), {})[r["judge"]] = r["winner_arm"]
    for verdicts in by_pair.values():
        if len(verdicts) == len(judges):
            total += 1
            agree += len(set(verdicts.values())) == 1
    for skill, js in sorted(per_skill.items()):
        summary["skills"][skill] = {j: {"skill_wins": v["skill"], "raw_wins": v["raw"], "ties": v["tie"],
                                        "mean_delta": round(statistics.mean(v["deltas"]), 2)} for j, v in js.items() if v["deltas"]}
    summary["pair_agreement"] = f"{agree}/{total}"
    dim_means = {d: round(statistics.mean(r["dims"][d]["skill"] - r["dims"][d]["raw"] for r in records), 2) for d in DIMENSIONS}
    summary["mean_dimension_delta"] = dim_means
    (out_dir / "tally.json").write_text(json.dumps(summary, indent=2))
    slim = [{k: r[k] for k in ("skill", "scenario", "rep", "judge", "served_model", "skill_is", "winner_arm",
                              "skill_total", "raw_total", "delta", "dims", "quote_found_in", "label_swap")}
            | {"winner_reason": r["verdict"].get("winner_reason")} for r in records]
    (out_dir / "verdicts.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tally", metavar="DIR", help="write tally.json and verdicts.json here")
    ap.add_argument("--skills", help="comma-separated short names to judge")
    ap.add_argument("--emit", metavar="DIR", help="write blind prompts for subagent judges, then stop")
    ap.add_argument("--ingest", metavar="DIR", nargs="+", help="read subagent verdicts from these dirs and tally")
    args = ap.parse_args()
    judges = args.judges.split(",")
    run = load_run(args.run)
    pairs = [p for p in pairs_of(run) if output_of(p["skill_key"]) and output_of(p["raw_key"])]
    if args.skills:
        wanted = {f"{BOX}-{s.strip()}" for s in args.skills.split(",")}
        pairs = [p for p in pairs if p["skill"] in wanted]
    prompts = scenario_prompts()
    if args.emit:
        emit(pairs, REPO / args.emit, args.seed, prompts, "claude-sonnet (subagent)")
        return 0
    if args.ingest:
        records = [r for d in args.ingest for r in ingest(REPO / d)]
        print(f"ingested {len(records)} verdicts; label swaps: {sum(r['label_swap'] for r in records)}")
        if args.tally:
            tally(records, REPO / args.tally, args.run, sorted({r["batch"] for r in records}))
        return 0
    jobs = [(p, j) for p in pairs for j in judges]
    print(f"run {args.run}: {len(pairs)} complete pairs x {len(judges)} judges = {len(jobs)} judgments")
    if args.dry_run:
        return 0
    api_key = load_env()["OPENROUTER_API_KEY"]
    records, failures, cost = [], [], 0.0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(judge_one, p, j, args.seed, prompts, api_key): (p, j) for p, j in jobs}
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            p, j = futs[fut]
            try:
                r = fut.result()
                records.append(r)
                cost += r.get("cost_usd") or 0
                flag = "  LABEL SWAP" if r["label_swap"] else ""
                print(f"  [{n}/{len(jobs)}] {p['skill']}/{p['scenario']}#{p['rep']} {j.split('/')[-1]}: {r['winner_arm']} ({r['delta']:+.2f}){flag}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{p['skill']}/{p['scenario']}#{p['rep']} {j}: {exc}")
                print(f"  [{n}/{len(jobs)}] FAILED {failures[-1]}", flush=True)
    print(f"\njudgments {len(records)}, failures {len(failures)}, cost ${cost:.2f} (cached judgments count their original cost)")
    for f in failures:
        print("  -", f)
    if args.tally:
        tally(records, REPO / args.tally, args.run, judges)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
