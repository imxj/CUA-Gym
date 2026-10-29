#!/usr/bin/env python3
"""Assignment records for failure-grounded task generation (div5 hand-off 2026-10-09, section 3A).

One JSON per parent task, built from the policy's screen rollouts (trajectory_collection.jsonl rows, one row per
rollout), the per-task pass-rate CSV and the gate report when they exist, and the parent's stored answer metadata:

  {parent_task_id, intent, sites, start_urls, pass_rate_at_8, mean_reward, rewards, termination_statuses, steps,
   most_common_failure, failure_labels, failed_rollouts: [2-3 x {rollout, reward, answer, final_urls,
   verifier_stdout, last_actions, label}], passing_rollout, gate_reasons, variant_rule, parent_meta}

Rows are large (screenshots inside `response`), so they are streamed and only compact fields are kept.
Failure labels are machine-made from the answer, the reward and the stored truth, e.g.
"step cap: no answer after 50 actions", "gave up: answered 'N/A'", "wrong set: 2 of 4 truth items, 1 extra".

  python3 assignments.py --rows '<glob>' --snapshot X.cc.jsonl --meta task_meta.json --out-dir DIR
                         [--per-task-csv CSV] [--rejected REJECTED.jsonl] [--gate GATE.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any

REPLACE_REASONS = re.compile(r"contamina|duplicate|give.?up", re.I)   # saturated/hopeless map to harder/simpler
GIVE_UP = re.compile(r"^\s*(n/?a|none|not found|no such|unable|cannot|can't|could not|couldn't|does not exist|"
                     r"doesn't exist|no (matching|results?|record|answer))\b", re.I)
SAT, HOPELESS = 0.75, 0.25


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.$/:-]+", " ", str(text).lower())).strip()


def truth_items(truth: Any) -> list[str]:
    """The atomic strings an answer must name: list items, pair names, dict values, or the value itself."""
    if truth is None:
        return []
    if isinstance(truth, list):
        out = []
        for t in truth:
            out += truth_items(t) if isinstance(t, (list, dict)) else [str(t)]
        return out
    if isinstance(truth, dict):
        return [str(v) for v in truth.values() if not isinstance(v, (list, dict))]
    return [str(truth)]


def actions(resp: Any, last: int = 5) -> tuple[int, list[str]]:
    out = resp.get("output", []) if isinstance(resp, dict) else []
    calls = [o for o in out if isinstance(o, dict) and o.get("type") in ("function_call", "computer_call")]
    shown = []
    for c in calls[-last:]:
        args = c.get("arguments") if isinstance(c.get("arguments"), str) else json.dumps(c.get("arguments") or c.get("action"))
        shown.append(f"{c.get('name', c.get('type'))} {args[:160]}")
    return len(calls), shown


def label(row: dict, meta: dict) -> str:
    """A short machine-made failure label with its evidence."""
    reward, answer = float(row["reward"]), (row["answer"] or "").strip()
    unach = bool(meta.get("unachievable"))
    truth = (meta.get("answer") or {}).get("truth")
    if row["term"] is None and not answer:
        return f"step cap: no answer after {row['steps']} actions (last: {row['last_actions'][-1][:70] if row['last_actions'] else '-'})"
    if not answer:
        if row["term"] == "success":
            return f"claimed success, no answer: terminate(status=success) without an answer after {row['steps']} actions"
        return f"ended without an answer (termination {row['term']})"
    first = answer.splitlines()[0][:90]
    if unach:
        return f"hallucinated: answered '{first}' although the target does not exist"
    if GIVE_UP.match(answer):
        return f"gave up: answered '{first}'" + (f" (trap: {meta['trap']})" if meta.get("trap") else "")
    if 0 < reward < 1:
        return f"partial: reward {reward:.2f}, answered '{first}'"
    items = truth_items(truth)
    if len(items) > 1:
        a = f" {norm(answer)} "
        hit = sum(1 for t in items if f" {norm(t)} " in a or norm(t) in a)
        extra = "" if hit < len(items) else ", all truth items present but rejected (extra items or format)"
        return f"wrong set: {hit} of {len(items)} truth items named{extra}; answered '{first}'"
    if items:
        t = norm(items[0])
        if t and t in norm(answer):
            return f"right value, rejected: answer contains '{items[0]}' but scored 0 (extra text or items); answered '{first}'"
        nums_a = re.findall(r"-?\d[\d,]*\.?\d*", answer)
        if re.fullmatch(r"-?[\d,.$ ]+", str(items[0]).strip()) and nums_a:
            return f"wrong number: answered {nums_a[0]}, truth {items[0]}"
        return f"wrong value: answered '{first}', truth '{str(items[0])[:60]}'"
    return f"wrong answer: '{first}'"


def category(lbl: str) -> str:
    return lbl.split(":", 1)[0]


def variant_rule(pass_rate: float | None, gate_reasons: list[str]) -> str:
    if gate_reasons:
        return "replacement"
    if pass_rate is None:
        return "unscreened"
    if pass_rate > SAT:
        return "harder"
    if pass_rate < HOPELESS:
        return "simpler_or_fix"
    return "keep"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="glob of trajectory_collection.jsonl files")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--meta", required=True, help="task_id -> {answer, unachievable, trap, family, ...}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-task-csv")
    ap.add_argument("--rejected", help="gate rejected.jsonl (task_id + reasons)")
    ap.add_argument("--gate", help="gate.json")
    ap.add_argument("--k", type=int, default=8, help="rollouts per task in the screen")
    a = ap.parse_args()

    tasks = {}
    for line in open(a.snapshot, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            tp = r["task_payload"]
            tasks[r["context_compaction_task_id"]] = {"intent": tp["intent"], "sites": tp.get("sites"),
                                                      "start_urls": tp.get("start_urls")}
    meta = json.load(open(a.meta, encoding="utf-8"))
    rolls: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    files = sorted(glob.glob(a.rows))
    for f in files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r.get("context_compaction_task_id")
            if tid not in tasks:
                continue
            ed = r.get("evaluator_details") or {}
            steps, last = actions(r.get("response"))
            idx = int(r.get("context_compaction_rollout_index", len(rolls[tid])))
            rolls[tid][idx] = {"rollout": idx, "reward": float(r.get("reward") or 0), "term": r.get("termination_status"),
                               "answer": r.get("answer"), "final_urls": ed.get("final_urls"),
                               "verifier_stdout": (ed.get("stdout_tail") or "")[-300:], "steps": steps,
                               "last_actions": last}      # a re-run of a shard overwrites the same rollout index
    csv_rows = {}
    if a.per_task_csv and Path(a.per_task_csv).is_file():
        for r in csv.DictReader(open(a.per_task_csv, encoding="utf-8")):
            csv_rows[r["task_id"]] = r
    rejected: dict[str, list[str]] = collections.defaultdict(list)
    if a.rejected and Path(a.rejected).is_file():
        for line in open(a.rejected, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                tid = r.get("task_id") or r.get("context_compaction_task_id")
                why = r.get("reasons") or r.get("reason") or r.get("gate_reasons") or []
                rejected[tid] += why if isinstance(why, list) else [str(why)]

    out = Path(a.out_dir)
    (out / "assignments").mkdir(parents=True, exist_ok=True)
    summary = []
    for tid, t in sorted(tasks.items()):
        m = meta.get(tid, {})
        rs = [rolls[tid][i] for i in sorted(rolls[tid])]
        for r in rs:
            r["label"] = "pass" if r["reward"] >= 1.0 else label(r, m)
        n = len(rs)
        pass_rate = (sum(r["reward"] >= 1.0 for r in rs) / n) if n else None
        mean_reward = (sum(r["reward"] for r in rs) / n) if n else None
        csv_n = 0
        if tid in csv_rows and str(csv_rows[tid].get("n") or "0").strip() not in ("", "0"):
            csv_n = int(float(csv_rows[tid]["n"]))       # the cluster's numbers win once it has scored the task
            pass_rate = float(csv_rows[tid].get("reward") or pass_rate or 0)
            mean_reward = float(csv_rows[tid].get("mean_reward") or mean_reward or 0)
        # Screened: all k rollouts are in, or the cluster scored the task from at least 5 (a finished shard can
        # lose a few rollouts; 4 shard-1 seeds have 5 or 7).
        screened = n >= a.k or csv_n >= 5
        fails = [r for r in rs if r["reward"] < 1.0]
        cats = collections.Counter(category(r["label"]) for r in fails)
        top = cats.most_common(1)[0][0] if cats else None
        rep = next((r["label"] for r in fails if category(r["label"]) == top), None)
        failed_rollouts = sorted([r for r in fails if category(r["label"]) == top], key=lambda r: r["rollout"])[:2] + \
            sorted([r for r in fails if category(r["label"]) != top], key=lambda r: r["rollout"])[:1]
        passing = next((r for r in rs if r["reward"] >= 1.0), None)
        gate_all = rejected.get(tid, [])
        gate = [g for g in gate_all if REPLACE_REASONS.search(json.dumps(g) if not isinstance(g, str) else g)]
        rec = {"parent_task_id": tid, **t, "pass_rate_at_8": pass_rate, "mean_reward": mean_reward, "n_rollouts": n,
               "csv_n": csv_n, "screened": screened,
               "rewards": [r["reward"] for r in rs], "termination_statuses": [r["term"] for r in rs],
               "steps": [r["steps"] for r in rs],
               "most_common_failure": rep, "failure_labels": dict(cats),
               "failed_rollouts": [{k: r[k] for k in ("rollout", "reward", "answer", "final_urls", "verifier_stdout",
                                                      "last_actions", "label")} for r in failed_rollouts],
               "passing_rollout": {k: passing[k] for k in ("rollout", "answer", "final_urls", "steps", "last_actions")}
               if passing else None,
               "gate_reasons": gate, "gate_reasons_all": gate_all, "variant_rule": variant_rule(pass_rate, gate) if screened or gate else "unscreened",
               "parent_meta": m}
        (out / "assignments" / f"{tid}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
        summary.append(rec)
    with (out / "assignments.jsonl").open("w", encoding="utf-8") as fh:
        for rec in summary:
            fh.write(json.dumps(rec, default=str) + "\n")
    rules = collections.Counter(r["variant_rule"] for r in summary)
    cats = collections.Counter(category(r["most_common_failure"]) for r in summary if r["most_common_failure"])
    print(f"{len(summary)} assignments from {len(files)} row file(s); rollouts per task: "
          f"{dict(collections.Counter(r['n_rollouts'] for r in summary))}")
    print("variant rule:", dict(rules))
    print("most common failure:", dict(cats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
