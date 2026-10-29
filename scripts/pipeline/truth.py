#!/usr/bin/env python3
"""Code-computed ground truth for answer-scored tasks (div5 hand-off, 2026-10-09, section 3C).

A truth module (one truth.py per task) defines

    compute(load)   -> the answer, JSON-serialisable; None when no record matches (an unachievable task)
    render(answer)  -> the answer string in the form the instruction asks for; "N/A" for None

and reads mock data only through load(app_dir, relpath), which parses hub/websites/<app_dir>/src/data/<relpath>
and records the file and its sha256. The answer therefore comes from the data files, never from a model.

check() runs the module twice in a fresh interpreter, then scores render(answer) with the task's own reward.py
(must be exactly 1.0) and two give-up answers, "N/A" and "" (each must score below 0.5; for an unachievable
task "N/A" is the reference and must score 1.0). Data paths are recorded under the training side's
layout (projects/CUA-Gym-Hub/websites/...); the files are byte-identical to hub/websites/ (checked 2026-10-10).

    python3 scripts/pipeline/truth.py check <bundle_dir> [--truth <truth.py>]
    python3 scripts/pipeline/truth.py run <truth.py>

Map tasks (div6-map, 2026-10-11): the map site is an external OpenStreetMap deployment with no data files. A map
truth module defines compute(load, api); api wraps scripts/_osm_common.py, the exact Nominatim and OSRM calls the
site itself makes (viewbox on geocoding, full-precision coordinates, the site's OSRM query string, its own time and
distance formatters). Every call is recorded with the sha256 of its result, and the raw OSRM duration and distance
go into provenance. For a stateless bundle the check also runs nemo_reward.py the way a rollout runs it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "hub" / "websites"
CLUSTER_PREFIX = "projects/CUA-Gym-Hub/websites"
GIVE_UPS = ("N/A", "")
FORBIDDEN = [r"\bopen\s*\(", r"\brequests\b", r"\burllib\b", r"\bsocket\b", r"\bsubprocess\b", r"\bos\.",
             r"\bimport\s+os\b", r"\bpathlib\b", r"\beval\s*\(", r"\bexec\s*\(", r"__import__", r"\bsys\.",
             r"\bimport\s+sys\b", r"\bshutil\b", r"\bglob\b"]

BOOT = r'''
import hashlib, importlib.util, json, sys
from pathlib import Path
truth_path, root = Path(sys.argv[1]), Path(sys.argv[2])
seen, cache = {}, {}
def load(app_dir, relpath):
    p = (root / app_dir / "src" / "data" / relpath).resolve()
    if root.resolve() not in p.parents:
        raise ValueError("load() outside the mock data: %s/%s" % (app_dir, relpath))
    if p not in cache:
        raw = p.read_bytes()
        seen["%s/src/data/%s" % (app_dir, relpath)] = hashlib.sha256(raw).hexdigest()
        cache[p] = json.loads(raw)
    return cache[p]
calls, names = [], []
class Api:
    """Map ground truth: the site's own Nominatim and OSRM calls (scripts/_osm_common.py), each one recorded."""
    def __init__(self):
        sys.path.insert(0, sys.argv[3])
        import _osm_common as C
        import os as _os
        _h = _os.environ.get("CUA_GYM_OSM_HOST")       # map host override: your own <OSM_HOST> (TASK_MAP.md)
        if _h:
            C.H, C.SITE = _h, f"http://{_h}:3000"
        self.C = C
        self.fmt_time, self.fmt_dist, self.VIEWBOX = C.fmt_time, C.fmt_dist, C.VIEWBOX
    def _rec(self, fn, args, result, extra=None):
        h = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
        calls.append(dict({"fn": fn, "args": args, "sha256": h}, **(extra or {})))
    def geocode(self, query):
        hits = self.C.nominatim(query)
        self._rec("geocode", [query], hits, {"hits": len(hits), "top": (hits[0].get("display_name") if hits else None)})
        for h in hits:
            nm = (h.get("name") or (h.get("display_name") or "").split(",")[0]).strip()
            if nm:
                names.append(nm)
        return hits
    def geocode_local(self, query, lat, lon, span):
        r = self.C.nominatim_local(query, lat, lon, span)
        self._rec("geocode_local", [query, lat, lon, span], r)
        return r
    def object(self, osm_type, osm_id):
        """The object page's data: Nominatim lookup with address details and OSM extratags (phone, website, ...)."""
        import urllib.parse
        key = {"node": "N", "way": "W", "relation": "R"}[osm_type] + str(osm_id)
        url = "http://%s:8085/lookup?osm_ids=%s&format=json&addressdetails=1&extratags=1" % (self.C.H, urllib.parse.quote(key))
        r = self.C._get(url, timeout=25)
        self._rec("object", [osm_type, osm_id], r)
        return r[0] if r else None
    def route(self, engine, a, b):
        d, m = self.C.osrm(engine, a, b)
        self._rec("route", [engine, list(a), list(b)], [d, m], {"duration_s": d, "distance_m": m})
        return d, m
spec = importlib.util.spec_from_file_location("truth_module", truth_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import inspect, re
wants_api = len(inspect.signature(mod.compute).parameters) >= 2
answer = mod.compute(load, Api()) if wants_api else mod.compute(load)
rendered = mod.render(answer)
def perturb(x):
    """Every number 1.5x + 2: outside any time or distance tolerance, still in the task's own answer format."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return int(round(x * 1.5)) + 2
    if isinstance(x, float):
        return round(x * 1.5 + 2, 3)
    if isinstance(x, (list, tuple)):
        return [perturb(i) for i in x]
    if isinstance(x, dict):
        return {k: perturb(v) for k, v in x.items()}
    if isinstance(x, str):
        return re.sub(r"\d+(\.\d+)?", lambda m: str(perturb(float(m.group(0)) if "." in m.group(0) else int(m.group(0)))), x)
    return x
probe_numeric = None
if re.search(r"\d", json.dumps(answer, default=str)):
    try:
        probe_numeric = mod.render(perturb(answer))
    except Exception:
        probe_numeric = None
print(json.dumps({"answer": answer, "rendered": rendered, "files": seen, "api_calls": calls,
                  "api_names": sorted(set(names)), "probe_numeric": probe_numeric}, sort_keys=True, default=str))
'''


def lint(code: str) -> list[str]:
    """Truth code may only read data through load(); no files, network, processes or dynamic code."""
    problems = [f"forbidden construct /{pat}/" for pat in FORBIDDEN if re.search(pat, code)]
    for name in ("compute", "render"):
        if not re.search(rf"^def {name}\s*\(", code, re.M):
            problems.append(f"missing top-level def {name}(...)")
    return problems


def run_truth(truth_path: Path, timeout: int = 900) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-I", "-c", BOOT, str(Path(truth_path).resolve()), str(DATA_ROOT),
                           str(REPO / "scripts")],
                          capture_output=True, text=True, timeout=timeout, cwd="/")
    if proc.returncode != 0:
        raise RuntimeError(f"truth.py failed (rc={proc.returncode}): {proc.stderr.strip()[-800:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    out["answer_sha256"] = hashlib.sha256(json.dumps(out["answer"], sort_keys=True, default=str)
                                          .encode()).hexdigest()
    out["data_files"] = [{"path": f"{CLUSTER_PREFIX}/{p}", "sha256": h} for p, h in sorted(out.pop("files").items())]
    # the answer hash already proves reproducibility; the call list (with result hashes) must match too
    out["api_sha256"] = hashlib.sha256(json.dumps(out.get("api_calls", []), sort_keys=True, default=str).encode()).hexdigest()
    return out


def live_state(app_dir: str) -> dict[str, Any]:
    """The mock's untouched state: what the agent sees and what an answer-only episode leaves behind. An external,
    stateless site (the map) has none."""
    if app_dir == "webarena_map_mock":
        return {}
    key = "CUA_GYM_" + re.sub(r"_mock$", "", app_dir).upper() + "_URL"
    base = os.environ.get(key)
    if not base:
        return {}
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/go?sid=truth-probe-{os.getpid()}", timeout=120) as resp:
            return json.load(resp).get("initial_state", {})
    except Exception:  # noqa: BLE001
        return {}

def score(bundle: Path, answer: str, states: dict[str, dict[str, Any]]) -> float:
    """The task's own reward.py on an answer-only episode (state untouched)."""
    manifest = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
    bundle = Path(bundle).resolve()
    spec = importlib.util.spec_from_file_location(f"_truth_{bundle.name}", bundle / manifest.get("reward_path", "reward.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    apps: dict[str, Any] = {}
    for app in manifest.get("apps") or []:
        st = states.get(app.get("name") or app.get("source_name"), {})
        doc = {"initial_state": st, "current_state": st, "state_diff": {}, "final_urls": [], "final_html": "",
               "final_text": ""}
        for key in (app.get("source_name"), app.get("name")):
            if key:
                apps[key] = doc
    evidence = {"schema_version": 1, "task_id": manifest.get("task_id"), "instruction": manifest.get("instruction"),
                "lane": "truth-check", "agent_answer": answer, "apps": apps, "observations": []}
    return float(module.evaluate(evidence)["score"])


def _shape(text: str) -> re.Pattern:
    """A regex for strings of the same form: letters, digits and punctuation runs kept in order."""
    parts = []
    for run in re.findall(r"[A-Za-z]+|\d+|\s+|[^A-Za-z\d\s]", text):
        parts.append(r"[A-Za-z]+" if run[0].isalpha() else r"\d{%d}" % len(run) if run[0].isdigit()
                     else r"\s+" if run[0].isspace() else re.escape(run))
    return re.compile("".join(parts))


def _atoms(v: Any) -> list[str]:
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, (list, tuple)):
        return [a for x in v for a in _atoms(x)]
    return [] if v is None else [str(v)]


def real_distractor(answer: Any, data_files: list[dict]) -> str | None:
    """Another value of the same data field as the answer's first item, not part of the answer: what a careless
    agent actually adds (another customer's email, another product of the catalogue, another reviewer). Found by
    locating a list of records where some record's field equals the item, then taking a sibling record's value of
    that field. None when the item is not a field value in the data (computed values such as month names)."""
    first = answer[0] if isinstance(answer, list) and answer else answer
    if isinstance(first, (list, tuple)):
        first = first[0] if first else None
    if first is None or isinstance(first, (float, bool, dict)):
        return None
    target, taken = str(first).strip().lower(), {a.strip().lower() for a in _atoms(answer)}
    for f in data_files:
        rel = f["path"].split("websites/", 1)[1]
        stack = [json.loads((DATA_ROOT / rel).read_bytes())]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack.extend(v for v in x.values() if isinstance(v, (dict, list)))
            elif isinstance(x, list):
                recs = [r for r in x if isinstance(r, dict)]
                keys = next(({k for k, v in r.items() if str(v).strip().lower() == target} for r in recs
                             if any(str(v).strip().lower() == target for v in r.values())), set())
                for key in keys:
                    for r in recs:
                        v = r.get(key)
                        if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip().lower() not in taken \
                                and str(v).strip():
                            return str(v).strip()
                stack.extend(v for v in x if isinstance(v, (dict, list)))
    return None


def false_item(answer: Any, rendered: str, data_files: list[dict] | None = None, kind: str = "") -> str | None:
    """The right answer plus one wrong element: a list gets a real same-form value from the data (else a made-up
    one), a number is off by one, a string is replaced by a real same-form value. Under the div4 rule (no false
    item, all-or-nothing values) it must score 0."""
    if answer is None or isinstance(answer, dict) or kind in ("range",):
        return None                                   # a range or named fields has no "extra item"
    if isinstance(answer, list) and len(answer) > 1:
        if len(answer) == 2 and all(isinstance(x, (int, float)) for x in answer) and kind != "list":
            return None                               # a [low, high] pair
        if any(isinstance(x, dict) for x in answer):
            return None
        d = real_distractor(answer, data_files or [])
        if not d:
            return None                               # the items are not field values in the data: no realistic probe
        if isinstance(answer[0], (list, tuple)) and len(answer[0]) == 2:
            return f"{rendered}, {d}: {answer[0][1]}"   # another real name with a plausible value
        return f"{rendered}, {d}"
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return rendered.replace(str(answer), str(answer + 1), 1) if str(answer) in rendered else str(answer + 1)
    if isinstance(answer, float):
        return rendered.replace(f"{answer:.2f}", f"{answer + 1:.2f}", 1) if f"{answer:.2f}" in rendered else str(answer + 1)
    return real_distractor(answer, data_files or [])


def nemo_score(bundle: Path, answer: str) -> float | None:
    """A stateless bundle's TRAINING reward, run the way a rollout runs it: nemo_reward.py piped to `python -` in an
    empty directory with only CUA_GYM_AGENT_ANSWER / CUA_GYM_FINAL_URLS set (TASK_MAP.md MAP-2). None if absent."""
    f = Path(bundle) / "nemo_reward.py"
    if not f.is_file():
        return None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        env = {"PATH": os.environ.get("PATH", ""), "CUA_GYM_AGENT_ANSWER": answer, "CUA_GYM_FINAL_URLS": "[]"}
        proc = subprocess.run([sys.executable, "-"], input=f.read_text(encoding="utf-8"), capture_output=True,
                              text=True, cwd=d, env=env, timeout=120)
    m = re.findall(r"REWARD:\s*([-0-9.eE]+)", proc.stdout)
    return float(m[-1]) if m else -1.0


def stateless(bundle: Path) -> bool:
    manifest = json.loads((Path(bundle) / "task.json").read_text(encoding="utf-8"))
    return any(a.get("stateless") for a in manifest.get("apps") or [])


def app_dirs(bundle: Path) -> list[str]:
    manifest = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
    return [a.get("name") or a.get("source_name") for a in manifest.get("apps") or []]


def check(bundle: Path, truth_path: Path, states: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every fact the export gate and the retrofit report need about one task's code-computed truth."""
    meta = json.loads((bundle / "task.json").read_text(encoding="utf-8")).get("metadata", {}) or {}
    stored = (meta.get("answer") or {}).get("truth") if isinstance(meta.get("answer"), dict) else None
    unachievable = bool(meta.get("unachievable"))
    rec: dict[str, Any] = {"task_id": bundle.name, "truth_py": str(truth_path), "stored_truth": stored,
                           "unachievable": unachievable, "problems": []}
    if not truth_path.is_file():
        return {**rec, "verdict": "no_truth", "problems": ["no truth.py"]}
    code = truth_path.read_text(encoding="utf-8")
    rec["reference_code"] = code
    if lint_problems := lint(code):
        return {**rec, "verdict": "error", "problems": lint_problems}
    try:
        first, second = run_truth(truth_path), run_truth(truth_path)
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "error", "problems": [f"{type(exc).__name__}: {exc}"[:900]]}
    rec.update({k: first[k] for k in ("answer", "rendered", "answer_sha256", "data_files")})
    rec["api_calls"] = first.get("api_calls", [])
    rec["reproducible"] = (first["answer_sha256"] == second["answer_sha256"] and first["rendered"] == second["rendered"]
                           and first.get("api_sha256") == second.get("api_sha256"))
    if states is None:
        states = {a: live_state(a) for a in app_dirs(bundle)}
    try:
        rec["reference_score"] = score(bundle, first["rendered"], states)
        rec["giveup_scores"] = {g or "<empty>": score(bundle, g, states) for g in GIVE_UPS}
        kind = str(((meta.get("answer") or {}) if isinstance(meta.get("answer"), dict) else {}).get("kind", ""))
        if first.get("api_calls"):          # map: numbers within tolerance are right; a name set gets a real candidate
            if kind in ("list", "names", "set") and isinstance(first["answer"], list):
                extra = next((n for n in first.get("api_names", [])
                              if n.strip().lower() not in {a.strip().lower() for a in _atoms(first["answer"])}), None)
                probe = f"{first['rendered']}, {extra}" if extra else None
            else:
                probe = first.get("probe_numeric")
        else:
            probe = false_item(first["answer"], first["rendered"], first["data_files"], kind)
        rec["false_item_probe"] = probe
        rec["false_item_score"] = score(bundle, probe, states) if probe is not None else None
        wrong = meta.get("wrong_move_answers")      # div8: the answers step_239's recorded wrong move produces
        if isinstance(wrong, list) and wrong:
            rec["wrong_move_scores"] = {str(w): score(bundle, str(w), states) for w in wrong[:5]}
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "error", "problems": [f"reward.py raised: {type(exc).__name__}: {exc}"[:900]]}
    if not rec["reproducible"]:
        rec["problems"].append("compute() is not reproducible: two runs gave different answers")
    if rec["reference_score"] != 1.0:
        rec["problems"].append(f"reward.py scores the code-computed answer {rec['reference_score']} (want 1.0)")
    if first["answer"] is None and not unachievable:
        rec["problems"].append("compute() found no matching record but the task is not marked unachievable")
    if unachievable and first["answer"] is not None:
        rec["problems"].append("the task is marked unachievable but compute() found an answer")
    for g, s in rec["giveup_scores"].items():
        if g == "N/A" and unachievable:
            continue
        if s >= 0.5:
            rec["problems"].append(f"give-up answer {g!r} scores {s} (want < 0.5)")
    if stateless(bundle):
        rec["nemo_reference_score"] = nemo_score(bundle, first["rendered"])
        rec["nemo_giveup_score"] = nemo_score(bundle, "N/A")
        if rec["nemo_reference_score"] is not None and rec["nemo_reference_score"] != rec["reference_score"]:
            rec["problems"].append(f"nemo_reward.py scores the code answer {rec['nemo_reference_score']} but reward.py "
                                   f"{rec['reference_score']}: training and verification disagree")
        if rec["nemo_giveup_score"] is not None and rec["nemo_giveup_score"] >= 0.5 and not unachievable:
            rec["problems"].append(f"nemo_reward.py pays {rec['nemo_giveup_score']} for 'N/A'")
        for w in rec.get("wrong_move_scores") or {}:
            if s := nemo_score(bundle, w):
                rec["problems"].append(f"nemo_reward.py pays {s} for the recorded wrong move's answer {w[:80]!r} (want 0)")
    for w, s in (rec.get("wrong_move_scores") or {}).items():
        if s:
            rec["problems"].append(f"the recorded wrong move's answer {w[:80]!r} scores {s} (want 0)")
    if rec["false_item_score"]:
        rec["problems"].append(f"lenient verifier: the right answer plus a wrong element ({rec['false_item_probe'][:80]!r}) "
                               f"scores {rec['false_item_score']} (want 0)")
    rec["verdict"] = "match" if not rec["problems"] else (
        "mismatch" if rec["reference_score"] != 1.0 or (first["answer"] is None) != unachievable else "defect")
    return rec


def check_mutation(bundle: Path, truth_path: Path) -> dict[str, Any]:
    """div7 C-family (state-change) tasks: truth.py computes the exact post-condition values the reward checks (the
    @handle, the verbatim text, the due date, the star rating), never written by hand. reward.py must define a
    module-level EXPECTED equal to compute()'s output. Reaching that state is proven by verification (golden replay
    1.0, initial 0.0, reward audit), which reads the back end through /go?sid=."""
    bundle = Path(bundle).resolve()
    rec: dict[str, Any] = {"task_id": bundle.name, "truth_py": str(truth_path), "problems": [], "kind": "mutation"}
    if not truth_path.is_file():
        return {**rec, "verdict": "no_truth", "problems": ["no truth.py"]}
    code = truth_path.read_text(encoding="utf-8")
    rec["reference_code"] = code
    if lint_problems := lint(code):
        return {**rec, "verdict": "error", "problems": lint_problems}
    try:
        first, second = run_truth(truth_path), run_truth(truth_path)
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "error", "problems": [f"{type(exc).__name__}: {exc}"[:900]]}
    rec.update({k: first[k] for k in ("answer", "rendered", "answer_sha256", "data_files")})
    rec["reproducible"] = first["answer_sha256"] == second["answer_sha256"]
    try:
        manifest = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location(f"_mut_{bundle.name}", bundle / manifest.get("reward_path", "reward.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = getattr(module, "EXPECTED", None)
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "error", "problems": [f"reward.py failed to import: {type(exc).__name__}: {exc}"[:600]]}
    norm = lambda v: json.dumps(v, sort_keys=True, default=str)
    rec["reward_expected"] = expected
    if expected is None:
        rec["problems"].append("reward.py defines no module-level EXPECTED (the post-condition values truth.py computes)")
    elif norm(json.loads(norm(expected))) != norm(json.loads(norm(first["answer"]))):
        rec["problems"].append(f"reward.py EXPECTED {norm(expected)[:200]} != truth.py compute() {norm(first['answer'])[:200]}")
    if not rec["reproducible"]:
        rec["problems"].append("compute() is not reproducible")
    rec["verdict"] = "match" if not rec["problems"] else "mismatch"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("bundle")
    c.add_argument("--truth", help="default: <bundle>/truth.py")
    r = sub.add_parser("run", help="lint and run one truth.py twice; print its answer (no bundle needed)")
    r.add_argument("truth")
    a = ap.parse_args()
    if a.cmd == "run":
        path = Path(a.truth).resolve()
        if problems := lint(path.read_text(encoding="utf-8")):
            print(json.dumps({"lint": problems}, indent=1))
            return 1
        first, second = run_truth(path), run_truth(path)
        first["reproducible"] = first["answer_sha256"] == second["answer_sha256"]
        print(json.dumps(first, indent=1, default=str))
        return 0 if first["reproducible"] else 1
    bundle = Path(a.bundle).resolve()
    rec = check(bundle, Path(a.truth).resolve() if a.truth else bundle / "truth.py")
    rec.pop("reference_code", None)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["verdict"] == "match" else 1


if __name__ == "__main__":
    sys.exit(main())
