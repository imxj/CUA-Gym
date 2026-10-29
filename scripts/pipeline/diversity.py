"""Diversity model for WebArena task batches: what the corpus already covers, where the gaps are,
how to plan slots into those gaps, how to check a bundle against its slot, and how to report a batch.

Corpus = every task already delivered: the exported rows of the six prior batches
(prior_batches/webarena_*.jsonl) plus every bundle under other pipeline batches' tasks/ directories.

Four axes, each measured the same way in planning, gating and reporting:

  family   the task's shape. Prior tasks: first three tokens of the id stem (the authoring lane/topic).
           New tasks: metadata.family, a snake_case slug that names the shape, not the entity.
  entity   named things in the instruction (projects, forums, products, customers, quoted strings).
           Overused = the ten most-mentioned per site in the corpus.
  field    state the reward checks: second-level fields of the live state document (and record fields
           of list-valued keys) that a reward names. Gaps = fields no reward has ever named.
  official the 4-site single-app WebArena intents; uncovered = no corpus intent within token-Jaccard 0.3.

Deliberately simple and deterministic (token sets, regexes, a seeded RNG) so a gate decision can be
reproduced and argued with. It measures lexical and structural diversity, not semantic diversity.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import random
import re
import statistics
import urllib.request
from pathlib import Path
from typing import Any

SITES = ("gitlab", "reddit", "shopping", "shopping_admin")
APP = {s: f"webarena_{s}_mock" for s in SITES}
SITE_OF_APP = {v: k for k, v in APP.items()}
SITE_OF_APP["webarena_map_mock"] = "map"         # div6-map: external OpenStreetMap site, stateless
STOP = set("""a an and are as at be by for from has have in into is it its must my of on one or so that the then
there this to two up want was with you your i me we us do does not no new page open go click use using make sure
should will when after before each all any find set change update create add""".split())
COVER_THRESHOLD = 0.3      # official intent counts as covered when some corpus intent is this close
NEAR_DUP = 0.55            # a new intent this close to a corpus intent is a near-duplicate
SEED_COPY = 0.75           # a new intent this close to its official seed is a copy, not a resemblance
FAMILY_CLASH = 0.67        # a new family slug this close to an existing family key is the same family
SHAPE_DUP = 0.55           # two intents this close once names, numbers and quotes are blanked are one template
SHAPE_STOP = {"who", "whom", "them", "their", "which", "what", "exactly", "same"}


def toks(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOP and len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def entities(text: str) -> set[str]:
    """Named things in an instruction: quoted strings, a/b paths, f/forums, Capitalised Multi Words."""
    found = set(re.findall(r"[\"“']([^\"”']{3,60})[\"”']", text or ""))
    found |= set(re.findall(r"\b[a-z0-9_.-]+/[a-z0-9_.-]+\b", text or ""))
    found |= set(re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b", text or ""))
    return {e.strip() for e in found if e.strip() and e.strip().lower() not in STOP}


def prior_family(task_id: str, site: str) -> str:
    stem = re.sub(r"_\d+$", "", task_id)
    stem = re.sub(rf"^{site}_", "", stem)
    return "_".join(stem.split("_")[:3])


def family_tokens(slug: str) -> set[str]:
    return {w for w in re.split(r"[_\W]+", (slug or "").lower()) if w and w not in STOP}


def _stem(w: str) -> str:
    if w.endswith("sses"):
        return w[:-2]
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 4:
            w = w[:-len(suf)]
            if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeious":     # cancell -> cancel, shipp -> ship
                w = w[:-1]
            return w
    return w


def shape(text: str) -> set[str]:
    """Template tokens of an intent: intent_template() lightly stemmed. div2 shipped twins the word-level check
    missed ("the customer who has the most canceled orders" / "the customer with the most cancelled orders", and
    four "In <repo>, comment <text> on the open issue whose description mentions <x>" tasks)."""
    return {_stem(w) for w in intent_template(text).split()
            if (len(w) > 2 or w in {"Q", "P", "N", "E"}) and w not in SHAPE_STOP}


# --------------------------------------------------------------------------- corpus

class Corpus:
    """Everything delivered so far, indexed per site."""

    def __init__(self, repo: Path, exclude_batch_dirs: list[Path] | None = None):
        self.repo = repo
        self.tasks: list[dict[str, Any]] = []
        for f in sorted(glob.glob(str(repo / "prior_batches" / "webarena_*.jsonl"))):
            batch = Path(f).stem.replace("webarena_", "")
            for line in open(f, encoding="utf-8"):
                p = json.loads(line)["task_payload"]
                site = SITE_OF_APP.get(p["sites"][0])
                if not site:
                    continue
                self.tasks.append(self._record(p["task_id"], site, batch, p["intent"],
                                               p["cuagym"].get("eval_reward_code") or "",
                                               prior_family(p["task_id"], site)))
        skip = {d.resolve() for d in exclude_batch_dirs or []}
        for manifest in sorted(repo.glob("output/**/tasks/*/*/task.json")):
            bundle = manifest.parent
            if any(s == bundle.resolve() or s in bundle.resolve().parents for s in skip):
                continue
            rec = bundle_record(bundle, batch=bundle.parents[2].name)
            if rec:
                self.tasks.append(rec)
        self.by_site: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for t in self.tasks:
            self.by_site[t["site"]].append(t)

    @staticmethod
    def _record(task_id, site, batch, intent, reward_code, family):
        return {"id": task_id, "site": site, "batch": batch, "intent": intent, "tok": toks(intent),
                "shape": shape(intent), "ents": entities(intent), "family": family, "lits": literals(reward_code)}

    def families(self, site: str) -> collections.Counter:
        return collections.Counter(t["family"] for t in self.by_site[site])

    def overused_entities(self, site: str, k: int = 10) -> list[str]:
        return [e for e, _ in collections.Counter(e for t in self.by_site[site] for e in t["ents"]).most_common(k)]

    def max_sim(self, site: str, tok: set[str], skip_id: str | set[str] | None = None,
                key: str = "tok") -> tuple[float, str | None]:
        best, who = 0.0, None
        skip = skip_id if isinstance(skip_id, set) else {skip_id}
        for t in self.by_site[site]:
            if t["id"] in skip:
                continue
            s = jaccard(tok, t[key])
            if s > best:
                best, who = s, t["id"]
        return best, who


def literals(code: str) -> set[str]:
    return set(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", code or ""))


def bundle_record(bundle: Path, batch: str = "") -> dict[str, Any] | None:
    try:
        manifest = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
        ti = json.loads((bundle / "task_instruction.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    site = SITE_OF_APP.get(ti.get("app_dir", ""), bundle.parent.name)
    meta = manifest.get("metadata", {}) or {}
    code = "".join((bundle / f).read_text(encoding="utf-8", errors="ignore")
                   for f in ("reward.py", "nemo_reward.py") if (bundle / f).is_file())
    intent = ti.get("task_instruction", "")
    return {"id": manifest.get("task_id", bundle.name), "site": site, "batch": batch, "intent": intent,
            "tok": toks(intent), "shape": shape(intent), "ents": entities(intent), "lits": literals(code),
            "family": meta.get("family") or prior_family(bundle.name, site), "meta": meta,
            "words": len(intent.split()), "bundle": str(bundle)}


# --------------------------------------------------------------------------- state fields and catalogs

def live_state(site: str, base: str) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/go?sid=diversity-probe"
    return json.load(urllib.request.urlopen(url, timeout=60))["initial_state"]


def state_fields(state: dict[str, Any]) -> dict[str, list[str]]:
    """{top-level key: [sub-field, ...]} for non-id dicts and for the records of list-valued keys."""
    out: dict[str, list[str]] = {}
    for k, v in state.items():
        if isinstance(v, dict) and v and not all(re.fullmatch(r"\d+", x) for x in v):
            out[k] = list(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = list(v[0])
        else:
            out[k] = []
    return out


EXCLUDE_KEYS = {"ui", "nextIds"}                      # UI chrome and id counters: nothing a task should be about
EXCLUDE_FIELDS = {"admin", "trusted", "id", "entity_id", "increment_id", "created_at", "updated_at",
                  "creation_time", "update_time", "store_id", "website_id"}   # ids, timestamps, permission flags


def field_gaps(corpus: Corpus, site: str, fields: dict[str, list[str]], rare: int = 3,
               exclude_keys: set[str] | None = None, exclude_fields: set[str] | None = None) -> list[dict[str, Any]]:
    """Fields no corpus reward names (or names rarely), grouped by parent key, emptiest groups first."""
    count = collections.Counter(x for t in corpus.by_site[site] for x in t["lits"])
    xk = EXCLUDE_KEYS if exclude_keys is None else exclude_keys
    xf = EXCLUDE_FIELDS if exclude_fields is None else exclude_fields
    gaps = []
    for key, subs in fields.items():
        if key in xk:
            continue
        never = [s for s in subs if count[s] == 0 and len(s) > 2 and s not in xf]
        if count[key] <= rare and not subs:
            gaps.append({"key": key, "fields": [], "key_checks": count[key]})
        elif never:
            gaps.append({"key": key, "fields": never, "key_checks": count[key]})
    return sorted(gaps, key=lambda g: (-len(g["fields"]), g["key_checks"]))


def catalog(repo: Path, site: str, state: dict[str, Any]) -> list[str]:
    """Names of seed entities an instruction could use, from the mock's data and live state."""
    data = repo / "hub" / "websites" / APP[site] / "src" / "data"
    names: list[str] = []

    def from_records(rows, *fields):
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            if "firstname" in r and "lastname" in r:
                names.append(f"{r['firstname']} {r['lastname']}")
            for f in fields:
                if isinstance(r.get(f), str) and r[f].strip():
                    names.append(r[f].strip())
                    break

    def load(name):
        p = data / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []

    if site == "gitlab":
        names += list(load("repo_files.json") or {})
    elif site == "reddit":
        names += [f"f/{f['name']}" for f in load("forums.json") if isinstance(f, dict) and f.get("name")]
        from_records(load("users.json"), "username")
    elif site == "shopping":
        from_records(load("products.json"), "name")
    elif site == "shopping_admin":
        from_records(load("products.json"), "name")
        for key in ("customers", "cmsPages", "cmsBlocks", "cartPriceRules"):
            from_records(state.get(key, []), "title", "name", "email")
    return sorted({n for n in names if 3 <= len(n) <= 80})


def long_tail(corpus: Corpus, site: str, names: list[str], limit: int = 4000) -> list[str]:
    """Catalog entries no corpus instruction mentions (case-insensitive substring)."""
    text = "\n".join(t["intent"].lower() for t in corpus.by_site[site])
    return [n for n in names[:limit] if n.lower() not in text]


# --------------------------------------------------------------------------- official intents

def official_intents(repo: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    path = repo / "webarena_benchmarks" / "webarena.jsonl"
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        web = r.get("web_name")
        if isinstance(web, list) and len(web) == 1 and web[0] in SITES:
            out[web[0]].append({"id": r.get("id"), "intent": r["ques"], "tok": toks(r["ques"])})
    return out


def uncovered_official(corpus: Corpus, official: dict[str, list[dict[str, Any]]], site: str) -> list[dict[str, Any]]:
    gen = [t["tok"] for t in corpus.by_site[site]]
    out = []
    for o in official[site]:
        best = max((jaccard(o["tok"], g) for g in gen), default=0.0)
        if best < COVER_THRESHOLD:
            out.append({**o, "nearest": round(best, 2)})
    return out


def spread(items: list[dict[str, Any]], k: int, rng: random.Random, key="tok", min_gap: float = 0.3):
    """Pick up to k items that are pairwise dissimilar (greedy over a shuffled order)."""
    pool = items[:]
    rng.shuffle(pool)
    chosen: list[dict[str, Any]] = []
    for it in pool:
        if all(jaccard(it[key], c[key]) < min_gap for c in chosen):
            chosen.append(it)
        if len(chosen) == k:
            break
    return chosen


# --------------------------------------------------------------------------- planning

def intent_template(text: str) -> str:
    """An official intent with its instance values blanked, so "most recent cancelled order" and "most recent
    pending order" are one template: quoted strings, numbers, paths/forums and Capitalised Names -> placeholders."""
    t = re.sub(r"[\"“'][^\"”']{1,80}[\"”']", " Q ", text or "")
    t = re.sub(r"\b[\w.-]+/[\w.-]+\b", " P ", t)
    t = re.sub(r"\b\d[\d,./:-]*\b", " N ", t)
    words = []
    for i, w in enumerate(t.split()):
        if w in ("Q", "P", "N"):
            words.append(w)
        elif i > 0 and w[:1].isupper():          # a name mid-sentence; the sentence's first word is kept
            words.append("E")
        else:
            words.append(w.lower())
    toks = [x for x in re.findall(r"[a-z]+|[QPNE]", " ".join(words)) if x in ("Q", "P", "N", "E") or x not in STOP]
    return " ".join(x for j, x in enumerate(toks) if not (x == "E" and j and toks[j - 1] == "E"))


# Skill pairs the corpus is thin on (batch-5 coverage: A5 12, A6 19, A13 39, R8 40, R7 59 of 600), grounded per site:
# A5 arithmetic edits and A6 attribute matrices exist only on shopping_admin.
OPEN_PAIRS = {
    "gitlab": [["R7", "A3"], ["R8", "A12"], ["R2", "A10"], ["R6", "A8"], ["R4", "A13"], ["R3", "A7"], ["R1", "A12"],
               ["R9", "A13"]],
    "reddit": [["R7", "A8"], ["R8", "A2"], ["R6", "A9"], ["R2", "A10"], ["R4", "A13"], ["R3", "A7"], ["R9", "A13"]],
    "shopping": [["R6", "A2"], ["R2", "A11"], ["R7", "A4"], ["R8", "A4"], ["R4", "A13"], ["R3", "A7"], ["R9", "A13"]],
    "shopping_admin": [["R10", "A5"], ["R1", "A6"], ["R6", "A10"], ["R2", "A13"], ["R7", "A3"], ["R8", "A12"],
                       ["R3", "A5"], ["R9", "A6"]],
}


# Hard chains (batch-5 rule: 3+ skills, two distinct retrievals or one retrieval feeding two dependent actions),
# built from the lookups that made div1 tasks learnable: two-hop joins (R7), counts (R3), ordinals (R4), content
# search (R6), date windows (R2), faceted lists (R5), superlatives (R1), reports (R10).
HARD_CHAINS = {
    "gitlab": [["R7", "R3", "A7"], ["R4", "R1", "A3"], ["R6", "R4", "A12"], ["R2", "R3", "A10"], ["R1", "A12", "A3"],
               ["R5", "R4", "A13"]],
    "reddit": [["R6", "R4", "A8"], ["R7", "R3", "A7"], ["R2", "R1", "A10"], ["R8", "R4", "A2"], ["R4", "A8", "A9"],
               ["R3", "R6", "A7"]],
    "shopping": [["R5", "R1", "A2"], ["R6", "R3", "A7"], ["R2", "R4", "A11"], ["R7", "R1", "A4"], ["R1", "A4", "A2"],
                 ["R3", "R5", "A13"]],
    "shopping_admin": [["R10", "R3", "A5"], ["R2", "R4", "A10"], ["R6", "R3", "A7"], ["R7", "R1", "A6"],
                       ["R1", "A5", "A10"], ["R5", "R4", "A13"]],
}


def skills_key(skills) -> tuple:
    """Order-free key for a skill set: retrievals first, then actions, numerically."""
    return tuple(sorted(set(skills or []), key=lambda x: (x[0] != "R", int(x[1:]) if x[1:].isdigit() else 0)))


def feedback_weights(repo: Path, sources: list[dict[str, str]]) -> tuple[dict, dict]:
    """Sampling weights per (site, skills_key) from earlier pass@16 screens.

    sources = [{"per_task_csv": <aggregate.py per_task.csv>, "tasks_dir": <batch tasks/ dir>}, ...]
    Mean k over a skill set's tasks: >= 14 of 16 (saturated, no GRPO signal) -> 0.25; 1..14 (learnable) -> 1.5;
    < 1 (never solved: too hard or broken) -> 0.6; unseen -> 1.0.
    """
    import csv
    meta = {}
    for src in sources:
        for m in (repo / src["tasks_dir"]).glob("*/*/task.json"):
            try:
                md = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            site = m.parent.parent.name
            meta[md.get("task_id", m.parent.name)] = (site, skills_key((md.get("metadata") or {}).get("skills")))
    ks: dict[tuple, list[int]] = collections.defaultdict(list)
    for src in sources:
        p = repo / src["per_task_csv"]
        if not p.is_file():
            continue
        for row in csv.DictReader(open(p, encoding="utf-8")):
            if row["task_id"] in meta:
                ks[meta[row["task_id"]]].append(int(row["n_success"]))
    weights = {}
    for key, vals in ks.items():
        mean = sum(vals) / len(vals)
        weights[key] = 0.25 if mean >= 14 else (1.5 if mean >= 1 else 0.6)
    return weights, {k: (len(v), round(sum(v) / len(v), 1)) for k, v in ks.items()}


def plan_slots(repo: Path, corpus: Corpus, plan: dict[str, Any], bases: dict[str, str]) -> list[dict[str, Any]]:
    """Slots with a novelty contract each.

    plan = {"sites": {site: n}, "seed": int, "seed_share": 0.5, "field_share": 0.2, "explicit_share": 0.3,
            "short_share": 0.4, "entity_candidates": 4, "open_pairs": {site: [[R, A], ...]},
            # harder batches (div3):
            "hard_share": 0.3,            share of all slots that are hard 3-skill chains (taken from open slots)
            "hard_chains": {site: [...]}, override HARD_CHAINS
            "drop_r9_only": true,         no open pair whose only retrieval is R9 (attribute of a named record)
            "require_lookup": true,       every slot must declare metadata.lookup; its value may not be in the text
            "golden_actions": [10, 20],   target golden-path length in UI actions
            "feedback": [{"per_task_csv": ..., "tasks_dir": ...}]   weight pairs/chains by earlier pass@16}

    Kinds: official_seed (an uncovered official intent, at most one per template), field (a never-checked
    state-field group, each used once), open (a corpus-thin skill pair or a hard chain). Shortfalls in seeds or
    fields become open slots, so any batch size plans.

    A plan with "families" (div4) plans by target failure class instead; see plan_family_slots.
    """
    if plan.get("assignments"):
        return plan_assignment_slots(repo, corpus, plan, bases)
    if plan.get("families"):
        return plan_family_slots(repo, corpus, plan, bases)
    rng = random.Random(plan.get("seed", 0))
    official = official_intents(repo)
    weights, _ = feedback_weights(repo, plan.get("feedback") or []) if plan.get("feedback") else ({}, {})
    slots: list[dict[str, Any]] = []
    n = 0
    for site, count in plan["sites"].items():
        state = live_state(site, bases[site])
        gaps = field_gaps(corpus, site, state_fields(state),
                          exclude_keys=set(plan.get("exclude_keys", EXCLUDE_KEYS)),
                          exclude_fields=set(plan.get("exclude_fields", EXCLUDE_FIELDS)))
        gaps = [g for g in gaps if g["fields"]]                       # need named sub-fields to target
        all_gap_fields = sorted({f for g in gaps for f in g["fields"]})
        n_field = min(round(count * plan.get("field_share", 0.2)), len(gaps))
        by_template: dict[str, dict[str, Any]] = {}
        uncovered = uncovered_official(corpus, official, site)
        rng.shuffle(uncovered)
        for o in uncovered:                                           # one representative per template
            by_template.setdefault(intent_template(o["intent"]), o)
        seeds = spread(list(by_template.values()), min(round(count * plan.get("seed_share", 0.5)), count - n_field),
                       rng, min_gap=plan.get("seed_min_gap", 0.35))
        n_open = count - n_field - len(seeds)
        n_hard = min(round(count * plan.get("hard_share", 0.0)), n_open)
        pairs = (plan.get("open_pairs") or {}).get(site) or OPEN_PAIRS[site]
        if plan.get("drop_r9_only"):
            pairs = [p for p in pairs if [x for x in p if x.startswith("R")] != ["R9"]] or pairs
        chains = (plan.get("hard_chains") or {}).get(site) or HARD_CHAINS[site]

        def draw(options: list[list[str]], k: int) -> list[list[str]]:
            """k picks with frequency proportional to the feedback weight (unseen options weigh 1.0)."""
            w = [weights.get((site, skills_key(o)), 1.0) for o in options]
            return rng.choices(options, weights=w, k=k) if k > 0 else []
        chain_picks, pair_picks = draw(chains, n_hard), draw(pairs, n_open - n_hard)
        tail = long_tail(corpus, site, catalog(repo, site, state))
        rng.shuffle(tail)
        avoid_ents = corpus.overused_entities(site)
        avoid_fams = [f for f, _ in corpus.families(site).most_common(12)]
        kinds = (["official_seed"] * len(seeds) + ["field"] * n_field + ["hard"] * n_hard
                 + ["open"] * (n_open - n_hard))
        rng.shuffle(kinds)
        seed_it, gap_it, chain_it, pair_it = iter(seeds), iter(gaps), iter(chain_picks), iter(pair_picks)
        k_ent = plan.get("entity_candidates", 4)
        for i, kind in enumerate(kinds):
            n += 1
            ents = [tail[(i * k_ent + j) % len(tail)] for j in range(k_ent)] if tail else []
            slot: dict[str, Any] = {
                "slot": n, "site": site, "kind": "open" if kind == "hard" else kind,
                "style": "explicit" if rng.random() < plan.get("explicit_share", 0.3) else "terse",
                "short": rng.random() < plan.get("short_share", 0.4),
                "entity_candidates": list(dict.fromkeys(ents)),
                "avoid_entities": avoid_ents, "avoid_families": avoid_fams,
            }
            if plan.get("require_lookup"):
                slot["require_lookup"] = True
            if plan.get("golden_actions"):
                slot["golden_actions"] = plan["golden_actions"]
            if kind == "official_seed":
                s = next(seed_it)
                slot.update({"seed_id": s["id"], "seed_intent": s["intent"]})
            elif kind == "field":
                g = next(gap_it)
                targets = g["fields"][:4]
                # Some never-checked fields have no UI writer (computed counters, flags). Any other never-checked
                # field on the site keeps the novelty and saves a futile re-author.
                slot.update({"target_key": g["key"], "target_fields": targets,
                             "alt_fields": [f for f in all_gap_fields if f not in targets][:30]})
            elif kind == "hard":
                slot.update({"pair": next(chain_it), "hard": True})
            else:
                slot["pair"] = next(pair_it)
            slots.append(slot)
    return slots


def plan_sites(plan: dict[str, Any]) -> list[str]:
    """Sites a plan touches, in first-seen order."""
    if plan.get("assignments"):
        return list(SITES)
    if plan.get("families"):
        return list(dict.fromkeys(s for f in plan["families"] for s in f["sites"]))
    return list(plan["sites"])


def plan_family_slots(repo: Path, corpus: Corpus, plan: dict[str, Any], bases: dict[str, str]) -> list[dict[str, Any]]:
    """Slots by target failure class (div4) instead of by skill pair.

    plan["families"] = [{"key": "f2_enumeration", "sites": {site: n}, "brief": "what the agent must learn",
                         "examples": {site: ["sub-type", ...]}, "answer_scored": true, "require_lookup": false,
                         "golden_actions": [lo, hi], "unachievable": 0}, ...]
    plan["intent_words"] = [10, 25] and plan["no_urls"] = true apply to every slot. Each slot gets one example
    sub-type (cycled, so every example is used before any repeats) plus the usual unused / over-used entities.
    "unachievable": k marks k slots of the family, spread across its sites, whose target must not exist.
    """
    rng = random.Random(plan.get("seed", 0))
    k_ent = plan.get("entity_candidates", 4)
    slots: list[dict[str, Any]] = []
    n = 0
    for fam in plan["families"]:
        fam_slots: list[dict[str, Any]] = []
        for site, count in fam["sites"].items():
            if site == "map":
                tail = []                               # entities come from the harvested place pool, per city
            else:
                tail = long_tail(corpus, site, catalog(repo, site, live_state(site, bases[site])))
            rng.shuffle(tail)
            avoid_ents = corpus.overused_entities(site)
            if site == "map":                           # city names are the slot's own setting, not entities to avoid
                cityset = {c.lower() for c in plan.get("map_cities", [])}
                avoid_ents = [e for e in avoid_ents if e.lower() not in cityset and e.lower() not in ("success",)]
            avoid_fams = [f for f, _ in corpus.families(site).most_common(12)]
            examples = list((fam.get("examples") or {}).get(site, []))
            rng.shuffle(examples)
            rungs = int(fam.get("ladder") or 1)
            evidence = []
            if fam.get("evidence"):                     # div7: the WebArena failures this family must answer
                rows = [json.loads(l) for l in open(repo / fam["evidence"], encoding="utf-8") if l.strip()]
                if fam.get("evidence_ids"):                 # div9: one family per source task
                    rows = [r for r in rows if r.get("task_id") in set(fam["evidence_ids"])]
                evidence = [r for r in rows if r.get("site") == site] or rows
                rng.shuffle(evidence)
            for i in range(count):
                n += 1
                ents = [tail[(i * k_ent + j) % len(tail)] for j in range(k_ent)] if tail else []
                city = None
                if site == "map" and plan.get("map_pool"):
                    if rungs > 1 and i % rungs:             # div8 map ladders: every rung shares the ladder's city
                        city, ents = fam_slots[-1].get("city"), list(fam_slots[-1].get("entity_candidates") or [])
                    else:                                   # round-robin by ladder, not by slot (27 cities, 3 rungs)
                        city, ents = map_city_places(repo, plan, (n - 1) // rungs + 1 if rungs > 1 else n, rng)
                slot: dict[str, Any] = {
                    "slot": n, "site": site, "kind": "family", "style": "terse", "short": False,
                    "div4_family": fam["key"], "family_brief": fam["brief"],
                    "entity_candidates": list(dict.fromkeys(ents)),
                    "avoid_entities": avoid_ents, "avoid_families": avoid_fams,
                }
                if examples:
                    slot["example"] = examples[i % len(examples)]
                for key in ("answer_scored", "require_lookup"):
                    if fam.get(key):
                        slot[key] = True
                if fam.get("golden_actions") or plan.get("golden_actions"):
                    slot["golden_actions"] = fam.get("golden_actions") or plan["golden_actions"]
                if plan.get("intent_words"):
                    slot["intent_words"] = plan["intent_words"]
                if plan.get("no_urls"):
                    slot["no_urls"] = True
                if city:
                    slot["city"] = city
                if rungs > 1:                               # div7 ladders: consecutive slots, one lane per ladder
                    k = i // rungs
                    slot.update({"ladder_id": f"{fam['key']}-{site}-{k + 1:02d}", "rung": i % rungs + 1,
                                 "ladder_rungs": rungs, "rung_spec": fam.get("rung_spec") or {}})
                    if examples:
                        slot["example"] = examples[k % len(examples)]
                    if evidence:
                        src = [evidence[(2 * k + j) % len(evidence)] for j in range(min(2, len(evidence)))]
                        slot["source_failures"] = [{key: r.get(key) for key in ("task_id", "intent", "subpattern", "steps",
                                                                                "agent_believed", "truth", "site",
                                                                                "data_fix")}
                                                   for r in src]
                elif evidence:
                    slot["source_failures"] = [{key: r.get(key) for key in ("task_id", "intent", "subpattern", "steps",
                                                                            "agent_believed", "truth", "site")}
                                               for r in [evidence[i % len(evidence)]]]
                if fam.get("div7_family"):
                    slot["div7_family"] = fam["div7_family"]
                for key in ("template", "fixed_steps", "wrong_move", "step_terms",     # div8 fixed-step templates
                            "source_task", "source_origin", "source_eval", "source_reference", "sibling_kinds",
                            "webarena_grading",                                     # div9 siblings
                            "div10"):                                               # div10 skill lanes
                    if fam.get(key):
                        slot[key] = fam[key]
                for key in ("banned_terms", "contamination_ngram", "code_truth", "stateless", "tolerance", "mutation",
                            "max_ui_actions"):
                    if plan.get(key) or fam.get(key):
                        slot[key] = fam.get(key, plan.get(key))
                fam_slots.append(slot)
        k = min(int(fam.get("unachievable", 0)), len(fam_slots))
        for i in range(k):                                            # evenly spaced, so spread over the sites
            fam_slots[int((i + 0.5) * len(fam_slots) / k)]["unachievable"] = True
        slots += fam_slots
    return slots


APP_SITE = {"webarena_shopping_mock": "shopping", "webarena_shopping_admin_mock": "shopping_admin",
            "webarena_gitlab_mock": "gitlab", "webarena_reddit_mock": "reddit"}
VARIANT_ORDER = ["replacement", "fix", "harder", "simpler"]


def plan_assignment_slots(repo: Path, corpus: Corpus, plan: dict[str, Any], bases: dict[str, str]) -> list[dict[str, Any]]:
    """Slots from assignment records (div5 hand-off 3A/3B): one new task per (parent, variant).

    plan["assignments"] = {"path": "<assignments.jsonl>", "truth_report": "<div4 truth/report.jsonl>",
                           "include": ["replacement", "fix", "harder", "simpler"], "max": 60}
    variant per parent: gate-rejected -> replacement; truth retrofit says the parent is defective (stored_wrong,
    reward_format, ambiguous, defect) -> fix; pass rate > 0.75 -> harder; < 0.25 -> simpler (genuine failures);
    otherwise none. Slots are taken in VARIANT_ORDER until "max"."""
    cfg = plan["assignments"]
    recs = [json.loads(l) for l in open(repo / cfg["path"], encoding="utf-8") if l.strip()]
    triage: dict[str, dict[str, Any]] = {}
    if cfg.get("truth_report") and (repo / cfg["truth_report"]).is_file():
        for l in open(repo / cfg["truth_report"], encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                triage[r["task_id"]] = r
    include = cfg.get("include", VARIANT_ORDER)
    picks: list[tuple[str, dict[str, Any]]] = []
    min_k = int(cfg.get("min_rollouts", 8))
    for rec in recs:
        tid, pr = rec["parent_task_id"], rec.get("pass_rate_at_8")
        tv = (triage.get(tid, {}).get("triage") or {}).get("verdict")
        if not rec.get("screened", (rec.get("n_rollouts") or 0) >= min_k) and not rec.get("gate_reasons"):
            continue                                  # not fully screened yet (incremental planning)
        if rec.get("gate_reasons"):
            v = "replacement"
        elif tv in ("stored_wrong", "reward_format", "ambiguous", "defect"):
            v = "fix"
        elif pr is None:
            continue
        elif pr > 0.75:
            v = "harder"
        elif pr < 0.25:
            v = "simpler"
        else:
            continue
        if v in include:
            picks.append((v, rec))
    # Incremental planning: (parent, variant) pairs already planned keep their slot numbers and their place under
    # the cap; new screened seeds take the next numbers.
    prior: dict[tuple[str, str], int] = {}
    keep = cfg.get("keep_numbers_from")
    if keep and (repo / keep).is_file():
        for old in json.loads((repo / keep).read_text(encoding="utf-8")).get("slots", []):
            if old.get("kind") == "assignment":
                prior[(old["parent_task_id"], old["variant_type"])] = int(old["slot"])
    kept = [vr for vr in picks if (vr[1]["parent_task_id"], vr[0]) in prior]
    # Within each variant type, take new parents round-robin across sites (sorted by pass rate distance from the
    # band inside a site), so the cap does not cut whole sites (2026-10-10: an id-sorted cut gave reddit 25, admin 6).
    def round_robin(items: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
        by_site: dict[str, list] = collections.defaultdict(list)
        for vr in sorted(items, key=lambda vr: vr[1]["parent_task_id"]):
            by_site[APP_SITE.get((vr[1].get("sites") or [""])[0], "z")].append(vr)
        out, queues = [], [by_site[k] for k in sorted(by_site)]
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop(0))
        return out
    fresh_all = [vr for vr in picks if (vr[1]["parent_task_id"], vr[0]) not in prior]
    fresh = [vr for v in VARIANT_ORDER for vr in round_robin([x for x in fresh_all if x[0] == v])]
    picks = kept + fresh[: max(0, int(cfg.get("max", 60)) - len(kept))]
    next_n = max(prior.values(), default=0)
    numbered = []
    for v, rec in sorted(picks, key=lambda vr: (APP_SITE.get((vr[1].get("sites") or [""])[0], "z"),
                                                VARIANT_ORDER.index(vr[0]), vr[1]["parent_task_id"])):
        key = (rec["parent_task_id"], v)
        if key not in prior:
            next_n += 1
            prior[key] = next_n
        numbered.append((prior[key], v, rec))
    slots: list[dict[str, Any]] = []
    for n, v, rec in sorted(numbered, key=lambda x: x[0]):
        site = APP_SITE.get((rec.get("sites") or [""])[0], "shopping")
        meta = rec.get("parent_meta") or {}
        tr = triage.get(rec["parent_task_id"], {})
        slots.append({
            "slot": n, "site": site, "kind": "assignment", "style": "terse", "short": False,
            "variant_type": v, "parent_task_id": rec["parent_task_id"], "parent_intent": rec.get("intent"),
            "parent_family": meta.get("div4_family_slug"), "parent_pass_rate": rec.get("pass_rate_at_8"),
            "parent_mean_reward": rec.get("mean_reward"), "failure_summary": rec.get("most_common_failure"),
            "failure_labels": rec.get("failure_labels"), "failed_rollouts": rec.get("failed_rollouts"),
            "passing_rollout": rec.get("passing_rollout"), "gate_reasons": rec.get("gate_reasons"),
            "parent_truth": (meta.get("answer") or {}).get("truth"), "parent_code_truth": tr.get("answer"),
            "parent_triage": tr.get("triage"), "parent_trap": meta.get("trap"),
            "parent_unachievable": meta.get("unachievable"),
            "answer_scored": True, "intent_words": plan.get("intent_words", [10, 25]), "no_urls": True,
            "golden_actions": plan.get("golden_actions", [8, 25]), "code_truth": True,
            "avoid_entities": corpus.overused_entities(site), "avoid_families": [],
        })
    return slots


VARIANT_TEXT = {
    "harder": ("The policy solves the parent too often (pass rate above 0.75). Write a HARDER SIBLING on the same skill: "
               "more items to collect, one extra constraint, the target on a deeper page, or two fields instead of one. "
               "Change one or two of these, not all; it must stay solvable in about 25 actions."),
    "simpler": ("The policy almost always fails the parent (pass rate below 0.25) for a genuine reason (the failed "
                "rollouts show it). Write a SIMPLER SIBLING that isolates the missing sub-skill: one page less, one "
                "field less, or one condition less, with an instruction that names the sub-goal plainly."),
    "replacement": ("The parent was rejected by the cluster's gate (reasons below). Write a REPLACEMENT task of the same "
                    "failure family that avoids the rejection reason (for contamination: share no 13-word span with any "
                    "WebArena intent and do not paraphrase one; for a give-up answer: the truth must be a real value)."),
    "fix": ("The parent is defective (the truth audit below explains how). Write the FIXED task: the same errand with "
            "the defect removed (a correct code-computed answer, an unambiguous instruction, or a reward that accepts "
            "the right answer). Do not generate around the defect."),
}


def assignment_brief(slot: dict[str, Any]) -> list[str]:
    """The assignment as the author reads it: parent, pass rate, failures and the variant to write."""
    pr = slot.get("parent_pass_rate")
    lines = [f"      ASSIGNMENT: {slot['variant_type'].upper()} variant of parent {slot['parent_task_id']}",
             f"      parent instruction: {json.dumps(slot.get('parent_intent'))}",
             f"      parent answer (stored): {json.dumps(slot.get('parent_truth'), default=str)[:300]}"
             + (f"; code-computed: {json.dumps(slot.get('parent_code_truth'), default=str)[:300]}"
                if slot.get("parent_code_truth") is not None else ""),
             f"      policy (step_239, T=0.6, 8 rollouts): pass rate {pr if pr is None else round(pr, 3)}, "
             f"failures {json.dumps(slot.get('failure_labels') or {})}",
             f"      most common failure: {slot.get('failure_summary') or '-'}"]
    for r in (slot.get("failed_rollouts") or [])[:3]:
        lines.append(f"      failed rollout {r.get('rollout')}: reward {r.get('reward')}; {r.get('label')}; "
                     f"last actions: {' | '.join((r.get('last_actions') or [])[-3:])[:400]}")
    if slot.get("passing_rollout"):
        p = slot["passing_rollout"]
        lines.append(f"      a passing rollout answered {json.dumps((p.get('answer') or '')[:200])} in {p.get('steps')} actions")
    if slot.get("gate_reasons"):
        lines.append(f"      gate rejection reasons: {'; '.join(map(str, slot['gate_reasons']))[:500]}")
    if slot.get("parent_triage"):
        t = slot["parent_triage"]
        lines.append(f"      truth audit of the parent: {t.get('verdict')}: {str(t.get('evidence'))[:500]}"
                     + (f" | proposed fix: {str(t.get('notes'))[:300]}" if t.get("notes") else ""))
    lines.append(f"      what to write: {VARIANT_TEXT[slot['variant_type']]}")
    if slot.get("parent_unachievable"):
        lines.append("      the parent is UNACHIEVABLE (its answer is N/A and the policy invents one): keep the truth N/A "
                     "(record metadata.unachievable = true, metadata.trap = \"unachievable\"; compute() returns None) and "
                     + ("make the absence easy to establish: one search or one page shows that nothing matches"
                        if slot["variant_type"] == "simpler" else
                        "make the absence harder to establish: a near-miss record exists under a similar name or on a "
                        "later page, and only a careful check shows it does not match"))
    lines.append(f"      record metadata.parent_task_id = \"{slot['parent_task_id']}\", metadata.variant_type = "
                 f"\"{slot['variant_type']}\", metadata.failure_summary (one line: the parent failure this variant "
                 f"targets), metadata.ground_truth_method = \"code\"")
    return lines


_POOL_CACHE: dict[str, list[dict[str, Any]]] = {}


def map_city_places(repo: Path, plan: dict[str, Any], n: int, rng: random.Random) -> tuple[str, list[str]]:
    """div6-map: slot n's city (round-robin over plan["map_cities"]) and 8 of its harvested places (round-tripped
    query strings, mixed categories), never a banned place."""
    path = str(repo / plan["map_pool"])
    if path not in _POOL_CACHE:
        _POOL_CACHE[path] = json.load(open(path, encoding="utf-8"))
    banned = re.compile(plan.get("banned_terms") or r"$^", re.I)
    cities = plan["map_cities"]
    city = cities[(n - 1) % len(cities)]
    places = [p for p in _POOL_CACHE[path] if p.get("city") == city and not banned.search(p.get("query", "") + " "
                                                                                          + p.get("display_name", ""))]
    by_cat: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for p in places:
        by_cat[p.get("category", "")].append(p)
    for v in by_cat.values():
        rng.shuffle(v)
    picks, cats = [], sorted(by_cat)
    rng.shuffle(cats)
    while len(picks) < 8 and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(picks) < 8:
                p = by_cat[c].pop()
                picks.append(f"{p['query']} [{p.get('category')}]")
    return city, picks


_WA_GRAMS: dict[int, set[tuple[str, ...]]] = {}


def webarena_ngrams(repo: Path, n: int) -> set[tuple[str, ...]]:
    if n not in _WA_GRAMS:
        grams: set[tuple[str, ...]] = set()
        for line in open(repo / "webarena_benchmarks" / "webarena.jsonl", encoding="utf-8"):
            r = json.loads(line)
            t = re.findall(r"[a-z0-9]+", str(r.get("ques") or r.get("intent") or "").lower())
            grams |= {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
        _WA_GRAMS[n] = grams
    return _WA_GRAMS[n]


URL_IN_TEXT = re.compile(r"https?://|localhost|127\.0\.0\.1|__CUA_GYM_", re.I)


def slot_brief(slot: dict[str, Any]) -> str:
    """The novelty contract as the author reads it."""
    lines = [f"  - slot {slot['slot']} ({slot['site']}, {slot['style']}"
             + (", short: <= 20 words" if slot.get("short") else "") + ")"]
    if slot.get("kind") == "assignment":
        lines += assignment_brief(slot)
        lines.append("      ANSWER-SCORED: the agent ends with terminate(answer=...); reward.py reads evidence[\"agent_answer\"] "
                     "(nemo_reward.py from CUA_GYM_AGENT_ANSWER) and compares it with the expected answer by exact match "
                     "or set equality after normalising case, whitespace, separators and order; lists get partial credit "
                     "|answer & truth| / |truth| when the answer has no false item, else 0; \"N/A\" and \"\" score 0 "
                     "(an unachievable task: \"N/A\" scores 1.0). metadata.answer = {\"kind\": ..., \"truth\": ...}")
    if slot.get("code_truth") and slot.get("mutation"):
        lines.append("      CODE-COMPUTED TARGET VALUES (required, state-change task): write truth.py with compute(load) "
                     "returning a dict of the exact post-condition values the reward checks (the @handle, the verbatim "
                     "text, the exact line, the due date, the star rating, the tracking number...), computed from the mock "
                     "data, and render(x) returning json.dumps(x). reward.py must define a module-level EXPECTED equal to "
                     "that dict and check the back-end current_state (/go?sid=) against it. The gate compares EXPECTED "
                     "with compute(); verification proves the golden replay reaches it. Also list in "
                     "metadata.near_misses the realistic wrong end states (display name instead of handle, paraphrase, "
                     "the 3rd star instead of the 4th, unsaved edit) that the reward must score below 1.0")
    elif slot.get("code_truth") and slot.get("stateless"):
        lines.append("      CODE-COMPUTED TRUTH (required): write truth.py in the bundle with compute(load, api) and "
                     "render(answer): api.geocode / api.route / api.object / api.fmt_time / api.fmt_dist are the site's own "
                     "Nominatim and OSRM calls (scripts/pipeline/truth.py). Run `python3 scripts/pipeline/truth.py run "
                     "<bundle>/truth.py` and inline exactly that answer in reward.py, nemo_reward.py and "
                     "metadata.answer.truth. The gate re-runs truth.py, scores render(answer) with reward.py AND "
                     "nemo_reward.py (both must pay 1.0), scores \"N/A\" and \"\" (< 0.5) and an answer outside the "
                     "tolerance (must be 0)")
    elif slot.get("code_truth"):
        lines.append("      CODE-COMPUTED TRUTH (required): write truth.py in the bundle with compute(load) and render(answer) "
                     "(contract: scripts/pipeline/truth.py; data only through load(app_dir, relpath)). Run "
                     "`python3 scripts/pipeline/truth.py run <bundle>/truth.py`, then inline exactly that answer as the "
                     "reward's expected value and as metadata.answer.truth. Never write the expected answer by hand: the "
                     "gate re-runs truth.py, scores render(answer) with your reward (must be 1.0) and scores \"N/A\" and "
                     "\"\" (must be below 0.5)")
    if slot.get("kind") == "family":
        lines.append(f"      failure class {slot['div4_family']}: {slot['family_brief']}")
        if slot.get("example") and slot.get("source_failures") and not slot.get("ladder_id"):
            lines.append(f"      sub-type idea, only if it fits the TARGET ERROR below (otherwise invent an errand that "
                         f"does): {slot['example']}")
        elif slot.get("example"):
            lines.append(f"      sub-type to build: {slot['example']} — choose your own entities, predicate and numbers; "
                         f"the slot's example is a kind of errand, not a text to copy")
        if slot.get("answer_scored") and slot.get("div10"):
            lines.append("      ANSWER-SCORED, GRADED LIKE WEBARENA (div10): the agent ends with terminate(answer=...). reward.py "
                         "(evidence[\"agent_answer\"]) and nemo_reward.py (CUA_GYM_AGENT_ANSWER) both score it with the "
                         "harness's _string_match: copy the scoring code of run_info/div10__answer_reward_reference.py "
                         "verbatim (rules first, then the GPT-4.1 judge through the literal __WEBARENA_JUDGE_URL__ token, "
                         "rules again when the judge is unavailable; the judge() wrapper scores a give-up with no reference "
                         "word 0). Pick the evaluator kind that fits the answer: exact_match for a value that must be copied "
                         "verbatim (an ID, a code), must_include for a set of names or numbers, otherwise fuzzy_match with ONE "
                         "reference string (never a list). Build SPEC.reference_answers from truth.py's answer. Put the habit "
                         "move's answers (the first candidate, the unqualified match, the page-1 count) in must_exclude and in "
                         "metadata.wrong_move_answers: each must score 0. Binary: no partial credit. \"N/A\" and \"\" score "
                         "0 unless the slot is unachievable. "
                         + ("Map: fill _KEY_NUMBERS with the truth's key durations / distances / clock time (outside "
                            "parentheses); the gate is the tolerance (+-1 min, +-0.15 km). " if slot.get("stateless") else "")
                         + "metadata.answer = {\"kind\": ..., \"truth\": ..., \"evaluator\": {...the SPEC "
                         "reference_answers...}}")
        elif slot.get("answer_scored") and slot.get("stateless"):
            lines.append("      ANSWER-SCORED (map): the agent ends with terminate(answer=...); reward.py reads "
                         "evidence[\"agent_answer\"], nemo_reward.py reads CUA_GYM_AGENT_ANSWER; both compare it with the "
                         "code-computed truth within the tolerance below (no LLM, no similarity score). Lists: names "
                         "normalised and order-free; "
                         + ("binary (div8): 1.0 only for the full set, else 0. " if slot.get("template") else
                            "partial credit |answer & truth| / |truth| only when no wrong place is named, else 0. ")
                         + "\"N/A\", \"\" and an answer outside tolerance score 0. metadata.answer = "
                         "{\"kind\": ..., \"truth\": ..., \"tolerance\": ...}")
        elif slot.get("answer_scored") and slot.get("webarena_grading"):
            lines.append("      ANSWER-SCORED, GRADED LIKE WEBARENA (div9): the agent ends with terminate(answer=...). reward.py "
                         "(evidence[\"agent_answer\"]) and nemo_reward.py (CUA_GYM_AGENT_ANSWER) both score it with the "
                         "harness's _string_match: copy the scoring functions of run_info/div9__answer_reward_reference.py "
                         "verbatim (exact_match / must_include / fuzzy_match, rules first, then the GPT-4.1 judge through the "
                         "literal __WEBARENA_JUDGE_URL__ token, rules again when the judge is unavailable). Use the same "
                         f"evaluator kind as the source ({slot.get('source_eval') or 'see the source'}); build "
                         "SPEC.reference_answers from truth.py's answer. A verbose answer that contains the right items "
                         "passes, as in the real evaluation. Put the recorded wrong move's answers (the first candidate, the "
                         "look-alike) in must_exclude so they score 0, and if extra candidates are excluded say the answer "
                         "form in the request ('just the name'). \"N/A\" and \"\" score 0. metadata.answer = {\"kind\": ..., "
                         "\"truth\": ..., \"evaluator\": {...the SPEC reference_answers...}}")
        elif slot.get("answer_scored"):
            lines.append("      ANSWER-SCORED: no writeback. The agent ends with terminate(answer=...); the reward reads "
                         "CUA_GYM_AGENT_ANSWER (see run_info/b4__TASK4.md) and compares it, after normalising case, "
                         "whitespace, separators and order, against the truth you compute from the mock's seed data and "
                         "inline as a fixture. Exact match or set equality only, no fuzzy or semantic matching. The "
                         "instruction states the answer form in a few words (\"comma-separated\", \"as a number\"). "
                         + ("Binary reward (div8): 1.0 only for the exact answer (set-equal for a list), else 0, so a "
                            "partial list from the recorded wrong move scores 0. " if slot.get("template") else
                            "Partial credit for a list: |answer & truth| / |truth| when the answer has no false item, else 0. ")
                         + "An empty answer, \"N/A\" or the untouched initial state scores 0. metadata.skills lists only retrieval "
                         "skills (R1-R10); metadata.answer = {\"kind\": \"list|number|pairs|range|text\", \"truth\": ...}; "
                         "golden_replay.py returns the answer string")
        if slot.get("unachievable"):
            lines.append("      UNACHIEVABLE slot (overrides the N/A rule above): ask for something that genuinely does not "
                         "exist on the site (check the seed data: no record matches, not even under another spelling or "
                         "on a later page). The correct answer is \"N/A\" (accept N/A, none, not found, does not exist, "
                         "case-insensitive); every other answer, and an empty one, scores 0. Record "
                         "metadata.unachievable = true and metadata.trap = \"unachievable\"")
        elif slot["div4_family"].startswith("f4"):
            lines.append("      the target EXISTS but a first glance misses it. Record metadata.trap as one of page_2_or_3, "
                         "behind_filter, spelling_variant, sibling_category, and make that trap real in the seed data. "
                         "Answering N/A or giving up scores 0")
    if slot.get("city"):
        lines.append(f"      MAP place set for this slot: {slot['city']}. Build the task from these harvested places (their "
                     f"query strings round-trip to a single Nominatim hit; geocode them with api.geocode and record the "
                     f"hit count): " + "; ".join(slot.get("entity_candidates") or [])
                     + (". div8: add other places in this city when the template needs them (several of one type near "
                        "a landmark, a same-name decoy in another town); each must round-trip to one hit, recorded in "
                        "metadata.places" if slot.get("template") else ""))
    if slot.get("tolerance"):
        lines.append(f"      answer tolerance the reward must apply: {slot['tolerance']}")
    if slot.get("source_task"):
        lines.append(f"      SIBLINGS OF A REAL FAILURE: this lane writes twins of WebArena task {slot['source_task']} "
                     f"({slot.get('source_origin')}). A twin keeps the source's site, page flow, UI controls, evaluator "
                     f"type and the decision the model got wrong, but uses ANOTHER entity and other values (never the "
                     f"source's own record, answer or wording; the 13-gram gate checks the wording). "
                     + "; ".join(f"rung {k}: {v}" for k, v in sorted((slot.get("sibling_kinds") or {}).items())))
    if slot.get("template"):
        lines.append(f"      TEMPLATE {slot['template']}. FIXED STEP the task must force (it lives only in the reward and the "
                     f"state checks, never in the instruction; every rung needs it): "
                     + " > ".join(slot.get("fixed_steps") or []))
        lines.append(f"      RECORDED WRONG MOVE ({'our policy E26 step_319' if slot.get('div10') else 'what step_239 did'}), "
                     f"which must score 0: {slot.get('wrong_move')}")
        if slot.get("step_terms"):
            lines.append(f"      the gate fails an instruction that names the step (case-insensitive regex): {slot['step_terms']}")
        lines.append("      record metadata.template, metadata.fixed_steps (list of strings), metadata.wrong_move, and "
                     + ("metadata.wrong_move_answers: 1 to 3 answers the recorded wrong move produces (the first or most "
                        "visible candidate, the page-1-only count, the look-alike's value); the gate scores each with "
                        "reward.py and requires 0" if slot.get("answer_scored") else
                        "metadata.near_misses with the recorded wrong move's end state FIRST: the reward must score that "
                        "state 0 (gate every component on the fixed step's post-condition) and the other near misses "
                        "below 1.0"))
    if d10 := slot.get("div10"):
        lines.append(f"      DIV10 SKILL {d10['skill']} ({d10['skill_name']}). Habit of OUR policy to break: {d10['habit']}. "
                     f"What Qwen does instead (the move the task must reward): {d10['qwen_move']}. Build a task that trains "
                     f"this skill on this lane's site ({slot['site']}); do not rebuild the source task.")
        rel = d10.get("relation")
        if rel == "near_sibling":
            ns = next((x for x in d10["sources"] if x["id"] == d10.get("near_source")), d10["sources"][0])
            lines.append(f"      LANE KIND (addendum A1): NEAR-SIBLING of webarena-{ns['id']} (TRAIN template "
                         f"{ns.get('template_id')}). Keep its page path and its kind of goal (right move: "
                         f"{ns.get('right_move')}); change the entity, the values, the wording and the answer (div9 twin "
                         f"design); never its record, answer or wording. metadata.relation = 'near_sibling', "
                         f"metadata.same_page_path_as_source = true, metadata.split = 'TRAIN', source_task_ids include "
                         f"webarena-{ns['id']}")
        elif rel == "skill" and d10.get("split") == "TEST":
            notes = "; ".join(f"template {x.get('template_id')} ({x.get('test_kind')}): forbidden page path "
                              f"'{x.get('page_path_note') or 'the source route'}'"
                              + (f", also the path of TRAIN templates {x['train_neighbours']}" if x.get("train_neighbours") else "")
                              for x in d10["sources"])
            lines.append(f"      LANE KIND (addendum A1 S3): SKILL TASKS ON TEST TEMPLATES. Build the habit on ANOTHER page "
                         f"path. {notes}. Use nothing from any task of those templates (entity, values, answer, page + goal); "
                         f"read the dossiers to learn the habit, never rebuild their scene. metadata.relation = 'skill', "
                         f"metadata.same_page_path_as_source = false, metadata.split = 'TEST'")
        elif rel == "skill":
            paths = "; ".join(f"webarena-{x['id']}: {x.get('right_move')}" for x in d10["sources"])[:900]
            lines.append(f"      LANE KIND (addendum A1 S5): SKILL TASKS on TRAIN sources. Build the habit on ANOTHER page "
                         f"path than each source's route ({paths}). metadata.relation = 'skill', "
                         f"metadata.same_page_path_as_source = false, metadata.split = 'TRAIN'")
        for x in d10["sources"]:
            lines.append(f"      SOURCE DOSSIER webarena-{x['id']} ({'+'.join(x['sites'])}, {x['trouble_class']}, priority "
                         f"{x['priority']}): READ {x['dossier']} in full first (trouble, wrong move vs right move, "
                         f"diagnostic signals, near misses, synthesis hints, do_not_copy, 'How Qwen did it'). Then open "
                         f"the failing E26 read {x['trajectories'].get('e26_s319')} around step {x.get('first_wrong_step')} "
                         f"and Qwen's read {x['trajectories'].get('qwen_r1')}; screenshots: "
                         + ", ".join(v for v in x["screens"].values())[:400]
                         + f". Do NOT use (do_not_copy): {'; '.join(x.get('do_not_copy') or [])[:400]}")
        dist = d10.get("rung_distance") or {}
        lines.append(f"      RUNGS by distance to the warning sign (T3); every rung keeps the trap, only the distance changes: "
                     + "; ".join(f"rung {k}: signal_distance {v} ({d10['distance_text'][str(v)]})" for k, v in sorted(dist.items()))
                     + (f". CONTRAST PAIR (K1): rungs {' and '.join(d10['contrast_rungs'])} are two tasks on the same page "
                        f"whose answers differ only because of one qualifier; give both metadata.contrast_pair_id = "
                        f"{d10['contrast_id']!r}" if d10.get("contrast_rungs") else ""))
        lines.append("      record in task.json metadata (hand-off section 7): source_task_ids, skill, trouble_class, trouble, "
                     "habit_move, trap_spec {wrong_move_analogue, right_move, diagnostic_signal, near_misses}, "
                     "discoverable_route (the right path with clicks, typing, keys and scrolls only), efficient_route "
                     "(e.g. a filter URL, or null), signal_distance (int), contrast_pair_id (or null), start_state_noise "
                     "(list, may be empty), mock_parity_check (how you confirmed the right move works the same on the mock "
                     "as on the real site), diversity_axes (entity kind, page type, cue, wording)")
    if slot.get("ladder_id"):
        spec = slot.get("rung_spec") or {}
        lines.append(f"      LADDER {slot['ladder_id']}, rung {slot['rung']} of {slot['ladder_rungs']} (1 = easiest). This "
                     f"lane writes all rungs of the ladder: one question, one shared truth program (truth.py identical in "
                     f"every rung except a RUNG constant that sets the scope), one bundle per rung with its own intent. "
                     + "; ".join(f"rung {k}: {v}" for k, v in sorted(spec.items())) + ". Record metadata.ladder_id, "
                     f"metadata.rung, metadata.source_failure_ids")
    for f in slot.get("source_failures") or []:
        lines.append(f"      TARGET ERROR, source WebArena failure {f.get('task_id')} ({f.get('site')}): "
                     f"{json.dumps(f.get('intent'))[:220]}"
                     f" — policy failure: {f.get('subpattern')}"
                     + (f" after {f.get('steps')} steps" if f.get("steps") else "")
                     + (f"; it believed: {str(f.get('agent_believed'))[:320]}" if f.get("agent_believed") else "")
                     + (f"; truth: {str(f.get('truth'))[:320]}" if f.get("truth") else "")
                     + (f"; analyst's data fix: {str(f.get('data_fix'))[:300]}" if f.get("data_fix") else "")
                     + ". Make that losing move fail and the truthful move pass; do NOT reuse the WebArena intent or "
                       "its entities (13-gram gate)")
    if slot.get("intent_words"):
        lo, hi = slot["intent_words"]
        lines.append(f"      instruction: {lo}-{hi} words, WebArena style: one plain request with an implicit target; name "
                     f"every required field once and never restate it; no site URL")
    if slot.get("pair"):
        if slot.get("hard"):
            lines.append(f"      skill chain (HARD): {' -> '.join(slot['pair'])} — each step depends on the previous one; "
                         f"metadata.difficulty \"hard\" with metadata.hard_criteria naming >= 2 of multi_entity, "
                         f"derived_target, ordering_dependency, cross_page, conditional_branch")
        elif slot.get("kind") == "open":
            lines.append(f"      skill pair: {' -> '.join(slot['pair'])} — a pairing the corpus is thin on (see "
                         f"docs/SKILL_TAXONOMY.md); invent a task shape no prior task has, using exactly these skills")
        else:
            lines.append(f"      skill pair: {' -> '.join(slot['pair'])}")
    if slot.get("kind") == "official_seed":
        lines.append(f"      inspiration ({slot['seed_id']}): \"{slot['seed_intent']}\" — build a task with this "
                     f"kind of everyday intent; resemble it, never copy it, use different entities and values")
    elif slot.get("kind") == "field":
        lines.append(f"      target state: {slot['target_key']} -> {', '.join(slot['target_fields'])} — no prior reward "
                     f"has ever checked these; your reward must check at least one of them")
        if slot.get("alt_fields"):
            lines.append(f"      if none of those has a UI control that writes it, use another never-checked field "
                         f"on this site instead: {', '.join(slot['alt_fields'])}")
    if slot.get("require_lookup"):
        lines.append("      the agent must FIND its target: never name the record the task acts on (no order number, "
                     "username, project path or title of it in the instruction); describe it by a property the agent "
                     "has to look up. Declare metadata.lookup = {\"what\": \"<how the target is identified>\", "
                     "\"value\": \"<the target's id or name>\"}; the gate fails the task if the value is in the text. "
                     "Keep the lookup bounded: one search, filter or sort plus at most 2-3 page visits must reach it "
                     "(about 20 agent steps). Never make the agent scan an unbounded list (every comment by a user, "
                     "all posts in a forum, every page of orders); div2's reddit tasks that did died at the step cap")
    if slot.get("max_ui_actions"):
        lines.append(f"      LENGTH LIMIT: the golden path of every rung is at most {slot['max_ui_actions']} UI actions (clicks, "
                     f"fills, key presses, selects, page loads), because an exploring policy needs about twice the "
                     f"shortest path and the cap is 50 steps. Record metadata.shortest_ui_actions (integer). The post "
                     f"gate counts the actions your verified golden_replay.py executes and rejects a rung above the "
                     f"limit. Rungs get harder by the decision they test (closer decoys, a subtler condition), never "
                     f"by more pages, more candidates to measure or more edits")
    if slot.get("golden_actions"):
        lo, hi = slot["golden_actions"]
        cap = 100 if hi > 50 else 50
        lines.append(f"      length: the golden path should need {lo}-{hi} UI actions" + (" (not 3-7)" if lo >= 8 else "")
                     + f"; the agent has {cap} steps"
                     + (" in this batch's screen and training (long-horizon family)" if cap > 50 else ""))
    if slot.get("entity_candidates") and not slot.get("city"):
        lines.append(f"      unused entities to prefer (or any other entity no prior task names): "
                     + "; ".join(slot["entity_candidates"]))
    if slot.get("avoid_entities"):
        lines.append(f"      do not name: " + "; ".join(slot["avoid_entities"]))
    if slot.get("avoid_families"):
        lines.append(f"      over-used task shapes to avoid: " + ", ".join(slot["avoid_families"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- per-bundle check

_A1: dict[str, Any] = {}


def _a1_tables(root: Path, ref: str, extra: list[str]) -> dict[str, Any]:
    """The addendum-A1 gate tables (check_split_gate.py's own functions), loaded once per process."""
    key = f"{root}|{ref}|{extra}"
    if key not in _A1:
        import importlib.util
        spec = importlib.util.spec_from_file_location("a1gate", root / Path(ref).parent / "check_split_gate.py")
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        wa = [json.loads(l) for l in open(root / ref, encoding="utf-8")]
        tmpl = {r["wa_task_id"]: r["template_id"] for r in wa}
        split_t = {r["template_id"]: r["split"] for r in wa}
        W = [(r["wa_task_id"], g.shingles(t)) for r in wa for t in r["intents"]]
        G13 = set()
        for r in wa:
            for t in r["intents"]:
                G13 |= g.grams(t)
        for f in extra:
            if (root / f).is_file():
                for l in open(root / f, encoding="utf-8"):
                    if l.strip():
                        G13 |= g.grams(g.row_intent(json.loads(l)))
        refs = {r["wa_task_id"]: {g.ans_norm(x) for x in r["refs"]} for r in wa}
        test_sets = {frozenset(g.ans_norm(x) for x in r["refs"]): r["wa_task_id"]
                     for r in wa if r["split"] == "TEST" and r["refs"]}
        _A1[key] = {"g": g, "tmpl": tmpl, "split_t": split_t, "W": W, "G13": G13, "refs": refs, "test_sets": test_sets}
    return _A1[key]


def a1_checks(bundle: Path, d10: dict[str, Any], meta: dict[str, Any], rec: dict[str, Any], ans: dict[str, Any],
              ids: set[str]) -> list[str]:
    """div10 addendum A1: relation / split / template fields, and the copy, TEST-text and answer checks of
    check_split_gate.py at authoring time (its WARN band fails here, so the delivery has no WARN rows to explain)."""
    out: list[str] = []
    rel, sp = d10["relation"], d10["split"]
    if meta.get("relation") != rel:
        out.append(f"A1: metadata.relation must be {rel!r} (this lane's kind)")
    if meta.get("split") != sp:
        out.append(f"A1: metadata.split must be {sp!r}")
    if meta.get("same_page_path_as_source") is not (rel == "near_sibling"):
        out.append(f"A1: metadata.same_page_path_as_source must be {str(rel == 'near_sibling').lower()}")
    if not meta.get("page_path"):
        out.append("A1: record metadata.page_path (the pages and controls of the shortest right route, short text)")
    if rel == "near_sibling" and str(d10.get("near_source")) not in ids:
        out.append(f"A1: a near-sibling names its source: source_task_ids must include webarena-{d10.get('near_source')}")
    root = next((p for p in Path(bundle).resolve().parents if (p / d10["a1_ref"]).is_file()), None)
    if root is None:
        return out + ["A1: gate tables not found (output/div10/a1); tell the operator"]
    t = _a1_tables(root, d10["a1_ref"], d10.get("a1_extra") or [])
    g = t["g"]
    srcs = [int(i) for i in ids if i.isdigit() and int(i) in t["tmpl"]]
    frozen = sorted({int(t["tmpl"][s]) for s in srcs})
    try:
        declared = sorted(int(x) for x in meta.get("wa_template_ids") or [])
    except (TypeError, ValueError):
        declared = None
    if declared != frozen:
        out.append(f"A1: metadata.wa_template_ids must be the frozen templates of your source_task_ids: {frozen}")
    sh = g.shingles(rec["intent"])
    best = max(((g.jac(sh, s), w) for w, s in t["W"]), default=(0.0, None))
    if best[0] >= 0.3:
        out.append(f"A1 G1: the instruction is too close to WebArena task {best[1]} (word 3-gram Jaccard {best[0]:.2f}, "
                   f"limit 0.3); reword it")
    bt = max(((g.jac(sh, s), w) for w, s in t["W"] if t["split_t"][t["tmpl"][w]] == "TEST"), default=(0.0, None))
    if bt[0] >= 0.3:
        out.append(f"A1 G2: the instruction is close to TEST task webarena-{bt[1]} (Jaccard {bt[0]:.2f}); TEST templates "
                   f"must stay untouched")
    if g.grams(rec["intent"]) & t["G13"]:
        out.append("A1 G1: the instruction shares a 13-word span with a WebArena intent or a div7-div9 intent")
    A = {g.ans_norm(x) for x in _answer_strings(ans)} - {""}
    for s_ in srcs:
        if A and A == t["refs"].get(s_):
            out.append(f"A1 G4: the answer equals the reference of source webarena-{s_}")
    if A and len(" ".join(sorted(A))) >= 6 and frozenset(A) in t["test_sets"]:
        out.append(f"A1 G4: the answer equals the reference of TEST task webarena-{t['test_sets'][frozenset(A)]}")
    return out


def _answer_strings(ans: dict[str, Any]) -> list[str]:
    ev = (ans or {}).get("evaluator") or {}
    out = []
    for k in ("exact_match", "must_include", "fuzzy_match"):
        v = ev.get(k) if isinstance(ev, dict) else None
        for x in (v if isinstance(v, list) else [v]):
            for y in (x if isinstance(x, list) else [x]):
                if isinstance(y, str) and y.strip():
                    out.append(y)
    if not out and isinstance((ans or {}).get("truth"), str):
        out.append(ans["truth"])
    return out


def check_bundle(bundle: Path, slot: dict[str, Any], corpus: Corpus, batch_peers: list[dict[str, Any]],
                 family_cap: int = 1) -> list[str]:
    """Diversity reasons this bundle fails its slot's contract (empty list = passes)."""
    rec = bundle_record(bundle)
    if rec is None:
        return ["diversity: bundle unreadable"]
    meta, site, problems = rec["meta"], rec["site"], []

    def earlier(p):        # a clash between two batch peers fails only the later slot
        try:
            return int((p.get("meta") or {}).get("slot")) < int(slot["slot"])
        except (TypeError, ValueError):
            return False
    batch_peers = [p for p in batch_peers if p and p["id"] != rec["id"] and earlier(p)
                   and not (slot.get("ladder_id") and (p.get("meta") or {}).get("ladder_id") == slot["ladder_id"])]
    # A sibling (harder / simpler / fix) is related to its parent on purpose: only the parent is exempt from the
    # novelty gates; every other prior task and batch peer still counts. A replacement gets no exemption.
    parent = slot.get("parent_task_id") if slot.get("variant_type") in ("harder", "simpler", "fix") else None
    skip = {rec["id"], parent} if parent else rec["id"]
    fam = meta.get("family")
    if not fam or not re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+){1,5}", str(fam)):
        problems.append("diversity: metadata.family must be a 2-6 token snake_case slug naming the task's shape")
    else:
        ft = family_tokens(fam)
        clash = [f for f in corpus.families(site) if jaccard(ft, family_tokens(f)) >= FAMILY_CLASH
                 and not (parent and f == slot.get("parent_family"))]
        if clash:
            problems.append(f"diversity: family '{fam}' repeats existing shape(s) {clash[:3]}; pick a genuinely new shape")
        same = [p["id"] for p in batch_peers if p["id"] != rec["id"] and p["site"] == site
                and jaccard(ft, family_tokens(p.get("family", ""))) >= FAMILY_CLASH]
        if len(same) >= family_cap:
            problems.append(f"diversity: family '{fam}' already used in this batch by {same[:2]}")
    if meta.get("slot") not in (slot["slot"], str(slot["slot"])):
        problems.append(f"diversity: metadata.slot must be {slot['slot']}")
    sim, who = corpus.max_sim(site, rec["tok"], skip_id=skip)
    if sim >= NEAR_DUP:
        problems.append(f"diversity: instruction is a near-duplicate of prior task {who} (token Jaccard {sim:.2f})")
    else:
        sim, who = corpus.max_sim(site, rec["shape"], skip_id=skip, key="shape")
        if sim >= SHAPE_DUP:
            problems.append(f"diversity: instruction repeats the template of prior task {who} with only names or values "
                            f"changed (template Jaccard {sim:.2f}); change what the agent has to find or do")
    for p in batch_peers:
        if p["id"] != rec["id"] and p["site"] == site and jaccard(rec["tok"], p["tok"]) >= NEAR_DUP:
            problems.append(f"diversity: instruction near-duplicates batch peer {p['id']}")
        elif p["id"] != rec["id"] and p["site"] == site and jaccard(rec["shape"], p["shape"]) >= SHAPE_DUP:
            problems.append(f"diversity: instruction repeats the template of batch peer {p['id']} with only names or "
                            f"values changed; change what the agent has to find or do")
    over = [e for e in slot.get("avoid_entities", []) if e.lower() in rec["intent"].lower()]
    if over:
        problems.append(f"diversity: instruction names over-used entit{'y' if len(over) == 1 else 'ies'} {over}")
    if slot.get("kind") == "field":
        targets = set(slot.get("target_fields", [])) | set(slot.get("alt_fields", []))
        if not targets & rec["lits"]:
            problems.append(f"diversity: reward checks none of the slot's target fields {sorted(slot.get('target_fields', []))}"
                            f" nor any listed never-checked alternative")
    if slot.get("kind") == "official_seed" and slot.get("seed_intent"):
        s = jaccard(rec["tok"], toks(slot["seed_intent"]))
        if s >= SEED_COPY:
            problems.append(f"diversity: instruction is too close to its official seed (Jaccard {s:.2f}); resemble, don't copy")
    if slot.get("require_lookup"):
        lookup = meta.get("lookup") if isinstance(meta.get("lookup"), dict) else None
        values = lookup.get("value") if lookup else None
        values = values if isinstance(values, list) else [values] if values not in (None, "") else []
        if not lookup or not lookup.get("what") or not values:
            problems.append("diversity: metadata.lookup {what, value} is required: declare what the agent must look up")
        else:
            text = rec["intent"].lower()
            handed = [str(v) for v in values if len(str(v).strip()) >= 2 and str(v).strip().lower() in text]
            if handed:
                problems.append(f"diversity: the instruction hands over the lookup target {handed}; describe it by "
                                f"a property the agent must look up instead")
    if slot.get("short") and rec["words"] > 24:
        problems.append(f"diversity: slot asks for a short instruction (<= 20 words), got {rec['words']}")
    if slot.get("intent_words"):
        lo, hi = slot["intent_words"]
        if not lo <= rec["words"] <= hi:
            problems.append(f"diversity: instruction must be {lo}-{hi} words (WebArena style), got {rec['words']}")
    if slot.get("banned_terms") and re.search(slot["banned_terms"], rec["intent"], re.I):
        problems.append(f"diversity: instruction names a banned place ({slot['banned_terms']}): WebArena's own map tasks "
                        f"live there; pick a place in another city")
    if slot.get("contamination_ngram"):
        n_ = int(slot["contamination_ngram"])
        t = re.findall(r"[a-z0-9]+", rec["intent"].lower())
        root = next((p for p in Path(rec["bundle"]).resolve().parents if (p / "webarena_benchmarks").is_dir()), None)
        if root and any(tuple(t[i:i + n_]) in webarena_ngrams(root, n_) for i in range(len(t) - n_ + 1)):
            problems.append(f"diversity: instruction shares a {n_}-word span with a WebArena intent (contamination)")
    if slot.get("no_urls") and URL_IN_TEXT.search(rec["intent"]):
        problems.append("diversity: instruction contains a URL or host; name the page or site in words instead")
    if slot.get("answer_scored"):
        ans = meta.get("answer") if isinstance(meta.get("answer"), dict) else None
        if not ans or "truth" not in ans or not ans.get("kind"):
            problems.append("diversity: answer-scored slot needs metadata.answer = {\"kind\": ..., \"truth\": ...}")
        if not meta.get("unachievable") and ans and str(ans.get("truth", "")).strip().lower() in ("n/a", "none", ""):
            problems.append("diversity: the truth of an achievable answer-scored task cannot be N/A or empty")
    if slot.get("ladder_id") and (meta.get("ladder_id") != slot["ladder_id"] or str(meta.get("rung")) != str(slot["rung"])):
        problems.append(f"diversity: ladder slot needs metadata.ladder_id = {slot['ladder_id']!r} and metadata.rung = "
                        f"{slot['rung']}")
    if slot.get("source_failures") and not meta.get("source_failure_ids"):
        problems.append("diversity: record metadata.source_failure_ids (the WebArena task ids this task answers)")
    if slot.get("step_terms") and (m := re.search(slot["step_terms"], rec["intent"], re.I)):
        problems.append(f"diversity: the instruction names the fixed step or a UI control ({m.group(0)!r}); state the goal "
                        f"only, the model must find the procedure itself")
    if slot.get("source_reference") and slot.get("answer_scored"):   # div9: another entity, never the test item's answer
        ans = meta.get("answer") if isinstance(meta.get("answer"), dict) else {}
        norm = lambda x: re.sub(r"[^a-z0-9]+", " ", json.dumps(x, ensure_ascii=False).lower()).strip()
        refs = [r for r in slot["source_reference"] if len(norm(r)) >= 3 and norm(r) not in ("n a", "")]
        if ans.get("truth") is not None and any(norm(ans["truth"]) == norm(r) for r in refs):
            problems.append(f"diversity: the twin's answer equals the WebArena source task's reference answer; build it "
                            f"on another entity so the test item is not reproduced")
    if cap := slot.get("max_ui_actions"):                      # div8 correction 2026-10-15
        v = meta.get("shortest_ui_actions")
        if not isinstance(v, int) or isinstance(v, bool):
            problems.append("diversity: record metadata.shortest_ui_actions (an integer: the UI actions of the golden "
                            "replay's path)")
        elif v > cap:
            problems.append(f"diversity: metadata.shortest_ui_actions = {v} exceeds {cap}; make the rung test a harder "
                            f"decision on a shorter path")
        if meta.get("needs_long_steps"):
            problems.append("diversity: needs_long_steps is set; a rung above the 50-step cap is not delivered")
    if slot.get("template"):
        if meta.get("template") != slot["template"]:
            problems.append(f"diversity: template slot needs metadata.template = {slot['template']!r}")
        if not isinstance(meta.get("fixed_steps"), list) or not meta.get("fixed_steps") or not meta.get("wrong_move"):
            problems.append("diversity: record metadata.fixed_steps (non-empty list) and metadata.wrong_move")
        if slot.get("answer_scored") and not meta.get("unachievable") and not (
                isinstance(meta.get("wrong_move_answers"), list) and meta["wrong_move_answers"]):
            problems.append("diversity: answer-scored template slot needs metadata.wrong_move_answers (1 to 3 answers the "
                            "recorded wrong move produces; each must score 0)")
        if slot.get("mutation") and not meta.get("near_misses"):
            problems.append("diversity: state-change template slot needs metadata.near_misses (the recorded wrong move "
                            "first)")
    if d10 := slot.get("div10"):                                    # div10 hand-off sections 3 and 7
        need = ("source_task_ids", "skill", "trouble_class", "trouble", "habit_move", "trap_spec", "discoverable_route",
                "signal_distance", "mock_parity_check", "diversity_axes")
        miss = [k for k in need if meta.get(k) in (None, "", [], {})]
        miss += [k for k in ("efficient_route", "contrast_pair_id", "start_state_noise") if k not in meta]
        if miss:
            problems.append(f"div10: record metadata {miss} (hand-off section 7)")
        ts = meta.get("trap_spec") if isinstance(meta.get("trap_spec"), dict) else {}
        if not all(ts.get(k) for k in ("wrong_move_analogue", "right_move", "diagnostic_signal", "near_misses")):
            problems.append("div10: metadata.trap_spec needs wrong_move_analogue, right_move, diagnostic_signal and "
                            "near_misses")
        if meta.get("skill") not in (None, d10["skill"]) and not meta.get("skill_change_reason"):
            problems.append(f"div10: metadata.skill is {meta.get('skill')!r} but the lane trains {d10['skill']}; keep it "
                            f"or explain in metadata.skill_change_reason")
        want = (d10.get("rung_distance") or {}).get(str(slot.get("rung")))
        sd = meta.get("signal_distance")
        if want is not None and not (isinstance(sd, int) and not isinstance(sd, bool) and
                                     (sd == want or (want >= 2 and sd >= 2))):
            problems.append(f"div10: rung {slot.get('rung')} must have metadata.signal_distance = {want} "
                            f"({d10['distance_text'][str(want)]}), got {sd!r}")
        if str(slot.get("rung")) in (d10.get("contrast_rungs") or []) and meta.get("contrast_pair_id") != d10["contrast_id"]:
            problems.append(f"div10: this rung is half of a K1 contrast pair: metadata.contrast_pair_id = "
                            f"{d10['contrast_id']!r}")
        ids = {str(x).split("-")[-1] for x in (meta.get("source_task_ids") or [])}
        lane_ids = [str(x["id"]) for x in d10["sources"]]
        if not ids & set(lane_ids):
            problems.append("div10: metadata.source_task_ids must include this lane's source ("
                            + ", ".join("webarena-" + i for i in lane_ids) + ")")
        normc = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()
        ans = meta.get("answer") if isinstance(meta.get("answer"), dict) else {}
        hay = " " + normc(rec["intent"]) + " " + normc(json.dumps(ans.get("truth"), ensure_ascii=False)) + " "
        hits = []
        for x in d10["sources"]:
            for item in x.get("do_not_copy") or []:
                for part in re.split(r"\s*(?:/|;|,| or | and )\s*", str(item)):
                    q = normc(part)
                    if len(q) >= 6 and not q.startswith(("the ", "a ", "any ", "same ")) and f" {q} " in hay:
                        hits.append(part.strip())
            for ref in x.get("reference") or []:
                q = normc(ref)
                if len(q) >= 4 and ans.get("truth") is not None and normc(json.dumps(ans["truth"], ensure_ascii=False)) == q:
                    hits.append(f"source reference {ref!r}")
        if hits:
            problems.append(f"div10 T9: the intent or answer reuses do_not_copy items of the source {sorted(set(hits))[:5]}; "
                            f"build the skill on other entities and values")
        if rel := d10.get("relation"):                                   # addendum A1 (template split)
            problems += a1_checks(bundle, d10, meta, rec, ans, ids)
    if slot.get("kind") == "assignment":
        if meta.get("parent_task_id") != slot["parent_task_id"] or meta.get("variant_type") != slot["variant_type"]:
            problems.append(f"diversity: assignment slot needs metadata.parent_task_id = {slot['parent_task_id']!r} and "
                            f"metadata.variant_type = {slot['variant_type']!r}")
        if not meta.get("failure_summary"):
            problems.append("diversity: assignment slot needs metadata.failure_summary")
    if slot.get("unachievable") and (meta.get("unachievable") is not True or meta.get("trap") != "unachievable"):
        problems.append("diversity: unachievable slot needs metadata.unachievable = true and metadata.trap = \"unachievable\"")
    if str(slot.get("div4_family", "")).startswith("f4") and not slot.get("unachievable") \
            and meta.get("trap") not in F4_TRAPS:
        problems.append(f"diversity: anti-give-up slot needs metadata.trap in {sorted(F4_TRAPS)}")
    return problems


F4_TRAPS = {"page_2_or_3", "behind_filter", "spelling_variant", "sibling_category"}


# --------------------------------------------------------------------------- report

def report(repo: Path, corpus: Corpus, batch: list[dict[str, Any]], bases: dict[str, str]) -> str:
    """Markdown: what this batch adds on each axis, against the corpus it was planned against."""
    official = official_intents(repo)
    lines = [f"# Diversity report — {len(batch)} tasks vs a corpus of {len(corpus.tasks)}", ""]
    if not batch:
        return "\n".join(lines + ["(no bundles)"])
    fams = collections.Counter((t["site"], t["family"]) for t in batch)
    new_fams = [f for f in fams if all(jaccard(family_tokens(f[1]), family_tokens(g)) < FAMILY_CLASH
                                       for g in corpus.families(f[0]))]
    H = -sum(c / len(batch) * math.log2(c / len(batch)) for c in fams.values())
    sims = [corpus.max_sim(t["site"], t["tok"], skip_id=t["id"])[0] for t in batch]
    prior_sims = []
    for site in SITES:
        pool = corpus.by_site[site][:300]
        prior_sims += [max((jaccard(t["tok"], u["tok"]) for u in corpus.by_site[site] if u["id"] != t["id"]),
                           default=0) for t in pool]
    over = collections.defaultdict(set, {s: set(corpus.overused_entities(s)) for s in set(SITES) | {t["site"] for t in batch}})
    over_share = sum(1 for t in batch if any(e.lower() in t["intent"].lower() for e in over[t["site"]])) / len(batch)
    prior_over = sum(1 for t in corpus.tasks if any(e.lower() in t["intent"].lower() for e in over[t["site"]])) \
        / max(len(corpus.tasks), 1)
    words = [len(t["intent"].split()) for t in batch]
    prior_words = [len(t["intent"].split()) for t in corpus.tasks]
    lines += [
        "| axis | this batch | corpus baseline |",
        "|---|---|---|",
        f"| task families | {len(fams)} distinct, **{len(new_fams)} new**, effective {2 ** H:.1f} | "
        f"{sum(len(corpus.families(s)) for s in SITES)} families |",
        f"| nearest-corpus similarity (mean max Jaccard) | **{statistics.mean(sims):.2f}** | "
        f"{statistics.mean(prior_sims):.2f} (corpus vs itself) |",
        f"| names an over-used entity | **{over_share:.0%}** | {prior_over:.0%} |",
        f"| instruction words (median / mean) | {statistics.median(words):.0f} / {statistics.mean(words):.1f} | "
        f"{statistics.median(prior_words):.0f} / {statistics.mean(prior_words):.1f} (official mean ~18) |",
    ]
    newly_fields, newly_official = [], []
    for site in SITES:
        mine = [t for t in batch if t["site"] == site]
        if not mine:
            continue
        before = collections.Counter(x for t in corpus.by_site[site] for x in t["lits"])
        fields = state_fields(live_state(site, bases[site]))
        subs = {s for v in fields.values() for s in v if len(s) > 2}
        newly_fields += [f"{site}:{f}" for f in sorted(subs) if before[f] == 0 and any(f in t["lits"] for t in mine)]
        for o in uncovered_official(corpus, official, site):
            if any(jaccard(o["tok"], t["tok"]) >= COVER_THRESHOLD for t in mine):
                newly_official.append(f"{site}: {o['intent'][:90]}")
    lines += [f"| state fields checked for the first time | **{len(newly_fields)}** | — |",
              f"| official intents newly covered (Jaccard >= {COVER_THRESHOLD}) | **{len(newly_official)}** | — |", ""]
    if newly_fields:
        lines += ["## First-time fields", ""] + [f"- {f}" for f in newly_fields] + [""]
    if newly_official:
        lines += ["## Newly covered official intents", ""] + [f"- {o}" for o in newly_official] + [""]
    skills = collections.Counter(s for t in batch for s in (t.get("meta", {}) or {}).get("skills", []))
    if skills:
        lines += ["## Skills", "", ", ".join(f"{k} {v}" for k, v in sorted(skills.items())), ""]
    lines += ["## Tasks", "", "| task | site | family | words | nearest corpus sim |", "|---|---|---|---|---|"]
    for t, s in sorted(zip(batch, sims), key=lambda ts: ts[0]["id"]):
        lines.append(f"| {t['id']} | {t['site']} | {t['family']} | {len(t['intent'].split())} | {s:.2f} |")
    return "\n".join(lines) + "\n"
