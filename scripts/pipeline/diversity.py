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

    def max_sim(self, site: str, tok: set[str], skip_id: str | None = None, key: str = "tok") -> tuple[float, str | None]:
        best, who = 0.0, None
        for t in self.by_site[site]:
            if t["id"] == skip_id:
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
    """
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


def slot_brief(slot: dict[str, Any]) -> str:
    """The novelty contract as the author reads it."""
    lines = [f"  - slot {slot['slot']} ({slot['site']}, {slot['style']}"
             + (", short: <= 20 words" if slot.get("short") else "") + ")"]
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
    if slot.get("golden_actions"):
        lo, hi = slot["golden_actions"]
        lines.append(f"      length: the golden path should need {lo}-{hi} UI actions (not 3-7); the agent has 50 steps")
    if slot.get("entity_candidates"):
        lines.append(f"      unused entities to prefer (or any other entity no prior task names): "
                     + "; ".join(slot["entity_candidates"]))
    if slot.get("avoid_entities"):
        lines.append(f"      do not name: " + "; ".join(slot["avoid_entities"]))
    if slot.get("avoid_families"):
        lines.append(f"      over-used task shapes to avoid: " + ", ".join(slot["avoid_families"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- per-bundle check

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
    batch_peers = [p for p in batch_peers if p and p["id"] != rec["id"] and earlier(p)]
    fam = meta.get("family")
    if not fam or not re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+){1,5}", str(fam)):
        problems.append("diversity: metadata.family must be a 2-6 token snake_case slug naming the task's shape")
    else:
        ft = family_tokens(fam)
        clash = [f for f in corpus.families(site) if jaccard(ft, family_tokens(f)) >= FAMILY_CLASH]
        if clash:
            problems.append(f"diversity: family '{fam}' repeats existing shape(s) {clash[:3]}; pick a genuinely new shape")
        same = [p["id"] for p in batch_peers if p["id"] != rec["id"] and p["site"] == site
                and jaccard(ft, family_tokens(p.get("family", ""))) >= FAMILY_CLASH]
        if len(same) >= family_cap:
            problems.append(f"diversity: family '{fam}' already used in this batch by {same[:2]}")
    if meta.get("slot") not in (slot["slot"], str(slot["slot"])):
        problems.append(f"diversity: metadata.slot must be {slot['slot']}")
    sim, who = corpus.max_sim(site, rec["tok"], skip_id=rec["id"])
    if sim >= NEAR_DUP:
        problems.append(f"diversity: instruction is a near-duplicate of prior task {who} (token Jaccard {sim:.2f})")
    else:
        sim, who = corpus.max_sim(site, rec["shape"], skip_id=rec["id"], key="shape")
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
    return problems


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
    over = {s: set(corpus.overused_entities(s)) for s in SITES}
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
