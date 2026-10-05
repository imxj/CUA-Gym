#!/usr/bin/env python3
"""WebArena task-generation pipeline, driven unattended through the Claude Agent SDK.

Runs one batch described by a spec (scripts/pipeline/specs/<batch>.json) end to end:

  preflight  deterministic environment checks: mocks, state API, endpoints, tests, CLI auth, data
  author     one `task-author` session per site lane, writing a bundle for each of that site's slots
  gate       deterministic: preflight_bundles.py, empty_state_probe.py, near_dup_scan.py (triage)
  repair     a gate-failed bundle moves to rejected/ and a fresh session re-authors its slot; re-gate
  verify     one `orchestrator` session per gate-passing task (reward-gen, golden-browser and
             reward-audit run inside it); completion is batch_orchestrator.task_is_complete()
  post       promote the verified bundle files from runs/<id>/ back into tasks/<site>/<id>/,
             re-gate them, check click-reachability, print the status report
  export     export_nemo_rollouts.py, filtered to tasks that cleared every stage
             -> <batch_dir>/<batch_id>_webarena.jsonl
  all        preflight, author, gate/repair loop, verify, one replace-and-reverify round, post, export
  status     per-slot table and cost totals
  ping       one tool-less SDK session: proves the SDK, the CLI and its login work

LLM sessions run with permission_mode="bypassPermissions" behind PreToolUse guard hooks: a session
may write only inside its own directory (a site's task directory for authors, a task's run
directory for verifiers, plus /tmp), git is read-only, the harness and hub/ are protected, the
mock lifecycle and the batch tooling belong to this driver, and authors may not run validation
code. The guard is a seatbelt against a confused agent, not a sandbox against a hostile one.

Deterministic steps are subprocesses. State is resumable in <batch_dir>/pipeline/<batch_id>/state.json
(the filesystem is the source of truth; state records decisions and history); every invocation
writes a stats file with per-session tokens, turns, duration and cost next to it.

Modelled on OSWorld-V2 scripts/python/v15_datagen_pipeline.py.

Usage, on the box that serves the mocks, with the claude CLI logged in:
  python3 scripts/webarena_datagen_pipeline.py --spec scripts/pipeline/specs/smk1.json ping
  python3 scripts/webarena_datagen_pipeline.py --spec scripts/pipeline/specs/smk1.json all
  python3 scripts/webarena_datagen_pipeline.py --spec scripts/pipeline/specs/smk1.json status
  ... --dry-run all     # control flow only: no API calls, no moves or copies, state in *.dry_run.json
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import fcntl
import fnmatch
import glob
import itertools
import json
import os
import re
import shlex
import shutil
import string
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
PROMPTS = SCRIPTS / "pipeline" / "prompts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "pipeline"))
from batch_orchestrator import load_env, task_is_complete  # noqa: E402
import diversity as div  # noqa: E402

PY = sys.executable
CANONICAL = ("task_instruction.json", "task.json", "reward.py", "nemo_reward.py", "nemo_task.json",
             "initial_setup.py", "requirements.txt")
DEFAULT_CLI = Path.home() / ".local" / "bin" / "claude"

# --------------------------------------------------------------------------- guard policy

PROTECTED_DIRS = ("hub", "cuagym", "cua_gym_web", "scripts", "run_info", "prior_batches", "docs", "tests",
                  "filter", "figures", ".claude", ".git", "output/reference")
PROTECTED_GLOBS = ("TASK*.md", "SETUP.md", "README.md", "CONTRIBUTING.md", ".env", ".env.*", ".gitmodules",
                   ".gitignore", "pyproject.toml", "vm_requirements.txt")
WRITE_VERBS = {"rm", "mv", "cp", "install", "ln", "rsync", "tee", "truncate", "dd", "touch", "rmdir", "chmod",
               "chown", "patch", "unzip", "shred", "sponge"}
GIT_READONLY = {"status", "diff", "log", "show", "blame", "ls-files", "rev-parse", "grep", "describe", "cat-file",
                "check-ignore", "shortlog"}
# Only enforced when a gateway token is injected into the session env (--base-url); with the CLI's own login
# the whitelisted env holds no secret, and agents routinely run `env | grep CUA_GYM` to find the endpoints.
ENV_DUMP_TEXT = [
    (r"(^|[;&|\s])(env|printenv)\s*($|[|>;&])", "dumping the environment can expose the gateway token"),
    (r"\bexport\s+-p\b|\bdeclare\s+-p|(^|[;&|\s])set\s*($|[|>;&])", "dumping shell variables can expose the token"),
    (r"/proc/[^\s/]+/environ", "reading process environments can expose the gateway token"),
]
DANGEROUS_TEXT = [
    (r"ANTHROPIC_|CLAUDE_CODE_OAUTH|\.credentials\.json|\.claude\.json|inference_token|INFERENCE_TOKEN",
     "credential names and files are off limits"),
    (r"(^|[\s/'\"=<])\.env(\s|$|['\";|&>])", ".env holds endpoint config and possibly secrets; it is off limits"),
    (r"(^|[;&|`(]\s*|\$\(\s*)eval\s", "eval hides the real command from the guard"),
    (r"\|\s*(ba|z|da)?sh\b(\s|$)", "piping into a shell hides the real command from the guard"),
    (r"base64\s+(-d|--decode)|xxd\s+-r", "decoding obfuscated commands is not allowed"),
    (r"\btmux\s+(kill|respawn)|\b(pkill|killall)\b", "the mock servers are managed by the operator, not by sessions"),
    (r"(^|[;&|]\s*|^\s*)(sudo|systemctl|apt-get|apt|reboot|shutdown)\s", "system administration is out of scope"),
]
# Scripts a session may READ but not RUN (matched on what a command segment executes, not on its text).
DRIVER_ONLY = {"batch_orchestrator.py", "export_nemo_rollouts.py", "webarena_datagen_pipeline.py",
               "deploy-all.sh", "start_webarena_mocks.sh"}
VALIDATION = {"preflight_bundles.py", "check_reachability.py", "empty_state_probe.py", "run_webarena_task.py",
              "near_dup_scan.py"}
_SKIP_PREFIX = {"command", "exec", "nohup", "time", "nice", "ionice", "stdbuf", "timeout"}
_SHELL_KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}"}
SYSTEM_APPEND = (
    "You are one worker inside an automated WebArena task-generation pipeline. Work autonomously; nobody will "
    "answer questions. Use PYTHONDONTWRITEBYTECODE=1 for Python runs. Never run git write commands, the batch "
    "tooling, or anything that restarts the mock servers; write only where your brief says (a guard hook denies "
    "everything else, including through Bash). Long commands are fine. Keep working until every deliverable "
    "exists; a partial result is a failure. Your final message is parsed by a program: when a JSON result is "
    "requested, produce exactly that object."
)

AUTHOR_SCHEMA = {"type": "object", "properties": {
    "tasks": {"type": "array", "items": {"type": "object", "properties": {
        "slot": {"type": "integer"}, "task_id": {"type": "string"}}, "required": ["slot", "task_id"]}},
    "notes": {"type": "string"}}, "required": ["tasks", "notes"]}
VERIFY_SCHEMA = {"type": "object", "properties": {
    "task_id": {"type": "string"}, "verified": {"type": "boolean"},
    "initial_score": {"type": ["number", "null"]}, "replay_score": {"type": ["number", "null"]},
    "rounds": {"type": "integer"}, "failed_conditions": {"type": "array", "items": {"type": "string"}},
    "notes": {"type": "string"}}, "required": ["task_id", "verified", "rounds", "failed_conditions", "notes"]}
PING_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "model_seen": {"type": "string"}},
               "required": ["ok", "model_seen"]}

_COUNTER = itertools.count(1)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def unique_name(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_{next(_COUNTER)}"


def schema_ok(data: Any, schema: dict[str, Any] | None) -> bool:
    return isinstance(data, dict) and all(k in data for k in (schema or {}).get("required", []))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- config / state / stats

@dataclasses.dataclass
class Cfg:
    spec: dict[str, Any]
    spec_path: Path
    model: str | None
    effort: str | None
    fallback_model: str | None
    max_budget_usd: float | None
    author_concurrency: int
    verify_concurrency: int
    author_timeout: int
    verify_timeout: int
    author_max_turns: int
    verify_max_turns: int
    repair_rounds: int
    replace_rounds: int
    fix_rounds: int
    cli_path: str | None
    base_url: str | None
    token_env: str
    dry_run: bool
    force: bool
    only: list[str] | None
    profiles: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    verify_retries: int = 2          # re-verify sessions that crashed or never finished a round
    retry_pause: int = 300           # seconds to wait before such a retry (rate limits, overload)

    def profile(self, role: str) -> dict[str, Any]:
        """Model routing for a session role: the spec's roles.<role> over the default profile.

        A profile with base_url goes through an Anthropic-compatible gateway (e.g. the NVIDIA inference API
        for DeepSeek) with the bearer token from token_env; without it the CLI uses its own login.
        base_url may name environment variables ("${CUAGYM_GATEWAY_BASE_URL}", set in .env) so the gateway
        host stays out of the committed specs.
        """
        base = dict(self.profiles.get("default", {}))
        role_key = {"ping": "default", "author": "author", "verify": "verify"}.get(role, role)
        base.update(self.profiles.get(role_key, {}))
        if base.get("base_url"):
            url = os.path.expandvars(str(base["base_url"]))
            if "$" in url:
                sys.exit(f"profile {role_key}: base_url {base['base_url']} is unset; export it or add it to .env")
            base["base_url"] = url
        return base

    @property
    def batch_id(self) -> str:
        return self.spec["batch_id"]

    @property
    def batch_dir(self) -> Path:
        return REPO / self.spec.get("batch_dir", f"output/{self.batch_id}")

    @property
    def tasks_dir(self) -> Path:
        return self.batch_dir / "tasks"

    @property
    def runs_dir(self) -> Path:
        return self.batch_dir / "runs"

    @property
    def rejected_dir(self) -> Path:
        return self.batch_dir / "rejected"

    @property
    def pipe_dir(self) -> Path:
        return self.batch_dir / "pipeline" / self.batch_id

    @property
    def plan_path(self) -> Path:
        return self.pipe_dir / "plan.json"

    @property
    def slots(self) -> list[dict[str, Any]]:
        """Fixed slots from the spec, or the slots `plan` wrote for a spec with a "plan" section."""
        if "slots" in self.spec:
            return self.spec["slots"]
        if self.plan_path.is_file():
            return json.loads(self.plan_path.read_text(encoding="utf-8"))["slots"]
        return []

    @property
    def diversity(self) -> bool:
        return bool(self.spec.get("diversity"))

    def endpoint_bases(self) -> dict[str, str]:
        """Mock base URLs from the environment (.env), as the runner resolves them."""
        out = {}
        for site in div.SITES:
            url = os.environ.get(f"CUA_GYM_WEBARENA_{site.upper()}_URL")
            if url:
                out[site] = url
        return out

    @property
    def sites(self) -> list[str]:
        return list(dict.fromkeys(s["site"] for s in self.slots))

    def site_dir(self, site: str) -> Path:
        return self.tasks_dir / site

    def bundle(self, task_id: str, site: str) -> Path:
        return self.tasks_dir / site / task_id

    def run_dir(self, task_id: str) -> Path:
        return self.runs_dir / task_id


class State:
    """Resumable decisions and history. One driver per batch at a time (see DriverLock)."""

    def __init__(self, cfg: Cfg):
        name = "state.dry_run.json" if cfg.dry_run else "state.json"
        self.path = cfg.pipe_dir / name
        self.data: dict[str, Any] = {"batch_id": cfg.batch_id, "slots": {}, "tasks": {}, "lanes": {}, "stages": {}}
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
        elif cfg.dry_run and (cfg.pipe_dir / "state.json").exists():
            self.data.update(json.loads((cfg.pipe_dir / "state.json").read_text(encoding="utf-8")))
        for s in cfg.slots:
            rec = self.data["slots"].setdefault(str(s["slot"]), {"task_id": None, "history": [], "replacements": 0})
            rec.update({k: v for k, v in s.items() if k != "slot"})      # the slot's contract, kept current

    def slot(self, n: int | str) -> dict[str, Any]:
        return self.data["slots"][str(n)]

    def task(self, task_id: str) -> dict[str, Any]:
        return self.data["tasks"].setdefault(task_id, {})

    def current(self) -> list[tuple[int, str, str]]:
        """(slot, site, task_id) for every filled slot."""
        out = []
        for n, rec in sorted(self.data["slots"].items(), key=lambda kv: int(kv[0])):
            if rec.get("task_id"):
                out.append((int(n), rec["site"], rec["task_id"]))
        return out

    def event(self, n: int, task_id: str | None, event: str, detail: str = "") -> None:
        self.slot(n)["history"].append({"at": now_iso(), "task_id": task_id, "event": event, "detail": detail[:2000]})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        tmp.replace(self.path)


class DriverLock:
    def __init__(self, cfg: Cfg):
        cfg.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.fh = (cfg.pipe_dir / "driver.lock").open("a+")

    def __enter__(self) -> "DriverLock":
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another pipeline driver holds this batch's lock; wait for it or stop it first")
        return self

    def __exit__(self, *exc: Any) -> None:
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)


class Stats:
    def __init__(self, cfg: Cfg, command: str):
        self.cfg, self.command = cfg, command
        self.started, self.t0 = now_iso(), time.time()
        self.sessions: list[dict[str, Any]] = []
        self.subprocesses: list[dict[str, Any]] = []

    def write(self) -> Path:
        out = self.cfg.pipe_dir / "stats"
        out.mkdir(parents=True, exist_ok=True)
        keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "duration_ms", "num_turns")
        totals: dict[str, Any] = {k: sum(int(s.get(k, 0) or 0) for s in self.sessions) for k in keys}
        totals["cost_usd"] = round(sum(float(s.get("cost_usd", 0) or 0) for s in self.sessions), 4)
        payload = {"command": " ".join(sys.argv), "subcommand": self.command, "start": self.started, "end": now_iso(),
                   "wall_seconds": round(time.time() - self.t0, 1), "model": self.cfg.model, "effort": self.cfg.effort,
                   "dry_run": self.cfg.dry_run, "totals": totals, "sessions": self.sessions,
                   "subprocesses": self.subprocesses}
        path = out / f"{unique_name(self.command)}.json"
        path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
        return path


# --------------------------------------------------------------------------- guard hooks

def _is_protected(path: Path) -> str | None:
    try:
        r = str(path.relative_to(REPO))
    except ValueError:
        return None
    for d in PROTECTED_DIRS:
        if r == d or r.startswith(d + "/"):
            return f"{r} is inside the protected tree {d}/"
    if "/" not in r and any(fnmatch.fnmatchcase(r, g) for g in PROTECTED_GLOBS):
        return f"{r} is a protected top-level file"
    return None


def _allowed_write(path: Path, writable: list[str]) -> bool:
    text = str(path)
    if any(fnmatch.fnmatchcase(text, pattern) for pattern in writable):
        return True
    tmpdir = os.environ.get("TMPDIR", "").rstrip("/")
    return (text == "/tmp" or text.startswith("/tmp/") or text.startswith("/dev/")
            or bool(tmpdir and (text == tmpdir or text.startswith(tmpdir + "/"))))


def _is_writable_ancestor(path: Path, writable: list[str]) -> bool:
    text = str(path).rstrip("/")
    return any(pattern.split("*")[0].rstrip("/").startswith(text) for pattern in writable)


def _split_segments(command: str) -> list[str]:
    """Split a shell command into simple-command segments on ; | || && newline $( ` ) — outside quotes only."""
    segments, buf, quote, i = [], [], None, 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < len(command):
                buf.append(command[i + 1])
                i += 1
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif command.startswith(("||", "&&", "$("), i):
            segments.append("".join(buf))
            buf = []
            i += 1
        elif ch in ";|\n`)":
            segments.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return segments


def _strip_heredoc_bodies(command: str) -> str:
    """Drop heredoc bodies (keep their header line) so file contents are not parsed as commands."""
    out, skip_until = [], None
    for line in command.splitlines():
        if skip_until is not None:
            if line.strip() == skip_until:
                skip_until = None
            continue
        out.append(line)
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            skip_until = m.group(1)
    return "\n".join(out)


def _resolve(cwd: Path, raw: str) -> Path:
    return Path(raw).resolve() if raw.startswith("/") else (cwd / raw).resolve()


def _executed(argv: list[str]) -> tuple[set[str], bool]:
    """Basenames of the scripts a command segment runs, and whether it drives Playwright."""
    exe = os.path.basename(argv[0])
    ran: set[str] = set()
    drives_playwright = False
    if re.match(r"^python[0-9.]*$", exe) or exe in {"bash", "sh", "zsh", "dash", "source", ".", "node", "uv"}:
        args = argv[1:]
        if "-m" in args[:2]:
            module = args[args.index("-m") + 1] if args.index("-m") + 1 < len(args) else ""
            ran.add(module.split(".")[-1] + ".py")
            drives_playwright = module.startswith("playwright")
        if "-c" in args[:2]:
            drives_playwright = drives_playwright or "playwright" in " ".join(args)
        ran.update(os.path.basename(a) for a in args if re.search(r"\.(py|sh)$", a))
    elif re.search(r"\.(py|sh)$", exe):
        ran.add(exe)
    elif exe == "playwright":
        drives_playwright = True
    return ran, drives_playwright


_VAR_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)


def _expand(text: str, shell_vars: dict[str, str]) -> str:
    """Substitute variables assigned earlier in the same command (`O=output/runs/x; mkdir -p $O/a`)."""
    return _VAR_REF.sub(lambda m: shell_vars.get(m.group(1) or m.group(2), m.group(0)), text)


def bash_verdict(command: str, writable: list[str], role: str, secrets_in_env: bool = False,
                 cwd: Path | None = None) -> str | None:
    """A denial reason for a Bash command, or None when it is acceptable.

    `cwd` is the shell's working directory when the command starts (the hook input's `cwd`): the Bash tool
    keeps its directory between calls, so `cd $R` in one call makes `cp x REVIEW.md` in the next relative to $R.
    """
    for pattern, reason in DANGEROUS_TEXT + (ENV_DUMP_TEXT if secrets_in_env else []):
        if re.search(pattern, command):
            return reason
    command = _strip_heredoc_bodies(command)
    cwd = cwd or REPO
    shell_vars: dict[str, str] = {}
    for segment in _split_segments(command):
        segment = _expand(segment.strip(), shell_vars)
        if not segment:
            continue
        try:
            words = shlex.split(segment, posix=True)
        except ValueError:
            words = []
        body = words[1:] if words[:1] == ["export"] else words
        if body and all(_ASSIGN.match(w) for w in body):     # pure assignment segment: remember, don't check
            for w in body:
                name, value = _ASSIGN.match(w).groups()
                shell_vars[name] = value
            continue
        unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", " Q ", segment)
        for target in re.findall(r"(?<![<>])>{1,2}\|?\s*([^\s;&|]+)", unquoted):
            if target.startswith("&") or target in {"/dev/null", "/dev/stderr", "/dev/stdout"}:
                continue
            if "$" in target or "~" in target:
                return f"redirection to a variable/tilde path ({target}) cannot be checked"
            path = _resolve(cwd, target)
            if (why := _is_protected(path)) or not _allowed_write(path, writable):
                return why or f"redirection writes outside this session's writable set: {target}"
        try:
            argv = shlex.split(segment, posix=True)
        except ValueError:
            return "unparseable shell segment"
        argv = [a for a in argv if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", a)]
        while argv and argv[0] in _SHELL_KEYWORDS:      # `do rm -rf $f`: the command is after the keyword
            argv = argv[1:]
        while argv and os.path.basename(argv[0]) in _SKIP_PREFIX:
            argv = argv[1:]
            if argv and re.fullmatch(r"\d+[smhd]?", argv[0]):
                argv = argv[1:]
        if not argv:
            continue
        exe = os.path.basename(argv[0])
        if exe == "cd" and len(argv) > 1 and "$" not in argv[1] and "~" not in argv[1]:
            cwd = _resolve(cwd, argv[1])
            continue
        if exe == "eval":
            return "eval hides the real command from the guard"
        ran, drives_playwright = _executed(argv)
        if ran & DRIVER_ONLY:
            return f"{', '.join(sorted(ran & DRIVER_ONLY))} runs only from the pipeline driver or the operator"
        if role == "author" and (ran & VALIDATION or drives_playwright):
            return "authors run no validation code (batch-5 rule 1); verification is a later stage"
        if exe in {"bash", "sh", "zsh", "dash"} and "-c" in argv:
            k = argv.index("-c") + 1
            if k < len(argv) and (nested := bash_verdict(argv[k], writable, role, secrets_in_env, cwd)):
                return f"nested shell: {nested}"
        if exe == "git":
            words = [a for a in argv[1:] if not a.startswith("-") and "=" not in a]
            sub = words[0] if words else ""
            if sub not in GIT_READONLY:
                return f"git {sub or '(no subcommand)'} is not read-only"
            continue
        if re.match(r"^python[0-9.]*$", exe) and "-c" in argv[1:3]:
            joined = " ".join(argv)
            if re.search(r"shutil\.(rmtree|move|copy)|os\.(remove|unlink|rename|replace)|open\([^)]*['\"](w|a)|"
                         r"write_text|write_bytes", joined) and not re.search(r"/tmp/", joined):
                return "inline Python that writes files is not allowed; write a script under /tmp or use the Write tool"
        paths: list[str] = []
        if exe == "mkdir":
            for raw in [a for a in argv[1:] if not a.startswith("-")]:
                if "$" in raw or raw.startswith("~"):
                    return f"mkdir with a variable/tilde path ({raw}) cannot be checked"
                path = _resolve(cwd, raw)
                if _is_protected(path) or not (_allowed_write(path, writable) or _is_writable_ancestor(path, writable)):
                    return f"mkdir outside this session's writable set: {raw}"
            continue
        if exe in WRITE_VERBS:
            args = [a for a in argv[1:] if not a.startswith("-")]
            if exe in {"cp", "mv", "install", "rsync", "ln"} and args:
                if exe == "mv":
                    paths = args          # mv also removes its source
                else:
                    args = args[-1:]
                    paths = args
            else:
                paths = args
        elif exe in {"sed", "perl"} and any(a.startswith("-i") for a in argv[1:]):
            paths = [a for a in argv[1:] if not a.startswith("-") and not re.match(r"^s[/|#]", a)]
        elif exe == "find" and any(a in {"-delete", "-exec", "-execdir", "-ok"} for a in argv):
            paths = [a for a in argv[1:2] if not a.startswith("-")] or ["."]
        elif exe == "tar" and any(a.startswith("-") and "x" in a for a in argv[1:3]):
            paths = [argv[argv.index("-C") + 1]] if "-C" in argv and argv.index("-C") + 1 < len(argv) else ["."]
        for raw in paths:
            if "$" in raw or "`" in raw or raw.startswith("~"):
                return f"write verb with a variable/tilde path ({raw}) cannot be checked"
            path = _resolve(cwd, raw)
            if why := _is_protected(path):
                return why
            if not _allowed_write(path, writable):
                return f"{exe} would write outside this session's writable set: {raw}"
    return None


def make_hooks(writable: list[str], role: str, log, secrets_in_env: bool = False):
    from claude_agent_sdk import HookMatcher

    def deny(reason: str) -> dict[str, Any]:
        log(f"\n[guard] DENY: {reason}\n")
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": f"pipeline guard: {reason}"}}

    def start_dir(hook_input) -> Path:
        cwd = hook_input.get("cwd")
        return Path(cwd) if cwd and Path(cwd).is_absolute() else REPO

    async def bash_guard(hook_input, tool_use_id, context):
        try:
            if hook_input.get("tool_name") != "Bash":
                return {}
            reason = bash_verdict(str(hook_input.get("tool_input", {}).get("command", "")), writable, role,
                                  secrets_in_env, start_dir(hook_input))
            return deny(reason) if reason else {}
        except Exception as exc:  # noqa: BLE001 - fail closed
            return deny(f"guard error ({exc!r})")

    async def write_guard(hook_input, tool_use_id, context):
        try:
            tool_input = hook_input.get("tool_input", {})
            target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
            if not target:
                return deny("write without a file path")
            path = _resolve(start_dir(hook_input), target)
            if why := _is_protected(path):
                return deny(why)
            if not _allowed_write(path, writable):
                return deny(f"{target} is outside this session's writable set")
            return {}
        except Exception as exc:  # noqa: BLE001 - fail closed
            return deny(f"guard error ({exc!r})")

    async def read_guard(hook_input, tool_use_id, context):
        try:
            tool_input = hook_input.get("tool_input", {})
            target = str(tool_input.get("file_path") or tool_input.get("path") or "")
            if target and _is_secret(_resolve(start_dir(hook_input), target)):
                return deny(f"{target} holds credentials or endpoint secrets")
            return {}
        except Exception as exc:  # noqa: BLE001 - fail closed
            return deny(f"guard error ({exc!r})")

    return {"PreToolUse": [HookMatcher(matcher="Bash", hooks=[bash_guard]),
                           HookMatcher(matcher="Write|Edit|MultiEdit|NotebookEdit", hooks=[write_guard]),
                           HookMatcher(matcher="Read|Grep|Glob", hooks=[read_guard])]}


SECRET_FILES: set[Path] = {REPO / ".env", Path.home() / ".claude" / ".credentials.json", Path.home() / ".claude.json"}


def _is_secret(path: Path) -> bool:
    return path in SECRET_FILES or path.name.startswith(".env") and path.parent == REPO and path.name != ".env.example"


def profile_token(prof: dict[str, Any]) -> str | None:
    """The gateway bearer token: from token_env if set in the environment, else from token_file. Never logged."""
    token = os.environ.get(prof.get("token_env") or "")
    if not token and prof.get("token_file"):
        path = Path(os.path.expanduser(prof["token_file"]))
        SECRET_FILES.add(path.resolve())
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
    return token or None


# --------------------------------------------------------------------------- session runner

_ENV_KEEP = re.compile(
    r"^(PATH|HOME|USER|LOGNAME|SHELL|LANG|LC_[A-Z_]+|TZ|TMPDIR|TERM|COLORTERM|http_proxy|https_proxy|no_proxy|"
    r"HTTP_PROXY|HTTPS_PROXY|NO_PROXY|SSL_CERT_FILE|SSL_CERT_DIR|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|"
    r"NODE_EXTRA_CA_CERTS|XDG_[A-Z_]+|PLAYWRIGHT_[A-Z_]+|VIRTUAL_ENV|PYTHONPATH|NVM_[A-Z_]+|CUA_GYM_[A-Z0-9_]+)$")
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def child_env(prof: dict[str, Any], sdir: Path) -> dict[str, str]:
    """Whitelisted environment for the CLI. Without a gateway base_url the CLI uses its own login."""
    env = {k: v for k, v in os.environ.items() if _ENV_KEEP.match(k)}
    if prof.get("base_url"):
        env["ANTHROPIC_BASE_URL"] = prof["base_url"]
        token = profile_token(prof)
        if token:
            env["ANTHROPIC_AUTH_TOKEN"] = token
        # Background calls (titles, summaries) use the "small" model; on a gateway it must be a model the
        # gateway serves, so pin it (OSWorld's dsv4 lanes pin it to the authoring model itself).
        small = prof.get("small_model") or prof.get("model")
        if small:
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_SMALL_FAST_MODEL"] = small
        # Sub-agents that ask for a model by alias must also resolve to a gateway model id.
        if prof.get("model"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = prof.get("opus_model") or prof["model"]
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = prof.get("sonnet_model") or prof["model"]
        env["API_TIMEOUT_MS"] = str(prof.get("api_timeout_ms", 30 * 60 * 1000))
        config_dir = sdir / ".claude_config"
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    env.update({
        # Sub-agents must run in the foreground. In SDK (streaming) mode a turn that ends while background
        # agents still run yields a ResultMessage, the driver closes the session and the agents die
        # (smk1, 2026-10-03: orchestrators "finished" at round 0 with reward-gen/golden-browser in flight).
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": "0",     # batch 6: print-mode sub-agents otherwise die at 600 s
        "CLAUDE_CODE_MAX_RETRIES": "8",
        # 429/529 are retried without limit (backoff up to 5 min) inside the session instead of ending it after
        # ~3.5 min: div3's gitlab/reddit authors lost 10 min of research each to gateway rate limits without it
        "CLAUDE_CODE_RETRY_WATCHDOG": "1",
        "CLAUDE_ENABLE_STREAM_WATCHDOG": "1",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(15 * 60 * 1000),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "BASH_DEFAULT_TIMEOUT_MS": str(10 * 60 * 1000),
        "BASH_MAX_TIMEOUT_MS": str(45 * 60 * 1000),
    })
    return env


@dataclasses.dataclass
class SessionResult:
    label: str
    ok: bool
    status: str
    data: dict[str, Any] | None
    metrics: dict[str, Any]
    session_dir: str


def _extract_json(text: str, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    decoder, found = json.JSONDecoder(), []
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    good = [o for o in found if schema_ok(o, schema)]
    return (good or found or [None])[-1]


def _dummy(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    t = t[0] if isinstance(t, list) else t
    if t == "object":
        return {k: _dummy(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    return {"boolean": False, "integer": 0, "number": 0.0}.get(t, "dry-run")


# The CLI retries a 429 for ~3.5 min, then ends the turn with this text as a "successful" reply. Without a shared
# pause the next queued session starts straight into the same limit: div3's first run lost ~50 lanes that way.
RATE_LIMITED = re.compile(r"API Error:[^\n]*(\b429\b|\b529\b|rate.?limit|overloaded)", re.I)
RATE_LIMIT_RETRIES = 6
_GATEWAY = {"cool_until": 0.0}


async def run_session(cfg: Cfg, sem: asyncio.Semaphore, stats: Stats, label: str, prompt: str, *, role: str,
                      agent: str | None, writable: list[str], schema: dict[str, Any] | None, timeout: int,
                      max_turns: int, tools: list[str] | None = None,
                      followup=None, max_followups: int = 2, profile_role: str | None = None,
                      _attempt: int = 0) -> SessionResult:
    """One headless SDK session. Never raises; returns a degraded SessionResult instead.

    `followup(structured) -> str | None` is asked after every completed turn; a returned prompt is sent in
    the same session (context intact) to make it finish work it stopped short of, up to max_followups times.
    A session the gateway rate-limited before it produced any output is re-run after a pause every session
    shares, up to RATE_LIMIT_RETRIES times; one that had already worked returns as a normal cut-off.
    """
    sdir = cfg.pipe_dir / "sessions" / unique_name(label)
    sdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (sdir / "prompt.md").write_text(prompt, encoding="utf-8")
    metrics: dict[str, Any] = {k: 0 for k in _TOKEN_FIELDS} | {"duration_ms": 0, "num_turns": 0, "cost_usd": 0.0}
    prof = cfg.profile(profile_role or role)
    if cfg.dry_run:
        data = _dummy(schema) if schema else None
        stats.sessions.append({"label": label, "role": role, "ok": True, "status": "dry_run", **metrics})
        print(f"[{now_iso()}] [dry-run] session {label} ({role}, agent={agent}, model={prof.get('model')}"
              f"{', gateway' if prof.get('base_url') else ''}) -> {rel(sdir)}", flush=True)
        return SessionResult(label, True, "dry_run", data, metrics, str(sdir))

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    transcript = (sdir / "transcript.md").open("w", encoding="utf-8")
    extra: dict[str, Any] = {}
    if role != "ping":
        extra["dangerously-skip-permissions"] = None
    if agent:
        extra["agent"] = agent
    options = ClaudeAgentOptions(
        cwd=str(REPO),
        cli_path=cfg.cli_path,
        setting_sources=None,              # CLI defaults: user + project + local, so .claude/ agents load
        permission_mode="default" if role == "ping" else "bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_APPEND},
        tools=tools,
        disallowed_tools=["WebSearch", "WebFetch"] + (["Agent", "Task"] if role == "author" else []),
        # the session's own CLI config dir (gateway sessions) is where the CLI spills large tool results
        hooks=None if role == "ping" else make_hooks(writable + [str(sdir) + "/*"], role, transcript.write,
                                                     secrets_in_env=bool(prof.get("base_url"))),
        model=prof.get("model"), fallback_model=prof.get("fallback_model"), effort=prof.get("effort"),
        env=child_env(prof, sdir), max_turns=max_turns, max_budget_usd=cfg.max_budget_usd,
        output_format={"type": "json_schema", "schema": schema} if schema else None,
        max_buffer_size=20 * 1024 * 1024,
        stderr=lambda line: transcript.write(f"[cli-stderr] {line}\n"),
        extra_args=extra,
    )
    text_out: list[str] = []
    structured: dict[str, Any] | None = None
    last_subtype: str | None = None
    session_id: str | None = None
    session_model: str | None = None

    def on_result(msg) -> None:
        nonlocal structured, last_subtype
        last_subtype = getattr(msg, "subtype", None)
        metrics["duration_ms"] += getattr(msg, "duration_ms", 0) or 0
        metrics["num_turns"] += getattr(msg, "num_turns", 0) or 0
        metrics["cost_usd"] = getattr(msg, "total_cost_usd", None) or metrics["cost_usd"]
        for field in _TOKEN_FIELDS:
            val = (getattr(msg, "usage", None) or {}).get(field) or 0
            if isinstance(val, (int, float)):
                metrics[field] += val
        if isinstance(so := getattr(msg, "structured_output", None), dict):
            structured = so
        if getattr(msg, "result", None):
            text_out.append(str(msg.result))
        transcript.write(f"\n\n[result subtype={last_subtype} turns={getattr(msg, 'num_turns', '?')} "
                         f"cost={getattr(msg, 'total_cost_usd', '?')}]\n")

    async def consume(client) -> None:
        nonlocal session_id, session_model
        async for msg in client.receive_response():
            name = type(msg).__name__
            if name == "SystemMessage" and getattr(msg, "subtype", "") == "init":
                data = getattr(msg, "data", None) or {}
                session_id, session_model = data.get("session_id"), data.get("model")
                transcript.write(f"session_id: {session_id}\nmodel: {session_model}\nagent: {data.get('agent')}\n\n")
            elif name == "AssistantMessage":
                for block in msg.content:
                    bname = type(block).__name__
                    if bname == "TextBlock":
                        transcript.write(block.text)
                        text_out.append(block.text)
                    elif bname == "ToolUseBlock":
                        transcript.write(f"\n[tool {block.name}] {json.dumps(block.input, default=str)[:400]}\n")
            elif name == "ResultMessage":
                on_result(msg)
            transcript.flush()

    def limited() -> bool:
        return bool(RATE_LIMITED.search("\n".join(text_out[-2:])))

    async def execute() -> None:
        nonlocal structured
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt=prompt)
            await consume(client)
            if limited():
                return
            for i in range(max_followups if followup else 0):
                more = followup(structured if schema_ok(structured, schema) else
                                _extract_json("\n".join(text_out[-3:]), schema))
                if not more or last_subtype != "success" or limited():
                    break
                transcript.write(f"\n\n[pipeline] follow-up {i + 1}: {more[:300]}\n")
                await client.query(prompt=more)
                await consume(client)
            if schema and not schema_ok(structured, schema):
                structured = _extract_json("\n".join(text_out[-3:]), schema)
            if schema and not schema_ok(structured, schema) and last_subtype == "success" and not limited():
                transcript.write("\n\n[pipeline] structured result missing; asking once more\n")
                await client.query(prompt="Return ONLY the requested JSON object now, with every required field.")
                await consume(client)
                if not schema_ok(structured, schema):
                    structured = _extract_json("\n".join(text_out[-3:]), schema)

    status = "exception"
    try:
        async with sem:
            while (wait := _GATEWAY["cool_until"] - time.time()) > 0:
                await asyncio.sleep(min(wait, 30))
            print(f"[{now_iso()}] session {label} started ({role}, agent={agent})", flush=True)
            await asyncio.wait_for(execute(), timeout=timeout)
        status = "ok" if not schema or schema_ok(structured, schema) else f"no_structured_output ({last_subtype})"
    except asyncio.TimeoutError:
        status = "timeout"
        transcript.write(f"\n\n[pipeline] TIMEOUT after {timeout}s\n")
    except Exception as exc:  # noqa: BLE001
        status = f"exception: {exc!r}"
        transcript.write("\n\n[pipeline] EXCEPTION\n```\n" + traceback.format_exc() + "\n```\n")
    finally:
        transcript.close()
    if status != "ok" and limited() and not metrics["output_tokens"] and _attempt < RATE_LIMIT_RETRIES:
        pause = min(900, 120 * 2 ** _attempt)
        _GATEWAY["cool_until"] = max(_GATEWAY["cool_until"], time.time() + pause)
        record = {"label": label, "role": role, "agent": agent, "model": session_model, "ok": False,
                  "status": "rate_limited", "last_subtype": last_subtype, "session_id": session_id, **metrics}
        (sdir / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        stats.sessions.append(record)
        print(f"[{now_iso()}] session {label}: rate_limited by the gateway before any output; every session "
              f"pauses {pause}s, then retry {_attempt + 1}/{RATE_LIMIT_RETRIES} -> {rel(sdir)}", flush=True)
        return await run_session(cfg, sem, stats, label, prompt, role=role, agent=agent, writable=writable,
                                 schema=schema, timeout=timeout, max_turns=max_turns, tools=tools, followup=followup,
                                 max_followups=max_followups, profile_role=profile_role, _attempt=_attempt + 1)
    ok = status == "ok"
    record = {"label": label, "role": role, "agent": agent, "model": session_model, "ok": ok, "status": status,
              "last_subtype": last_subtype, "session_id": session_id, "structured": structured, **metrics}
    (sdir / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    stats.sessions.append({k: v for k, v in record.items() if k != "structured"})
    metrics["model"] = session_model
    print(f"[{now_iso()}] session {label}: {status} model={session_model} turns={metrics['num_turns']} "
          f"cost=${metrics['cost_usd']:.2f} -> {rel(sdir)}", flush=True)
    return SessionResult(label, ok, status, structured if schema else None, metrics, str(sdir))


# --------------------------------------------------------------------------- subprocesses and helpers

def run_cmd(cfg: Cfg, stats: Stats, argv: list[str], *, name: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Deterministic, read-only step. Runs even under --dry-run; output logged under pipe_dir/logs."""
    log_dir = cfg.pipe_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not re.match(r"^(CLAUDECODE$|CLAUDE_CODE_)", k)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    shown = " ".join(shlex.quote(a) for a in argv)
    proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    log = log_dir / f"{unique_name(name)}.log"
    log.write_text(f"$ {shown}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n", encoding="utf-8")
    stats.subprocesses.append({"name": name, "command": shown, "returncode": proc.returncode, "log": rel(log)})
    return proc


def render(name: str, **values: Any) -> str:
    return string.Template((PROMPTS / f"{name}.md").read_text(encoding="utf-8")).safe_substitute(
        {k: str(v) for k, v in values.items()})


def brief_values(cfg: Cfg, site: str) -> dict[str, Any]:
    spec = cfg.spec
    task_doc = spec.get("task_doc")
    return {
        "batch_id": cfg.batch_id, "site": site, "site_dir": rel(cfg.site_dir(site)),
        "task_doc_line": f"{task_doc} (the batch brief)." if task_doc else "",
        "read_first": ", ".join(spec.get("read_first", [])) or "(none)",
        "corrections": " and ".join(spec.get("corrections", [])) or "(none)",
        "reference_bundle": spec.get("reference_bundle", ""),
        "difficulty": spec.get("difficulty", "medium"), "difficulty_rule": spec.get("difficulty_rule", ""),
        "inventory": spec.get("inventory", "prior_batches/inventory/{site}.tsv").format(site=site),
        "extra_rules": "\n".join(f"- {r}" for r in spec.get("extra_rules", [])),
        "contract_rules": CONTRACT_RULES if cfg.diversity else "",
    }


CONTRACT_RULES = """
Each slot carries a NOVELTY CONTRACT; the batch exists to widen the corpus, so the contract outranks convenience:
- The task's shape must be new. Record it as metadata.family: a 2-6 token snake_case slug naming the errand's
  shape (e.g. "bio_rewrite_from_profile_fact"), not its entity. It must not repeat a shape in the corpus or in
  "over-used task shapes"; a gate compares it against every existing family.
- Prefer the slot's unused entities, or any entity no prior task names; never name a "do not name" entity.
- An "inspiration" slot: write the kind of short everyday intent the official benchmark has (the official mean
  is ~18 words), grounded in this mock, with different entities and values. Resembling it is the point;
  copying or lightly paraphrasing it fails the gate. Many official intents are one-step ("Set my status to
  X"); the corpus rule still needs a retrieval, so keep the errand and make its target something the agent
  must look up ("Set my status to the title of my oldest open issue").
- A "target state" slot: the reward must check at least one listed field; trace the UI control that writes it.
- Record metadata.slot (the slot number), metadata.family, metadata.entities (names used),
  metadata.targeted_fields (state paths the reward checks) and metadata.seed_id (inspiration slots).
- Keep it solvable: the golden path should need at most ~25 UI actions (the agent has a 50-step budget) and the
  reward at most 3 components. Difficulty is secondary to novelty: derive it from the skill chain, don't pad it.
"""


def read_manifest(bundle: Path) -> dict[str, Any] | None:
    try:
        return json.loads((bundle / "task.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rebuild_site_index(cfg: Cfg, site: str) -> None:
    """index.json and nemo_tasks.jsonl are derived files: regenerate them from the bundles on disk."""
    sdir = cfg.site_dir(site)
    bundles = sorted(p.parent for p in sdir.glob("*/task.json"))
    if cfg.dry_run:
        print(f"[dry-run] would rebuild {rel(sdir)}/index.json and nemo_tasks.jsonl ({len(bundles)} bundles)")
        return
    sdir.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": 2, "tasks": [{"task_id": b.name, "path": f"{b.name}/task.json"} for b in bundles]}
    (sdir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    lines = []
    for b in bundles:
        try:
            lines.append(json.dumps(json.loads((b / "nemo_task.json").read_text(encoding="utf-8")), ensure_ascii=False))
        except (OSError, json.JSONDecodeError):
            pass
    (sdir / "nemo_tasks.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def adopt(cfg: Cfg, state: State, claims: list[dict[str, Any]] | None = None, site: str | None = None,
          only_slots: set[int] | None = None) -> None:
    """Bind on-disk bundles of this batch to unfilled slots: session claims first, then metadata.slot or
    (pair, style). A session passes only_slots so it never adopts a bundle another session is still writing."""
    taken = {tid for _, _, tid in state.current()}
    for claim in claims or []:
        n, tid = claim.get("slot"), claim.get("task_id")
        rec = state.data["slots"].get(str(n))
        if not rec or rec.get("task_id") or tid in taken or (site and rec["site"] != site) \
                or (only_slots is not None and int(n) not in only_slots):
            continue
        if (cfg.bundle(tid, rec["site"]) / "task.json").is_file():
            rec["task_id"] = tid
            taken.add(tid)
            state.task(tid).update({"slot": int(n), "site": rec["site"]})
            state.event(int(n), tid, "authored", "claimed by session")
    for s in cfg.slots:
        if site and s["site"] != site:
            continue
        for bundle in sorted(cfg.site_dir(s["site"]).glob(f"{cfg.batch_id}_*/task.json")):
            tid = bundle.parent.name
            if tid in taken or state.data["tasks"].get(tid, {}).get("rejected"):
                continue
            meta = (read_manifest(bundle.parent) or {}).get("metadata", {})
            for n, rec in sorted(state.data["slots"].items(), key=lambda kv: int(kv[0])):
                if rec["site"] != s["site"] or rec.get("task_id") or (only_slots is not None and int(n) not in only_slots):
                    continue
                by_slot = str(meta.get("slot")) == n
                by_pair = rec.get("pair") and set(meta.get("skills", [])) == set(rec["pair"]) \
                    and meta.get("style") == rec.get("style")
                if by_slot or by_pair:
                    rec["task_id"] = tid
                    taken.add(tid)
                    state.task(tid).update({"slot": int(n), "site": rec["site"]})
                    state.event(int(n), tid, "authored",
                                "adopted from disk by metadata.slot" if by_slot else "adopted from disk by (pair, style)")
                    break


def reject(cfg: Cfg, state: State, n: int, task_id: str, why: str, kind: str) -> Path:
    """Move a bundle (and its run dir, if any) to rejected/ and free its slot."""
    rec = state.slot(n)
    dest = cfg.rejected_dir / task_id
    if cfg.dry_run:
        print(f"[dry-run] would move {rel(cfg.bundle(task_id, rec['site']))} -> {rel(dest)} ({kind})")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        src = cfg.bundle(task_id, rec["site"])
        if src.exists():
            shutil.move(str(src), str(dest / "bundle"))
        if cfg.run_dir(task_id).exists():
            shutil.move(str(cfg.run_dir(task_id)), str(dest / "run"))
        (dest / "REJECTED.md").write_text(f"# {task_id}\n\nslot {n}, {kind} failure, {now_iso()}\n\n{why}\n",
                                          encoding="utf-8")
        rebuild_site_index(cfg, rec["site"])
    state.task(task_id).update({"rejected": True, "rejected_reason": why[:2000], "rejected_kind": kind})
    rec["task_id"] = None
    rec["replacements"] = rec.get("replacements", 0) + 1
    by_kind = rec.setdefault("replacements_by_kind", {})
    by_kind[kind] = by_kind.get(kind, 0) + 1
    state.event(n, task_id, f"rejected:{kind}", why)
    state.save()
    return dest


# --------------------------------------------------------------------------- stages

def stage_preflight(cfg: Cfg, stats: Stats) -> bool:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    snippet = ("import sys; sys.path.insert(0, 'scripts'); from batch_orchestrator import load_env; load_env(); "
               "from cua_gym_web.registry import EndpointRegistry; from cuagym.hub_apps import APP_DIRS; "
               "r = EndpointRegistry.from_sources(); [print(a, r.resolve(a).base_url) for a in APP_DIRS]")
    proc = run_cmd(cfg, stats, [PY, "-c", snippet], name="preflight_endpoints", timeout=120)
    urls = dict(line.split(None, 1) for line in proc.stdout.splitlines() if " http" in line)
    check("endpoints resolve", proc.returncode == 0 and len(urls) >= 5, f"{len(urls)} apps")
    for site in cfg.sites:
        app, url = f"webarena_{site}_mock", urls.get(f"webarena_{site}_mock", "").rstrip("/")
        try:
            with urllib.request.urlopen(f"{url}/go?sid=pipeline-preflight-{os.getpid()}", timeout=15) as resp:
                doc = json.loads(resp.read())
            check(f"{app} state API", set(doc) >= {"initial_state", "current_state", "state_diff"}
                  and doc["state_diff"] == {}, url)
        except Exception as exc:  # noqa: BLE001
            check(f"{app} state API", False, f"{url}: {exc}")
    proc = run_cmd(cfg, stats, [PY, "-m", "pytest", "tests/", "-q"], name="preflight_pytest", timeout=600)
    check("pytest", proc.returncode == 0, (proc.stdout.strip().splitlines() or [""])[-1])
    try:
        import claude_agent_sdk  # noqa: F401
        from importlib.metadata import version
        check("claude-agent-sdk", True, version("claude-agent-sdk"))
    except Exception as exc:  # noqa: BLE001
        check("claude-agent-sdk", False, repr(exc))
    roles = ("author", "verify")
    if any(not cfg.profile(r).get("base_url") for r in roles):
        cli = cfg.cli_path or shutil.which("claude") or "claude"
        proc = run_cmd(cfg, stats, [cli, "auth", "status"], name="preflight_auth", timeout=60)
        try:
            auth = json.loads(proc.stdout)
            check("claude CLI login", bool(auth.get("loggedIn")), auth.get("authMethod", ""))
        except json.JSONDecodeError:
            check("claude CLI login", False, proc.stdout[:200] + proc.stderr[:200])
    for r in roles:
        prof = cfg.profile(r)
        if prof.get("base_url"):
            ok, detail = gateway_probe(prof)
            check(f"{r} model {prof.get('model')} via gateway", ok, detail)
    if "plan" in cfg.spec:
        check("plan.json (run `plan` first)", cfg.plan_path.is_file(), rel(cfg.plan_path))
    spec = cfg.spec
    for path in ([spec.get("task_doc")] + spec.get("read_first", []) + spec.get("corrections", [])
                 + [spec.get("reference_bundle"), "webarena_benchmarks/webarena.jsonl", "cuagym/data/example.jsonl"]
                 + spec.get("prior_batch_roots", [])):
        if path:
            check(f"exists {path}", (REPO / path).exists())
    ref = REPO / spec.get("reference_bundle", "")
    check("reference bundle verified", (ref / "REVIEW.md").is_file() and task_is_complete(ref))
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    (cfg.pipe_dir / "preflight.json").write_text(json.dumps(
        [{"check": c[0], "ok": c[1], "detail": c[2]} for c in checks], indent=1) + "\n", encoding="utf-8")
    return all(ok for _, ok, _ in checks)


async def stage_author(cfg: Cfg, state: State, stats: Stats, on_lane_done=None,
                       sem: asyncio.Semaphore | None = None) -> None:
    """Author every empty slot, in sessions of at most lane_size slots. on_lane_done(task_ids) is awaited as each
    session finishes, so its tasks can be gated and verified while other sessions are still writing."""
    adopt(cfg, state)
    state.save()
    by_site: dict[str, list[dict[str, Any]]] = {}
    for s in cfg.slots:
        if not state.slot(s["slot"]).get("task_id"):
            by_site.setdefault(s["site"], []).append(s)
    size = max(1, int(cfg.spec.get("lane_size", 5)))      # one session writes at most this many tasks
    lanes: list[tuple[str, list[dict[str, Any]]]] = [(site, slots[i:i + size]) for site, slots in by_site.items()
                                                     for i in range(0, len(slots), size)]
    if not lanes:
        print("author: every slot already has a bundle")
        return
    sem = sem or asyncio.Semaphore(cfg.author_concurrency)

    def brief(s: dict[str, Any]) -> str:
        text = div.slot_brief(s)
        history = state.slot(s["slot"]).get("history", [])
        rejected = [h for h in history if h["event"].startswith("rejected")]
        if rejected:      # re-authoring after an interrupted repair: say what went wrong last time
            text += (f"\n      a previous attempt ({rejected[-1]['task_id']}) was rejected: "
                     + rejected[-1]["detail"][:600].replace("\n", " "))
        if any(h["event"] == "author_cutoff" for h in history):
            text += ("\n      an earlier session ran out of turns or time before writing this slot: keep the research "
                     "short and write the bundle early")
        return text

    async def lane(site: str, slots: list[dict[str, Any]]) -> None:
        lines = "\n".join(brief(s) for s in slots)
        prompt = render("author", **brief_values(cfg, site), n=len(slots), slot_lines=lines)
        label = f"author-{site}-s{slots[0]['slot']}"
        res = await run_session(cfg, sem, stats, label, prompt, role="author", agent="task-author",
                                writable=[str(cfg.site_dir(site)), str(cfg.site_dir(site)) + "/*"],
                                schema=AUTHOR_SCHEMA, timeout=cfg.author_timeout, max_turns=cfg.author_max_turns)
        state.data["lanes"].setdefault(site, []).append(
            {"at": now_iso(), "status": res.status, "session": rel(Path(res.session_dir)), **res.metrics})
        if not cfg.dry_run:
            rebuild_site_index(cfg, site)
            adopt(cfg, state, (res.data or {}).get("tasks"), site=site, only_slots={s["slot"] for s in slots})
        state.save()
        for s in slots:
            if not state.slot(s["slot"]).get("task_id"):
                state.event(s["slot"], None, "author_cutoff",
                            f"session {label} ended ({res.status}) without writing this slot")
        state.save()
        if on_lane_done:
            await on_lane_done([state.slot(s["slot"])["task_id"] for s in slots if state.slot(s["slot"]).get("task_id")])

    await asyncio.gather(*(lane(site, slots) for site, slots in lanes))
    missing = [s["slot"] for s in cfg.slots if not state.slot(s["slot"]).get("task_id")]
    if missing:
        print(f"author: slots still empty after authoring: {missing}")


async def stage_gate(cfg: Cfg, state: State, stats: Stats, only: set[str] | None = None) -> dict[str, list[str]]:
    """Run the mechanical gate; returns {task_id: [failure reasons]} for current tasks that failed.

    Subprocesses and corpus scans run in threads: verify sessions may be streaming (and waiting on guard
    hooks) on the event loop while the gate runs. State is only touched on the loop thread.
    """
    def cmd(argv, name):
        return asyncio.to_thread(run_cmd, cfg, stats, argv, name=name)
    failures: dict[str, list[str]] = {}
    summary: list[str] = []
    prior = [a for root in cfg.spec.get("prior_batch_roots", []) for a in ("--prior-batch", root)]
    for site in cfg.sites:
        sdir = cfg.site_dir(site)
        if not any(sdir.glob("*/task.json")):
            continue
        proc = await cmd([PY, "scripts/preflight_bundles.py", rel(sdir), *cfg.spec.get("gate_flags", []),
                          *prior, "--quiet"], f"gate_preflight_{site}")
        current = None
        for line in proc.stdout.splitlines():
            if m := re.match(r"FAIL (\S+)$", line):
                current = Path(m.group(1)).name
                failures.setdefault(current, [])
            elif current and line.startswith(" ") and line.strip().startswith("-"):
                failures[current].append("preflight_bundles: " + line.strip()[2:])
            else:
                current = None
                if line.startswith(f"{site}:"):
                    summary.append(line)
    proc = await cmd([PY, "scripts/empty_state_probe.py", rel(cfg.tasks_dir), "--quiet"], "gate_empty")
    current = None
    for line in proc.stdout.splitlines():
        if m := re.match(r"PAYS-ON-EMPTY (\S+)", line):
            current = m.group(1)
            failures.setdefault(current, []).append("empty_state_probe: reward pays on an empty current_state")
        elif current and line.strip().startswith("-"):
            failures[current].append("empty_state_probe: " + line.strip()[2:])
        else:
            current = None
    if cfg.diversity and state.current():
        targets = [(cfg.bundle(t, s), {"slot": n, **state.slot(n)}, t) for n, s, t in state.current()
                   if not only or t in only]
        peer_dirs = [cfg.bundle(t, s) for _, s, t in state.current()]

        def diversity_checks() -> dict[str, list[str]]:
            corpus = div.Corpus(REPO, exclude_batch_dirs=[cfg.tasks_dir])
            peers = [div.bundle_record(d) for d in peer_dirs]
            return {tid: div.check_bundle(b, slot, corpus, peers, family_cap=cfg.spec.get("family_cap", 1))
                    for b, slot, tid in targets}
        for tid, reasons in (await asyncio.to_thread(diversity_checks)).items():
            if reasons:
                failures.setdefault(tid, []).extend(reasons)
    flags: dict[str, str] = {}
    for site in cfg.sites:
        prior_dir = cfg.spec.get("near_dup_prior", "prior_batches/all/{site}").format(site=site)
        if not any(cfg.site_dir(site).glob("*/task_instruction.json")) or not (REPO / prior_dir).is_dir():
            continue
        proc = await cmd([PY, "scripts/near_dup_scan.py", "--new", rel(cfg.site_dir(site)),
                          "--prior", prior_dir, "--top", "2"], f"gate_neardup_{site}")
        for m in re.finditer(r"^FLAG (\S+)\s+\(max overlap ([0-9.]+)\)", proc.stdout, re.M):
            flags[m.group(1)] = m.group(2)
    result: dict[str, list[str]] = {}
    for n, site, tid in state.current():
        if only and tid not in only:
            continue
        reasons = failures.get(tid, [])
        state.task(tid)["gate"] = {"pass": not reasons, "reasons": reasons, "near_dup_overlap": flags.get(tid),
                                   "at": now_iso()}
        if reasons:
            result[tid] = reasons
    state.data["stages"]["gate"] = {"at": now_iso(), "summary": summary, "failed": sorted(result)}
    state.save()
    for line in summary:
        print(f"  {line}")
    print(f"gate: {len(state.current()) - len(result)}/{len(state.current())} current bundles pass"
          + (f"; failing: {sorted(result)}" if result else "") + (f"; near-dup flags: {flags}" if flags else ""))
    return result


async def stage_repair(cfg: Cfg, state: State, stats: Stats, failed: dict[str, str], kind: str,
                       sem: asyncio.Semaphore | None = None) -> list[str]:
    """Reject each failed task and re-author its slot in a fresh session. Returns the new task ids."""
    sem = sem or asyncio.Semaphore(cfg.author_concurrency)
    new_ids: list[str] = []
    limit = cfg.repair_rounds if kind == "gate" else cfg.replace_rounds

    async def one(tid: str, why: str) -> None:
        n, site = state.task(tid)["slot"], state.task(tid)["site"]
        if state.slot(n).get("replacements_by_kind", {}).get(kind, 0) >= limit:
            state.event(n, tid, "exhausted", f"{kind} failure and no {kind} replacements left")
            print(f"repair: slot {n} ({tid}) is out of replacements; leaving it")
            return
        dest = reject(cfg, state, n, tid, why, kind)
        rec = state.slot(n)
        prompt = render("repair", **brief_values(cfg, site), slot=n, contract=div.slot_brief({"slot": n, **rec}),
                        old_id=tid, rejected_path=rel(dest), reason=why,
                        nnn=f"{n:03d}", round=rec["replacements"])
        res = await run_session(cfg, sem, stats, f"repair-{site}-slot{n}", prompt, role="author",
                                agent="task-author",
                                writable=[str(cfg.site_dir(site)), str(cfg.site_dir(site)) + "/*"],
                                schema=AUTHOR_SCHEMA, timeout=cfg.author_timeout, max_turns=cfg.author_max_turns)
        if not cfg.dry_run:
            rebuild_site_index(cfg, site)
            adopt(cfg, state, (res.data or {}).get("tasks"), site=site, only_slots={n})
        if new := state.slot(n).get("task_id"):
            new_ids.append(new)
        state.save()

    await asyncio.gather(*(one(tid, why) for tid, why in failed.items()))
    return new_ids


async def stage_verify(cfg: Cfg, state: State, stats: Stats, only: set[str] | None = None,
                       sem: asyncio.Semaphore | None = None) -> dict[str, bool]:
    sem = sem or asyncio.Semaphore(cfg.verify_concurrency)
    results: dict[str, bool] = {}

    async def one(n: int, site: str, tid: str) -> None:
        rd = cfg.run_dir(tid)
        if task_is_complete(rd) and not cfg.force:
            state.task(tid)["verify"] = {**state.task(tid).get("verify", {}), "complete": True}
            results[tid] = True
            return
        if not cfg.dry_run:
            rd.mkdir(parents=True, exist_ok=True)
        bundle = cfg.bundle(tid, site)
        prompt = render("verify", task_json=str(bundle / "task.json"), bundle_dir=rel(bundle), run_dir=str(rd),
                        mode=cfg.spec.get("session_mode", "legacy"), task_id=tid, runs_dir=rel(cfg.runs_dir))

        def unfinished(report: dict[str, Any] | None) -> str | None:
            """Ask again only when the workflow stopped short, never after a genuine three-round failure."""
            if cfg.dry_run or task_is_complete(rd):
                return None
            report = report or {}
            stopped_short = (report.get("rounds", 0) or 0) < 1 or any(
                re.search(r"not yet|still running|not reported|had not|pending", str(x), re.I)
                for x in [report.get("notes", "")] + list(report.get("failed_conditions", [])))
            if not stopped_short:
                return None
            return ("The workflow stopped before it finished: no round has completed (or sub-agents had not "
                    "reported back), so nothing is verified yet. Background work is disabled in this pipeline. "
                    "Continue from where you stopped, running the remaining agents in the foreground, until "
                    "verification.json and REVIEW.md exist in the output directory or three rounds have failed. "
                    "Then return the JSON object again.")

        res = await run_session(cfg, sem, stats, f"verify-{tid}", prompt, role="verify", agent="orchestrator",
                                writable=[str(rd), str(rd) + "/*"], schema=VERIFY_SCHEMA,
                                timeout=cfg.verify_timeout, max_turns=cfg.verify_max_turns, followup=unfinished)
        complete = (not cfg.dry_run) and task_is_complete(rd)
        rec = state.task(tid).setdefault("verify", {})
        rec.update({"complete": complete, "status": res.status, "attempts": rec.get("attempts", 0) + 1,
                    "report": res.data, "session": rel(Path(res.session_dir)), "cost_usd": res.metrics["cost_usd"],
                    "at": now_iso()})
        state.event(n, tid, "verified" if complete else "verify_failed", json.dumps(res.data or {})[:1000])
        results[tid] = complete
        state.save()

    todo = [(n, s, t) for n, s, t in state.current()
            if (not only or t in only) and state.task(t).get("gate", {}).get("pass")]
    await asyncio.gather(*(one(*item) for item in todo))
    print(f"verify: {sum(results.values())}/{len(results)} complete")
    return results


def verified_genuinely(task: dict[str, Any]) -> bool:
    """Did the last verify session actually judge the task (a normal finish after at least one round)?"""
    v = task.get("verify", {})
    report = v.get("report") or {}
    return v.get("status") == "ok" and (report.get("rounds") or 0) >= 1


def verify_failure_reason(cfg: Cfg, state: State, tid: str) -> str:
    v = state.task(tid).get("verify", {})
    report = v.get("report") or {}
    parts = [f"verification status: {v.get('status')}"]
    if report.get("failed_conditions"):
        parts.append("failed agreement conditions: " + "; ".join(report["failed_conditions"]))
    if report.get("notes"):
        parts.append("orchestrator notes: " + report["notes"])
    review = next(iter(sorted(cfg.run_dir(tid).glob("**/REVIEW.md"))), None)
    if review:
        parts.append("REVIEW.md excerpt:\n" + review.read_text(errors="ignore")[:3000])
    return "\n".join(parts)


async def stage_post(cfg: Cfg, state: State, stats: Stats) -> None:
    verified = [(n, s, t) for n, s, t in state.current() if task_is_complete(cfg.run_dir(t))]
    for n, site, tid in verified:
        rd, bd = cfg.run_dir(tid), cfg.bundle(tid, site)
        changed = [f for f in CANONICAL if (rd / f).is_file()
                   and (not (bd / f).is_file() or (rd / f).read_bytes() != (bd / f).read_bytes())]
        if changed and not cfg.dry_run:
            backup = cfg.pipe_dir / "pre_promote" / tid
            backup.mkdir(parents=True, exist_ok=True)
            for f in changed:
                if (bd / f).is_file():
                    shutil.copy2(bd / f, backup / f)
                shutil.copy2(rd / f, bd / f)
        elif changed:
            print(f"[dry-run] would promote {changed} from {rel(rd)} into {rel(bd)}")
        state.task(tid)["promote"] = {"changed": changed, "at": now_iso()}
        if changed:
            state.event(n, tid, "promoted", ", ".join(changed))
    for site in {s for _, s, _ in verified}:
        rebuild_site_index(cfg, site)
    regate = await stage_gate(cfg, state, stats, only={t for _, _, t in verified}) if verified else {}
    reach_fail: dict[str, list[str]] = {}
    if verified:
        proc = await asyncio.to_thread(run_cmd, cfg, stats, [PY, "scripts/check_reachability.py",
                                       *[rel(cfg.run_dir(t)) for _, _, t in verified], "--quiet"], name="post_reach")
        current = None
        for line in proc.stdout.splitlines():
            if m := re.match(r"UNREACHABLE (\S+)", line):
                current = m.group(1)
                reach_fail.setdefault(current, [])
            elif current and line.strip().startswith("-"):
                reach_fail[current].append(line.strip()[2:])
            else:
                current = None
    for n, site, tid in verified:
        problems = regate.get(tid, []) + [f"reachability: {p}" for p in reach_fail.get(tid, [])]
        if tid in reach_fail and not reach_fail[tid]:
            problems.append("reachability: UNREACHABLE")
        state.task(tid)["post"] = {"pass": not problems, "problems": problems, "at": now_iso()}
    await asyncio.to_thread(run_cmd, cfg, stats, [PY, "scripts/_b5_status.py", "--runs", rel(cfg.runs_dir),
                                                   "--failures"], name="post_status")
    state.save()
    ok = sum(1 for _, _, t in verified if state.task(t)["post"]["pass"])
    print(f"post: {len(verified)} verified, {ok} clear the post-promotion gate and reachability")


async def stage_fix(cfg: Cfg, state: State, stats: Stats, sem: asyncio.Semaphore | None = None) -> list[str]:
    """Verified tasks whose promoted bundle fails the post gate go back to an orchestrator session with the gate's
    reasons: reward repaired in the run dir, replay re-run twice, re-audited. The next `post` promotes and re-gates.
    (div1, 2026-10-03: 11 of 20 verified tasks failed S7 because verification rewrote rewards to read initial_state.)"""
    sem = sem or asyncio.Semaphore(cfg.verify_concurrency)
    todo = [(n, s, t) for n, s, t in state.current()
            if task_is_complete(cfg.run_dir(t)) and state.task(t).get("post", {}).get("pass") is False
            and (not cfg.only or t in cfg.only)]

    async def one(n: int, site: str, tid: str) -> None:
        rd = cfg.run_dir(tid)
        reasons = "\n".join(f"- {p}" for p in state.task(tid)["post"].get("problems", []))
        prompt = render("fix", task_id=tid, run_dir=str(rd), runs_dir=rel(cfg.runs_dir), reasons=reasons)
        res = await run_session(cfg, sem, stats, f"fix-{tid}", prompt, role="verify", agent="orchestrator",
                                writable=[str(rd), str(rd) + "/*"], schema=VERIFY_SCHEMA,
                                timeout=cfg.verify_timeout, max_turns=cfg.verify_max_turns)
        rec = state.task(tid).setdefault("fix", {})
        rec.update({"attempts": rec.get("attempts", 0) + 1, "status": res.status, "report": res.data,
                    "session": rel(Path(res.session_dir)), "cost_usd": res.metrics["cost_usd"], "at": now_iso()})
        state.event(n, tid, "fix", json.dumps(res.data or {})[:1000])
        state.save()

    await asyncio.gather(*(one(*item) for item in todo))
    if todo:
        print(f"fix: {len(todo)} verified task(s) sent back for gate repairs")
    return [t for _, _, t in todo]


async def post_and_fix(cfg: Cfg, state: State, stats: Stats, sem: asyncio.Semaphore | None = None) -> None:
    await stage_post(cfg, state, stats)
    for _ in range(cfg.fix_rounds):
        if not await stage_fix(cfg, state, stats, sem=sem):
            return
        await stage_post(cfg, state, stats)


def normalize_attempt_dirs(cfg: Cfg, state: State) -> list[str]:
    """export_nemo_rollouts.py reads verification.json only from attempt-*/ or the run root. Orchestrators sometimes
    name their attempts run1/, round-2/, ... (div2: div2_reddit_negcount_gated_vote_083 verified twice in run1/ and
    run2/ and was excluded as 'not verified'). Copy such dirs to attempt-<n>/, keeping the originals."""
    fixed = []
    for _, _, tid in state.current():
        rd = cfg.run_dir(tid)
        if not task_is_complete(rd) or (rd / "verification.json").is_file() or list(rd.glob("attempt-*/verification.json")):
            continue
        dirs = sorted({p.parent for p in rd.glob("*/verification.json") if not p.parent.name.startswith("audit")},
                      key=lambda d: (int(m.group()) if (m := re.search(r"\d+", d.name)) else 0, d.stat().st_mtime))
        for i, d in enumerate(dirs, 1):
            dest = rd / f"attempt-{i}"
            if cfg.dry_run:
                print(f"[dry-run] would copy {rel(d)} -> {rel(dest)}")
            elif not dest.exists():
                shutil.copytree(d, dest)
        if dirs:
            fixed.append(tid)
            state.task(tid)["attempt_dirs_normalized"] = [d.name for d in dirs]
    if fixed:
        print(f"export: normalized attempt directories for {len(fixed)} task(s): {fixed}")
    return fixed


def stage_export(cfg: Cfg, state: State, stats: Stats) -> Path | None:
    normalize_attempt_dirs(cfg, state)
    raw = cfg.pipe_dir / ("export_raw.dry_run.jsonl" if cfg.dry_run else "export_raw.jsonl")
    proc = run_cmd(cfg, stats, [PY, "scripts/export_nemo_rollouts.py", rel(cfg.batch_dir), "--output", rel(raw),
                                "--tasks-subdir", os.path.relpath(cfg.tasks_dir, cfg.batch_dir),
                                "--runs-subdir", os.path.relpath(cfg.runs_dir, cfg.batch_dir)], name="export")
    print("\n".join("  " + line for line in proc.stdout.strip().splitlines()))
    eligible = {t for _, _, t in state.current() if state.task(t).get("post", {}).get("pass")}
    rows = []
    if raw.is_file():
        for line in raw.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("context_compaction_task_id") in eligible:
                rows.append(line)
    final = cfg.batch_dir / f"{cfg.batch_id}_webarena.jsonl"
    if cfg.dry_run:
        print(f"[dry-run] would write {len(rows)} rows to {rel(final)}")
        return None
    final.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    exported = {json.loads(r)["context_compaction_task_id"] for r in rows}
    for _, _, tid in state.current():
        state.task(tid)["exported"] = tid in exported
    state.data["stages"]["export"] = {"at": now_iso(), "rows": len(rows), "path": rel(final)}
    state.save()
    print(f"export: {len(rows)}/{len(cfg.slots)} slots -> {rel(final)}")
    return final


def stage_plan(cfg: Cfg, stats: Stats) -> bool:
    """Write <pipe_dir>/plan.json: one novelty contract per slot, drawn from the corpus's gaps."""
    if "plan" not in cfg.spec:
        print("plan: this spec has fixed slots and no \"plan\" section; nothing to do")
        return False
    if cfg.plan_path.is_file() and not cfg.force:
        print(f"plan: {rel(cfg.plan_path)} already exists; --force re-plans (bound slots keep their numbers)")
        return False
    bases = cfg.endpoint_bases()
    missing = [s for s in cfg.spec["plan"]["sites"] if s not in bases]
    if missing:
        sys.exit(f"plan: no endpoint for {missing} (CUA_GYM_WEBARENA_*_URL in .env)")
    corpus = div.Corpus(REPO, exclude_batch_dirs=[cfg.tasks_dir])
    slots = div.plan_slots(REPO, corpus, cfg.spec["plan"], bases)
    cfg.pipe_dir.mkdir(parents=True, exist_ok=True)
    cfg.plan_path.write_text(json.dumps({"batch_id": cfg.batch_id, "created": now_iso(), "corpus_tasks": len(corpus.tasks),
                                         "plan": cfg.spec["plan"], "slots": slots}, indent=1) + "\n", encoding="utf-8")
    print(f"plan: {len(slots)} slots against a corpus of {len(corpus.tasks)} tasks -> {rel(cfg.plan_path)}")
    for s in slots:
        what = {"official_seed": lambda: f"seed {s['seed_id']}: {s['seed_intent'][:70]}",
                "field": lambda: f"field {s['target_key']}: {', '.join(s['target_fields'][:3])}",
                "open": lambda: f"open pair {'+'.join(s['pair'])}"}[s["kind"]]()
        print(f"  {s['slot']:>3} {s['site']:<14} {s['style']:<8} {'short' if s.get('short') else '     '} {what}")
    return True


def stage_diversity(cfg: Cfg, state: State) -> Path:
    corpus = div.Corpus(REPO, exclude_batch_dirs=[cfg.tasks_dir])
    batch = [r for _, s, t in state.current() if (r := div.bundle_record(cfg.bundle(t, s)))]
    md = div.report(REPO, corpus, batch, cfg.endpoint_bases())
    path = cfg.pipe_dir / ("DIVERSITY.dry_run.md" if cfg.dry_run else "DIVERSITY.md")
    path.write_text(md, encoding="utf-8")
    print("\n".join(md.splitlines()[:12]))
    print(f"diversity report -> {rel(path)}")
    return path


def gateway_probe(prof: dict[str, Any]) -> tuple[bool, str]:
    """One tiny /v1/messages call through the profile's gateway. The token is sent, never printed."""
    token = profile_token(prof)
    if not token:
        where = " or ".join(x for x in (f"${prof['token_env']}" if prof.get("token_env") else "",
                                        prof.get("token_file", "")) if x)
        return False, f"no gateway token ({where or 'set token_env or token_file in the profile'})"
    req = urllib.request.Request(
        prof["base_url"].rstrip("/") + "/v1/messages",
        data=json.dumps({"model": prof["model"], "max_tokens": 16,
                         "messages": [{"role": "user", "content": "ping"}]}).encode(),
        headers={"Authorization": f"Bearer {token}", "content-type": "application/json",
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:120]


def print_status(cfg: Cfg, state: State) -> None:
    print(f"batch {cfg.batch_id}  ({rel(cfg.batch_dir)})  author={cfg.profile('author').get('model')}  "
          f"verify={cfg.profile('verify').get('model')}")
    if not cfg.slots:
        print("  no slots yet" + (" — run `plan`" if "plan" in cfg.spec else ""))
    print(f"  {'slot':>4}  {'site':<14} {'contract':<9} {'style':<8} {'gate':<5} {'verify':<7} {'post':<5} {'exp':<4} "
          f"{'repl':>4}  task_id")
    for s in cfg.slots:
        rec = state.slot(s["slot"])
        tid = rec.get("task_id") or "-"
        t = state.data["tasks"].get(tid, {})

        def mark(v: Any) -> str:
            return "-" if v is None else ("ok" if v else "FAIL")
        contract = "+".join(s["pair"]) if s.get("pair") else {"official_seed": "seed", "field": "field"}.get(
            s.get("kind"), "-")
        print(f"  {s['slot']:>4}  {s['site']:<14} {contract:<9} {s['style']:<8} "
              f"{mark(t.get('gate', {}).get('pass')):<5} {mark(t.get('verify', {}).get('complete')):<7} "
              f"{mark(t.get('post', {}).get('pass')):<5} {mark(t.get('exported')):<4} "
              f"{rec.get('replacements', 0):>4}  {tid}")
    costs = []
    for p in sorted((cfg.pipe_dir / "stats").glob("*.json")):
        try:
            costs.append(json.loads(p.read_text())["totals"]["cost_usd"])
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    print(f"  sessions cost so far: ${sum(costs):.2f} across {len(costs)} invocations")


# --------------------------------------------------------------------------- commands

async def cmd_all(cfg: Cfg, state: State, stats: Stats, skip_preflight: bool) -> int:
    if not skip_preflight and not stage_preflight(cfg, stats):
        print("preflight failed; fix the environment or pass --skip-preflight")
        return 2
    vsem = asyncio.Semaphore(cfg.verify_concurrency)      # one verification budget for the whole run
    asem = asyncio.Semaphore(cfg.author_concurrency)      # one authoring budget: lanes and repairs share it
    verifying: list[asyncio.Task] = []

    def verify_now(task_ids) -> None:
        if task_ids:
            verifying.append(asyncio.create_task(stage_verify(cfg, state, stats, only=set(task_ids), sem=vsem)))

    async def repair_loop(failed: dict[str, list[str]]) -> None:
        """Re-author gate failures, re-gate the replacements, verify the ones that pass; up to repair_rounds."""
        for _ in range(cfg.repair_rounds):
            if not failed:
                return
            new_ids = await stage_repair(cfg, state, stats, {t: "\n".join(r) for t, r in failed.items()}, kind="gate",
                                         sem=asem)
            if not new_ids:
                return
            failed = await stage_gate(cfg, state, stats, only=set(new_ids))
            verify_now([t for t in new_ids if t not in failed])

    def passing(ids) -> list[str]:
        return [t for t in ids if state.task(t).get("gate", {}).get("pass")]

    loops: list[asyncio.Task] = []

    async def gate_and_queue(task_ids) -> None:
        """Gate fresh bundles, verify the ones that pass, start a repair loop for the rest."""
        ids = {t for t in task_ids if t}
        if not ids:
            return
        failed = await stage_gate(cfg, state, stats, only=ids)
        verify_now(passing(ids))
        if failed:
            loops.append(asyncio.create_task(repair_loop(failed)))

    async def drain(tasks: list[asyncio.Task]) -> None:
        while pending := [t for t in tasks if not t.done()]:
            await asyncio.gather(*pending)

    # Every author session's tasks are gated and verified as soon as that session finishes; bundles that already
    # exist (a resumed run) are gated straight away. A task that already passed verification is left to post/fix.
    adopt(cfg, state)
    bound = {t for _, _, t in state.current()}
    existing = {t for t in bound if not task_is_complete(cfg.run_dir(t))}
    authoring = (asyncio.create_task(stage_author(cfg, state, stats, on_lane_done=gate_and_queue, sem=asem))
                 if len(bound) < len(cfg.slots) else None)
    await gate_and_queue(existing)
    if authoring:
        await authoring
    for _ in range(cfg.spec.get("author_retries", 1)):      # slots a cut-off session never wrote
        if all(state.slot(s["slot"]).get("task_id") for s in cfg.slots):
            break
        print("author: re-authoring slots that earlier sessions did not write")
        await stage_author(cfg, state, stats, on_lane_done=gate_and_queue, sem=asem)
    await drain(loops)
    await drain(verifying)
    # A session that crashed, timed out or never finished a round (API errors, rate limits) says nothing about
    # the task: verify it again before considering a replacement. Only a completed, failing verification replaces.
    for attempt in range(cfg.verify_retries):
        unrun = [t for _, _, t in state.current() if state.task(t).get("gate", {}).get("pass")
                 and not task_is_complete(cfg.run_dir(t)) and not verified_genuinely(state.task(t))]
        if not unrun:
            break
        print(f"verify: retrying {len(unrun)} task(s) whose session did not complete a round "
              f"(attempt {attempt + 2}); pausing {cfg.retry_pause}s first")
        await asyncio.sleep(cfg.retry_pause)
        await stage_verify(cfg, state, stats, only=set(unrun), sem=vsem)
    for _ in range(cfg.replace_rounds):
        failed_v = {t: verify_failure_reason(cfg, state, t) for _, _, t in state.current()
                    if state.task(t).get("gate", {}).get("pass") and not state.task(t).get("verify", {}).get("complete")
                    and not task_is_complete(cfg.run_dir(t)) and verified_genuinely(state.task(t))}
        if not failed_v:
            break
        new_ids = await stage_repair(cfg, state, stats, failed_v, kind="verify", sem=asem)
        if not new_ids:
            break
        regate = await stage_gate(cfg, state, stats, only=set(new_ids))
        await stage_verify(cfg, state, stats, only=set(new_ids) - set(regate), sem=vsem)
    await post_and_fix(cfg, state, stats, sem=vsem)
    stage_export(cfg, state, stats)
    if cfg.diversity:
        stage_diversity(cfg, state)
    print_status(cfg, state)
    return 0


async def cmd_ping(cfg: Cfg, stats: Stats, role: str) -> int:
    res = await run_session(cfg, asyncio.Semaphore(1), stats, f"ping-{role}",
                            "Reply with the JSON object {\"ok\": true, \"model_seen\": \"<your model id>\"}.",
                            role="ping", agent=None, writable=[], schema=PING_SCHEMA, timeout=600, max_turns=3,
                            tools=[], profile_role=role)
    print(json.dumps({"status": res.status, "data": res.data, "cost_usd": res.metrics["cost_usd"]}))
    return 0 if res.ok else 1


def build_cfg(args: argparse.Namespace) -> Cfg:
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cli = args.cli_path or (str(DEFAULT_CLI) if DEFAULT_CLI.exists() else None)
    # Pin the model: without it SDK sessions fall back to a different default than an interactive CLI does
    # (measured 2026-10-02: SDK -> claude-sonnet-4-6, `claude -p` -> claude-opus-5-5).
    model = args.model or spec.get("model")
    if not model:
        sys.exit("no model: set \"model\" in the spec or pass --model")
    default = {"model": model, "effort": args.effort or spec.get("effort"), "fallback_model": args.fallback_model,
               "base_url": args.base_url or spec.get("base_url"), "token_env": args.token_env}
    profiles = {"default": {k: v for k, v in default.items() if v is not None}}
    if not args.default_profile_only:
        profiles.update(spec.get("roles") or {})
    return Cfg(profiles=profiles, spec=spec, spec_path=spec_path, model=model, effort=args.effort or spec.get("effort"),
               fallback_model=args.fallback_model, max_budget_usd=args.max_budget_usd,
               author_concurrency=args.author_concurrency or (spec.get("concurrency") or {}).get("author", 4),
               verify_concurrency=args.verify_concurrency or (spec.get("concurrency") or {}).get("verify", 5),
               author_timeout=60 * (args.author_timeout_min or spec.get("author_timeout_min", 120)),
               verify_timeout=60 * (args.verify_timeout_min or spec.get("verify_timeout_min", 60)),
               author_max_turns=args.author_max_turns or spec.get("author_max_turns", 300),
               verify_max_turns=args.verify_max_turns or spec.get("verify_max_turns", 220),
               repair_rounds=args.repair_rounds, replace_rounds=args.replace_rounds, fix_rounds=args.fix_rounds,
               cli_path=cli,
               base_url=args.base_url, token_env=args.token_env, dry_run=args.dry_run, force=args.force,
               only=args.tasks)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["ping", "preflight", "plan", "author", "gate", "repair", "verify", "post",
                                       "fix", "export", "diversity", "status", "all"])
    p.add_argument("--role", default="default", choices=["default", "author", "verify"],
                   help="ping: which session profile to test (author = the spec's authoring model)")
    p.add_argument("--default-profile-only", action="store_true",
                   help="ignore the spec's per-role models; every session uses the default profile")
    p.add_argument("--spec", required=True, help="batch spec JSON (scripts/pipeline/specs/<batch>.json)")
    p.add_argument("--model", help="model for every session (default: the spec's \"model\")")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                   help="reasoning effort (default: the spec's \"effort\", else the CLI default)")
    p.add_argument("--fallback-model")
    p.add_argument("--max-budget-usd", type=float, help="per-session spend cap")
    p.add_argument("--author-concurrency", type=int, help="default: the spec's concurrency.author, else 4")
    p.add_argument("--verify-concurrency", type=int, help="default: the spec's concurrency.verify, else 5")
    p.add_argument("--author-timeout-min", type=int, help="default: the spec's author_timeout_min, else 120")
    p.add_argument("--verify-timeout-min", type=int, help="default: the spec's verify_timeout_min, else 60")
    p.add_argument("--author-max-turns", type=int, help="default: the spec's author_max_turns, else 300")
    p.add_argument("--verify-max-turns", type=int, help="default: the spec's verify_max_turns, else 220")
    p.add_argument("--repair-rounds", type=int, default=2, help="gate-failure re-authoring rounds per slot")
    p.add_argument("--replace-rounds", type=int, default=1, help="verification-failure replacement rounds")
    p.add_argument("--fix-rounds", type=int, default=2, help="post-gate repair rounds for verified tasks")
    p.add_argument("--cli-path", help=f"claude CLI to drive (default: {DEFAULT_CLI} if present)")
    p.add_argument("--base-url", help="route through an Anthropic-compatible gateway instead of the CLI login")
    p.add_argument("--token-env", default="ANTHROPIC_AUTH_TOKEN", help="env var holding the gateway bearer token")
    p.add_argument("--tasks", nargs="*", help="restrict gate/verify/repair to these task ids")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="no API calls, no moves or copies; state in *.dry_run.json")
    p.add_argument("--force", action="store_true", help="re-verify tasks that already have a passing run")
    args = p.parse_args()
    load_env()                      # CUA_GYM_WEBARENA_*_URL from .env, as batch_orchestrator.py does
    cfg = build_cfg(args)
    cfg.pipe_dir.mkdir(parents=True, exist_ok=True)
    stats = Stats(cfg, args.command)
    only = set(args.tasks) if args.tasks else None
    if args.command == "status":    # read-only: no lock, nothing saved, safe while a driver is running
        state = State(cfg)
        adopt(cfg, state)
        print_status(cfg, state)
        return 0
    with DriverLock(cfg):
        if args.command == "plan" or (args.command in ("all", "author") and "plan" in cfg.spec
                                      and not cfg.plan_path.is_file()):
            stage_plan(cfg, stats)
            if args.command == "plan":
                return 0
        state = State(cfg)                  # after planning, so planned slots are in state
        try:
            if args.command == "ping":
                return asyncio.run(cmd_ping(cfg, stats, args.role))
            if args.command == "preflight":
                return 0 if stage_preflight(cfg, stats) else 1
            if args.command == "diversity":
                adopt(cfg, state)
                stage_diversity(cfg, state)
                return 0
            if args.command == "author":
                asyncio.run(stage_author(cfg, state, stats))
            elif args.command == "gate":
                adopt(cfg, state)
                return 1 if asyncio.run(stage_gate(cfg, state, stats, only=only)) else 0
            elif args.command == "repair":
                failed = {t: "\n".join(state.task(t).get("gate", {}).get("reasons", [])) for _, _, t in state.current()
                          if state.task(t).get("gate", {}).get("pass") is False and (not only or t in only)}
                asyncio.run(stage_repair(cfg, state, stats, failed, kind="gate"))
            elif args.command == "verify":
                adopt(cfg, state)
                asyncio.run(stage_verify(cfg, state, stats, only=only))
            elif args.command == "post":
                asyncio.run(stage_post(cfg, state, stats))
            elif args.command == "fix":                 # post-gate repairs for verified tasks, then export
                adopt(cfg, state)
                asyncio.run(post_and_fix(cfg, state, stats))
                stage_export(cfg, state, stats)
                if cfg.diversity:
                    stage_diversity(cfg, state)
                print_status(cfg, state)
            elif args.command == "export":
                stage_export(cfg, state, stats)
            elif args.command == "all":
                return asyncio.run(cmd_all(cfg, state, stats, args.skip_preflight))
            return 0
        finally:
            state.save()
            if stats.sessions or stats.subprocesses:
                print(f"stats: {rel(stats.write())}")


if __name__ == "__main__":
    sys.exit(main())
