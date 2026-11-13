#!/usr/bin/env python3
"""Promptimize install doctor: verify a gate setup end to end.

Runs the checks a fresh machine (the macOS dogfooding install is the
first) needs before the hook can be trusted, and prints one line per
check with an actionable fix when something is wrong:

  python3 plugin/promptimize_doctor.py          # human output
  python3 plugin/promptimize_doctor.py --json   # machine output

Exit code: 0 when nothing FAILs (WARN/INFO/SKIP are fine), 1 otherwise.

Statuses:
  PASS  the check succeeded
  WARN  the hook still works, but something is degraded (e.g. no API key
        configured: decisions stay local, question drafting is off)
  FAIL  the gate cannot score prompts correctly as configured
  SKIP  a prerequisite is absent, so the check does not apply
  INFO  neutral fact worth seeing (e.g. decision server not running -
        the hook falls back to in-process scoring by design)

Nothing here sends prompts anywhere, spends money, or writes state
(except appending one usage-log line to prove the log path is writable).
"""
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "plugin"))

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"


def resolve_paths(env):
    plugin_dir = os.path.join(REPO_ROOT, "plugin")
    return {
        "model": env.get("PROMPTIMIZE_LAYA_ONNX", os.path.join(plugin_dir, "model.onnx")),
        "head": env.get("PROMPTIMIZE_GATE_HEAD", os.path.join(plugin_dir, "gate_head.json")),
        "budget": os.path.expanduser("~/.promptimize/budget.json"),
        "usage_log": os.path.expanduser("~/.promptimize/usage.jsonl"),
        "token": env.get("PROMPTIMIZE_SERVE_TOKEN_FILE",
                         os.path.expanduser("~/.promptimize/serve_token")),
        "serve_url": env.get("PROMPTIMIZE_SERVE_URL", "http://127.0.0.1:8735"),
        "claude_settings": os.path.expanduser("~/.claude/settings.json"),
        "probes": os.path.join(REPO_ROOT, "experiments", "probe2_probes.jsonl"),
        "probe_result": os.path.join(REPO_ROOT, "experiments", "probe2_result.json"),
    }


def check_python(env, paths):
    v = sys.version_info
    if v >= (3, 9):
        return (PASS, "python", "python %d.%d.%d" % (v[0], v[1], v[2]), None)
    return (FAIL, "python", "python %d.%d is too old" % (v[0], v[1]),
            "use python 3.9+ (the hook and laya need it)")


def check_deps(env, paths):
    import importlib.util
    missing = [m for m in ("numpy", "onnxruntime", "laya")
               if importlib.util.find_spec(m) is None]
    if not missing:
        return (PASS, "deps", "numpy, onnxruntime and laya all importable", None)
    return (FAIL, "deps", "missing python packages: %s" % ", ".join(missing),
            "pip install laya onnxruntime   (numpy comes with them)")


def check_model_file(env, paths):
    p = paths["model"]
    if not os.path.exists(p):
        return (FAIL, "model", "no ONNX model at %s" % p,
                "curl -L -o plugin/model.onnx "
                "https://huggingface.co/codenamev/laya-onnx/resolve/main/english/model.onnx")
    size = os.path.getsize(p)
    if size < 100_000_000:
        return (FAIL, "model", "model file is only %.1f MB - truncated download"
                % (size / 1e6), "re-download the model (expect ~850 MB)")
    return (PASS, "model", "model present (%.0f MB)" % (size / 1e6), None)


def check_head_artifact(env, paths):
    p = paths["head"]
    if not os.path.exists(p):
        return (WARN, "head", "no gate_head.json - decider falls back to zero-shot "
                "noul (measured val acc 0.682 vs 0.977 for the head)",
                "keep plugin/gate_head.json next to the hook")
    try:
        with open(p) as f:
            a = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return (FAIL, "head", "gate_head.json unreadable: %s" % e,
                "restore the artifact from the repo")
    if a.get("format") != "promptimize-gate-head/v1":
        return (FAIL, "head", "unsupported artifact format %r" % a.get("format"),
                "use the artifact from this repo version")
    dims = {len(a.get(k, [])) for k in ("w", "mu", "sd")}
    if len(dims) != 1:
        return (FAIL, "head", "artifact arrays disagree on length: %s" % sorted(dims),
                "restore the artifact from the repo")
    return (PASS, "head", "trained head artifact OK (%d-dim)" % dims.pop(), None)


def check_normalization(env, paths):
    try:
        from gate_normalize import normalize
    except ImportError as e:
        return (FAIL, "normalize", "scripts/gate_normalize.py not importable: %s" % e,
                "run the doctor from the repo root; the repo layout must stay intact")
    got = normalize("Fix the bug in src/auth/login.py please")
    if got != "Fix the bug in please":
        return (FAIL, "normalize", "normalization contract broken: got %r" % got,
                "gate_normalize.py must strip path-like tokens; restore it from the repo")
    return (PASS, "normalize", "T5 normalization contract intact", None)


def check_decider(env, paths):
    import promptimize_gate as gate
    d = gate.decider_name()
    if d == "head":
        return (PASS, "decider", "decider resolves to the trained head", None)
    return (WARN, "decider", "decider resolves to %r (zero-shot fallback)" % d,
            "restore plugin/gate_head.json for the measured-better head decider")


def _deps_available():
    try:
        import onnxruntime, laya, numpy  # noqa: F401
        return True
    except ImportError:
        return False


def check_scoring(env, paths):
    """Smoke-test the real pipeline on two frozen diagnostic probes with
    committed expected scores: A10 (ambiguous, p_amb 0.9262) must score
    ask-side, E1 (executable, p_amb 0.0) must score pass-side. Bands are
    deliberately loose - this catches a broken pipeline, not jitter."""
    if not (os.path.exists(paths["model"]) and os.path.exists(paths["head"])):
        return (SKIP, "scoring", "model or head artifact missing", None)
    if not (os.path.exists(paths["probes"]) and os.path.exists(paths["probe_result"])):
        return (SKIP, "scoring", "frozen probe expectations not in this copy "
                "(bare plugin install)", None)
    if not _deps_available():
        return (SKIP, "scoring", "deps missing", None)
    probes = {json.loads(l)["id"]: json.loads(l)
              for l in open(paths["probes"])}
    expect = {p["id"]: p["p_ambiguous"]
              for p in json.load(open(paths["probe_result"]))["per_probe"]}
    import promptimize_gate as gate
    head = gate.load_head(paths["head"])
    t0 = time.time()
    failures = []
    details = []
    for pid, band in (("A10", ("ask", 0.5)), ("E1", ("pass", 0.05))):
        p_amb = gate.score_head(gate.embed_prompt(probes[pid]["prompt"]), head)
        side, bound = band
        ok = p_amb >= bound if side == "ask" else p_amb <= bound
        details.append("%s p_amb=%.3f (frozen %.4f)" % (pid, p_amb, expect[pid]))
        if not ok:
            failures.append("%s scored %.3f, expected %s-side of %.2f"
                            % (pid, p_amb, side, bound))
    ms = int((time.time() - t0) * 1000)
    if failures:
        return (FAIL, "scoring", "; ".join(failures),
                "pipeline/artifact mismatch - re-clone the repo and re-download the model")
    return (PASS, "scoring", "pipeline smoke OK in %d ms: %s" % (ms, "; ".join(details)), None)


def check_key(env, paths):
    if env.get("PROMPTIMIZE_ANTHROPIC_KEY_CMD"):
        return (PASS, "api-key", "key command configured (e.g. macOS Keychain)", None)
    f = env.get("PROMPTIMIZE_ANTHROPIC_KEY_FILE")
    if f:
        if not os.path.exists(f):
            return (FAIL, "api-key", "PROMPTIMIZE_ANTHROPIC_KEY_FILE points at a "
                    "missing file: %s" % f, "create it (install -m 600 /dev/null %s, "
                    "then edit) or unset the variable" % f)
        mode = os.stat(f).st_mode & 0o777
        if mode & 0o077:
            return (FAIL, "api-key", "key file %s is group/world-readable (%03o) - "
                    "the hook refuses it" % (f, mode), "chmod 600 %s" % f)
        return (PASS, "api-key", "key file configured with 0600 permissions", None)
    return (WARN, "api-key", "no Anthropic key configured - decisions stay local and "
            "free, but no clarifying question is drafted on flagged prompts",
            "see plugin/README.md (Keychain or 0600 key file)")


def check_budget(env, paths):
    p = paths["budget"]
    if not os.path.exists(p):
        return (WARN, "budget", "budget ledger not provisioned - every paid call is "
                "refused (decisions still work, question drafting does not)",
                "python3 scripts/budget_ledger.py init --cap <remaining allowance> "
                "--note '...' (see plugin/README.md - never re-provision spent money)")
    try:
        with open(p) as f:
            json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return (FAIL, "budget", "budget ledger unreadable: %s" % e,
                "the ledger fails closed; restore or re-provision it")
    return (PASS, "budget", "budget ledger present and parseable", None)


def check_usage_log(env, paths):
    p = paths["usage_log"]
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"ts": time.time(), "kind": "doctor"}) + "\n")
    except OSError as e:
        return (FAIL, "usage-log", "cannot append to %s: %s" % (p, e),
                "check permissions on ~/.promptimize/")
    return (PASS, "usage-log", "usage log writable at %s" % p, None)


def check_server(env, paths):
    if env.get("PROMPTIMIZE_SERVE", "1").lower() in ("0", "off", "false"):
        return (INFO, "server", "decision server disabled (PROMPTIMIZE_SERVE=0); "
                "hook scores in-process", None)
    import urllib.request
    tok = None
    try:
        with open(paths["token"]) as f:
            tok = f.read().strip() or None
    except OSError:
        pass
    headers = {"X-Promptimize-Token": tok} if tok else {}
    url = paths["serve_url"].rstrip("/") + "/healthz"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=2) as resp:
            json.loads(resp.read())
    except Exception:
        return (INFO, "server", "decision server not reachable - hook falls back to "
                "in-process scoring by design (slower per prompt)",
                "optional: start plugin/laya_serve.py for warm scoring (see README)")
    if tok and os.path.exists(paths["token"]):
        mode = os.stat(paths["token"]).st_mode & 0o777
        if mode != 0o600:
            return (WARN, "server", "server reachable but token file is %03o, "
                    "expected 600" % mode, "chmod 600 %s" % paths["token"])
    return (PASS, "server", "decision server reachable with valid token auth", None)


def check_claude_wiring(env, paths):
    p = paths["claude_settings"]
    if not os.path.exists(p):
        return (WARN, "claude-wiring", "no ~/.claude/settings.json found - the hook "
                "is not wired into Claude Code yet",
                "add the UserPromptSubmit hook block from plugin/README.md")
    try:
        with open(p) as f:
            text = f.read()
    except OSError as e:
        return (WARN, "claude-wiring", "cannot read %s: %s" % (p, e), None)
    if "promptimize_gate.py" in text:
        return (PASS, "claude-wiring", "settings.json references the gate hook", None)
    return (WARN, "claude-wiring", "settings.json does not mention promptimize_gate.py",
            "add the UserPromptSubmit hook block from plugin/README.md")


ALL_CHECKS = [
    check_python, check_deps, check_model_file, check_head_artifact,
    check_normalization, check_decider, check_scoring, check_key,
    check_budget, check_usage_log, check_server, check_claude_wiring,
]


def run_checks(env=None):
    env = os.environ if env is None else env
    paths = resolve_paths(env)
    results = []
    for fn in ALL_CHECKS:
        try:
            status, name, detail, fix = fn(env, paths)
        except Exception as e:
            status, name, detail, fix = (FAIL, fn.__name__.replace("check_", ""),
                                         "check itself crashed: %s: %s"
                                         % (type(e).__name__, e), None)
        results.append({"status": status, "check": name, "detail": detail, "fix": fix})
    return results


def main():
    results = run_checks()
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1))
    else:
        for r in results:
            line = "%-5s %-14s %s" % (r["status"], r["check"], r["detail"])
            print(line)
            if r["fix"]:
                print("      -> %s" % r["fix"])
        fails = sum(1 for r in results if r["status"] == FAIL)
        warns = sum(1 for r in results if r["status"] == WARN)
        print("\n%d checks, %d failed, %d warnings." % (len(results), fails, warns))
        if fails:
            print("Fix the FAIL lines above and re-run; the hook itself always "
                  "fails open, but a FAIL here means it is not scoring as configured.")
    sys.exit(1 if any(r["status"] == FAIL for r in results) else 0)


if __name__ == "__main__":
    main()
