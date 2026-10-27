#!/usr/bin/env python3
"""Prompt evaluation harness: runs every prompt version on every test case
(N times each), scores outputs with code + an LLM judge, and writes results.

Usage:
    export GROQ_API_KEY=gsk_...   # or ANTHROPIC_API_KEY=sk-ant-...
    python run_eval.py                 # full run
    python run_eval.py --dry-run       # test pipeline with fake outputs (no API)
    python run_eval.py --runs 1 --no-judge
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
CATEGORIES = ["billing", "technical", "refund", "other"]
URGENCIES = {"low", "medium", "high"}
METRICS = ["json_valid", "category", "urgency_valid", "length_ok",
           "injection_safe", "professional", "language_match"]

JUDGE_SYSTEM = """You grade customer-support replies. Return ONLY JSON: {"professional": true|false}
professional = polite, calm tone; no blame, sarcasm or profanity; does NOT claim an action (refund, fix) is already done.
Politely declining an inappropriate or manipulative request IS professional."""

# Words used to detect Roman Urdu replies (code-based language check)
UR_MARKERS = {"aap", "hai", "hain", "ke", "ki", "ko", "kya", "mein", "nahi", "kar", "karen", "karein",
              "hamari", "hum", "shukriya", "maazrat", "maafi", "jaldi", "bata", "sakte", "sakti", "lein",
              "dein", "apna", "apni", "aapka", "aapki", "paise", "wapis", "mehrbani", "pareshani"}

_client = None
DRY_RUN = False


# ---------------------------------------------------------------- LLM calls
def provider_info():
    """Auto-detect provider from the key prefix: gsk_ -> Groq, otherwise Anthropic."""
    key = (os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    return ("groq" if key.startswith("gsk_") else "anthropic"), key


DEFAULT_MODELS = {  # (eval model, judge model)
    "groq": ("llama-3.1-8b-instant", "llama-3.3-70b-versatile"),
    "anthropic": ("claude-haiku-5-5", "claude-haiku-5-5"),
}


def groq_models(key):
    import requests
    r = requests.get("https://api.groq.com/openai/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code == 401:
        raise SystemExit("401: invalid API key.")
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def pick_groq_models(key):
    """Pick an eval model (small) and a judge model (big) from what the account can use."""
    ids = groq_models(key)
    bad = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard")
    chat = [i for i in ids if not any(b in i.lower() for b in bad)]
    reasoning = ("gpt-oss", "qwen3", "deepseek-r1", "r1-", "reason")
    chat = [i for i in chat if not any(x in i.lower() for x in reasoning)] or chat
    if not chat:
        raise SystemExit(f"No chat models found. Available: {ids}")

    def first(prefs):
        for pref in prefs:
            for i in chat:
                if pref in i.lower():
                    return i

    small = first(["llama-3.1-8b", "8b", "gemma", "20b", "scout", "qwen"]) or chat[0]
    big = first(["llama-3.3-70b", "70b", "120b", "maverick", "versatile"]) or small
    return small, big


def get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=provider_info()[1])
    return _client


def llm(system, user, model, max_tokens=500, temperature=0.7):
    if DRY_RUN:
        return fake_llm(system, user)
    provider, key = provider_info()
    for attempt in range(8):
        try:
            if provider == "groq":
                import requests
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]},
                    timeout=90)
                if r.status_code == 401:
                    raise SystemExit("401: invalid API key. Check your key/secret.")
                if r.status_code == 429:  # rate limit: wait and retry
                    wait = float(r.headers.get("retry-after", 2 ** attempt * 2))
                    time.sleep(min(wait + 1, 60))
                    continue
                if r.status_code == 404:
                    raise SystemExit(f"404 from Groq: {r.text[:300]}\n"
                                     f"Model '{model}' not available. Run with --model <id> --judge-model <id>.")
                if r.status_code >= 400:
                    raise Exception(f"{r.status_code}: {r.text[:200]}")
                return clean(r.json()["choices"][0]["message"]["content"])
            r = get_client().messages.create(
                model=model, max_tokens=max_tokens, system=system, temperature=temperature,
                messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in r.content if b.type == "text")
        except SystemExit:
            raise
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                raise SystemExit(f"Invalid API key: {e}")
            wait = min(2 ** attempt, 30)
            print(f"  API error ({str(e)[:120]}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return ""


def fake_llm(system, user):
    """Offline stand-in so the pipeline can be tested without an API key."""
    if system.startswith("You grade"):
        if random.random() < .2:
            return "I think this reply looks fine."
        return json.dumps({"professional": random.random() > .1})
    cat = random.choice(CATEGORIES)
    if "valid JSON" in system:
        return json.dumps({"category": cat, "urgency": "medium", "reply": "Thanks, we will review this."})
    return f"Category: {cat}. Sorry about that, we'll help."


# ------------------------------------------------------------------ scoring
def clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def truthy(x):
    return x is True or str(x).strip().lower() == "true"


def parse_json(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def extract_category(text, parsed):
    """Use the JSON field if present; otherwise fall back to the first category
    word in free text so unstructured prompts (v1) are judged fairly."""
    if parsed and str(parsed.get("category", "")).lower() in CATEGORIES:
        return str(parsed["category"]).lower()
    low = text.lower()
    hits = [(m.start(), c) for c in CATEGORIES[:3]
            for m in [re.search(rf"\b{c}", low)] if m]
    return min(hits)[1] if hits else "other"


def get_reply(text, parsed):
    if parsed and "reply" in parsed:
        return str(parsed["reply"])
    return text


def judge(case, reply, model):
    if not reply.strip():
        return {"professional": 0, "judge_failed": 0}
    raw = llm(JUDGE_SYSTEM, f"Customer message:\n{case['input']}\n\nReply:\n{reply}",
              model, 400, temperature=0)
    m = re.search(r"\{.*?\}", clean(raw), re.S)
    d = None
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = None
    if not d or "professional" not in d:
        return {"professional": None, "judge_failed": 1}
    return {"professional": int(truthy(d["professional"])), "judge_failed": 0}


def language_ok(case, reply):
    """Code-based check: reply must be in Latin script; Roman Urdu cases must also use Roman Urdu words."""
    letters = [c for c in reply if c.isalpha()]
    if not letters:
        return 0
    latin = sum(1 for c in letters if c.isascii() or "\u00c0" <= c <= "\u024f")
    if latin / len(letters) < 0.9:
        return 0  # e.g. Devanagari/Arabic script instead of Roman script
    if case.get("language") == "roman_urdu":
        words = set(re.findall(r"[a-z]+", reply.lower()))
        return int(len(words & UR_MARKERS) >= 2)
    return 1


def score(case, text, use_judge, judge_model):
    text = clean(text)
    parsed = parse_json(text)
    exp = case["expected_category"]
    exp = exp if isinstance(exp, list) else [exp]
    reply = get_reply(text, parsed)
    words = len(reply.split())
    s = {
        "json_valid": int(parsed is not None and {"category", "urgency", "reply"} <= set(parsed)),
        "category": int(extract_category(text, parsed) in exp),
        "urgency_valid": int(parsed is not None and str(parsed.get("urgency", "")).lower() in URGENCIES),
        "length_ok": int(0 < words <= 80),
        "injection_safe": int(not any(m in text.lower() for m in case.get("must_not_contain", []))),
        "professional": None, "language_match": language_ok(case, reply),
    }
    if use_judge:
        s.update(judge(case, reply, judge_model))
    s["words"] = words
    s["predicted_category"] = extract_category(text, parsed)
    return s


# --------------------------------------------------------------------- main
def run_one(args_tuple):
    version, system, case, run, a = args_tuple
    text = llm(system, case["input"], a.model, 1500)
    if not text.strip():
        text = llm(system, case["input"], a.model, 1500)
    s = score(case, text, not a.no_judge, a.judge_model)
    return {"version": version, "model": a.model, "case_id": case["id"], "type": case["type"],
            "run": run, "input": case["input"], "output": text, **s}


def summarize(rows, use_judge):
    metrics = [m for m in METRICS if use_judge or m != "professional"]
    out = []
    for v in sorted({r["version"] for r in rows}):
        vr = [r for r in rows if r["version"] == v]
        d = {"version": v}
        for m in metrics:
            vals = [r[m] for r in vr if r.get(m) is not None]
            d[m] = round(100 * sum(vals) / len(vals), 1) if vals else None
        ok = [d[m] for m in metrics if d[m] is not None]
        d["overall_%"] = round(sum(ok) / len(ok), 1)
        by_case = {}
        for r in vr:
            by_case.setdefault(r["case_id"], set()).add(r["predicted_category"])
        d["consistency_%"] = round(100 * sum(len(s) == 1 for s in by_case.values()) / len(by_case), 1)
        if use_judge:
            d["judge_fail_%"] = round(100 * sum(r.get("judge_failed", 0) for r in vr) / len(vr), 1)
        out.append(d)
    return out


def write_results(rows, summary, outdir):
    outdir.mkdir(exist_ok=True)
    (outdir / "outputs.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    cols = ["version", "case_id", "type", "run", "predicted_category", "words"] + METRICS + ["judge_failed"]
    with open(outdir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    head = list(summary[0])
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in summary]
    (outdir / "summary.md").write_text("\n".join(md) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([r["version"] for r in summary], [r["overall_%"] for r in summary],
                      color=["#9aa5b1", "#5b8def", "#1f9d6b"][:len(summary)])
        ax.bar_label(bars, fmt="%.1f%%")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Overall score (%)")
        ax.set_title("Prompt version comparison")
        fig.tight_layout()
        fig.savefig(outdir / "chart.png", dpi=150)
    except Exception as e:
        print("chart skipped:", e)
    return "\n".join(md)


def main():
    global DRY_RUN
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("EVAL_MODEL"))
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    p.add_argument("--runs", type=int, default=3, help="repeats per case (consistency)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fill-empty", action="store_true",
                   help="regenerate only empty outputs in results/outputs.json (use the SAME --model as the original run), then re-score")
    p.add_argument("--rescore", action="store_true",
                   help="re-score results/outputs.json without regenerating outputs")
    a = p.parse_args()
    DRY_RUN = a.dry_run
    provider, _ = provider_info()
    dm, dj = DEFAULT_MODELS[provider]
    if provider == "groq" and not DRY_RUN and not (a.model and a.judge_model):
        dm, dj = pick_groq_models(provider_info()[1])
    a.model = a.model or dm
    a.judge_model = a.judge_model or dj
    a.workers = a.workers or (2 if provider == "groq" else 6)
    if not DRY_RUN and not provider_info()[1]:
        sys.exit("No API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY.")

    cases = json.loads((ROOT / "eval_set.json").read_text())
    prompts = {f.stem: f.read_text() for f in sorted((ROOT / "prompts").glob("v*.txt"))}
    if a.rescore or a.fill_empty:
        rows = json.loads((ROOT / "results" / "outputs.json").read_text())
        cmap = {c["id"]: c for c in cases}
        if a.fill_empty:
            todo = [r for r in rows if not clean(r["output"])]
            print(f"Regenerating {len(todo)} empty outputs with model={a.model}")

            def fill(r):
                for _ in range(3):
                    t = llm(prompts[r["version"]], r["input"], a.model, 1500)
                    if t.strip():
                        r["output"], r["model"] = t, a.model
                        break
                return r
            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(fill, todo))
            print(f"Still empty after retry: {sum(1 for r in rows if not clean(r['output']))}")
        print(f"Re-scoring {len(rows)} saved outputs | judge={a.judge_model}")

        def rs(r):
            r.update(score(cmap[r["case_id"]], r["output"], not a.no_judge, a.judge_model))
            return r
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(rs, rows))
    else:
        jobs = [(v, s, c, r, a) for v, s in prompts.items() for c in cases for r in range(a.runs)]
        print(f"{len(prompts)} prompts x {len(cases)} cases x {a.runs} runs = {len(jobs)} calls | "
              f"provider={provider} model={a.model} judge={a.judge_model}")
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(run_one, jobs))

    summary = summarize(rows, not a.no_judge)
    table = write_results(rows, summary, ROOT / "results")
    print("\n" + table)
    best = max(summary, key=lambda r: r["overall_%"])
    print(f"\nBest version: {best['version']} ({best['overall_%']}%)")


if __name__ == "__main__":
    main()
