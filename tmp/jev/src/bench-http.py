"""Time Jev vs SiliconFlow LLM from HTTP send to full response body. No queue waits inside the timer."""

from __future__ import annotations

import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "gold.json"
POLICY_PATH = ROOT / "src" / "policy.ts"
SAMPLE_INDEXES = [0, 10, 50, 86, 108, 120]

load_dotenv(ROOT / ".env.local")
load_dotenv(Path(r"C:\code\susmessagebot\.env"))


def load_policy() -> tuple[str, dict[str, str]]:
    text = POLICY_PATH.read_text(encoding="utf-8")
    instructions = re.search(r"POLICY_INSTRUCTIONS = `(.*)`;", text, re.S).group(1)
    ban = re.search(r'BAN: "(.*)"', text).group(1)
    safe = re.search(r'SAFE: "(.*)"', text).group(1)
    return instructions, {"BAN": ban, "SAFE": safe}


def load_samples() -> list[dict]:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    by_index = {row["index"]: row for row in gold}
    return [by_index[i] for i in SAMPLE_INDEXES]


def post_json(url: str, headers: dict[str, str], payload: dict, timeout: float = 60.0):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, status, raw


def sleep_outside_timer(seconds: float) -> None:
    time.sleep(seconds)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize(name: str, rows: list[dict]) -> None:
    ms = [row["ms"] for row in rows]
    print(f"\n{name}")
    print(f"  n={len(ms)}  min={min(ms):.0f}  p50={percentile(ms, 50):.0f}  "
          f"p90={percentile(ms, 90):.0f}  mean={statistics.mean(ms):.0f}  max={max(ms):.0f}")
    for row in rows:
        print(
            f"  [{row['index']:03}] {row['ms']:7.0f} ms  http={row['status']}  "
            f"pred={row['pred']}  expect={row['expected']}"
        )


def bench_jev(samples: list[dict]) -> list[dict]:
    key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if not key:
        raise SystemExit("AI_GATEWAY_API_KEY missing")
    instructions, criteria = load_policy()
    questions = {
        "verdict": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    url = "https://ai-gateway.vercel.sh/v1/evaluate"

    def one(text: str) -> tuple[float, int, str]:
        payload = {
            "model": "typesafe-ai/jev",
            "state": {"message": text},
            "questions": questions,
        }
        ms, status, raw = post_json(url, headers, payload)
        pred = "?"
        if status == 200:
            data = json.loads(raw)
            answer = data.get("answers", {}).get("verdict", {})
            pred = answer.get("choice", "?")
        return ms, status, pred

    print("warmup jev (discarded)...")
    one(samples[0]["text"])
    sleep_outside_timer(2.5)

    rows = []
    for sample in samples:
        while True:
            ms, status, pred = one(sample["text"])
            if status == 429:
                print(f"  jev 429, wait 65s, this attempt not counted ({ms:.0f} ms)")
                sleep_outside_timer(65)
                continue
            rows.append({
                "index": sample["index"],
                "expected": sample["expected"],
                "ms": ms,
                "status": status,
                "pred": pred,
            })
            print(f"  jev [{sample['index']:03}] {ms:.0f} ms http={status}")
            sleep_outside_timer(2.5)
            break
    return rows


def bench_llm(samples: list[dict]) -> list[dict]:
    key = (os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise SystemExit("SILICONFLOW_API_KEY missing")
    base = os.environ.get("SILICONFLOW_BASE_URL") or os.environ.get(
        "OPENROUTER_BASE_URL", "https://api.siliconflow.cn/v1"
    )
    model = os.environ.get("SILICONFLOW_MODEL") or os.environ.get(
        "OPENROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct"
    )
    print(f"llm model={model} base={base}")
    instructions, _criteria = load_policy()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    url = base.rstrip("/") + "/chat/completions"

    def one(text: str) -> tuple[float, int, str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"<message>{text}</message>\n只输出 BAN 或 SAFE。"},
            ],
            "max_tokens": 64,
            "temperature": 0,
            "enable_thinking": False,
        }
        ms, status, raw = post_json(url, headers, payload)
        pred = "?"
        if status == 200:
            data = json.loads(raw)
            content = data["choices"][0]["message"].get("content") or ""
            pred = "BAN" if "BAN" in content.upper() else "SAFE" if "SAFE" in content.upper() else content[:20]
        return ms, status, pred

    print("warmup llm (discarded)...")
    one(samples[0]["text"])

    rows = []
    for sample in samples:
        ms, status, pred = one(sample["text"])
        rows.append({
            "index": sample["index"],
            "expected": sample["expected"],
            "ms": ms,
            "status": status,
            "pred": pred,
        })
        print(f"  llm [{sample['index']:03}] {ms:.0f} ms http={status}")
        sleep_outside_timer(0.3)
    return rows


def main() -> None:
    samples = load_samples()
    print(f"samples={len(samples)} indexes={SAMPLE_INDEXES}")
    print("timer = urllib request start -> full response body; retries/waits excluded")
    jev_rows = bench_jev(samples)
    llm_rows = bench_llm(samples)
    summarize("Jev typesafe-ai/jev  HTTP /v1/evaluate", jev_rows)
    summarize("LLM SiliconFlow HTTP /chat/completions", llm_rows)


if __name__ == "__main__":
    main()
