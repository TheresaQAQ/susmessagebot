"""Average eval summaries across multiple runs of the same prompt version."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path("eval_results")
METRICS = [
    "accuracy",
    "ban_precision",
    "ban_recall",
    "ban_f1",
    "fp",
    "fn",
    "tp",
    "tn",
    "avg_seconds",
    "correct",
]


def parse_runs(spec: str) -> list[int]:
    """Accept '2-5' or '1,2,3' or '1'."""
    runs: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            runs.extend(range(int(a), int(b) + 1))
        else:
            runs.append(int(part))
    return sorted(set(runs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--runs", required=True, help="e.g. 2-5 or 1,2,3,4")
    args = parser.parse_args()

    prompt_id = args.prompt_version
    runs = parse_runs(args.runs)
    by_model: dict[str, list[dict]] = defaultdict(list)

    for run in runs:
        run_dir = RESULTS_ROOT / f"prompt_{prompt_id}" / f"run{run}"
        if not run_dir.exists():
            raise SystemExit(f"missing run dir: {run_dir}")
        for p in sorted(run_dir.glob("*.summary.json")):
            s = json.loads(p.read_text(encoding="utf-8"))
            by_model[s["model"]].append(s)

    out_dir = RESULTS_ROOT / f"prompt_{prompt_id}" / f"avg_run{runs[0]}-{runs[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    avg_rows = []
    for model, items in sorted(by_model.items(), key=lambda x: x[0]):
        found_runs = sorted(i.get("run", -1) for i in items)
        if len(items) != len(runs):
            print(f"WARN {model}: expected {len(runs)} runs, got {found_runs}")
        avg = {
            "prompt_version": prompt_id,
            "runs": runs,
            "n_runs": len(items),
            "model": model,
            "provider": "siliconflow",
        }
        for m in METRICS:
            vals = [float(i[m]) for i in items if m in i]
            avg[f"mean_{m}"] = round(sum(vals) / len(vals), 4) if vals else None
            if vals:
                avg[f"min_{m}"] = round(min(vals), 4)
                avg[f"max_{m}"] = round(max(vals), 4)
        # per-run accuracy list for transparency
        avg["per_run_accuracy"] = [i.get("accuracy") for i in sorted(items, key=lambda x: x.get("run", 0))]
        avg["per_run_fp"] = [i.get("fp") for i in sorted(items, key=lambda x: x.get("run", 0))]
        avg["per_run_fn"] = [i.get("fn") for i in sorted(items, key=lambda x: x.get("run", 0))]
        avg_rows.append(avg)
        path = out_dir / f"{model.replace('/', '_')}.avg.json"
        path.write_text(json.dumps(avg, ensure_ascii=False, indent=2), encoding="utf-8")

    # sort by mean accuracy desc
    avg_rows.sort(key=lambda x: (-(x.get("mean_accuracy") or 0), x["model"]))

    lines = [
        f"# 多轮平均 — prompt `{prompt_id}` / run{runs[0]}–run{runs[-1]}（共 {len(runs)} 轮）",
        "",
        "| 模型 | 平均正确率 | 平均FP | 平均FN | 平均BAN F1 | 均耗时 | 各轮Acc |",
        "|------|------------|--------|--------|------------|--------|---------|",
    ]
    for s in avg_rows:
        accs = "/".join(f"{a:.0%}" for a in s["per_run_accuracy"])
        lines.append(
            f"| `{s['model']}` | {s['mean_accuracy']:.1%} | {s['mean_fp']:.2f} | "
            f"{s['mean_fn']:.2f} | {s['mean_ban_f1']:.1%} | {s['mean_avg_seconds']:.2f}s | {accs} |"
        )
    lines += [
        "",
        f"明细 JSON：`{out_dir.as_posix()}/`",
        "",
    ]
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # also drop a copy at prompt root for quick find
    (RESULTS_ROOT / f"prompt_{prompt_id}" / f"AVERAGE_run{runs[0]}-{runs[-1]}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
