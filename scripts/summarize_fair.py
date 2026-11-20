"""Build results/fair/summary.{md,csv} from the per-arm evaluation JSONs and training logs (no torch needed)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fair"
ARMS = ["baseline_b0", "baseline_continue", "caff_equal", "caff_static", "caff_learned"]
SETS = [("512", "512-query dev set"), ("2048", "2,048-query holdout"), ("1536", "1,536 queries not used for selection")]


def load(arm, tag):
    path = RES / f"{arm}_{tag}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def last_log(arm):
    path = ROOT / "checkpoints" / "fair" / f"{arm}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []
    return rows[-1] if rows else None


def main() -> int:
    lines, csv_rows = [], []
    for tag, title in SETS:
        lines += [f"### {title}", "", "| Arm | R@1 | R@5 | R@10 | valid queries |", "|---|---|---|---|---|"]
        for arm in ARMS:
            rec = load(arm, tag)
            if rec is None:
                lines.append(f"| {arm} | not run | | | |")
                continue
            m = rec["query_metrics"]; r = m["recall"]
            lines.append(f"| {arm} | {100*r['R@1']:.2f} | {100*r['R@5']:.2f} | {100*r['R@10']:.2f} | {m['valid_queries']} |")
            csv_rows.append([arm, tag, r["R@1"], r["R@5"], r["R@10"], m["valid_queries"]])
        lines.append("")
    lines += ["### Provenance", "", "| Arm | trained params (total) | breakdown | gate mode | config | init ckpt sha256 | checkpoint sha256 | mean gate weights [global, local] |", "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rec = load(arm, "2048") or load(arm, "512")
        if rec is None:
            continue
        log = last_log(arm) or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm,
            (rec.get("trained_parameter_counts") or {}).get("trainable", "no training (reference)" if arm == "baseline_b0" else "n/a"),
            rec.get("trained_parameter_breakdown") or "n/a", rec.get("caff_gate_mode"),
            rec.get("training_config_path") or rec.get("config"),
            (rec.get("training_init_checkpoint_sha256") or "")[:12], rec["checkpoint_sha256"][:12],
            log.get("mean_gate_weights")))
    eq, bc = load("caff_equal", "2048"), load("baseline_continue", "2048")
    if eq and bc:
        d = 100 * (eq["query_metrics"]["recall"]["R@1"] - bc["query_metrics"]["recall"]["R@1"])
        lines += ["", f"Sanity check: caff_equal minus baseline_continue R@1 on the 2,048 set = {d:+.2f} pt "
                      "(should be ~0: the equal gate is a no-op with identical data order and seed)."]
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (RES / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["arm", "eval_set", "R@1", "R@5", "R@10", "valid_queries"]); w.writerows(csv_rows)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
