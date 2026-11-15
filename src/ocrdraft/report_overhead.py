"""`ocrdraft report`: overhead-study raw cells -> results/overhead/summary.json. No GPU."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import overhead as oh

TAUS = [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7]
# Per-configuration files written next to the cells by overhead_study.run_group.
NOT_CELLS = {"startup.json", "command.json", "startup_failed.json", "run_failed.json"}


def load_cells(raw: Path) -> dict[str, dict[str, dict]]:
    """{config: {cell: run_cell result}} from <raw>/<config>/<cell>.json."""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(raw.glob("*/*.json")):
        if f.name not in NOT_CELLS:
            out.setdefault(f.parent.name, {})[f.stem] = json.loads(f.read_text())
    return out


def failures(raw: Path) -> dict[str, str]:
    """Configurations that failed to start or failed while running (see overhead_study.run_group)."""
    return {
        f.parent.name: json.loads(f.read_text()).get("error", "")[:300]
        for pattern in ("*/startup_failed.json", "*/run_failed.json")
        for f in sorted(raw.glob(pattern))
    }


def fixed_summaries(cells: dict[str, dict]) -> dict[int, oh.CellSummary]:
    return {
        r["concurrency"]: oh.summarize(r) for name, r in cells.items() if name.startswith("fixed-")
    }


def natural_stats(base: dict, other: dict) -> dict:
    """Matched-page comparison of two natural-length runs (same pages, same concurrency)."""
    b = {r["page"]: r for r in base["requests"] if r.get("ok")}
    o = {r["page"]: r for r in other["requests"] if r.get("ok")}
    common = sorted(set(b) & set(o))
    if not common:
        return {}
    exact = sum(b[p]["text"] == o[p]["text"] for p in common)
    return {
        "pages": len(common),
        "page_speedup": sum(b[p]["e2e"] for p in common) / sum(o[p]["e2e"] for p in common),
        "tau": oh.mean_acceptance(other.get("spec")),
        "exact_match": exact / len(common),
    }


def build(raw: Path, config_path: Path) -> str:
    cfg = json.loads(config_path.read_text())
    cells = load_cells(raw)
    out_dir = raw.parent
    base_fixed = fixed_summaries(cells["baseline"])
    nat16 = cells["baseline"]["natural-c16"]
    ok16 = [r for r in nat16["requests"] if r.get("ok")]
    natural_tokens = statistics.median(r["tokens"] for r in ok16)
    decode_share = statistics.median(
        (r["e2e"] - r["ttft"]) / r["e2e"] for r in ok16 if r.get("e2e")
    )
    capped = sum(r["finish_reason"] == "length" for r in ok16) / len(ok16)

    ngram = {}
    for c in (1, 16):
        key = f"natural-c{c}"
        if "ngram" in cells and key in cells["ngram"]:
            ngram[c] = natural_stats(cells["baseline"][key], cells["ngram"][key])
    ngram_speedup = {c: v["page_speedup"] for c, v in ngram.items() if v}

    drafters = {
        name: fixed_summaries(c) for name, c in cells.items() if name not in ("baseline", "ngram")
    }
    analytic = {r["variant"]: r for r in oh.analytic_table(oh.LIGHTONOCR_2_1B, cfg["variants"])}
    train_cost = {k: v["draft_params_read_m"] for k, v in analytic.items()}
    decision = oh.decide(
        base_fixed, drafters, natural_tokens, ngram_speedup, cfg["decision"], train_cost
    )

    rows = []
    for name, by_c in drafters.items():
        for c, s in sorted(by_c.items()):
            b = base_fixed[c]
            rows.append(
                {
                    "candidate": name,
                    "concurrency": c,
                    "tau_measured": round(s.tau, 3),
                    "step_ms": round(s.step * 1e3, 3),
                    "baseline_tpot_ms": round(b.tpot * 1e3, 3),
                    "cost_ratio": round(s.step / b.tpot, 3),
                    "ttft_ms": round(s.ttft * 1e3, 1),
                    "baseline_ttft_ms": round(b.ttft * 1e3, 1),
                    "page_speedup": {
                        str(t): round(oh.page_speedup(b, s, t, natural_tokens), 3)
                        for t in TAUS
                        if t <= max_tau(name)
                    },
                }
            )
    summary = {
        "natural_tokens_median": natural_tokens,
        "decode_share_of_page_latency_c16": round(decode_share, 3),
        "natural_hit_max_tokens_frac": round(capped, 3),
        "baseline": {
            str(c): {"tpot_ms": round(s.tpot * 1e3, 3), "ttft_ms": round(s.ttft * 1e3, 1)}
            for c, s in sorted(base_fixed.items())
        },
        "ngram": {str(c): v for c, v in ngram.items()},
        "analytic": list(analytic.values()),
        "measured": rows,
        "startup_failures": failures(raw),
        "decision": decision,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    choice = decision["choice"] or "none (no candidate is worth training)"
    return f"chosen drafter: {choice} -> {out_dir / 'summary.json'}"


def max_tau(candidate: str) -> float:
    """A step can emit at most K drafted tokens plus the bonus token."""
    return int(candidate.rsplit("@", 1)[1]) + 1 if "@" in candidate else float("inf")
