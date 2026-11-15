"""Trained-drafter report: DSpark acceptance, measured speedup, output agreement, run costs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import overhead as oh
from .report_overhead import load_cells


def eval_categories(cfg: dict) -> dict[str, str]:
    """Rebuild page id -> category exactly as data.render_pages chose them (same seed)."""
    from . import data

    chosen = data.bench_sample(cfg, cfg["eval_pages"], cfg["page_seed"])
    return {f"{i:03d}": data.category_of(p) for i, p in enumerate(chosen)}


def matched(base: dict, spec: dict) -> list[tuple[dict, dict]]:
    b = {r["page"]: r for r in base["requests"] if r.get("ok")}
    s = {r["page"]: r for r in spec["requests"] if r.get("ok")}
    return [(b[p], s[p]) for p in sorted(set(b) & set(s))]


def cell_summary(base: dict, spec: dict, cats: dict[str, str]) -> dict:
    pairs = matched(base, spec)
    # tau per category is a ratio of sums over its pages (1 + accepted / drafts, as for the cell
    # tau), so long pages weigh by their draft steps instead of each page counting once.
    per_cat: dict[str, dict[str, float]] = defaultdict(
        lambda: {"num_drafts": 0, "num_accepted_tokens": 0}
    )
    for _, s in pairs:
        m = (s.get("spec_metrics") or {}).get("speculative_decoding")
        if m:
            acc = per_cat[cats.get(s["page"], "?")]
            acc["num_drafts"] += m["num_spec_steps"]
            acc["num_accepted_tokens"] += m["num_accepted_draft_tokens"]
    return {
        "pages": len(pairs),
        "tau": round(oh.mean_acceptance(spec.get("spec")), 3),
        "page_latency_speedup": round(
            sum(b["e2e"] for b, _ in pairs) / sum(s["e2e"] for _, s in pairs), 3
        ),
        "throughput_speedup": round(spec["output_tokens_per_s"] / base["output_tokens_per_s"], 3),
        "exact_match": round(sum(b["text"] == s["text"] for b, s in pairs) / len(pairs), 3),
        "tau_by_category": {c: round(oh.mean_acceptance(v), 2) for c, v in sorted(per_cat.items())},
        "median_tpot_ms": {
            "baseline": round(1e3 * statistics.median(b["tpot"] for b, _ in pairs if b["tpot"]), 3),
            "spec": round(1e3 * statistics.median(s["tpot"] for _, s in pairs if s["tpot"]), 3),
        },
    }


def compare_configs(cells: dict, cats: dict[str, str]) -> dict:
    """cell_summary of every spec config against no-spec, for the cells both have."""
    base = cells["baseline"]
    return {
        name: {
            cell: cell_summary(base[cell], cells[name][cell], cats)
            for cell in sorted(cells[name])
            if cell in base
        }
        for name in sorted(cells)
        if name != "baseline"
    }


def per_position(spec: dict) -> list[float]:
    """Acceptance probability at draft position i, from per-request histograms."""
    hist = None
    for r in spec["requests"]:
        m = (r.get("spec_metrics") or {}).get("speculative_decoding")
        if not m:
            continue
        h = m["acceptance_histogram"]
        hist = h if hist is None else [a + b for a, b in zip(hist, h)]
    if not hist:
        return []
    total = sum(hist)
    # hist[j] = steps that accepted exactly j drafts; P(accept >= i+1) for each position.
    return [round(sum(hist[i + 1 :]) / total, 3) for i in range(len(hist) - 1)]


def build(raw: Path, config_path: Path, overhead_summary: Path | None = None) -> str:
    cfg = json.loads(config_path.read_text())
    cells = load_cells(raw / "eval")
    cats = eval_categories(cfg)
    base = cells["baseline"]
    rows = compare_configs(cells, cats)
    for name, r in rows.items():
        r["position_acceptance"] = per_position(cells[name]["natural-c16"])
    agree = {
        "control: no-spec c1 vs c16": agreement(base["natural-c1"], base["natural-c16"]),
        "control: no-spec c16 vs c32": agreement(base["natural-c16"], base["natural-c32"]),
    }
    for name in sorted(n for n in cells if n != "baseline"):
        for cell in sorted(cells[name]):
            if cell in base:
                agree[f"no-spec vs {name} {cell}"] = agreement(base[cell], cells[name][cell])
    # `ocrdraft eval` writes into `raw`; the published runs were fetched into sibling trees.
    sampled = sampled_results(_tree(raw, "eval_sampled", "sampled_raw"), cats)
    throughput = throughput_results(cells, _tree(raw, "eval_throughput", "throughput_raw"))
    summary = {
        "note": cfg["drafter_pipeline"].get("report_note"),
        "results": rows,
        "agreement": agree,
        "sampled": sampled,
        "throughput": throughput,
        "gpu_util": gpu_util(raw / "state/gpu_dmon.log"),
    }
    if overhead_summary and overhead_summary.exists():
        summary["predicted_vs_measured"] = predicted(
            overhead_summary, rows, cfg["dspark"]["name"], cfg["dspark"].get("step_cost_proxy")
        )
    out = raw.parent / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    return str(out)


def _tree(raw: Path, part: str, legacy: str) -> Path:
    return raw if (raw / part).is_dir() else raw.parent / legacy


def same_pages(manifest: Path, reference: Path) -> dict:
    """PNG sha256 agreement between a run's eval-page manifest and a reference render."""
    a = {p["id"]: p["sha256"] for p in json.loads(manifest.read_text())["pages"]}
    b = {p["id"]: p["sha256"] for p in json.loads(reference.read_text())["pages"]}
    common = sorted(set(a) & set(b))
    return {"compared": len(common), "identical": sum(a[i] == b[i] for i in common)}


def sampled_results(raw: Path, cats: dict[str, str]) -> dict:
    """`ocrdraft eval`, sampled mode: the same cells with the model card's sampling (no exact match)."""
    if not (raw / "eval_sampled").is_dir():
        return {}
    cells = load_cells(raw / "eval_sampled")
    out = compare_configs(cells, cats)
    for by_cell in out.values():
        for s in by_cell.values():
            del s["exact_match"]  # sampled outputs differ run to run
    sampling = next(iter(next(iter(cells.values())).values())).get("sampling")
    res = {"sampling": sampling, "results": out}
    ref = Path("data/overhead_pages/manifest.json")
    if (raw / "eval_pages_manifest.json").exists() and ref.exists():
        res["pages_vs_overhead_render"] = same_pages(raw / "eval_pages_manifest.json", ref)
    return res


def steady_state(cell: dict, ramp_s: float = 5.0) -> dict:
    """Output tok/s while every slot was busy, from a rebuilt dispatch timeline.

    bench.run_cell's `concurrency` workers take requests in index order, each starting the next
    as soon as its previous one ends, so start times follow from the measured e2e alone. The
    window runs from `ramp_s` after the first token until the last request starts decoding (the
    queue is empty after that); each request's tokens are spread evenly over its decode span.
    `makespan_s` should match the measured wall time, which checks the reconstruction.
    """
    import heapq

    reqs = sorted((q for q in cell["requests"] if q.get("ok")), key=lambda q: q["index"])
    free = [0.0] * cell["concurrency"]
    spans = []
    for q in reqs:
        s = heapq.heappop(free)
        heapq.heappush(free, s + q["e2e"])
        spans.append((s + q["ttft"], s + q["e2e"], q["tokens"]))
    t0 = min(a for a, _, _ in spans) + ramp_s
    t1 = max(a for a, _, _ in spans)
    tok = sum(
        n * (min(b, t1) - max(a, t0)) / (b - a)
        for a, b, n in spans
        if b > a and min(b, t1) > max(a, t0)
    )
    return {
        "tok_s": round(tok / (t1 - t0)) if t1 > t0 else None,
        "window_s": round(t1 - t0, 1),
        "makespan_s": round(max(b for _, b, _ in spans), 1),
    }


def throughput_point(cell: dict) -> dict:
    ok = [r for r in cell["requests"] if r.get("ok")]
    tp = [r["tpot"] for r in ok if r.get("tpot")]
    ss = steady_state(cell) if cell["concurrency"] > 1 else {"tok_s": None}
    return {
        "output_tok_s": round(cell["output_tokens_per_s"]),
        "steady_state_tok_s": ss["tok_s"],
        "steady_window_s": ss.get("window_s"),
        "pages_s": round(len(ok) / cell["wall_s"], 2),
        "requests": len(ok),
        "tau": round(oh.mean_acceptance(cell.get("spec")), 3),
        "median_tpot_ms": round(1e3 * statistics.median(tp), 2) if tp else None,
    }


def throughput_results(greedy: dict, raw: Path) -> dict:
    """Greedy throughput by concurrency: the c=1/16/32 latency cells plus `evaluate throughput`."""
    if not (raw / "eval_throughput").is_dir():
        return {}
    cells = {k: dict(v) for k, v in greedy.items()}
    for name, cs in load_cells(raw / "eval_throughput").items():
        cells.setdefault(name, {}).update(cs)
    out: dict = {}
    for name, cs in cells.items():
        pts = {
            int(c.rsplit("c", 1)[1]): throughput_point(v)
            for c, v in cs.items()
            if c.startswith("natural-c")
        }
        out[name] = dict(sorted(pts.items()))
    base = out.get("baseline", {})
    for name, pts in out.items():
        for c, pt in pts.items():
            if c in base:
                pt["vs_nospec_same_c"] = round(pt["output_tok_s"] / base[c]["output_tok_s"], 3)

    def best(p: dict) -> int:
        return p["steady_state_tok_s"] or p["output_tok_s"]

    peak = {n: max(best(p) for p in pts.values()) for n, pts in out.items() if pts}
    return {
        "by_concurrency": out,
        "peak_output_tok_s": peak,
        "peak_vs_nospec_peak": {n: round(v / peak["baseline"], 3) for n, v in peak.items()}
        if "baseline" in peak
        else {},
    }


def agreement(a: dict, b: dict) -> dict:
    """Exact-match rate and, for differing pages, character similarity of two greedy runs."""
    import difflib

    pairs = matched(a, b)
    diff = [(x["text"], y["text"]) for x, y in pairs if x["text"] != y["text"]]
    sims = [difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() for x, y in diff]
    return {
        "pages": len(pairs),
        "exact_match": round(1 - len(diff) / len(pairs), 3),
        "differing_char_sim_median": round(statistics.median(sims), 4) if sims else None,
    }


def predicted(overhead_summary: Path, rows: dict, variant: str, proxy: str | None = None) -> dict:
    """The overhead study's step-cost model evaluated at the measured tau, for the trained `variant`.

    Overhead-study rows timed with exactly that architecture (`<variant>@<K>`) are used. Failing
    that, an explicitly configured `proxy` variant (configs/run.json dspark.step_cost_proxy)
    is used and named in `step_cost_from` and `note`: its step cost differs from the trained
    variant's, so the prediction is approximate. With neither, `cells` is empty and `note`
    says why. The page prediction uses oh.page_speedup at the overhead study's median page length.
    """
    s = json.loads(overhead_summary.read_text())
    timed = {m["candidate"].rpartition("@")[0] for m in s["measured"]}
    source = variant if variant in timed else proxy if proxy in timed else None
    out: dict = {"variant": variant, "step_cost_from": source, "cells": {}}
    tokens = s["natural_tokens_median"]
    for m in s["measured"]:
        name, _, k = m["candidate"].rpartition("@")
        c = m["concurrency"]
        cell = rows.get(f"dspark@{k}", {}).get(f"natural-c{c}")
        if name != source or not cell:
            continue
        tau = cell["tau"]
        # The overhead study's drafters accept nothing (tau 1), so their TPOT is the step cost.
        base = oh.CellSummary(
            "baseline", c, m["baseline_tpot_ms"], m["baseline_ttft_ms"], tokens, 1.0, 0
        )
        spec = oh.CellSummary(name, c, m["step_ms"], m["ttft_ms"], tokens, 1.0, 0)
        out["cells"][f"K={k} c{c}"] = {
            "measured_tau": tau,
            "predicted_decode_speedup": round(tau / m["cost_ratio"], 3),
            "predicted_page_speedup": round(oh.page_speedup(base, spec, tau, tokens), 3),
            "measured_page_speedup": cell["page_latency_speedup"],
        }
    if source is None:
        out["note"] = (
            f"The overhead study did not time {variant} (timed: {', '.join(sorted(timed))}), "
            "so no step-cost prediction is made for it."
        )
    elif source != variant:
        out["note"] = (
            f"The overhead study did not time {variant}; step costs are those of {source}, "
            "a proxy with a different step cost, so the predictions are approximate."
        )
    return out


def gpu_util(path: Path) -> dict:
    if not path.exists():
        return {}
    sm = [
        int(p[2])
        for line in path.read_text().splitlines()
        if not line.startswith("#") and len(p := line.split()) > 3 and p[2].isdigit()
    ]
    return {"samples": len(sm), "mean_sm_pct": round(statistics.fmean(sm), 1) if sm else None}
