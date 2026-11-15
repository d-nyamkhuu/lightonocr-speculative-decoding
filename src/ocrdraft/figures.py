"""`ocrdraft report`: the README charts (docs/figures/*.svg) from the committed result summaries.

Needs only results/overhead/summary.json and results/dspark/summary.json, so it runs on any
machine. The diagram_*.svg files in the same folder are hand-drawn; both share the fonts and
colours below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .util import read_json  # noqa: E402

# Same fonts as the diagrams; viewers without IBM Plex fall back along the stack.
SANS = ["IBM Plex Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
MONO = ["IBM Plex Mono", "SFMono-Regular", "Menlo", "Consolas", "DejaVu Sans Mono"]
INK, MUTED, AXIS, GRID, REF = "#1b2a3a", "#5b6472", "#b6bec9", "#e4e8ee", "#8a93a3"
# One hue: navy for the main series, a tint of it for a second, grey for the baseline.
MAIN, SECOND, BASE = "#2b4c7e", "#9fb1cc", "#a3abb8"

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.monospace": MONO,
        "svg.fonttype": "none",  # keep text as text: smaller, selectable, sharp
        "svg.hashsalt": "ocrdraft",  # stable ids, so re-runs give identical files
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
    }
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Plex is usually absent

PROXY = "dspark-l3-b8-v32k@7"  # the overhead study's DSpark, the trained drafter's step-cost proxy
USERS = {1: "1 user", 16: "16 users", 32: "32 users"}


def _axes(w: float = 7.5, h: float = 4.2, grid: str = "y"):
    """A white figure: spines left/bottom only, light gridlines along `grid`."""
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=4, color=AXIS)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)


def _labels(ax, xlabel: str | None, ylabel: str | None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10.5)


def _note(ax, x, y, text: str, color: str = INK, **kw) -> None:
    """A monospace value or reference label."""
    ax.text(x, y, text, family="monospace", fontsize=9, color=color, **kw)


def _ref(ax, axis: str, at: float) -> None:
    (ax.axhline if axis == "h" else ax.axvline)(at, color=REF, linestyle="--", lw=1)


def _legend(ax, **kw) -> None:
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, **kw)


def _line(ax, xs, ys, label: str, color: str, ls: str = "-", marker: str | None = "o") -> None:
    """A series labelled at its right end instead of in a legend."""
    ax.plot(xs, ys, color=color, linestyle=ls, marker=marker)
    ax.annotate(
        label,
        (xs[-1], ys[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=9.5,
        color=INK,
    )


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, facecolor="white", metadata={"Date": None})
    plt.close(fig)


# ---------------------------------------------------------------- overhead study


def decision_ranking(overhead: dict, out: Path) -> None:
    """Predicted page speedup of every candidate at its expected τ, against break-even (1×)."""
    d = overhead["decision"]
    rows = sorted(d["ranking"], key=lambda r: r["score"])
    f, ax = _axes(7.5, 4.4, grid="x")
    one, many = ([r["page_speedup"][str(c)] for r in rows] for c in (1, 16))
    ys = range(len(rows))
    ax.hlines(ys, one, many, color=SECOND, lw=2)
    ax.scatter(one, ys, s=46, facecolor="white", edgecolor=MAIN, lw=1.6, zorder=3, label=USERS[1])
    ax.scatter(many, ys, s=46, color=MAIN, zorder=3, label=USERS[16])
    _ref(ax, "v", 1.0)
    _note(ax, 1.015, len(rows) - 0.4, "break-even (1×)", MUTED, va="bottom")
    names = [r["candidate"] + ("  ← chosen" if r["candidate"] == d["choice"] else "") for r in rows]
    ax.set_yticks(ys, names)
    ax.set_ylim(-0.75, len(rows) + 0.1)
    ax.set_xlim(0.9, None)
    _labels(ax, "predicted page speedup at the expected τ (×)", None)
    _title(ax, "Every candidate predicted above break-even")
    _legend(ax, loc="lower right", handletextpad=0.3)
    _save(f, out, "decision_ranking.svg")


def step_cost(overhead: dict, out: Path) -> None:
    """One speculative step relative to one normal step (random drafters, tau = 1)."""
    show = [
        (PROXY, "DSpark (chosen)", MAIN, "-"),
        ("eagle3-l1-v32k@3", "EAGLE-3", BASE, "-"),
        ("dflash2-l4-b16@15", "DFlash2 (16-token block)", BASE, "--"),
    ]
    f, ax = _axes(7.5, 3.9)
    for cand, label, color, ls in show:
        pts = sorted(
            (m["concurrency"], m["cost_ratio"])
            for m in overhead["measured"]
            if m["candidate"] == cand
        )
        _line(ax, *zip(*pts), label, color, ls)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 32], ["1", "4", "16", "32"])
    ax.set_xlim(0.85, 70)
    ax.set_ylim(1, 2.2)
    _labels(ax, "concurrent requests", "speculative step / normal step")
    _title(ax, "Overhead study: cost of one speculative step")
    _save(f, out, "step_cost.svg")


def predicted_speedup(overhead: dict, out: Path) -> None:
    cells = {m["concurrency"]: m for m in overhead["measured"] if m["candidate"] == PROXY}
    tokens = overhead["natural_tokens_median"]
    taus = [1 + i / 20 for i in range(141)]
    f, ax = _axes()
    for c, color in ((16, MAIN), (1, SECOND)):
        m = cells[c]
        base = m["baseline_ttft_ms"] + tokens * m["baseline_tpot_ms"]
        ys = [base / (m["ttft_ms"] + tokens * m["step_ms"] / t) for t in taus]
        _line(ax, taus, ys, USERS[c], color, marker=None)
    _ref(ax, "h", 1.0)
    _note(ax, 8.95, 1.06, "break-even", MUTED, ha="right")
    _ref(ax, "v", 5.4)
    _note(ax, 5.5, 0.4, "trained drafter\nτ = 5.4", MUTED)
    ax.set_xlim(1, 9)
    ax.set_xticks(range(1, 9))
    ax.set_ylim(bottom=0)
    _labels(ax, "acceptance length τ", "predicted page speedup (×)")
    _title(ax, "Predicted page speedup vs acceptance length")
    _save(f, out, "predicted_speedup_vs_tau.svg")


# ---------------------------------------------------------------- trained drafter


def page_speedup(summary: dict, out: Path) -> None:
    r = summary["results"]
    f, ax = _axes(7.5, 3.9)
    w, xs, ticks = 0.36, [], []
    for i, c in enumerate(USERS):
        for j, k in enumerate((4, 7)):
            x, v = i + (j - 0.5) * w, r[f"dspark@{k}"][f"natural-c{c}"]["page_latency_speedup"]
            ax.bar(x, v, width=w - 0.04, color=MAIN)
            _note(ax, x, v + 0.06, f"{v:.2f}×", ha="center")
            xs.append(x)
            ticks.append(f"K={k}")
        ax.text(
            i,
            -0.13,
            USERS[c],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.5,
            color=INK,
        )
    ax.set_xticks(xs, ticks)
    ax.set_ylim(0, 3.2)
    _ref(ax, "h", 1)
    _note(ax, 2.95, 1.05, "model alone", MUTED, ha="right")
    ax.set_xlim(-0.6, 2.97)
    _labels(ax, None, "page processing speedup (×)")
    _title(ax, "Page processing speedup vs the model alone")
    _save(f, out, "page_speedup.svg")


def throughput(summary: dict, out: Path) -> None:
    tp = summary["throughput"]
    by_c = tp["by_concurrency"]
    f, ax = _axes(7.5, 3.9)
    for name, label, color, ls in (
        ("dspark@7", "K=7", MAIN, "-"),
        ("dspark@4", "K=4", SECOND, "-"),
        ("baseline", "model alone", BASE, "--"),
    ):
        pts = sorted(
            (int(c), p["steady_state_tok_s"] or p["output_tok_s"]) for c, p in by_c[name].items()
        )
        _line(ax, *zip(*pts), label, color, ls)
    peak = tp["peak_output_tok_s"]
    _note(
        ax,
        0.02,
        0.96,
        f"peak: {peak['dspark@7']:,} vs {peak['baseline']:,} tok/s",
        transform=ax.transAxes,
        va="top",
    )
    xs = sorted(int(c) for c in by_c["baseline"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlim(xs[0] * 0.85, xs[-1] * 2.6)
    ax.set_ylim(bottom=0)
    _labels(ax, "concurrent requests", "output tokens/s (steady state)")
    _title(ax, "Throughput vs concurrency")
    _save(f, out, "throughput_vs_concurrency.svg")


def tau_by_category(summary: dict, out: Path) -> None:
    tc = summary["results"]["dspark@7"]["natural-c16"]["tau_by_category"]
    rows = sorted(tc.items(), key=lambda kv: kv[1])
    f, ax = _axes(7.5, 3.5, grid="x")
    ax.barh([k.replace("_", " ") for k, _ in rows], [v for _, v in rows], color=MAIN, height=0.6)
    for i, (_, v) in enumerate(rows):
        _note(ax, v + 0.08, i, f"{v:.2f}", va="center")
    ax.set_xlim(0, 8)
    _labels(ax, "acceptance length τ (K=7, 16 users)", None)
    _title(ax, "Acceptance length by document category")
    _save(f, out, "tau_by_category.svg")


def position_acceptance(summary: dict, out: Path) -> None:
    f, ax = _axes(7.5, 3.9)
    for name, color in (("dspark@7", MAIN), ("dspark@4", SECOND)):
        pos = summary["results"][name]["position_acceptance"]
        _line(ax, list(range(1, len(pos) + 1)), pos, f"K={name[-1]}", color)
    ax.set_xlim(0.7, 7.8)
    ax.set_ylim(0, 1)
    _labels(ax, "draft position", "P(accepted)")
    _title(ax, "Acceptance by draft position (16 users)")
    _save(f, out, "position_acceptance.svg")


def build(
    results: Path = Path("results"),
    out: Path = Path("docs/figures"),
) -> list[Path]:
    overhead = read_json(results / "overhead/summary.json")
    summary = read_json(results / "dspark/summary.json")
    out.mkdir(parents=True, exist_ok=True)
    decision_ranking(overhead, out)
    for chart in (step_cost, predicted_speedup):
        chart(overhead, out)
    for chart in (page_speedup, throughput, tau_by_category, position_acceptance):
        chart(summary, out)
    return sorted(p for p in out.glob("*.svg") if not p.name.startswith("diagram_"))
