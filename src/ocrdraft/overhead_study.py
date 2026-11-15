"""The overhead study: timings of the model alone, ngram and random-init drafters.

The README serves one configuration at a time; `bench_group` runs its cells against that server.
Every cell is written atomically to <out>/<config>/<cell>.json; complete cells are skipped, so a
rerun only redoes what is missing or failed.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from . import bench, data
from .util import log, write_atomic


def natural_subset(pages: list[dict], per_category: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for p in pages:
        if seen.get(p["category"], 0) < per_category:
            out.append(p)
            seen[p["category"]] = seen.get(p["category"], 0) + 1
    return out


def cell(name: str, concurrency: int, max_tokens: int, natural: bool, pages: str = "all") -> dict:
    # Fixed-length cells ignore EOS so every request decodes exactly `max_tokens`.
    return {
        "name": name,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "ignore_eos": not natural,
        "pages": pages,
    }


def plan(cfg: dict) -> list[dict]:
    """Ordered configurations and their cells."""
    fixed, nat = cfg["fixed_tokens"], cfg["natural_max_tokens"]
    natural_cells = [
        cell("natural-c1", 1, nat, natural=True, pages="subset"),
        cell("natural-c16", 16, nat, natural=True),
    ]
    base_conc = cfg["concurrency"] + cfg["baseline_extra_concurrency"]
    groups = [
        {
            "config": "baseline",
            "cells": [cell(f"fixed-c{c}", c, fixed, natural=False) for c in base_conc]
            + natural_cells,
        },
        {"config": "ngram", "cells": natural_cells},
    ]
    for v in cfg["variants"]:
        for k in v["k"]:
            groups.append(
                {
                    "config": f"{v['name']}@{k}",
                    "cells": [
                        cell(f"fixed-c{c}", c, fixed, natural=False) for c in cfg["concurrency"]
                    ],
                }
            )
    return groups


def bench_group(cfg: dict, group: dict, pages_dir: Path, out: Path, server: str) -> None:
    """The unfinished cells of `group` against the running server.

    A failure (e.g. the server died) is recorded in run_failed.json and re-raised; a cell with
    failed requests stays unfinished. Either way the cells left are retried on the next run.
    """
    gdir = out / group["config"]
    pages = data.load_pages(pages_dir)
    subset = natural_subset(pages, 2)
    (gdir / "startup_failed.json").unlink(missing_ok=True)
    (gdir / "run_failed.json").unlink(missing_ok=True)
    try:
        warmup = cfg["warmup_requests"]
        asyncio.run(bench.run_cell(server, cfg["target"], pages[:4], 2, warmup, 256, True))
        for c in group["cells"]:
            if not bench.cell_complete(gdir / f"{c['name']}.json"):
                run_one_cell(cfg, group, c, pages if c["pages"] == "all" else subset, gdir, server)
    except Exception as exc:
        write_atomic(gdir / "run_failed.json", {"error": repr(exc)[-4000:], "at": time.time()})
        raise
    bench.require_complete(gdir, [c["name"] for c in group["cells"]])


def run_one_cell(
    cfg: dict, group: dict, c: dict, pages: list[dict], gdir: Path, server: str
) -> None:
    # Fixed-length runs cycle the pages; natural runs send every selected page exactly once.
    per_c = cfg["requests_per_concurrency"][str(c["concurrency"])]
    n = per_c if c["ignore_eos"] else len(pages)
    res = asyncio.run(
        bench.run_cell(
            server,
            cfg["target"],
            pages,
            c["concurrency"],
            n,
            c["max_tokens"],
            c["ignore_eos"],
        )
    )
    res.update({"config": group["config"], "cell": c["name"], "finished_at": time.time()})
    write_atomic(gdir / f"{c['name']}.json", res)
    ok = sum(r.get("ok", False) for r in res["requests"])
    log(
        f"[cell] {group['config']} {c['name']} ok={ok}/{n} "
        f"tok/s={res['output_tokens_per_s']:.0f} spec={res['spec']}"
    )
