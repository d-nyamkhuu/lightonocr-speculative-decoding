"""The `ocrdraft` command. Run it from the repo root; `ocrdraft <command> -h` lists its arguments.

  any machine: render-bench, report
  GPU steps:   the README's steps, run inside the vLLM container next to a running `vllm serve`

`ocrdraft` never starts a process: the README gives the `vllm serve` and `speculators` commands.
Fixed choices (model and dataset revisions, drafter shapes, benchmark cells) are in
configs/run.json; paths and sizes are arguments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .bench import SERVER
from .util import read_json

CONFIG = Path("configs/run.json")


# ------------------------------------------------------------------ any machine


def cmd_render_bench(a) -> None:
    from . import data

    m = data.render_pages(read_json(CONFIG), Path(a.out), a.pages, a.seed)
    print(f"{len(m['pages'])} pages -> {a.out}")


def cmd_report(a) -> None:
    from . import figures, report_drafter, report_overhead

    overhead_raw, drafter_raw = Path(a.overhead_raw), Path(a.drafter_raw)
    if overhead_raw.is_dir():
        print(report_overhead.build(overhead_raw, CONFIG))
    else:
        print(f"no {overhead_raw}: keeping results/overhead/summary.json")
    if (drafter_raw / "eval").is_dir():
        print(report_drafter.build(drafter_raw, CONFIG, overhead_raw.parent / "summary.json"))
    else:
        print(f"no {drafter_raw / 'eval'}: keeping results/dspark/summary.json")
    for p in figures.build():
        print(p)


# ------------------------------------------------------------------ GPU steps


def cmd_drafter_args(a) -> None:
    from . import drafters

    print(" ".join(drafters.arch_args(drafters.variant(read_json(CONFIG), a.name))))


def cmd_token_freq(a) -> None:
    from . import drafters

    drafters.write_token_freq(Path(a.out))


def cmd_overhead_bench(a) -> None:
    from . import bench, overhead_study

    cfg = read_json(CONFIG)
    groups = {g["config"]: g for g in overhead_study.plan(cfg)}
    if a.config not in groups:
        raise SystemExit(f"unknown config {a.config!r}; one of: {', '.join(groups)}")
    bench.wait_until_ready(a.server)
    overhead_study.bench_group(cfg, groups[a.config], Path(a.pages), Path(a.out), a.server)


def cmd_render_train(a) -> None:
    from . import mix

    print(json.dumps(mix.prepare(read_json(CONFIG), Path(a.out), a.workers)))


def cmd_generate(a) -> None:
    from . import bench, drafter_pipeline

    bench.wait_until_ready(a.server)
    stats = drafter_pipeline.generate(
        read_json(CONFIG), Path(a.pages), Path(a.out), a.concurrency, a.server
    )
    print(json.dumps(stats))


def cmd_conversations(a) -> None:
    from . import drafter_pipeline

    drafter_pipeline.write_split_conversations(Path(a.answers), Path(a.pages), Path(a.out))


def cmd_merge_prepared(a) -> None:
    from . import drafter_pipeline

    print(json.dumps(drafter_pipeline.merge_prepared(Path(a.parts), Path(a.out))))


def cmd_eval(a) -> None:
    from . import bench, drafter_pipeline

    bench.wait_until_ready(a.server)
    cfg = read_json(CONFIG)
    drafter_pipeline.run_eval(cfg, a.config, Path(a.pages), Path(a.out), a.server)


def main(argv=None) -> int:
    cfg = read_json(CONFIG)
    ap = argparse.ArgumentParser(prog="ocrdraft", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    def command(name, fn, help, *args):
        p = sub.add_parser(name, help=help, description=help)
        for flags, kw in args:
            p.add_argument(*flags, **kw)
        p.set_defaults(fn=fn)

    def arg(*flags, **kw):
        if "default" in kw and "help" in kw:
            kw["help"] += " (default: %(default)s)"
        return flags, kw

    server = arg(
        "--server",
        default=SERVER,
        help="vLLM server to use; the command waits until it is ready",
    )

    command(
        "render-bench",
        cmd_render_bench,
        "Render page 1 of olmOCR-bench PDFs (the revision pinned in configs/run.json), sampled "
        "evenly across its document categories, to PNGs plus manifest.json.",
        arg("--out", default="data/overhead_pages", help="output folder"),
        arg("--pages", type=int, default=cfg["pages"], help="how many pages"),
        arg("--seed", type=int, default=cfg["page_seed"], help="sampling seed"),
    )
    command(
        "report",
        cmd_report,
        "Rebuild results/overhead/summary.json and results/dspark/summary.json from raw results "
        "where they exist, then redraw the README charts in docs/figures.",
        arg("--overhead-raw", default="results/overhead/raw", help="`overhead-bench` results"),
        arg("--drafter-raw", default="results/dspark/raw", help="`eval` results"),
    )
    command(
        "drafter-args",
        cmd_drafter_args,
        "Print the `speculators train` architecture flags for a drafter in configs/run.json.",
        arg("name", help="a `variants` name, or `dspark` for the trained drafter"),
    )
    command(
        "token-freq",
        cmd_token_freq,
        "Write a token-frequency table, which `speculators train` needs to build "
        "pruned-vocabulary drafters.",
        arg("--out", default="data/drafters/_data/token_freq.pt", help="output file"),
    )
    command(
        "overhead-bench",
        cmd_overhead_bench,
        "Time one overhead-study configuration against the running server; one JSON file per "
        "benchmark cell. Finished cells are skipped.",
        arg("config", help="baseline, ngram or <variant>@<K>"),
        arg("--pages", default="data/overhead_pages", help="`ocrdraft render-bench` output"),
        arg("--out", default="results/overhead/raw", help="results folder"),
        server,
    )
    command(
        "render-train",
        cmd_render_train,
        "Download olmOCR-mix tarballs and render the training pages (subsets and counts in "
        "configs/run.json `train_data`), dropping pages identical to a benchmark page.",
        arg("--out", default="data/train_pages", help="output folder"),
        arg("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="processes"),
    )
    command(
        "generate",
        cmd_generate,
        "Ask the running server to transcribe every training page; the answers become the "
        "drafter's training targets. Pages already answered are skipped.",
        arg("--pages", default="data/train_pages", help="`ocrdraft render-train` output"),
        arg("--out", default="data/answers.jsonl", help="answers file"),
        arg("--concurrency", type=int, default=512, help="requests in flight"),
        server,
    )
    command(
        "conversations",
        cmd_conversations,
        "Turn complete answers into `speculators` conversations, split into train.jsonl and "
        "val.jsonl by source document.",
        arg("--answers", default="data/answers.jsonl", help="`ocrdraft generate` output"),
        arg("--pages", default="data/train_pages", help="the page images the answers refer to"),
        arg("--out", default="data/conversations", help="output folder"),
    )
    command(
        "merge-prepared",
        cmd_merge_prepared,
        "Join the separately prepared train and val data, train first, and write the "
        "--train-data-ratio that splits them again to <out>/train_data_ratio.",
        arg("--parts", default="data/prep_parts", help="folder holding train/ and val/"),
        arg("--out", default="data/prep", help="output folder"),
    )
    command(
        "eval",
        cmd_eval,
        "Time one configuration on the held-out pages against the running server: greedy and "
        "sampled decoding at 1, 16 and 32 users, then greedy throughput at 64 and 128. One JSON "
        "file per cell; finished cells are skipped.",
        arg("config", help="a name for the served configuration: baseline or dspark@<K>"),
        arg("--pages", default="data/eval_pages", help="`ocrdraft render-bench` output"),
        arg("--out", default="results/dspark/raw", help="results folder"),
        server,
    )
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
