"""Training data and evaluation for the DSpark drafter.

Generation and evaluation talk to the running vLLM server; the rest prepares files for
`speculators`. Generation follows the training-page manifest as `ocrdraft render-train` writes
it, so the two can also run at the same time.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from . import bench, data
from .util import jsonl_ids, log, write_atomic

# Generation tolerates pages the server rejects outright (HTTP 4xx other than 408/429, e.g. a
# prompt over max_model_len) up to this fraction of the pages; they are recorded and skipped
# on rerun. Any other failure fails the step so a rerun retries it.
MAX_REJECTED_FRAC = 0.01


# ------------------------------------------------------------------ generation


def read_new_rows(manifest: Path, pos: int) -> tuple[list[dict], int]:
    """Complete lines appended since byte offset `pos`, and the new offset."""
    rows = []
    if manifest.exists():
        with manifest.open() as f:
            f.seek(pos)
            while (line := f.readline()).endswith("\n"):
                pos = f.tell()
                rows.append(json.loads(line))
    return rows, pos


async def aiter_manifest(pages_dir: Path):
    """Yield manifest rows as `ocrdraft render-train` appends them; stop after its DONE marker."""
    pos = 0
    while True:
        # Read the marker before the rows, so rows written just before it still count.
        finished = (pages_dir / "DONE").exists()
        rows, pos = read_new_rows(pages_dir / "manifest.jsonl", pos)
        for r in rows:
            yield r
        if finished:
            return
        await asyncio.sleep(2)


def is_permanent_rejection(exc: BaseException) -> bool:
    """A 4xx answer that retrying the same request cannot fix (not a timeout or rate limit)."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    code = exc.response.status_code
    return 400 <= code < 500 and code not in (408, 429)


async def generate_all(
    cfg: dict, pages_dir: Path, out: Path, concurrency: int, server: str
) -> dict:
    """Answers go to `out`; pages the server rejects (4xx) to `<out stem>.rejected.jsonl`."""
    rejected_path = out.with_name(out.stem + ".rejected.jsonl")
    pages_dir = pages_dir.resolve()  # the server needs absolute file:// URLs
    done_ids, rejected_ids = jsonl_ids(out), jsonl_ids(rejected_path)
    seen = done_ids | rejected_ids
    stats = {
        "ok": 0,
        "failed": 0,
        "rejected": 0,
        "tokens": 0,
        "skipped": len(done_ids),
        "previously_rejected": len(rejected_ids),
    }
    t0 = time.time()
    in_flight = asyncio.Semaphore(concurrency)
    queued = asyncio.Semaphore(2 * concurrency)  # don't read the manifest far ahead of the GPU
    timeout, limits = httpx.Timeout(1800, connect=30), httpx.Limits(max_connections=concurrency + 8)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        with out.open("a") as fout, rejected_path.open("a") as frej:

            async def one(row: dict) -> None:
                body = {
                    "model": cfg["target"],
                    "messages": bench.image_messages(f"file://{pages_dir / row['png']}"),
                    "max_tokens": cfg["natural_max_tokens"],
                    "temperature": 0.0,
                }
                try:
                    async with in_flight:
                        r = await client.post(f"{server}/v1/chat/completions", json=body)
                    r.raise_for_status()
                    j = r.json()
                    ch = j["choices"][0]
                    rec = {
                        **row,
                        "text": ch["message"]["content"],
                        "finish_reason": ch["finish_reason"],
                        "prompt_tokens": j["usage"]["prompt_tokens"],
                        "completion_tokens": j["usage"]["completion_tokens"],
                    }
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                    if is_permanent_rejection(exc):
                        stats["rejected"] += 1
                        frej.write(json.dumps({"id": row["id"], "error": repr(exc)[:500]}) + "\n")
                        frej.flush()
                    else:
                        stats["failed"] += 1
                    log(f"gen fail {row['id']}: {exc!r}"[:300])
                    return
                finally:
                    queued.release()
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                stats["ok"] += 1
                stats["tokens"] += rec["completion_tokens"]
                if stats["ok"] % 500 == 0:
                    log(
                        f"gen {stats['ok']} pages, {stats['tokens'] / (time.time() - t0):.0f} tok/s"
                    )

            tasks = []
            async for row in aiter_manifest(pages_dir):
                if row["id"] in seen:
                    continue
                await queued.acquire()
                tasks.append(asyncio.create_task(one(row)))
            await asyncio.gather(*tasks)
    stats["wall_s"] = round(time.time() - t0, 1)
    stats["tok_per_s"] = round(stats["tokens"] / max(stats["wall_s"], 1), 1)
    return stats


def generate(cfg: dict, pages_dir: Path, out: Path, concurrency: int, server: str) -> dict:
    """The target's answers to the pages in `pages_dir`, appended to `out`.

    Pages already answered are skipped. Raises unless (nearly) all pages succeeded.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = asyncio.run(generate_all(cfg, pages_dir, out, concurrency, server))
    if stats["failed"]:
        raise RuntimeError(f"{stats['failed']} generation requests failed; rerun to retry")
    rejected = stats["rejected"] + stats["previously_rejected"]
    total = rejected + stats["ok"] + stats["skipped"]
    if rejected > MAX_REJECTED_FRAC * total:
        raise RuntimeError(
            f"server rejected {rejected}/{total} pages (over "
            f"{MAX_REJECTED_FRAC:.0%}); see {out.stem}.rejected.jsonl"
        )
    return stats


# ------------------------------------------------------------------ files for speculators


def write_conversations(
    responses: Path, pages_dir: Path, out: Path, split: str | None = None
) -> int:
    """Clean (finished, non-empty) responses as speculators conversations.

    Images are referenced in `pages_dir`. `split` keeps only rows of that document-level split
    (ocrdraft.mix's `split` field).
    """
    n = 0
    with out.open("w") as f:
        for line in responses.read_text().splitlines():
            r = json.loads(line)
            if not r["text"] or r["finish_reason"] != "stop":
                continue  # truncated or empty answers are not clean targets
            if split is not None and r.get("split") != split:
                continue
            conv = [
                {
                    "role": "user",
                    "content": [{"type": "image", "path": str(pages_dir / r["png"])}],
                },
                {"role": "assistant", "content": r["text"]},
            ]
            f.write(json.dumps({"conversations": conv}) + "\n")
            n += 1
    return n


def write_split_conversations(responses: Path, pages_dir: Path, out_dir: Path) -> dict[str, Path]:
    """train.jsonl / val.jsonl from the document-level split, so validation is held-out docs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    convs = {split: out_dir / f"{split}.jsonl" for split in ("train", "val")}
    for split, path in convs.items():
        if write_conversations(responses, pages_dir.resolve(), path, split=split) == 0:
            raise RuntimeError(f"no clean {split}-split responses in {responses}")
    return convs


# speculators 0.9 train takes one --data-path and holds out its *last* rows for validation
# (ArrowDataset: rows [0, int(N * train_data_ratio)) train, the rest val), but prepare-data
# shuffles rows. So each split is prepared on its own and the two are concatenated train-first;
# train_split_ratio() then puts the boundary exactly between them. token_freq.pt comes from the
# train part only.
def merge_prepared(parts: Path, out: Path) -> dict:
    """Concatenate <parts>/train and <parts>/val into `out`, with the --train-data-ratio in a file."""
    import shutil

    from datasets import concatenate_datasets, load_from_disk

    train, val = load_from_disk(str(parts / "train")), load_from_disk(str(parts / "val"))
    shutil.rmtree(out, ignore_errors=True)
    concatenate_datasets([train, val]).save_to_disk(str(out))
    if (parts / "train/token_freq.pt").exists():
        shutil.copy(parts / "train/token_freq.pt", out / "token_freq.pt")
    split = {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_data_ratio": train_split_ratio(len(train), len(val)),
    }
    (out / "train_data_ratio").write_text(str(split["train_data_ratio"]))
    return split


def train_split_ratio(train_rows: int, val_rows: int) -> float:
    """--train-data-ratio whose int(N * ratio) boundary is exactly `train_rows`."""
    return (train_rows + 0.5) / (train_rows + val_rows)


# ------------------------------------------------------------------ evaluation

# Evaluation modes, in the order they run, and the results folder of each.
EVAL_DIRS = {"greedy": "eval", "sampled": "eval_sampled", "throughput": "eval_throughput"}


def eval_cells(cfg: dict, mode: str) -> list[tuple[str, int, int]]:
    """(name, concurrency, requests) per cell; requests cycle over the pages.

    greedy and sampled: c=1 on 56 pages (8 per category), c=16 and c=32 on all of them;
    throughput: greedy at high concurrency.
    """
    if mode == "throughput":
        tc = cfg["eval_throughput"]
        return [(f"natural-c{c}", c, tc["requests"]) for c in tc["concurrency"]]
    n = cfg["eval_pages"]
    return [("natural-c1", 1, 56), ("natural-c16", 16, n), ("natural-c32", 32, n)]


def run_eval(
    cfg: dict, name: str, pages_dir: Path, results: Path, server: str, modes=tuple(EVAL_DIRS)
) -> None:
    """Every unfinished cell of configuration `name`, in each mode, against the running server.

    Cells go to <results>/<eval dir for mode>/<name>/<cell>.json. Exits naming the cells left
    unfinished, which a rerun retries.
    """
    pages = data.load_pages(pages_dir)
    asyncio.run(bench.run_cell(server, cfg["target"], pages[:4], 2, 2, 256, True))  # warm-up
    for mode in modes:
        run_mode(cfg, mode, name, pages, results / EVAL_DIRS[mode] / name, server)
    missing = []
    for mode in modes:
        out = results / EVAL_DIRS[mode] / name
        missing += [
            f"{mode}/{c[0]}"
            for c in eval_cells(cfg, mode)
            if not bench.cell_complete(out / f"{c[0]}.json")
        ]
    if missing:
        raise SystemExit(f"{name}: incomplete cells {', '.join(missing)}; run the command again")


def run_mode(cfg: dict, mode: str, name: str, pages: list[dict], out: Path, server: str) -> None:
    sampling = cfg["eval_sampling"] if mode == "sampled" else None
    todo = [c for c in eval_cells(cfg, mode) if not bench.cell_complete(out / f"{c[0]}.json")]
    for cell, c, n in todo:
        res = asyncio.run(
            bench.run_cell(
                server,
                cfg["target"],
                pages[:n],
                c,
                n,
                cfg["natural_max_tokens"],
                False,
                sampling=sampling,
            )
        )
        res.update({"config": name, "cell": cell, "sampling": sampling})
        write_atomic(out / f"{cell}.json", res)
        log(f"eval {name} {mode} {cell} tok/s={res['output_tokens_per_s']:.0f} spec={res['spec']}")
