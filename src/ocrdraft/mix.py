"""Sample, extract and render olmOCR-mix-1025 training pages, streaming a manifest as pages finish.

Each PDF in the mix is a single page named `<source document>-<page>` (National Archives uses
`-page-<n>`). Train/val are split by source document so near-duplicate pages of one document
never straddle the split. PDFs byte-identical to any olmOCR-bench PDF are dropped.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .data import render_pdf
from .util import jsonl_ids

_PAGE_SUFFIX = re.compile(r"(-page)?-\d+$")


def group_key(member: str) -> str:
    stem = member[:-4] if member.endswith(".pdf") else member
    return _PAGE_SUFFIX.sub("", stem)


def is_val(group: str, val_per_mille: int, seed: int) -> bool:
    h = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(h[:8], 16) % 1000 < val_per_mille


def select(
    members: dict[str, list[str]], quotas: dict[str, int], val_per_mille: int, seed: int
) -> list[dict]:
    """members: subset -> PDF member names (tar:arcname). Returns quota-capped, shuffled picks."""
    rng = random.Random(seed)
    out = []
    for subset, names in sorted(members.items()):
        names = sorted(names)
        rng.shuffle(names)
        for name in names[: quotas[subset]]:
            g = group_key(name.split(":", 1)[1])
            out.append(
                {
                    "subset": subset,
                    "member": name,
                    "group": g,
                    "split": "val" if is_val(g, val_per_mille, seed) else "train",
                }
            )
    rng.shuffle(out)  # interleave subsets so a streaming consumer sees a mixed stream
    return out


def tarballs_for(subset: str, count: int) -> list[str]:
    return [f"pdf_tarballs/{subset}_{i:05d}.tar.gz" for i in range(count)]


def _render_job(args):
    page_id, pdf_bytes, png, longest = args
    try:
        return page_id, render_pdf(pdf_bytes, png, longest), None
    except Exception as exc:  # noqa: BLE001 - a bad PDF is skipped, not fatal
        return page_id, None, repr(exc)


def bench_hashes(repo: str, revision: str) -> set[str]:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo, repo_type="dataset", revision=revision, allow_patterns=["bench_data/pdfs/*"]
        )
    )
    return {hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.pdf")}


def prepare(cfg: dict, out: Path, workers: int) -> dict:
    """Download tarballs, pick pages, render in parallel, append to out/manifest.jsonl."""
    from huggingface_hub import hf_hub_download

    t = cfg["train_data"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done_ids = jsonl_ids(manifest)
    banned = bench_hashes(cfg["bench_dataset"], cfg["bench_revision"])

    # 1) Download tarballs and list their PDFs.
    local: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for subset, spec in t["subsets"].items():
        for tb in tarballs_for(subset, spec["tarballs"]):
            path = hf_hub_download(t["dataset"], tb, repo_type="dataset", revision=t["revision"])
            local[tb] = path
            with tarfile.open(path) as tf:
                members.setdefault(subset, []).extend(
                    f"{tb}:{m.name}"
                    for m in tf.getmembers()
                    if m.isfile() and m.name.endswith(".pdf")
                )
    picks = select(
        members, {s: v["pages"] for s, v in t["subsets"].items()}, t["val_per_mille"], t["seed"]
    )
    for i, p in enumerate(picks):
        p["id"] = f"{i:05d}"

    # 2) Read the chosen PDFs tarball by tarball and render with a process pool.
    by_tar: dict[str, list[dict]] = {}
    for p in picks:
        if p["id"] not in done_ids:
            by_tar.setdefault(p["member"].split(":", 1)[0], []).append(p)
    stats = {"picked": len(picks), "rendered": len(done_ids), "bench_dupes": 0, "errors": 0}
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool, manifest.open("a") as mf:
        futures: dict = {}

        def drain(block: bool) -> None:
            ready = list(as_completed(futures)) if block else [f for f in futures if f.done()]
            for fut in ready:
                p = futures.pop(fut)
                page_id, size, err = fut.result()
                if err:
                    stats["errors"] += 1
                    continue
                row = {
                    "id": page_id,
                    "png": f"{page_id}.png",
                    "size": size,
                    "subset": p["subset"],
                    "group": p["group"],
                    "split": p["split"],
                    "source": p["member"],
                    "pdf_sha256": p["pdf_sha256"],
                }
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                stats["rendered"] += 1

        for tb, plist in by_tar.items():
            want = {p["member"].split(":", 1)[1]: p for p in plist}
            with tarfile.open(local[tb]) as tf:
                for m in tf:
                    p = want.get(m.name)
                    if p is None:
                        continue
                    data = tf.extractfile(m).read()
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in banned:
                        stats["bench_dupes"] += 1
                        continue
                    p["pdf_sha256"] = digest
                    png = out / f"{p['id']}.png"
                    futures[
                        pool.submit(_render_job, (p["id"], data, str(png), t["render_longest_px"]))
                    ] = p
                    if len(futures) % 64 == 0:
                        drain(block=False)  # stream finished pages to the manifest early
        drain(block=True)
    stats["render_s"] = round(time.time() - t0, 1)
    (out / "DONE").write_text(json.dumps(stats))
    return stats
