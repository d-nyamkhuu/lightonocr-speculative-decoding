"""Sample olmOCR-bench pages and render them as the model card recommends (longest side 1540 px)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

PDF_ROOT = "bench_data/pdfs/"


def category_of(path: str) -> str:
    return path[len(PDF_ROOT) :].split("/", 1)[0]


def categories_of(paths: list[str]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for p in paths:
        if p.startswith(PDF_ROOT) and p.endswith(".pdf"):
            by.setdefault(category_of(p), []).append(p)
    return {k: sorted(v) for k, v in sorted(by.items())}


def stratified_sample(by_cat: dict[str, list[str]], n: int, seed: int) -> list[str]:
    """Round-robin over categories so each gets n/len(cats) (+1 for the first n%len)."""
    rng = random.Random(seed)
    pools = {c: rng.sample(v, len(v)) for c, v in by_cat.items()}
    out: list[str] = []
    while len(out) < n and any(pools.values()):
        for c in by_cat:
            if pools[c] and len(out) < n:
                out.append(pools[c].pop())
    return out


def bench_sample(cfg: dict, n: int, seed: int) -> list[str]:
    """The `n` olmOCR-bench PDFs to evaluate on; page id i is the i-th path."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        cfg["bench_dataset"], repo_type="dataset", revision=cfg["bench_revision"]
    )
    return stratified_sample(categories_of(files), n, seed)


def render_pdf(pdf, png: Path | str, longest_px: int) -> list[int]:
    """Render page 1 of `pdf` (a path or bytes) to an RGB PNG; returns its [width, height]."""
    import pypdfium2 as pdfium

    page = pdfium.PdfDocument(pdf)[0]
    w, h = page.get_size()
    img = page.render(scale=longest_px / max(w, h)).to_pil().convert("RGB")
    img.save(png, format="PNG")
    return list(img.size)


def render_pages(cfg: dict, out: Path, n: int, seed: int) -> dict:
    """Page 1 of `n` olmOCR-bench PDFs, sampled evenly across categories, as PNGs in `out`."""
    from huggingface_hub import hf_hub_download

    repo, rev = cfg["bench_dataset"], cfg["bench_revision"]
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    for i, path in enumerate(bench_sample(cfg, n, seed)):
        local = hf_hub_download(repo, path, repo_type="dataset", revision=rev)
        png = out / f"{i:03d}.png"
        size = render_pdf(local, png, cfg["render_longest_px"])
        pages.append(
            {
                "id": png.stem,
                "category": category_of(path),
                "source": path,
                "png": png.name,
                "size": size,
                "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            }
        )
    manifest = {"dataset": repo, "revision": rev, "seed": seed, "pages": pages}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_pages(pages_dir: Path) -> list[dict]:
    """Manifest rows of a render_pages directory, with `png` as an absolute path."""
    pages_dir = Path(pages_dir).resolve()
    manifest = json.loads((pages_dir / "manifest.json").read_text())
    return [{**p, "png": str(pages_dir / p["png"])} for p in manifest["pages"]]
