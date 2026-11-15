"""Closed-loop replay client for a vLLM OpenAI endpoint serving LightOnOCR.

Each request sends one page image (the model card's image-only prompt) and streams the reply.
TTFT is time to the first content chunk; TPOT is (last chunk - first chunk) / (tokens - 1), so
bursts of several speculative tokens per chunk are not miscounted. Acceptance comes from the
server's Prometheus counters, diffed around each cell.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

SERVER = "http://127.0.0.1:8000"  # the default: host and port in configs/vllm.yaml

SPEC_COUNTERS = {
    "vllm:spec_decode_num_drafts_total": "num_drafts",
    "vllm:spec_decode_num_draft_tokens_total": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens",
}


def parse_spec_counters(text: str) -> dict:
    """Sum the spec-decode counters over all label sets (per-position series excluded).

    Counter families come back without `_total`; their samples are `<name>_total` (the value)
    and `<name>_created` (a timestamp, ignored). Untyped lines keep their name as written.
    """
    out = {v: 0.0 for v in SPEC_COUNTERS.values()}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = SPEC_COUNTERS.get(sample.name)
            if key is None or "position" in sample.labels:
                continue
            out[key] += sample.value
    return out


def image_payload(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def image_messages(image_url: str) -> list[dict]:
    """The model card's prompt: one user turn holding only the page image."""
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}]


def build_request(
    model: str, image_url: str, max_tokens: int, ignore_eos: bool, sampling: dict | None = None
) -> dict:
    """Greedy by default; `sampling` (e.g. temperature/top_p) overrides it."""
    body = {
        "model": model,
        "messages": image_messages(image_url),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if sampling:
        body.update(sampling)
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def cell_complete(path: Path) -> bool:
    """A saved run_cell result in which every request succeeded; anything else is rerun."""
    if not path.exists():
        return False
    run = json.loads(path.read_text())
    reqs = run["requests"]
    return 0 < run["n_requests"] == len(reqs) and all(r.get("ok") for r in reqs)


def wait_until_ready(server: str, timeout_s: float = 1800, poll_s: float = 5) -> None:
    """Block until the vLLM server answers /health, so a client can start before the server."""
    t0 = time.time()
    said = False
    while True:
        try:
            if httpx.get(f"{server}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"no vLLM server ready at {server} after {timeout_s:.0f}s")
        if not said:
            print(f"waiting for the vLLM server at {server} ...", flush=True)
            said = True
        time.sleep(poll_s)


def require_complete(out: Path, cells: list[str]) -> None:
    """Exit with an error naming the cells under `out` that are missing or had failed requests."""
    missing = [c for c in cells if not cell_complete(out / f"{c}.json")]
    if missing:
        raise SystemExit(f"{out}: incomplete cells {', '.join(missing)}; run the command again")


def timing_from_chunks(t0: float, chunk_times: list[float], tokens: int) -> dict:
    if not chunk_times:
        return {"ttft": None, "tpot": None}
    first, last = chunk_times[0], chunk_times[-1]
    tpot = (last - first) / (tokens - 1) if tokens > 1 and last > first else None
    return {"ttft": first - t0, "tpot": tpot, "e2e": last - t0}


async def one_request(client: httpx.AsyncClient, base_url: str, body: dict, page_id: str) -> dict:
    t0 = time.perf_counter()
    chunk_times: list[float] = []
    parts: list[str] = []
    usage, metrics, finish = None, None, None
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode(errors="replace")[:500]
                return {"page": page_id, "ok": False, "error": f"HTTP {r.status_code}: {err}"}
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("metrics"):
                    metrics = obj["metrics"]
                for ch in obj.get("choices", []):
                    text = (ch.get("delta") or {}).get("content")
                    if text:
                        chunk_times.append(time.perf_counter())
                        parts.append(text)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return {"page": page_id, "ok": False, "error": repr(exc)}
    tokens = (usage or {}).get("completion_tokens", 0)
    rec = {
        "page": page_id,
        "ok": tokens > 0,
        "tokens": tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "finish_reason": finish,
        "chunks": len(chunk_times),
        "text": "".join(parts),
        "spec_metrics": metrics,
    }
    rec.update(timing_from_chunks(t0, chunk_times, tokens))
    return rec


async def run_cell(
    base_url: str,
    model: str,
    pages: list[dict],
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    ignore_eos: bool,
    timeout_s: float = 900.0,
    sampling: dict | None = None,
) -> dict:
    """Closed loop: `concurrency` workers issue `n_requests` requests cycling over `pages`."""
    images = {p["id"]: image_payload(Path(p["png"])) for p in pages}
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n_requests):
        queue.put_nowait(i)
    results: list[dict] = []
    limits = httpx.Limits(max_connections=concurrency + 4)
    timeout = httpx.Timeout(timeout_s, connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        before = parse_spec_counters((await client.get(f"{base_url}/metrics")).text)

        async def worker():
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                page = pages[i % len(pages)]
                body = build_request(model, images[page["id"]], max_tokens, ignore_eos, sampling)
                rec = await one_request(client, base_url, body, page["id"])
                rec["index"] = i
                results.append(rec)

        t_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t_start
        after = parse_spec_counters((await client.get(f"{base_url}/metrics")).text)
    results.sort(key=lambda r: r["index"])
    out_tokens = sum(r.get("tokens", 0) for r in results if r.get("ok"))
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "wall_s": wall,
        "output_tokens_per_s": out_tokens / wall if wall else None,
        "spec": {k: after[k] - before[k] for k in before},
        "requests": results,
    }
