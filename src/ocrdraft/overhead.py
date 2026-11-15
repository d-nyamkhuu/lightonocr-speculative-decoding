"""Drafter-overhead model: analytic weight-bytes estimate and fits of measured step costs.

Speedup model. A speculative step verifies K+1 positions and emits tau tokens on average
(tau = mean acceptance length, including the bonus token). If one no-spec decode step costs
t0 and one spec step (draft + verify) costs ts, then decode speedup = tau * t0 / ts. The
break-even acceptance length is ts / t0. Page latency adds the time to first token (vision
encoder + image prefill), which speculation does not shorten.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# ---------------------------------------------------------------- analytic model


@dataclass(frozen=True)
class TargetShape:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int

    @classmethod
    def from_text_config(cls, c: dict) -> "TargetShape":
        return cls(
            hidden=c["hidden_size"],
            intermediate=c["intermediate_size"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            vocab=c["vocab_size"],
        )


# LightOnOCR-2-1B text_config (Qwen3 decoder) at configs/run.json target_revision.
LIGHTONOCR_2_1B = TargetShape(
    hidden=1024, intermediate=3072, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936
)


def layer_params(t: TargetShape, attn_input: int | None = None) -> int:
    """Qwen3/Llama decoder layer weights (norms ignored). attn_input overrides q/k/v input width."""
    a = attn_input or t.hidden
    q = a * t.heads * t.head_dim
    kv = 2 * a * t.kv_heads * t.head_dim
    o = t.heads * t.head_dim * t.hidden
    mlp = 3 * t.hidden * t.intermediate
    return q + kv + o + mlp


def target_step_params(t: TargetShape) -> int:
    """Weights read per target decode step: all layers + LM head (embedding rows are negligible)."""
    return t.layers * layer_params(t) + t.vocab * t.hidden


def draft_step_params(t: TargetShape, v: dict) -> int:
    """Weights read to draft one block (parallel drafters) or K tokens (EAGLE-3)."""
    kind = v["type"]
    vocab = v.get("draft_vocab_size") or t.vocab
    n_aux = len(v["target_layer_ids"])
    head = vocab * t.hidden
    fc = n_aux * t.hidden * t.hidden
    if kind in ("dflash", "dflash2", "dspark"):
        total = v["num_layers"] * layer_params(t) + fc + head
        if kind == "dflash2":
            rank = v.get("selector_rank", 256)
            # predecessor + successor codebooks over the full vocab, plus hidden projection.
            total += 2 * t.vocab * rank + t.hidden * rank
        if kind == "dspark":
            rank = v.get("markov_rank", 256)
            total += 2 * vocab * rank + t.hidden  # Markov W1/W2 + confidence head
        return total
    if kind == "eagle3":
        k = max(v["k"])
        # First draft layer takes [embedding; hidden] -> 2x q/k/v input width.
        per_step = layer_params(t, attn_input=2 * t.hidden) + head
        return fc + k * per_step
    raise ValueError(f"unknown drafter type {kind}")


def analytic_table(t: TargetShape, variants: list[dict]) -> list[dict]:
    """Bandwidth-bound, batch-1 estimate. Verify of K+1 tokens is taken to cost one target step."""
    base = target_step_params(t)
    rows = []
    for v in variants:
        d = draft_step_params(t, v)
        ratio = 1.0 + d / base
        rows.append(
            {
                "variant": v["name"],
                "draft_params_read_m": round(d / 1e6, 1),
                "draft_over_target": round(d / base, 3),
                "break_even_tau": round(ratio, 2),
                "speedup_at_tau3": round(3.0 / ratio, 2),
                "speedup_at_tau4": round(4.0 / ratio, 2),
            }
        )
    return rows


# ---------------------------------------------------------------- measured model


def mean_acceptance(spec: dict | None) -> float:
    """tau from Prometheus deltas: 1 + accepted / drafts. 1.0 when speculation is off."""
    if not spec or not spec.get("num_drafts"):
        return 1.0
    return 1.0 + spec["num_accepted_tokens"] / spec["num_drafts"]


@dataclass
class CellSummary:
    config: str
    concurrency: int
    tpot: float  # median seconds per output token
    ttft: float  # median seconds to first token
    tokens: float  # median output tokens per request
    tau: float  # measured mean acceptance length
    n: int

    @property
    def step(self) -> float:
        """Seconds per verify step: each step emits tau tokens."""
        return self.tpot * self.tau


def summarize(run: dict) -> CellSummary:
    ok = [r for r in run["requests"] if r.get("ok") and r.get("tpot") is not None]
    if not ok:
        raise ValueError(f"no successful requests in {run['config']} c={run['concurrency']}")
    return CellSummary(
        config=run["config"],
        concurrency=run["concurrency"],
        tpot=statistics.median(r["tpot"] for r in ok),
        ttft=statistics.median(r["ttft"] for r in ok),
        tokens=statistics.median(r["tokens"] for r in ok),
        tau=mean_acceptance(run.get("spec")),
        n=len(ok),
    )


def page_speedup(base: CellSummary, spec: CellSummary, tau: float, tokens: float) -> float:
    """Per-request latency speedup for a page with `tokens` output tokens."""
    t_base = base.ttft + tokens * base.tpot
    t_spec = spec.ttft + tokens * spec.step / tau
    return t_base / t_spec


def decide(
    base: dict[int, CellSummary],
    drafters: dict[str, dict[int, CellSummary]],
    natural_tokens: float,
    ngram_speedup: dict[int, float],
    rule: dict,
    train_cost: dict[str, float],
) -> dict:
    """Apply the frozen GO/NO-GO rule. drafters maps 'variant@K' -> {concurrency: summary}."""
    conc = rule["score_concurrency"]
    scored = []
    for name, cells in drafters.items():
        if not all(c in cells and c in base for c in conc):
            continue
        tau = rule["prior_tau"][name.split("-", 1)[0]]  # by drafter type, e.g. "dspark"
        per = {c: page_speedup(base[c], cells[c], tau, natural_tokens) for c in conc}
        score = statistics.fmean(per.values())
        passes = all(per[c] >= rule["min_speedup"][str(c)] for c in conc)
        beats = all(
            per[c] >= (1 + rule["min_gain_over_ngram"]) * ngram_speedup.get(c, 1.0) for c in conc
        )
        scored.append(
            {
                "candidate": name,
                "prior_tau": tau,
                "page_speedup": {str(c): round(v, 3) for c, v in per.items()},
                "score": round(score, 3),
                "meets_thresholds": passes,
                "beats_ngram": beats,
                "break_even_tau": {str(c): round(cells[c].step / base[c].tpot, 2) for c in conc},
            }
        )
    scored.sort(key=lambda s: -s["score"])
    eligible = [s for s in scored if s["meets_thresholds"] and s["beats_ngram"]]
    if not eligible:
        return {"decision": "NO-GO", "choice": None, "ranking": scored}
    top = eligible[0]
    close = [s for s in eligible if s["score"] >= top["score"] * (1 - rule["tie_margin"])]
    choice = min(close, key=lambda s: train_cost.get(s["candidate"].split("@")[0], float("inf")))
    return {"decision": "GO", "choice": choice["candidate"], "ranking": scored}
