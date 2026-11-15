"""`speculators train` flags for each drafter shape (the README's random-init drafters).

The overhead study serves random-init drafters (`speculators train --dry-run`): step latency
does not depend on weight values, and random weights accept ~nothing (tau ~ 1), so they
measure the pure cost of one speculative step.
"""

from __future__ import annotations

from pathlib import Path

PARALLEL = ("dflash", "dflash2", "dspark")  # drafters that propose a whole block per step


def arch_args(v: dict) -> list[str]:
    """`speculators train` flags for a drafter shape (a configs/run.json variant or `dspark`)."""
    args = ["--speculator-type", v["type"], "--num-layers", str(v["num_layers"])]
    args += ["--target-layer-ids", *map(str, v["target_layer_ids"])]
    if v.get("block_size"):
        args += ["--block-size", str(v["block_size"])]
    if v.get("draft_vocab_size"):
        args += ["--draft-vocab-size", str(v["draft_vocab_size"])]
    if v["type"] in PARALLEL:
        # Last draft layer uses full attention, as in published small DFlash2 recipes.
        args += ["--full-attention-indices", str(v["num_layers"] - 1)]
    if v["type"] == "dspark":
        args += ["--markov-rank", str(v.get("markov_rank", 256))]
        args += ["--enable-confidence-head", "--confidence-head-with-markov"]
    return args


def write_token_freq(path: Path, vocab: int = 151936) -> None:
    """A token-frequency table; pruned-vocabulary drafters need one, and any works for timing."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({i: vocab - i for i in range(vocab)}, path)


def variant(cfg: dict, name: str) -> dict:
    """A configs/run.json drafter shape by name: a `variants` entry, or `dspark`."""
    return (
        cfg["dspark"] if name == "dspark" else next(v for v in cfg["variants"] if v["name"] == name)
    )
