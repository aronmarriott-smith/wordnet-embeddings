"""Benchmark the exported model against MTEB's English STS task family.

Requires bin/export_sentence_transformer.sh to have been run first (loads
data/sentence_transformers/). Writes mteb's raw per-task results under
benchmarks/raw/ and a compact summary (one score per task, plus the run's
training config) to benchmarks/sts_summary_<timestamp>.json — the summary
files are meant to be committed, so scores are comparable across iterations
via git history.

Usage::

    python -m wordnet_embeddings.benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from wordnet_embeddings.config import EMBED_DIM, NUM_EPOCHS, ST_EXPORT_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = Path("benchmarks")
RAW_DIR = BENCHMARK_DIR / "raw"


def run_sts_benchmark(model_dir: Path = ST_EXPORT_DIR, output_dir: Path = BENCHMARK_DIR) -> dict:
    import mteb
    from sentence_transformers import SentenceTransformer

    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} not found — run bin/export_sentence_transformer.sh first"
        )

    model = SentenceTransformer(str(model_dir))
    # languages=["eng"] alone isn't enough: MTEB includes a task if English is
    # *any* of its language pairs, so cross-lingual tasks (STS17, STS22, ...)
    # pass too even though most of their content isn't English. Filtering on
    # is_multilingual drops those, leaving only monolingual English STS tasks
    # — the right scope per CUSTOM_EMBEDDINGS_RESEARCH.md's English-only scope.
    all_tasks = mteb.get_tasks(task_types=["STS"], languages=["eng"])
    tasks = [t for t in all_tasks if not t.metadata.is_multilingual]
    log.info("Running %d STS tasks: %s", len(tasks), ", ".join(t.metadata.name for t in tasks))

    raw_dir = output_dir / "raw"
    results = mteb.MTEB(tasks=tasks).run(model, output_folder=str(raw_dir), overwrite_results=True)

    scores = {r.task.metadata.name: r.get_score() for r in results}
    overall = sum(scores.values()) / len(scores) if scores else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"embed_dim": EMBED_DIM, "num_epochs": NUM_EPOCHS},
        "overall_mean_score": overall,
        "scores": scores,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"sts_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("--- STS summary ---")
    for name, score in sorted(scores.items()):
        log.info("%-20s %.4f", name, score)
    log.info("%-20s %.4f", "OVERALL (mean)", overall)
    log.info("Saved %s", summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ST_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR)
    args = parser.parse_args()
    run_sts_benchmark(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
