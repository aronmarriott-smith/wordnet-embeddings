"""Train 128-dim TransE knowledge-graph embeddings on WordNet triples.

Usage::

    python -m wordnet_embeddings.train                # full run (200 epochs)
    python -m wordnet_embeddings.train --epochs 1     # quick smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

EMBED_DIM = 128   # our own embedding space, not tied to any other model
NUM_EPOCHS = 200

DEFAULT_TRIPLES = Path("data/triples.tsv")
DEFAULT_OUTPUT = Path("data/model")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def train(
    triples_path: Path = DEFAULT_TRIPLES,
    output_path: Path = DEFAULT_OUTPUT,
    num_epochs: int = NUM_EPOCHS,
    evaluate: bool = False,
) -> None:
    import torch
    from pykeen.models import TransE
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    log.info("Loading triples from %s ...", triples_path)
    tf = TriplesFactory.from_path(triples_path)

    if evaluate:
        # 80/10/10 split for held-out evaluation.
        # Scores every test triple against all 117k entities —
        # fast on GPU (CUDA auto-detected by PyTorch), ~1hr on CPU.
        training, _, testing = tf.split([0.8, 0.1, 0.1], random_state=42)
        log.info("Split: %d train / %d test", training.num_triples, testing.num_triples)
    else:
        # Train on all triples — better for a final model since no data is
        # reserved for evaluation splits we aren't using.
        training = tf
        log.info("Evaluation disabled; training on all %d triples", tf.num_triples)

    log.info(
        "%d entities | %d relations | %d dims | %d epochs",
        tf.num_entities, tf.num_relations, EMBED_DIM, num_epochs,
    )

    model = TransE(
        triples_factory=training,
        embedding_dim=EMBED_DIM,
        loss="MarginRankingLoss",  # original TransE loss; NSSALoss is a stronger alternative
        random_seed=42,
    )

    optimizer = torch.optim.Adam(model.parameters())  # adaptive lr; SGD is the simpler alternative

    # SLCWATrainingLoop = Stochastic Local Closed World Assumption,
    # the standard training loop for TransE with margin ranking loss.
    loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    loop.train(
        triples_factory=training,
        num_epochs=num_epochs,
        checkpoint_directory=str(output_path / "checkpoints"),
        checkpoint_frequency=10,  # save to disk every N epochs (allows resume)
    )

    # Save entity_to_id and raw float embeddings for export.py
    entity_to_id = training.entity_to_id  # synset_name -> integer row index
    (output_path / "entity_to_id.json").write_text(json.dumps(entity_to_id))

    embeddings = (
        model.entity_representations[0](indices=None)  # all entity vectors
        .detach()
        .cpu()
        .numpy()
    )  # shape: (num_entities, EMBED_DIM)
    np.save(output_path / "entity_embeddings.npy", embeddings)

    # Save model weights for potential future use (resume, evaluation, etc.)
    torch.save(model.state_dict(), output_path / "model.pt")

    log.info("Saved entity_to_id.json, entity_embeddings.npy, model.pt -> %s", output_path)

    if evaluate:
        from pykeen.evaluation import RankBasedEvaluator
        evaluator = RankBasedEvaluator()
        result = evaluator.evaluate(
            model=model,
            mapped_triples=testing.mapped_triples,
            additional_filter_triples=[training.mapped_triples],
        )
        log.info(
            "Hits@10: %.4f | MRR: %.4f",
            result.get_metric("hits@10"),
            result.get_metric("mean_reciprocal_rank"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Training epochs (default: {NUM_EPOCHS}). Use --epochs 1 to smoke-test.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run ranking evaluation after training. Fast on GPU (CUDA auto-detected); ~1hr on CPU.",
    )
    args = parser.parse_args()
    train(args.triples, args.output, args.epochs, args.evaluate)


if __name__ == "__main__":
    main()
