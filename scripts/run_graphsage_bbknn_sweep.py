"""CLI runner for 5-seed GraphSAGE benchmark on BBKNN graph (backward-compatible wrapper)."""

from __future__ import annotations

import argparse

from scripts.run_graphsage_sweep import DEFAULT_SEEDS, run_graphsage_sweep

DEFAULT_GRAPH = "bbknn_kperbatch2_donors12"


def run_graphsage_bbknn_sweep(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    graph_name: str = DEFAULT_GRAPH,
    seeds: list[int] | None = None,
    device: str = "auto",
    max_epochs: int = 500,
    patience: int = 50,
) -> list[dict[str, float | str | int]]:
    """Execute the GraphSAGE benchmark sweep across seeds on the specified BBKNN graph."""
    return run_graphsage_sweep(
        dataset_name=dataset_name,
        split_id=split_id,
        graph_name=graph_name,
        seeds=seeds,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-seed GraphSAGE benchmark on BBKNN graph.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument(
        "--graph",
        type=str,
        default=DEFAULT_GRAPH,
        help="Graph variant name.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (auto, cuda, mps, cpu).",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    run_graphsage_bbknn_sweep(
        dataset_name=args.dataset,
        split_id=args.split,
        graph_name=args.graph,
        seeds=args.seeds,
        device=args.device,
        max_epochs=args.epochs,
        patience=args.patience,
    )
