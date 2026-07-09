"""Evaluation metrics including Expected Calibration Error (ECE)."""

from __future__ import annotations

from collections import defaultdict


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Compute ECE and per-bin stats.

    confidences and correct must be the same length.
    """
    if not confidences:
        return 0.0, []

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for confidence, is_correct in zip(confidences, correct):
        idx = min(int(confidence * n_bins), n_bins - 1)
        bins[idx].append((confidence, is_correct))

    total = len(confidences)
    ece = 0.0
    bin_stats = []
    for idx, entries in enumerate(bins):
        if not entries:
            continue
        avg_conf = sum(item[0] for item in entries) / len(entries)
        avg_acc = sum(item[1] for item in entries) / len(entries)
        weight = len(entries) / total
        ece += weight * abs(avg_acc - avg_conf)
        bin_stats.append(
            {
                "bin": idx,
                "count": len(entries),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "lo": idx / n_bins,
                "hi": (idx + 1) / n_bins,
            }
        )
    return ece, bin_stats


def summarize_confusion(results: list[dict]) -> dict[tuple[str, str], int]:
    confusions: dict[tuple[str, str], int] = defaultdict(int)
    for row in results:
        if not row["correct"]:
            confusions[(row["gold"], str(row["pred"]))] += 1
    return dict(confusions)


def save_reliability_diagram(bin_stats: list[dict], path: str, title: str = "Reliability") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for reliability diagrams. Install with: pip install matplotlib"
        ) from exc

    xs = [item["avg_confidence"] for item in bin_stats]
    ys = [item["avg_accuracy"] for item in bin_stats]
    counts = [item["count"] for item in bin_stats]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.scatter(xs, ys, s=[max(20, c) for c in counts], alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
