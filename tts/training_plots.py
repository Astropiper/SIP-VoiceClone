"""Render training and validation metrics to a headless PNG."""

from dataclasses import dataclass
import math
from pathlib import Path

from .plotting import save_png


TRAIN = "train"
VALIDATION = "validation"
BLUE = "#1763a6"
ORANGE = "#b54817"
PURPLE = "#65469b"
GREEN = "#08796a"


@dataclass(frozen=True)
class _Series:
    metric: str
    split: str
    label: str
    color: str
    scale: float = 1.0


@dataclass(frozen=True)
class _Panel:
    title: str
    ylabel: str
    series: tuple[_Series, ...]


def _split_series(metric: str, scale: float = 1.0) -> tuple[_Series, ...]:
    return (
        _Series(metric, TRAIN, "Train", BLUE, scale),
        _Series(metric, VALIDATION, "Validation", ORANGE, scale),
    )


def _encoder_series(metric: str) -> tuple[_Series, ...]:
    return (
        _Series(f"student_{metric}", VALIDATION, "Student", BLUE),
        _Series(f"teacher_{metric}", VALIDATION, "EMA teacher", ORANGE),
    )


_LEARNING_RATE = _Panel("Learning rate", "Rate", (_Series("lr", TRAIN, "Train", GREEN),))
_CONTENT_DISTANCE = _Panel("Content sensitivity", "Aligned L2 distance", _encoder_series("aligned_content_distance"))
_STAGES = {
    "pretrain": (
        "Stage 1 · Encoder pretraining",
        (
            _Panel("Training objective", "Loss", _split_series("loss")),
            _Panel("Masked frame regression", "MSE", _split_series("regression_loss")),
            _Panel("Variance penalty", "Unweighted loss", _split_series("variance_loss")),
            _CONTENT_DISTANCE,
            _Panel("Representation rank", "Effective rank", _encoder_series("effective_rank")),
            _Panel("Masked frame variation", "Mean channel std", (
                _Series("masked_target_std", TRAIN, "Target (train)", BLUE),
                _Series("masked_target_std", VALIDATION, "Target (validation)", PURPLE),
                _Series("masked_prediction_std", TRAIN, "Prediction (train)", ORANGE),
                _Series("masked_prediction_std", VALIDATION, "Prediction (validation)", GREEN),
            )),
            _Panel("Mask coverage", "Valid frames (%)", _split_series("mask_coverage", scale=100)),
            _Panel("Mask span", "Mean frames per span", _split_series("mask_mean_span")),
            _LEARNING_RATE,
        ),
    ),
    "train-text": (
        "Stage 2 · Text-to-acoustic model",
        (
            _Panel("Training objective", "Loss", _split_series("loss")),
            _Panel("Acoustic prediction", "L1", _split_series("acoustic_l1")),
            _Panel("Stop prediction", "Binary cross-entropy", _split_series("stop_bce")),
            _Panel("Free generation length", "Mean absolute error (frames)", (
                _Series("generation_length_mae_frames", VALIDATION, "Validation", ORANGE),
            )),
            _Panel("Free generation hitting frame limit", "Validation utterances (%)", (
                _Series("generation_limit_rate", VALIDATION, "Validation", ORANGE, scale=100),
            )),
            _LEARNING_RATE,
        ),
    ),
    "train-mel": (
        "Stage 3 · Mel decoder",
        (
            _Panel("Mel prediction", "L1", _split_series("loss")),
            _LEARNING_RATE,
        ),
    ),
}


def _metric_values(records: list[dict]) -> dict[tuple[str, str], dict[int, float]]:
    values: dict[tuple[str, str], dict[int, float]] = {}
    for record in records:
        split, update = record.get("split"), record.get("update")
        if split not in (TRAIN, VALIDATION) or type(update) is not int:
            continue
        for name, value in record.items():
            if name != "update" and type(value) in (int, float) and math.isfinite(value):
                values.setdefault((split, name), {})[update] = value
        samples, hits = record.get("generation_samples"), record.get("generation_hit_limit")
        if type(samples) is int and samples > 0 and type(hits) is int and 0 <= hits <= samples:
            values.setdefault((split, "generation_limit_rate"), {})[update] = hits / samples
    return values


def plot_training(
    records: list[dict], stage: str, output: Path, *, min_content_distance: float | None = None,
) -> None:
    """Write a PNG for the available metrics and replace any previous plot atomically."""
    title, stage_panels = _STAGES[stage]
    values = _metric_values(records)
    panels = [panel for panel in stage_panels if any((s.split, s.metric) in values for s in panel.series)]
    if not panels:
        return

    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, MaxNLocator

    columns = min(math.ceil(math.sqrt(len(panels))), 3)
    rows = math.ceil(len(panels) / columns)
    figure = Figure(figsize=(4.5 * columns, 3.05 * rows + 0.55), dpi=150, layout="constrained")
    axes = figure.subplots(rows, columns, sharex=True, squeeze=False).ravel()
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.supxlabel("Optimizer update", fontsize=11)

    plotted_steps = {
        step for panel in panels for series in panel.series
        for step in values.get((series.split, series.metric), {})
    }
    if len(plotted_steps) == 1:
        axes[0].xaxis.set_major_locator(FixedLocator(list(plotted_steps)))
    else:
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    selection_metric = "regression_loss" if stage == "pretrain" else "loss"
    validation_scores = values.get((VALIDATION, selection_metric), {})
    selected_steps = sorted({
        record["update"] for record in records
        if record.get("split") == VALIDATION and record.get("best_checkpoint") is True
        and record.get("update") in validation_scores
    })
    thresholds = values.get((VALIDATION, "min_content_distance"), {})
    for panel, axis in zip(panels, axes):
        for series in panel.series:
            points = values.get((series.split, series.metric))
            if not points:
                continue
            updates, measurements = zip(*sorted(points.items()))
            is_validation = series.split == VALIDATION
            axis.plot(
                updates, [series.scale * value for value in measurements],
                color=series.color, label=series.label, linewidth=1.6,
                linestyle="--" if is_validation else "-",
                marker="." if is_validation or len(updates) == 1 else None,
                markersize=5,
            )
        if panel.series[0].metric == selection_metric and selected_steps:
            axis.scatter(
                selected_steps, [validation_scores[step] for step in selected_steps], s=25,
                color=GREEN, marker="D", linewidths=0.5, edgecolors="white", label="Saved best", zorder=4,
            )
        if panel is _CONTENT_DISTANCE:
            if len(set(thresholds.values())) > 1:
                steps, limits = zip(*sorted(thresholds.items()))
                axis.step(steps, limits, where="post", color="#4b5563", linestyle=":", label="Selection threshold")
            else:
                threshold = next(iter(thresholds.values()), min_content_distance)
                if threshold is not None:
                    axis.axhline(threshold, color="#4b5563", linewidth=1,
                                 linestyle=":", label=f"Selection threshold ({threshold:g})")
        axis.set_title(panel.title, fontsize=11, loc="left", fontweight="bold")
        axis.set_ylabel(panel.ylabel, fontsize=9)
        axis.tick_params(axis="both", labelsize=8, labelbottom=True)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.ticklabel_format(axis="y", scilimits=(-3, 3), useMathText=True)
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axis.get_legend_handles_labels()
        if len(handles) > 1:
            axis.legend(handles, labels, fontsize=8, framealpha=0.9)

    for axis in axes[len(panels):]:
        axis.set_visible(False)

    save_png(figure, output)
