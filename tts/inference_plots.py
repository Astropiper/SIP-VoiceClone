"""Diagnostic figures for feature extraction, resynthesis and synthesis."""

from pathlib import Path

import numpy as np
from torch import Tensor

from .data import MEL_SAMPLE_RATE
from .encoder import FIRST_FRAME_SECONDS, FRAME_STEP_SECONDS
from .mel import MEL_HOP
from .plotting import save_png


MEL_FRAME_SECONDS = MEL_HOP / MEL_SAMPLE_RATE


def _figure(height):
    from matplotlib.figure import Figure

    return Figure(figsize=(10, height), layout="constrained")


def _heatmap(axis, frames, step, first_frame=0, **style):
    start = first_frame - step / 2
    end = start + len(frames) * step
    image = axis.imshow(
        frames.T, extent=(start, end, -0.5, frames.shape[1] - 0.5), origin="lower",
        aspect="auto", interpolation="nearest", **style,
    )
    axis.set_xlim(max(0, start), end)
    return image


def save_feature_plot(features: Tensor, output: Path) -> None:
    values = features.detach().float().cpu().numpy()
    scale = max(float(np.abs(values).max()), 1e-6)
    figure = _figure(4)
    axis = figure.subplots()
    image = _heatmap(axis, values, FRAME_STEP_SECONDS, FIRST_FRAME_SECONDS, cmap="RdBu_r", vmin=-scale, vmax=scale)
    axis.set(title="EMA encoder features", xlabel="Time (s)", ylabel="Feature dimension")
    figure.colorbar(image, ax=axis, label="Activation")
    save_png(figure, output)


def save_reconstruction_plot(reference: Tensor, prediction: Tensor, output: Path) -> None:
    from matplotlib.colors import Normalize

    reference = reference.detach().float().cpu().numpy()
    prediction = prediction.detach().float().cpu().numpy()
    error = np.abs(prediction - reference)
    scale = Normalize(vmin=min(reference.min(), prediction.min()), vmax=max(reference.max(), prediction.max()))

    figure = _figure(7.5)
    reference_axis, prediction_axis, error_axis = figure.subplots(3, sharex=True)
    image = _heatmap(reference_axis, reference, MEL_FRAME_SECONDS, cmap="magma", norm=scale)
    _heatmap(prediction_axis, prediction, MEL_FRAME_SECONDS, cmap="magma", norm=scale)
    error_image = _heatmap(error_axis, error, MEL_FRAME_SECONDS, cmap="viridis", vmin=0, vmax=max(error.max(), 1e-6))
    reference_axis.set(title="Reference log-Mel", ylabel="Mel bin")
    prediction_axis.set(title="Reconstructed log-Mel", ylabel="Mel bin")
    error_axis.set(title=f"Absolute log-Mel error (MAE {error.mean():.4f})", xlabel="Time (s)", ylabel="Mel bin")
    figure.colorbar(image, ax=[reference_axis, prediction_axis], label="Log-Mel")
    figure.colorbar(error_image, ax=error_axis, label="Absolute error")
    save_png(figure, output)


def save_synthesis_plot(mel: Tensor, waveform: Tensor, output: Path) -> None:
    values = mel.detach().float().cpu().numpy()
    samples = waveform.detach().float().cpu().numpy()
    figure = _figure(5.5)
    mel_axis, waveform_axis = figure.subplots(2, sharex=True, height_ratios=(2, 1))
    image = _heatmap(mel_axis, values, MEL_FRAME_SECONDS, cmap="magma")
    mel_axis.set(title="Generated log-Mel", ylabel="Mel bin")
    waveform_axis.plot(np.arange(len(samples)) / MEL_SAMPLE_RATE, samples, color="#225f85", linewidth=0.75)
    waveform_axis.set(title="Synthesized waveform", xlabel="Time (s)", ylabel="Amplitude", ylim=(-1, 1))
    waveform_axis.set_xlim(0, max(len(samples) / MEL_SAMPLE_RATE, (len(values) - 0.5) * MEL_FRAME_SECONDS))
    figure.colorbar(image, ax=mel_axis, label="Log-Mel")
    save_png(figure, output)
