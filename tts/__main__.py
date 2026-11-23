"""Command-line interface for training and baseline synthesis."""

import argparse
import json
import math
from pathlib import Path

import soundfile
import torch

from .data import MEL_SAMPLE_RATE, load_clean_waveforms, load_waveform
from .encoder import FIRST_FRAME_SECONDS, FRAME_STEP_SECONDS, SAMPLE_RATE
from .inference_plots import save_feature_plot, save_reconstruction_plot, save_synthesis_plot
from .mel import MelDecoder, MelDecoderConfig, estimate_mel_lengths, load_vocos
from .text import TextAcousticModel, TextModelConfig, TextVocabulary
from .train import (
    MEL, PRETRAIN, TEXT, check_encoder, load_checkpoint, load_frozen_encoder,
    read_training_metrics, select_device, train,
)
from .training_plots import plot_training


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def _device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or a specific CUDA device")


def _encoder_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-checkpoint", required=True, type=Path, help="checkpoint from pretrain")


def _plot_arguments(parser: argparse.ArgumentParser) -> None:
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--plot-output", type=Path, help="PNG path (defaults to a file next to the output)")
    options.add_argument("--no-plot", action="store_true", help="skip the diagnostic figure")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stages = {
        PRETRAIN: "train the EMA waveform encoder on clips with background music",
        MEL: "train the Mel decoder on clean clips and a frozen encoder",
        TEXT: "train the autoregressive text model on transcribed clips and a frozen encoder",
    }
    for stage, help_text in stages.items():
        command = commands.add_parser(stage, help=help_text)
        command.set_defaults(run=train)
        command.add_argument("--device", help="auto, cpu, mps, cuda, or a specific CUDA device")
        command.add_argument("--train-manifest", type=Path)
        command.add_argument("--valid-manifest", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--model-config", dest="model_config_path", type=Path, help="model JSON (defaults to configs/model_config.json)")
        command.add_argument("--training-config", dest="training_config_path", type=Path, help="training JSON (defaults to configs/training_config.json)")
        command.add_argument("--updates", type=_positive_int, help="total optimizer updates")
        command.add_argument("--batch-size", type=_positive_int)
        command.add_argument("--accumulate", type=_positive_int, help="mini-batches per optimizer update")
        command.add_argument("--lr", type=_positive_float)
        command.add_argument("--weight-decay", type=float)
        command.add_argument("--seed", type=_nonnegative_int)
        command.add_argument("--workers", type=_nonnegative_int)
        command.add_argument("--log-every", type=_positive_int)
        command.add_argument("--eval-every", type=_positive_int)
        command.add_argument("--resume", type=Path, help="resume a checkpoint with the original training settings")
        plot = command.add_mutually_exclusive_group()
        plot.add_argument("--no-plot", action="store_true", default=None, help="skip training.png while saving training logs")
        plot.add_argument("--plot", dest="no_plot", action="store_false", help="write training.png")
        if stage != PRETRAIN:
            command.add_argument("--encoder-checkpoint", type=Path, help="checkpoint from pretrain")
        if stage == PRETRAIN:
            command.add_argument("--mask-fraction", type=float)
            command.add_argument("--mask-span", type=_positive_int)
            command.add_argument("--ema-decay", type=float)
            command.add_argument(
                "--variance-weight", type=float,
                help="local feature variance loss weight (0 disables)",
            )
            command.add_argument(
                "--min-content-distance", type=_positive_float,
                help="minimum aligned feature distance for both encoders to qualify for best.pt",
            )
            command.add_argument(
                "--background-manifest", type=Path, help="JSONL of independent accompaniment for the student input",
            )
            command.add_argument(
                "--background-probability", type=float,
                help="probability of mixing each training clip when a background manifest is supplied",
            )
            command.add_argument(
                "--background-snr-db", type=float, nargs=2, metavar=("MIN", "MAX"),
                help="range of original-to-added background ratios in dB",
            )
        if stage == TEXT:
            command.add_argument("--generation-samples", type=_nonnegative_int)
            command.add_argument("--max-frames", type=_positive_int)

    extract = commands.add_parser("extract", help="export the frozen EMA encoder's features for an audio clip")
    _device_argument(extract)
    _encoder_argument(extract)
    extract.add_argument("--audio", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    _plot_arguments(extract)
    extract.set_defaults(run=_extract)

    reconstruct = commands.add_parser("reconstruct", help="resynthesize an audio clip through the frozen encoder")
    _device_argument(reconstruct)
    _encoder_argument(reconstruct)
    reconstruct.add_argument("--mel-checkpoint", required=True, type=Path)
    reconstruct.add_argument("--audio", required=True, type=Path)
    reconstruct.add_argument("--output", required=True, type=Path)
    reconstruct.add_argument("--reference-output", type=Path, help="also vocode the clip's native reference Mel")
    _plot_arguments(reconstruct)
    reconstruct.set_defaults(run=_reconstruct)

    synthesize = commands.add_parser("synthesize", help="generate audio from new text")
    _device_argument(synthesize)
    synthesize.add_argument("--mel-checkpoint", required=True, type=Path)
    synthesize.add_argument("--text-checkpoint", required=True, type=Path)
    synthesize.add_argument("--text", required=True)
    synthesize.add_argument("--max-frames", type=_positive_int, default=500)
    synthesize.add_argument("--output", required=True, type=Path)
    _plot_arguments(synthesize)
    synthesize.set_defaults(run=_synthesize)

    report = commands.add_parser("plot-training", help="draw a PNG from saved training metrics")
    report.add_argument("--stage", choices=stages, required=True)
    report.add_argument("--metrics", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--min-content-distance", type=_positive_float, help="pretraining threshold for older logs")
    report.set_defaults(run=_plot_training_log)
    return parser


def _plot_training_log(args) -> None:
    if args.output.suffix.lower() != ".png" or args.output.resolve() == args.metrics.resolve():
        raise ValueError("The output must be a separate .png file")
    records = read_training_metrics(args.metrics)
    if not records:
        raise ValueError(f"No training or validation metrics found in {args.metrics}")
    plot_training(records, args.stage, args.output, min_content_distance=args.min_content_distance)


def _load_mel_decoder(path: Path, device: torch.device) -> tuple[MelDecoder, str]:
    checkpoint = load_checkpoint(path, MEL)
    model = MelDecoder(MelDecoderConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint["encoder_sha256"]


def _vocode(vocoder, mel: torch.Tensor, output_path: Path) -> torch.Tensor:
    waveform = vocoder.decode(mel.transpose(1, 2).contiguous().cpu()).squeeze(0).clamp(-1, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(output_path, waveform.numpy(), MEL_SAMPLE_RATE)
    return waveform


def _plot_path(args, inputs: tuple[Path, ...], *extra_outputs: Path) -> Path | None:
    occupied = {path.resolve() for path in inputs}
    for path in (args.output, *extra_outputs):
        resolved = path.resolve()
        if resolved in occupied:
            raise ValueError("Outputs must be separate from each other, input files and checkpoints")
        occupied.add(resolved)
    if args.no_plot:
        return None

    plot = args.plot_output
    if plot is None:
        plot = args.output.with_suffix(".png")
        if plot.resolve() in occupied:
            plot = args.output.with_name(f"{args.output.stem}.plot.png")
    if plot.suffix.lower() != ".png":
        raise ValueError("The plot output path must end in .png")
    if plot.resolve() in occupied:
        raise ValueError("The plot must be separate from inputs, checkpoints and other outputs")
    return plot


@torch.no_grad()
def _extract(args) -> None:
    plot = _plot_path(args, (args.audio, args.encoder_checkpoint))
    device = select_device(args.device)
    encoder, encoder_id = load_frozen_encoder(args.encoder_checkpoint, device)
    waveform = load_waveform(args.audio).unsqueeze(0).to(device)
    features, _ = encoder(waveform, torch.tensor([waveform.size(1)]))
    features = features.squeeze(0).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": features,
        "encoder_sha256": encoder_id,
        "sample_rate": SAMPLE_RATE,
        "frame_step_seconds": FRAME_STEP_SECONDS,
        "first_frame_seconds": FIRST_FRAME_SECONDS,
    }, args.output)
    if plot is not None:
        save_feature_plot(features, plot)


@torch.no_grad()
def _reconstruct(args) -> None:
    extra_outputs = () if args.reference_output is None else (args.reference_output,)
    plot = _plot_path(args, (args.audio, args.encoder_checkpoint, args.mel_checkpoint), *extra_outputs)
    device = select_device(args.device)
    encoder, encoder_id = load_frozen_encoder(args.encoder_checkpoint, device)
    decoder, decoder_encoder_id = _load_mel_decoder(args.mel_checkpoint, device)
    check_encoder({"encoder_sha256": decoder_encoder_id}, encoder_id)
    waveform, mel_waveform = load_clean_waveforms(args.audio)
    length = torch.tensor([waveform.numel()])
    features, valid = encoder(waveform.unsqueeze(0).to(device), length)
    vocoder = load_vocos()
    reference = vocoder.feature_extractor(mel_waveform.unsqueeze(0)).transpose(1, 2)
    mel_lengths = torch.tensor([reference.size(1)], device=device)
    prediction = decoder(features, valid.sum(dim=1), mel_lengths)
    _vocode(vocoder, prediction, args.output)
    if args.reference_output:
        _vocode(vocoder, reference, args.reference_output)
    if plot is not None:
        save_reconstruction_plot(reference.squeeze(0), prediction.squeeze(0), plot)


@torch.no_grad()
def _synthesize(args) -> None:
    plot = _plot_path(args, (args.text_checkpoint, args.mel_checkpoint))
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.text_checkpoint, TEXT)
    decoder, encoder_id = _load_mel_decoder(args.mel_checkpoint, device)
    check_encoder(checkpoint, encoder_id)
    text_model = TextAcousticModel(TextModelConfig(**checkpoint["model_config"]))
    text_model.load_state_dict(checkpoint["model"])
    text_model.to(device).eval()
    vocab = TextVocabulary(checkpoint["vocab"])
    tokens = torch.tensor([vocab.encode(args.text)], device=device)
    lengths = torch.tensor([tokens.size(1)], device=device)
    generation = text_model.generate(tokens, lengths, max_frames=args.max_frames)
    mel_lengths = estimate_mel_lengths(generation.lengths)
    mel = decoder(generation.features, generation.lengths, mel_lengths)
    waveform = _vocode(load_vocos(), mel, args.output)
    if plot is not None:
        save_synthesis_plot(mel.squeeze(0), waveform, plot)
    print(json.dumps({
        "generated_frames": generation.lengths.item(), "hit_limit": generation.hit_limit.item(),
    }))


def main() -> None:
    args = _parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
