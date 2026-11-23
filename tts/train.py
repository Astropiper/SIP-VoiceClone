"""Training entry points and checkpoint handling for the three baseline stages."""

from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
from itertools import islice
from pathlib import Path
import json
import math
import os
import sys
import tempfile
import traceback

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .config import (
    DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, MEL, PRETRAIN, TEXT, read_config, resolve_training_config,
)
from .data import (
    BackgroundMix, CleanDataset, Clip,
    TextDataset, WaveformDataset, collate_clean, collate_text, collate_waveforms,
    read_manifest, validate_disjoint_sources,
)
from .encoder import (
    DIAGNOSTIC_CLIPS, AcousticEncoder, EncoderConfig,
    PretrainingModel, diagnostics, frame_lengths,
)
from .mel import MelDecoder, MelDecoderConfig, extract_mel_targets, make_mel_extractor, mel_valid_mask
from .text import TextAcousticModel, TextModelConfig, TextVocabulary
from .training_plots import plot_training


CHECKPOINT_VERSION = 1
Batch = dict[str, Tensor]


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    device = torch.device(name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver, the CUDA-enabled PyTorch installation, "
            "and CUDA_VISIBLE_DEVICES."
        )
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    count = torch.cuda.device_count()
    if device.index >= count:
        raise ValueError(f"CUDA device {device} is out of range ({count} visible CUDA devices)")
    return device


def _check_checkpoint(checkpoint: dict, stage: str) -> dict:
    if checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("stage") != stage:
        raise ValueError(f"Expected a version {CHECKPOINT_VERSION} checkpoint from {stage}")
    return checkpoint


def load_checkpoint(path: Path, stage: str) -> dict:
    return _check_checkpoint(torch.load(path, map_location="cpu", weights_only=True), stage)


def load_frozen_encoder(path: Path, device: torch.device) -> tuple[AcousticEncoder, str]:
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        checkpoint = _check_checkpoint(torch.load(stream, map_location="cpu", weights_only=True), PRETRAIN)

    encoder = AcousticEncoder(EncoderConfig(**checkpoint["model_config"]))
    prefix = "teacher."
    encoder.load_state_dict({
        name.removeprefix(prefix): weight for name, weight in checkpoint["model"].items()
        if name.startswith(prefix)
    })
    return encoder.to(device).eval().requires_grad_(False), digest.hexdigest()


def check_encoder(checkpoint: dict, encoder_id: str) -> None:
    if checkpoint["encoder_sha256"] != encoder_id:
        raise ValueError("The supplied encoder checkpoint differs from the one used for this model")


def _save_checkpoint(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _rng_state(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state(), "device": device.type}
    if device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    if device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if state["device"] == device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)
    if state["device"] == device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])


@dataclass
class StepResult:
    loss: Tensor
    frames: int
    metrics: dict[str, float]
    counts: dict[str, int]


@dataclass
class Task:
    model: nn.Module
    training: Dataset
    validation: Dataset
    collate: Callable
    step: Callable[[Batch, torch.Generator | None], StepResult]
    metadata: dict
    update_teacher: Callable[[], None] | None = None
    evaluate_samples: Callable[[list[Batch]], dict[str, float]] | None = None


def _save_config_copy(path: Path, content: bytes) -> None:
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary_path = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_config(path: Path, config: dict) -> None:
    content = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _save_config_copy(path, content.encode("utf-8"))


def _resolve_model_config(config_type, options: dict, checkpoint: dict | None, **derived):
    field_types = {field.name: field.type for field in fields(config_type)}
    unknown = options.keys() - field_types.keys()
    if unknown:
        raise ValueError(f"Unknown model config fields for {config_type.__name__}: {', '.join(sorted(unknown))}")
    for name, value in options.items():
        if field_types[name] is int:
            minimum = 0 if config_type is MelDecoderConfig and name == "layers" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"Model config {name} must be an integer >= {minimum}")
        elif type(value) not in (int, float) or not -math.inf < value < math.inf:
            raise ValueError(f"Model config {name} must be a finite number")
    if "dropout" in options and not 0 <= options["dropout"] <= 1:
        raise ValueError("Model config dropout must be between 0 and 1")
    if "stop_pos_weight" in options and options["stop_pos_weight"] <= 0:
        raise ValueError("Model config stop_pos_weight must be positive")
    for name, value in derived.items():
        if name in options and options[name] != value:
            raise ValueError(f"Model config {name} is determined by the data or encoder checkpoint and must be {value}")

    values = options
    if checkpoint is not None:
        values = checkpoint["model_config"]
        for name, value in options.items():
            if values.get(name) != value:
                raise ValueError(f"Model config {name} must match the resume checkpoint ({values.get(name)})")
    config = config_type(**(values | derived))
    if isinstance(config, (EncoderConfig, TextModelConfig)) and config.dim % config.heads:
        raise ValueError("Model config dim must be divisible by heads")
    return config


def _prepare_background_mix(
    args, training: list[Clip], validation: list[Clip],
) -> tuple[BackgroundMix | None, dict | None]:
    path = getattr(args, "background_manifest", None)
    probability = getattr(args, "background_probability", None)
    snr = getattr(args, "background_snr_db", None)
    if path is None:
        return None, None

    backgrounds = read_manifest(path)
    validate_disjoint_sources(training, validation, backgrounds)
    snr_db = tuple(snr)
    mix = BackgroundMix(backgrounds, probability, snr_db)
    return (mix if probability else None), {
        "manifest_sha256": sha256(path.expanduser().read_bytes()).hexdigest(),
        "probability": probability,
        "snr_db": list(snr_db),
    }


def _prepare_task(args, device: torch.device, options: dict, checkpoint: dict | None = None) -> Task:
    training = read_manifest(args.train_manifest)
    validation = read_manifest(args.valid_manifest)
    validate_disjoint_sources(training, validation)

    if args.command == PRETRAIN:
        background_mix, background_config = _prepare_background_mix(args, training, validation)
        config = _resolve_model_config(EncoderConfig, options, checkpoint)
        variance_weight = args.variance_weight
        model = PretrainingModel(
            config, mask_fraction=args.mask_fraction, mask_span=args.mask_span, variance_weight=variance_weight,
        ).to(device)

        def step(batch, generator):
            waveforms, lengths = batch["waveforms"], batch["lengths"]
            student_waveforms = None
            counts = {}
            if model.training and background_mix is not None:
                mixed, mixed_clips = background_mix(waveforms, lengths)
                if mixed_clips:
                    student_waveforms = mixed.to(device)
                counts.update(background_mixed_clips=mixed_clips, background_total_clips=len(lengths))
            result = model(waveforms.to(device), lengths, generator=generator, student_waveforms=student_waveforms)
            counts.update(
                masked_frames=result.masked_frames, valid_frames=result.valid_frames, mask_spans=result.span_count,
            )
            return StepResult(result.loss, result.masked_frames, {
                "regression_loss": result.regression_loss.detach().item(),
                "variance_loss": result.variance_loss.detach().item(),
                "masked_target_std": result.masked_target_std.item(),
                "masked_prediction_std": result.masked_prediction_std.item(),
            }, counts)

        def representation_stats(batches):
            return diagnostics(model, (
                (batch["waveforms"].to(device), batch["lengths"]) for batch in batches
            ))

        return Task(
            model, WaveformDataset(training), WaveformDataset(validation), collate_waveforms,
            step, {"model_config": asdict(config), "mask_fraction": args.mask_fraction,
                   "mask_span": args.mask_span, "ema_decay": args.ema_decay, "variance_weight": variance_weight,
                   "background_augmentation": background_config},
            update_teacher=lambda: model.update_teacher(args.ema_decay),
            evaluate_samples=representation_stats,
        )

    encoder, encoder_id = load_frozen_encoder(args.encoder_checkpoint, device)

    @torch.no_grad()
    def encode(batch):
        features, valid = encoder(batch["waveforms"].to(device), batch["lengths"])
        return features, valid.sum(dim=1)

    if args.command == TEXT:
        vocab = TextVocabulary.build(clip.text for clip in training if clip.text is not None)
        config = _resolve_model_config(
            TextModelConfig, options, checkpoint, vocab_size=len(vocab), acoustic_dim=encoder.config.dim,
        )
        model = TextAcousticModel(config).to(device)

        def step(batch, _generator):
            targets, lengths = encode(batch)
            output = model(batch["tokens"].to(device), batch["token_lengths"].to(device), targets, lengths)
            return StepResult(output.loss, int(lengths.sum().item()), {
                "acoustic_l1": output.acoustic_loss.detach().item(),
                "stop_bce": output.stop_loss.detach().item(),
            }, {})

        def generation_stats(batches):
            absolute_error = hit_limit = samples = 0
            for batch in batches:
                remaining = args.generation_samples - samples
                tokens = batch["tokens"][:remaining].to(device)
                lengths = batch["token_lengths"][:remaining].to(device)
                output = model.generate(tokens, lengths, max_frames=args.max_frames)
                target_lengths = frame_lengths(batch["lengths"][:remaining]).to(device)
                absolute_error += int((output.lengths - target_lengths).abs().sum().item())
                hit_limit += int(output.hit_limit.sum().item())
                samples += len(tokens)
            return {
                "generation_samples": samples,
                "generation_length_mae_frames": absolute_error / samples,
                "generation_hit_limit": hit_limit,
            }

        return Task(
            model, TextDataset(training, vocab.encode), TextDataset(validation, vocab.encode),
            collate_text, step,
            {"model_config": asdict(config), "encoder_sha256": encoder_id, "vocab": list(vocab.tokens)},
            evaluate_samples=generation_stats if args.generation_samples else None,
        )

    config = _resolve_model_config(MelDecoderConfig, options, checkpoint, acoustic_dim=encoder.config.dim)
    model = MelDecoder(config).to(device)
    extractor = make_mel_extractor()

    def step(batch, _generator):
        features, lengths = encode(batch)
        targets, mel_lengths = extract_mel_targets(extractor, batch["mel_waveforms"], batch["mel_waveform_lengths"])
        targets, mel_lengths = targets.to(device), mel_lengths.to(device)
        prediction = model(features, lengths, mel_lengths, targets.shape[1])
        valid = mel_valid_mask(mel_lengths, targets.shape[1])
        error = F.l1_loss(prediction, targets, reduction="none").mean(dim=-1)
        return StepResult(
            error.masked_fill(~valid, 0).sum() / valid.sum(),
            int(mel_lengths.sum().item()), {}, {},
        )

    return Task(
        model, CleanDataset(training), CleanDataset(validation), collate_clean, step,
        {"model_config": asdict(config), "encoder_sha256": encoder_id},
    )


class Metrics:
    def __init__(self):
        self.frames = 0
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, result: StepResult) -> None:
        self.frames += result.frames
        self.sums["loss"] += result.loss.detach().item() * result.frames
        for name, value in result.metrics.items():
            self.sums[name] += value * result.frames
        for name, count in result.counts.items():
            self.counts[name] += count

    def values(self) -> dict[str, float]:
        values = {name: value / self.frames for name, value in self.sums.items()}
        if self.counts:
            values.update(
                mask_coverage=self.counts["masked_frames"] / self.counts["valid_frames"],
                mask_mean_span=self.counts["masked_frames"] / self.counts["mask_spans"],
            )
        if self.counts.get("background_total_clips"):
            values["background_mix_fraction"] = self.counts["background_mixed_clips"] / self.counts["background_total_clips"]
        return values


def _schedule(update: int, total: int) -> float:
    warmup = max(1, int(total * 0.05))
    if update < warmup:
        return (update + 1) / warmup
    if update >= total:
        return 0.0
    progress = (update - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _loader(dataset, collate, workers, seed, *, device: torch.device, **batch_options) -> DataLoader:
    # Workers start after the model has already initialized CUDA in the parent.
    context = "spawn" if workers and device.type == "cuda" else None
    return DataLoader(
        dataset, collate_fn=collate, num_workers=workers,
        multiprocessing_context=context,
        generator=torch.Generator().manual_seed(seed), **batch_options,
    )


@torch.no_grad()
def _validate(task: Task, args, device: torch.device) -> dict[str, float]:
    task.model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    loader = _loader(task.validation, task.collate, args.workers, args.seed, device=device, batch_size=args.batch_size)
    sample_batch_count = 0
    if args.command == PRETRAIN:
        sample_batch_count = math.ceil(DIAGNOSTIC_CLIPS / args.batch_size)
    if args.command == TEXT:
        sample_batch_count = math.ceil(args.generation_samples / args.batch_size)
    samples = []
    metrics = Metrics()
    for index, batch in enumerate(loader):
        metrics.add(task.step(batch, generator))
        if index < sample_batch_count:
            samples.append(batch)
    values = metrics.values()
    if task.evaluate_samples is not None:
        values.update(task.evaluate_samples(samples))
    task.model.train()
    return values


def read_training_metrics(path: Path, through_update: int | None = None) -> list[dict]:
    """Read metrics from a run log or a captured console stream in update order."""
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("rejected_checkpoint"):
                continue
            split = record.get("split")
            update = record.get("update")
            loss = record.get("loss")
            if split not in ("train", "validation") or type(update) is not int or update < 0:
                continue
            if type(loss) not in (int, float) or any(
                isinstance(value, float) and not math.isfinite(value) for value in record.values()
            ):
                continue
            if through_update is None or update <= through_update:
                records[update, split] = record
    return [records[key] for key in sorted(records)]


def _prepare_metrics_log(path: Path, resume: Path | None, update: int) -> list[dict]:
    history = []
    if resume is not None:
        source = resume.parent / path.name
        if source.is_file():
            history = read_training_metrics(source, through_update=update)

    stream = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary_path = Path(stream.name)
    try:
        with stream:
            for record in history:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return history


def _log(split: str, update: int, metrics: dict, path: Path | None = None, **extra) -> dict:
    record = {"split": split, "update": update, **metrics, **extra}
    line = json.dumps(record, allow_nan=False)
    if path is not None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    print(line, flush=True)
    return record


def _content_check_passed(values: dict, minimum: float) -> bool:
    return all(values[f"{name}_aligned_content_distance"] >= minimum for name in ("student", "teacher"))


def _resume_best_loss(task: Task, args, path: Path, settings: dict, device: torch.device) -> float:
    checkpoint = load_checkpoint(path, args.command)
    if args.command == PRETRAIN:
        checkpoint.setdefault("variance_weight", 0.0)
    if checkpoint["settings"] != settings or any(
        checkpoint.get(key) != value for key, value in task.metadata.items()
    ):
        raise ValueError(f"Existing {path} was trained with different settings. Resume into a new output directory.")
    if args.command != PRETRAIN or checkpoint.get("min_content_distance") == args.min_content_distance:
        return checkpoint["best_loss"]

    task.model.load_state_dict(checkpoint["model"])
    values = _validate(task, args, device)
    passed = _content_check_passed(values, args.min_content_distance)
    if passed:
        return values["regression_loss"]

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="rejected-best-", suffix=".pt", delete=False) as stream:
        rejected = Path(stream.name)
    try:
        path.replace(rejected)
    except OSError:
        rejected.unlink(missing_ok=True)
        raise
    _log("validation", checkpoint["update"], values, content_check_passed=False, rejected_checkpoint=rejected.name)
    return math.inf


class _LoggedStream:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, message: str) -> int:
        self.log.write(message)
        return self.console.write(message)

    def flush(self) -> None:
        self.log.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def train(args) -> None:
    args = Namespace(**vars(args))
    requested_training = getattr(args, "training_config_path", None)
    training_source = (requested_training or DEFAULT_TRAINING_CONFIG).expanduser().absolute()
    defaults, default_content = read_config(DEFAULT_TRAINING_CONFIG, args.command, training=True)
    if training_source == DEFAULT_TRAINING_CONFIG:
        training_options, training_content = {}, default_content
    else:
        training_options, training_content = read_config(training_source, args.command, training=True)

    resume = args.resume.expanduser() if args.resume is not None else None
    checkpoint = load_checkpoint(resume, args.command) if resume is not None else None
    if checkpoint is not None and args.command == PRETRAIN:
        checkpoint.setdefault("variance_weight", 0.0)
    args.resume = resume
    vars(args).update(resolve_training_config(args, checkpoint, defaults, training_options))

    requested_model = getattr(args, "model_config_path", None)
    model_source = None
    model_content = None
    model_options = {}
    if requested_model is not None or checkpoint is None:
        model_source = (requested_model or DEFAULT_MODEL_CONFIG).expanduser().absolute()
        model_options, model_content = read_config(model_source, args.command)
        if checkpoint is None and model_source != DEFAULT_MODEL_CONFIG:
            default_model, _ = read_config(DEFAULT_MODEL_CONFIG, args.command)
            model_options = default_model | model_options

    log_path = args.output_dir / "train.log"
    model_config_path = args.output_dir / "model_config.json"
    training_config_path = args.output_dir / "training_config.json"
    originals = args.output_dir / "input_configs"
    model_copy = originals / model_config_path.name
    training_copy = originals / training_config_path.name
    artifacts = {(args.output_dir / name).resolve() for name in ("last.pt", "best.pt", "metrics.jsonl", "training.png")}
    report_paths = (log_path, model_config_path, training_config_path, model_copy, training_copy)
    report_targets = [path.resolve() for path in report_paths]
    inputs = (args.train_manifest, args.valid_manifest, args.resume, getattr(args, "encoder_checkpoint", None),
              getattr(args, "background_manifest", None))
    input_targets = {path.expanduser().resolve() for path in inputs if path is not None}
    for path, target in zip(report_paths, report_targets):
        if target in artifacts or target in input_targets or report_targets.count(target) > 1:
            raise ValueError(f"{path.name} must be separate from training inputs and other outputs")
    for source, snapshot, copy in ((model_source, model_config_path, model_copy),
                                   (training_source, training_config_path, training_copy)):
        if source is None:
            continue
        target = source.resolve()
        allowed = {copy.resolve()}
        if resume is not None:
            allowed.add(snapshot.resolve())
        if target in (artifacts.union(report_targets) - allowed):
            raise ValueError(f"Config input {source} must be separate from training outputs")
    defaults_targets = {DEFAULT_MODEL_CONFIG.resolve(), DEFAULT_TRAINING_CONFIG.resolve()}
    if defaults_targets.intersection(artifacts.union(report_targets)):
        raise ValueError("Training outputs must be separate from the default configuration files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {key: value for key, value in vars(args).items() if key != "run"}
    with log_path.open("a", encoding="utf-8", errors="backslashreplace", buffering=1) as log:
        print(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Started {args.command}", file=log)
        print(json.dumps(config, ensure_ascii=False, default=str), file=log)
        with redirect_stdout(_LoggedStream(sys.stdout, log)), redirect_stderr(_LoggedStream(sys.stderr, log)):
            try:
                _train(args, log, checkpoint, model_options, model_config_path, training_config_path,
                       model_content, training_content)
            except BaseException:
                traceback.print_exc(file=log)
                raise
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Finished {args.command}", file=log)


def _train(args, log, checkpoint: dict | None, model_options: dict, model_config_path: Path,
           training_config_path: Path, model_content: bytes | None, training_content: bytes) -> None:
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    task = _prepare_task(args, device, model_options, checkpoint)
    model_config = task.metadata["model_config"]
    print(json.dumps({"model_config": model_config}), file=log)
    model = task.model
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: _schedule(update, args.updates))
    settings = {name: getattr(args, name) for name in (
        "updates", "batch_size", "accumulate", "lr", "weight_decay", "seed",
    )}
    settings["manifest_sha256"] = [
        sha256(manifest.read_bytes()).hexdigest() for manifest in (args.train_manifest, args.valid_manifest)
    ]
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    metrics_path = args.output_dir / "metrics.jsonl"
    plot_path = args.output_dir / "training.png"
    inputs = [args.train_manifest, args.valid_manifest, args.resume, getattr(args, "encoder_checkpoint", None),
              getattr(args, "background_manifest", None)]
    report_paths = [metrics_path] if args.no_plot else [metrics_path, plot_path]
    if {path.resolve() for path in report_paths}.intersection(path.expanduser().resolve() for path in inputs if path is not None):
        raise ValueError("Training reports must be separate from manifests and input checkpoints")
    update = epoch = next_batch = 0
    best = math.inf
    if checkpoint is not None:
        if checkpoint["settings"] != settings or any(
            checkpoint.get(key) != value for key, value in task.metadata.items()
        ):
            raise ValueError("Resuming requires the original training settings, manifests and model configuration")
        if best_path.exists():
            best = _resume_best_loss(task, args, best_path, settings, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        update, epoch, next_batch = (checkpoint[key] for key in ("update", "epoch", "next_batch"))
        _restore_rng(checkpoint["rng"], device)
        checkpoint.clear()
    elif last_path.exists() or best_path.exists():
        raise FileExistsError(f"{args.output_dir} already contains a training checkpoint. Use --resume or a new directory.")

    training_config = {
        key: str(value.expanduser().absolute()) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key != "run"
    }
    training_config.update(
        requested_device=args.device, device=str(device), manifest_sha256=settings["manifest_sha256"],
    )
    if args.command == PRETRAIN:
        training_config["variance_weight"] = task.metadata["variance_weight"]
        training_config["background_augmentation"] = task.metadata["background_augmentation"]

    _save_config(model_config_path, model_config)
    _save_config(training_config_path, training_config)
    originals = args.output_dir / "input_configs"
    originals.mkdir(exist_ok=True)
    model_copy = originals / model_config_path.name
    if model_content is not None:
        _save_config_copy(model_copy, model_content)
    elif not model_copy.exists():
        _save_config(model_copy, model_config)
    _save_config_copy(originals / training_config_path.name, training_content)
    history = _prepare_metrics_log(metrics_path, args.resume, update)
    minimum = args.min_content_distance if args.command == PRETRAIN else None
    if not args.no_plot and update >= args.updates and history:
        plot_training(history, args.command, plot_path, min_content_distance=minimum)

    model.train()
    metrics = Metrics()
    while update < args.updates:
        seed = args.seed + epoch
        order = torch.randperm(len(task.training), generator=torch.Generator().manual_seed(seed)).tolist()
        batches = list(BatchSampler(order, args.batch_size, drop_last=False))
        loader = _loader(task.training, task.collate, args.workers, seed, device=device, batch_sampler=batches[next_batch:])
        iterator = iter(loader)
        while next_batch < len(batches) and update < args.updates:
            optimizer.zero_grad(set_to_none=True)
            frames = 0
            for batch in islice(iterator, args.accumulate):
                result = task.step(batch, None)
                if not torch.isfinite(result.loss).item():
                    raise FloatingPointError(f"Non-finite training loss before update {update + 1}")
                (result.loss * result.frames).backward()
                metrics.add(result)
                frames += result.frames
                next_batch += 1
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(frames)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if task.update_teacher is not None:
                task.update_teacher()
            scheduler.step()
            update += 1
            if update % args.log_every == 0 or update == args.updates:
                history.append(_log("train", update, metrics.values(), metrics_path, lr=lr))
                metrics = Metrics()
            if update % args.eval_every == 0 or update == args.updates:
                values = _validate(task, args, device)
                score = values["loss"]
                eligible = True
                if args.command == PRETRAIN:
                    score = values["regression_loss"]
                    eligible = _content_check_passed(values, minimum)
                    values["content_check_passed"] = eligible
                    values["min_content_distance"] = minimum
                improved = eligible and score < best
                if improved:
                    best = score
                payload = {
                    "version": CHECKPOINT_VERSION, "stage": args.command, **task.metadata,
                    "settings": settings, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": _rng_state(device), "best_loss": best,
                    "update": update, "epoch": epoch, "next_batch": next_batch,
                }
                if args.command == PRETRAIN:
                    payload["min_content_distance"] = minimum
                if improved:
                    _save_checkpoint(best_path, payload)
                _save_checkpoint(last_path, payload)
                history.append(_log("validation", update, values, metrics_path, best_checkpoint=improved))
                if not args.no_plot:
                    plot_training(history, args.command, plot_path, min_content_distance=minimum)
        epoch += 1
        next_batch = 0
