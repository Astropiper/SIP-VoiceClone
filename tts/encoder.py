"""Waveform encoder and EMA masked prediction for stage one."""

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SAMPLE_RATE = 16_000
KERNEL_SIZES = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)
FRAME_STRIDE = math.prod(STRIDES)
RECEPTIVE_FIELD = 1 + sum(
    (kernel - 1) * math.prod(STRIDES[:i]) for i, kernel in enumerate(KERNEL_SIZES)
)
FRAME_STEP_SECONDS = FRAME_STRIDE / SAMPLE_RATE
FIRST_FRAME_SECONDS = RECEPTIVE_FIELD / (2 * SAMPLE_RATE)
DEFAULT_VARIANCE_WEIGHT = 0.1
LOCAL_STD_FLOOR = 0.1
DIAGNOSTIC_CLIPS = 8


def frame_lengths(lengths: Tensor) -> Tensor:
    """Return the number of full waveform windows produced by the convolutions."""
    return (1 + (lengths - RECEPTIVE_FIELD) // FRAME_STRIDE).clamp_min(0)


def sinusoidal_positions(sequence: Tensor) -> Tensor:
    positions = torch.arange(sequence.size(-2), device=sequence.device, dtype=torch.float32)
    dim = sequence.size(-1)
    channels = torch.arange(0, dim, 2, device=sequence.device, dtype=torch.float32)
    angles = positions[:, None] * torch.exp(channels * (-math.log(10_000) / dim))
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :dim].to(sequence.dtype)


def _frame_counts(waveforms: Tensor, lengths: Tensor) -> list[int]:
    if waveforms.ndim != 2 or lengths.shape != (waveforms.size(0),) or waveforms.size(0) == 0:
        raise ValueError("Expected nonempty waveforms (batch, samples) and lengths (batch,)")
    sample_counts = lengths.tolist()
    if any(not RECEPTIVE_FIELD <= n <= waveforms.size(-1) for n in sample_counts):
        raise ValueError(f"Each waveform must have at least {RECEPTIVE_FIELD} valid samples")
    return [1 + (n - RECEPTIVE_FIELD) // FRAME_STRIDE for n in sample_counts]


def _valid_frames(features: Tensor, counts: list[int]) -> Tensor:
    lengths = torch.tensor(counts, device=features.device)
    return torch.arange(features.size(1), device=features.device)[None, :] < lengths[:, None]


@dataclass(frozen=True)
class EncoderConfig:
    conv_channels: int = 128
    dim: int = 256
    layers: int = 6
    heads: int = 4
    ff_dim: int = 1024
    teacher_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 1 <= self.teacher_layers <= self.layers:
            raise ValueError("teacher_layers must be between 1 and the number of encoder layers")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x).transpose(1, 2)
        return F.gelu(self.norm(x)).transpose(1, 2)


class _TransformerBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.dim)
        self.attention = nn.MultiheadAttention(
            config.dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(config.dim)
        self.ff_in = nn.Linear(config.dim, config.ff_dim)
        self.ff_out = nn.Linear(config.ff_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, padding: Tensor) -> tuple[Tensor, Tensor]:
        normed = self.attention_norm(x)
        attended, _ = self.attention(
            normed, normed, normed, key_padding_mask=padding, need_weights=False
        )
        x = x + self.dropout(attended)
        ff = self.ff_out(self.dropout(F.gelu(self.ff_in(self.ff_norm(x)))))
        return x + self.dropout(ff), ff


class AcousticEncoder(nn.Module):
    """Export unmasked contextual features and their valid-frame mask."""

    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config
        self.conv = nn.Sequential(
            *(
                _ConvBlock(1 if i == 0 else config.conv_channels, config.conv_channels, kernel, stride)
                for i, (kernel, stride) in enumerate(zip(KERNEL_SIZES, STRIDES))
            )
        )
        self.projection = nn.Linear(config.conv_channels, config.dim)
        self.blocks = nn.ModuleList(_TransformerBlock(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(config.dim)

    def _extract(self, waveforms: Tensor) -> Tensor:
        local = self.conv(waveforms[:, None, :]).transpose(1, 2)
        return self.projection(local)

    def _contextualize(
        self, local: Tensor, valid: Tensor, target_layers: int = 0
    ) -> tuple[Tensor, list[Tensor]]:
        x = local + sinusoidal_positions(local)
        padding = ~valid
        targets = []
        for i, block in enumerate(self.blocks):
            x, ff = block(x, padding)
            if i >= len(self.blocks) - target_layers:
                targets.append(ff)
        return self.output_norm(x).masked_fill(padding[:, :, None], 0), targets

    def forward(self, waveforms: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        counts = _frame_counts(waveforms, lengths)
        local = self._extract(waveforms)
        valid = _valid_frames(local, counts)
        features, _ = self._contextualize(local, valid)
        return features, valid


def _sample_mask(
    counts: list[int], total_frames: int, fraction: float, span: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, int, int]:
    """Sample separated, nonpadding runs with a bounded length and a per-clip budget."""
    mask = torch.zeros(len(counts), total_frames, dtype=torch.bool)
    masked_frames = span_count = 0
    for row, count in enumerate(counts):
        # Leave at least one frame visible and preserve a gap between consecutive runs.
        budget = min(max(1, int(count * fraction + 0.5)), count - 1, count - count // (span + 1))
        runs = (budget + span - 1) // span
        sizes = [span] * runs
        if budget % span:
            short_run = torch.randint(runs, (), generator=generator).item()
            sizes[short_run] = budget % span

        # Randomly distribute the remaining visible frames around the separated runs.
        slack = count - budget - (runs - 1)
        cuts = torch.randint(slack + 1, (runs,), generator=generator).sort().values.tolist()
        visible_before = 0
        cursor = 0
        for cut, size in zip(cuts, sizes):
            cursor += cut - visible_before
            mask[row, cursor:cursor + size] = True
            cursor += size + 1
            visible_before = cut
        masked_frames += budget
        span_count += runs
    return mask, masked_frames, span_count


def _teacher_target(layers: list[Tensor], valid: Tensor) -> Tensor:
    padding = ~valid[:, :, None]
    counts = valid.sum(dim=1).float()[:, None, None]
    normalized = []
    for layer in layers:
        values = layer.float().masked_fill(padding, 0)
        mean = values.sum(dim=1, keepdim=True) / counts
        centered = (values - mean).masked_fill(padding, 0)
        variance = centered.square().sum(dim=1, keepdim=True) / counts
        normalized.append(centered * torch.rsqrt(variance + 1e-5))
    return torch.stack(normalized).mean(dim=0)


def _frame_variance(features: Tensor, included: Tensor, count: int) -> Tensor:
    excluded = ~included[:, :, None]
    values = features.float().masked_fill(excluded, 0)
    mean = values.sum(dim=(0, 1)) / count
    centered = (values - mean).masked_fill(excluded, 0)
    return centered.square().sum(dim=(0, 1)) / count


@dataclass
class PretrainingOutput:
    loss: Tensor
    masked_frames: int
    valid_frames: int
    span_count: int
    regression_loss: Tensor
    variance_loss: Tensor
    masked_target_std: Tensor
    masked_prediction_std: Tensor


class PretrainingModel(nn.Module):
    def __init__(
        self, config: EncoderConfig = EncoderConfig(), mask_fraction: float = 0.5, mask_span: int = 10,
        variance_weight: float = DEFAULT_VARIANCE_WEIGHT,
    ):
        super().__init__()
        if not 0 < mask_fraction < 1 or mask_span < 1:
            raise ValueError("Expected 0 < mask_fraction < 1 and mask_span >= 1")
        if not 0 <= variance_weight < math.inf:
            raise ValueError("Variance weight must be finite and nonnegative")
        self.config = config
        self.mask_fraction = mask_fraction
        self.mask_span = mask_span
        self.variance_weight = variance_weight
        self.student = AcousticEncoder(config)
        self.teacher = copy.deepcopy(self.student).requires_grad_(False).eval()
        self.mask_embedding = nn.Parameter(torch.empty(config.dim))
        nn.init.normal_(self.mask_embedding, std=0.02)
        self.predictor = nn.Linear(config.dim, config.dim)

    def train(self, mode: bool = True) -> "PretrainingModel":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self, waveforms: Tensor, lengths: Tensor, generator: torch.Generator | None = None,
        *, student_waveforms: Tensor | None = None,
    ) -> PretrainingOutput:
        counts = _frame_counts(waveforms, lengths)
        if min(counts) < 2:
            raise ValueError(f"Pretraining requires at least {RECEPTIVE_FIELD + FRAME_STRIDE} samples per clip")
        if student_waveforms is None:
            student_waveforms = waveforms
        if student_waveforms.shape != waveforms.shape:
            raise ValueError("Student and teacher waveform batches must have the same shape")
        local = self.student._extract(student_waveforms)
        valid = _valid_frames(local, counts)
        valid_frames = sum(counts)
        # Measure the acoustic projection before masking or positional encoding.
        local_variance = _frame_variance(local, valid, valid_frames)
        variance_loss = F.relu(LOCAL_STD_FLOOR - (local_variance + 1e-4).sqrt()).mean()
        masked, masked_frames, span_count = _sample_mask(
            counts, local.size(1), self.mask_fraction, self.mask_span, generator
        )
        masked = masked.to(local.device)
        local = torch.where(masked[:, :, None], self.mask_embedding.to(local.dtype), local)
        features, _ = self.student._contextualize(local, valid)
        predictions = self.predictor(features).float()

        with torch.no_grad(), torch.autocast(device_type=waveforms.device.type, enabled=False):
            teacher_local = self.teacher._extract(waveforms.float())
            _, layers = self.teacher._contextualize(teacher_local, valid, self.config.teacher_layers)
            targets = _teacher_target(layers, valid)

        errors = (predictions - targets).masked_fill(~masked[:, :, None], 0)
        regression_loss = errors.square().sum() / (masked_frames * self.config.dim)
        loss = regression_loss
        if self.variance_weight:
            loss = loss + self.variance_weight * variance_loss
        with torch.no_grad():
            masked_target_std = _frame_variance(targets, masked, masked_frames).sqrt().mean()
            masked_prediction_std = _frame_variance(predictions, masked, masked_frames).sqrt().mean()
        return PretrainingOutput(
            loss, masked_frames, valid_frames, span_count, regression_loss, variance_loss,
            masked_target_std, masked_prediction_std,
        )

    @torch.no_grad()
    def update_teacher(self, decay: float) -> None:
        if not 0 <= decay <= 1:
            raise ValueError("EMA decay must be in [0, 1]")
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters(), strict=True):
            teacher.copy_(torch.lerp(teacher.float(), student.detach().float(), 1 - decay))


def _feature_statistics(utterances: list[Tensor]) -> dict[str, float]:
    within = torch.stack([x.std(dim=0, unbiased=False) for x in utterances]).mean()
    means = torch.stack([x.mean(dim=0) for x in utterances])
    frames = torch.cat(utterances).double()
    centered = frames - frames.mean(dim=0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(frames)).clamp_min(0)
    total = eigenvalues.sum()
    rank = 0.0
    if total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / total
        rank = (-torch.sum(probabilities * probabilities.log())).exp().item()
    stats = {"within_utterance_std": within.item(), "effective_rank": rank}
    if len(utterances) > 1:
        stats["across_utterance_std"] = means.std(dim=0, unbiased=False).mean().item()
    return stats


@torch.no_grad()
def diagnostics(
    model: PretrainingModel, batches: Iterable[tuple[Tensor, Tensor]]
) -> dict[str, float]:
    """Inspect the unmasked exported layers on fixed evaluation waveforms."""
    encoders = {"student": model.student, "teacher": model.teacher}
    utterances: dict[str, list[Tensor]] = {name: [] for name in encoders}
    clips = []
    metrics = {}
    was_training = model.training
    model.eval()
    try:
        for waveforms, lengths in batches:
            remaining = DIAGNOSTIC_CLIPS - len(clips)
            if not remaining:
                break
            lengths = lengths[:remaining]
            sample_counts = lengths.tolist()
            waveforms = waveforms[:remaining, :max(sample_counts)]
            counts = frame_lengths(lengths).tolist()
            for row, size in enumerate(sample_counts):
                clips.append(waveforms[row, :size].clone())
            shortest, longest = min(sample_counts), max(sample_counts)
            shortest_row = sample_counts.index(shortest)
            for name, encoder in encoders.items():
                features, _ = encoder(waveforms, lengths)
                utterances[name].extend(
                    features[row, :count].float().cpu() for row, count in enumerate(counts)
                )
                key = f"{name}_padding_max_abs_error"
                if key not in metrics and shortest < longest:
                    single, _ = encoder(
                        waveforms[shortest_row:shortest_row + 1, :shortest],
                        lengths[shortest_row:shortest_row + 1],
                    )
                    padded = features[shortest_row:shortest_row + 1, :single.size(1)]
                    metrics[key] = (single - padded).abs().max().item()

        if len(clips) == 1:
            clip = clips[0]
            length = max(RECEPTIVE_FIELD, clip.numel() // 2)
            clips = [clip[:length], clip[-length:]]
        pairs = list(zip(clips[::2], clips[1::2]))
        metrics["content_probe_pairs"] = len(pairs)
        for name, encoder in encoders.items():
            distances = []
            for left, right in pairs:
                length = min(left.numel(), right.numel())
                pair = torch.stack([left[:length], right[:length]])
                lengths = torch.full((2,), length, dtype=torch.long)
                features, _ = encoder(pair, lengths)
                distance = (features[0].float() - features[1].float()).norm(dim=-1).mean()
                distances.append(distance.item())
            if distances:
                metrics[f"{name}_aligned_content_distance"] = median(distances)
            sequences = utterances[name]
            if sequences:
                metrics.update(
                    {f"{name}_{key}": value for key, value in _feature_statistics(sequences).items()}
                )
        return metrics
    finally:
        model.train(was_training)
