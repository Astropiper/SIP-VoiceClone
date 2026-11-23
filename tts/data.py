"""Data contracts for the three training stages."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

from .encoder import FRAME_STRIDE, RECEPTIVE_FIELD, SAMPLE_RATE
MEL_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class Clip:
    """A pre-cut utterance and the ID of the recording it was cut from."""

    path: Path | str
    source: str
    text: str | None = None

    def __post_init__(self):
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("Audio path must be a nonempty path")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Source recording ID must be a nonempty string")
        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("Clip text must be a string")
        object.__setattr__(self, "path", Path(self.path).expanduser())
        object.__setattr__(self, "source", self.source.strip())


def read_manifest(path: Path | str) -> list[Clip]:
    """Read JSON Lines records with audio, source, and optional text fields."""
    path = Path(path).expanduser().absolute()
    clips = []
    with path.open(encoding="utf-8") as manifest:
        for number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audio = row.get("audio") if isinstance(row, dict) else None
                if not isinstance(audio, str) or not audio.strip():
                    raise ValueError("Expected an object with a nonempty audio path string")
                audio_path = path.parent / Path(audio).expanduser()
                clips.append(Clip(audio_path, row.get("source"), row.get("text")))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not clips:
        raise ValueError(f"{path}: manifest has no clips")
    return clips


def validate_disjoint_sources(*splits: Sequence[Clip]) -> None:
    """Check that one original recording never appears in multiple splits."""
    seen = set()
    for split in splits:
        sources = {clip.source for clip in split}
        overlap = seen.intersection(sources)
        if overlap:
            raise ValueError(f"Source recording appears in multiple splits: {min(overlap)!r}")
        seen.update(sources)


def _decode_waveform(path: Path | str) -> tuple[Tensor, int]:
    path = Path(path).expanduser()
    try:
        samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Could not decode audio {path}: {exc}") from exc
    waveform = torch.from_numpy(samples)
    if not waveform.numel() or not torch.isfinite(waveform).all():
        raise ValueError(f"Audio is empty or contains non-finite samples: {path}")
    return waveform.mean(dim=1), rate


@lru_cache(maxsize=16)
def _resampler(original_rate: int, target_rate: int) -> Resample:
    return Resample(original_rate, target_rate)


def _resample(waveform: Tensor, original_rate: int, target_rate: int) -> Tensor:
    if original_rate != target_rate:
        waveform = _resampler(original_rate, target_rate)(waveform)
    return waveform.clamp(-1, 1)


def load_waveform(path: Path | str, sample_rate: int = SAMPLE_RATE) -> Tensor:
    """Decode, downmix and resample a clip to a CPU float32 waveform."""
    waveform, rate = _decode_waveform(path)
    return _resample(waveform, rate, sample_rate)


class BackgroundMix:
    """Mix independent background audio into valid samples of a CPU waveform batch."""

    def __init__(self, clips: Sequence[Clip], probability: float, snr_db: tuple[float, float]):
        if not clips:
            raise ValueError("Background mixing requires at least one audio clip")
        if not 0 <= probability <= 1:
            raise ValueError("--background-probability must be in [0, 1]")
        low, high = snr_db
        if not 0 <= low <= high < float("inf"):
            raise ValueError("--background-snr-db must be finite, nonnegative and in ascending order")
        self.paths = tuple(clip.path for clip in clips)
        self.probability = probability
        self.snr_db = snr_db
        self._load = lru_cache(maxsize=8)(load_waveform)

    def __call__(self, waveforms: Tensor, lengths: Tensor) -> tuple[Tensor, int]:
        selected = torch.rand(waveforms.size(0)) < self.probability
        if not selected.any():
            return waveforms, 0
        mixed = waveforms.clone()
        mixed_clips = 0
        low, high = self.snr_db
        for row in selected.nonzero(as_tuple=True)[0].tolist():
            signal = mixed[row, :lengths[row].item()]
            background = self._load(self.paths[torch.randint(len(self.paths), ()).item()])
            # Crop longer backgrounds and place shorter ones within the utterance.
            overlap = min(signal.numel(), background.numel())
            offset = torch.randint(abs(signal.numel() - background.numel()) + 1, ()).item()
            signal_start = offset if signal.numel() > background.numel() else 0
            background_start = offset if background.numel() > signal.numel() else 0
            region = signal[signal_start:signal_start + overlap]
            background = background[background_start:background_start + overlap]
            signal_rms = region.square().mean().sqrt().item()
            background_rms = background.square().mean().sqrt().item()
            if min(signal_rms, background_rms) < 1e-8:
                continue
            snr = low + (high - low) * torch.rand(()).item()
            gain = (signal_rms / background_rms) * 10 ** (-snr / 20)
            if gain == 0:
                continue
            region.add_(background, alpha=gain)
            signal.div_(signal.abs().amax().clamp_min(1))
            mixed_clips += 1
        return (mixed if mixed_clips else waveforms), mixed_clips


def _check_encoder_length(waveform: Tensor, path: Path | str, min_frames: int) -> None:
    min_samples = RECEPTIVE_FIELD + (min_frames - 1) * FRAME_STRIDE
    if waveform.numel() < min_samples:
        raise ValueError(
            f"{path}: need at least {min_frames} encoder frame(s) "
            f"({min_samples} samples at {SAMPLE_RATE} Hz), got {waveform.numel()} samples"
        )


def load_clean_waveforms(path: Path | str) -> tuple[Tensor, Tensor]:
    """Load 16 kHz encoder and 24 kHz Vocos inputs from one decoded clip."""
    waveform, rate = _decode_waveform(path)
    encoder_waveform = _resample(waveform, rate, SAMPLE_RATE)
    _check_encoder_length(encoder_waveform, path, min_frames=1)
    mel_waveform = _resample(waveform, rate, MEL_SAMPLE_RATE)
    if mel_waveform.numel() <= 512:  # Vocos centers its 1024-sample STFT with reflect padding.
        raise ValueError(f"{path}: waveform is too short for Vocos reflect padding")
    return encoder_waveform, mel_waveform


class WaveformDataset(Dataset[Tensor]):
    def __init__(self, clips: Sequence[Clip], min_frames: int = 2):
        if min_frames < 1:
            raise ValueError("Minimum encoder frame count must be positive")
        self.clips = tuple(clips)
        self.min_frames = min_frames

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Tensor:
        path = self.clips[index].path
        waveform = load_waveform(path)
        _check_encoder_length(waveform, path, self.min_frames)
        return waveform


class TextDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, clips: Sequence[Clip], tokenize: Callable[[str], Sequence[int] | Tensor]):
        self.audio = WaveformDataset(clips, min_frames=1)
        self.tokens = []
        for clip in self.audio.clips:
            if clip.text is None or not clip.text.strip():
                raise ValueError(f"{clip.path}: text training requires an ASR transcript")
            ids = torch.as_tensor(tokenize(clip.text), dtype=torch.long)
            if ids.ndim != 1 or not ids.numel() or (ids <= 0).any():
                raise ValueError(f"{clip.path}: tokenizer must return nonempty positive token IDs")
            self.tokens.append(ids.tolist())

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.audio[index], torch.tensor(self.tokens[index], dtype=torch.long)


class CleanDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, clips: Sequence[Clip]):
        self.clips = tuple(clips)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return load_clean_waveforms(self.clips[index].path)


def _pad(sequences: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), lengths


def collate_waveforms(samples: Sequence[Tensor]) -> dict[str, Tensor]:
    waveforms, lengths = _pad(samples)
    return {"waveforms": waveforms, "lengths": lengths}


def collate_text(samples: Sequence[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    waveforms, tokens = zip(*samples)
    token_ids, token_lengths = _pad(tokens)
    return dict(collate_waveforms(waveforms), tokens=token_ids, token_lengths=token_lengths)


def collate_clean(samples: Sequence[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    waveforms, mel_waveforms = zip(*samples)
    mel_waveforms, mel_lengths = _pad(mel_waveforms)
    return dict(
        collate_waveforms(waveforms),
        mel_waveforms=mel_waveforms,
        mel_waveform_lengths=mel_lengths,
    )
