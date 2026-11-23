"""Vocos Mel targets and a time-aligned acoustic decoder."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import MEL_SAMPLE_RATE
from .encoder import FIRST_FRAME_SECONDS, FRAME_STEP_SECONDS, FRAME_STRIDE, RECEPTIVE_FIELD, SAMPLE_RATE

MEL_HOP = 256
N_MELS = 100
VOCOS_CHECKPOINT = "charactr/vocos-mel-24khz"


def load_vocos():
    try:
        from vocos import Vocos
    except ModuleNotFoundError as exc:
        if exc.name != "vocos":
            raise
        raise RuntimeError("Mel training and synthesis require Vocos: install tts/requirements.txt") from exc

    # Keep the native STFT and inverse STFT on CPU, including with an MPS decoder.
    return Vocos.from_pretrained(VOCOS_CHECKPOINT).cpu().eval().requires_grad_(False)


def make_mel_extractor() -> nn.Module:
    # The checkpoint includes frontend buffers such as the Mel filter bank.
    return load_vocos().feature_extractor


@torch.no_grad()
def extract_mel_targets(extractor: nn.Module, waveforms: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    # Native centered STFT reflects each clip's real edge, so run it before padding.
    sequences = [
        extractor(waveform[:length].unsqueeze(0)).squeeze(0).transpose(0, 1)
        for waveform, length in zip(waveforms, lengths.tolist(), strict=True)
    ]
    return pad_sequence(sequences, batch_first=True), lengths.new_tensor([len(mel) for mel in sequences])


def mel_valid_mask(lengths: Tensor, frames: int) -> Tensor:
    return torch.arange(frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def align_features(features: Tensor, lengths: Tensor, mel_frames: int) -> Tensor:
    """Interpolate onto centered Vocos Mel timestamps, extending each endpoint."""
    mel_times = torch.arange(mel_frames, device=features.device, dtype=torch.float32)
    mel_times = mel_times * (MEL_HOP / MEL_SAMPLE_RATE)
    acoustic_indices = ((mel_times - FIRST_FRAME_SECONDS) / FRAME_STEP_SECONDS).clamp_min(0)
    last_index = (lengths - 1).unsqueeze(1)
    acoustic_indices = torch.minimum(acoustic_indices.unsqueeze(0), last_index)
    left = acoustic_indices.floor().long()
    right = torch.minimum(left + 1, last_index)
    width = features.size(-1)
    left_values = features.gather(1, left.unsqueeze(-1).expand(-1, -1, width))
    right_values = features.gather(1, right.unsqueeze(-1).expand(-1, -1, width))
    weight = (acoustic_indices - left).unsqueeze(-1).to(features.dtype)
    return torch.lerp(left_values, right_values, weight)


def estimate_mel_lengths(acoustic_lengths: Tensor) -> Tensor:
    # Use the shortest duration consistent with the number of valid acoustic frames.
    samples_16k = RECEPTIVE_FIELD + (acoustic_lengths - 1) * FRAME_STRIDE
    return samples_16k * MEL_SAMPLE_RATE // (SAMPLE_RATE * MEL_HOP) + 1


@dataclass(frozen=True)
class MelDecoderConfig:
    acoustic_dim: int = 256
    width: int = 128
    layers: int = 4


class MelResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, kernel_size=7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, 4 * width)
        self.contract = nn.Linear(4 * width, width)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.depthwise(features.transpose(1, 2)).transpose(1, 2)
        hidden = self.contract(F.gelu(self.expand(self.norm(hidden))))
        return features + hidden


class MelDecoder(nn.Module):
    def __init__(self, config: MelDecoderConfig = MelDecoderConfig()):
        super().__init__()
        self.input_proj = nn.Linear(config.acoustic_dim, config.width)
        self.blocks = nn.ModuleList(MelResidualBlock(config.width) for _ in range(config.layers))
        self.output_proj = nn.Linear(config.width, N_MELS)

    def forward(
        self, features: Tensor, lengths: Tensor, mel_lengths: Tensor, mel_frames: int | None = None
    ) -> Tensor:
        if mel_frames is None:
            mel_frames = int(mel_lengths.max().item())
        padding = ~mel_valid_mask(mel_lengths, mel_frames).unsqueeze(-1)
        hidden = align_features(self.input_proj(features), lengths, mel_frames).masked_fill(padding, 0)
        for block in self.blocks:
            hidden = block(hidden).masked_fill(padding, 0)
        return self.output_proj(hidden).masked_fill(padding, 0)
