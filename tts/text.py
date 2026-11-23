"""Mandarin text frontend and autoregressive prediction of acoustic frames."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import sinusoidal_positions


PAD_ID = 0
UNK_ID = 1
_SPECIAL_TOKENS = ("<pad>", "<unk>")
_PUNCTUATION = str.maketrans({"。": ".", "、": ",", "’": "'", "‘": "'", "—": "-", "–": "-"})
_NUMBER = re.compile(
    r"(?P<sign>(?<![a-z0-9])[+-])?"
    r"(?P<whole>[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)"
    r"(?:\.(?P<fraction>[0-9]+))?(?P<percent>%)?"
)
_NON_HAN_TOKEN = re.compile(r"[a-z]|[,.!?;:'-]|(?<=[a-z])\s+(?=[a-z])")
_DIGIT_TABLE = str.maketrans("0123456789", "零一二三四五六七八九")


def _cardinal_words(digits: str) -> str:
    """Read an unsigned, unpadded integer with up to 12 digits."""
    digits = digits.zfill(12)
    parts = []
    needs_zero = False
    for start, group_unit in zip(range(0, 12, 4), ("亿", "万", "")):
        group = digits[start : start + 4]
        if group == "0000":
            needs_zero = bool(parts)
            continue
        for digit, place in zip(group, ("千", "百", "十", "")):
            if digit == "0":
                needs_zero = bool(parts)
                continue
            if needs_zero:
                parts.append("零")
            if parts or digit != "1" or place != "十":
                parts.append(digit.translate(_DIGIT_TABLE))
            parts.append(place)
            needs_zero = False
        parts.append(group_unit)
    return "".join(parts) or "零"


def _number_words(match: re.Match[str]) -> str:
    digits = match["whole"].replace(",", "")
    fraction = match["fraction"]
    following = match.string[match.end() : match.end() + 1]
    read_year = len(digits) == 4 and fraction is None and following == "年"
    read_digits = read_year or len(digits) > 12 or (len(digits) > 1 and digits.startswith("0"))
    number = digits.translate(_DIGIT_TABLE) if read_digits else _cardinal_words(digits)
    if fraction is not None:
        number += "点" + fraction.translate(_DIGIT_TABLE)
    if match["percent"]:
        number = "百分之" + number
    sign = {"+": "正", "-": "负", None: ""}[match["sign"]]
    return sign + number


def _non_han_tokens(text: str) -> list[str]:
    tokens = []
    for match in _NON_HAN_TOKEN.finditer(text):
        token = match.group()
        if token.isalpha():
            token = f"en:{token}"
        elif token.isspace():
            token = "<space>"
        tokens.append(token)
    return tokens


def tokenize_text(text: str) -> list[str]:
    """Apply the same normalization to ASR transcripts and inference input."""
    try:
        from pypinyin import Style, lazy_pinyin
    except ModuleNotFoundError as exc:
        if exc.name != "pypinyin":
            raise
        raise RuntimeError("The text frontend requires pypinyin: install tts/requirements.txt") from exc

    normalized = unicodedata.normalize("NFKC", text).lower().translate(_PUNCTUATION)
    normalized = _NUMBER.sub(_number_words, normalized)
    tokens = lazy_pinyin(
        normalized,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        tone_sandhi=True,
        errors=_non_han_tokens,
    )
    if not tokens:
        raise ValueError("text contains no Mandarin syllables, Latin letters, or pause marks")
    return tokens


class TextVocabulary:
    """Vocabulary ordered by token string, with stable padding and unknown IDs."""

    def __init__(self, tokens: Sequence[str]):
        self.tokens = list(tokens)
        if tuple(self.tokens[:2]) != _SPECIAL_TOKENS or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary must start with <pad>, <unk> and contain unique tokens")
        self._ids = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, texts: Iterable[str]) -> "TextVocabulary":
        tokens = {token for text in texts for token in tokenize_text(text)}
        if not tokens:
            raise ValueError("at least one transcript is required to build a vocabulary")
        return cls([*_SPECIAL_TOKENS, *sorted(tokens)])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self._ids.get(token, UNK_ID) for token in tokenize_text(text)]


@dataclass(frozen=True)
class TextModelConfig:
    vocab_size: int
    acoustic_dim: int = 256
    dim: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    stop_pos_weight: float = 5.0


@dataclass(frozen=True)
class TextOutput:
    prediction: Tensor
    stop_logits: Tensor
    loss: Tensor
    acoustic_loss: Tensor
    stop_loss: Tensor


@dataclass(frozen=True)
class Generation:
    features: Tensor
    lengths: Tensor
    hit_limit: Tensor


def _padding_mask(lengths: Tensor, width: int) -> Tensor:
    return torch.arange(width, device=lengths.device)[None, :] >= lengths[:, None]


def _causal_mask(width: int, device: torch.device) -> Tensor:
    steps = torch.arange(width, device=device)
    return steps[None, :] > steps[:, None]


class TextAcousticModel(nn.Module):
    def __init__(self, config: TextModelConfig):
        super().__init__()
        if min(config.encoder_layers, config.decoder_layers) < 1 or config.stop_pos_weight <= 0:
            raise ValueError("layer counts and stop_pos_weight must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim, padding_idx=PAD_ID)
        self.frame_projection = nn.Linear(config.acoustic_dim, config.dim)
        self.frame_output = nn.Linear(config.dim, config.acoustic_dim)
        self.stop_output = nn.Linear(config.dim, 1)
        layer_options = dict(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options),
            config.encoder_layers,
            norm=nn.LayerNorm(config.dim),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**layer_options),
            config.decoder_layers,
            norm=nn.LayerNorm(config.dim),
        )
        # TransformerEncoder and TransformerDecoder begin with cloned layer weights.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()

    def _encode(self, tokens: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        padding = _padding_mask(lengths, tokens.shape[1])
        embeddings = self.token_embedding(tokens)
        embeddings = embeddings * self.config.dim**0.5 + sinusoidal_positions(embeddings)
        return self.encoder(embeddings, src_key_padding_mask=padding), padding

    def _decode(
        self,
        frames: Tensor,
        memory: Tensor,
        memory_padding: Tensor,
        causal: Tensor,
        frame_padding: Tensor | None = None,
    ) -> Tensor:
        projected = self.frame_projection(frames)
        return self.decoder(
            projected + sinusoidal_positions(projected),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=frame_padding,
            memory_key_padding_mask=memory_padding,
            tgt_is_causal=True,
        )

    def forward(
        self, tokens: Tensor, token_lengths: Tensor, targets: Tensor, target_lengths: Tensor
    ) -> TextOutput:
        """Train with right-padded token and target batches with positive lengths."""
        memory, memory_padding = self._encode(tokens, token_lengths)
        frame_padding = _padding_mask(target_lengths, targets.shape[1])
        prefix = torch.cat((torch.zeros_like(targets[:, :1]), targets[:, :-1]), dim=1)
        prefix = prefix.masked_fill(frame_padding.unsqueeze(-1), 0)
        hidden = self._decode(
            prefix,
            memory,
            memory_padding,
            _causal_mask(targets.shape[1], targets.device),
            frame_padding,
        )
        prediction = self.frame_output(hidden).masked_fill(frame_padding.unsqueeze(-1), 0)
        stop_logits = self.stop_output(hidden).squeeze(-1).masked_fill(frame_padding, 0)

        valid_count = (~frame_padding).sum()
        target_frames = targets.float().masked_fill(frame_padding.unsqueeze(-1), 0)
        acoustic_loss = (prediction.float() - target_frames).abs().sum()
        acoustic_loss = acoustic_loss / (valid_count * self.config.acoustic_dim)
        steps = torch.arange(targets.shape[1], device=targets.device)
        stop_targets = (steps[None, :] == target_lengths[:, None] - 1).float()
        stop_errors = F.binary_cross_entropy_with_logits(
            stop_logits.float(),
            stop_targets,
            pos_weight=stop_targets.new_tensor(self.config.stop_pos_weight),
            reduction="none",
        )
        stop_loss = stop_errors.masked_fill(frame_padding, 0).sum() / valid_count
        return TextOutput(prediction, stop_logits, acoustic_loss + stop_loss, acoustic_loss, stop_loss)

    @torch.inference_mode()
    def generate(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        max_frames: int = 500,
        stop_threshold: float = 0.5,
    ) -> Generation:
        """Generate padded sequences and flag samples that ran out of frames."""
        if max_frames < 1 or not 0 <= stop_threshold <= 1:
            raise ValueError("max_frames must be positive and stop_threshold must be in [0, 1]")
        was_training = self.training
        self.eval()
        try:
            memory, memory_padding = self._encode(tokens, token_lengths)
            causal = _causal_mask(max_frames, tokens.device)
            frames = memory.new_zeros(tokens.shape[0], max_frames + 1, self.config.acoustic_dim)
            lengths = torch.zeros_like(token_lengths)
            for step in range(1, max_frames + 1):
                hidden = self._decode(frames[:, :step], memory, memory_padding, causal[:step, :step])
                current = hidden[:, -1]
                active = lengths == 0
                prediction = self.frame_output(current)
                frames[:, step] = prediction.masked_fill(~active[:, None], 0)
                stop = self.stop_output(current).squeeze(-1).float().sigmoid() >= stop_threshold
                lengths = lengths.masked_fill(active & stop, step)
                if torch.all(lengths > 0):
                    break
            hit_limit = lengths == 0
            lengths = lengths.masked_fill(hit_limit, max_frames)
            return Generation(frames[:, 1 : step + 1].contiguous(), lengths, hit_limit)
        finally:
            if was_training:
                self.train()
