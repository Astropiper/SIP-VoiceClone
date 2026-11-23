"""Read the independent model and training configuration files."""

import json
import math
from pathlib import Path


PRETRAIN = "pretrain"
TEXT = "train-text"
MEL = "train-mel"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_MODEL_CONFIG = CONFIG_DIR / "model_config.json"
DEFAULT_TRAINING_CONFIG = CONFIG_DIR / "training_config.json"

_PATH_FIELDS = {"train_manifest", "valid_manifest", "output_dir", "encoder_checkpoint", "background_manifest"}
_SNAPSHOT_FIELDS = {"command", "resume", "requested_device", "manifest_sha256", "background_augmentation",
                    "model_config_path", "training_config_path"}
_OPTIMIZER_FIELDS = ("updates", "batch_size", "accumulate", "lr", "weight_decay", "seed")
_PRETRAIN_FIELDS = ("mask_fraction", "mask_span", "ema_decay", "variance_weight", "min_content_distance")


def read_config(path: Path, stage: str, *, training: bool = False) -> tuple[dict, bytes]:
    content = path.read_bytes()
    try:
        config = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a JSON object")
    sections = {PRETRAIN, TEXT, MEL}
    if training:
        sections.add("common")
    if config.keys() & sections:
        if config.keys() - sections or any(not isinstance(value, dict) for value in config.values()):
            raise ValueError(f"Config {path} must map section names to JSON objects")
        config = config.get("common", {}) | config.get(stage, {})
    if training:
        if config.get("command", stage) != stage:
            raise ValueError(f"Training config {path} is for {config['command']}, expected {stage}")
        if "requested_device" in config:
            config["device"] = config["requested_device"]
        config = {key: value for key, value in config.items() if key not in _SNAPSHOT_FIELDS}
        for key in config.keys() & _PATH_FIELDS:
            if config[key] is not None:
                if not isinstance(config[key], str) or not config[key].strip():
                    raise ValueError(f"Training config {key} must be a nonempty path string")
                config[key] = path.parent / Path(config[key]).expanduser()
    return config, content


def resolve_training_config(args, checkpoint: dict | None, defaults: dict, options: dict) -> dict:
    fields = vars(args).keys() - {"command", "run", "resume", "model_config_path", "training_config_path"}
    unknown = (defaults.keys() | options.keys()) - fields
    if unknown:
        raise ValueError(f"Unknown training config fields for {args.command}: {', '.join(sorted(unknown))}")
    missing = fields - defaults.keys()
    if missing:
        raise ValueError(f"Default training config for {args.command} is missing: {', '.join(sorted(missing))}")
    inherited = {}
    if checkpoint is not None:
        inherited = {key: checkpoint["settings"][key] for key in _OPTIMIZER_FIELDS}
        if args.command == PRETRAIN:
            inherited.update({key: checkpoint[key] for key in _PRETRAIN_FIELDS if key in checkpoint})
            # Version 1 checkpoints from before the variance loss used a weight of zero.
            inherited.setdefault("variance_weight", 0.0)
            background = checkpoint.get("background_augmentation")
            if background is not None:
                inherited.update(background_probability=background["probability"], background_snr_db=background["snr_db"])

    cli = {key: value for key in defaults if (value := getattr(args, key, None)) is not None}
    overrides = options | cli
    values = defaults | inherited | overrides
    for key in values.keys() & _PATH_FIELDS:
        if values[key] is not None:
            values[key] = values[key].expanduser()
    for key in ("train_manifest", "valid_manifest", "output_dir", "encoder_checkpoint"):
        if key in values and values[key] is None:
            raise ValueError(f"Specify {key} in the training config or pass --{key.replace('_', '-')}")
    for key in ("updates", "batch_size", "accumulate", "log_every", "eval_every", "mask_span", "max_frames",
                "seed", "workers", "generation_samples"):
        if key in values:
            minimum = 0 if key in {"seed", "workers", "generation_samples"} else 1
            if type(values[key]) is not int or values[key] < minimum:
                raise ValueError(f"Training config {key} must be an integer >= {minimum}")
    if not isinstance(values["device"], str) or not values["device"].strip():
        raise ValueError("Training config device must be a nonempty string")
    if type(values["no_plot"]) is not bool:
        raise ValueError("Training config no_plot must be a boolean")
    _check_number(values, "lr", 0, math.inf, inclusive_min=False)
    _check_number(values, "weight_decay", 0, math.inf)
    if args.command == PRETRAIN:
        _check_number(values, "mask_fraction", 0, 1, inclusive_min=False, inclusive_max=False)
        _check_number(values, "ema_decay", 0, 1)
        _check_number(values, "variance_weight", 0, math.inf)
        _check_number(values, "min_content_distance", 0, math.inf, inclusive_min=False)
        if values["background_manifest"] is None:
            if any(overrides.get(key) is not None for key in ("background_probability", "background_snr_db")):
                raise ValueError("background_probability and background_snr_db require background_manifest")
            values.update(background_probability=None, background_snr_db=None)
        else:
            for key in ("background_probability", "background_snr_db"):
                if values[key] is None:
                    values[key] = defaults[key]
            _check_number(values, "background_probability", 0, 1)
            snr = values["background_snr_db"]
            if (not isinstance(snr, (list, tuple)) or len(snr) != 2
                    or any(type(value) not in (int, float) for value in snr)
                    or not 0 <= snr[0] <= snr[1] < math.inf):
                raise ValueError("Training config background_snr_db must contain two finite, nonnegative numbers in ascending order")
    return values


def _check_number(values: dict, key: str, minimum: float, maximum: float, *,
                  inclusive_min: bool = True, inclusive_max: bool = True) -> None:
    value = values[key]
    if (type(value) not in (int, float) or not -math.inf < value < math.inf
            or not minimum <= value <= maximum
            or (not inclusive_min and value == minimum) or (not inclusive_max and value == maximum)):
        raise ValueError(f"Training config {key} must be a finite number in the allowed range")
