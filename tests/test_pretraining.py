import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import soundfile
import torch

from tts.__main__ import _parser
from tts.config import DEFAULT_TRAINING_CONFIG, read_config, resolve_training_config
from tts.data import BackgroundMix, Clip
from tts.encoder import EncoderConfig, PretrainingModel, SAMPLE_RATE
from tts.train import PRETRAIN, _prepare_metrics_log, _prepare_task, load_checkpoint, train


CONFIG = EncoderConfig(
    conv_channels=4, dim=8, layers=1, heads=2, ff_dim=16, teacher_layers=1, dropout=0,
)


def _position_only_model(config, **kwargs):
    model = PretrainingModel(config, **kwargs)
    with torch.no_grad():
        for layer in (model.student.projection, model.student.blocks[0].ff_out, model.predictor):
            for parameter in layer.parameters():
                parameter.zero_()
    model.update_teacher(0)
    return model


class PretrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_position_only_features_are_rejected_by_checkpoint_selection(self):
        generator = torch.Generator().manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for split in ("train", "valid"):
                records = []
                for index in range(2):
                    source = f"{split}-{index}"
                    path = root / f"{source}.wav"
                    waveform = torch.randn(1920, generator=generator) * 0.1
                    soundfile.write(path, waveform.numpy(), SAMPLE_RATE)
                    records.append(json.dumps({"audio": path.name, "source": source}))
                manifest = root / f"{split}.jsonl"
                manifest.write_text("\n".join(records) + "\n", encoding="utf-8")
                manifests.append(manifest)

            output = root / "run"
            model_config = root / "encoder.json"
            model_config.write_text(json.dumps(asdict(CONFIG)), encoding="utf-8")
            training_input = root / "settings.json"
            training_input.write_text(json.dumps({
                "common": {
                    "device": "cpu", "updates": 1, "batch_size": 1, "eval_every": 5,
                    "train_manifest": "train.jsonl", "valid_manifest": "valid.jsonl", "output_dir": "run",
                },
                "pretrain": {"variance_weight": 0.25},
            }), encoding="utf-8")
            command = [
                "pretrain", "--training-config", str(training_input), "--batch-size", "2",
            ]
            args = _parser().parse_args(command + ["--model-config", str(model_config)])
            log = io.StringIO()
            with patch("tts.train.PretrainingModel", side_effect=_position_only_model), redirect_stdout(log):
                train(args)

            validation = json.loads(log.getvalue().splitlines()[-1])
            self.assertEqual(validation["split"], "validation")
            self.assertEqual(validation["regression_loss"], 0)
            self.assertGreater(validation["teacher_within_utterance_std"], 0)
            self.assertGreater(validation["teacher_effective_rank"], 1)
            self.assertEqual(validation["teacher_aligned_content_distance"], 0)
            self.assertFalse(validation["content_check_passed"])
            checkpoint = output / "last.pt"
            payload = load_checkpoint(checkpoint, PRETRAIN)
            self.assertEqual(payload["model_config"], asdict(CONFIG))
            saved_config = output / "model_config.json"
            self.assertEqual(json.loads(saved_config.read_text(encoding="utf-8")), asdict(CONFIG))
            training_config_path = output / "training_config.json"
            training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
            for key, value in payload["settings"].items():
                self.assertEqual(training_config[key], value)
            self.assertEqual(training_config["train_manifest"], str(manifests[0].absolute()))
            self.assertEqual(training_config["eval_every"], 5)
            self.assertEqual(training_config["batch_size"], 2)
            self.assertEqual(training_config["variance_weight"], 0.25)
            self.assertIsNone(training_config["resume"])
            for name, source in (("model_config.json", model_config), ("training_config.json", training_input)):
                self.assertEqual((output / "input_configs" / name).read_bytes(), source.read_bytes())
            self.assertFalse((output / "best.pt").exists())
            persisted = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(persisted, log.getvalue().splitlines())
            training_log = (output / "train.log").read_text(encoding="utf-8")
            self.assertIn(log.getvalue(), training_log)
            self.assertIn(json.dumps({"model_config": asdict(CONFIG)}), training_log)
            self.assertIn("Finished pretrain", training_log)
            self.assertFalse(validation["best_checkpoint"])
            with (output / "training.png").open("rb") as plot:
                self.assertEqual(plot.read(8), b"\x89PNG\r\n\x1a\n")

            model_config.unlink()
            saved_config.unlink()
            resume = ["pretrain", "--training-config", str(training_config_path),
                      "--resume", str(checkpoint), "--no-plot"]
            train(_parser().parse_args(resume))
            self.assertEqual(json.loads(saved_config.read_text(encoding="utf-8")), asdict(CONFIG))
            resumed_config = json.loads(training_config_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed_config["resume"], str(checkpoint.absolute()))
            self.assertEqual(resumed_config["variance_weight"], 0.25)
            self.assertTrue(resumed_config["no_plot"])
            model_config.write_text('{"dim": 16}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dim must match the resume checkpoint"):
                train(_parser().parse_args(resume + ["--model-config", str(model_config)]))
            self.assertEqual(json.loads(training_config_path.read_text(encoding="utf-8")), resumed_config)

    def test_training_log_preserves_traceback_when_retrying(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "训练"
            args = _parser().parse_args([
                "pretrain", "--device", "cpu", "--updates", "1", "--no-plot",
                "--train-manifest", "train.jsonl", "--valid-manifest", "valid.jsonl",
                "--output-dir", str(output),
            ])

            def fail(*_):
                print("音频读取失败", file=sys.stderr)
                raise ValueError("invalid manifest")

            stderr = io.StringIO()
            with redirect_stderr(stderr), patch("tts.train._prepare_task", side_effect=fail):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "invalid manifest"):
                        train(args)
                    self.assertIs(sys.stderr, stderr)

            log = (output / "train.log").read_text(encoding="utf-8")
            self.assertEqual(log.count("Started pretrain"), 2)
            self.assertEqual(log.count("Traceback (most recent call last):"), 2)
            self.assertEqual(log.count("ValueError: invalid manifest"), 2)
            self.assertEqual(log.count("音频读取失败"), 2)
            self.assertEqual(stderr.getvalue(), "音频读取失败\n" * 2)

    def test_resume_discards_uncommitted_and_duplicate_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "metrics.jsonl"
            records = [
                {"split": "train", "update": 2, "loss": 0.6},
                {"split": "train", "update": 2, "loss": 0.5},
                {"split": "validation", "update": 2, "loss": 0.7},
                {"split": "train", "update": 3, "loss": 0.4},
            ]
            path.write_text("\n".join(map(json.dumps, records)) + '\n{"split":', encoding="utf-8")

            self.assertEqual(_prepare_metrics_log(path, root / "last.pt", 2), records[1:3])
            self.assertEqual(
                [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()], records[1:3],
            )

    def test_variance_loss_backpropagates_through_small_acoustic_variation(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            model = PretrainingModel(CONFIG, variance_weight=0.5)
            waveforms = torch.randn(1, 1920) * 0.1
        projection = model.student.projection.weight
        with torch.no_grad():
            projection.mul_(0.01)

        result = model(waveforms, torch.tensor([waveforms.size(1)]), torch.Generator().manual_seed(1))
        self.assertGreater(result.variance_loss.item(), 0)
        regularization = result.loss - result.regression_loss
        torch.testing.assert_close(regularization, 0.5 * result.variance_loss)
        gradient, = torch.autograd.grad(regularization, projection)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0)

    def test_background_mix_respects_snr_lengths_and_padding(self):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            path = Path(directory) / "background.wav"
            soundfile.write(path, torch.full((800,), 0.5).numpy(), SAMPLE_RATE)
            mix = BackgroundMix([Clip(path, "instrumental")], 1.0, (10.0, 10.0))
            waveforms = torch.zeros(2, 1600)
            waveforms[0, :1200] = 0.25
            lengths = torch.tensor([1200, 1600])
            result, count = mix(waveforms, lengths)

            self.assertEqual(count, 1)
            torch.testing.assert_close(waveforms[0, :1200], torch.full((1200,), 0.25))
            torch.testing.assert_close(result[0, 1200:], waveforms[0, 1200:])
            torch.testing.assert_close(result[1], waveforms[1])
            added = result[0] - waveforms[0]
            self.assertEqual(added.count_nonzero().item(), 800)
            torch.testing.assert_close(added[added != 0], torch.full((800,), 0.25 * 10 ** -0.5))

            mix = BackgroundMix([Clip(path, "instrumental")], 1.0, (0.0, 0.0))
            waveforms[0, :720] = 0.8
            waveforms[0, 720:] = 0
            result, count = mix(waveforms, torch.tensor([720, 1600]))
            self.assertEqual(count, 1)
            torch.testing.assert_close(result[0, :720], torch.ones(720))
            torch.testing.assert_close(result[0, 720:], waveforms[0, 720:])

    def test_pretraining_background_is_only_seen_by_student_during_training(self):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            root = Path(directory)
            for split, scale in (("train", 0.1), ("valid", 0.2), ("background", 0.3)):
                soundfile.write(root / f"{split}.wav", (torch.randn(1920) * scale).numpy(), SAMPLE_RATE)
                (root / f"{split}.jsonl").write_text(
                    json.dumps({"audio": f"{split}.wav", "source": split}) + "\n", encoding="utf-8",
                )
            model_config = root / "encoder.json"
            model_config.write_text(json.dumps(asdict(CONFIG)), encoding="utf-8")
            args = _parser().parse_args([
                "pretrain", "--updates", "1", "--output-dir", str(root / "run"),
                "--train-manifest", str(root / "train.jsonl"), "--valid-manifest", str(root / "valid.jsonl"),
                "--background-manifest", str(root / "background.jsonl"), "--background-probability", "1",
                "--model-config", str(model_config),
            ])
            defaults, _ = read_config(DEFAULT_TRAINING_CONFIG, PRETRAIN, training=True)
            vars(args).update(resolve_training_config(args, None, defaults, {}))
            options, _ = read_config(model_config, PRETRAIN)
            task = _prepare_task(args, torch.device("cpu"), options)
            student = task.model.student
            teacher = task.model.teacher
            with (
                patch.object(student, "_extract", wraps=student._extract) as student_extract,
                patch.object(teacher, "_extract", wraps=teacher._extract) as teacher_extract,
            ):
                batch = task.collate([task.training[0]])
                originals = batch["waveforms"].clone()
                result = task.step(batch, None)
                self.assertEqual(result.counts["background_mixed_clips"], 1)
                self.assertFalse(torch.equal(student_extract.call_args.args[0], originals))
                torch.testing.assert_close(teacher_extract.call_args.args[0], originals)

                task.model.eval()
                batch = task.collate([task.validation[0]])
                result = task.step(batch, torch.Generator().manual_seed(0))
                self.assertNotIn("background_mixed_clips", result.counts)
                torch.testing.assert_close(student_extract.call_args.args[0], batch["waveforms"])
                torch.testing.assert_close(teacher_extract.call_args.args[0], batch["waveforms"])

            args.background_manifest = args.valid_manifest
            with self.assertRaisesRegex(ValueError, "Source recording appears in multiple splits"):
                _prepare_task(args, torch.device("cpu"), options)
