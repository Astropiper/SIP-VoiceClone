"""Transcribe single-speaker segments that match a local voiceprint gallery."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
VAD_LIMIT_MS = 9000
MIN_SPEAKER_MS = 120
MIN_VOICEPRINT_MS = 1000
MATCH_MARGIN = 0.05
EMBEDDING_DIM = 192
VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VOICEPRINT_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"
ASR_MODEL = "iic/SenseVoiceSmall"
SEGMENTATION_MODEL = (
    Path(__file__).resolve().parent
    / "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
)
# Class order of the pyannote 3.0 powerset output: silence, A, B, C, AB, AC, BC.
ACTIVE_SPEAKERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
)


@contextmanager
def decode_audio(path):
    if not path.is_file():
        raise ValueError(f"找不到音频文件 {path}")
    with tempfile.TemporaryDirectory(prefix="speech-pipeline-") as directory:
        pcm = Path(directory) / "audio.f32"
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path.resolve()),
                "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "f32le", str(pcm),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise ValueError(f"无法解码 {path}\n{result.stderr.strip()}")
        if pcm.stat().st_size == 0:
            raise ValueError(f"音频流为空 {path}")
        # VAD only copies one block at a time, even for hour-long input.
        yield np.memmap(pcm, dtype="<f4", mode="r")


def load_model(model_id, **options):
    from funasr import AutoModel

    cache = Path(os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope/hub")).expanduser()
    local_model = cache / "models" / model_id
    model = str(local_model) if (local_model / "configuration.json").is_file() else model_id
    loaded = AutoModel(
        model=model, device="cpu", ncpu=4, check_latest=False,
        disable_update=True, disable_pbar=True, **options,
    )
    weights = loaded.kwargs.get("init_param")
    if not weights or not Path(weights).is_file():
        raise ValueError(f"缺少 {model_id} 的模型权重，请检查 ModelScope 缓存")
    return loaded


def vad_segments(vad, audio):
    cache = {}
    start_ms = None
    audio_end_ms = len(audio) * 1000 // SAMPLE_RATE
    # Extra silence flushes the VAD and ensures the last block has usable features.
    for offset in range(0, len(audio), SAMPLE_RATE):
        chunk = np.array(audio[offset:offset + SAMPLE_RATE])
        final = offset + SAMPLE_RATE >= len(audio)
        if final:
            chunk = np.pad(chunk, (0, SAMPLE_RATE + SAMPLE_RATE // 4 - len(chunk)))
        events = vad.generate(
            input=chunk, fs=SAMPLE_RATE, cache=cache, chunk_size=1000,
            is_final=final, max_end_silence_time=600,
        )[0]["value"]
        for start, end in events:
            if start >= 0:
                start_ms = start
            if end >= 0 and start_ms is not None:
                end = min(end, audio_end_ms)
                if end > start_ms:
                    yield start_ms, end
                start_ms = None
    if start_ms is not None and start_ms < audio_end_ms:
        yield start_ms, audio_end_ms


def clip_audio(audio, start_ms, end_ms):
    return np.array(audio[start_ms * SAMPLE_RATE // 1000:end_ms * SAMPLE_RATE // 1000])


class SpeakerCounter:
    def __init__(self, model):
        if not model.is_file():
            raise ValueError(f"缺少说话人分段模型 {model}，下载方式见 README.md")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        try:
            self.session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ValueError(f"无法加载说话人分段模型 {model}\n{exc}") from exc
        self.input_name = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        try:
            self.window_size = int(metadata["window_size"])
            self.frame_size = int(metadata["receptive_field_size"])
            self.frame_step = int(metadata["receptive_field_shift"])
            sample_rate = int(metadata["sample_rate"])
            num_classes = int(metadata["num_classes"])
        except (KeyError, ValueError) as exc:
            raise ValueError("请使用 sherpa-onnx 发布的 pyannote segmentation 3.0 模型") from exc
        if (sample_rate != SAMPLE_RATE or num_classes != len(ACTIVE_SPEAKERS)
                or self.window_size < SAMPLE_RATE * 10 or min(self.frame_step, self.frame_size) < 1):
            raise ValueError("说话人分段模型的采样率、窗口或输出类别与 pyannote 3.0 不兼容")

    def measure(self, clip):
        if len(clip) > self.window_size:
            raise ValueError("VAD 片段超过说话人模型的窗口长度")
        left = (self.window_size - len(clip)) // 2
        waveform = np.zeros((1, 1, self.window_size), dtype=np.float32)
        waveform[0, 0, left:left + len(clip)] = clip
        scores, = self.session.run(None, {self.input_name: waveform})
        if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[2] != len(ACTIVE_SPEAKERS):
            raise ValueError(f"说话人分段模型返回了错误的输出形状 {scores.shape}")
        centers = np.arange(scores.shape[1]) * self.frame_step + (self.frame_size - 1) / 2
        valid_scores = scores[0, (centers >= left) & (centers < left + len(clip))]
        if not np.isfinite(valid_scores).all():
            raise ValueError("说话人分段模型返回了无效分数")
        active = ACTIVE_SPEAKERS[valid_scores.argmax(axis=1)]
        ms_per_frame = self.frame_step * 1000 / SAMPLE_RATE
        speaker_ms = active.sum(axis=0) * ms_per_frame
        overlap_ms = np.count_nonzero(active.sum(axis=1) > 1) * ms_per_frame
        return int((speaker_ms >= MIN_SPEAKER_MS).sum()), round(speaker_ms.max()), round(overlap_ms)


def unit_vector(data):
    try:
        values = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("声纹向量只能包含数值") from exc
    if values.shape != (EMBEDDING_DIM,) or not np.isfinite(values).all():
        raise ValueError(f"声纹必须是含 {EMBEDDING_DIM} 个有限数值的一维向量")
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("声纹向量的模长无效")
    return values / norm


def extract_voiceprint(model, clip):
    result = model.generate(input=clip, fs=SAMPLE_RATE)[0]["spk_embedding"]
    return unit_vector(result[0].detach().cpu().numpy())


def load_gallery(path):
    gallery = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gallery, dict) or gallery.get("model") != VOICEPRINT_MODEL:
        raise ValueError(f"声纹库的 model 字段必须为 {VOICEPRINT_MODEL}")
    speakers = gallery.get("speakers")
    if not isinstance(speakers, dict) or not speakers:
        raise ValueError("声纹库的 speakers 必须是非空对象")
    result = {}
    for speaker, vectors in speakers.items():
        if not speaker.strip() or not isinstance(vectors, list) or not vectors:
            raise ValueError(f"声纹库条目 {speaker!r} 缺少有效 ID 或向量列表")
        result[speaker] = np.stack([unit_vector(vector) for vector in vectors])
    return result


def match_voiceprint(gallery, embedding, threshold):
    ranked = sorted(
        ((float(np.clip((vectors @ embedding).max(), -1, 1)), speaker)
         for speaker, vectors in gallery.items()),
        reverse=True,
    )
    score, speaker = ranked[0]
    if score < threshold:
        return "no_match", None, score
    if len(ranked) > 1 and score - ranked[1][0] < MATCH_MARGIN:
        return "ambiguous_match", None, score
    return "transcribed", speaker, score


def enroll(references, vad):
    model = load_model(VOICEPRINT_MODEL)
    speakers = {}
    for speaker, filename in references:
        if not speaker.strip():
            raise ValueError("说话人 ID 不能为空")
        with decode_audio(Path(filename)) as audio:
            vectors = [
                extract_voiceprint(model, clip_audio(audio, start, end)).tolist()
                for start, end in vad_segments(vad, audio) if end - start >= MIN_VOICEPRINT_MS
            ]
        if not vectors:
            raise ValueError(f"参考录音 {filename} 没有至少 {MIN_VOICEPRINT_MS} 毫秒的语音片段")
        speakers.setdefault(speaker, []).extend(vectors)
    return {"model": VOICEPRINT_MODEL, "speakers": speakers}


def transcribe(audio_path, vad, counter, gallery, threshold, language):
    voiceprint_model = asr = None
    segments = []
    with decode_audio(audio_path) as audio:
        for start, end in vad_segments(vad, audio):
            clip = clip_audio(audio, start, end)
            count, longest_ms, overlap_ms = counter.measure(clip)
            segment = {
                "start_ms": start, "end_ms": end, "speaker_count": count,
                "overlap_ms": overlap_ms,
            }
            segments.append(segment)
            if count != 1:
                segment["status"] = "multiple_speakers" if count > 1 else "no_speaker"
                continue
            if longest_ms < MIN_VOICEPRINT_MS:
                segment["status"] = "too_short"
                continue

            if voiceprint_model is None:
                voiceprint_model = load_model(VOICEPRINT_MODEL)
            embedding = extract_voiceprint(voiceprint_model, clip)
            status, speaker, score = match_voiceprint(gallery, embedding, threshold)
            segment.update(status=status, similarity=round(score, 4))
            if speaker is None:
                continue

            if asr is None:
                asr = load_model(ASR_MODEL)
            result = asr.generate(input=clip, fs=SAMPLE_RATE, language=language, use_itn=True)
            text = re.sub(r"<\|[^|]*\|>", "", result[0]["text"]).strip()
            segment.update(speaker_id=speaker, text=text)
    return {"audio": str(audio_path), "match_threshold": threshold, "segments": segments}


def write_json(path, data):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as temp:
        try:
            json.dump(data, temp, ensure_ascii=False, indent=2, allow_nan=False)
            temp.write("\n")
            temp.close()
            os.replace(temp.name, path)
        finally:
            Path(temp.name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    enrollment = commands.add_parser("enroll", help="从单人参考录音创建声纹库")
    enrollment.add_argument("--speaker", action="append", nargs=2, metavar=("ID", "AUDIO"), required=True)
    enrollment.add_argument("--output", "-o", required=True, type=Path)
    transcription = commands.add_parser("transcribe", help="仅转写匹配声纹库的单人片段")
    transcription.add_argument("audio", type=Path)
    transcription.add_argument("--voiceprints", required=True, type=Path)
    transcription.add_argument("--output", "-o", required=True, type=Path)
    transcription.add_argument("--segmentation-model", type=Path, default=SEGMENTATION_MODEL)
    transcription.add_argument("--match-threshold", type=float, default=0.5)
    transcription.add_argument("--language", choices=["auto", "zh", "en", "yue", "ja", "ko"], default="auto")
    args = parser.parse_args()

    make_vad = partial(load_model, VAD_MODEL, max_single_segment_time=VAD_LIMIT_MS)
    try:
        if args.command == "enroll":
            inputs = [Path(filename) for _, filename in args.speaker]
        else:
            inputs = [args.audio, args.voiceprints, args.segmentation_model]
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("输出路径必须与输入文件分开")
        if args.command == "enroll":
            data = enroll(args.speaker, make_vad())
        else:
            if not 0 <= args.match_threshold <= 1:
                raise ValueError("--match-threshold 必须在 [0, 1] 内")
            gallery = load_gallery(args.voiceprints)
            counter = SpeakerCounter(args.segmentation_model)
            data = transcribe(args.audio, make_vad(), counter, gallery, args.match_threshold, args.language)
        write_json(args.output, data)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1
    print(f"已写入 {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
