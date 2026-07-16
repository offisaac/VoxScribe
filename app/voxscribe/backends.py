from pathlib import Path

import numpy as np

from voxscribe.transcription import Segment, TranscriptionResult, normalize_audio


ROOT = Path(__file__).resolve().parents[2]


BACKEND_INFO = {
    "qwen3_asr": {
        "label": "Qwen3-ASR 1.7B",
        "default_path": str(ROOT / "models" / "Qwen3-ASR-1.7B"),
    },
    "faster_whisper": {
        "label": "Faster Whisper Large-V3",
        "default_path": str(ROOT / "models" / "faster-whisper-large-v3"),
    },
    "external_cli": {
        "label": "通用本地命令行模型",
        "default_path": "",
    },
}


class QwenBackend:
    label = BACKEND_INFO["qwen3_asr"]["label"]

    def __init__(self, model_path):
        import torch
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA 显卡")
        self.torch = torch
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=2048,
        )

    def transcribe_result(self, audio, language=None):
        samples, sample_rate = normalize_audio(audio)
        segments = []
        for start_sample, end_sample in _speech_intervals(samples, sample_rate):
            chunk = samples[start_sample:end_sample]
            if len(chunk) < sample_rate // 2:
                continue
            results = self.model.transcribe(audio=(chunk, sample_rate), language=language)
            text = (results[0].text or "").strip() if results else ""
            if text:
                segments.append(
                    Segment(
                        start=start_sample / sample_rate,
                        end=end_sample / sample_rate,
                        text=text,
                    )
                )
        return TranscriptionResult(
            segments=segments,
            language=language or "",
            duration=len(samples) / sample_rate,
        )

    def transcribe(self, audio, language=None):
        return self.transcribe_result(audio, language).text


class FasterWhisperBackend:
    label = BACKEND_INFO["faster_whisper"]["label"]

    def __init__(self, model_path):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            str(model_path),
            device="cuda",
            compute_type="float16",
        )

    def transcribe_result(self, audio, language=None):
        language_map = {"Chinese": "zh", "English": "en"}
        if isinstance(audio, tuple):
            audio = audio[0]
        segments, info = self.model.transcribe(
            audio,
            language=language_map.get(language),
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
        )
        converted = []
        for segment in segments:
            words = []
            for word in segment.words or []:
                words.append({"start": word.start, "end": word.end, "word": word.word})
            converted.append(
                Segment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    words=words,
                )
            )
        return TranscriptionResult(
            segments=converted,
            language=info.language or "",
            duration=float(getattr(info, "duration", 0.0) or 0.0),
        )

    def transcribe(self, audio, language=None):
        return self.transcribe_result(audio, language).text


class ExternalCLIBackend:
    label = BACKEND_INFO["external_cli"]["label"]

    def __init__(self, model_path, command_template):
        if not command_template.strip():
            raise RuntimeError("请先填写本地模型的命令模板")
        self.model_path = str(model_path or "")
        self.command_template = command_template

    def transcribe(self, audio, language=None):
        import os
        import subprocess
        import tempfile

        import soundfile as sf

        temporary_path = None
        if isinstance(audio, tuple):
            samples, sample_rate = audio
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temporary_path = handle.name
            handle.close()
            sf.write(temporary_path, samples, sample_rate)
            input_path = temporary_path
        else:
            input_path = str(audio)
        quote = lambda value: subprocess.list2cmdline([str(value)])
        command = self.command_template.format(
            input=quote(input_path),
            model=quote(self.model_path),
            language=language or "auto",
        )
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"命令退出码 {result.returncode}")
            return result.stdout.strip()
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def transcribe_result(self, audio, language=None):
        from voxscribe.transcription import normalize_audio

        text = self.transcribe(audio, language)
        samples, sample_rate = normalize_audio(audio)
        duration = len(samples) / sample_rate
        segments = [Segment(0.0, duration, text)] if text else []
        return TranscriptionResult(segments=segments, language=language or "", duration=duration)


def create_backend(name, model_path, options=None):
    options = options or {}
    model_path = Path(model_path)
    if name != "external_cli" and not model_path.exists():
        raise RuntimeError(f"模型目录不存在：{model_path}")
    if name == "qwen3_asr":
        return QwenBackend(model_path)
    if name == "faster_whisper":
        return FasterWhisperBackend(model_path)
    if name == "external_cli":
        return ExternalCLIBackend(model_path, options.get("command_template", ""))
    raise RuntimeError(f"尚未安装模型适配器：{name}")


def _speech_intervals(samples, sample_rate, max_seconds=25.0):
    import librosa

    detected = librosa.effects.split(
        samples,
        top_db=35,
        frame_length=1024,
        hop_length=256,
    )
    if len(detected) == 0:
        return [(0, len(samples))]
    padding = int(0.15 * sample_rate)
    max_samples = int(max_seconds * sample_rate)
    intervals = []
    for raw_start, raw_end in detected:
        start = max(0, int(raw_start) - padding)
        end = min(len(samples), int(raw_end) + padding)
        if intervals and start - intervals[-1][1] < int(0.6 * sample_rate) and end - intervals[-1][0] <= max_samples:
            intervals[-1] = (intervals[-1][0], end)
        else:
            while end - start > max_samples:
                intervals.append((start, start + max_samples))
                start += max_samples
            intervals.append((start, end))
    return intervals
