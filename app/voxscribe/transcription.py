import json
import threading
from dataclasses import asdict, dataclass, field

import numpy as np


_SIMPLIFIER = None
_SIMPLIFIER_LOCK = threading.Lock()


def normalize_recognition_text(text):
    text = (text or "").strip()
    if not text:
        return ""
    global _SIMPLIFIER
    if _SIMPLIFIER is None:
        with _SIMPLIFIER_LOCK:
            if _SIMPLIFIER is None:
                from opencc import OpenCC

                _SIMPLIFIER = OpenCC("t2s")
    return _SIMPLIFIER.convert(text)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""
    words: list = field(default_factory=list)


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: str = ""
    duration: float = 0.0

    def __post_init__(self):
        for segment in self.segments:
            segment.text = normalize_recognition_text(segment.text)

    @property
    def text(self):
        return "\n".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, value):
        data = json.loads(value)
        return cls(
            segments=[Segment(**segment) for segment in data.get("segments", [])],
            language=data.get("language", ""),
            duration=float(data.get("duration", 0.0)),
        )


def normalize_audio(audio, target_rate=16000):
    if isinstance(audio, tuple):
        samples, sample_rate = audio
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if int(sample_rate) != target_rate:
            from scipy.signal import resample_poly

            samples = resample_poly(samples, target_rate, int(sample_rate)).astype(np.float32)
        return samples, target_rate
    return load_media_audio(audio, target_rate)


def load_media_audio(path, target_rate=16000):
    import av

    chunks = []
    with av.open(str(path)) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError("媒体文件中没有音频轨道")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
        for frame in container.decode(audio_stream):
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for item in converted:
                chunks.append(item.to_ndarray().reshape(-1).astype(np.float32))
        for item in resampler.resample(None):
            chunks.append(item.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        raise RuntimeError("无法解码媒体音频")
    return np.concatenate(chunks), target_rate
