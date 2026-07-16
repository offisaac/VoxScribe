import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SETTINGS = {
    "general": {
        "language": "zh_CN",
        "theme": "dark",
        "font_size": 32,
    },
    "live": {
        "device_keyword": "CABLE Output",
        "backend": "faster_whisper",
        "streaming_url": "http://127.0.0.1:8765",
        "wsl_distro": "Ubuntu",
        "stream_chunk_seconds": 0.8,
        "export_enabled": True,
        "export_folder": r"D:\Meeting-Transcripts\Voxscribe",
        "file_name": "Meeting Transcript {date_time}",
        "mode": "append_below",
        "chunk_seconds": 3.5,
        "silence_threshold": 0.0025,
        "obs_file_name": "obs_live_caption.txt",
    },
    "folder_watch": {
        "enabled": True,
        "input_folder": r"D:\Meeting-Transcripts\Voxscribe\视频",
        "output_folder": r"D:\Meeting-Transcripts\Voxscribe\离线转录",
        "delete_processed_files": False,
        "export_formats": ["txt"],
    },
    "audio_processing": {
        "mode": "noise_reduce",
        "demucs_model": "htdemucs",
        "speaker_identification": False,
        "speaker_count": 0,
    },
    "model": {
        "backend": "qwen3_asr",
        "model_path": str(ROOT / "models" / "Qwen3-ASR-1.7B"),
        "qwen3_asr_path": str(ROOT / "models" / "Qwen3-ASR-1.7B"),
        "faster_whisper_path": str(ROOT / "models" / "faster-whisper-large-v3"),
        "fun_asr_nano_path": str(ROOT / "models" / "Fun-ASR-Nano-2512"),
        "external_cli_path": "",
        "external_cli_command": "",
        "language": "auto",
        "use_gpu": True,
    },
    "hotkeys": {
        "record_toggle": "Ctrl+Shift+R",
        "floating_window": "Ctrl+Shift+F",
    },
}


def _merge(defaults, values):
    result = deepcopy(defaults)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


class SettingsStore:
    def __init__(self, path):
        self.path = Path(path)
        self.data = deepcopy(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if self.path.exists():
            try:
                values = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = _merge(DEFAULT_SETTINGS, values)
            except (OSError, ValueError):
                self.data = deepcopy(DEFAULT_SETTINGS)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def get(self, section, key, default=None):
        return self.data.get(section, {}).get(key, default)

    def update_section(self, section, values):
        current = self.data.setdefault(section, {})
        current.update(values)
        self.save()
