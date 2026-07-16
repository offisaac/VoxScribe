import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from voxscribe.backends import _speech_intervals
from voxscribe.config import SettingsStore
from voxscribe.exports import write_exports
from voxscribe.tasks import TaskStore
from voxscribe.transcription import Segment, TranscriptionResult
from voxscribe.viewer import TranscriptionViewer


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def test_result_round_trip():
    result = TranscriptionResult(
        [Segment(0.14, 1.2, "你好", "说话人 1", [{"start": 0.14, "end": 0.4, "word": "你"}])],
        "zh",
        1.2,
    )
    loaded = TranscriptionResult.from_json(result.to_json())
    assert loaded.text == "你好"
    assert loaded.segments[0].speaker == "说话人 1"
    assert loaded.segments[0].words[0]["word"] == "你"


def test_precise_exports(tmp_path):
    source = tmp_path / "meeting.wav"
    source.touch()
    result = TranscriptionResult(
        [Segment(0.14, 1.2, "第一句"), Segment(2.345, 4.567, "第二句")],
        "zh",
        4.567,
    )
    outputs = write_exports(result, source, tmp_path, ["txt", "srt", "vtt", "json"], "Test Model")
    assert len(outputs) == 4
    srt = (tmp_path / "meeting - Test Model.srt").read_text(encoding="utf-8")
    assert "00:00:00,140 --> 00:00:01,200" in srt
    assert "00:00:02,345 --> 00:00:04,567" in srt
    assert json.loads((tmp_path / "meeting - Test Model.json").read_text(encoding="utf-8"))["duration"] == 4.567


def test_task_store_lifecycle(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    source = tmp_path / "audio.wav"
    source.touch()
    queued = store.enqueue(source, "manual", "faster_whisper")
    assert store.get(queued)["status"] == "queued"
    store.cancel(queued)
    assert store.get(queued)["status"] == "cancelled"
    interrupted_queued = store.enqueue(source, "folder", "qwen3_asr")
    running = store.start(source, "manual", "qwen3_asr")
    assert store.get(running)["status"] == "running"
    assert store.recover_interrupted() == 2
    assert store.get(interrupted_queued)["status"] == "failed"
    assert store.get(running)["status"] == "failed"
    completed = store.start(source, "manual", "qwen3_asr")
    result = TranscriptionResult([Segment(0, 1, "完成")], "zh", 1)
    store.complete(completed, [tmp_path / "out.txt"], result)
    assert store.get(completed)["progress"] == 100
    assert TranscriptionResult.from_json(store.get(completed)["result_json"]).text == "完成"


def test_settings_atomic_merge(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"general":{"font_size":48}}', encoding="utf-8")
    store = SettingsStore(path)
    assert store.get("general", "font_size") == 48
    assert store.get("folder_watch", "enabled") is True
    store.update_section("live", {"chunk_seconds": 5.0})
    assert json.loads(path.read_text(encoding="utf-8"))["live"]["chunk_seconds"] == 5.0
    assert not path.with_suffix(".json.tmp").exists()


def test_vad_intervals_follow_speech():
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])
    intervals = _speech_intervals(audio, sample_rate)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert 0.7 * sample_rate < start < 1.1 * sample_rate
    assert 1.9 * sample_rate < end < 2.3 * sample_rate


def test_viewer_constructs_from_persisted_result(tmp_path, app):
    source = tmp_path / "viewer.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    store = TaskStore(tmp_path / "viewer.db")
    task_id = store.start(source, "manual", "faster_whisper")
    result = TranscriptionResult([Segment(0.1, 0.9, "可编辑文本")], "zh", 1)
    store.complete(task_id, [tmp_path / "viewer.txt"], result)
    viewer = TranscriptionViewer(store.get(task_id), store, "Faster Whisper", None)
    assert viewer.table.rowCount() == 1
    assert viewer.table.item(0, 3).text() == "可编辑文本"
    viewer.close()
