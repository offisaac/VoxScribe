import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from voxscribe.backends import FunASRNanoBackend, _speech_intervals
from voxscribe.config import SettingsStore
from voxscribe.exports import write_exports
from voxscribe.streaming import QwenStreamingService, QwenStreamingSession
from voxscribe.tasks import TaskStore
from voxscribe.transcription import Segment, TranscriptionResult, normalize_recognition_text
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


def test_recognition_text_is_simplified_chinese_and_keeps_english():
    assert normalize_recognition_text("開放時間：早上九點。") == "开放时间：早上九点。"
    assert normalize_recognition_text("Hello, ChatGPT.") == "Hello, ChatGPT."


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
    assert store.get("live", "recognition_mode") == "standard"
    assert store.get("live", "standard_backend") == "qwen3_asr"
    store.update_section("live", {"chunk_seconds": 5.0})
    assert json.loads(path.read_text(encoding="utf-8"))["live"]["chunk_seconds"] == 5.0
    assert not path.with_suffix(".json.tmp").exists()


def test_qwen_is_available_in_both_live_modes(tmp_path, app, monkeypatch):
    application_path = Path(__file__).resolve().parents[1] / "app" / "voxscribe.py"
    spec = importlib.util.spec_from_file_location("voxscribe_desktop_test", application_path)
    desktop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desktop)
    store = SettingsStore(tmp_path / "settings.json")
    monkeypatch.setattr(desktop, "SETTINGS", store)

    dialog = desktop.SettingsDialog(store)
    assert dialog.live_recognition_mode.findData("streaming") >= 0
    assert dialog.live_recognition_mode.findData("standard") >= 0
    assert dialog.live_backend.findData("qwen3_asr") >= 0
    dialog.live_recognition_mode.setCurrentIndex(dialog.live_recognition_mode.findData("standard"))
    dialog.live_backend.setCurrentIndex(dialog.live_backend.findData("qwen3_asr"))
    dialog._save()

    assert store.get("live", "recognition_mode") == "standard"
    assert store.get("live", "backend") == "qwen3_asr"
    recorder = desktop.LiveRecorder(None, None, desktop.Events(), TaskStore(tmp_path / "live.db"))
    assert recorder.recognition_mode == "standard"
    assert recorder.backend_name == "qwen3_asr"
    assert recorder.chunk_seconds == store.get("live", "chunk_seconds")

    store.update_section(
        "live",
        {"recognition_mode": "streaming", "backend": "qwen3_asr"},
    )
    recorder = desktop.LiveRecorder(None, None, desktop.Events(), TaskStore(tmp_path / "stream.db"))
    assert recorder.recognition_mode == "streaming"
    assert recorder.backend_name == "qwen3_asr"
    assert recorder.chunk_seconds == store.get("live", "stream_chunk_seconds")


def test_live_stream_failure_falls_back_to_standard_qwen(tmp_path, app, monkeypatch):
    application_path = Path(__file__).resolve().parents[1] / "app" / "voxscribe.py"
    spec = importlib.util.spec_from_file_location("voxscribe_fallback_test", application_path)
    desktop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desktop)
    store = SettingsStore(tmp_path / "settings.json")
    store.update_section(
        "live",
        {
            "recognition_mode": "streaming",
            "backend": "qwen3_asr",
            "standard_backend": "qwen3_asr",
        },
    )
    monkeypatch.setattr(desktop, "SETTINGS", store)

    class FakeManager:
        def __init__(self):
            self.loaded = []
            self.transcribed = []

        def ensure_loaded(self, backend_name):
            self.loaded.append(backend_name)

        def transcribe(self, audio, language=None, backend_name=None):
            self.transcribed.append((language, backend_name))
            return "普通模式继续识别"

    class FakeService:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class BrokenSession:
        def push(self, _audio):
            raise RuntimeError("HTTP 500")

    manager = FakeManager()
    service = FakeService()
    events = desktop.Events()
    mode_changes = []
    live_text = []
    events.live_mode_changed.connect(mode_changes.append)
    events.live_text.connect(live_text.append)
    recorder = desktop.LiveRecorder(manager, service, events, TaskStore(tmp_path / "fallback.db"))
    recorder.sample_rate = 16000
    recorder.chunk_seconds = 0.8
    recorder.export_enabled = False
    recorder.streaming_session = BrokenSession()
    recorder.session_started = 0.0
    recorder.audio_queue.put(np.ones(12800, dtype=np.float32) * 0.01)
    recorder.audio_queue.put(None)

    recorder._worker()

    assert service.stopped is True
    assert manager.loaded == ["qwen3_asr"]
    assert manager.transcribed == [(None, "qwen3_asr")]
    assert recorder.recognition_mode == "standard"
    assert mode_changes == ["qwen3_asr"]
    assert live_text == ["普通模式继续识别"]


def test_error_dialog_is_non_modal_and_reuses_single_window(app):
    application_path = Path(__file__).resolve().parents[1] / "app" / "voxscribe.py"
    spec = importlib.util.spec_from_file_location("voxscribe_error_dialog_test", application_path)
    desktop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desktop)
    window = QMainWindow()
    window.status_label = QLabel(window)
    window.error_dialog = None

    desktop.MainWindow._show_error(window, "第一次错误")
    first_dialog = window.error_dialog
    desktop.MainWindow._show_error(window, "第二次错误")

    assert window.error_dialog is first_dialog
    assert first_dialog.text() == "第二次错误"
    assert first_dialog.windowModality() == Qt.WindowModality.NonModal
    first_dialog.accept()
    app.processEvents()
    assert window.error_dialog is None
    window.close()


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


def test_fun_asr_result_mapping():
    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["language"] == "中文"
            self.input_path = Path(kwargs["input"])
            assert self.input_path.exists()
            return [{"text": "高速字幕。", "timestamps": [{"token": "高", "start_time": 0.1, "end_time": 0.2}]}]

    backend = FunASRNanoBackend.__new__(FunASRNanoBackend)
    backend.model = FakeModel()
    result = backend.transcribe_result((np.zeros(16000, dtype=np.float32), 16000), "Chinese")
    assert result.text == "高速字幕。"
    assert result.segments[0].words[0] == {"start": 0.1, "end": 0.2, "word": "高"}
    assert not backend.model.input_path.exists()


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


def test_qwen_streaming_http_session(monkeypatch):
    calls = []
    responses = iter(
        [
            {"session_id": "session-1"},
            {"language": "Chinese", "text": "partial"},
            {"language": "Chinese", "text": "final"},
        ]
    )

    class FakeResponse:
        status = 200

        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.value).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.data, request.headers, timeout))
        return FakeResponse(next(responses))

    monkeypatch.setattr("voxscribe.streaming.urlopen", fake_urlopen)
    session = QwenStreamingSession("http://127.0.0.1:8765").start()
    partial = session.push(np.zeros(16000, dtype=np.float32))
    final = session.finish()

    assert partial["text"] == "partial"
    assert final["text"] == "final"
    assert calls[0][0].endswith("/api/start")
    assert json.loads(calls[0][1].decode("utf-8")) == {
        "chunk_size_sec": 0.8,
        "unfixed_chunk_num": 4,
        "unfixed_token_num": 5,
    }
    assert calls[1][0].endswith("/api/chunk?session_id=session-1")
    assert len(calls[1][1]) == 16000 * 4
    assert calls[2][0].endswith("/api/finish?session_id=session-1")


def test_qwen_streaming_session_rotates_before_audio_grows_unbounded(monkeypatch):
    responses = iter(
        [
            {"session_id": "session-1"},
            {"language": "Chinese", "text": "第一段进行中"},
            {"language": "Chinese", "text": "第一段完成"},
            {"session_id": "session-2"},
            {"language": "Chinese", "text": "第二段进行中"},
            {"language": "Chinese", "text": "第二段完成"},
        ]
    )
    calls = []

    class FakeResponse:
        status = 200

        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.value).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return FakeResponse(next(responses))

    monkeypatch.setattr("voxscribe.streaming.urlopen", fake_urlopen)
    session = QwenStreamingSession(
        "http://127.0.0.1:8765",
        max_session_audio_seconds=1.0,
    ).start()
    session.push(np.zeros(12800, dtype=np.float32))
    rotated = session.push(np.zeros(12800, dtype=np.float32))
    final = session.finish()

    assert rotated["text"] == "第一段完成\n第二段进行中"
    assert final["text"] == "第一段完成\n第二段完成"
    assert sum("/api/start" in call for call in calls) == 2
    assert sum("/api/finish" in call for call in calls) == 2


def test_qwen_streaming_session_retries_transient_failure(monkeypatch):
    attempts = []

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("service is restarting")
            return self

    service = QwenStreamingService()
    monkeypatch.setattr(service, "ensure_started", lambda: None)
    monkeypatch.setattr("voxscribe.streaming.QwenStreamingSession", FakeSession)
    monkeypatch.setattr("voxscribe.streaming.time.sleep", lambda _seconds: None)

    assert service.create_session().__class__ is FakeSession
    assert len(attempts) == 3


def test_qwen_service_stop_terminates_wsl_immediately(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)

    service = QwenStreamingService(distro="Ubuntu")
    service.ready = True
    monkeypatch.setattr("voxscribe.streaming.subprocess.run", fake_run)

    service.stop()

    assert calls == [["wsl.exe", "--shutdown"]]
    assert service.ready is False
