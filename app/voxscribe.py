from __future__ import annotations

import os
import logging
from logging.handlers import RotatingFileHandler
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
TEMP_DIR = ROOT / "cache" / "temp"

os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / "models" / "huggingface" / "hub"))
os.environ.setdefault("TEMP", str(TEMP_DIR))
os.environ.setdefault("TMP", str(TEMP_DIR))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

from voxscribe.config import SettingsStore
from voxscribe.backends import BACKEND_INFO, create_backend
from voxscribe.tasks import TaskStore
from voxscribe.hotkeys import HotkeyManager
from voxscribe.exports import write_exports
from voxscribe.streaming import QwenStreamingService
from voxscribe.transcription import Segment, TranscriptionResult

SETTINGS = SettingsStore(ROOT / "config" / "settings.json")


def refresh_settings_paths():
    global MODEL_DIR, LIVE_EXPORT_DIR, WATCH_INPUT_DIR, FILE_OUTPUT_DIR
    MODEL_DIR = Path(SETTINGS.get("model", "model_path"))
    LIVE_EXPORT_DIR = Path(SETTINGS.get("live", "export_folder"))
    WATCH_INPUT_DIR = Path(SETTINGS.get("folder_watch", "input_folder"))
    FILE_OUTPUT_DIR = Path(SETTINGS.get("folder_watch", "output_folder"))
    LIVE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    WATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


refresh_settings_paths()


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("voxscribe")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_DIR / "voxscribe.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def configure_windows_taskbar(window):
    if sys.platform != "win32":
        return
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        @classmethod
        def parse(cls, value):
            data = uuid.UUID(value).bytes_le
            return cls.from_buffer_copy(data)

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]

    class PROPVARIANT_VALUE(ctypes.Union):
        _fields_ = [("pwszVal", ctypes.c_wchar_p), ("ullVal", ctypes.c_ulonglong)]

    class PROPVARIANT(ctypes.Structure):
        _anonymous_ = ("value",)
        _fields_ = [
            ("vt", ctypes.c_ushort),
            ("wReserved1", ctypes.c_ushort),
            ("wReserved2", ctypes.c_ushort),
            ("wReserved3", ctypes.c_ushort),
            ("value", PROPVARIANT_VALUE),
        ]

    shell32 = ctypes.windll.shell32
    store_pointer = ctypes.c_void_p()
    iid_store = GUID.parse("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")
    result = shell32.SHGetPropertyStoreForWindow(
        wintypes.HWND(int(window.winId())), ctypes.byref(iid_store), ctypes.byref(store_pointer)
    )
    if result != 0 or not store_pointer.value:
        raise OSError(result, "无法取得 Windows 窗口属性")

    table = ctypes.cast(store_pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    set_value = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT)
    )(table[6])
    commit = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(table[7])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])
    app_model_guid = GUID.parse("9f4c2855-9f79-4b39-a8d0-e1d42de1d5f3")
    properties = {
        5: "VoxScribe.Desktop.1",
        2: f'"{ROOT / "VoxScribe.exe"}"',
        4: "VoxScribe",
        3: f"{ROOT / 'VoxScribe.exe'},0",
    }
    try:
        for property_id, value in properties.items():
            key = PROPERTYKEY(app_model_guid, property_id)
            variant = PROPVARIANT(vt=31, pwszVal=value)
            result = set_value(store_pointer, ctypes.byref(key), ctypes.byref(variant))
            if result != 0:
                raise OSError(result, f"无法设置任务栏属性 {property_id}")
        result = commit(store_pointer)
        if result != 0:
            raise OSError(result, "无法提交 Windows 任务栏属性")
    finally:
        release(store_pointer)


def write_transcript_exports(result, source, output_dir, formats, backend_name=None):
    backend_name = backend_name or SETTINGS.get("model", "backend", "qwen3_asr")
    backend_label = BACKEND_INFO.get(backend_name, {}).get("label", backend_name)
    return write_exports(result, source, output_dir, formats, backend_label)

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QEvent, QLockFile, QObject, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizeGrip,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class Events(QObject):
    status = Signal(str)
    live_text = Signal(str)
    live_snapshot = Signal(str)
    offline_text = Signal(str)
    error = Signal(str)
    model_ready = Signal()
    history_changed = Signal()
    record_hotkey = Signal()
    floating_hotkey = Signal()
    audio_level = Signal(float)


class SettingsDialog(QDialog):
    def __init__(self, store: SettingsStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("偏好设置")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        content = QHBoxLayout()
        self.navigation = QListWidget()
        self.navigation.addItems(["常规", "实时录制", "文件转写", "音频处理", "识别模型", "快捷键"])
        self.navigation.setFixedWidth(132)
        self.navigation.setSpacing(3)
        self.navigation.setStyleSheet(
            "QListWidget{background:#111720;border:0;border-radius:9px;padding:6px;}"
            "QListWidget::item{padding:11px 12px;border-radius:7px;color:#aab6c6;}"
            "QListWidget::item:hover{background:#1d2633;color:#e8edf5;}"
            "QListWidget::item:selected{background:#285fae;color:white;font-weight:600;}"
        )
        self.pages = QStackedWidget()
        for page in [self._general_tab(), self._live_tab(), self._folder_tab(), self._audio_tab(), self._model_tab(), self._hotkey_tab()]:
            self.pages.addWidget(page)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        content.addWidget(self.navigation)
        content.addWidget(self.pages, 1)
        layout.addLayout(content, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _general_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.settings_font_size = QSpinBox()
        self.settings_font_size.setRange(18, 72)
        self.settings_font_size.setValue(self.store.get("general", "font_size", 32))
        form.addRow("实时字幕字号", self.settings_font_size)
        theme = QComboBox()
        theme.addItem("深色", "dark")
        form.addRow("界面主题", theme)
        note = QLabel("配置保存在软件目录 config\\settings.json，可随整个软件文件夹迁移。")
        note.setWordWrap(True)
        note.setObjectName("hintText")
        form.addRow(note)
        return page

    def _folder_picker(self, line_edit):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("浏览")
        button.clicked.connect(lambda: self._browse_folder(line_edit))
        layout.addWidget(button)
        return row

    def _browse_folder(self, line_edit):
        selected = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if selected:
            line_edit.setText(selected)

    def _live_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.live_export_enabled = QCheckBox("启用实时录制转录导出")
        self.live_export_enabled.setChecked(self.store.get("live", "export_enabled", True))
        form.addRow(self.live_export_enabled)
        self.live_export_folder = QLineEdit(self.store.get("live", "export_folder"))
        form.addRow("导出文件夹", self._folder_picker(self.live_export_folder))
        self.live_file_name = QLineEdit(self.store.get("live", "file_name"))
        form.addRow("文件名模板", self.live_file_name)
        self.live_chunk_seconds = QDoubleSpinBox()
        self.live_chunk_seconds.setRange(1.5, 30.0)
        self.live_chunk_seconds.setSingleStep(0.5)
        self.live_chunk_seconds.setValue(float(self.store.get("live", "chunk_seconds", 3.5)))
        self.live_chunk_seconds.setSuffix(" 秒")
        form.addRow("识别分段", self.live_chunk_seconds)
        self.live_silence = QDoubleSpinBox()
        self.live_silence.setDecimals(4)
        self.live_silence.setRange(0.0, 0.1)
        self.live_silence.setSingleStep(0.0005)
        self.live_silence.setValue(float(self.store.get("live", "silence_threshold", 0.0025)))
        form.addRow("静音阈值", self.live_silence)
        self.live_device_keyword = QLineEdit(self.store.get("live", "device_keyword", "CABLE Output"))
        form.addRow("默认音频设备关键词", self.live_device_keyword)
        self.live_backend = QComboBox()
        for backend_name, info in BACKEND_INFO.items():
            self.live_backend.addItem(info["label"], backend_name)
        live_backend_index = self.live_backend.findData(self.store.get("live", "backend", "faster_whisper"))
        if live_backend_index >= 0:
            self.live_backend.setCurrentIndex(live_backend_index)
        form.addRow("实时识别模型", self.live_backend)
        self.live_backend_note = QLabel()
        self.live_backend_note.setObjectName("hintText")
        self.live_backend_note.setWordWrap(True)
        form.addRow(self.live_backend_note)
        self.live_backend.currentIndexChanged.connect(self._live_backend_changed)
        self._live_backend_changed()
        return page

    def _live_backend_changed(self):
        if self.live_backend.currentData() == "fun_asr_nano":
            if self.live_chunk_seconds.value() > 2.0:
                self.live_chunk_seconds.setValue(2.0)
            self.live_backend_note.setText("高速模式使用 2 秒分段，优先降低字幕等待时间；Qwen3-ASR 1.7B 仍是默认精度模式。")
        else:
            self.live_backend_note.setText("实时模型只影响录制字幕；文件转写模型可在“识别模型”中单独设置。")

    def _folder_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.watch_enabled = QCheckBox("开启文件夹监控")
        self.watch_enabled.setChecked(self.store.get("folder_watch", "enabled", True))
        form.addRow(self.watch_enabled)
        self.watch_input = QLineEdit(self.store.get("folder_watch", "input_folder"))
        form.addRow("输入文件夹", self._folder_picker(self.watch_input))
        self.watch_output = QLineEdit(self.store.get("folder_watch", "output_folder"))
        form.addRow("输出文件夹", self._folder_picker(self.watch_output))
        self.watch_delete = QCheckBox("转写成功后删除输入文件")
        self.watch_delete.setChecked(self.store.get("folder_watch", "delete_processed_files", False))
        form.addRow(self.watch_delete)
        formats = QWidget()
        formats_layout = QHBoxLayout(formats)
        formats_layout.setContentsMargins(0, 0, 0, 0)
        enabled_formats = self.store.get("folder_watch", "export_formats", ["txt"])
        self.format_txt = QCheckBox("TXT")
        self.format_txt.setChecked("txt" in enabled_formats)
        self.format_srt = QCheckBox("SRT")
        self.format_srt.setChecked("srt" in enabled_formats)
        self.format_vtt = QCheckBox("VTT")
        self.format_vtt.setChecked("vtt" in enabled_formats)
        formats_layout.addWidget(self.format_txt)
        formats_layout.addWidget(self.format_srt)
        formats_layout.addWidget(self.format_vtt)
        formats_layout.addStretch(1)
        form.addRow("自动导出", formats)
        return page

    def _model_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.model_backend = QComboBox()
        for backend_name, info in BACKEND_INFO.items():
            self.model_backend.addItem(info["label"], backend_name)
        active_backend = self.store.get("model", "backend", "qwen3_asr")
        backend_index = self.model_backend.findData(active_backend)
        if backend_index >= 0:
            self.model_backend.setCurrentIndex(backend_index)
        form.addRow("模型后端", self.model_backend)
        self.model_path = QLineEdit(self.store.get("model", "model_path"))
        form.addRow("模型路径", self._folder_picker(self.model_path))
        self.model_command = QLineEdit(self.store.get("model", "external_cli_command", ""))
        self.model_command.setPlaceholderText('例如：whisper-cli.exe -m {model} -f {input} -l {language}')
        self.model_command_label = QLabel("本地命令模板")
        form.addRow(self.model_command_label, self.model_command)
        self.model_backend.currentIndexChanged.connect(self._model_backend_changed)
        self.model_language = QComboBox()
        self.model_language.addItem("自动检测（中英混说）", "auto")
        self.model_language.addItem("中文", "Chinese")
        self.model_language.addItem("英文", "English")
        language_index = self.model_language.findData(self.store.get("model", "language", "auto"))
        if language_index >= 0:
            self.model_language.setCurrentIndex(language_index)
        form.addRow("语言", self.model_language)
        note = QLabel("Qwen3-ASR 1.7B 是默认精度模式；Fun-ASR-Nano 是低延迟高速模式。切换模型无需改动录制和文件夹工作流。")
        note.setWordWrap(True)
        note.setObjectName("hintText")
        form.addRow(note)
        self._model_backend_changed()
        return page

    def _audio_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.audio_mode = QComboBox()
        self.audio_mode.addItem("关闭（保留原始音频）", "off")
        self.audio_mode.addItem("智能降噪（推荐）", "noise_reduce")
        self.audio_mode.addItem("人声分离 Demucs（嘈杂环境）", "vocals")
        current = self.audio_mode.findData(self.store.get("audio_processing", "mode", "noise_reduce"))
        if current >= 0:
            self.audio_mode.setCurrentIndex(current)
        form.addRow("转写前处理", self.audio_mode)
        self.demucs_model = QLineEdit(self.store.get("audio_processing", "demucs_model", "htdemucs"))
        form.addRow("Demucs 模型", self.demucs_model)
        self.speaker_identification = QCheckBox("启用本地说话人识别")
        self.speaker_identification.setChecked(
            self.store.get("audio_processing", "speaker_identification", False)
        )
        form.addRow(self.speaker_identification)
        self.speaker_count = QSpinBox()
        self.speaker_count.setRange(0, 10)
        self.speaker_count.setValue(self.store.get("audio_processing", "speaker_count", 0))
        self.speaker_count.setSpecialValueText("自动")
        form.addRow("说话人数", self.speaker_count)
        note = QLabel("智能降噪适合会议和面试；人声分离更强但速度较慢。全部在本机执行。")
        note.setWordWrap(True)
        note.setObjectName("hintText")
        form.addRow(note)
        return page

    def _model_backend_changed(self):
        backend_name = self.model_backend.currentData()
        if not backend_name:
            return
        default_path = BACKEND_INFO[backend_name]["default_path"]
        model_path = self.store.get("model", f"{backend_name}_path", default_path)
        self.model_path.setText(model_path)
        is_external = backend_name == "external_cli"
        self.model_command_label.setVisible(is_external)
        self.model_command.setVisible(is_external)

    def _hotkey_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.record_hotkey = QLineEdit(self.store.get("hotkeys", "record_toggle", "Ctrl+Shift+R"))
        form.addRow("录制/停止", self.record_hotkey)
        self.floating_hotkey = QLineEdit(self.store.get("hotkeys", "floating_window", "Ctrl+Shift+F"))
        form.addRow("演示悬浮窗", self.floating_hotkey)
        note = QLabel("快捷键配置已经持久化；全局热键监听将在快捷键模块中启用。")
        note.setWordWrap(True)
        note.setObjectName("hintText")
        form.addRow(note)
        return page

    def _save(self):
        formats = []
        if self.format_txt.isChecked():
            formats.append("txt")
        if self.format_srt.isChecked():
            formats.append("srt")
        if self.format_vtt.isChecked():
            formats.append("vtt")
        if not formats:
            formats = ["txt"]
        self.store.update_section("general", {"font_size": self.settings_font_size.value()})
        self.store.update_section(
            "live",
            {
                "device_keyword": self.live_device_keyword.text().strip() or "CABLE Output",
                "backend": self.live_backend.currentData(),
                "export_enabled": self.live_export_enabled.isChecked(),
                "export_folder": self.live_export_folder.text().strip(),
                "file_name": self.live_file_name.text().strip() or "Meeting Transcript {date_time}",
                "chunk_seconds": self.live_chunk_seconds.value(),
                "silence_threshold": self.live_silence.value(),
            },
        )
        self.store.update_section(
            "folder_watch",
            {
                "enabled": self.watch_enabled.isChecked(),
                "input_folder": self.watch_input.text().strip(),
                "output_folder": self.watch_output.text().strip(),
                "delete_processed_files": self.watch_delete.isChecked(),
                "export_formats": formats,
            },
        )
        backend_name = self.model_backend.currentData()
        model_path = self.model_path.text().strip()
        self.store.update_section(
            "model",
            {
                "backend": backend_name,
                "model_path": model_path,
                f"{backend_name}_path": model_path,
                "external_cli_command": self.model_command.text().strip(),
                "language": self.model_language.currentData(),
            },
        )
        self.store.update_section(
            "audio_processing",
            {
                "mode": self.audio_mode.currentData(),
                "demucs_model": self.demucs_model.text().strip() or "htdemucs",
                "speaker_identification": self.speaker_identification.isChecked(),
                "speaker_count": self.speaker_count.value(),
            },
        )
        self.store.update_section(
            "hotkeys",
            {
                "record_toggle": self.record_hotkey.text().strip(),
                "floating_window": self.floating_hotkey.text().strip(),
            },
        )
        refresh_settings_paths()
        self.accept()


class ModelManager:
    def __init__(self, events: Events):
        self.events = events
        self.backend = None
        self.backend_key = None
        self.lock = threading.Lock()

    def ensure_loaded(self, backend_name=None):
        backend_name = backend_name or SETTINGS.get("model", "backend", "qwen3_asr")
        default_path = BACKEND_INFO.get(backend_name, {}).get("default_path", "")
        model_path = SETTINGS.get("model", f"{backend_name}_path", default_path)
        backend_key = (backend_name, model_path)
        with self.lock:
            if self.backend is not None and self.backend_key == backend_key:
                return self.backend
            self.backend = None
            self.backend_key = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            label = BACKEND_INFO.get(backend_name, {}).get("label", backend_name)
            self.events.status.emit(f"正在加载 {label}…")
            self.backend = create_backend(
                backend_name,
                model_path,
                {"command_template": SETTINGS.get("model", "external_cli_command", "")},
            )
            self.backend_key = backend_key
            self.events.status.emit(f"{label} 就绪 · 全程本地")
            self.events.model_ready.emit()
            return self.backend

    def transcribe(self, audio, language=None, backend_name=None):
        return self.transcribe_result(audio, language, backend_name).text

    def transcribe_result(self, audio, language=None, backend_name=None):
        backend = self.ensure_loaded(backend_name)
        with self.lock:
            return backend.transcribe_result(audio, language)

    def unload(self):
        with self.lock:
            self.backend = None
            self.backend_key = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LiveRecorder:
    def __init__(self, manager: ModelManager, streaming_service, events: Events, tasks: TaskStore):
        self.manager = manager
        self.streaming_service = streaming_service
        self.events = events
        self.tasks = tasks
        self.audio_queue = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.stream = None
        self.thread = None
        self.sample_rate = 48000
        backend_name = SETTINGS.get("live", "backend", "faster_whisper")
        configured_chunk = float(SETTINGS.get("live", "chunk_seconds", 3.5))
        if backend_name == "qwen3_asr":
            self.chunk_seconds = float(SETTINGS.get("live", "stream_chunk_seconds", 0.8))
        else:
            self.chunk_seconds = min(configured_chunk, 2.0) if backend_name == "fun_asr_nano" else configured_chunk
        self.silence_threshold = float(SETTINGS.get("live", "silence_threshold", 0.0025))
        self.export_enabled = SETTINGS.get("live", "export_enabled", True)
        self.session_file = None
        self.fixed_obs_file = LIVE_EXPORT_DIR / SETTINGS.get("live", "obs_file_name", "obs_live_caption.txt")
        self.task_id = None
        self.segments = []
        self.session_started = 0.0
        self.last_text = ""
        self.last_level_at = 0.0
        self.streaming_session = None

    def start(self, device_index: int):
        self.stop()
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        refresh_settings_paths()
        LIVE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.stop_event.clear()
        backend_name = SETTINGS.get("live", "backend", "faster_whisper")
        configured_chunk = float(SETTINGS.get("live", "chunk_seconds", 3.5))
        if backend_name == "qwen3_asr":
            self.chunk_seconds = float(SETTINGS.get("live", "stream_chunk_seconds", 0.8))
        else:
            self.chunk_seconds = min(configured_chunk, 2.0) if backend_name == "fun_asr_nano" else configured_chunk
        self.silence_threshold = float(SETTINGS.get("live", "silence_threshold", 0.0025))
        self.export_enabled = SETTINGS.get("live", "export_enabled", True)
        self.fixed_obs_file = LIVE_EXPORT_DIR / SETTINGS.get("live", "obs_file_name", "obs_live_caption.txt")
        self.sample_rate = int(sd.query_devices(device_index)["default_samplerate"])
        stamp = datetime.now().strftime("%d-%b-%Y %H-%M-%S")
        template = SETTINGS.get("live", "file_name", "Meeting Transcript {date_time}")
        try:
            file_name = template.format(date_time=stamp)
        except (KeyError, ValueError):
            file_name = f"Meeting Transcript {stamp}"
        self.session_file = LIVE_EXPORT_DIR / f"{file_name}.txt"
        if self.export_enabled:
            self.session_file.write_text("", encoding="utf-8")
            self.fixed_obs_file.write_text("", encoding="utf-8")
        self.task_id = self.tasks.start(self.session_file, "live", backend_name)
        self.segments = []
        self.session_started = time.monotonic()
        self.last_text = ""
        self.last_level_at = 0.0
        if backend_name == "qwen3_asr":
            self.streaming_session = self.streaming_service.create_session()
        self.events.history_changed.emit()

        def callback(indata, frames, time_info, status):
            if status:
                self.events.status.emit(f"音频状态：{status}")
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            now = time.monotonic()
            if now - self.last_level_at >= 0.08:
                self.last_level_at = now
                self.events.audio_level.emit(float(np.sqrt(np.mean(np.square(mono)) + 1e-12)))
            try:
                self.audio_queue.put_nowait(mono)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(mono)
                except queue.Empty:
                    pass
                self.events.status.emit("实时音频缓存已满，已丢弃最旧数据以保持低延迟")

        self.stream = sd.InputStream(
            device=device_index,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=0,
        )
        self.stream.start()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.events.status.emit(f"正在监听 CABLE Output · {self.sample_rate} Hz")

    def stop(self):
        self.stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=15)
            self.thread = None
        if self.streaming_session is not None:
            try:
                response = self.streaming_session.finish()
                text = (response.get("text") or "").strip()
                if text:
                    elapsed = time.monotonic() - self.session_started
                    self.last_text = text
                    self.segments = [Segment(0.0, elapsed, text)]
                    self._save_snapshot(text)
                    self.events.live_snapshot.emit(text)
            except Exception:
                LOGGER.exception("Qwen streaming session finalization failed")
            finally:
                self.streaming_session = None
        self._finalize_session()

    def _worker(self):
        chunks = []
        total = 0
        target = int(self.sample_rate * self.chunk_seconds)
        backend_name = SETTINGS.get("live", "backend", "faster_whisper")
        while not self.stop_event.is_set():
            try:
                block = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            chunks.append(block)
            total += len(block)
            if total < target:
                continue
            audio = np.concatenate(chunks)
            chunks = []
            total = 0
            if backend_name != "qwen3_asr" and float(np.sqrt(np.mean(np.square(audio)) + 1e-12)) < self.silence_threshold:
                continue
            try:
                if self.sample_rate != 16000:
                    from scipy.signal import resample_poly

                    audio = resample_poly(audio, 16000, self.sample_rate).astype(np.float32)
                if backend_name == "qwen3_asr":
                    response = self.streaming_session.push(audio)
                    text = (response.get("text") or "").strip()
                    if text and text != self.last_text:
                        self.last_text = text
                        elapsed = time.monotonic() - self.session_started
                        self.segments = [Segment(0.0, elapsed, text)]
                        self._save_snapshot(text)
                        self.events.live_snapshot.emit(text)
                    continue
                language = SETTINGS.get("model", "language", "auto")
                text = self.manager.transcribe(
                    (audio, 16000),
                    language=None if language == "auto" else language,
                    backend_name=SETTINGS.get("live", "backend", "faster_whisper"),
                )
                if text:
                    text = self._remove_overlap(text)
                    if text:
                        elapsed = time.monotonic() - self.session_started
                        self.segments.append(Segment(max(0.0, elapsed - self.chunk_seconds), elapsed, text))
                        self._save(text)
                        self.events.live_text.emit(text)
            except Exception as exc:
                LOGGER.exception("实时识别失败")
                self.events.error.emit(f"实时识别失败：{exc}")

    def _remove_overlap(self, text):
        text = text.strip()
        if not text or text == self.last_text:
            return ""
        maximum = min(len(self.last_text), len(text), 24)
        matched = False
        for trailing_trim in range(0, 3):
            previous = self.last_text[:-trailing_trim] if trailing_trim else self.last_text
            for size in range(min(len(previous), maximum), 1, -1):
                if previous[-size:] == text[:size]:
                    text = text[size:].lstrip()
                    matched = True
                    break
            if matched:
                break
        if text:
            self.last_text = (self.last_text + text)[-80:]
        return text

    def _finalize_session(self):
        if self.task_id is None:
            return
        duration = max(0.0, time.monotonic() - self.session_started)
        result = TranscriptionResult(self.segments, SETTINGS.get("model", "language", ""), duration)
        try:
            self.tasks.complete(self.task_id, [self.session_file] if self.session_file else [], result)
            self.events.history_changed.emit()
        except Exception:
            LOGGER.exception("实时任务收尾失败")
        finally:
            self.task_id = None

    def _save(self, text: str):
        if not self.export_enabled:
            return
        line = text.strip() + "\n"
        with self.session_file.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self.fixed_obs_file.write_text(text.strip(), encoding="utf-8")

    def _save_snapshot(self, text: str):
        if not self.export_enabled:
            return
        self.session_file.write_text(text.strip() + "\n", encoding="utf-8")
        self.fixed_obs_file.write_text(text.strip(), encoding="utf-8")


class FileTaskQueue:
    def __init__(self, manager: ModelManager, streaming_service, events: Events, tasks: TaskStore, is_live):
        self.manager = manager
        self.streaming_service = streaming_service
        self.events = events
        self.tasks = tasks
        self.is_live = is_live
        self.pending = queue.Queue()
        self.active_signatures = set()
        self.cancelled = set()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def enqueue(self, source, trigger="manual"):
        source = Path(source)
        if not source.exists() or not source.is_file():
            self.events.error.emit(f"文件不存在：{source}")
            return None
        stat = source.stat()
        backend_name = SETTINGS.get("model", "backend", "qwen3_asr")
        signature = (str(source.resolve()).lower(), stat.st_size, stat.st_mtime_ns, backend_name)
        with self.lock:
            if signature in self.active_signatures:
                self.events.status.emit(f"已在队列中，跳过重复任务：{source.name}")
                return None
            self.active_signatures.add(signature)
        task_id = self.tasks.enqueue(source, trigger, backend_name)
        self.pending.put((task_id, source, trigger, backend_name, signature))
        self.events.history_changed.emit()
        self.events.status.emit(f"已加入转写队列：{source.name}")
        return task_id

    def cancel(self, task_id):
        self.cancelled.add(int(task_id))
        self.tasks.cancel(task_id)
        self.events.history_changed.emit()

    def stop(self):
        self.stop_event.set()
        self.pending.put(None)

    def _worker(self):
        while not self.stop_event.is_set():
            item = self.pending.get()
            if item is None:
                break
            task_id, source, trigger, backend_name, signature = item
            try:
                if task_id in self.cancelled:
                    continue
                while self.is_live() and not self.stop_event.wait(0.5):
                    pass
                if self.stop_event.is_set():
                    break
                self.tasks.mark_running(task_id)
                self.events.history_changed.emit()
                self.events.status.emit(f"正在转写：{source.name}")
                language = SETTINGS.get("model", "language", "auto")
                processing_mode = SETTINGS.get("audio_processing", "mode", "noise_reduce")
                audio_input = str(source)
                if processing_mode != "off":
                    from voxscribe.preprocessing import preprocess_audio

                    self.events.status.emit(f"正在进行音频预处理：{source.name}")
                    audio_input = preprocess_audio(
                        source,
                        processing_mode,
                        SETTINGS.get("audio_processing", "demucs_model", "htdemucs"),
                    )
                live_backend = SETTINGS.get("live", "backend", "faster_whisper")
                if backend_name == "qwen3_asr":
                    result = self.streaming_service.transcribe_result(audio_input)
                else:
                    if live_backend == "qwen3_asr":
                        self.streaming_service.stop_engine()
                    result = self.manager.transcribe_result(
                        audio_input,
                        language=None if language == "auto" else language,
                        backend_name=backend_name,
                    )
                if SETTINGS.get("audio_processing", "speaker_identification", False):
                    from voxscribe.diarization import assign_speakers

                    self.events.status.emit(f"正在识别说话人：{source.name}")
                    result = assign_speakers(
                        result,
                        audio_input,
                        SETTINGS.get("audio_processing", "speaker_count", 0),
                    )
                formats = SETTINGS.get("folder_watch", "export_formats", ["txt"])
                outputs = write_transcript_exports(result, source, FILE_OUTPUT_DIR, formats, backend_name)
                live_backend = SETTINGS.get("live", "backend", "faster_whisper")
                if live_backend == "qwen3_asr" and backend_name != "qwen3_asr":
                    self.manager.unload()
                    self.streaming_service.ensure_started()
                    self.events.model_ready.emit()
                elif backend_name != live_backend:
                    self.events.status.emit("文件转写完成，正在恢复实时识别模型…")
                    self.manager.ensure_loaded(live_backend)
                self.tasks.complete(task_id, outputs, result)
                self.events.offline_text.emit(result.text)
                self.events.status.emit(f"转写完成：{outputs[0] if outputs else FILE_OUTPUT_DIR}")
                if trigger == "folder_watch" and SETTINGS.get("folder_watch", "delete_processed_files", False):
                    source.unlink()
            except Exception as exc:
                self.tasks.fail(task_id, exc)
                self.events.error.emit(f"文件转写失败：{source.name} · {exc}")
            finally:
                with self.lock:
                    self.active_signatures.discard(signature)
                self.events.history_changed.emit()


class FolderWatcher:
    MEDIA_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}

    def __init__(self, task_queue: FileTaskQueue, events: Events, is_live):
        self.task_queue = task_queue
        self.events = events
        self.is_live = is_live
        self.enabled = SETTINGS.get("folder_watch", "enabled", True)
        self.stop_event = threading.Event()
        self.sizes = {}
        self.retry_after = {}
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _worker(self):
        while not self.stop_event.wait(2):
            if not self.enabled or self.is_live():
                continue
            for source in sorted(WATCH_INPUT_DIR.iterdir()):
                if self.stop_event.is_set() or not self.enabled or self.is_live():
                    break
                if not source.is_file() or source.suffix.lower() not in self.MEDIA_SUFFIXES:
                    continue
                if self.retry_after.get(source, 0) > time.time():
                    continue
                formats = SETTINGS.get("folder_watch", "export_formats", ["txt"])
                backend_name = SETTINGS.get("model", "backend", "qwen3_asr")
                backend_label = BACKEND_INFO.get(backend_name, {}).get("label", backend_name)
                output = FILE_OUTPUT_DIR / f"{source.stem} - {backend_label}.{formats[0]}"
                if output.exists():
                    continue
                try:
                    size = source.stat().st_size
                except OSError:
                    continue
                if self.sizes.get(source) != size:
                    self.sizes[source] = size
                    continue
                if self.task_queue.enqueue(source, "folder_watch"):
                    self.retry_after[source] = time.time() + 60


class FloatingCaptionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.drag_position = QPoint()
        self.font_size = 34
        self.position_initialized = False
        self.setWindowTitle("VoxScribe 演示字幕")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(760, 260)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("captionFrame")
        frame.setStyleSheet(
            "QFrame#captionFrame{background:rgba(8,10,14,235);border:1px solid #434a58;border-radius:12px;}"
            "QPushButton{background:#29303a;border:0;border-radius:5px;padding:5px 10px;color:white;}"
            "QLabel{color:#aeb8c8;}"
            "QTextEdit{background:transparent;border:0;color:white;padding:8px;}"
        )
        outer.addWidget(frame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 8, 6)

        self.title_bar = QFrame()
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(2, 0, 0, 0)
        self.drag_label = QLabel("拖动此处移动 · 滚轮查看历史")
        title_layout.addWidget(self.drag_label)
        title_layout.addStretch(1)
        smaller = QPushButton("A−")
        smaller.clicked.connect(lambda: self.change_font(-2))
        title_layout.addWidget(smaller)
        larger = QPushButton("A+")
        larger.clicked.connect(lambda: self.change_font(2))
        title_layout.addWidget(larger)
        close = QPushButton("关闭悬浮窗")
        close.clicked.connect(self.hide)
        title_layout.addWidget(close)
        layout.addWidget(self.title_bar)
        self.title_bar.installEventFilter(self)
        self.drag_label.installEventFilter(self)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.text, 1)
        grip_row = QHBoxLayout()
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self))
        layout.addLayout(grip_row)
        self.apply_font()

    def change_font(self, delta):
        self.font_size = max(18, min(72, self.font_size + delta))
        self.apply_font()

    def apply_font(self):
        self.text.setFont(QFont("Microsoft YaHei UI", self.font_size))

    def append_text(self, text):
        self.text.append(text)
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text.setTextCursor(cursor)

    def set_text(self, text):
        self.text.setPlainText(text)
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text.setTextCursor(cursor)

    def show_for(self, anchor):
        self.show()
        screens = QApplication.screens()
        is_visible = any(screen.availableGeometry().intersects(self.frameGeometry()) for screen in screens)
        if not self.position_initialized or not is_visible:
            screen = QApplication.screenAt(anchor.frameGeometry().center()) or QApplication.primaryScreen()
            area = screen.availableGeometry()
            x = area.left() + max(16, (area.width() - self.width()) // 2)
            y = area.top() + max(16, (area.height() - self.height()) // 3)
            self.move(x, y)
            self.position_initialized = True
        self.raise_()
        self.activateWindow()

    def eventFilter(self, watched, event):
        if watched in (self.title_bar, self.drag_label) and event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            return True
        if watched in (self.title_bar, self.drag_label) and event.type() == QEvent.Type.MouseMove and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            return True
        if watched in (self.title_bar, self.drag_label) and event.type() == QEvent.Type.MouseButtonRelease:
            self.drag_position = QPoint()
            return True
        return super().eventFilter(watched, event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        LIVE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        WATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        FILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.events = Events()
        self.tasks = TaskStore(ROOT / "data" / "tasks.db")
        self.tasks.recover_interrupted()
        self.manager = ModelManager(self.events)
        self.streaming_service = QwenStreamingService(
            SETTINGS.get("live", "streaming_url", "http://127.0.0.1:8765"),
            SETTINGS.get("live", "wsl_distro", "Ubuntu"),
        )
        self.recorder = LiveRecorder(self.manager, self.streaming_service, self.events, self.tasks)
        self.floating = FloatingCaptionWindow()
        self.device_indices = []
        self.model_is_ready = False
        self.setWindowTitle("VoxScribe")
        self.setWindowIcon(QIcon(str(ROOT / "assets" / "voxscribe.ico")))
        self.resize(1080, 720)
        self.setMinimumSize(860, 580)
        self._build_ui()
        self._connect_events()
        self._refresh_devices()
        self.file_queue = FileTaskQueue(
            self.manager,
            self.streaming_service,
            self.events,
            self.tasks,
            lambda: self.recorder.stream is not None,
        )
        self.folder_watcher = FolderWatcher(
            self.file_queue,
            self.events,
            lambda: self.recorder.stream is not None,
        )
        self.hotkeys = HotkeyManager(
            self.events.record_hotkey.emit,
            self.events.floating_hotkey.emit,
        )
        self._start_hotkeys()
        threading.Thread(target=self._warm_model, daemon=True).start()

    def _build_ui(self):
        self.setStyleSheet(
            "QMainWindow{background:#10141c;}"
            "QWidget{background:transparent;color:#e8edf5;font-family:'Microsoft YaHei UI';}"
            "QFrame#headerCard{background:#161c25;border:0;border-radius:12px;}"
            "QLabel#appTitle{font-size:22px;font-weight:700;color:#ffffff;}"
            "QLabel#subTitle{font-size:12px;color:#94a0b2;}"
            "QLabel#liveState{font-size:13px;color:#aab7c8;font-weight:600;}"
            "QLabel#sectionTitle{font-size:14px;font-weight:600;color:#dce5f1;}"
            "QLabel#hintText{font-size:12px;color:#99a6b8;padding:1px 0 6px 0;}"
            "QLabel#emptyCaptionTitle{font-size:18px;font-weight:600;color:#dce5f1;}"
            "QLabel#emptyCaptionText{font-size:14px;color:#9eabba;}"
            "QPushButton{background:#232b37;border:1px solid #344052;border-radius:8px;padding:9px 16px;color:#e8edf5;}"
            "QPushButton:hover{background:#2c3645;border-color:#45546a;}"
            "QPushButton#primaryButton{background:#3978df;border-color:#3978df;font-weight:600;}"
            "QPushButton#primaryButton:hover{background:#4a87e8;border-color:#4a87e8;}"
            "QPushButton#stopButton{background:#6f333a;border-color:#824048;}"
            "QProgressBar{height:6px;background:#202733;border:0;border-radius:3px;text-align:center;color:transparent;}"
            "QProgressBar::chunk{background:#4a8cff;border-radius:3px;}"
            "QPushButton:disabled{background:#1b212b;border-color:#29313e;color:#687485;}"
            "QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit{background:#171d26;border:1px solid #344052;border-radius:8px;padding:8px;color:#e8edf5;}"
            "QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus,QLineEdit:focus{border-color:#4a8cff;}"
            "QTextEdit{background:#0b0f15;border:1px solid #293343;border-radius:10px;padding:14px;selection-background-color:#315f9f;}"
            "QTableWidget{background:#0b0f15;border:1px solid #293343;border-radius:10px;gridline-color:#202938;alternate-background-color:#111824;}"
            "QHeaderView::section{background:#1b2330;color:#c9d4e3;border:0;border-right:1px solid #303b4c;padding:8px;}"
            "QTabWidget::pane{border:0;margin-top:8px;}"
            "QTabBar::tab{padding:10px 22px;margin-right:4px;background:transparent;border:0;border-bottom:2px solid transparent;color:#8f9bad;}"
            "QTabBar::tab:hover{color:#d9e2ef;}"
            "QTabBar::tab:selected{background:transparent;border-bottom-color:#4a8cff;color:white;font-weight:600;}"
            "QCheckBox{spacing:7px;}"
        )
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("headerCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        heading = QVBoxLayout()
        title = QLabel("VoxScribe")
        title.setObjectName("appTitle")
        heading.addWidget(title)
        subtitle = QLabel("实时录制与文件转写")
        subtitle.setObjectName("subTitle")
        heading.addWidget(subtitle)
        header_layout.addLayout(heading)
        header_layout.addStretch(1)
        settings_button = QPushButton("设置")
        settings_button.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_button)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._live_tab(), "实时录制")
        self.tabs.addTab(self._offline_tab(), "文件转写")
        self.tabs.addTab(self._history_tab(), "任务历史")
        layout.addWidget(self.tabs)
        self.status_label = QLabel("正在初始化…")
        self.status_label.setStyleSheet("color:#9ca8ba;padding:6px;")
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)

    def _live_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 6, 2, 2)
        layout.setSpacing(9)
        section = QLabel("实时录制")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)
        state_row = QHBoxLayout()
        self.live_model_state = QLabel("● 正在准备实时识别模型…")
        self.live_model_state.setObjectName("liveState")
        state_row.addWidget(self.live_model_state)
        state_row.addStretch(1)
        self.record_time = QLabel("00:00")
        self.record_time.setObjectName("hintText")
        self.record_time.hide()
        state_row.addWidget(self.record_time)
        layout.addLayout(state_row)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("字幕音源"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(420)
        controls.addWidget(self.device_combo, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_devices)
        controls.addWidget(refresh)
        self.start_button = QPushButton("开始录制")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._toggle_live)
        self.start_button.setEnabled(False)
        controls.addWidget(self.start_button)
        layout.addLayout(controls)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("输入音量"))
        self.audio_meter = QProgressBar()
        self.audio_meter.setRange(0, 100)
        self.audio_meter.setTextVisible(False)
        level_row.addWidget(self.audio_meter, 1)
        layout.addLayout(level_row)

        view_controls = QHBoxLayout()
        self.always_top = QCheckBox("窗口置顶")
        self.always_top.toggled.connect(self._toggle_top)
        view_controls.addWidget(self.always_top)
        view_controls.addWidget(QLabel("字号"))
        self.font_size = QSpinBox()
        self.font_size.setRange(18, 72)
        self.font_size.setValue(SETTINGS.get("general", "font_size", 32))
        self.font_size.valueChanged.connect(self._apply_font)
        view_controls.addWidget(self.font_size)
        clear = QPushButton("清空窗口")
        clear.clicked.connect(self._clear_live)
        view_controls.addWidget(clear)
        copy = QPushButton("复制全部")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.live_edit.toPlainText()))
        view_controls.addWidget(copy)
        floating = QPushButton("打开演示悬浮窗")
        floating.clicked.connect(self._show_floating)
        view_controls.addWidget(floating)
        view_controls.addStretch(1)
        layout.addLayout(view_controls)

        self.caption_stack = QStackedWidget()
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.addStretch(1)
        empty_title = QLabel("会议软件音频配置")
        empty_title.setObjectName("emptyCaptionTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_route = QLabel(
            "扬声器：CABLE Input    ·    麦克风：AirPods 麦克风\n"
            "VoxScribe 字幕音源：CABLE Output（只监听）"
        )
        empty_route.setObjectName("emptyCaptionText")
        empty_route.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_route.setWordWrap(True)
        empty_layout.addWidget(empty_route)
        empty_note = QLabel("开始录制后，识别到的字幕会自动显示在这里")
        empty_note.setObjectName("hintText")
        empty_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_note)
        empty_layout.addStretch(1)
        self.caption_stack.addWidget(empty_page)

        self.live_edit = QTextEdit()
        self.live_edit.setReadOnly(True)
        self.live_edit.setPlaceholderText("")
        self.caption_stack.addWidget(self.live_edit)
        self.caption_stack.setCurrentIndex(0)
        layout.addWidget(self.caption_stack, 1)
        self._apply_font()
        return page

    def _offline_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 6, 2, 2)
        layout.setSpacing(9)
        section = QLabel("文件转写")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)
        hint = QLabel("选择音频、视频、文件夹或 URL，确认后开始本地转写")
        hint.setObjectName("hintText")
        layout.addWidget(hint)
        controls = QHBoxLayout()
        self.file_label = QLabel("尚未选择音频或视频文件")
        controls.addWidget(self.file_label, 1)
        choose = QPushButton("添加文件")
        choose.setObjectName("primaryButton")
        choose.clicked.connect(self._choose_file)
        controls.addWidget(choose)
        choose_folder = QPushButton("添加文件夹")
        choose_folder.clicked.connect(self._choose_folder)
        controls.addWidget(choose_folder)
        add_url = QPushButton("添加 URL")
        add_url.clicked.connect(self._add_url)
        controls.addWidget(add_url)
        self.offline_button = QPushButton("开始转写")
        self.offline_button.setObjectName("primaryButton")
        self.offline_button.clicked.connect(self._offline_transcribe)
        self.offline_button.setEnabled(False)
        controls.addWidget(self.offline_button)
        layout.addLayout(controls)
        input_row = QHBoxLayout()
        self.input_path_label = QLabel(f"默认读取：{WATCH_INPUT_DIR}")
        self.input_path_label.setObjectName("hintText")
        input_row.addWidget(self.input_path_label, 1)
        open_input = QPushButton("打开读取文件夹")
        open_input.clicked.connect(lambda: os.startfile(WATCH_INPUT_DIR))
        input_row.addWidget(open_input)
        layout.addLayout(input_row)
        output_row = QHBoxLayout()
        self.output_path_label = QLabel(f"默认输出：{FILE_OUTPUT_DIR}")
        self.output_path_label.setObjectName("hintText")
        output_row.addWidget(self.output_path_label, 1)
        open_output = QPushButton("打开输出文件夹")
        open_output.clicked.connect(lambda: os.startfile(FILE_OUTPUT_DIR))
        output_row.addWidget(open_output)
        layout.addLayout(output_row)
        self.folder_watch_checkbox = QCheckBox("启用文件夹自动转录")
        self.folder_watch_checkbox.setChecked(SETTINGS.get("folder_watch", "enabled", True))
        self.folder_watch_checkbox.toggled.connect(self._toggle_folder_watch)
        layout.addWidget(self.folder_watch_checkbox)
        self.offline_edit = QTextEdit()
        self.offline_edit.setReadOnly(True)
        self.offline_edit.setFont(QFont("Microsoft YaHei UI", 18))
        layout.addWidget(self.offline_edit, 1)
        return page

    def _history_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 6, 2, 2)
        top = QHBoxLayout()
        title = QLabel("自动转录与手动转写历史")
        title.setObjectName("sectionTitle")
        top.addWidget(title)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索文件名")
        self.history_search.setMaximumWidth(220)
        self.history_search.textChanged.connect(self._refresh_history)
        top.addWidget(self.history_search)
        self.history_filter = QComboBox()
        self.history_filter.addItem("全部状态", "")
        self.history_filter.addItem("已完成", "completed")
        self.history_filter.addItem("进行中", "running")
        self.history_filter.addItem("失败", "failed")
        self.history_filter.currentIndexChanged.connect(self._refresh_history)
        top.addWidget(self.history_filter)
        top.addStretch(1)
        retry = QPushButton("重试选中")
        retry.clicked.connect(self._retry_selected_task)
        top.addWidget(retry)
        cancel = QPushButton("取消排队")
        cancel.clicked.connect(self._cancel_selected_task)
        top.addWidget(cancel)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_history)
        top.addWidget(refresh)
        open_outputs = QPushButton("打开输出文件夹")
        open_outputs.clicked.connect(lambda: os.startfile(FILE_OUTPUT_DIR))
        top.addWidget(open_outputs)
        layout.addLayout(top)
        hint = QLabel("双击已完成任务打开可搜索、可编辑、可播放的转录查看器")
        hint.setObjectName("hintText")
        layout.addWidget(hint)
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["时间", "文件", "来源", "模型", "状态", "输出/错误"])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setColumnWidth(0, 150)
        self.history_table.setColumnWidth(1, 210)
        self.history_table.setColumnWidth(2, 90)
        self.history_table.setColumnWidth(3, 160)
        self.history_table.setColumnWidth(4, 80)
        self.history_table.cellDoubleClicked.connect(self._open_history_item)
        layout.addWidget(self.history_table, 1)
        self._refresh_history()
        return page

    def _connect_events(self):
        self.events.status.connect(self.status_label.setText)
        self.events.live_text.connect(self._append_live)
        self.events.live_snapshot.connect(self._replace_live)
        self.events.offline_text.connect(self._show_offline)
        self.events.error.connect(self._show_error)
        self.events.model_ready.connect(self._model_ready)
        self.events.history_changed.connect(self._refresh_history)
        self.events.record_hotkey.connect(self._hotkey_record_toggle)
        self.events.floating_hotkey.connect(self._show_floating)
        self.events.audio_level.connect(self._update_audio_meter)

    def _start_hotkeys(self):
        try:
            self.hotkeys.start(
                SETTINGS.get("hotkeys", "record_toggle", "Ctrl+Shift+R"),
                SETTINGS.get("hotkeys", "floating_window", "Ctrl+Shift+F"),
            )
        except Exception as exc:
            self.events.status.emit(f"全局快捷键启用失败：{exc}")

    def _hotkey_record_toggle(self):
        if self.recorder.stream is None:
            self._start_live()
        else:
            self._stop_live()

    def _refresh_history(self):
        if not hasattr(self, "history_table"):
            return
        all_rows = self.tasks.recent()
        rows = all_rows
        query = self.history_search.text().strip().lower() if hasattr(self, "history_search") else ""
        status_filter = self.history_filter.currentData() if hasattr(self, "history_filter") else ""
        if query:
            rows = [row for row in rows if query in row["source_name"].lower()]
        if status_filter:
            rows = [row for row in rows if row["status"] == status_filter]
        status_labels = {"queued": "排队中", "running": "进行中", "completed": "完成", "failed": "失败", "cancelled": "已取消"}
        trigger_labels = {"manual": "手动", "folder_watch": "监控", "url": "URL", "live": "实时"}
        self.history_table.setRowCount(len(rows))
        active_count = sum(
            row["status"] in {"queued", "running"} and row["trigger"] != "live"
            for row in all_rows
        )
        if hasattr(self, "tabs"):
            self.tabs.setTabText(1, f"文件转写  {active_count}" if active_count else "文件转写")
        for row_index, row in enumerate(rows):
            backend_label = BACKEND_INFO.get(row["backend"], {}).get("label", row["backend"])
            detail = row["output_paths"] if row["status"] == "completed" else row["error"]
            values = [
                row["updated_at"].replace("T", " "),
                row["source_name"],
                trigger_labels.get(row["trigger"], row["trigger"]),
                backend_label,
                status_labels.get(row["status"], row["status"]),
                detail,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value or "")
                item.setToolTip(value or "")
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row["id"])
                if column == 4:
                    colors = {
                        "queued": "#e7c66b",
                        "running": "#67a9ff",
                        "completed": "#6fdaa0",
                        "failed": "#ff7d88",
                        "cancelled": "#9aa6b6",
                    }
                    item.setForeground(QColor(colors.get(row["status"], "#e8edf5")))
                self.history_table.setItem(row_index, column, item)

    def _open_history_item(self, row, _column):
        item = self.history_table.item(row, 0)
        if item is None:
            return
        task = self.tasks.get(item.data(Qt.ItemDataRole.UserRole))
        if task is None:
            return
        if not task["result_json"]:
            QMessageBox.information(self, "VoxScribe", "这条旧任务没有分段时间轴，请重新转写后查看。")
            return
        backend_label = BACKEND_INFO.get(task["backend"], {}).get("label", task["backend"])
        from voxscribe.viewer import TranscriptionViewer

        viewer = TranscriptionViewer(task, self.tasks, backend_label, self)
        viewer.exec()
        self._refresh_history()

    def _model_ready(self):
        live_backend = SETTINGS.get("live", "backend", "faster_whisper")
        streaming_ready = live_backend == "qwen3_asr" and self.streaming_service.ready
        manager_ready = self.manager.backend_key and self.manager.backend_key[0] == live_backend
        if streaming_ready or manager_ready:
            label = BACKEND_INFO.get(live_backend, {}).get("label", live_backend)
            self.model_is_ready = True
            self.live_model_state.setText(f"● {label} 已就绪")
            self.live_model_state.setStyleSheet("color:#6fdaa0;font-weight:600;")
            if self.recorder.stream is None:
                self.start_button.setEnabled(True)
            self.status_label.clear()

    def _open_settings(self):
        dialog = SettingsDialog(SETTINGS, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_settings_to_ui()

    def _apply_settings_to_ui(self):
        refresh_settings_paths()
        self.font_size.setValue(SETTINGS.get("general", "font_size", 32))
        self.folder_watch_checkbox.setChecked(SETTINGS.get("folder_watch", "enabled", True))
        self.folder_watcher.enabled = SETTINGS.get("folder_watch", "enabled", True)
        self.input_path_label.setText(f"默认读取：{WATCH_INPUT_DIR}")
        self.output_path_label.setText(f"默认输出：{FILE_OUTPUT_DIR}")
        self.recorder.chunk_seconds = float(SETTINGS.get("live", "chunk_seconds", 3.5))
        self.recorder.silence_threshold = float(SETTINGS.get("live", "silence_threshold", 0.0025))
        self._refresh_devices()
        self._start_hotkeys()
        live_backend = SETTINGS.get("live", "backend", "faster_whisper")
        if live_backend != "qwen3_asr" and self.streaming_service.ready:
            self.streaming_service.stop_engine()
        default_path = BACKEND_INFO.get(live_backend, {}).get("default_path", "")
        model_path = SETTINGS.get("model", f"{live_backend}_path", default_path)
        streaming_ready = live_backend == "qwen3_asr" and self.streaming_service.ready
        if streaming_ready or self.manager.backend_key == (live_backend, model_path):
            self._model_ready()
        else:
            self.model_is_ready = False
            self.live_model_state.setText("● 正在应用识别模型设置…")
            self.live_model_state.setStyleSheet("color:#f6c85f;font-weight:600;")
            self.start_button.setEnabled(False)
            threading.Thread(target=self._warm_model, daemon=True).start()
        self.status_label.setText("设置已保存并应用")

    def _warm_model(self):
        try:
            live_backend = SETTINGS.get("live", "backend", "faster_whisper")
            if live_backend == "qwen3_asr":
                self.events.status.emit("正在启动 Qwen3-ASR 1.7B 流式服务…")
                self.streaming_service.ensure_started()
                self.events.status.emit("Qwen3-ASR 1.7B 流式服务已就绪 · 全程本地")
                self.events.model_ready.emit()
            else:
                self.manager.ensure_loaded(live_backend)
        except Exception as exc:
            self.events.error.emit(f"模型加载失败：{exc}")

    def _refresh_devices(self):
        self.device_combo.clear()
        self.device_indices.clear()
        devices = sd.query_devices()
        apis = sd.query_hostapis()
        candidates = []
        for index, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue
            api = apis[device["hostapi"]]["name"]
            label = f"{device['name']} · {api}"
            candidates.append((index, label, api, device["name"]))
        preferred = None
        device_keyword = SETTINGS.get("live", "device_keyword", "CABLE Output")
        for index, label, api, name in candidates:
            self.device_indices.append(index)
            self.device_combo.addItem(label)
            if device_keyword.lower() in name.lower() and api == "Windows WASAPI":
                preferred = self.device_combo.count() - 1
        if preferred is None:
            for pos, (_, _, _, name) in enumerate(candidates):
                if device_keyword.lower() in name.lower():
                    preferred = pos
                    break
        if preferred is not None:
            self.device_combo.setCurrentIndex(preferred)

    def _start_live(self):
        pos = self.device_combo.currentIndex()
        if pos < 0:
            self._show_error("没有可用的音频输入设备。")
            return
        try:
            self._clear_live()
            self.recorder.start(self.device_indices[pos])
            self.start_button.setEnabled(False)
            self.start_button.setText("停止并保存")
            self.start_button.setObjectName("stopButton")
            self.start_button.style().unpolish(self.start_button)
            self.start_button.style().polish(self.start_button)
            self.start_button.setEnabled(True)
            self.live_model_state.setText("● 正在录制并识别")
            self.live_model_state.setStyleSheet("color:#67a9ff;font-weight:600;")
            self.record_time.show()
        except Exception as exc:
            self._show_error(f"无法启动音频输入：{exc}")

    def _stop_live(self):
        self.recorder.stop()
        self.start_button.setEnabled(True)
        self.start_button.setText("开始录制")
        self.start_button.setObjectName("primaryButton")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.audio_meter.setValue(0)
        self.record_time.hide()
        self._model_ready()
        self.status_label.setText("实时字幕已停止；已自动保存文本。")

    def _toggle_live(self):
        if self.recorder.stream is None:
            self._start_live()
        else:
            self._stop_live()

    def _update_audio_meter(self, rms):
        level = max(0, min(100, int(rms * 900)))
        self.audio_meter.setValue(level)
        if self.recorder.stream is not None:
            elapsed = max(0, int(time.monotonic() - self.recorder.session_started))
            self.record_time.setText(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

    def _append_live(self, text):
        self.caption_stack.setCurrentIndex(1)
        self.live_edit.append(text)
        self.floating.append_text(text)
        cursor = self.live_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.live_edit.setTextCursor(cursor)

    def _replace_live(self, text):
        self.caption_stack.setCurrentIndex(1)
        self.live_edit.setPlainText(text)
        self.floating.set_text(text)
        cursor = self.live_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.live_edit.setTextCursor(cursor)

    def _clear_live(self):
        self.live_edit.clear()
        self.caption_stack.setCurrentIndex(0)

    def _apply_font(self):
        if hasattr(self, "live_edit"):
            self.live_edit.setFont(QFont("Microsoft YaHei UI", self.font_size.value()))

    def _show_floating(self):
        self.floating.set_text(self.live_edit.toPlainText())
        self.floating.show_for(self)

    def _toggle_top(self, enabled):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.show()

    def _choose_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择一个或多个音频/视频",
            str(WATCH_INPUT_DIR),
            "媒体文件 (*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mkv *.mov *.webm);;所有文件 (*.*)",
        )
        if paths:
            self.selected_files = paths
            self.file_label.setText(paths[0] if len(paths) == 1 else f"已选择 {len(paths)} 个媒体文件")
            self.offline_button.setEnabled(True)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含媒体的文件夹", str(WATCH_INPUT_DIR))
        if not folder:
            return
        paths = [
            str(path)
            for path in Path(folder).rglob("*")
            if path.is_file() and path.suffix.lower() in FolderWatcher.MEDIA_SUFFIXES
        ]
        if not paths:
            QMessageBox.information(self, "VoxScribe", "所选文件夹中没有支持的媒体文件。")
            return
        self.selected_files = paths
        self.file_label.setText(f"已从文件夹选择 {len(paths)} 个媒体文件")
        self.offline_button.setEnabled(True)

    def _add_url(self):
        url, accepted = QInputDialog.getText(self, "添加 URL", "YouTube 或媒体页面地址：")
        if not accepted or not url.strip():
            return
        self.status_label.setText("正在下载 URL 媒体…")

        def download():
            try:
                import yt_dlp

                before = {path.resolve() for path in WATCH_INPUT_DIR.iterdir() if path.is_file()}
                options = {
                    "format": "bestaudio/best",
                    "outtmpl": str(WATCH_INPUT_DIR / "%(title).150s [%(id)s].%(ext)s"),
                    "noplaylist": False,
                    "quiet": True,
                    "no_warnings": True,
                }
                with yt_dlp.YoutubeDL(options) as downloader:
                    downloader.extract_info(url.strip(), download=True)
                after = {path.resolve() for path in WATCH_INPUT_DIR.iterdir() if path.is_file()}
                downloaded = sorted(after - before)
                if not downloaded:
                    raise RuntimeError("下载完成但没有找到新媒体文件")
                for path in downloaded:
                    self.file_queue.enqueue(path, "url")
                self.events.status.emit(f"URL 下载完成，已加入 {len(downloaded)} 个任务")
            except Exception as exc:
                self.events.error.emit(f"URL 下载失败：{exc}")

        threading.Thread(target=download, daemon=True).start()

    def _offline_transcribe(self):
        paths = getattr(self, "selected_files", [])
        if not paths:
            return
        added = sum(bool(self.file_queue.enqueue(path, "manual")) for path in paths)
        if added:
            self.selected_files = []
            self.file_label.setText(f"已加入 {added} 个任务")
            self.offline_button.setEnabled(False)
            self.tabs.setCurrentIndex(2)

    def _selected_task(self):
        row = self.history_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "VoxScribe", "请先在任务历史中选择一行。")
            return None
        item = self.history_table.item(row, 0)
        return self.tasks.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _retry_selected_task(self):
        task = self._selected_task()
        if task is None:
            return
        if not Path(task["source_path"]).exists():
            QMessageBox.warning(self, "VoxScribe", "原始媒体文件不存在，无法重试。")
            return
        self.file_queue.enqueue(task["source_path"], task["trigger"])

    def _cancel_selected_task(self):
        task = self._selected_task()
        if task is None:
            return
        if task["status"] != "queued":
            QMessageBox.information(self, "VoxScribe", "只有尚未开始的排队任务可以安全取消。")
            return
        self.file_queue.cancel(task["id"])

    def _show_offline(self, text):
        self.offline_edit.setPlainText(text)

    def _toggle_folder_watch(self, enabled):
        if hasattr(self, "folder_watcher"):
            self.folder_watcher.enabled = enabled
        SETTINGS.update_section("folder_watch", {"enabled": enabled})
        state = "开启" if enabled else "关闭"
        self.status_label.setText(f"文件夹监控已{state}")

    def _show_error(self, message):
        self.status_label.setText(message)
        QMessageBox.critical(self, "VoxScribe", message)

    def closeEvent(self, event):
        self.recorder.stop()
        self.folder_watcher.stop()
        self.file_queue.stop()
        self.hotkeys.stop()
        self.floating.close()
        self.streaming_service.stop()
        event.accept()


def main():
    def exception_hook(exc_type, exc_value, exc_traceback):
        LOGGER.critical("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = exception_hook
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VoxScribe.Desktop.1")
        except Exception:
            LOGGER.exception("无法设置 Windows 应用标识")
    app = QApplication(sys.argv)
    app.setApplicationName("VoxScribe")
    app.setWindowIcon(QIcon(str(ROOT / "assets" / "voxscribe.ico")))
    instance_lock = QLockFile(str(ROOT / "cache" / "voxscribe.lock"))
    instance_lock.setStaleLockTime(10000)
    if not instance_lock.tryLock(0):
        QMessageBox.information(None, "VoxScribe", "VoxScribe 已经在运行。")
        return
    app.instance_lock = instance_lock
    window = MainWindow()
    try:
        configure_windows_taskbar(window)
    except Exception:
        LOGGER.exception("无法配置 Windows 任务栏入口")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
