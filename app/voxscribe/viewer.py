from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from voxscribe.exports import write_exports
from voxscribe.transcription import Segment, TranscriptionResult


def _clock(milliseconds):
    seconds = max(0, int(milliseconds / 1000))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TranscriptionViewer(QDialog):
    def __init__(self, task, task_store, backend_label, parent=None):
        super().__init__(parent)
        self.task = task
        self.task_store = task_store
        self.backend_label = backend_label
        self.result = TranscriptionResult.from_json(task["result_json"])
        self.source = Path(task["source_path"])
        self.setWindowTitle(f"转录查看器 · {task['source_name']}")
        self.resize(1080, 720)
        self._build_ui()
        self._load_segments()
        self._setup_player()
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search.setFocus)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        QShortcut(QKeySequence("Space"), self, activated=self._toggle_play)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        toolbar = QHBoxLayout()
        title = QLabel(self.task["source_name"])
        title.setObjectName("viewerTitle")
        toolbar.addWidget(title)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索转录内容 · Ctrl+F")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        toolbar.addWidget(self.search, 1)
        export = QPushButton("重新导出")
        export.clicked.connect(self._export)
        toolbar.addWidget(export)
        save = QPushButton("保存修改")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        toolbar.addWidget(save)
        layout.addLayout(toolbar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["开始", "结束", "说话人", "转录文本"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 110)
        self.table.cellClicked.connect(self._seek_row)
        layout.addWidget(self.table, 1)

        self.notes = QLineEdit(self.task["notes"] or "")
        self.notes.setPlaceholderText("任务备注（会保存到历史记录）")
        layout.addWidget(self.notes)

        player_row = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self._toggle_play)
        player_row.addWidget(self.play_button)
        self.position_label = QLabel("00:00:00 / 00:00:00")
        player_row.addWidget(self.position_label)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.sliderMoved.connect(self._seek)
        player_row.addWidget(self.position, 1)
        player_row.addWidget(QLabel("速度"))
        self.speed = QComboBox()
        for value in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        player_row.addWidget(self.speed)
        self.follow = QCheckBox("跟随播放")
        self.follow.setChecked(True)
        player_row.addWidget(self.follow)
        layout.addLayout(player_row)

    def _load_segments(self):
        self.table.setRowCount(len(self.result.segments))
        for row, segment in enumerate(self.result.segments):
            start = QTableWidgetItem(f"{segment.start:.3f}")
            end = QTableWidgetItem(f"{segment.end:.3f}")
            start.setFlags(start.flags() & ~Qt.ItemFlag.ItemIsEditable)
            end.setFlags(end.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, start)
            self.table.setItem(row, 1, end)
            self.table.setItem(row, 2, QTableWidgetItem(segment.speaker))
            self.table.setItem(row, 3, QTableWidgetItem(segment.text))

    def _setup_player(self):
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        if self.source.exists() and self.source.suffix.lower() not in {".txt", ".srt", ".vtt", ".json"}:
            self.player.setSource(QUrl.fromLocalFile(str(self.source)))
        else:
            self.play_button.setEnabled(False)
            self.play_button.setToolTip("原始媒体文件已移动或删除")
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(lambda duration: self.position.setMaximum(duration))
        self.player.playbackStateChanged.connect(
            lambda state: self.play_button.setText("暂停" if state == QMediaPlayer.PlaybackState.PlayingState else "播放")
        )

    def _position_changed(self, position):
        if not self.position.isSliderDown():
            self.position.setValue(position)
        self.position_label.setText(f"{_clock(position)} / {_clock(self.player.duration())}")
        if self.follow.isChecked():
            seconds = position / 1000
            for row, segment in enumerate(self.result.segments):
                if segment.start <= seconds <= segment.end:
                    self.table.selectRow(row)
                    self.table.scrollToItem(self.table.item(row, 3))
                    break

    def _seek(self, position):
        self.player.setPosition(position)

    def _seek_row(self, row, _column):
        self.player.setPosition(int(float(self.table.item(row, 0).text()) * 1000))

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _filter(self, value):
        query = value.strip().lower()
        for row in range(self.table.rowCount()):
            text = self.table.item(row, 3).text().lower()
            speaker = self.table.item(row, 2).text().lower()
            self.table.setRowHidden(row, bool(query and query not in text and query not in speaker))

    def _collect_result(self):
        segments = []
        for row in range(self.table.rowCount()):
            original = self.result.segments[row]
            segments.append(
                Segment(
                    start=float(self.table.item(row, 0).text()),
                    end=float(self.table.item(row, 1).text()),
                    speaker=self.table.item(row, 2).text().strip(),
                    text=self.table.item(row, 3).text().strip(),
                    words=original.words,
                )
            )
        return TranscriptionResult(segments, self.result.language, self.result.duration)

    def _save(self):
        self.result = self._collect_result()
        self.task_store.update_result(self.task["id"], self.result)
        self.task_store.update_notes(self.task["id"], self.notes.text().strip())
        QMessageBox.information(self, "VoxScribe", "转录修改和备注已保存。")

    def _export(self):
        directory = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(self.source.parent))
        if not directory:
            return
        self.result = self._collect_result()
        outputs = write_exports(
            self.result,
            self.source,
            directory,
            ["txt", "srt", "vtt", "json"],
            self.backend_label,
        )
        QMessageBox.information(self, "VoxScribe", f"已导出 {len(outputs)} 个文件到：\n{directory}")
