import sqlite3
from datetime import datetime
from pathlib import Path


class TaskStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_paths TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
            migrations = {
                "result_json": "TEXT DEFAULT ''",
                "progress": "REAL DEFAULT 0",
                "notes": "TEXT DEFAULT ''",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {name} {definition}")

    def start(self, source_path, trigger, backend):
        task_id = self.enqueue(source_path, trigger, backend)
        self.mark_running(task_id)
        return task_id

    def enqueue(self, source_path, trigger, backend):
        now = datetime.now().isoformat(timespec="seconds")
        source = Path(source_path)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks
                    (source_path, source_name, trigger, backend, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', ?, ?)
                """,
                (str(source), source.name, trigger, backend, now, now),
            )
            return cursor.lastrowid

    def mark_running(self, task_id):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='running', progress=1, notes='准备处理', error='', updated_at=? WHERE id=?",
                (now, int(task_id)),
            )

    def update_progress(self, task_id, progress, notes=""):
        now = datetime.now().isoformat(timespec="seconds")
        progress = max(0.0, min(100.0, float(progress)))
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET progress=?, notes=?, updated_at=? WHERE id=? AND status IN ('queued', 'running')",
                (progress, str(notes), now, int(task_id)),
            )

    def complete(self, task_id, output_paths, result=None):
        now = datetime.now().isoformat(timespec="seconds")
        outputs = "\n".join(str(path) for path in output_paths)
        with self._connect() as connection:
            result_json = result.to_json() if result is not None else ""
            connection.execute(
                "UPDATE tasks SET status='completed', progress=100, notes='完成', output_paths=?, result_json=?, updated_at=? WHERE id=?",
                (outputs, result_json, now, task_id),
            )

    def fail(self, task_id, error):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='failed', error=?, updated_at=? WHERE id=?",
                (str(error), now, task_id),
            )

    def cancel(self, task_id):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status='cancelled', error='用户取消', updated_at=? WHERE id=? AND status='queued'",
                (now, int(task_id)),
            )

    def get(self, task_id):
        with self._connect() as connection:
            return connection.execute("SELECT * FROM tasks WHERE id=?", (int(task_id),)).fetchone()

    def update_result(self, task_id, result):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET result_json=?, updated_at=? WHERE id=?",
                (result.to_json(), now, int(task_id)),
            )

    def update_notes(self, task_id, notes):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET notes=?, updated_at=? WHERE id=?",
                (notes, now, int(task_id)),
            )

    def recover_interrupted(self):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            return connection.execute(
                "UPDATE tasks SET status='failed', error='程序上次退出时任务尚未完成，可从历史记录重试', updated_at=? WHERE status IN ('queued', 'running')",
                (now,),
            ).rowcount

    def recent(self, limit=200):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM tasks ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
