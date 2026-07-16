import json
import os
import subprocess
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from voxscribe.transcription import Segment, TranscriptionResult, normalize_audio, normalize_recognition_text


class QwenStreamingSession:
    def __init__(
        self,
        base_url,
        timeout=15,
        chunk_seconds=0.8,
        unfixed_chunk_num=4,
        unfixed_token_num=5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chunk_seconds = float(chunk_seconds)
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)
        self.session_id = None

    def _post(self, path, data=b"", content_type="application/json", timeout=None):
        request = Request(
            self.base_url + path,
            data=data,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urlopen(request, timeout=timeout or self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            if isinstance(result.get("text"), str):
                result["text"] = normalize_recognition_text(result["text"])
            return result

    def start(self):
        payload = json.dumps(
            {
                "chunk_size_sec": self.chunk_seconds,
                "unfixed_chunk_num": self.unfixed_chunk_num,
                "unfixed_token_num": self.unfixed_token_num,
            }
        ).encode("utf-8")
        self.session_id = self._post("/api/start", payload)["session_id"]
        return self

    def push(self, samples):
        if not self.session_id:
            raise RuntimeError("流式识别会话尚未开始")
        audio = np.ascontiguousarray(samples, dtype="<f4")
        query = urlencode({"session_id": self.session_id})
        return self._post(
            f"/api/chunk?{query}",
            audio.tobytes(),
            "application/octet-stream",
        )

    def finish(self):
        if not self.session_id:
            return {"language": "", "text": ""}
        query = urlencode({"session_id": self.session_id})
        try:
            return self._post(f"/api/finish?{query}", timeout=3)
        finally:
            self.session_id = None


class QwenStreamingService:
    def __init__(
        self,
        base_url="http://127.0.0.1:8765",
        distro="Ubuntu",
        chunk_seconds=0.8,
        unfixed_chunk_num=4,
        unfixed_token_num=5,
    ):
        self.base_url = base_url.rstrip("/")
        self.distro = distro
        self.chunk_seconds = float(chunk_seconds)
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)
        self.keepalive = None
        self.ready = False
        self.cache_trimmed = False
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()

    @staticmethod
    def _creation_flags():
        return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def _healthy(self):
        try:
            with urlopen(self.base_url + "/", timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def _trim_wsl_cache(self):
        if self.cache_trimmed:
            return
        try:
            subprocess.run(
                [
                    "wsl.exe",
                    "-d",
                    self.distro,
                    "-u",
                    "root",
                    "--",
                    "sh",
                    "-lc",
                    "sync; echo 3 > /proc/sys/vm/drop_caches",
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=self._creation_flags(),
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self.cache_trimmed = True

    def ensure_started(self, timeout=240):
        with self.lock:
            self.stop_requested.clear()
            if self._healthy():
                self.ready = True
                self._trim_wsl_cache()
                return
            if self.keepalive is None or self.keepalive.poll() is not None:
                self.keepalive = subprocess.Popen(
                    ["wsl.exe", "-d", self.distro, "-u", "voxscribe", "--", "sleep", "infinity"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=self._creation_flags(),
                )
            subprocess.run(
                ["wsl.exe", "-d", self.distro, "-u", "root", "--", "systemctl", "start", "voxscribe-qwen-stream"],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=self._creation_flags(),
                timeout=30,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.stop_requested.is_set():
                    raise RuntimeError("Qwen 流式服务启动已取消")
                if self._healthy():
                    self.ready = True
                    self._trim_wsl_cache()
                    return
                time.sleep(2)
            raise RuntimeError("Qwen 流式服务启动超时")

    def create_session(self):
        last_error = None
        for attempt in range(3):
            try:
                self.ensure_started()
                return QwenStreamingSession(
                    self.base_url,
                    chunk_seconds=self.chunk_seconds,
                    unfixed_chunk_num=self.unfixed_chunk_num,
                    unfixed_token_num=self.unfixed_token_num,
                ).start()
            except Exception as exc:
                last_error = exc
                self.ready = False
                self.cache_trimmed = False
                if self.stop_requested.is_set():
                    raise RuntimeError("Qwen 流式服务启动已取消") from exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"无法建立 Qwen 流式会话：{last_error}") from last_error

    def transcribe_result(self, audio):
        samples, sample_rate = normalize_audio(audio)
        session = self.create_session()
        result = {"language": "", "text": ""}
        try:
            for start in range(0, len(samples), sample_rate):
                result = session.push(samples[start : start + sample_rate])
            result = session.finish()
        finally:
            if session.session_id:
                session.finish()
        text = (result.get("text") or "").strip()
        duration = len(samples) / sample_rate
        segments = [Segment(0.0, duration, text)] if text else []
        return TranscriptionResult(segments, result.get("language") or "", duration)

    def stop_engine(self):
        self.stop()

    def stop(self):
        self.stop_requested.set()
        self.ready = False
        self.cache_trimmed = False
        try:
            subprocess.run(
                ["wsl.exe", "--shutdown"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=self._creation_flags(),
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            if self.keepalive is not None and self.keepalive.poll() is None:
                self.keepalive.terminate()
                try:
                    self.keepalive.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.keepalive.kill()
            self.keepalive = None
