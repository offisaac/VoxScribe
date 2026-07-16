import json
import os
import subprocess
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from voxscribe.transcription import Segment, TranscriptionResult, normalize_audio


class QwenStreamingSession:
    def __init__(self, base_url, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = None

    def _post(self, path, data=b"", content_type="application/json"):
        request = Request(
            self.base_url + path,
            data=data,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def start(self):
        self.session_id = self._post("/api/start")["session_id"]
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
            return self._post(f"/api/finish?{query}")
        finally:
            self.session_id = None


class QwenStreamingService:
    def __init__(self, base_url="http://127.0.0.1:8765", distro="Ubuntu"):
        self.base_url = base_url.rstrip("/")
        self.distro = distro
        self.keepalive = None
        self.ready = False
        self.lock = threading.Lock()

    @staticmethod
    def _creation_flags():
        return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def _healthy(self):
        try:
            with urlopen(self.base_url + "/", timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def ensure_started(self, timeout=240):
        with self.lock:
            if self._healthy():
                self.ready = True
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
                if self._healthy():
                    self.ready = True
                    return
                time.sleep(2)
            raise RuntimeError("Qwen 流式服务启动超时")

    def create_session(self):
        self.ensure_started()
        return QwenStreamingSession(self.base_url).start()

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
        with self.lock:
            self.ready = False
            if self.keepalive is None or self.keepalive.poll() is not None:
                return
            subprocess.run(
                ["wsl.exe", "-d", self.distro, "-u", "root", "--", "systemctl", "stop", "voxscribe-qwen-stream"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=self._creation_flags(),
                timeout=30,
            )

    def stop(self):
        try:
            try:
                self.stop_engine()
            except (OSError, subprocess.SubprocessError):
                pass
        finally:
            if self.keepalive is not None and self.keepalive.poll() is None:
                self.keepalive.terminate()
                try:
                    self.keepalive.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.keepalive.kill()
            self.keepalive = None
            self.ready = False
            try:
                subprocess.run(
                    ["wsl.exe", "--terminate", self.distro],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=self._creation_flags(),
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                pass
