# VoxScribe

**Private, local-first speech transcription for meetings, presentations, and media files.**

VoxScribe is a Windows desktop app for real-time captions and offline media transcription. It is designed for workflows that need local processing, a floating presenter window, and OBS-compatible live captions without sending audio to a cloud API.

## Features

- Real-time recording from Windows audio devices, including VB-CABLE.
- Local Qwen3-ASR streaming captions through a WSL 2 service, with partial results that update as speech arrives.
- Fun-ASR-Nano as an optional low-latency local live-caption backend.
- Floating, always-on-top caption window for presentations.
- Offline audio/video transcription and automatic folder monitoring.
- Faster Whisper, Qwen3-ASR, Fun-ASR-Nano, and an external CLI adapter for local backends.
- TXT, SRT, VTT, and JSON transcript exports.
- Task history, retries, cancellation, transcript viewer, search, editing, and media playback.
- Optional noise reduction, vocal isolation, speaker clustering, and global hotkeys.
- OBS-ready text output that always contains the newest live caption.

## Built with Codex & GPT-5.6

VoxScribe was developed with Codex and GPT-5.6 as collaborative engineering tools. They were used to shape the local-first product workflow, implement the Windows desktop interface and model-adapter architecture, refine audio-routing and presenter-caption interactions, create the test suite, and document a reproducible open-source setup. The final application runs transcription locally with user-controlled models and audio devices; Codex and GPT-5.6 supported the development process rather than acting as a runtime dependency.

## Architecture

```text
Meeting audio -> VB-CABLE -> VoxScribe -> live captions / OBS text source
Microphone -> meeting application -> remote participant
```

VoxScribe reads audio for transcription only. Keep the meeting application's microphone set to a real microphone rather than `CABLE Output` to avoid echoing remote participants back into the call.

## Requirements

- Windows 10 or 11
- Python 3.12
- NVIDIA GPU with CUDA recommended for local models
- FFmpeg available on `PATH` for broad media-format support
- Local model weights for Qwen3-ASR and/or Faster Whisper (not included in this repository)
- WSL 2 with an Ubuntu distribution and systemd, only when using Qwen3-ASR streaming mode

## Setup

1. Create and activate a virtual environment.
2. Install the Python dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `config/settings.example.json` to `config/settings.json` and update model and output paths.
4. Download the local model weights you intend to use and set their paths in `config/settings.json`.
5. Start the app:

   ```powershell
   python app/voxscribe.py
   ```

Run the test suite with:

```powershell
python -m pytest tests
```

## Live transcription modes

VoxScribe offers two local live-caption approaches:

| Mode | Backend | Best for |
| --- | --- | --- |
| Standard live captions | Faster Whisper or Fun-ASR-Nano | Low-latency chunked captions directly on Windows |
| Streaming live captions | Qwen3-ASR 1.7B | Continuously updated captions with local WSL 2 inference |

### Qwen3-ASR streaming mode

Qwen streaming is fully local. The Windows app sends 16 kHz audio chunks only to a service on `http://127.0.0.1:8765`; no cloud endpoint or API key is used.

The checked-in streaming components are:

- `app/voxscribe/streaming.py` — Windows client and session lifecycle.
- `scripts/qwen_stream_server.py` — local Qwen streaming HTTP service.
- `scripts/voxscribe-qwen-stream.service` — systemd service definition for WSL 2.

The bundled service definition expects a prepared WSL environment with the service script at `/opt/voxscribe/services/qwen_stream_server.py`, the Qwen model at `/opt/voxscribe/models/Qwen3-ASR-1.7B`, and a Python runtime at `/opt/voxscribe/runtime/bin/python`. Update those paths if your WSL installation differs, then install and start the service:

```bash
sudo cp scripts/voxscribe-qwen-stream.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voxscribe-qwen-stream
```

In VoxScribe settings, select **Qwen3-ASR 1.7B** as the live model. The app checks the local service and starts it through the configured WSL distribution when necessary. `streaming_url` and `wsl_distro` can be changed in `config/settings.json`.

### Fun-ASR-Nano fast mode

Set the live backend to `fun_asr_nano` and place its local model under the configured `fun_asr_nano_path`. VoxScribe uses short audio chunks in this mode to prioritize responsiveness. It does not require WSL.

## Privacy

VoxScribe is designed for local operation. Model weights, recordings, transcripts, task history, logs, and user settings are intentionally excluded from version control.

## License

Released under the [MIT License](LICENSE).
