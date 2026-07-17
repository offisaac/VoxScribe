# VoxScribe

**Private, local-first speech transcription for meetings, presentations, and media files.**

VoxScribe is a Windows desktop app for real-time captions and offline media transcription. It is designed for workflows that need local processing, a floating presenter window, and OBS-compatible live captions without sending audio to a cloud API.

## Features

- Real-time recording from Windows audio devices, including VB-CABLE.
- Quick audio-source presets for meeting audio (`CABLE Output`) and currently playing system audio (automatic Windows loopback detection).
- Local Qwen3-ASR streaming captions through a WSL 2 service, with partial results that update as speech arrives.
- Fun-ASR-Nano as an optional low-latency local live-caption backend.
- Floating, always-on-top caption window for presentations.
- Offline audio/video transcription and automatic folder monitoring.
- Faster Whisper, Qwen3-ASR, Fun-ASR-Nano, and an external CLI adapter for local backends.
- TXT, SRT, VTT, and JSON transcript exports.
- Task history, retries, cancellation, transcript viewer, search, editing, and media playback.
- Optional noise reduction, vocal isolation, speaker clustering, and global hotkeys.
- Selectable local Demucs vocal-separation models, from fast MDX variants to higher-quality HTDemucs variants.
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
sudo systemctl disable voxscribe-qwen-stream
```

In VoxScribe settings, select **Qwen3-ASR 1.7B** as the live model. The app checks the local service and starts it through the configured WSL distribution when necessary. It keeps the model loaded while VoxScribe is open and shuts WSL down when VoxScribe exits. `streaming_url` and `wsl_distro` can be changed in `config/settings.json`.

The bundled service is tuned for one live transcription session: an isolated multiprocess vLLM engine, eager CUDA execution, no CPU swap reservation, an 8K context, a 55% GPU memory target, and no multimodal processor cache. VoxScribe automatically rotates the server-side streaming session every 20 seconds of audio so inference cost stays bounded during long interviews while committed captions remain in the transcript. The processor cache is intentionally disabled because live audio chunks are unique and do not benefit from reuse.

Live recording exposes two independent Qwen modes. **True streaming** uses the WSL service for low-latency revisions. **Standard segmented** loads the same Qwen3-ASR 1.7B model directly on Windows and does not depend on the WSL streaming session. Standard Qwen is the safe default and automatic fallback when streaming startup or a live streaming request fails. Audio already entering the recorder stays queued during fallback, the poisoned WSL service is stopped, and recording continues with standard Qwen. Switching modes unloads the previous runtime before loading the new one so both model copies are not kept in RAM together.

Recognition is scoped to Simplified Chinese and English. The language selector supports automatic Chinese/English detection, forced Simplified Chinese, or forced English. Translation is disabled, and all recognized Chinese text is normalized to Simplified Chinese before display and export.

### Reliability during long sessions

- The example configuration starts in **Standard segmented** mode so the app remains usable without a running WSL service.
- If a true-streaming request fails, VoxScribe preserves queued audio, stops the failed WSL runtime, loads the configured standard backend, and continues the active recording.
- Runtime recognition errors trigger a safe stop and transcript finalization instead of leaving the audio device or export file in an indeterminate state.
- Error notifications are non-modal and coalesced, so an unavailable service does not lock the recording interface behind repeated dialog windows.

### Fun-ASR-Nano fast mode

Set the live backend to `fun_asr_nano` and place its local model under the configured `fun_asr_nano_path`. VoxScribe uses short audio chunks in this mode to prioritize responsiveness. It does not require WSL.

## Audio source shortcuts

The **Quick source** menu on the live-recording screen reduces device-selection mistakes:

- **Meeting · CABLE Output** selects the `CABLE Output` device through the Windows WASAPI host API. Use this when a meeting application or browser is deliberately routed through VB-CABLE.
- **Testing · current system audio** detects the active physical Windows playback device and captures it through a local loopback. This is useful for transcribing a video already playing on the computer without changing that application's output device.

System loopback capture is local and read-only. It captures playback audio only; a physical microphone must still be routed separately if both sides of a conversation need to be transcribed.

## Audio processing

VoxScribe offers selectable local Demucs models for vocal isolation: balanced **HTDemucs**, higher-quality **HTDemucs FT**, six-source **HTDemucs 6S**, and faster **MDX** variants. The selected model is downloaded by Demucs on first use if it is not already cached. For normal meetings, start with noise reduction or HTDemucs; reserve larger separation models for offline media where processing time is less important.

## Offline task progress and safety

File and folder-monitor transcription tasks now report their current stage in **Task History**: preparing, audio preprocessing, recognition, speaker identification, export, and completion. Running tasks show an estimated stage percentage and the current status note instead of an empty result field.

VoxScribe verifies that each requested transcript export exists before marking a task as complete. If an output file cannot be written, the task remains failed rather than being presented as a successful transcription. Folder monitoring and file processing pause whenever either a normal input stream or system-audio loopback recording is active, preventing the two workloads from competing for the active model.

## Privacy

VoxScribe is designed for local operation. Model weights, recordings, transcripts, task history, logs, and user settings are intentionally excluded from version control.

## License

Released under the [MIT License](LICENSE).
