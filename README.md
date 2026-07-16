# VoxScribe

**Private, local-first speech transcription for meetings, presentations, and media files.**

VoxScribe is a Windows desktop app for real-time captions and offline media transcription. It is designed for workflows that need local processing, a floating presenter window, and OBS-compatible live captions without sending audio to a cloud API.

## Features

- Real-time recording from Windows audio devices, including VB-CABLE.
- Floating, always-on-top caption window for presentations.
- Offline audio/video transcription and automatic folder monitoring.
- Faster Whisper and Qwen3-ASR local backends, plus an external CLI adapter.
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

## Privacy

VoxScribe is designed for local operation. Model weights, recordings, transcripts, task history, logs, and user settings are intentionally excluded from version control.

## License

Released under the [MIT License](LICENSE).
