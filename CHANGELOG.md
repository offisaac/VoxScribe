# Changelog

All notable changes to VoxScribe are documented in this file.

## 0.5.0 — 2026-07-17

### Added

- Quick live-audio source menu with an exact WASAPI `CABLE Output` preset for meetings.
- Local Windows system-audio loopback capture that automatically selects an active playback endpoint for media testing and transcription.
- Selectable Demucs vocal-isolation models: HTDemucs, HTDemucs FT, HTDemucs 6S, MDX, MDX Extra, and MDX Extra Q.

### Changed

- Live input initialization now retries compatible sample rates and prefers 48 kHz for Windows WDM-KS devices.
- Settings controls use fixed model choices rather than free-form Demucs model names.
- Refined dark-theme focus, menu, and combo-box styling for clearer keyboard and mouse interaction.

### Fixed

- Meeting preset selection now matches both the device name and Windows WASAPI host API, preventing accidental selection of an identically named MME endpoint.

## 0.4.0 — 2026-07-17

### Added

- Session rotation for Qwen true streaming: the local service commits and renews its server-side session every 20 seconds of audio while preserving the accumulated transcript.
- Automated regression coverage for streaming-session rotation, standard-Qwen fallback, non-modal error notification reuse, Simplified Chinese normalization, and WSL shutdown behavior.

### Changed

- Standard segmented Qwen3-ASR is now the safe default in the shipped example configuration.
- The WSL streaming service now uses isolated vLLM multiprocess execution and remains on-demand; VoxScribe starts it when needed and releases the WSL runtime when it exits.
- The local streaming client retains committed caption text across session rotation, preventing long recordings from losing earlier recognized content.
- Documentation now distinguishes true streaming from standard segmented recognition and documents their memory and recovery behavior.

### Fixed

- A failed true-streaming request now falls back to the configured standard local backend without stopping the recording or dropping queued audio.
- Runtime recognition failures now stop and finalize the active transcript safely.
- Repeated runtime errors no longer create blocking dialog stacks; VoxScribe reuses a single non-modal error notification.

## 0.3.0 — 2026-07-16

### Added

- Local Qwen3-ASR streaming client, WSL systemd service definition, and configurable streaming parameters.
- Fun-ASR-Nano low-latency local backend.
- Simplified Chinese output normalization and local-only Chinese/English recognition settings.

### Changed

- Separated live streaming recognition from offline high-accuracy transcription.
- Added local model switching, WSL lifecycle handling, and streaming service health checks.
