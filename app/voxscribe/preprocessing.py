import threading

import numpy as np
from scipy.signal import resample_poly

from voxscribe.transcription import load_media_audio


_demucs_model = None
_demucs_lock = threading.Lock()


def preprocess_audio(source, mode="off", demucs_model="htdemucs"):
    if mode == "off":
        return str(source)
    if mode == "noise_reduce":
        import noisereduce as nr

        audio, sample_rate = load_media_audio(source, 16000)
        enhanced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.8,
            n_jobs=1,
        )
        return np.asarray(enhanced, dtype=np.float32), sample_rate
    if mode == "vocals":
        return _extract_vocals(source, demucs_model)
    raise RuntimeError(f"未知音频预处理模式：{mode}")


def _extract_vocals(source, model_name):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    global _demucs_model
    with _demucs_lock:
        if _demucs_model is None:
            _demucs_model = get_model(model_name)
            _demucs_model.eval()
        model = _demucs_model
    audio, sample_rate = load_media_audio(source, model.samplerate)
    stereo = np.stack([audio, audio])
    waveform = torch.from_numpy(stereo).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with torch.inference_mode():
        separated = apply_model(
            model,
            waveform,
            device=device,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    vocals_index = model.sources.index("vocals")
    vocals = separated[vocals_index].mean(dim=0).detach().cpu().numpy()
    if model.samplerate != 16000:
        vocals = resample_poly(vocals, 16000, model.samplerate)
    return np.asarray(vocals, dtype=np.float32), 16000

