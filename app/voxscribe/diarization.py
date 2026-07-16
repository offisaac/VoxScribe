import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from voxscribe.transcription import normalize_audio


def assign_speakers(result, audio, requested_speakers=0):
    eligible = [segment for segment in result.segments if segment.end - segment.start >= 0.5]
    if len(eligible) < 2:
        return result
    samples, sample_rate = normalize_audio(audio)
    features = []
    valid_segments = []
    for segment in eligible:
        start = max(0, int(segment.start * sample_rate))
        end = min(len(samples), int(segment.end * sample_rate))
        clip = samples[start:end]
        if len(clip) < sample_rate // 2:
            continue
        features.append(_speaker_features(clip, sample_rate))
        valid_segments.append(segment)
    if len(features) < 2:
        return result
    values = StandardScaler().fit_transform(np.vstack(features))
    speakers = _choose_speaker_count(values, requested_speakers)
    if speakers <= 1:
        labels = np.zeros(len(valid_segments), dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=speakers, linkage="ward").fit_predict(values)
    mapping = {}
    for segment, label in zip(valid_segments, labels):
        if label not in mapping:
            mapping[label] = f"说话人 {len(mapping) + 1}"
        segment.speaker = mapping[label]
    return result


def _speaker_features(audio, sample_rate):
    import librosa

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
        ]
    )


def _choose_speaker_count(values, requested):
    if requested and requested > 0:
        return min(int(requested), len(values))
    if len(values) < 4:
        return 1
    best_count = 1
    best_score = 0.12
    for count in range(2, min(4, len(values) - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(values)
        score = silhouette_score(values, labels)
        if score > best_score:
            best_count = count
            best_score = score
    return best_count

