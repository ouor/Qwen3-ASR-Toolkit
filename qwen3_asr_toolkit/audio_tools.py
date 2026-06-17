import io
import sys
import srt
import subprocess
import numpy as np
import soundfile as sf

from datetime import timedelta


WAV_SAMPLE_RATE = 16000


def load_audio(file_path: str, verbose: bool = True) -> np.ndarray:
    """Load any local/remote media into a 16 kHz mono float32 waveform.

    Tries librosa first (fast for common audio formats); falls back to FFmpeg,
    which also handles video containers (mp4/mkv/mov, ...) and exotic codecs.
    """
    try:
        if file_path.startswith(("http://", "https://")):
            # Force the FFmpeg path for remote files (robust for any container).
            raise ValueError("remote file -> ffmpeg")
        import librosa
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        return wav_data
    except Exception as e:
        if verbose and not str(e).endswith("ffmpeg"):
            print(f"librosa load failed, falling back to ffmpeg: {e}", file=sys.stderr)
        try:
            command = [
                'ffmpeg',
                '-i', file_path,
                '-ar', str(WAV_SAMPLE_RATE),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-f', 'wav',
                '-'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data, stderr_data = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr_data.decode('utf-8', errors='ignore')}")

            with io.BytesIO(stdout_data) as data_io:
                wav_data, _ = sf.read(data_io, dtype='float32')
            return wav_data
        except Exception as ffmpeg_e:
            raise RuntimeError(f"Failed to load audio from '{file_path}' even with ffmpeg. Error: {ffmpeg_e}")


def build_srt(items, max_cue_sec: float = 6.0, gap_sec: float = 0.8, max_chars: int = 42) -> str:
    """Group word-level forced-alignment items into readable SRT cues.

    A new cue starts when the current one would exceed ``max_cue_sec`` in duration
    or ``max_chars`` in length, or when the silent gap before the next word exceeds
    ``gap_sec``. Word items carry no punctuation, so cues are joined with spaces for
    space-delimited languages and concatenated for CJK.

    Args:
        items: Iterable of objects with ``.text``, ``.start_time``, ``.end_time`` (seconds).
        max_cue_sec: Maximum duration of a single subtitle cue.
        gap_sec: Silence (seconds) between words that forces a cue break.
        max_chars: Soft maximum characters per cue.

    Returns:
        SRT-formatted string.
    """
    items = [it for it in items if str(getattr(it, "text", "")).strip() != ""]
    if not items:
        return ""

    def join_words(words):
        # Use spaces unless the run is dominated by CJK single-char tokens.
        cjk = sum(1 for w in words if len(w) == 1 and ord(w[0]) >= 0x3000)
        sep = "" if cjk > len(words) / 2 else " "
        return sep.join(words)

    cues = []  # list of (start, end, [words])
    cur_words = [items[0].text]
    cur_start = items[0].start_time
    cur_end = items[0].end_time

    for prev, it in zip(items, items[1:]):
        gap = it.start_time - prev.end_time
        cur_text_len = len(join_words(cur_words))
        too_long = (it.end_time - cur_start) > max_cue_sec
        too_wide = cur_text_len + 1 + len(it.text) > max_chars
        if gap > gap_sec or too_long or too_wide:
            cues.append((cur_start, cur_end, cur_words))
            cur_words = [it.text]
            cur_start = it.start_time
            cur_end = it.end_time
        else:
            cur_words.append(it.text)
            cur_end = it.end_time
    cues.append((cur_start, cur_end, cur_words))

    subtitles = []
    for idx, (start, end, words) in enumerate(cues, start=1):
        # Guard against zero/negative durations from alignment noise.
        if end <= start:
            end = start + 0.1
        subtitles.append(srt.Subtitle(
            index=idx,
            start=timedelta(seconds=float(start)),
            end=timedelta(seconds=float(end)),
            content=join_words(words),
        ))
    return srt.compose(subtitles)
