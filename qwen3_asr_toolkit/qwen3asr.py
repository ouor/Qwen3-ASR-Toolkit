import sys

import torch

from qwen_asr import Qwen3ASRModel
import qwen_asr.inference.qwen3_asr as _qwen3_asr_mod


# Empirically-measured peak GPU memory of a single ASR chunk for Qwen3-ASR-1.7B in
# bfloat16 without FlashAttention (RTX 3090, transformers backend). Each entry is
# (chunk_seconds, peak_reserved_GB_including_weights). Memory grows super-linearly with
# chunk length, so we size chunks up front to fit available VRAM instead of risking OOM.
# The list is capped at 300s because beyond that a single chunk's transcription starts
# getting truncated by max_new_tokens (output stops growing), silently losing text.
_CHUNK_MEM_TABLE = [
    (60, 4.4),
    (120, 4.8),
    (180, 6.3),
    (240, 7.6),
    (300, 9.3),
]
# Weights resident on the GPU (GB) — subtracted so we budget only the *activation* memory
# against whatever VRAM is free at transcribe time (the weights are already allocated).
_WEIGHTS_GB = 3.8
# Fraction of free VRAM we allow a chunk's activations to use (headroom for fragmentation).
_VRAM_SAFETY = 0.80
# Absolute floor if even the smallest table entry does not fit.
MIN_CHUNK_SEC = 60
# Fallback when free VRAM cannot be queried (e.g. CPU device).
DEFAULT_CHUNK_SEC = 300


# Map CLI dtype strings to torch dtypes.
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _device_index(device: str):
    """Return the CUDA device index for a 'cuda[:N]' string, or None for CPU/unknown."""
    d = str(device).lower()
    if not d.startswith("cuda"):
        return None
    if ":" in d:
        try:
            return int(d.split(":", 1)[1])
        except ValueError:
            return None
    return 0


def pick_chunk_sec(device: str) -> int:
    """Choose the largest ASR chunk length whose measured activation memory fits free VRAM.

    Deterministic: reads the currently-free VRAM and picks from the measured memory table
    so a single chunk never exceeds the budget (no OOM, no CPU offload, no token-truncation).
    Falls back to ``DEFAULT_CHUNK_SEC`` when VRAM cannot be queried.
    """
    idx = _device_index(device)
    if idx is None:
        return DEFAULT_CHUNK_SEC
    try:
        free_bytes, _ = torch.cuda.mem_get_info(idx)
    except Exception:
        return DEFAULT_CHUNK_SEC
    free_gb = free_bytes / 1024**3
    budget_gb = free_gb * _VRAM_SAFETY

    chosen = MIN_CHUNK_SEC
    for secs, peak_gb in _CHUNK_MEM_TABLE:
        activation_gb = peak_gb - _WEIGHTS_GB  # weights are already resident
        if activation_gb <= budget_gb:
            chosen = secs
        else:
            break
    return chosen


class QwenASR:
    """Local, fully-offline Qwen3-ASR backend.

    Thin wrapper around ``qwen_asr.Qwen3ASRModel`` (transformers backend) that runs
    the model on a local GPU instead of calling the DashScope cloud API. The
    underlying library already handles long-audio chunking, batching, language
    identification and repetition cleanup, so this wrapper only loads the model
    once and exposes a single ``transcribe`` call.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        aligner_path: str = None,
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 4096,
        attn_implementation: str = None,
    ):
        self.device = device
        torch_dtype = DTYPE_MAP.get(str(dtype).lower(), torch.bfloat16)

        forced_aligner_kwargs = None
        if aligner_path:
            forced_aligner_kwargs = dict(dtype=torch_dtype, device_map=device)
            if attn_implementation:
                forced_aligner_kwargs["attn_implementation"] = attn_implementation

        model_kwargs = dict(
            dtype=torch_dtype,
            device_map=device,
            forced_aligner=aligner_path,
            forced_aligner_kwargs=forced_aligner_kwargs,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)

    def transcribe(self, audio, context: str = "", language: str = None, return_time_stamps: bool = False,
                   chunk_sec: int = None, verbose: bool = True):
        """Transcribe a single audio input.

        Args:
            audio: A path/URL/base64 string or an ``(np.ndarray, sample_rate)`` tuple.
            context: Optional text context to bias recognition of specific terms.
            language: Optional forced language (canonical name, e.g. "Korean"). ``None`` = auto-detect.
            return_time_stamps: If True, also return word-level timestamps via the forced aligner.
            chunk_sec: Internal ASR chunk length (seconds). ``None`` auto-sizes it to fit free VRAM
                via :func:`pick_chunk_sec` so long audio never OOMs and never truncates.
            verbose: Print the chosen chunk length to stderr.

        Returns:
            A single ``ASRTranscription`` (``.language``, ``.text``, ``.time_stamps``).
        """
        cur = int(chunk_sec) if chunk_sec else pick_chunk_sec(self.device)
        if verbose:
            print(f"[qwen3-asr] ASR chunk length: {cur}s", file=sys.stderr)

        # The library reads its chunk length from module constants inside transcribe();
        # patch them so each chunk's activation memory stays within the VRAM budget.
        _qwen3_asr_mod.MAX_ASR_INPUT_SECONDS = cur
        _qwen3_asr_mod.MAX_FORCE_ALIGN_INPUT_SECONDS = min(_qwen3_asr_mod.MAX_FORCE_ALIGN_INPUT_SECONDS, cur)

        results = self.model.transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        return results[0]
