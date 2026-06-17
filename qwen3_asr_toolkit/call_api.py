import os
import argparse

import requests

from urllib.parse import urlparse

from qwen3_asr_toolkit.qwen3asr import QwenASR
from qwen3_asr_toolkit.audio_tools import load_audio, build_srt, WAV_SAMPLE_RATE


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully offline local-GPU toolkit for Qwen3-ASR: long-audio transcription with optional SRT timestamps."
    )
    parser.add_argument("--input-file", "-i", type=str, required=True, help="Input media file path or http(s) URL")
    parser.add_argument("--context", "-c", type=str, default="", help="Optional text context to bias recognition")
    parser.add_argument("--language", "-l", type=str, default=None, help="Force a language (e.g. Korean, English). Default: auto-detect")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_ASR_MODEL, help="ASR model repo id or local path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Inference dtype")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Max internal inference batch size. Raise only when transcribing many short files; keep at 1 for long audio to avoid OOM")
    parser.add_argument("--chunk-sec", type=int, default=None, help="Internal ASR chunk length in seconds. Default: auto-sized to fit free VRAM (no OOM, no truncation)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max tokens generated per audio chunk")
    parser.add_argument("--attn", type=str, default=None, help="attn_implementation, e.g. flash_attention_2 or sdpa")
    parser.add_argument("--save-srt", "-srt", action="store_true", help="Also write a timestamped .srt subtitle file")
    parser.add_argument("--aligner", type=str, default=DEFAULT_ALIGNER_MODEL, help="Forced-aligner model (only loaded with --save-srt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path base (without extension). Default: alongside input")
    parser.add_argument("--silence", "-s", action="store_true", help="Reduce terminal output")
    return parser.parse_args()


def resolve_output_base(input_file: str, output: str) -> str:
    if output:
        return os.path.splitext(output)[0]
    if os.path.exists(input_file):
        return os.path.splitext(input_file)[0]
    # Remote URL: derive a name from the path component.
    name = os.path.splitext(urlparse(input_file).path)[0].split("/")[-1] or "transcription"
    return name


def main():
    args = parse_args()
    input_file = args.input_file
    silence = args.silence

    def info(msg):
        if not silence:
            print(msg)

    # Validate input.
    if input_file.startswith(("http://", "https://")):
        try:
            response = requests.head(input_file, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                raise FileNotFoundError(f"returned status code {response.status_code}")
        except Exception as e:
            raise FileNotFoundError(f"HTTP link {input_file} does not exist or is inaccessible: {str(e)}")
    elif not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file "{input_file}" does not exist!')

    # Load and resample audio (handles video via ffmpeg fallback).
    info("Loading audio...")
    wav = load_audio(input_file, verbose=not silence)
    info(f"Loaded audio duration: {len(wav) / WAV_SAMPLE_RATE:.2f}s")

    # Load the model (forced aligner only when SRT is requested).
    info(f"Loading model '{args.model}' on {args.device} ({args.dtype})...")
    qwen3asr = QwenASR(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        aligner_path=args.aligner if args.save_srt else None,
        max_inference_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn,
    )

    info("Transcribing...")
    result = qwen3asr.transcribe(
        audio=(wav, WAV_SAMPLE_RATE),
        context=args.context,
        language=args.language,
        return_time_stamps=args.save_srt,
        chunk_sec=args.chunk_sec,
        verbose=not silence,
    )

    language = result.language or "Unknown"
    full_text = result.text

    info(f"Detected Language: {language}")
    info(f"Full Transcription: {full_text}")

    out_base = resolve_output_base(input_file, args.output)

    # Save transcription text (UTF-8 for multilingual safety on Windows).
    txt_path = out_base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(language + "\n")
        f.write(full_text + "\n")
    print(f'Transcription of "{input_file}" saved to "{txt_path}"')

    # Save SRT subtitles.
    if args.save_srt:
        items = list(result.time_stamps) if result.time_stamps is not None else []
        srt_content = build_srt(items)
        srt_path = out_base + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f'SRT subtitles of "{input_file}" saved to "{srt_path}"')


if __name__ == "__main__":
    main()
