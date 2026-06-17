# Qwen3-ASR-Toolkit

[![PyPI version](https://badge.fury.io/py/qwen3-asr-toolkit.svg)](https://badge.fury.io/py/qwen3-asr-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Also in](https://img.shields.io/badge/Also%20in-Java-orange.svg)](#-implementations-in-other-languages)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 😊 Important Notice

Qwen3-ASR is now **open-sourced** 🎉🎉🎉. Welcome to visit the [**GitHub**](https://github.com/QwenLM/Qwen3-ASR) and [**blog**](https://qwen.ai/blog?id=qwen3asr) for more information. The open-source model offers functionality comparable to the API and supports free, fast local deployment. Qwen3-ASR open-source model includes two powerful **all-in-one speech recognition models (0.6B/1.7B)** that support language identification and ASR for **52 languages and dialects**, as well as a novel non-autoregressive speech forced-alignment model that can align text–speech pairs in 11 languages. Its powerful performance is sufficient to deliver highly compelling speech-to-text transcription capabilities. Welcome to use it!

## Overview 

A high-performance Python command-line toolkit that runs the **open-source Qwen3-ASR models fully offline on your local GPU**. It wraps the official [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) inference library with a simple CLI that transcribes long audio/video files of any length to `.txt`, and optionally generates timestamped `.srt` subtitles via the Qwen3-ForcedAligner. No API key, no cloud — your audio never leaves your machine.

> **Note:** This is a local-GPU fork. The upstream project drives the cloud **Qwen-ASR API** instead; if you want the API version, see the [original repository](https://github.com/QwenLM/Qwen3-ASR-Toolkit).

## 🚀 Key Features

-   **Fully Offline**: Runs the Qwen3-ASR models locally on your own NVIDIA GPU. No API key and no network calls during transcription.
-   **Any Length**: The underlying library automatically chunks long audio internally (up to ~20 min per ASR window), so hours-long files just work.
-   **52 Languages**: Automatic language identification and recognition across 52 languages and dialects, including Korean, English, Chinese, Japanese, and more.
-   **Intelligent Post-Processing**: Common ASR **hallucinations and repetitive artifacts** are detected and removed automatically by the library for cleaner transcripts.
-   **SRT Subtitle Generation**: Optionally produce timestamped **`.srt` subtitle files** using the Qwen3-ForcedAligner model for word-level alignment.
-   **Automatic Audio Resampling**: Converts audio from any sample rate and channel count to the 16kHz mono format the model expects — no pre-processing needed.
-   **Universal Media Support**: Handles virtually any audio and video format (e.g., `.mp4`, `.mov`, `.mkv`, `.mp3`, `.wav`, `.m4a`) thanks to its FFmpeg fallback.
-   **Simple & Easy to Use**: A straightforward command-line interface gets you transcribing with a single command.

## ⚙️ How It Works

This tool follows a simple, fully-local pipeline:

1.  **Media Loading**: Loads your media file (**local file or remote URL**), using librosa with an FFmpeg fallback for video and exotic codecs, and resamples to 16kHz mono.
2.  **Model Loading**: Loads the Qwen3-ASR model onto your GPU once (and the Qwen3-ForcedAligner too, when `--save-srt` is requested).
3.  **Transcription**: Hands the waveform to `qwen_asr.Qwen3ASRModel.transcribe()`, which internally chunks long audio, runs batched inference, identifies the language, and cleans up repetitions.
4.  **Output Generation**: The transcription is printed to the console and saved to a UTF-8 `.txt` file. **Optionally, a timestamped `.srt` subtitle file is also generated** from forced-alignment word timestamps.

## 🏁 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   **An NVIDIA GPU with CUDA.** A 1.7B model needs roughly 5–6 GB of VRAM in bfloat16; an RTX 3090 (24 GB) runs it comfortably. CPU-only inference works but is very slow.
-   **Python 3.10+** (3.12 recommended). A clean conda environment avoids dependency conflicts.
-   **FFmpeg** on your PATH, for video and non-standard audio formats.
    -   **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
    -   **macOS**: `brew install ffmpeg`
    -   **Windows**: Download from the [official FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's PATH.

### Installation

#### 1. Create an environment and install CUDA PyTorch

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr

# Install a CUDA build of torch matching your driver (cu128 shown here).
# Keep torch / torchaudio / torchvision on the SAME CUDA build.
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 2. Install the toolkit

```bash
git clone https://github.com/QwenLM/Qwen3-ASR-Toolkit.git
cd Qwen3-ASR-Toolkit
pip install .
```

This pulls in the official [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) inference library and makes the `qwen3-asr` command available.

#### 3. (Optional) Pre-download the models

The model weights download automatically on first run. To fetch them ahead of time (or on a machine that can't download during execution):

```bash
hf download Qwen/Qwen3-ASR-1.7B
hf download Qwen/Qwen3-ForcedAligner-0.6B   # only needed for --save-srt
```

> **Windows tip:** if Hugging Face downloads stall, disable the Xet transfer backend and use plain HTTPS:
> ```powershell
> $env:HF_HUB_DISABLE_XET=1
> hf download Qwen/Qwen3-ASR-1.7B
> ```

## 📖 Usage

Once installed, you can use the `qwen3-asr` command directly from your terminal. By default, the tool will print progress information.

### Command

```bash
qwen3-asr -i <input_file_or_url> [-c <context>] [-l <language>] [-m <model>] [--device <dev>] [--dtype <dtype>] [--save-srt] [-o <output>] [-s]
```

### Arguments

| Argument            | Short  | Description                                                                                  | Required/Optional                               |
| ------------------- | ------ | -------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `--input-file`      | `-i`   | Path to the local media file or a remote URL (http/https) to transcribe.                     | **Required**                                    |
| `--context`         | `-c`   | Text context to bias the model toward specific terms/jargon.                                  | Optional, Default: `""`                         |
| `--language`        | `-l`   | Force a language (e.g. `Korean`, `English`). Omit to auto-detect.                             | Optional, Default: auto-detect                  |
| `--model`           | `-m`   | ASR model repo id or local path.                                                             | Optional, Default: `Qwen/Qwen3-ASR-1.7B`        |
| `--device`          |        | Torch device, e.g. `cuda:0` or `cpu`.                                                         | Optional, Default: `cuda:0`                      |
| `--dtype`           |        | Inference dtype: `bfloat16`, `float16`, or `float32`.                                         | Optional, Default: `bfloat16`                    |
| `--max-batch-size`  |        | Max internal inference batch size. Raise only for many short files; keep at 1 for long audio.| Optional, Default: `1`                          |
| `--chunk-sec`       |        | Internal ASR chunk length (seconds). Auto-sized to free VRAM by default — no OOM, no truncation. | Optional, Default: auto                       |
| `--max-new-tokens`  |        | Max tokens generated per audio chunk.                                                        | Optional, Default: `4096`                       |
| `--attn`            |        | `attn_implementation`, e.g. `flash_attention_2` or `sdpa`.                                    | Optional                                        |
| `--save-srt`        | `-srt` | Also generate a timestamped `.srt` subtitle file (loads the forced aligner).                 | Optional                                        |
| `--aligner`         |        | Forced-aligner model (only used with `--save-srt`).                                           | Optional, Default: `Qwen/Qwen3-ForcedAligner-0.6B` |
| `--output`          | `-o`   | Output path base (without extension). Default: alongside the input file.                      | Optional                                        |
| `--silence`         | `-s`   | Silence mode. Suppresses progress messages on the terminal.                                  | Optional                                        |

### Output

The transcription is printed to the terminal (unless in `--silence` mode) and saved as a UTF-8 `.txt` file next to the input (e.g. `my_video.mp4` → `my_video.txt`). The first line is the detected language, the second is the transcript.

**If you use the `--save-srt` flag, a corresponding `my_video.srt` subtitle file is also created.**

---

## ✨ Examples

Here are a few examples of how to use the tool.

#### 1. Basic Transcription of a Local File

Transcribe a video file with auto language detection on the default GPU. The model downloads automatically on first run.

```bash
qwen3-asr -i "/path/to/my/long_lecture.mp4"
```

#### 2. Transcribe a Remote Audio File

Directly process an audio file from a URL.

```bash
qwen3-asr -i "https://somewebsite.com/audios/podcast_episode.mp3"
```

#### 3. Generate an SRT Subtitle File

Use the `--save-srt` (or `-srt`) flag to also generate a timestamped subtitle file via the forced aligner. This is ideal for video captioning.

```bash
qwen3-asr -i "/path/to/my/documentary.mp4" -srt
```
*This command will create `documentary.txt` and `documentary.srt`.*

#### 4. Force a Language

Skip language detection when you already know the language (e.g. `Korean`).

```bash
qwen3-asr -i "/path/to/my/interview.wav" -l Korean
```

#### 5. Provide Context and Pick a Model

Bias recognition toward domain jargon with `-c`, and choose a model with `-m` (e.g. the lighter `0.6B`).

```bash
qwen3-asr -i "/path/to/my/tech_talk.mp4" -c "Qwen3-ASR, FFmpeg, bfloat16" -m Qwen/Qwen3-ASR-0.6B -srt
```

#### 6. Manage GPU Memory

Long audio works out of the box — the ASR chunk length is **auto-sized to fit your free VRAM** (no OOM, no offload, no truncation). On a smaller GPU you can force a shorter chunk, or pick a specific device/dtype:

```bash
qwen3-asr -i "/path/to/my/podcast.wav" --device cuda:0 --dtype float16 --chunk-sec 180
```

#### 7. Run in Silence Mode

Use `-s` / `--silence` to suppress progress messages. The transcript is still saved to a file.

```bash
qwen3-asr -i "/path/to/my/meeting_recording.m4a" -s
```

## 🌍 Implementations in Other Languages

While this project provides a full-featured Python toolkit, we also host implementations in other programming languages to demonstrate how the same core logic can be applied across different technology stacks. We warmly welcome the community to contribute examples in more languages!

### ☕ Java Example

We have provided a Java version as a standalone example located in the `examples/java-example` directory of this repository. This example showcases how to implement the key features of the toolkit—including VAD-based audio chunking, parallel API requests, and result aggregation—using Java. It serves as a great starting point for Java developers looking to integrate Qwen-ASR into their applications.

### How to Contribute Your Version

If you have implemented a similar toolkit in another language (e.g., **Go**, **Rust**, **C#**, **JavaScript/Node.js**), we would love to feature it! Please open a pull request to add your implementation to the `examples` directory. For more details on contributing, see the [Contributing](#-contributing) section below.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please feel free to fork the repo, create a feature branch, and open a pull request. You can also open an issue with the "enhancement" tag.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
