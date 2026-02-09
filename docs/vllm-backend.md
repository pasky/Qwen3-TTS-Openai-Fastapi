# vLLM-Omni Backend for Qwen3-TTS

This document provides detailed information about using the vLLM-Omni backend for faster Qwen3-TTS inference.

## Overview

The vLLM-Omni backend provides day-0 support for Qwen3-TTS with optimized inference performance. While vLLM-Omni currently only supports offline inference (not true audio streaming), it can significantly speed up generation compared to the official backend.

### Key Features

- ⚡ **Faster Inference** - Optimized for throughput and latency
- 🎯 **Same API** - Drop-in replacement, no client changes needed
- 🔧 **Thread-Safe** - Built-in concurrency handling with async locks
- 🚀 **Warmup Support** - Optional warmup on startup to reduce first-request latency
- 📊 **GPU Optimized** - Best performance on NVIDIA GPUs (tested on RTX 3090)

## Installation

### Option 1: Docker (Recommended)

The easiest way to use the vLLM backend is via Docker:

```bash
# Build the vLLM-enabled image
docker-compose --profile vllm build qwen3-tts-vllm

# Run the vLLM backend service
docker-compose --profile vllm up qwen3-tts-vllm
```

### Option 2: Manual Installation

1. Install the base package:

```bash
pip install -e .
```

2. Install vLLM **and vLLM-Omni** (requires CUDA):

```bash
pip install vllm>=0.4.0 vllm-omni
```

Or install with the vllm extras (recommended):

```bash
pip install -e ".[vllm]"
```

If you use `uv`:

```bash
uv sync --extra api --extra vllm
```

3. **Apply required patches** to the installed `vllm-omni` package:

```bash
uv run python patches/apply.py
```

See [Required Patches](#required-patches) below for details.

## Configuration

### Environment Variables

Configure the backend using environment variables:

```bash
# Select backend (required)
export TTS_BACKEND=vllm_omni

# Optional: Override model (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
export TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

# Optional: Enable warmup on startup (recommended)
export TTS_WARMUP_ON_START=true

# Server settings
export HOST=0.0.0.0
export PORT=8880
export WORKERS=1
```

### Model Selection

We recommend the **0.6B model** for best speed/quality tradeoff:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | **Recommended** - Best for production |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | Higher quality, slower |

The 0.6B model provides excellent quality with significantly faster inference, making it ideal for real-time applications.

## Usage

### Starting the Server

```bash
# Set backend to vLLM-Omni
export TTS_BACKEND=vllm_omni
export TTS_WARMUP_ON_START=true

# Start the server
python -m api.main
```

### API Usage

The API is identical to the official backend:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hello! This is Qwen3-TTS with vLLM-Omni backend."
)
response.stream_to_file("output.mp3")
```

### Health Check

Check backend status:

```bash
curl http://localhost:8880/health
```

Response:

```json
{
  "status": "healthy",
  "backend": {
    "name": "vllm_omni",
    "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "ready": true
  },
  "device": {
    "type": "cuda:0",
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "vram_total": "24.00 GB",
    "vram_used": "3.45 GB"
  },
  "version": "0.1.0"
}
```

## Performance Tuning

### GPU Memory

vLLM requires sufficient GPU memory. For different models:

- **0.6B model**: ~3-4 GB VRAM
- **1.7B model**: ~8-10 GB VRAM

### Concurrency

The vLLM backend uses an async lock to ensure thread safety. For high-throughput scenarios:

1. Use multiple workers (if supported by your vLLM version)
2. Deploy multiple instances behind a load balancer
3. Consider request batching at the application level

### Warmup

Enable warmup to reduce first-request latency:

```bash
export TTS_WARMUP_ON_START=true
```

This runs a test inference during startup to pre-load the model.

## Voice and Language Settings

### Supported Voices

The vLLM backend supports the same voices as the official backend:

- Vivian (Female)
- Ryan (Male)
- Sophia (Female)
- Isabella (Female)
- Evan (Male)
- Lily (Female)

Plus OpenAI-compatible aliases: alloy, echo, fable, nova, onyx, shimmer

### Supported Languages

- English
- Chinese (中文)
- Japanese (日本語)
- Korean (한국어)
- German (Deutsch)
- French (Français)
- Spanish (Español)
- Russian (Русский)
- Portuguese (Português)
- Italian (Italiano)

### Language Selection

Specify language in three ways:

1. **Auto-detection** (default):
```python
response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hello world"
)
```

2. **Model suffix**:
```python
response = client.audio.speech.create(
    model="tts-1-es",  # Spanish
    voice="Vivian",
    input="Hola mundo"
)
```

3. **Request parameter** (if extended):
```python
response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hola mundo",
    language="Spanish"  # via extended schema
)
```

## Required Patches

The vLLM-Omni backend requires small patches to the installed `vllm-omni`
package. These are tracked as unified diff files in `patches/` and applied
directly to the files inside your virtualenv's `site-packages`.

### Why patches are needed

`vllm-omni==0.14.0` has two issues that affect this project:

1. **`forward()` dict mutation** — `Qwen3TTSModelForGeneration.forward()` calls
   `.pop()` on the shared `runtime_additional_information` dict, mutating
   per-request state that may be referenced across iterations. The fix is a
   one-line shallow copy before the `.pop()` calls.

2. **Voice-clone IPC serialisation** — vLLM-Omni uses msgpack to serialise
   `additional_information` between the API process and GPU worker processes.
   Nested torch tensors inside list/dict payloads are not reliably
   reconstructed on the worker side. For `CUSTOM_VOICE`, this project sends
   precomputed voice-clone conditioning tensors as top-level
   `additional_information` keys. The patch adds support for these flattened
   keys in `generate_voice_clone()`, reconstructing a `VoiceClonePromptItem`
   on the worker side.

### Applying patches

```bash
# Apply all patches (idempotent — safe to run multiple times)
uv run python patches/apply.py

# Check whether patches are applied without modifying anything
uv run python patches/apply.py --check
```

> **Important:** Patches must be re-applied after every `uv sync` or
> `pip install` that reinstalls `vllm-omni`, since those operations replace
> the patched files with fresh copies.

### Patch files

| File | Target package | Description |
|------|---------------|-------------|
| `patches/vllm_omni_qwen3_tts.patch` | `vllm-omni==0.14.0` | Dict mutation fix + voice-clone IPC support |

### Adding new patches

1. Copy the original file from `site-packages` to `/tmp/original.py`
2. Make your changes in a copy at `/tmp/patched.py`
3. Generate a diff: `diff -u /tmp/original.py /tmp/patched.py > patches/my_fix.patch`
4. Edit the patch header paths to be relative to `site-packages`
5. Register the patch in `PATCH_TARGETS` inside `patches/apply.py`

## Troubleshooting

### vLLM Installation Issues

**Problem**: vLLM fails to install

**Solution**:
- Ensure CUDA is installed (CUDA 11.8+ or 12.1+)
- Check PyTorch compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
- Try: `pip install vllm --no-build-isolation`

### CUDA Out of Memory

**Problem**: `CUDA out of memory` error

**Solutions**:
1. Use the smaller 0.6B model instead of 1.7B
2. Reduce `max_model_len` (set via code modification)
3. Close other GPU applications
4. Use a GPU with more VRAM

### Slow First Request

**Problem**: First request takes a long time

**Solutions**:
1. Enable warmup: `TTS_WARMUP_ON_START=true`
2. This is expected for model loading; subsequent requests will be fast
3. Consider keeping the service running rather than starting/stopping

### Backend Not Loading

**Problem**: Server starts but backend initialization fails

**Solutions**:
1. Check logs for detailed error messages
2. Verify vLLM is installed: `pip list | grep vllm`
3. Verify model accessibility from HuggingFace
4. Check GPU availability: `nvidia-smi`

### Different Audio Quality

**Problem**: Audio quality differs from official backend

**Explanation**:
- Both backends use the same models
- Minor differences may occur due to different inference paths
- If quality is critical, stick with the official backend
- The 0.6B model trades slight quality for speed

## Docker Deployment

### Build vLLM Image

```bash
docker build -t qwen3-tts:vllm --target vllm-production .
```

### Run vLLM Container

```bash
docker run -d \
  --name qwen3-tts-vllm \
  --gpus all \
  -p 8880:8880 \
  -e TTS_BACKEND=vllm_omni \
  -e TTS_WARMUP_ON_START=true \
  -e TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  qwen3-tts:vllm
```

### Docker Compose

Use the provided compose configuration:

```bash
# Start vLLM backend
docker-compose --profile vllm up -d qwen3-tts-vllm

# View logs
docker-compose logs -f qwen3-tts-vllm

# Stop
docker-compose --profile vllm down
```

## Comparison: vLLM vs Official Backend

| Feature | Official Backend | vLLM-Omni Backend |
|---------|------------------|-------------------|
| **Speed** | ⚡⚡ Medium | ⚡⚡⚡ Fast |
| **Memory** | ~6-8 GB (1.7B) | ~3-4 GB (0.6B) |
| **Setup** | ✅ Simple | ⚠️ Requires vLLM |
| **Quality** | ⭐⭐⭐⭐ Best | ⭐⭐⭐ Good |
| **True Streaming** | ❌ No | ❌ No |
| **Chunk Streaming** | ✅ Yes | ✅ Yes |
| **OpenWebUI Compatible** | ✅ Yes | ✅ Yes |

## Notes

- **True Streaming**: Neither backend supports true audio streaming over HTTP currently. Both use OpenWebUI's chunk-based approach.
- **Chunk Streaming**: Long text is split into chunks, each processed as a separate TTS request.
- **Production Use**: vLLM backend is suitable for production but test thoroughly in your environment first.
- **Model Updates**: As vLLM-Omni and Qwen3-TTS evolve, check for updates to both packages.

## Support

For issues specific to:
- **vLLM-Omni**: Check [vLLM GitHub](https://github.com/vllm-project/vllm)
- **Qwen3-TTS**: Check [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen-TTS)
- **This Implementation**: Open an issue in this repository
