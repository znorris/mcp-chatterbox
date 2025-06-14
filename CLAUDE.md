# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server for Chatterbox TTS that provides text-to-speech generation with voice cloning capabilities. The server uses quantized models (6-bit q6_k.gguf) for efficient memory usage and supports Apple Silicon MPS acceleration and CUDA.

## Architecture

- **Single-file MCP server**: `mcp_server.py` contains the complete ChatterboxMCPServer class
- **Five main tools**:
  - `chatterbox_tts_generate`: Basic TTS with optional voice cloning or voice store selection
  - `chatterbox_tts_clone_voice`: Voice cloning from audio samples  
  - `chatterbox_tts_info`: Model and system information
  - `chatterbox_tts_prepare_audio`: Audio file optimization for voice cloning
  - `chatterbox_tts_voice_store`: Voice state management (save/load/list/delete)
- **Dual transport support**: STDIO (default) and SSE (HTTP-based)
- **Device auto-detection**: Automatically selects CUDA > MPS > CPU
- **Lazy loading**: ChatterboxTTS model loads on first use

## Development Commands

### Setup virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install with dev dependencies
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest
```

### Format code
```bash
black .
isort .
```

### Type checking
```bash
mypy mcp_server.py
```

## Running the Server

### STDIO transport (default for MCP clients)
```bash
python mcp_server.py
```

### SSE transport (HTTP-based)
```bash
python mcp_server.py --transport sse --host localhost --port 8000
```

### Model selection
```bash
# Use environment variable
export CHATTERBOX_MODEL="t3_cfg-q4_k.gguf"
python mcp_server.py

# Or use command line argument
python mcp_server.py --model t3_cfg-q8_0.gguf

# Enable hot reload with specific model
python mcp_server.py --hot-reload --model t3_cfg-f16.gguf
```

## Key Implementation Details

- Model initialization is deferred until first tool call to avoid startup delays
- Audio files are saved to temp directory by default unless output_path specified
- Voice cloning requires existing audio file path validation
- Sample rate is determined by the loaded model (typically 24kHz)
- Exaggeration parameter (0.0-1.0) controls voice characteristics
- Voice store system preserves voice states in memory for seamless switching
- PyTorch 2.6 compatibility: Uses `weights_only=False` for voice state restoration
- Hot reload support available with `--hot-reload` flag for development

## Voice Store Usage

The voice store system allows saving and reusing voice configurations:

```bash
# Save a voice from audio sample
chatterbox_tts_voice_store(action="save", voice_name="narrator", voice_sample_path="./audio/narrator.wav")

# List available voices
chatterbox_tts_voice_store(action="list")

# Generate speech with saved voice
chatterbox_tts_generate(text="Hello world", voice_name="narrator", output_path="./output.wav")

# Delete a voice
chatterbox_tts_voice_store(action="delete", voice_name="narrator")
```