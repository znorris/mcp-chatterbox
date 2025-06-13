# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server for Chatterbox TTS that provides text-to-speech generation with voice cloning capabilities. The server uses quantized models (6-bit q6_k.gguf) for efficient memory usage and supports Apple Silicon MPS acceleration and CUDA.

## Architecture

- **Single-file MCP server**: `mcp_server.py` contains the complete ChatterboxMCPServer class
- **Three main tools**:
  - `chatterbox_tts_generate`: Basic TTS with optional voice cloning
  - `chatterbox_tts_clone_voice`: Voice cloning from audio samples  
  - `chatterbox_tts_info`: Model and system information
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

## Key Implementation Details

- Model initialization is deferred until first tool call to avoid startup delays
- Audio files are saved to temp directory by default unless output_path specified
- Voice cloning requires existing audio file path validation
- Sample rate is determined by the loaded model (typically 24kHz)
- Exaggeration parameter (0.0-1.0) controls voice characteristics