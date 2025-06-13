# MCP Chatterbox

A Model Context Protocol (MCP) server for Chatterbox TTS with quantized model support, voice cloning capabilities, and optimized performance on Apple Silicon.

## Features

- **Quantized Model Support**: Uses 6-bit quantized models (q6_k.gguf) for efficient memory usage
- **Voice Cloning**: Clone voices from audio samples for personalized speech generation
- **Apple Silicon Optimization**: Leverages MPS (Metal Performance Shaders) for accelerated inference on Mac
- **CUDA Support**: Full CUDA acceleration on compatible systems
- **Adjustable Voice Control**: Fine-tune voice characteristics with exaggeration parameters

## Installation

### Prerequisites

This MCP server requires Python 3.8+ and the Chatterbox TTS library with its quantized models.

### Step 1: Install System Dependencies

#### macOS
```bash
# Install Python 3.8+ if not already installed
brew install python

# For Apple Silicon Macs, ensure you have the latest macOS for MPS support
```

#### Linux (Ubuntu/Debian)
```bash
# Install Python and development tools
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential

# For NVIDIA GPU support, install CUDA toolkit (optional)
# Follow: https://developer.nvidia.com/cuda-downloads
```

#### Windows
```bash
# Install Python 3.8+ from https://python.org or via Chocolatey
choco install python

# For NVIDIA GPU support, install CUDA toolkit (optional)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### Step 2: Install Chatterbox TTS and Models

The Chatterbox TTS system uses quantized models for efficient inference. You have two options:

#### Option A: Automatic Model Download (Recommended)
```bash
# Install the Chatterbox TTS library - models will download automatically
pip install chatterbox-tts
```

The first time you run the MCP server, it will automatically download the required models:
- **Text-to-Audio Model**: `t3_cfg-q6_k.gguf` (~417MB) - 6-bit quantized transformer model
- **Voice Encoder Models**: `ve_fp32-f16.gguf` and `ve_fp32-f32.gguf` - For voice cloning
- **Neural Vocoder**: Various other supporting models

#### Option B: Manual Model Download
If you prefer to download models manually or have specific model requirements:

```bash
# Clone the original Chatterbox repository
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox

# Download models using their provided scripts
python download_models.py

# Available quantized model variants:
# - t3_cfg-q4_k.gguf (smaller, faster, lower quality)
# - t3_cfg-q6_k.gguf (recommended balance)
# - t3_cfg-q8_0.gguf (larger, slower, higher quality)
# - t3_cfg-f16.gguf (unquantized, highest quality)
```

### Step 3: Install This MCP Server

Clone and install the MCP server:

```bash
# Clone this repository
git clone https://github.com/znorris/mcp-chatterbox.git
cd mcp-chatterbox

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the MCP server and dependencies
pip install -e .
```

### Step 4: Configure Claude Code

Add the Chatterbox MCP server to your Claude Code configuration:

#### macOS/Linux
Edit `~/.claude.json` and add to the `mcpServers` section:

```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "type": "stdio",
      "command": "/path/to/mcp-chatterbox/venv/bin/python",
      "args": ["/path/to/mcp-chatterbox/mcp_server.py"],
      "env": {}
    }
  }
}
```

**Example with actual paths:**
```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "type": "stdio",
      "command": "/Users/username/mcp-chatterbox/venv/bin/python",
      "args": ["/Users/username/mcp-chatterbox/mcp_server.py"],
      "env": {}
    }
  }
}
```

#### Windows
Edit `%USERPROFILE%\.claude.json` and add to the `mcpServers` section:

```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "type": "stdio",
      "command": "C:\\path\\to\\mcp-chatterbox\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\mcp-chatterbox\\mcp_server.py"],
      "env": {}
    }
  }
}
```

**Important Notes:**
- Use the full path to the Python executable in your project's virtual environment
- The `env` object should be empty since the venv contains all dependencies
- Replace `/path/to/mcp-chatterbox` with your actual project path

### Step 5: Restart Claude Code

Restart Claude Code to load the new MCP server. You should see "chatterbox-tts" available in the MCP server list.

## Usage

### As an MCP Server

#### STDIO Transport (default)
Configure your MCP client to use this server over STDIO:

```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "command": "python",
      "args": ["/path/to/mcp-chatterbox/mcp_server.py"]
    }
  }
}
```

#### SSE Transport (HTTP)
Run the server with SSE transport for HTTP-based communication:

```bash
python mcp_server.py --transport sse --host localhost --port 8000
```

Then configure your MCP client:

```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### Available Tools

#### `chatterbox_tts_generate`
Generate speech from text using the default voice or with voice cloning.

Parameters:
- `text` (required): The text to convert to speech
- `audio_prompt_path` (optional): Path to audio file for voice cloning
- `output_path` (optional): Output file path (defaults to temp file)
- `exaggeration` (optional): Voice exaggeration factor (0.0 to 1.0, default: 0.5)

#### `chatterbox_tts_clone_voice`
Generate speech with voice cloning from an audio sample.

Parameters:
- `text` (required): The text to convert to speech
- `voice_sample_path` (required): Path to audio file containing voice to clone
- `output_path` (optional): Output file path (defaults to temp file)
- `exaggeration` (optional): Voice exaggeration factor (0.0 to 1.0, default: 0.5)

#### `chatterbox_tts_info`
Get information about the Chatterbox TTS model and system capabilities.

## Model Requirements

This MCP server requires the Chatterbox TTS quantized models. The models will be automatically downloaded when first initializing the ChatterboxTTS class.

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchAudio 2.0+
- For Apple Silicon: macOS with MPS support
- For NVIDIA GPUs: CUDA-compatible PyTorch installation

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
isort .
```

## License

MIT License - see LICENSE file for details.

## Credits

Built on top of the excellent [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) library by Resemble AI.