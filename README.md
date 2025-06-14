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

#### `chatterbox_tts_prepare_audio`
Prepare and optimize audio files for voice cloning (convert MP3/other formats to optimal WAV).

Parameters:
- `input_path` (required): Path to input audio file (MP3, WAV, etc.)
- `output_path` (optional): Output path for optimized WAV file (defaults to temp file)
- `target_sample_rate` (optional): Target sample rate in Hz (default: 24000)
- `mono` (optional): Convert to mono (recommended for voice cloning, default: true)
- `normalize` (optional): Normalize audio volume (default: true)
- `trim_silence` (optional): Trim silence from beginning and end (default: true)

#### `chatterbox_tts_clone_voice`
Generate speech with voice cloning from an audio sample.

Parameters:
- `text` (required): The text to convert to speech
- `voice_sample_path` (required): Path to audio file containing voice to clone
- `output_path` (optional): Output file path (defaults to temp file)
- `exaggeration` (optional): Voice exaggeration factor (0.0 to 1.0, default: 0.5)

#### `chatterbox_tts_info`
Get information about the Chatterbox TTS model and system capabilities.

#### `chatterbox_tts_voice_store`
Manage voice store - save, load, list, and delete different voice states for seamless voice switching.

Parameters:
- `action` (required): Action to perform on voice store ("save", "load", "list", "delete")
- `voice_name` (optional): Name of the voice to save/load/delete
- `voice_sample_path` (optional): Path to audio file for saving new voice (required for 'save' action)

### Voice Store System

The voice store system allows you to save and manage multiple voice configurations for seamless switching between different cloned voices without re-processing audio samples each time.

#### Save a Voice
```python
# Save a voice from an audio sample
result = await client.call_tool("chatterbox_tts_voice_store", {
    "action": "save",
    "voice_name": "narrator_voice",
    "voice_sample_path": "/path/to/narrator_sample.wav"
})
```

#### List Available Voices
```python
# List all saved voices
result = await client.call_tool("chatterbox_tts_voice_store", {
    "action": "list"
})
```

#### Generate Speech with Saved Voice
```python
# Use a saved voice by name
result = await client.call_tool("chatterbox_tts_generate", {
    "text": "Hello, this uses my saved narrator voice!",
    "voice_name": "narrator_voice",
    "output_path": "/path/to/output.wav"
})
```

#### Delete a Voice
```python
# Remove a voice from the store
result = await client.call_tool("chatterbox_tts_voice_store", {
    "action": "delete",
    "voice_name": "narrator_voice"
})
```

The voice store automatically saves a "default" voice when the model first loads, and all voice states are preserved in memory for the duration of the server session.

### Voice Cloning Workflow

The typical workflow for voice cloning with optimal results:

1. **Prepare your audio file** using `chatterbox_tts_prepare_audio`:
   ```python
   # This will convert your MP3 to an optimized WAV file
   result = await client.call_tool("chatterbox_tts_prepare_audio", {
       "input_path": "/path/to/your/voice_sample.mp3",
       "output_path": "/path/to/optimized_voice.wav"
   })
   ```

2. **Generate speech with voice cloning** using `chatterbox_tts_clone_voice`:
   ```python
   # Use the optimized audio for best cloning results
   result = await client.call_tool("chatterbox_tts_clone_voice", {
       "text": "Your text to synthesize",
       "voice_sample_path": "/path/to/optimized_voice.wav",
       "exaggeration": 0.6
   })
   ```

The `chatterbox_tts_prepare_audio` tool automatically handles:
- **Format conversion**: MP3/other formats → WAV
- **Sample rate optimization**: Any rate → 24kHz (optimal for Chatterbox)
- **Channel optimization**: Stereo → Mono (reduces complexity)
- **Volume normalization**: Ensures consistent levels
- **Silence trimming**: Removes dead air for better cloning

## Voice Cloning Audio Preparation

### Preparing Audio Files for Voice Cloning

To achieve the best voice cloning results with Chatterbox TTS, your reference audio files should meet these specifications:

#### Duration Requirements
- **Minimum**: 5 seconds (system can work with as little as this)
- **Recommended**: 10+ seconds for optimal quality
- **Sweet spot**: 7-20 seconds provides excellent zero-shot voice cloning

#### Format and Quality Specifications
- **Preferred format**: WAV (highest quality)
- **Supported formats**: WAV, MP3
- **Sample rate**: 24kHz or higher (minimum)
- **Channels**: Mono preferred, stereo acceptable
- **Bit depth**: 16-bit minimum, 24-bit preferred

#### Audio Quality Requirements
- **Single speaker only**: No background voices or overlapping speech
- **Clean recording**: No background noise, music, or interference
- **Professional quality**: Use a good microphone if possible
- **Consistent volume**: Avoid clipping or very quiet sections

#### Content and Style Guidelines
- **Emotional matching**: The emotion in your reference audio should match your desired output
- **Speaking style consistency**: Use similar speaking style to your intended use case
  - For audiobook generation: Use audiobook-style narration samples
  - For conversational speech: Use natural conversation samples
  - For dramatic content: Use expressive, dramatic samples
- **Clear pronunciation**: Ensure words are clearly articulated
- **Natural pacing**: Avoid rushed or unnaturally slow speech

#### Converting MP3 to Optimal Format

If you have an MP3 file, you can convert it to the optimal WAV format using common tools:

##### Using FFmpeg (recommended)
```bash
# Convert MP3 to high-quality WAV
ffmpeg -i input.mp3 -ar 24000 -ac 1 -f wav output.wav

# For even higher quality (48kHz)
ffmpeg -i input.mp3 -ar 48000 -ac 1 -f wav output.wav
```

##### Using Python (programmatically)
```python
import torchaudio

# Load and resample audio
waveform, sample_rate = torchaudio.load("input.mp3")
resampled = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
torchaudio.save("output.wav", resampled, 24000)
```

#### Quality Impact on Output
- **High-quality reference audio**: Results in natural, clear voice cloning
- **Poor quality reference audio**: May produce distorted or unnatural speech
- **Background noise**: Can significantly reduce cloning accuracy
- **Multiple speakers**: Will confuse the model and produce poor results

#### Built-in Watermarking
All audio generated by Chatterbox TTS includes Resemble AI's Perth (Perceptual Threshold) Watermarker - imperceptible neural watermarks that:
- Survive MP3 compression and audio editing
- Maintain nearly 100% detection accuracy
- Are inaudible to human listeners

## Model Requirements and Selection

This MCP server requires the Chatterbox TTS quantized models. The models will be automatically downloaded when first initializing the ChatterboxTTS class.

### Choosing the Right Model for Your System

Chatterbox TTS supports several quantized model variants optimized for different hardware configurations:

#### Model Variants

| Model | File Size | Memory Usage | Speed | Quality | Best For |
|-------|-----------|--------------|-------|---------|----------|
| `t3_cfg-q4_k.gguf` | ~312MB | ~2GB RAM | Fastest | Good | Low-memory systems, embedded devices |
| `t3_cfg-q6_k.gguf` | ~417MB | ~3GB RAM | Fast | Excellent | **Recommended balance** |
| `t3_cfg-q8_0.gguf` | ~555MB | ~4GB RAM | Moderate | Excellent+ | High-memory systems |
| `t3_cfg-f16.gguf` | ~1.1GB | ~6GB RAM | Slowest | Best | High-end systems, maximum quality |

#### Hardware Recommendations

**For Raspberry Pi / Edge Devices (2-4GB RAM):**
```bash
# Use the lightest model
export CHATTERBOX_MODEL="t3_cfg-q4_k.gguf"
```

**For Standard Laptops/Desktops (8-16GB RAM):**
```bash
# Use the recommended balanced model (default)
export CHATTERBOX_MODEL="t3_cfg-q6_k.gguf"
```

**For High-End Workstations (32GB+ RAM):**
```bash
# Use the highest quality model
export CHATTERBOX_MODEL="t3_cfg-f16.gguf"
```

### Model Configuration

#### Environment Variable Method (Recommended)
Set the model before starting the MCP server:

```bash
# Set desired model
export CHATTERBOX_MODEL="t3_cfg-q6_k.gguf"

# Start the MCP server
python mcp_server.py
```

#### Manual Model Selection
You can also manually download and specify models:

```bash
# Download specific model
python -c "
from chatterbox_tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(model_id='t3_cfg-q4_k.gguf')
"

# Or download all models for switching
python -c "
from chatterbox_tts import ChatterboxTTS
for model_id in ['t3_cfg-q4_k.gguf', 't3_cfg-q6_k.gguf', 't3_cfg-q8_0.gguf']:
    ChatterboxTTS.from_pretrained(model_id=model_id)
"
```

#### Performance Testing
Test different models on your system to find the optimal balance:

```bash
# Test q4_k model
CHATTERBOX_MODEL="t3_cfg-q4_k.gguf" python -c "
from chatterbox_tts import ChatterboxTTS
import time
model = ChatterboxTTS.from_pretrained()
start = time.time()
wav = model.generate('Testing q4k model performance')
print(f'q4_k generation time: {time.time() - start:.2f}s')
"

# Test q6_k model  
CHATTERBOX_MODEL="t3_cfg-q6_k.gguf" python -c "
from chatterbox_tts import ChatterboxTTS
import time
model = ChatterboxTTS.from_pretrained()
start = time.time()
wav = model.generate('Testing q6k model performance')
print(f'q6_k generation time: {time.time() - start:.2f}s')
"
```

### Memory Requirements

- **q4_k**: Minimum 2GB RAM, 4GB recommended
- **q6_k**: Minimum 3GB RAM, 6GB recommended  
- **q8_0**: Minimum 4GB RAM, 8GB recommended
- **f16**: Minimum 6GB RAM, 12GB recommended

**Note**: These are rough estimates. Actual memory usage depends on text length, voice cloning complexity, and system overhead.

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