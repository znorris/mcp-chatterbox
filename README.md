# MCP Chatterbox TTS

A Claude Code MCP server that adds text-to-speech and voice cloning capabilities to your Claude sessions.

## What This Does

- **Generate Speech**: Convert any text to high-quality audio
- **Voice Cloning**: Clone voices from audio samples in seconds
- **Voice Library**: Save and switch between multiple cloned voices
- **Smart Audio Processing**: Automatically optimize audio files for best results

## Quick Start for Claude Code Users

### 1. Install the Server

```bash
# Clone and set up the project
git clone https://github.com/znorris/mcp-chatterbox.git
cd mcp-chatterbox

# Create isolated environment and install
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### 2. Add to Claude Code

Edit your `~/.claude.json` file to add this server:

**macOS/Linux:**
```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "command": "/path/to/mcp-chatterbox/venv/bin/python",
      "args": ["/path/to/mcp-chatterbox/mcp_server.py"]
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "chatterbox-tts": {
      "command": "C:\\path\\to\\mcp-chatterbox\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\mcp-chatterbox\\mcp_server.py"]
    }
  }
}
```

> **💡 Tip**: Replace `/path/to/mcp-chatterbox` with your actual project location. Use `pwd` (macOS/Linux) or `cd` (Windows) to find your current path.

### 3. Restart Claude Code

Restart Claude Code to load the TTS server. You'll see "chatterbox-tts" in your available tools.

### 4. First Use

On first use, the system will automatically download AI models (~500MB). This happens once and takes 2-3 minutes depending on your internet speed.

## How to Use

### Basic Text-to-Speech

Ask Claude to generate speech from any text:

> "Generate speech for: Hello, this is my first TTS sample!"

### Voice Cloning  

1. **Prepare an audio file** (MP3 or WAV, 7-20 seconds of clear speech)
2. **Clone the voice**: "Clone this voice from my audio file at `/path/to/voice_sample.wav` and say 'Hello world'"
3. **Save for reuse**: "Save this voice as 'narrator' in the voice store"

### Voice Library Management

- **List voices**: "Show me all voices in the voice store"
- **Use saved voice**: "Generate speech using the 'narrator' voice: Welcome to my podcast"
- **Delete voice**: "Remove the 'narrator' voice from the store"

## Model Selection (Optional)

Choose the right model for your system:

| Your System | Recommended Model | Setup |
|-------------|------------------|-------|
| MacBook Air, basic laptop | `q4_k` (fastest) | `export CHATTERBOX_MODEL="t3_cfg-q4_k.gguf"` |
| Standard laptop/desktop | `q6_k` (default) | No setup needed |
| High-end system | `f16` (best quality) | `export CHATTERBOX_MODEL="t3_cfg-f16.gguf"` |

Set the environment variable before starting Claude Code to use a different model.

## Audio Quality Tips

For best voice cloning results, use audio that is:
- **7-20 seconds long** (sweet spot for quality)
- **Single speaker only** (no background voices)
- **Clear recording** (minimal background noise)
- **Consistent volume** (not too quiet or distorted)

The server includes an audio preparation tool that automatically optimizes your files.

## Troubleshooting

**"ModuleNotFoundError" when starting Claude Code:**
- Ensure you activated the virtual environment before installing: `source venv/bin/activate`
- Check that the Python path in `~/.claude.json` points to the venv Python

**Models downloading slowly:**
- First-time setup downloads ~500MB of AI models
- This is normal and only happens once
- Subsequent starts are fast (2-3 seconds)

**Voice cloning sounds distorted:**
- Use the audio preparation tool: "Prepare this audio file for voice cloning: `/path/to/audio.mp3`"
- Ensure your source audio is clean and clear
- Try a longer audio sample (10+ seconds)

**Server not appearing in Claude Code:**
- Double-check the paths in `~/.claude.json` match your installation
- Ensure you restarted Claude Code after editing the config
- Check the server logs for error messages

## Available Tools

When you ask Claude to work with audio, it will automatically use these tools:

- **`chatterbox_tts_generate`**: Create speech from text
- **`chatterbox_tts_clone_voice`**: Clone a voice from audio sample  
- **`chatterbox_tts_voice_store`**: Manage your voice library
- **`chatterbox_tts_prepare_audio`**: Optimize audio files for cloning
- **`chatterbox_tts_info`**: Check system status and capabilities

You don't need to remember these names - just ask Claude in natural language what you want to do with audio.

## Advanced Configuration

### Custom Model Selection
```bash
# Use a specific model for your session
python mcp_server.py --model t3_cfg-q8_0.gguf

# Enable development hot-reload
python mcp_server.py --hot-reload --model t3_cfg-f16.gguf
```

### Memory Requirements
- **q4_k**: 2GB+ RAM (basic systems)
- **q6_k**: 3GB+ RAM (recommended default) 
- **q8_0**: 4GB+ RAM (high quality)
- **f16**: 6GB+ RAM (maximum quality)

### Performance Testing
Test different models on your system:
```bash
# Compare model performance
CHATTERBOX_MODEL="t3_cfg-q4_k.gguf" python -c "
from chatterbox_tts import ChatterboxTTS
import time
model = ChatterboxTTS.from_pretrained()
start = time.time()
wav = model.generate('Performance test')
print(f'q4_k: {time.time() - start:.2f}s')
"
```

## Technical Details

- **Models**: Uses quantized AI models for efficient memory usage
- **Hardware**: Optimized for Apple Silicon (MPS) and NVIDIA GPUs (CUDA)
- **Audio**: 24kHz sample rate, supports WAV and MP3 input/output
- **Watermarking**: All generated audio includes imperceptible watermarks for authenticity

## License & Credits

MIT License - Built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI.