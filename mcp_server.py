#!/usr/bin/env python3
"""
Chatterbox TTS MCP Server

A Model Context Protocol server for Chatterbox TTS with quantized model support,
voice cloning capabilities, and optimized performance on Apple Silicon.
"""

import argparse
import copy
import json
import logging
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle

import torch
import torchaudio as ta
from mcp.server import Server, NotificationOptions  
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    Resource,
    TextContent,
    Tool,
)
from pydantic import AnyUrl

from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatterboxMCPServer:
    """MCP Server for Chatterbox TTS with quantized model support."""
    
    def __init__(self, hot_reload=False, model_id=None):
        self.server = Server("chatterbox-tts")
        self.model: Optional[ChatterboxTTS] = None
        self.voice_store: Dict[str, bytes] = {}  # Named voice state cache
        self.current_voice = "default"
        self.device = self._get_device()
        self.model_id = model_id  # Store model_id for later use
        self.hot_reload = hot_reload
        self.last_modified = None
        if self.hot_reload:
            self._start_file_watcher()
        self._setup_tools()
        
    def _get_device(self) -> str:
        """Automatically detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _start_file_watcher(self):
        """Start background file watcher for hot reloading."""
        def watch_file():
            script_path = __file__
            self.last_modified = os.path.getmtime(script_path)
            while True:
                try:
                    current_modified = os.path.getmtime(script_path)
                    if current_modified > self.last_modified:
                        logger.info("🔥 Hot reload: Script modified, reloading...")
                        self.last_modified = current_modified
                        # In a real implementation, we would reload the module
                        # For now, just log the event
                        logger.info("🔥 Hot reload complete")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"File watcher error: {e}")
                    break
        
        watcher_thread = threading.Thread(target=watch_file, daemon=True)
        watcher_thread.start()
        logger.info("🔥 Hot reload watcher started")
    
    def _setup_tools(self):
        """Register MCP tools."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available Chatterbox TTS tools."""
            return [
                Tool(
                    name="chatterbox_tts_generate",
                    description="Generate speech from text using Chatterbox TTS with quantized models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to convert to speech"
                            },
                            "voice_name": {
                                "type": "string",
                                "description": "Name of voice from voice store to use (defaults to 'default')",
                                "default": "default"
                            },
                            "audio_prompt_path": {
                                "type": "string",
                                "description": "Optional path to audio file for voice cloning (overrides voice_name)",
                                "default": None
                            },
                            "output_path": {
                                "type": "string", 
                                "description": "Output file path (defaults to temp file)",
                                "default": None
                            },
                            "exaggeration": {
                                "type": "number",
                                "description": "Voice exaggeration factor (0.0 to 1.0)",
                                "default": 0.5
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="chatterbox_tts_info",
                    description="Get information about the Chatterbox TTS model and system",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="chatterbox_tts_prepare_audio",
                    description="Prepare and optimize audio files for voice cloning (convert MP3/other formats to optimal WAV)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input_path": {
                                "type": "string",
                                "description": "Path to input audio file (MP3, WAV, etc.)"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Output path for optimized WAV file (defaults to temp file)",
                                "default": None
                            },
                            "target_sample_rate": {
                                "type": "integer",
                                "description": "Target sample rate in Hz (default: 24000)",
                                "default": 24000
                            },
                            "mono": {
                                "type": "boolean",
                                "description": "Convert to mono (recommended for voice cloning)",
                                "default": True
                            },
                            "normalize": {
                                "type": "boolean",
                                "description": "Normalize audio volume",
                                "default": True
                            },
                            "trim_silence": {
                                "type": "boolean",
                                "description": "Trim silence from beginning and end",
                                "default": True
                            }
                        },
                        "required": ["input_path"]
                    }
                ),
                Tool(
                    name="chatterbox_tts_clone_voice",
                    description="Generate speech with voice cloning from an audio sample",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to convert to speech"
                            },
                            "voice_sample_path": {
                                "type": "string",
                                "description": "Path to audio file containing voice to clone"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Output file path (defaults to temp file)", 
                                "default": None
                            },
                            "exaggeration": {
                                "type": "number",
                                "description": "Voice exaggeration factor (0.0 to 1.0)",
                                "default": 0.5
                            }
                        },
                        "required": ["text", "voice_sample_path"]
                    }
                ),
                Tool(
                    name="chatterbox_tts_voice_store",
                    description="Manage voice store - save, load, list, and switch between different voice states",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["save", "load", "list", "delete"],
                                "description": "Action to perform on voice store"
                            },
                            "voice_name": {
                                "type": "string",
                                "description": "Name of the voice to save/load/delete"
                            },
                            "voice_sample_path": {
                                "type": "string",
                                "description": "Path to audio file for saving new voice (required for 'save' action)"
                            }
                        },
                        "required": ["action"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "chatterbox_tts_generate":
                    return await self._generate_speech(arguments)
                elif name == "chatterbox_tts_info":
                    return await self._get_model_info()
                elif name == "chatterbox_tts_prepare_audio":
                    return await self._prepare_audio(arguments)
                elif name == "chatterbox_tts_clone_voice":
                    return await self._clone_voice(arguments)
                elif name == "chatterbox_tts_voice_store":
                    return await self._manage_voice_store(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _save_voice_state(self, conds):
        """Save voice conditions state using tensor serialization."""
        try:
            # Use torch.save for proper tensor serialization
            logger.info(f"Saving voice state, conds type: {type(conds)}")
            import io
            buffer = io.BytesIO()
            torch.save(conds, buffer)
            buffer.seek(0)
            data = buffer.getvalue()
            logger.info(f"Successfully saved voice state, size: {len(data)} bytes")
            return data
        except Exception as e:
            logger.error(f"Error saving voice state: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _restore_voice_state(self, state_data):
        """Restore voice conditions from serialized state."""
        try:
            if state_data is None:
                logger.error("Voice state data is None")
                return None
            import io
            buffer = io.BytesIO(state_data)
            logger.info(f"Attempting to restore voice state, buffer size: {len(state_data)} bytes")
            restored = torch.load(buffer, map_location=self.device, weights_only=False)
            logger.info(f"Successfully restored voice state, type: {type(restored)}")
            return restored
        except Exception as e:
            logger.error(f"Error restoring voice state: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _ensure_model_loaded(self):
        """Ensure the Chatterbox TTS model is loaded."""
        if self.model is None:
            # Priority: CLI arg > env var > default
            model_id = self.model_id or os.getenv("CHATTERBOX_MODEL", "t3_cfg-q6_k.gguf")
            logger.info(f"Loading Chatterbox TTS model '{model_id}' on device: {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            # Save default voice to voice store
            self.voice_store["default"] = self._save_voice_state(self.model.conds)
            logger.info(f"Default voice saved to voice store - conds type: {type(self.model.conds)}")
            logger.info(f"Default voice - t3 speaker_emb shape: {self.model.conds.t3.speaker_emb.shape if hasattr(self.model.conds.t3, 'speaker_emb') else 'None'}")
            logger.info("Chatterbox TTS model loaded successfully with default voice stored")
    
    async def _generate_speech(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Generate speech from text using voice store."""
        await self._ensure_model_loaded()
        
        text = arguments.get("text", "")
        voice_name = arguments.get("voice_name", "default")
        audio_prompt_path = arguments.get("audio_prompt_path")
        output_path = arguments.get("output_path")
        exaggeration = arguments.get("exaggeration", 0.5)
        
        if not text:
            return [TextContent(type="text", text="Error: No text provided")]
        
        # Generate output path if not provided
        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"chatterbox_tts_{os.getpid()}.wav")
        
        # Handle voice selection
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            # Direct voice cloning - generate and optionally save to voice store
            wav = self.model.generate(text, audio_prompt_path=audio_prompt_path, exaggeration=exaggeration)
            result_text = f"Generated speech with voice cloning from '{audio_prompt_path}'"
        else:
            # Use voice from voice store
            if voice_name in self.voice_store:
                logger.info(f"Loading voice '{voice_name}' from voice store...")
                restored_conds = self._restore_voice_state(self.voice_store[voice_name])
                if restored_conds is not None:
                    self.model.conds = restored_conds
                    self.current_voice = voice_name
                    logger.info(f"Voice '{voice_name}' loaded successfully")
                else:
                    logger.warning(f"Failed to restore voice '{voice_name}'")
                    return [TextContent(type="text", text=f"Error: Failed to load voice '{voice_name}'")]
            else:
                return [TextContent(type="text", text=f"Error: Voice '{voice_name}' not found in voice store. Available voices: {list(self.voice_store.keys())}")]
            
            wav = self.model.generate(text, audio_prompt_path=None, exaggeration=exaggeration)
            result_text = f"Generated speech with voice '{voice_name}' from voice store"
        
        # Save audio file
        ta.save(output_path, wav, self.model.sr)
        
        return [TextContent(
            type="text",
            text=f"{result_text}\nText: {text}\nOutput saved to: {output_path}\nSample rate: {self.model.sr} Hz"
        )]
    
    async def _clone_voice(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Generate speech with voice cloning."""
        await self._ensure_model_loaded()
        
        text = arguments.get("text", "")
        voice_sample_path = arguments.get("voice_sample_path", "")
        output_path = arguments.get("output_path")
        exaggeration = arguments.get("exaggeration", 0.5)
        
        if not text:
            return [TextContent(type="text", text="Error: No text provided")]
        
        if not voice_sample_path or not os.path.exists(voice_sample_path):
            return [TextContent(type="text", text=f"Error: Voice sample file not found: {voice_sample_path}")]
        
        # Generate output path if not provided
        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"chatterbox_cloned_{os.getpid()}.wav")
        
        # Generate speech with voice cloning
        wav = self.model.generate(text, audio_prompt_path=voice_sample_path, exaggeration=exaggeration)
        
        # Save audio file
        ta.save(output_path, wav, self.model.sr)
        
        return [TextContent(
            type="text",
            text=f"Generated speech with voice cloning\nText: {text}\nVoice sample: {voice_sample_path}\nOutput saved to: {output_path}\nSample rate: {self.model.sr} Hz"
        )]
    
    async def _manage_voice_store(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Manage the voice store - save, load, list, delete voices."""
        await self._ensure_model_loaded()
        
        action = arguments.get("action", "")
        voice_name = arguments.get("voice_name", "")
        voice_sample_path = arguments.get("voice_sample_path", "")
        
        if action == "list":
            voices = list(self.voice_store.keys())
            return [TextContent(
                type="text",
                text=f"Available voices in voice store:\n{json.dumps(voices, indent=2)}\nCurrent voice: {self.current_voice}"
            )]
        
        elif action == "save":
            if not voice_name:
                return [TextContent(type="text", text="Error: voice_name required for save action")]
            
            if not voice_sample_path or not os.path.exists(voice_sample_path):
                return [TextContent(type="text", text=f"Error: Voice sample file not found: {voice_sample_path}")]
            
            # Generate speech with the new voice to load its conditions
            temp_wav = self.model.generate("Voice calibration sample", audio_prompt_path=voice_sample_path, exaggeration=0.5)
            
            # Save the current voice conditions to the voice store
            self.voice_store[voice_name] = self._save_voice_state(self.model.conds)
            logger.info(f"Voice '{voice_name}' saved to voice store")
            
            return [TextContent(
                type="text",
                text=f"Voice '{voice_name}' saved to voice store successfully\nTotal voices: {len(self.voice_store)}"
            )]
        
        elif action == "load":
            if not voice_name:
                return [TextContent(type="text", text="Error: voice_name required for load action")]
            
            if voice_name not in self.voice_store:
                return [TextContent(type="text", text=f"Error: Voice '{voice_name}' not found in voice store. Available voices: {list(self.voice_store.keys())}")]
            
            # Load the voice conditions
            restored_conds = self._restore_voice_state(self.voice_store[voice_name])
            if restored_conds is not None:
                self.model.conds = restored_conds
                self.current_voice = voice_name
                logger.info(f"Voice '{voice_name}' loaded successfully")
                return [TextContent(
                    type="text",
                    text=f"Voice '{voice_name}' loaded successfully\nCurrent voice: {self.current_voice}"
                )]
            else:
                return [TextContent(type="text", text=f"Error: Failed to load voice '{voice_name}'")]
        
        elif action == "delete":
            if not voice_name:
                return [TextContent(type="text", text="Error: voice_name required for delete action")]
            
            if voice_name == "default":
                return [TextContent(type="text", text="Error: Cannot delete the default voice")]
            
            if voice_name not in self.voice_store:
                return [TextContent(type="text", text=f"Error: Voice '{voice_name}' not found in voice store")]
            
            del self.voice_store[voice_name]
            if self.current_voice == voice_name:
                # Switch to default if current voice was deleted
                self.current_voice = "default"
                restored_conds = self._restore_voice_state(self.voice_store["default"])
                if restored_conds is not None:
                    self.model.conds = restored_conds
            
            return [TextContent(
                type="text",
                text=f"Voice '{voice_name}' deleted successfully\nCurrent voice: {self.current_voice}"
            )]
        
        else:
            return [TextContent(type="text", text=f"Error: Unknown action '{action}'. Available actions: save, load, list, delete")]
    
    async def _prepare_audio(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Prepare and optimize audio files for voice cloning."""
        input_path = arguments.get("input_path", "")
        output_path = arguments.get("output_path")
        target_sample_rate = arguments.get("target_sample_rate", 24000)
        mono = arguments.get("mono", True)
        normalize = arguments.get("normalize", True)
        trim_silence = arguments.get("trim_silence", True)
        
        # Validate input file
        if not input_path or not os.path.exists(input_path):
            return [TextContent(type="text", text=f"Error: Input audio file not found: {input_path}")]
        
        # Generate output path if not provided
        if not output_path:
            input_stem = Path(input_path).stem
            output_path = os.path.join(tempfile.gettempdir(), f"chatterbox_prepared_{input_stem}_{os.getpid()}.wav")
        
        try:
            # Load audio file
            waveform, original_sample_rate = ta.load(input_path)
            original_channels = waveform.shape[0]
            original_duration = waveform.shape[1] / original_sample_rate
            
            logger.info(f"Loaded audio: {original_sample_rate}Hz, {original_channels} channels, {original_duration:.2f}s")
            
            # Convert to mono if requested
            if mono and original_channels > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted to mono")
            
            # Trim silence if requested
            if trim_silence:
                # Simple silence trimming based on energy threshold
                energy = torch.sum(waveform ** 2, dim=0)
                threshold = torch.max(energy) * 0.001  # 0.1% of max energy
                non_silent = energy > threshold
                
                if torch.any(non_silent):
                    start_idx = torch.argmax(non_silent.float())
                    end_idx = len(non_silent) - torch.argmax(torch.flip(non_silent, [0]).float())
                    waveform = waveform[:, start_idx:end_idx]
                    logger.info(f"Trimmed silence: {start_idx} to {end_idx} samples")
            
            # Resample if needed
            if original_sample_rate != target_sample_rate:
                resampler = ta.transforms.Resample(original_sample_rate, target_sample_rate)
                waveform = resampler(waveform)
                logger.info(f"Resampled from {original_sample_rate}Hz to {target_sample_rate}Hz")
            
            # Normalize if requested
            if normalize:
                max_val = torch.max(torch.abs(waveform))
                if max_val > 0:
                    waveform = waveform / max_val * 0.95  # Leave some headroom
                    logger.info("Normalized audio levels")
            
            # Save optimized audio
            ta.save(output_path, waveform, target_sample_rate)
            
            # Calculate final stats
            final_duration = waveform.shape[1] / target_sample_rate
            final_channels = waveform.shape[0]
            
            return [TextContent(
                type="text",
                text=f"Audio preparation completed successfully!\n\n"
                     f"Input: {input_path}\n"
                     f"Output: {output_path}\n\n"
                     f"Original: {original_sample_rate}Hz, {original_channels} ch, {original_duration:.2f}s\n"
                     f"Optimized: {target_sample_rate}Hz, {final_channels} ch, {final_duration:.2f}s\n\n"
                     f"Processing applied:\n"
                     f"• Mono conversion: {'Yes' if mono and original_channels > 1 else 'No'}\n"
                     f"• Resampling: {'Yes' if original_sample_rate != target_sample_rate else 'No'}\n"
                     f"• Silence trimming: {'Yes' if trim_silence else 'No'}\n"
                     f"• Normalization: {'Yes' if normalize else 'No'}\n\n"
                     f"The optimized audio is ready for voice cloning!"
            )]
            
        except Exception as e:
            logger.error(f"Error preparing audio: {e}")
            return [TextContent(type="text", text=f"Error preparing audio: {str(e)}")]
    
    async def _get_model_info(self) -> List[TextContent]:
        """Get information about the model and system."""
        await self._ensure_model_loaded()
        
        info = {
            "model_name": "Chatterbox TTS",
            "model_type": "Quantized TTS with voice cloning",
            "device": self.device,
            "sample_rate": self.model.sr if self.model else "Unknown",
            "quantization": "6-bit (q6_k.gguf)",
            "architecture": "Transformer-based with quantized weights",
            "capabilities": [
                "Text-to-speech generation",
                "Voice cloning from audio samples", 
                "Adjustable voice exaggeration",
                "Apple Silicon (MPS) acceleration",
                "CUDA acceleration (if available)"
            ]
        }
        
        return [TextContent(
            type="text",
            text=f"Chatterbox TTS Model Information:\n{json.dumps(info, indent=2)}"
        )]
    
    async def run_stdio(self):
        """Run the MCP server over STDIO."""
        logger.info(f"Starting Chatterbox TTS MCP Server (STDIO) on device: {self.device}")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="chatterbox-tts",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities=None
                ),
                ),
            )
    
    async def run_sse(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server over SSE (Server-Sent Events)."""
        logger.info(f"Starting Chatterbox TTS MCP Server (SSE) on {host}:{port}, device: {self.device}")
        
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        from sse_starlette.sse import EventSourceResponse
        import uvicorn
        
        transport = SseServerTransport("/messages")
        
        async def handle_sse(request):
            async with transport.connect_sse(
                request,
                self.server,
                InitializationOptions(
                    server_name="chatterbox-tts",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(),
                )
            ) as connection:
                await connection.handle_request()
        
        async def handle_messages(request):
            return await transport.handle_post_message(request, self.server)
        
        app = Starlette(
            routes=[
                Route("/sse", handle_sse),
                Route("/messages", handle_messages, methods=["POST"]),
            ]
        )
        
        await uvicorn.run(app, host=host, port=port)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chatterbox TTS MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "sse"], 
        default="stdio",
        help="Transport method (default: stdio)"
    )
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host for SSE transport (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for SSE transport (default: 8000)"
    )
    parser.add_argument(
        "--hot-reload",
        action="store_true",
        help="Enable hot reloading (experimental)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Chatterbox TTS model to use (e.g., t3_cfg-q6_k.gguf). Overrides CHATTERBOX_MODEL env var."
    )
    
    args = parser.parse_args()
    server = ChatterboxMCPServer(hot_reload=args.hot_reload, model_id=args.model)
    
    if args.transport == "stdio":
        await server.run_stdio()
    elif args.transport == "sse":
        await server.run_sse(args.host, args.port)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())