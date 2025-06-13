#!/usr/bin/env python3
"""
Chatterbox TTS MCP Server

A Model Context Protocol server for Chatterbox TTS with quantized model support,
voice cloning capabilities, and optimized performance on Apple Silicon.
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    
    def __init__(self):
        self.server = Server("chatterbox-tts")
        self.model: Optional[ChatterboxTTS] = None
        self.device = self._get_device()
        self._setup_tools()
        
    def _get_device(self) -> str:
        """Automatically detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
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
                            "audio_prompt_path": {
                                "type": "string",
                                "description": "Optional path to audio file for voice cloning",
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
                elif name == "chatterbox_tts_clone_voice":
                    return await self._clone_voice(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _ensure_model_loaded(self):
        """Ensure the Chatterbox TTS model is loaded."""
        if self.model is None:
            logger.info(f"Loading Chatterbox TTS model on device: {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("Chatterbox TTS model loaded successfully")
    
    async def _generate_speech(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Generate speech from text."""
        await self._ensure_model_loaded()
        
        text = arguments.get("text", "")
        audio_prompt_path = arguments.get("audio_prompt_path")
        output_path = arguments.get("output_path")
        exaggeration = arguments.get("exaggeration", 0.5)
        
        if not text:
            return [TextContent(type="text", text="Error: No text provided")]
        
        # Generate output path if not provided
        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"chatterbox_tts_{os.getpid()}.wav")
        
        # Generate speech
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            wav = self.model.generate(text, audio_prompt_path=audio_prompt_path, exaggeration=exaggeration)
            result_text = f"Generated speech with voice cloning from '{audio_prompt_path}'"
        else:
            wav = self.model.generate(text, exaggeration=exaggeration)
            result_text = "Generated speech with default voice"
        
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
    
    args = parser.parse_args()
    server = ChatterboxMCPServer()
    
    if args.transport == "stdio":
        await server.run_stdio()
    elif args.transport == "sse":
        await server.run_sse(args.host, args.port)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())