#!/usr/bin/env python3
"""Test script for the Chatterbox MCP server."""

import asyncio
import json
import tempfile
from mcp_server import ChatterboxMCPServer

async def test_server():
    print("🎙️  Testing Chatterbox MCP Server...")
    server = ChatterboxMCPServer()
    
    # Test getting model info first (this will load the model)
    print("\n📊 Getting model information...")
    try:
        info_result = await server._get_model_info()
        for content in info_result:
            print(content.text)
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return
    
    # Test generating fun speech
    print("\n🎭 Generating fun speech...")
    fun_texts = [
        "Hello! I'm an AI voice speaking through Chatterbox TTS. This is absolutely magical!",
        "Welcome to the world of quantized neural text-to-speech synthesis. Isn't technology amazing?",
        "I can speak with various levels of exaggeration. This is me being quite dramatic!"
    ]
    
    for i, text in enumerate(fun_texts, 1):
        print(f"\n🎵 Generating speech sample {i}...")
        try:
            result = await server._generate_speech({
                'text': text,
                'exaggeration': 0.6 + (i * 0.1)  # Increasing exaggeration
            })
            
            for content in result:
                print(content.text)
                
        except Exception as e:
            print(f"❌ Error generating speech {i}: {e}")
            continue
    
    print("\n✅ MCP Server test completed!")

if __name__ == '__main__':
    asyncio.run(test_server())