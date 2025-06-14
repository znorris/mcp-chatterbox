#!/usr/bin/env python3
"""
Performance testing script for Chatterbox TTS
Measures generation times under different conditions
"""

import time
import asyncio
import tempfile
import os
from pathlib import Path
import statistics

# Test text samples
SHORT_TEXT = "Hello world"
MEDIUM_TEXT = "I was generated locally on a macbook pro using Chatterbox with the default voice."
LONG_TEXT = """
Machine learning has revolutionized the field of artificial intelligence, 
enabling computers to learn and make decisions without explicit programming.
Deep neural networks, inspired by the human brain, can process vast amounts
of data and identify complex patterns that were previously impossible to detect.
This breakthrough technology now powers everything from image recognition
and natural language processing to autonomous vehicles and medical diagnosis.
"""

def time_function(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        return result, duration
    return wrapper

async def main():
    """Run comprehensive performance tests"""
    print("🚀 Starting Chatterbox TTS Performance Tests")
    print("=" * 60)
    
    # Import the server class
    from mcp_server import ChatterboxMCPServer
    
    # Create server instance
    server = ChatterboxMCPServer()
    
    results = {
        'cold_start': {},
        'warm_start': {},
        'text_length': {},
        'voice_cloning': {},
        'exaggeration': {}
    }
    
    # Test 1: Cold Start (includes model loading)
    print("\n📊 Test 1: Cold Start Performance")
    print("-" * 40)
    
    start_time = time.time()
    await server._generate_speech({
        'text': SHORT_TEXT,
        'output_path': './performance_tests/cold_start.wav'
    })
    cold_start_time = time.time() - start_time
    
    results['cold_start']['time'] = cold_start_time
    print(f"Cold start time (with model loading): {cold_start_time:.2f}s")
    
    # Test 2: Warm Start (multiple generations)
    print("\n🔥 Test 2: Warm Start Performance (5 runs)")
    print("-" * 40)
    
    warm_times = []
    for i in range(5):
        start_time = time.time()
        await server._generate_speech({
            'text': SHORT_TEXT,
            'output_path': f'./performance_tests/warm_start_{i}.wav'
        })
        warm_time = time.time() - start_time
        warm_times.append(warm_time)
        print(f"Run {i+1}: {warm_time:.2f}s")
    
    results['warm_start']['times'] = warm_times
    results['warm_start']['average'] = statistics.mean(warm_times)
    results['warm_start']['min'] = min(warm_times)
    results['warm_start']['max'] = max(warm_times)
    
    print(f"Average warm start time: {results['warm_start']['average']:.2f}s")
    print(f"Range: {results['warm_start']['min']:.2f}s - {results['warm_start']['max']:.2f}s")
    
    # Test 3: Different Text Lengths
    print("\n📝 Test 3: Text Length Performance")
    print("-" * 40)
    
    text_tests = [
        ('short', SHORT_TEXT),
        ('medium', MEDIUM_TEXT),
        ('long', LONG_TEXT.strip())
    ]
    
    for name, text in text_tests:
        start_time = time.time()
        await server._generate_speech({
            'text': text,
            'output_path': f'./performance_tests/length_{name}.wav'
        })
        gen_time = time.time() - start_time
        
        # Estimate audio duration (rough calculation: ~150 words per minute)
        word_count = len(text.split())
        estimated_audio_duration = (word_count / 150) * 60  # seconds
        
        results['text_length'][name] = {
            'generation_time': gen_time,
            'word_count': word_count,
            'estimated_audio_duration': estimated_audio_duration,
            'real_time_factor': gen_time / max(estimated_audio_duration, 0.1)
        }
        
        print(f"{name.capitalize()} text ({word_count} words): {gen_time:.2f}s")
        print(f"  Estimated audio duration: {estimated_audio_duration:.1f}s")
        print(f"  Real-time factor: {results['text_length'][name]['real_time_factor']:.2f}x")
    
    # Test 4: Voice Cloning Performance
    print("\n🎭 Test 4: Voice Cloning Performance")
    print("-" * 40)
    
    # Test with Tim Curry voice if available
    tim_curry_path = './audio/original/CLUE_Movie_monkey_brains.mp3'
    if os.path.exists(tim_curry_path):
        start_time = time.time()
        await server._clone_voice({
            'text': MEDIUM_TEXT,
            'voice_sample_path': tim_curry_path,
            'output_path': './performance_tests/voice_clone.wav'
        })
        clone_time = time.time() - start_time
        
        results['voice_cloning']['time'] = clone_time
        print(f"Voice cloning time: {clone_time:.2f}s")
        
        # Compare with default voice
        start_time = time.time()
        await server._generate_speech({
            'text': MEDIUM_TEXT,
            'output_path': './performance_tests/default_voice.wav'
        })
        default_time = time.time() - start_time
        
        results['voice_cloning']['default_time'] = default_time
        results['voice_cloning']['overhead'] = clone_time - default_time
        
        print(f"Default voice time: {default_time:.2f}s")
        print(f"Voice cloning overhead: {results['voice_cloning']['overhead']:.2f}s")
    else:
        print(f"Voice sample not found: {tim_curry_path}")
        results['voice_cloning']['error'] = "Sample file not found"
    
    # Test 5: Exaggeration Settings
    print("\n🎚️  Test 5: Exaggeration Settings Performance")
    print("-" * 40)
    
    exaggeration_levels = [0.0, 0.5, 1.0]
    for level in exaggeration_levels:
        start_time = time.time()
        await server._generate_speech({
            'text': MEDIUM_TEXT,
            'exaggeration': level,
            'output_path': f'./performance_tests/exag_{level}.wav'
        })
        exag_time = time.time() - start_time
        
        results['exaggeration'][f'level_{level}'] = exag_time
        print(f"Exaggeration {level}: {exag_time:.2f}s")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"\n🥶 Cold Start (first generation): {results['cold_start']['time']:.2f}s")
    print(f"🔥 Warm Start (average): {results['warm_start']['average']:.2f}s")
    print(f"⚡ Speed improvement: {results['cold_start']['time'] / results['warm_start']['average']:.1f}x faster after warmup")
    
    print(f"\n📊 Text Length Analysis:")
    for name, data in results['text_length'].items():
        print(f"  {name.capitalize()}: {data['generation_time']:.2f}s ({data['real_time_factor']:.1f}x real-time)")
    
    if 'time' in results['voice_cloning']:
        print(f"\n🎭 Voice Cloning:")
        print(f"  Default voice: {results['voice_cloning']['default_time']:.2f}s")
        print(f"  Voice cloning: {results['voice_cloning']['time']:.2f}s")
        print(f"  Overhead: {results['voice_cloning']['overhead']:.2f}s")
    
    print(f"\n🎚️  Exaggeration Impact:")
    for level, time_val in results['exaggeration'].items():
        print(f"  {level}: {time_val:.2f}s")
    
    # Live Call Assessment
    print(f"\n" + "=" * 60)
    print("🎯 LIVE CALL SUITABILITY ASSESSMENT")
    print("=" * 60)
    
    avg_warm_time = results['warm_start']['average']
    short_text_rtf = results['text_length']['short']['real_time_factor']
    
    print(f"\nKey Metrics:")
    print(f"• Average warm generation time: {avg_warm_time:.2f}s")
    print(f"• Short text real-time factor: {short_text_rtf:.1f}x")
    print(f"• Cold start penalty: {results['cold_start']['time'] - avg_warm_time:.1f}s")
    
    # Live call verdict
    if avg_warm_time < 2.0 and short_text_rtf < 3.0:
        verdict = "✅ SUITABLE for live calls"
        details = "Fast enough for real-time interaction"
    elif avg_warm_time < 5.0:
        verdict = "⚠️  MARGINAL for live calls"
        details = "May work for slower-paced conversations"
    else:
        verdict = "❌ NOT SUITABLE for live calls"
        details = "Better suited for content creation"
    
    print(f"\n{verdict}")
    print(f"Assessment: {details}")
    
    print(f"\nRecommendations:")
    if avg_warm_time < 2.0:
        print("• Suitable for live calls, chatbots, and interactive applications")
        print("• Pre-warm the model to avoid cold start delays")
    print("• Excellent for content creation and batch processing")
    print("• Voice cloning adds minimal overhead for pre-saved voices")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())