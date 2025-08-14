#!/usr/bin/env python3
"""
Text-to-Speech Tool for A-Modular-Kingdom
Supports multiple TTS engines for CLI-compatible audio output
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Literal
import json

def clear_proxy_settings():
    """Clear proxy settings for TTS operations"""
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

class TTSEngine:
    """Text-to-Speech engine with multiple backend support"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "amk_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
    def speak(
        self, 
        text: str, 
        engine: Literal["pyttsx3", "gtts", "kokoro"] = "pyttsx3",
        voice: Optional[str] = None,
        speed: float = 200,
        play_audio: bool = True
    ) -> dict:
        """
        Convert text to speech and optionally play it
        
        Args:
            text: Text to convert to speech
            engine: TTS engine to use ("pyttsx3", "gtts", "kokoro")
            voice: Voice ID/name (engine-specific)
            speed: Speech rate (pyttsx3 only)
            play_audio: Whether to play audio immediately
            
        Returns:
            dict with success status, file path, and error if any
        """
        try:
            if engine == "pyttsx3":
                return self._pyttsx3_speak(text, voice, speed, play_audio)
            elif engine == "gtts":
                return self._gtts_speak(text, play_audio)
            elif engine == "kokoro":
                return self._kokoro_speak(text, voice, play_audio)
            else:
                return {"success": False, "error": f"Unknown engine: {engine}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _pyttsx3_speak(self, text: str, voice: Optional[str], speed: float, play_audio: bool) -> dict:
        """Use pyttsx3 for local TTS"""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Set voice if specified
            if voice:
                voices = engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            
            # Set speed
            engine.setProperty('rate', speed)
            
            if play_audio:
                engine.say(text)
                engine.runAndWait()
                return {"success": True, "method": "direct_playback", "engine": "pyttsx3"}
            else:
                # Save to file
                output_path = self.temp_dir / f"tts_output_{hash(text)}.wav"
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                return {"success": True, "file_path": str(output_path), "engine": "pyttsx3"}
                
        except ImportError:
            return {"success": False, "error": "pyttsx3 not installed"}
    
    def _gtts_speak(self, text: str, play_audio: bool) -> dict:
        """Use gTTS for cloud-based TTS"""
        try:
            from gtts import gTTS
            import pygame
            
            # Generate audio file
            tts = gTTS(text=text, lang='en')
            output_path = self.temp_dir / f"gtts_output_{hash(text)}.mp3"
            tts.save(str(output_path))
            
            if play_audio:
                # Play using pygame
                pygame.mixer.init()
                pygame.mixer.music.load(str(output_path))
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
                return {"success": True, "method": "pygame_playback", "engine": "gtts"}
            else:
                return {"success": True, "file_path": str(output_path), "engine": "gtts"}
                
        except ImportError as e:
            return {"success": False, "error": f"Missing dependency: {e}"}
    
    def _kokoro_speak(self, text: str, voice: Optional[str], play_audio: bool) -> dict:
        """Use Kokoro TTS (if available)"""
        kokoro_script = "/home/masih/My_freelance_website/scripts/kokoro_tts.py"
        
        if not os.path.exists(kokoro_script):
            return {"success": False, "error": "Kokoro TTS script not found"}
        
        try:
            output_path = self.temp_dir / f"kokoro_output_{hash(text)}.wav"
            
            # Call Kokoro script
            cmd = [
                "python", kokoro_script,
                "--text", text,
                "--output", str(output_path)
            ]
            
            if voice:
                cmd.extend(["--voice", voice])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"success": False, "error": f"Kokoro failed: {result.stderr}"}
            
            if play_audio and os.path.exists(output_path):
                # Play using system audio player
                if os.name == 'posix':  # Linux/Mac
                    subprocess.run(["aplay", str(output_path)], check=False)
                elif os.name == 'nt':   # Windows
                    subprocess.run(["start", str(output_path)], shell=True, check=False)
                
                return {"success": True, "method": "system_playback", "engine": "kokoro"}
            else:
                return {"success": True, "file_path": str(output_path), "engine": "kokoro"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_voices(self, engine: str = "pyttsx3") -> dict:
        """List available voices for the specified engine"""
        try:
            if engine == "pyttsx3":
                import pyttsx3
                engine_obj = pyttsx3.init()
                voices = engine_obj.getProperty('voices')
                return {
                    "success": True, 
                    "voices": [{"id": v.id, "name": v.name, "languages": v.languages} for v in voices]
                }
            else:
                return {"success": False, "error": f"Voice listing not supported for {engine}"}
                
        except ImportError:
            return {"success": False, "error": f"{engine} not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Main interface for MCP integration
def text_to_speech(text: str, **kwargs) -> str:
    """
    MCP tool interface for text-to-speech
    
    Args:
        text: Text to convert to speech
        **kwargs: Additional parameters for TTS engine
        
    Returns:
        JSON string with operation result
    """
    tts = TTSEngine()
    result = tts.speak(text, **kwargs)
    return json.dumps(result, indent=2)

def list_tts_voices(engine: str = "pyttsx3") -> str:
    """
    MCP tool interface for listing available voices
    
    Args:
        engine: TTS engine to query
        
    Returns:
        JSON string with available voices
    """
    tts = TTSEngine()
    result = tts.list_voices(engine)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    # Test the TTS system
    tts = TTSEngine()
    
    print("🔊 Testing A-Modular-Kingdom TTS System...")
    
    # Test pyttsx3
    print("\nTesting pyttsx3...")
    result = tts.speak("Hello from your A-Modular-Kingdom TTS system!", engine="pyttsx3")
    print(f"Result: {result}")
    
    # List voices
    print("\nAvailable voices:")
    voices = tts.list_voices()
    print(voices)