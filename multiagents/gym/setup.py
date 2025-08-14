#!/usr/bin/env python3
"""
Setup script for AI Gym Assistant
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing dependencies...")
    subprocess.check_call(["uv", "pip", "install", "-r", "requirements.txt"])

def setup_ollama():
    """Setup Ollama for local LLM"""
    print("\nSetting up Ollama...")
    print("1. Install Ollama from https://ollama.com")
    print("2. Run: ollama pull qwen3:8b")
    print("3. Start Ollama service")

def setup_google_ai():
    """Setup Google AI"""
    print("\nSetting up Google AI...")
    print("1. Get API key from https://makersuite.google.com/app/apikey")
    print("2. Set environment variable: export GOOGLE_API_KEY=your_key_here")

def main():
    """Main setup function"""
    print("🏋️ AI Gym Assistant Setup")
    print("=" * 30)
    
    try:
        install_requirements()
        
        print("\n📋 Configuration Options:")
        print("1. For local LLM (Ollama + qwen3:8b) - Default")
        print("2. For Google AI (Gemini)")
        
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            setup_ollama()
            print("\n✅ To use Ollama, ensure it's running and has qwen2.5:3b model")
        elif choice == "2":
            setup_google_ai()
            print("\n✅ To use Google AI, set your API key in environment variables")
        else:
            print("Invalid choice. Defaulting to Ollama setup.")
            setup_ollama()
        
        print("\n🚀 Setup complete! Run the application with:")
        print("python main.py")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 