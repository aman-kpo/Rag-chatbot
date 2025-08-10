#!/usr/bin/env python3
"""
Launcher script for the Law RAG Chatbot FastAPI application
"""

import subprocess
import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    missing_vars = []
    
    if not os.getenv('HF_TOKEN'):
        missing_vars.append('HF_TOKEN')
    
    if not os.getenv('GROQ_API_KEY'):
        missing_vars.append('GROQ_API_KEY')
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables:")
        print("Windows:")
        print(f"   set {var}=your_value_here")
        print("Linux/Mac:")
        print(f"   export {var}=your_value_here")
        return False
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import langchain
        import groq
        import sentence_transformers
        import chromadb
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def main():
    """Main launcher function"""
    print("🚀 Law RAG Chatbot API Launcher")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("✅ Environment and dependencies ready")
    print("\n🌐 Starting FastAPI server...")
    print(f"   URL: http://localhost:8000")
    print(f"   API Docs: http://localhost:8000/docs")
    print(f"   Health Check: http://localhost:8000/health")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Start the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 