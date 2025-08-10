#!/usr/bin/env python3
"""
Setup script for Law RAG Chatbot
Helps configure environment variables and test the system
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'langchain', 'langchain-groq', 
        'sentence-transformers', 'chromadb', 'datasets', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def setup_environment():
    """Set up environment variables"""
    print("\n🔧 Setting up environment variables...")
    
    env_file = Path('.env')
    
    if env_file.exists():
        print("📁 .env file already exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if 'HF_TOKEN' in content and 'GROQ_API_KEY' in content:
                print("✅ Environment variables are configured")
                return True
            else:
                print("⚠️  .env file exists but missing required variables")
    
    print("📝 Creating .env file...")
    
    # Get Hugging Face token
    print("\n🤗 Hugging Face Token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with read permissions")
    hf_token = input("Enter your Hugging Face token: ").strip()
    
    if not hf_token:
        print("❌ Hugging Face token is required")
        return False
    
    # Get Groq API key
    print("\n🚀 Groq API Key:")
    print("1. Go to https://console.groq.com/")
    print("2. Sign up and get your API key")
    groq_key = input("Enter your Groq API key: ").strip()
    
    if not groq_key:
        print("❌ Groq API key is required")
        return False
    
    # Create .env file
    env_content = f"""# Law RAG Chatbot Environment Variables
HF_TOKEN={hf_token}
GROQ_API_KEY={groq_key}

# Optional: Customize these settings
# HOST=0.0.0.0
# PORT=8000
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully")
    
    # Set environment variables for current session
    os.environ['HF_TOKEN'] = hf_token
    os.environ['GROQ_API_KEY'] = groq_key
    
    return True

def test_api_keys():
    """Test if the API keys are valid"""
    print("\n🧪 Testing API keys...")
    
    # Test Hugging Face token
    try:
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            response = requests.get(
                'https://huggingface.co/api/datasets/ymoslem/Law-StackExchange',
                headers={'Authorization': f'Bearer {hf_token}'}
            )
            if response.status_code == 200:
                print("✅ Hugging Face token is valid")
            else:
                print("❌ Hugging Face token is invalid")
                return False
        else:
            print("❌ HF_TOKEN not found in environment")
            return False
    except Exception as e:
        print(f"❌ Error testing Hugging Face token: {e}")
        return False
    
    # Test Groq API key (basic validation)
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and len(groq_key) > 20:  # Basic length check
        print("✅ Groq API key format looks valid")
    else:
        print("❌ GROQ_API_KEY format is invalid")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting FastAPI server...")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start the server
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def run_demo():
    """Run the demo script"""
    print("\n🎭 Running demo...")
    try:
        subprocess.run([sys.executable, 'demo.py'])
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    """Main setup function"""
    print("🏛️  Law RAG Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Test API keys
    if not test_api_keys():
        print("\n❌ API key validation failed. Please check your tokens and try again.")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    
    # Ask user what to do next
    while True:
        print("\n🎯 What would you like to do next?")
        print("1. Start the server")
        print("2. Run the demo")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            start_server()
        elif choice == '2':
            run_demo()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 