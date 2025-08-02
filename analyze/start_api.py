#!/usr/bin/env python3
"""
Startup script for the enhanced CV Analysis API with LangChain integration
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-ollama",
        "requests",
        "pydantic"
    ]
    
    # Optional free LLM packages
    optional_packages = [
        "langchain-huggingface",
        "langchain-fireworks", 
        "langchain-together"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️ {package} - Not installed (optional)")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_environment():
    """Check environment variables and configuration"""
    print("\n🔧 Checking environment configuration...")
    
    # Check for free LLM API keys
    api_keys = {
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
        "FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY"),
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY")
    }
    
    available_models = []
    for key, value in api_keys.items():
        if value:
            model_name = key.replace("_API_KEY", "").lower()
            if model_name == "huggingface":
                available_models.extend(["deepseek_coder", "mistral_7b", "llama3_8b"])
            elif model_name == "fireworks":
                available_models.append("fireworks_llama3")
            elif model_name == "together":
                available_models.append("together_llama3")
            elif model_name == "openrouter":
                available_models.append("openrouter")
            print(f"✅ {key} - Configured")
        else:
            print(f"⚠️ {key} - Not configured (optional)")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama - Running locally")
            available_models.extend(["ollama_llama3", "ollama_legacy"])
        else:
            print("⚠️ Ollama - Not responding")
    except:
        print("⚠️ Ollama - Not available")
    
    print(f"\n📋 Available LLM models: {', '.join(available_models) if available_models else 'None'}")
    
    if not available_models:
        print("⚠️ No LLM models available. The API will use fallback responses.")
        print("💡 To get free LLMs working:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Pull Llama 3: ollama pull llama2:7b")
        print("   3. Get HuggingFace API key: https://huggingface.co/settings/tokens")
        print("   4. Get Fireworks API key: https://fireworks.ai/")
        print("   5. Get Together AI key: https://together.ai/")
    
    return True

def start_api():
    """Start the FastAPI server"""
    print("\n🚀 Starting CV Analysis API...")
    print("=" * 50)
    
    # Change to the analyze directory
    analyze_dir = Path(__file__).parent
    os.chdir(analyze_dir)
    
    # Start the API
    try:
        import uvicorn
        from main import app
        
        print("🌐 Starting server on http://localhost:8001")
        print("📚 API Documentation: http://localhost:8001/docs")
        print("🔍 Health Check: http://localhost:8001/health")
        print("\n" + "=" * 50)
        print("✨ Features Available:")
        print("• LangChain LLM Integration")
        print("• Free LLM Support (Llama 3, DeepSeek, Mistral)")
        print("• Multilingual Support (English/French)")
        print("• Enhanced Error Handling")
        print("• Smart Fallback Mechanisms")
        print("=" * 50)
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return False

def test_api():
    """Test the API endpoints"""
    print("\n🧪 Testing API endpoints...")
    
    base_url = "http://localhost:8001"
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is running!")
                break
        except:
            time.sleep(1)
            print(f"⏳ Waiting... ({i+1}/30)")
    else:
        print("❌ API failed to start")
        return False
    
    # Test endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/models/available", "Available models"),
        ("/languages/supported", "Supported languages"),
        ("/example", "Example requests")
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {description} - Working")
            else:
                print(f"⚠️ {description} - Status {response.status_code}")
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
    
    return True

def main():
    """Main startup function"""
    print("🚀 CV Analysis API Startup")
    print("=" * 50)
    print("✨ LangChain Integration + Free LLM Support")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed.")
        return False
    
    # Test API if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if not test_api():
            print("\n❌ API test failed.")
            return False
        print("\n✅ API test completed successfully!")
        return True
    
    # Start the API
    return start_api()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 API stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 