#!/usr/bin/env python3
"""
Setup script for free LLMs in CV Analysis API
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")
    
    try:
        # Check if Ollama is installed
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print(f"📋 Version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def install_ollama():
    """Install Ollama"""
    print("\n🚀 Installing Ollama...")
    
    try:
        # Download and install Ollama
        install_script = "curl -fsSL https://ollama.ai/install.sh | sh"
        print(f"Running: {install_script}")
        
        result = subprocess.run(install_script, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ollama installed successfully!")
            return True
        else:
            print(f"❌ Failed to install Ollama: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing Ollama: {e}")
        return False

def pull_llama_models():
    """Pull Llama models for Ollama"""
    print("\n📥 Pulling Llama models...")
    
    models = [
        "llama2:7b",
        "llama2:13b",
        "mistral:7b",
        "deepseek-coder:6.7b"
    ]
    
    for model in models:
        print(f"📥 Pulling {model}...")
        try:
            result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {model} pulled successfully!")
            else:
                print(f"⚠️ Failed to pull {model}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error pulling {model}: {e}")

def check_huggingface():
    """Check HuggingFace API key"""
    print("\n🔍 Checking HuggingFace configuration...")
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key:
        print("✅ HuggingFace API key is set")
        
        # Test the API key
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                print("✅ HuggingFace API key is valid")
                return True
            else:
                print(f"⚠️ HuggingFace API key test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing HuggingFace API key: {e}")
            return False
    else:
        print("⚠️ HuggingFace API key not set")
        return False

def setup_huggingface():
    """Setup HuggingFace API key"""
    print("\n🔧 Setting up HuggingFace...")
    
    print("📋 To get a HuggingFace API key:")
    print("1. Visit: https://huggingface.co/settings/tokens")
    print("2. Create a new token")
    print("3. Copy the token")
    
    api_key = input("\n🔑 Enter your HuggingFace API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Add to environment
        os.environ["HUGGINGFACE_API_KEY"] = api_key
        
        # Add to .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "a") as f:
                f.write(f"\nHUGGINGFACE_API_KEY={api_key}")
        else:
            with open(env_file, "w") as f:
                f.write(f"HUGGINGFACE_API_KEY={api_key}")
        
        print("✅ HuggingFace API key saved to .env file")
        return True
    else:
        print("⚠️ Skipping HuggingFace setup")
        return False

def check_fireworks():
    """Check Fireworks API key"""
    print("\n🔍 Checking Fireworks configuration...")
    
    api_key = os.getenv("FIREWORKS_API_KEY")
    if api_key:
        print("✅ Fireworks API key is set")
        return True
    else:
        print("⚠️ Fireworks API key not set")
        return False

def setup_fireworks():
    """Setup Fireworks API key"""
    print("\n🔧 Setting up Fireworks...")
    
    print("📋 To get a Fireworks API key:")
    print("1. Visit: https://fireworks.ai/")
    print("2. Sign up for free account")
    print("3. Get your API key from dashboard")
    
    api_key = input("\n🔑 Enter your Fireworks API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Add to environment
        os.environ["FIREWORKS_API_KEY"] = api_key
        
        # Add to .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "a") as f:
                f.write(f"\nFIREWORKS_API_KEY={api_key}")
        else:
            with open(env_file, "w") as f:
                f.write(f"FIREWORKS_API_KEY={api_key}")
        
        print("✅ Fireworks API key saved to .env file")
        return True
    else:
        print("⚠️ Skipping Fireworks setup")
        return False

def check_together():
    """Check Together AI API key"""
    print("\n🔍 Checking Together AI configuration...")
    
    api_key = os.getenv("TOGETHER_API_KEY")
    if api_key:
        print("✅ Together AI API key is set")
        return True
    else:
        print("⚠️ Together AI API key not set")
        return False

def setup_together():
    """Setup Together AI API key"""
    print("\n🔧 Setting up Together AI...")
    
    print("📋 To get a Together AI API key:")
    print("1. Visit: https://together.ai/")
    print("2. Sign up for free account")
    print("3. Get your API key from dashboard")
    
    api_key = input("\n🔑 Enter your Together AI API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Add to environment
        os.environ["TOGETHER_API_KEY"] = api_key
        
        # Add to .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "a") as f:
                f.write(f"\nTOGETHER_API_KEY={api_key}")
        else:
            with open(env_file, "w") as f:
                f.write(f"TOGETHER_API_KEY={api_key}")
        
        print("✅ Together AI API key saved to .env file")
        return True
    else:
        print("⚠️ Skipping Together AI setup")
        return False

def test_models():
    """Test available models"""
    print("\n🧪 Testing available models...")
    
    try:
        from llm_prompter import llm_manager
        
        models = list(llm_manager.models.keys())
        if models:
            print(f"✅ Available models: {', '.join(models)}")
            
            # Test with a simple prompt
            test_prompt = "Analyze this ingredient: Sugar, Salt"
            print("🔄 Testing model with sample prompt...")
            
            result = llm_manager.prompt_llm(test_prompt)
            if result:
                print("✅ Model test successful!")
                return True
            else:
                print("❌ Model test failed")
                return False
        else:
            print("❌ No models available")
            return False
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Free LLM Setup for CV Analysis API")
    print("=" * 50)
    print("✨ Setting up Llama 3, DeepSeek, Mistral, and other free LLMs")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    if not (current_dir / "llm_prompter.py").exists():
        print("❌ Please run this script from the analyze directory")
        return False
    
    setup_results = []
    
    # Setup Ollama (recommended)
    print("\n" + "=" * 50)
    print("🦙 Setting up Ollama (Recommended)")
    print("=" * 50)
    
    if not check_ollama():
        install_choice = input("Install Ollama? (y/n): ").lower().strip()
        if install_choice == 'y':
            if install_ollama():
                pull_llama_models()
                setup_results.append(("Ollama", True))
            else:
                setup_results.append(("Ollama", False))
        else:
            setup_results.append(("Ollama", False))
    else:
        pull_llama_models()
        setup_results.append(("Ollama", True))
    
    # Setup HuggingFace
    print("\n" + "=" * 50)
    print("🤗 Setting up HuggingFace")
    print("=" * 50)
    
    if not check_huggingface():
        setup_choice = input("Setup HuggingFace? (y/n): ").lower().strip()
        if setup_choice == 'y':
            if setup_huggingface():
                setup_results.append(("HuggingFace", True))
            else:
                setup_results.append(("HuggingFace", False))
        else:
            setup_results.append(("HuggingFace", False))
    else:
        setup_results.append(("HuggingFace", True))
    
    # Setup Fireworks
    print("\n" + "=" * 50)
    print("🎆 Setting up Fireworks AI")
    print("=" * 50)
    
    if not check_fireworks():
        setup_choice = input("Setup Fireworks AI? (y/n): ").lower().strip()
        if setup_choice == 'y':
            if setup_fireworks():
                setup_results.append(("Fireworks", True))
            else:
                setup_results.append(("Fireworks", False))
        else:
            setup_results.append(("Fireworks", False))
    else:
        setup_results.append(("Fireworks", True))
    
    # Setup Together AI
    print("\n" + "=" * 50)
    print("🤝 Setting up Together AI")
    print("=" * 50)
    
    if not check_together():
        setup_choice = input("Setup Together AI? (y/n): ").lower().strip()
        if setup_choice == 'y':
            if setup_together():
                setup_results.append(("Together AI", True))
            else:
                setup_results.append(("Together AI", False))
        else:
            setup_results.append(("Together AI", False))
    else:
        setup_results.append(("Together AI", True))
    
    # Test models
    print("\n" + "=" * 50)
    print("🧪 Testing Models")
    print("=" * 50)
    
    if test_models():
        setup_results.append(("Model Testing", True))
    else:
        setup_results.append(("Model Testing", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    print("=" * 50)
    
    successful = 0
    total = len(setup_results)
    
    for service, success in setup_results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} {service}")
        if success:
            successful += 1
    
    print(f"\n📈 Results: {successful}/{total} services configured successfully")
    
    if successful > 0:
        print("\n🎉 Setup completed! You can now run the API:")
        print("python start_api.py")
    else:
        print("\n⚠️ No services configured. The API will use fallback responses.")
    
    return successful > 0

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 