#!/usr/bin/env python3
"""
Quick start script for Intelligent Video Surveillance System
"""
import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 
        'flask', 'numpy', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n🔧 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model():
    """Check if a pre-trained model exists (prefer v2)."""
    candidates = [
        "trained_models_v2/best_model.pth",
        "trained_models/best_model.pth",
    ]
    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break
    if found:
        print(f"✅ Pre-trained model found: {found}")
        return True
    print("❌ No pre-trained model found")
    print("   Looked for:")
    for p in candidates:
        print(f"   - {p}")
    print("\n🔧 Options:")
    print("1. Place your checkpoint at one of the above paths")
    print("2. Train a new model using: python train.py --help")
    return False

def main():
    print("🎥 Intelligent Video Surveillance System")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check model
    print("\n🤖 Checking model...")
    model_exists = check_model()
    
    print("\n🚀 Starting the application...")
    print("   - Web interface: http://localhost:5000")
    print("   - Press Ctrl+C to stop")
    print("=" * 50)
    
    if not model_exists:
        response = input("\nModel not found. Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Exiting...")
            sys.exit(1)
    
    # Start the application
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
