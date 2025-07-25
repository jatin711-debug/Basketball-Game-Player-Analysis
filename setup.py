"""
Setup and Installation Script for Basketball Analysis System
Automated setup for RTX 3050 optimization
"""

import subprocess
import sys
import os
from pathlib import Path
import urllib.request
import shutil

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected - will use CPU (slower)")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - CUDA may not be installed")
        return False

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "supervision>=0.16.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "Pillow>=9.5.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0"
    ]
    
    print("📦 Installing required packages...")
    
    # Install PyTorch with CUDA support for RTX 3050
    cuda_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    if not run_command(cuda_command, "Installing PyTorch with CUDA support"):
        # Fallback to CPU version
        cpu_command = "pip install torch torchvision torchaudio"
        run_command(cpu_command, "Installing PyTorch (CPU version)")
    
    # Install other requirements
    for package in requirements[3:]:  # Skip torch packages already installed
        if not run_command(f"pip install {package}", f"Installing {package.split('>=')[0]}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "media", 
        "output",
        "output/analysis",
        "output/videos",
        "output/statistics",
        "output/visualizations"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def download_models():
    """Download required models"""
    models = {
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt",
        "sam2.1_b.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    }
    
    print("📥 Downloading models...")
    models_dir = Path("models")
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✅ {model_name} already exists")
            continue
            
        try:
            print(f"⬇️  Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Downloaded: {model_name}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            print(f"Please manually download from: {url}")
    
    return True

def check_custom_model():
    """Check if custom basketball model exists"""
    custom_model_path = Path("runs/detect/train9/weights/best.pt")
    if custom_model_path.exists():
        print("✅ Custom basketball model found")
        return True
    else:
        print("⚠️  Custom basketball model not found")
        print(f"Expected location: {custom_model_path}")
        print("The system will use the general YOLO model instead")
        print("To train a custom model, use your basketball dataset")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    config_content = '''# Sample Video Configuration
# Edit this file to point to your video files

# Video files (add your own)
SAMPLE_VIDEOS = [
    "./media/video_1.mp4",
    "./media/video_2.mp4"
    # Add more video paths here
]

# Analysis settings
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Output settings  
GENERATE_VISUALIZATIONS = True
SAVE_ANNOTATED_VIDEO = False  # Set to True for video output

print("Configuration loaded successfully!")
'''
    
    config_path = Path("user_config.py")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Created sample configuration file: user_config.py")
    
    return True

def verify_installation():
    """Verify the installation"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        import ultralytics
        import supervision
        import cv2
        import pandas
        import matplotlib
        
        print("✅ All packages imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - using CPU")
        
        # Check models
        models_exist = True
        for model in ["models/yolo11n.pt", "models/sam2.1_b.pt"]:
            if Path(model).exists():
                print(f"✅ Found: {model}")
            else:
                print(f"❌ Missing: {model}")
                models_exist = False
        
        return models_exist
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🏀 Basketball Analysis System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Some packages failed to install, but continuing...")
    
    # Download models
    if not download_models():
        print("⚠️  Some models failed to download")
    
    # Check custom model
    check_custom_model()
    
    # Create sample config
    create_sample_config()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add basketball video files to ./media/ directory")
        print("2. Run: python quick_start.py")
        print("3. Or run: python script.py --video ./media/your_video.mp4 --visualize")
        print("\nFor RTX 3050 users:")
        print("- The system is optimized for your GPU")
        print("- Monitor GPU temperature during analysis")
        print("- Close other GPU-intensive applications")
        return 0
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and try again")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
