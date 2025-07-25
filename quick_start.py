"""
Quick Start Script for Basketball Analysis
Simple interface for running analysis with common settings
"""

import os
import sys
from pathlib import Path
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import ultralytics
        import supervision
        import cv2
        import pandas
        import matplotlib
        import plotly
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_models():
    """Check if required models are available"""
    models_to_check = [
        "./models/yolo11n.pt",
        "./models/sam2.1_b.pt",
        "./runs/detect/train9/weights/best.pt"
    ]
    
    missing_models = []
    for model_path in models_to_check:
        if not Path(model_path).exists():
            missing_models.append(model_path)
        else:
            print(f"✓ Found: {model_path}")
    
    if missing_models:
        print("\n✗ Missing models:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease ensure all models are downloaded and placed correctly.")
        return False
    
    return True

def list_videos():
    """List available video files"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    # Check media directory
    media_dir = Path("./media")
    if media_dir.exists():
        for ext in video_extensions:
            video_files.extend(media_dir.glob(f"*{ext}"))
    
    # Check current directory
    current_dir = Path(".")
    for ext in video_extensions:
        video_files.extend(current_dir.glob(f"*{ext}"))
    
    return video_files

def main():
    print("🏀 Basketball Game Analysis System - Quick Start")
    print("=" * 50)
    
    # Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        return 1
    
    # Check models
    print("\n2. Checking models...")
    if not check_models():
        return 1
    
    # List available videos
    print("\n3. Available video files:")
    video_files = list_videos()
    
    if not video_files:
        print("No video files found. Please add video files to ./media/ directory")
        return 1
    
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video}")
    
    # Get user selection
    try:
        choice = int(input(f"\nSelect video (1-{len(video_files)}): ")) - 1
        if choice < 0 or choice >= len(video_files):
            print("Invalid selection")
            return 1
        
        selected_video = video_files[choice]
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")
        return 1
    
    # Analysis options
    print("\n4. Analysis options:")
    print("  1. Basic analysis (no video output)")
    print("  2. Full analysis with visualizations")
    print("  3. Complete analysis with video output")
    
    try:
        analysis_choice = int(input("Select option (1-3): "))
        if analysis_choice not in [1, 2, 3]:
            print("Invalid selection")
            return 1
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")
        return 1
    
    # Build command
    cmd = [sys.executable, "script.py", "--video", str(selected_video)]
    
    if analysis_choice >= 2:
        cmd.append("--visualize")
    
    if analysis_choice == 3:
        cmd.append("--save-video")
    
    # Run analysis
    print(f"\n5. Running analysis on: {selected_video}")
    print("Command:", " ".join(cmd))
    print("\nStarting analysis... This may take several minutes.\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Analysis completed successfully!")
        print("Check the ./output/ directory for results.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis failed with error code: {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Analysis cancelled by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
