"""
Test Script for Basketball Analysis System
Validates installation and basic functionality
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('supervision', 'Supervision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful")
    return True

def test_cuda():
    """Test CUDA functionality"""
    print("\n🔍 Testing CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA available: {device_name}")
            print(f"✅ GPU Memory: {memory_gb:.1f} GB")
            
            # Test tensor operations
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✅ CUDA tensor operations working")
            
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return False
            
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_models():
    """Test model loading"""
    print("\n🔍 Testing model loading...")
    
    models_to_test = [
        ("./models/yolo11n.pt", "YOLO11"),
        ("./models/sam2.1_b.pt", "SAM2"),
        ("./runs/detect/train9/weights/best.pt", "Custom Basketball Model (optional)")
    ]
    
    model_status = {}
    
    for model_path, model_name in models_to_test:
        if Path(model_path).exists():
            print(f"✅ Found: {model_name}")
            model_status[model_name] = True
            
            # Test loading for critical models
            if "Custom" not in model_name:
                try:
                    if "YOLO" in model_name:
                        from ultralytics import YOLO
                        model = YOLO(model_path)
                        print(f"✅ {model_name} loaded successfully")
                    elif "SAM" in model_name:
                        from ultralytics import SAM
                        model = SAM(model_path)
                        print(f"✅ {model_name} loaded successfully")
                except Exception as e:
                    print(f"⚠️  {model_name} file exists but failed to load: {e}")
                    model_status[model_name] = False
        else:
            print(f"❌ Missing: {model_name} at {model_path}")
            model_status[model_name] = False
    
    # Check if critical models are available
    critical_models = ["YOLO11", "SAM2"]
    critical_available = all(model_status.get(model, False) for model in critical_models)
    
    if critical_available:
        print("✅ All critical models available")
        return True
    else:
        print("❌ Missing critical models - run setup.py first")
        return False

def test_core_functionality():
    """Test core basketball analyzer functionality"""
    print("\n🔍 Testing core functionality...")
    
    try:
        # Test config
        from config import Config
        Config.optimize_gpu()
        print("✅ Configuration loaded")
        
        # Test analyzer initialization (without loading heavy models)
        print("✅ Core modules importable")
        
        # Test data structures
        from basketball_analyzer import PlayerStats, GameAnalytics
        stats = PlayerStats(player_id=1)
        analytics = GameAnalytics()
        print("✅ Data structures working")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_sample_video():
    """Check for sample video files"""
    print("\n🔍 Checking for video files...")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    # Check media directory
    media_dir = Path("./media")
    if media_dir.exists():
        for ext in video_extensions:
            video_files.extend(media_dir.glob(f"*{ext}"))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video file(s):")
        for video in video_files[:3]:  # Show first 3
            print(f"   - {video}")
        if len(video_files) > 3:
            print(f"   ... and {len(video_files) - 3} more")
        return True
    else:
        print("⚠️  No video files found in ./media/ directory")
        print("Add some basketball video files to test the system")
        return False

def run_quick_test():
    """Run a quick functionality test without heavy processing"""
    print("\n🔍 Running quick functionality test...")
    
    try:
        # Test minimal analyzer setup
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Device selection: {device}")
        
        # Test basic image processing
        import cv2
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[:] = (0, 100, 0)  # Green background
        
        # Test basic operations
        gray = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(dummy_image, (320, 240))
        
        print("✅ Basic image processing working")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🏀 Basketball Analysis System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Model Test", test_models),
        ("Core Functionality", test_core_functionality),
        ("Video Files Check", test_sample_video),
        ("Quick Functionality Test", run_quick_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("🏀 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Add basketball videos to ./media/ directory")
        print("2. Run: python quick_start.py")
        print("3. Or: python script.py --video your_video.mp4 --visualize")
        return 0
    elif passed >= total - 2:  # Allow for minor issues
        print("\n⚠️  Most tests passed. System should work with minor limitations.")
        print("Check failed tests above for specific issues.")
        return 0
    else:
        print("\n❌ Multiple test failures. Please run setup.py first.")
        print("Or check the installation guide in README.md")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
