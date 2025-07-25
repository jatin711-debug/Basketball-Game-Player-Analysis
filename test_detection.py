"""
Quick Detection Test Script
Test object detection on a single frame or short video clip
"""

import cv2
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from basketball_analyzer import BasketballAnalyzer
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_detection_on_frame(video_path: str, frame_number: int = 100):
    """Test detection on a specific frame"""
    print(f"🔍 Testing detection on frame {frame_number} of {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Jump to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_number}")
        return False
    
    print(f"✅ Frame loaded: {frame.shape}")
    
    # Initialize analyzer with debug mode
    try:
        analyzer = BasketballAnalyzer(device="auto")
        print("✅ Analyzer initialized")
        
        # Test detection
        print("\n🔍 Running detection test...")
        analyzer.debug_detection(frame)
        
        # Process the frame
        annotated_frame = analyzer.process_frame(frame)
        
        # Save the result
        output_path = "test_detection_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"✅ Result saved to: {output_path}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"Players detected: {len(analyzer.player_stats)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_on_video_clip(video_path: str, start_frame: int = 100, num_frames: int = 50):
    """Test detection on a short video clip"""
    print(f"🔍 Testing detection on {num_frames} frames starting from frame {start_frame}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Initialize analyzer
    try:
        analyzer = BasketballAnalyzer(device="auto")
        print("✅ Analyzer initialized")
        
        # Jump to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_processed = 0
        total_detections = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_processed += 1
            detections = analyzer.detect_objects(frame)
            total_detections += len(detections)
            
            # Show progress
            if i % 10 == 0:
                print(f"  Frame {start_frame + i}: {len(detections)} detections")
        
        cap.release()
        
        print(f"\n📊 Test Results:")
        print(f"Frames processed: {frames_processed}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections / frames_processed:.1f}")
        print(f"Players tracked: {len(analyzer.player_stats)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🏀 Basketball Detection Test")
    print("=" * 40)
    
    # Check for video files
    video_files = []
    media_dir = Path("./media")
    if media_dir.exists():
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(media_dir.glob(f"*{ext}"))
    
    # Also check current directory
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(".").glob(f"*{ext}"))
    
    if not video_files:
        print("❌ No video files found. Please add video files to test.")
        return 1
    
    print("📹 Available videos:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video}")
    
    try:
        choice = int(input(f"\nSelect video (1-{len(video_files)}): ")) - 1
        if choice < 0 or choice >= len(video_files):
            print("Invalid selection")
            return 1
        
        selected_video = str(video_files[choice])
        
        print("\n🔧 Test options:")
        print("  1. Single frame test")
        print("  2. Short video clip test (50 frames)")
        
        test_choice = int(input("Select test (1-2): "))
        
        if test_choice == 1:
            frame_num = int(input("Frame number to test (default 100): ") or 100)
            success = test_detection_on_frame(selected_video, frame_num)
        elif test_choice == 2:
            start_frame = int(input("Start frame (default 100): ") or 100)
            success = test_detection_on_video_clip(selected_video, start_frame)
        else:
            print("Invalid test choice")
            return 1
        
        if success:
            print("\n✅ Test completed successfully!")
            return 0
        else:
            print("\n❌ Test failed!")
            return 1
            
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
