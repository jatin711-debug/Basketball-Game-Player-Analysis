"""
Configuration settings for Basketball Analysis System
RTX 3050 optimized settings
"""

import torch
from pathlib import Path

class Config:
    # Hardware Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPU_MEMORY_FRACTION = 0.8  # Reserve 20% for system
    BATCH_SIZE = 4  # Optimal for RTX 3050
    USE_HALF_PRECISION = True  # FP16 for better performance
    
    # Model Paths
    MODEL_DIR = Path("./models")
    YOLO_GENERAL_PATH = MODEL_DIR / "yolo11n.pt"
    SAM_MODEL_PATH = MODEL_DIR / "sam2.1_b.pt"
    CUSTOM_MODEL_PATH = Path("./runs/detect/train9/weights/best.pt")
    
    # Video Processing
    DEFAULT_FPS = 30
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    MAX_TRAJECTORY_LENGTH = 30
    
    # Tracking Parameters
    TRACK_ACTIVATION_THRESHOLD = 0.25
    LOST_TRACK_BUFFER = 50
    MINIMUM_MATCHING_THRESHOLD = 0.8
    
    # Analysis Parameters
    BALL_POSSESSION_DISTANCE = 80  # pixels
    SHOT_DETECTION_DISTANCE = 100  # pixels to rim
    PLAYER_INTERACTION_DISTANCE = 150  # pixels
    
    # Court Zones (normalized coordinates 0-1, will be scaled to frame size)
    COURT_ZONES = {
        "paint_left": (0.078, 0.278, 0.313, 0.722),
        "paint_right": (0.688, 0.278, 0.922, 0.722),
        "three_point_left": (0.039, 0.139, 0.352, 0.861),
        "three_point_right": (0.648, 0.139, 0.961, 0.861),
        "mid_court": (0.313, 0.0, 0.688, 1.0),
        "free_throw_left": (0.078, 0.347, 0.313, 0.653),
        "free_throw_right": (0.688, 0.347, 0.922, 0.653)
    }
    
    # Class IDs (based on your dataset)
    CLASS_NAMES = {
        0: 'unknown',
        1: 'basketball', 
        2: 'people',
        3: 'rim'
    }
    
    # Output Settings
    OUTPUT_DIR = Path("./output")
    ANALYSIS_DIR = OUTPUT_DIR / "analysis"
    VIDEO_DIR = OUTPUT_DIR / "videos"
    STATS_DIR = OUTPUT_DIR / "statistics"
    
    # Visualization
    COLORS = {
        'player': (0, 255, 0),
        'ball': (255, 0, 0),
        'rim': (0, 0, 255),
        'trajectory': (255, 255, 0),
        'zone': (128, 128, 128)
    }
    
    # Performance Monitoring
    LOG_INTERVAL = 100  # frames
    SAVE_INTERVAL = 1000  # frames
    
    @classmethod
    def create_directories(cls):
        """Create necessary output directories"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.ANALYSIS_DIR.mkdir(exist_ok=True)
        cls.VIDEO_DIR.mkdir(exist_ok=True)
        cls.STATS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def optimize_gpu(cls):
        """Optimize GPU settings for RTX 3050"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cls.GPU_MEMORY_FRACTION)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Clear cache
            torch.cuda.empty_cache()
    
    @classmethod
    def get_scaled_zones(cls, frame_width: int, frame_height: int) -> dict:
        """Scale court zones to actual frame dimensions"""
        scaled_zones = {}
        for zone_name, (x1, y1, x2, y2) in cls.COURT_ZONES.items():
            scaled_zones[zone_name] = (
                int(x1 * frame_width),
                int(y1 * frame_height),
                int(x2 * frame_width),
                int(y2 * frame_height)
            )
        return scaled_zones
