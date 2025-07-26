"""
Basketball Game Analysis System using YOLO11 and SAM2
Production-ready implementation for RTX 3050 GPU optimization
Enhanced with improved multi-player detection and SAM2 segmentation
"""

import cv2
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO, SAM
import supervision as sv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from pathlib import Path
import time
from datetime import datetime
import colorsys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PlayerStats:
    """Comprehensive player statistics tracking"""
    player_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    shot_attempts: int = 0
    shots_made: int = 0
    ball_touches: int = 0
    speed_data: List[float] = field(default_factory=list)
    time_in_zones: Dict[str, float] = field(default_factory=dict)
    movements: List[str] = field(default_factory=list)
    team: Optional[str] = None
    jersey_color: Optional[Tuple[int, int, int]] = None
    assigned_color: Optional[Tuple[int, int, int]] = None  # Color for segmentation visualization
    
    @property
    def shooting_percentage(self) -> float:
        return (self.shots_made / self.shot_attempts * 100) if self.shot_attempts > 0 else 0.0
    
    @property
    def average_speed(self) -> float:
        return np.mean(self.speed_data) if self.speed_data else 0.0
    
    @property
    def distance_covered(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            x1, y1 = self.positions[i-1]
            x2, y2 = self.positions[i]
            total_distance += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return total_distance

@dataclass
class GameAnalytics:
    """Game-level analytics and insights"""
    total_possessions: int = 0
    fast_breaks: int = 0
    turnovers: int = 0
    rebounds: int = 0
    assists: int = 0
    game_flow: List[str] = field(default_factory=list)
    heat_map_data: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)

class BasketballAnalyzer:
    """Main basketball analysis system with enhanced multi-player detection"""
    
    def __init__(self, 
                 yolo_model_path: str = "./models/yolo11l.pt",
                 sam_model_path: str = "./models/sam2.1_b.pt",
                 custom_model_path: str = "./runs/detect/train9/weights/best.pt",
                 device: str = "auto"):
        
        # Device configuration optimized for RTX 3050
        if device == "auto":
            self.device = self._configure_optimal_device()
        else:
            self.device = device
            
        logger.info(f"Initializing Basketball Analyzer on device: {self.device}")
        
        # Initialize models
        self._load_models(yolo_model_path, sam_model_path, custom_model_path)
        
        # Enhanced tracking configuration for multiple players
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.15,  # Lower threshold for better detection
            lost_track_buffer=60,  # Longer buffer to maintain tracking
            minimum_matching_threshold=0.7,  # Adjusted for better player association
            frame_rate=30,
            minimum_consecutive_frames=3  # Reduce false positives
        )
        
        # Initialize annotators
        self._setup_annotators()
        
        # Analytics storage
        self.player_stats: Dict[int, PlayerStats] = {}
        self.game_analytics = GameAnalytics()
        self.frame_count = 0
        self.fps = 30
        
        # Court analysis
        self.court_zones = self._initialize_court_zones()
        self.ball_trajectory = deque(maxlen=30)
        
        # Performance optimization settings for RTX 3050
        self.batch_size = 2  # Reduced for SAM2 processing
        self.confidence_threshold = 0.15  # Lower threshold for better player detection
        self.iou_threshold = 0.4  # Adjusted for overlapping players
        
        # SAM2 segmentation settings
        self.use_sam_segmentation = True
        self.segmentation_cache = {}  # Cache segmentations to improve performance
        self.player_colors = {}  # Store assigned colors for each player
        self.color_palette = self._generate_color_palette(20)  # Generate colors for up to 20 players
        
        # Enhanced detection settings
        self.detection_history = defaultdict(list)  # Track detection confidence over time
        self.min_detection_confidence = 0.1  # Very low for initial detection
        self.stable_detection_confidence = 0.25  # Higher for stable tracking
        
    def _configure_optimal_device(self) -> str:
        """Configure optimal device settings for RTX 3050"""
        if torch.cuda.is_available():
            # RTX 3050 specific optimizations
            torch.cuda.set_per_process_memory_fraction(0.75)  # Leave more memory for SAM2
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            return 'cuda'
        return 'cpu'
    
    def _load_models(self, yolo_path: str, sam_path: str, custom_path: str):
        """Load and configure all models"""
        try:
            # Load general YOLO model for person detection
            logger.info(f"Loading general YOLO model from: {yolo_path}")
            self.yolo_general = YOLO(yolo_path)
            self.yolo_general.to(self.device)
            
            # Always use general model for better person detection accuracy
            logger.info("Using general YOLO model for enhanced person detection")
            self.yolo_basketball = self.yolo_general
            
            # Load custom basketball-specific model as secondary
            if Path(custom_path).exists():
                try:
                    logger.info(f"Loading custom basketball model as secondary: {custom_path}")
                    self.yolo_custom = YOLO(custom_path)
                    self.yolo_custom.to(self.device)
                    self.has_custom_model = True
                    logger.info("Custom basketball model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load custom model: {e}")
                    self.has_custom_model = False
            else:
                logger.warning(f"Custom basketball model not found at: {custom_path}")
                self.has_custom_model = False
            
            # Load SAM2 for precise segmentation
            logger.info(f"Loading SAM2 model from: {sam_path}")
            self.sam_model = SAM(sam_path)
            self.sam_model.to(self.device)
            
            # Optimize models for inference
            if self.device == 'cuda':
                try:
                    self.yolo_general.model.half()  # Use FP16 for better performance
                    if self.has_custom_model:
                        self.yolo_custom.model.half()
                    logger.info("Enabled FP16 optimization for CUDA")
                except Exception as e:
                    logger.warning(f"Could not enable FP16: {e}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _generate_color_palette(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for player visualization"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 + (i % 2) * 0.1  # Vary brightness slightly
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # Convert to BGR for OpenCV
            colors.append(bgr)
        
        return colors
    
    def _setup_annotators(self):
        """Setup visualization components"""
        self.box_annotator = sv.BoxAnnotator(
            thickness=3,
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=3,
            trace_length=50  # Longer traces for better movement visualization
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.7,
            text_color=sv.Color.WHITE,
            text_padding=5
        )
        
        # Heat map for player positions
        self.heat_map = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def _initialize_court_zones(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Initialize basketball court zones for analysis"""
        return {
            "paint_left": (100, 200, 400, 520),
            "paint_right": (880, 200, 1180, 520),
            "three_point_left": (50, 100, 450, 620),
            "three_point_right": (830, 100, 1230, 620),
            "mid_court": (400, 0, 880, 720),
            "free_throw_left": (100, 250, 400, 470),
            "free_throw_right": (880, 250, 1180, 470)
        }
    
    def detect_objects(self, frame: np.ndarray) -> sv.Detections:
        """Enhanced object detection with multi-model approach for better accuracy"""
        try:
            # Primary detection using general YOLO model (better for person detection)
            results_general = self.yolo_general(
                frame,
                conf=self.min_detection_confidence,  # Very low initial threshold
                iou=self.iou_threshold,
                device=self.device,
                half=True if self.device == 'cuda' else False,
                classes=[0]  # Only detect persons (class 0 in COCO)
            )[0]
            
            detections_general = sv.Detections.from_ultralytics(results_general)
            
            # Secondary detection using custom model for basketball-specific objects
            basketball_detections = sv.Detections.empty()
            if self.has_custom_model:
                try:
                    results_custom = self.yolo_custom(
                        frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        device=self.device,
                        half=True if self.device == 'cuda' else False
                    )[0]
                    
                    basketball_detections = sv.Detections.from_ultralytics(results_custom)
                    
                    # Filter for basketball and rim detections only
                    if hasattr(basketball_detections, 'class_id') and basketball_detections.class_id is not None:
                        basketball_mask = np.isin(basketball_detections.class_id, [1, 3])  # basketball, rim
                        basketball_detections = basketball_detections[basketball_mask]
                
                except Exception as e:
                    logger.warning(f"Custom model detection failed: {e}")
            
            # Combine detections
            if len(basketball_detections) > 0:
                # Merge person detections with basketball detections
                all_detections = self._merge_detections(detections_general, basketball_detections)
            else:
                all_detections = detections_general
            
            # Enhanced filtering for better person detection
            all_detections = self._filter_person_detections(all_detections, frame.shape)
            
            # Debug logging
            if len(all_detections) > 0:
                logger.debug(f"Enhanced detection: {len(all_detections)} objects found")
                if hasattr(all_detections, 'class_id') and all_detections.class_id is not None:
                    person_count = np.sum(all_detections.class_id == 0)
                    logger.debug(f"Person detections: {person_count}")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return sv.Detections.empty()
    
    def _merge_detections(self, person_detections: sv.Detections, basketball_detections: sv.Detections) -> sv.Detections:
        """Merge person and basketball object detections"""
        if len(person_detections) == 0:
            return basketball_detections
        if len(basketball_detections) == 0:
            return person_detections
        
        # Combine detection arrays
        combined_xyxy = np.vstack([person_detections.xyxy, basketball_detections.xyxy])
        combined_confidence = np.hstack([person_detections.confidence, basketball_detections.confidence])
        combined_class_id = np.hstack([person_detections.class_id, basketball_detections.class_id])
        
        # Create merged detection object
        merged_detections = sv.Detections(
            xyxy=combined_xyxy,
            confidence=combined_confidence,
            class_id=combined_class_id
        )
        
        return merged_detections
    
    def _filter_person_detections(self, detections: sv.Detections, frame_shape: Tuple[int, int, int]) -> sv.Detections:
        """Enhanced filtering for person detections to improve accuracy"""
        if len(detections) == 0:
            return detections
        
        height, width = frame_shape[:2]
        valid_indices = []
        
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            confidence = detections.confidence[i]
            
            # Calculate detection properties
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
            
            # Enhanced filtering criteria for person detection
            is_valid = True
            
            # Size constraints (people should be reasonably sized)
            min_area = (width * height) * 0.001  # Minimum 0.1% of frame
            max_area = (width * height) * 0.3    # Maximum 30% of frame
            if bbox_area < min_area or bbox_area > max_area:
                is_valid = False
            
            # Aspect ratio constraints (people are typically taller than wide)
            if aspect_ratio < 0.8 or aspect_ratio > 4.0:
                is_valid = False
            
            # Minimum dimensions
            if bbox_width < 20 or bbox_height < 40:
                is_valid = False
            
            # Position constraints (not too close to edges unless confidence is high)
            edge_margin = 10
            if confidence < 0.3:
                if (x1 < edge_margin or x2 > width - edge_margin or 
                    y1 < edge_margin or y2 > height - edge_margin):
                    is_valid = False
            
            # Dynamic confidence threshold based on detection stability
            if hasattr(detections, 'class_id') and detections.class_id[i] == 0:  # Person class
                min_conf = self.stable_detection_confidence if len(self.player_stats) > 3 else self.min_detection_confidence
                if confidence < min_conf:
                    is_valid = False
            
            if is_valid:
                valid_indices.append(i)
        
        # Apply filtering
        if valid_indices:
            filtered_detections = detections[valid_indices]
            logger.debug(f"Filtered detections: {len(filtered_detections)}/{len(detections)} kept")
            return filtered_detections
        else:
            return sv.Detections.empty()
    
    def track_players(self, detections: sv.Detections) -> sv.Detections:
        """Enhanced player tracking with improved multi-player handling"""
        if len(detections) == 0:
            return detections
            
        try:
            # Apply tracking
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Debug tracking
            if len(tracked_detections) > 0:
                logger.debug(f"Tracking: {len(tracked_detections)} objects tracked")
            
            # Update player statistics and assign colors
            for i in range(len(tracked_detections)):
                # Safe check for tracker_id existence and validity
                if (hasattr(tracked_detections, 'tracker_id') and 
                    tracked_detections.tracker_id is not None and 
                    i < len(tracked_detections.tracker_id)):
                    
                    player_id = tracked_detections.tracker_id[i]
                    
                    # Skip if player_id is None
                    if player_id is None:
                        continue
                        
                    # Initialize new player
                    if player_id not in self.player_stats:
                        self.player_stats[player_id] = PlayerStats(player_id=player_id)
                        # Assign unique color
                        color_index = len(self.player_colors) % len(self.color_palette)
                        self.player_colors[player_id] = self.color_palette[color_index]
                        self.player_stats[player_id].assigned_color = self.color_palette[color_index]
                        logger.info(f"New player detected: {player_id}, assigned color: {self.color_palette[color_index]}")
                    
                    # Extract position and update statistics
                    if i < len(tracked_detections.xyxy):
                        x1, y1, x2, y2 = tracked_detections.xyxy[i]
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        # Update position history
                        self.player_stats[player_id].positions.append((center_x, center_y))
                        
                        # Limit position history to prevent memory issues
                        if len(self.player_stats[player_id].positions) > 100:
                            self.player_stats[player_id].positions = self.player_stats[player_id].positions[-50:]
                        
                        # Calculate speed if we have previous positions
                        if len(self.player_stats[player_id].positions) >= 2:
                            prev_pos = self.player_stats[player_id].positions[-2]
                            current_pos = self.player_stats[player_id].positions[-1]
                            distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + 
                                             (current_pos[1] - prev_pos[1])**2)
                            speed = distance * self.fps  # pixels per second
                            self.player_stats[player_id].speed_data.append(speed)
                            
                            # Limit speed data history
                            if len(self.player_stats[player_id].speed_data) > 50:
                                self.player_stats[player_id].speed_data = self.player_stats[player_id].speed_data[-30:]
                        
                        # Analyze court zone
                        self._analyze_player_zone(player_id, center_x, center_y)
            
            return tracked_detections
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return detections
    
    def _analyze_player_zone(self, player_id: int, x: float, y: float):
        """Analyze which court zone the player is in"""
        for zone_name, (x1, y1, x2, y2) in self.court_zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if zone_name not in self.player_stats[player_id].time_in_zones:
                    self.player_stats[player_id].time_in_zones[zone_name] = 0
                self.player_stats[player_id].time_in_zones[zone_name] += 1.0 / self.fps
    
    def segment_players_with_sam2(self, frame: np.ndarray, detections: sv.Detections) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Segment players using SAM2 and apply color coding"""
        if len(detections) == 0 or not self.use_sam_segmentation:
            return frame, {}
        
        segmented_frame = frame.copy()
        player_masks = {}
        
        try:
            # Process each detected player
            for i in range(len(detections)):
                # Safe check for tracker_id
                if (hasattr(detections, 'tracker_id') and 
                    detections.tracker_id is not None and 
                    i < len(detections.tracker_id) and
                    detections.tracker_id[i] is not None):
                    
                    player_id = detections.tracker_id[i]
                    bbox = detections.xyxy[i]
                    
                    # Check if this is a person detection
                    if hasattr(detections, 'class_id') and i < len(detections.class_id) and detections.class_id[i] != 0:
                        continue  # Skip non-person detections for segmentation
                    
                    # Generate cache key for this frame region
                    cache_key = f"{player_id}_{self.frame_count}_{int(bbox[0])}_{int(bbox[1])}"
                    
                    # Use cached mask if available and recent
                    if cache_key in self.segmentation_cache:
                        mask = self.segmentation_cache[cache_key]
                    else:
                        # Generate new segmentation
                        mask = self._generate_sam2_mask(frame, bbox)
                        
                        # Cache the mask (limit cache size)
                        if len(self.segmentation_cache) > 50:
                            # Remove oldest entries
                            oldest_keys = list(self.segmentation_cache.keys())[:10]
                            for key in oldest_keys:
                                del self.segmentation_cache[key]
                        
                        self.segmentation_cache[cache_key] = mask
                    
                    if mask is not None:
                        player_masks[player_id] = mask
                        # Apply colored segmentation
                        segmented_frame = self._apply_colored_mask(segmented_frame, mask, player_id)
        
        except Exception as e:
            logger.warning(f"SAM2 segmentation failed: {e}")
            return frame, {}
        
        return segmented_frame, player_masks
    
    def _generate_sam2_mask(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Generate segmentation mask using SAM2"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Expand bbox slightly for better segmentation
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            # Use bbox as prompt for SAM2
            box_prompt = np.array([[x1, y1, x2, y2]])
            
            # Run SAM2 prediction
            results = self.sam_model(frame, bboxes=box_prompt)
            
            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                # Get the first (best) mask
                mask = results[0].masks.data[0].cpu().numpy()
                
                # Ensure mask is boolean
                if mask.dtype != bool:
                    mask = mask > 0.5
                
                return mask.astype(np.uint8)
            
        except Exception as e:
            logger.debug(f"SAM2 mask generation failed: {e}")
        
        return None
    
    def _apply_colored_mask(self, frame: np.ndarray, mask: np.ndarray, player_id: int) -> np.ndarray:
        """Apply colored mask overlay to frame"""
        if player_id not in self.player_colors:
            return frame
        
        color = self.player_colors[player_id]
        
        # Create colored overlay
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = color
        
        # Apply overlay with transparency
        alpha = 0.4  # Transparency level
        overlay_area = mask > 0
        frame[overlay_area] = (frame[overlay_area] * (1 - alpha) + 
                              colored_mask[overlay_area] * alpha).astype(np.uint8)
        
        return frame
    
    def analyze_basketball_events(self, frame: np.ndarray, detections: sv.Detections):
        """Analyze basketball-specific events and actions"""
        ball_detections = []
        rim_detections = []
        player_detections = []
        
        # Separate detections by class
        for i, detection in enumerate(detections):
            if hasattr(detections, 'class_id') and len(detections.class_id) > i:
                class_id = detections.class_id[i]
                if class_id == 1:  # basketball
                    ball_detections.append(detection)
                elif class_id == 3:  # rim
                    rim_detections.append(detection)
                elif class_id == 0:  # person (from general model)
                    player_detections.append(detection)
        
        # Analyze ball trajectory
        if ball_detections:
            ball_pos = self._get_detection_center(ball_detections[0])
            self.ball_trajectory.append(ball_pos)
            self._analyze_shot_attempt(ball_pos, rim_detections)
        
        # Analyze player interactions
        self._analyze_player_interactions(player_detections, ball_detections)
    
    def _get_detection_center(self, detection) -> Tuple[float, float]:
        """Get center point of detection"""
        if hasattr(detection, 'xyxy') and len(detection.xyxy) > 0:
            x1, y1, x2, y2 = detection.xyxy[0]
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return (0, 0)
    
    def _analyze_shot_attempt(self, ball_pos: Tuple[float, float], rim_detections: List):
        """Analyze potential shot attempts"""
        if len(self.ball_trajectory) < 5:
            return
        
        # Analyze ball trajectory for upward movement toward rim
        recent_positions = list(self.ball_trajectory)[-5:]
        y_movement = [pos[1] for pos in recent_positions]
        
        # Check for upward trajectory (decreasing y values)
        if len(y_movement) >= 3 and y_movement[0] > y_movement[-1]:
            # Potential shot detected
            if rim_detections:
                rim_pos = self._get_detection_center(rim_detections[0])
                distance_to_rim = np.sqrt((ball_pos[0] - rim_pos[0])**2 + 
                                        (ball_pos[1] - rim_pos[1])**2)
                
                if distance_to_rim < 100:  # Close to rim
                    self._record_shot_attempt(ball_pos, rim_pos)
    
    def _record_shot_attempt(self, ball_pos: Tuple[float, float], rim_pos: Tuple[float, float]):
        """Record shot attempt and outcome"""
        # Find closest player to attribute the shot
        closest_player = None
        min_distance = float('inf')
        
        for player_id, stats in self.player_stats.items():
            if stats.positions:
                last_pos = stats.positions[-1]
                distance = np.sqrt((ball_pos[0] - last_pos[0])**2 + 
                                 (ball_pos[1] - last_pos[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_player = player_id
        
        if closest_player and min_distance < 150:  # Within reasonable range
            self.player_stats[closest_player].shot_attempts += 1
            logger.info(f"Shot attempt recorded for player {closest_player}")
    
    def _analyze_player_interactions(self, player_detections: List, ball_detections: List):
        """Analyze player interactions with ball and each other"""
        if not ball_detections:
            return
        
        ball_pos = self._get_detection_center(ball_detections[0])
        
        # Check for ball possession
        for detection in player_detections:
            if hasattr(detection, 'tracker_id') and detection.tracker_id:
                player_pos = self._get_detection_center(detection)
                distance_to_ball = np.sqrt((ball_pos[0] - player_pos[0])**2 + 
                                         (ball_pos[1] - player_pos[1])**2)
                
                if distance_to_ball < 80:  # Player is close to ball
                    if detection.tracker_id in self.player_stats:
                        self.player_stats[detection.tracker_id].ball_touches += 1
    
    def generate_player_analysis(self, player_id: int) -> Dict[str, Any]:
        """Generate comprehensive player analysis"""
        if player_id not in self.player_stats:
            return {}
        
        stats = self.player_stats[player_id]
        
        analysis = {
            "player_id": player_id,
            "assigned_color": stats.assigned_color,
            "performance_metrics": {
                "shots_attempted": stats.shot_attempts,
                "shots_made": stats.shots_made,
                "shooting_percentage": stats.shooting_percentage,
                "ball_touches": stats.ball_touches,
                "distance_covered": stats.distance_covered,
                "average_speed": stats.average_speed
            },
            "court_analysis": {
                "time_in_zones": stats.time_in_zones,
                "dominant_zone": max(stats.time_in_zones.items(), key=lambda x: x[1])[0] if stats.time_in_zones else "unknown"
            },
            "play_style": self._determine_play_style(stats),
            "strengths": self._identify_strengths(stats),
            "weaknesses": self._identify_weaknesses(stats)
        }
        
        return analysis
    
    def _determine_play_style(self, stats: PlayerStats) -> str:
        """Determine player's play style based on statistics"""
        if not stats.time_in_zones:
            return "Unknown"
        
        total_time = sum(stats.time_in_zones.values())
        zone_percentages = {zone: time/total_time for zone, time in stats.time_in_zones.items()}
        
        # Analyze dominant zones
        paint_time = zone_percentages.get("paint_left", 0) + zone_percentages.get("paint_right", 0)
        three_point_time = zone_percentages.get("three_point_left", 0) + zone_percentages.get("three_point_right", 0)
        mid_court_time = zone_percentages.get("mid_court", 0)
        
        if paint_time > 0.4:
            return "Post Player/Center"
        elif three_point_time > 0.3:
            return "Perimeter Shooter"
        elif stats.average_speed > 200:  # High mobility
            return "Point Guard/Playmaker"
        elif mid_court_time > 0.4:
            return "Mid-Range Specialist"
        else:
            return "Versatile Player"
    
    def _identify_strengths(self, stats: PlayerStats) -> List[str]:
        """Identify player strengths"""
        strengths = []
        
        if stats.shooting_percentage > 45:
            strengths.append("Accurate Shooter")
        if stats.average_speed > 250:
            strengths.append("High Mobility")
        if stats.ball_touches > 50:
            strengths.append("Ball Handler")
        if stats.distance_covered > 1000:
            strengths.append("High Work Rate")
        
        paint_time = sum(stats.time_in_zones.get(zone, 0) for zone in ["paint_left", "paint_right"])
        if paint_time > 30:  # seconds
            strengths.append("Interior Presence")
        
        return strengths if strengths else ["Developing Player"]
    
    def _identify_weaknesses(self, stats: PlayerStats) -> List[str]:
        """Identify areas for improvement"""
        weaknesses = []
        
        if stats.shooting_percentage < 30 and stats.shot_attempts > 5:
            weaknesses.append("Shooting Efficiency")
        if stats.average_speed < 100:
            weaknesses.append("Court Mobility")
        if len(stats.time_in_zones) < 3:
            weaknesses.append("Court Coverage")
        if stats.ball_touches < 20:
            weaknesses.append("Ball Involvement")
        
        return weaknesses if weaknesses else ["Well-Rounded Player"]
    
    def create_heat_map(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create heat map of player positions"""
        heat_map = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        
        for player_id, stats in self.player_stats.items():
            for pos in stats.positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                    # Add Gaussian blur around position
                    for i in range(max(0, y-20), min(frame_shape[0], y+20)):
                        for j in range(max(0, x-20), min(frame_shape[1], x+20)):
                            distance = np.sqrt((i-y)**2 + (j-x)**2)
                            if distance <= 20:
                                heat_map[i, j] += np.exp(-distance**2 / 200)
        
        # Normalize and convert to color
        heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
        heat_map_color = cv2.applyColorMap(heat_map.astype(np.uint8), cv2.COLORMAP_JET)
        
        return heat_map_color
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """Process complete video and generate analysis"""
        logger.info(f"Starting enhanced video analysis: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.fps = fps
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process frame with enhanced detection and segmentation
                annotated_frame = self.process_frame(frame)
                
                # Write frame if output specified
                if output_path:
                    out.write(annotated_frame)
                
                # Progress logging
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / self.frame_count) * (total_frames - self.frame_count)
                    logger.info(f"Progress: {progress:.1f}% - Players: {len(self.player_stats)} - ETA: {eta:.1f}s")
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        # Generate final analysis
        analysis_results = self.generate_game_analysis()
        
        logger.info("Enhanced video analysis completed successfully")
        logger.info(f"Total players detected and tracked: {len(self.player_stats)}")
        return analysis_results
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with enhanced detection and SAM2 segmentation"""
        # Enhanced object detection
        detections = self.detect_objects(frame)
        
        # Debug first few frames
        if self.frame_count <= 3:
            self.debug_detection(frame, detections)
        
        # Enhanced player tracking
        tracked_detections = self.track_players(detections)
        
        # SAM2 segmentation for players
        segmented_frame, player_masks = self.segment_players_with_sam2(frame, tracked_detections)
        
        # Basketball event analysis
        self.analyze_basketball_events(segmented_frame, tracked_detections)
        
        # Create comprehensive annotations
        annotated_frame = self._annotate_frame(segmented_frame, tracked_detections, player_masks)
        
        return annotated_frame
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections, player_masks: Dict[int, np.ndarray]) -> np.ndarray:
        """Create comprehensive frame annotations with enhanced visualization"""
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and enhanced labels
        if len(detections) > 0:
            # Create enhanced labels for each detection
            labels = []
            colors = []
            
            for i in range(len(detections)):
                # Safe check for tracker_id
                if (hasattr(detections, 'tracker_id') and 
                    detections.tracker_id is not None and 
                    i < len(detections.tracker_id)):
                    
                    player_id = detections.tracker_id[i]
                    
                    # Check if this is a person detection and has valid tracker_id
                    is_person = (hasattr(detections, 'class_id') and 
                               i < len(detections.class_id) and
                               detections.class_id[i] == 0)
                    
                    if is_person and player_id is not None and player_id in self.player_stats:
                        stats = self.player_stats[player_id]
                        label = (f"P{player_id} | "
                               f"Speed: {stats.average_speed:.1f} | "
                               f"Touches: {stats.ball_touches}")
                        
                        # Use assigned color for this player
                        if player_id in self.player_colors:
                            color = self.player_colors[player_id]
                            colors.append(sv.Color(r=color[2], g=color[1], b=color[0]))  # Convert BGR to RGB
                        else:
                            colors.append(sv.Color.WHITE)
                    else:
                        # Non-person detection or untracked detection
                        if hasattr(detections, 'class_id') and i < len(detections.class_id):
                            class_names = {1: "Ball", 3: "Rim", 0: "Person"}
                            class_id = detections.class_id[i]
                            label = class_names.get(class_id, f"Object{class_id}")
                        else:
                            label = "Detection"
                        colors.append(sv.Color.YELLOW)
                else:
                    label = "Detection"
                    colors.append(sv.Color.WHITE)
                
                labels.append(label)
            
            # Apply enhanced annotations with custom colors
            for i in range(len(detections)):
                # Draw individual bounding box with custom color
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                color = colors[i] if i < len(colors) else sv.Color.WHITE
                color_bgr = (color.b, color.g, color.r)  # Convert to BGR for OpenCV
                
                # Draw thicker bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 3)
                
                # Draw label with background
                label = labels[i] if i < len(labels) else "Detection"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), color_bgr, -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add traces for player movement with custom colors
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        
        # Add court zones overlay (semi-transparent)
        annotated_frame = self._draw_court_zones(annotated_frame)
        
        # Add enhanced statistics overlay
        annotated_frame = self._add_enhanced_stats_overlay(annotated_frame)
        
        # Add player legend
        annotated_frame = self._add_player_legend(annotated_frame)
        
        return annotated_frame
    
    def _draw_court_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw basketball court zones with enhanced visualization"""
        overlay = frame.copy()
        
        zone_colors = {
            "paint_left": (0, 100, 255),      # Orange
            "paint_right": (0, 100, 255),     # Orange
            "three_point_left": (255, 100, 0), # Blue
            "three_point_right": (255, 100, 0), # Blue
            "mid_court": (0, 255, 100),       # Green
            "free_throw_left": (100, 255, 255), # Yellow  
            "free_throw_right": (100, 255, 255) # Yellow
        }
        
        for zone_name, (x1, y1, x2, y2) in self.court_zones.items():
            color = zone_colors.get(zone_name, (0, 255, 0))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add zone label with background
            label = zone_name.replace("_", " ").title()
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, (x1, y1 - label_size[1] - 5), 
                        (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend with original frame (more transparent)
        return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
    
    def _add_enhanced_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add enhanced real-time statistics overlay"""
        overlay = frame.copy()
        
        # Create semi-transparent background for stats
        stats_bg = np.zeros((150, 350, 3), dtype=np.uint8)
        stats_bg[:] = (0, 0, 0)  # Black background
        
        y_offset = 20
        line_height = 25
        
        # Game statistics
        cv2.putText(stats_bg, f"Frame: {self.frame_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(stats_bg, f"Players Tracked: {len(self.player_stats)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Active players in current frame
        active_players = len([p for p in self.player_stats.values() 
                            if p.positions and len(p.positions) > 0])
        cv2.putText(stats_bg, f"Active Players: {active_players}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        # Ball trajectory status
        ball_status = "Tracking" if len(self.ball_trajectory) > 0 else "Not Detected"
        cv2.putText(stats_bg, f"Ball: {ball_status}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height
        
        # SAM2 segmentation status
        seg_status = "Active" if self.use_sam_segmentation else "Disabled"
        cv2.putText(stats_bg, f"SAM2 Segmentation: {seg_status}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Overlay stats background with transparency
        overlay[10:160, 10:360] = cv2.addWeighted(
            overlay[10:160, 10:360], 0.3, stats_bg, 0.7, 0)
        
        return overlay
    
    def _add_player_legend(self, frame: np.ndarray) -> np.ndarray:
        """Add player color legend"""
        if not self.player_colors:
            return frame
        
        legend_height = min(200, len(self.player_colors) * 25 + 30)
        legend_width = 200
        
        # Create legend background
        legend_bg = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        legend_bg[:] = (0, 0, 0)  # Black background
        
        # Legend title
        cv2.putText(legend_bg, "Player Legend", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Player colors
        y_pos = 45
        for i, (player_id, color) in enumerate(self.player_colors.items()):
            if i >= 6:  # Limit to first 6 players in legend
                break
                
            # Draw color square
            cv2.rectangle(legend_bg, (10, y_pos - 10), (30, y_pos + 5), color, -1)
            
            # Player info
            stats = self.player_stats.get(player_id)
            if stats:
                label = f"P{player_id} ({len(stats.positions)} pts)"
            else:
                label = f"Player {player_id}"
            
            cv2.putText(legend_bg, label, (35, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
        
        # Position legend in top-right corner
        start_y = 10
        start_x = frame.shape[1] - legend_width - 10
        end_y = start_y + legend_height
        end_x = start_x + legend_width
        
        # Ensure legend fits within frame bounds
        if end_y > frame.shape[0]:
            start_y = frame.shape[0] - legend_height - 10
            end_y = frame.shape[0] - 10
        if end_x > frame.shape[1]:
            start_x = frame.shape[1] - legend_width - 10
            end_x = frame.shape[1] - 10
        
        # Overlay legend with transparency
        frame[start_y:end_y, start_x:end_x] = cv2.addWeighted(
            frame[start_y:end_y, start_x:end_x], 0.3, legend_bg, 0.7, 0)
        
        return frame
    
    def generate_game_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive game analysis report"""
        analysis = {
            "game_summary": {
                "total_frames_processed": self.frame_count,
                "players_detected": len(self.player_stats),
                "segmentation_enabled": self.use_sam_segmentation,
                "analysis_timestamp": datetime.now().isoformat(),
                "player_colors": {str(k): v for k, v in self.player_colors.items()}
            },
            "player_analyses": {},
            "team_statistics": self._generate_team_stats(),
            "game_flow": self._analyze_game_flow(),
            "key_insights": self._generate_insights(),
            "detection_quality": self._assess_detection_quality()
        }
        
        # Generate individual player analyses
        for player_id in self.player_stats:
            analysis["player_analyses"][player_id] = self.generate_player_analysis(player_id)
        
        return analysis
    
    def _assess_detection_quality(self) -> Dict[str, Any]:
        """Assess the quality of player detection and tracking"""
        if not self.player_stats:
            return {"quality": "No players detected", "score": 0.0}
        
        # Calculate detection quality metrics
        avg_positions = np.mean([len(stats.positions) for stats in self.player_stats.values()])
        tracking_consistency = np.mean([
            len(stats.positions) / max(1, self.frame_count) 
            for stats in self.player_stats.values()
        ])
        
        # Quality assessment
        if tracking_consistency > 0.8:
            quality = "Excellent"
            score = 95.0
        elif tracking_consistency > 0.6:
            quality = "Good"
            score = 80.0
        elif tracking_consistency > 0.4:
            quality = "Fair"
            score = 65.0
        else:
            quality = "Poor"
            score = 40.0
        
        return {
            "quality": quality,
            "score": score,
            "tracking_consistency": tracking_consistency,
            "average_positions_per_player": avg_positions,
            "total_unique_players": len(self.player_stats)
        }
    
    def _generate_team_stats(self) -> Dict[str, Any]:
        """Generate team-level statistics"""
        total_shots = sum(stats.shot_attempts for stats in self.player_stats.values())
        total_makes = sum(stats.shots_made for stats in self.player_stats.values())
        total_touches = sum(stats.ball_touches for stats in self.player_stats.values())
        
        return {
            "total_shot_attempts": total_shots,
            "total_shots_made": total_makes,
            "team_shooting_percentage": (total_makes / total_shots * 100) if total_shots > 0 else 0,
            "total_ball_touches": total_touches,
            "average_player_speed": np.mean([stats.average_speed for stats in self.player_stats.values()]) if self.player_stats else 0,
            "most_active_player": max(self.player_stats.items(), 
                                    key=lambda x: len(x[1].positions))[0] if self.player_stats else None
        }
    
    def _analyze_game_flow(self) -> Dict[str, Any]:
        """Analyze game flow and momentum"""
        return {
            "possession_changes": self.game_analytics.total_possessions,
            "fast_breaks": self.game_analytics.fast_breaks,
            "turnovers": self.game_analytics.turnovers,
            "game_pace": "High" if len(self.player_stats) > 8 else "Moderate",
            "player_interaction_density": len(self.player_stats) / max(1, self.frame_count / 100)
        }
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from the enhanced analysis"""
        insights = []
        
        if self.player_stats:
            # Detection quality insight
            detection_quality = self._assess_detection_quality()
            insights.append(f"Player detection quality: {detection_quality['quality']} "
                          f"(Tracking consistency: {detection_quality['tracking_consistency']:.1%})")
            
            # Best shooter
            best_shooter = max(self.player_stats.items(), 
                             key=lambda x: x[1].shooting_percentage, default=(None, None))
            if best_shooter[0] and best_shooter[1].shot_attempts > 3:
                insights.append(f"Player {best_shooter[0]} has the highest shooting percentage at {best_shooter[1].shooting_percentage:.1f}%")
            
            # Most active player
            most_active = max(self.player_stats.items(), 
                            key=lambda x: x[1].distance_covered, default=(None, None))
            if most_active[0]:
                insights.append(f"Player {most_active[0]} covered the most distance with {most_active[1].distance_covered:.1f} pixels")
            
            # Court coverage analysis
            zone_usage = defaultdict(int)
            for stats in self.player_stats.values():
                for zone in stats.time_in_zones:
                    zone_usage[zone] += 1
            
            if zone_usage:
                most_used_zone = max(zone_usage.items(), key=lambda x: x[1])
                insights.append(f"Most contested area: {most_used_zone[0]} ({most_used_zone[1]} players)")
            
            # SAM2 segmentation insight
            if self.use_sam_segmentation:
                insights.append(f"SAM2 whole-body segmentation active for {len(self.player_colors)} players with unique color coding")
        
        return insights
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to file"""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the analysis to JSON-serializable format
        serializable_analysis = convert_numpy_types(analysis)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        logger.info(f"Enhanced analysis saved to: {output_path}")
    
    def export_player_stats_csv(self, output_path: str):
        """Export enhanced player statistics to CSV"""
        data = []
        for player_id, stats in self.player_stats.items():
            row = {
                'player_id': player_id,
                'assigned_color_bgr': str(stats.assigned_color),
                'shot_attempts': stats.shot_attempts,
                'shots_made': stats.shots_made,
                'shooting_percentage': stats.shooting_percentage,
                'ball_touches': stats.ball_touches,
                'distance_covered': stats.distance_covered,
                'average_speed': stats.average_speed,
                'total_positions_tracked': len(stats.positions),
                'dominant_zone': max(stats.time_in_zones.items(), key=lambda x: x[1])[0] if stats.time_in_zones else 'unknown',
                'play_style': self._determine_play_style(stats)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Enhanced player statistics exported to: {output_path}")
    
    def debug_detection(self, frame: np.ndarray, detections: sv.Detections) -> None:
        """Enhanced debug method to analyze detection performance"""
        logger.info("=== ENHANCED DEBUG DETECTION ===")
        logger.info(f"Frame: {self.frame_count}, Shape: {frame.shape}")
        logger.info(f"Enhanced detection settings:")
        logger.info(f"  - Min confidence: {self.min_detection_confidence}")
        logger.info(f"  - Stable confidence: {self.stable_detection_confidence}")
        logger.info(f"  - IoU threshold: {self.iou_threshold}")
        logger.info(f"  - SAM2 segmentation: {self.use_sam_segmentation}")
        
        # Test general model detection
        results = self.yolo_general(frame, conf=0.05, verbose=False, classes=[0])  # Very low confidence, person only
        raw_detections = sv.Detections.from_ultralytics(results[0])
        
        logger.info(f"Raw person detections (conf=0.05): {len(raw_detections)}")
        if len(raw_detections) > 0:
            for i in range(min(5, len(raw_detections))):  # Show first 5
                conf = raw_detections.confidence[i] if hasattr(raw_detections, 'confidence') else 'N/A'
                bbox = raw_detections.xyxy[i]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                logger.info(f"  Detection {i}: confidence={conf:.3f}, area={area:.0f}")
        
        # Current processed detections
        logger.info(f"Processed detections: {len(detections)}")
        logger.info(f"Current players tracked: {len(self.player_stats)}")
        
        # Safe check for tracker_id
        if len(detections) > 0 and hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            tracked_count = len([tid for tid in detections.tracker_id if tid is not None])
            logger.info(f"Successfully tracked: {tracked_count}/{len(detections)}")
        else:
            logger.info("No tracking IDs available (detections not yet tracked)")
        
        logger.info("=== END ENHANCED DEBUG ===")