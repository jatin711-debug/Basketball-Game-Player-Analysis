"""
Basketball Game Analysis System using YOLO11 and SAM2
Production-ready implementation for RTX 3050 GPU optimization
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
    """Main basketball analysis system"""
    
    def __init__(self, 
                 yolo_model_path: str = "./models/yolo11n.pt",
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
        
        # Initialize tracking components
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=50,
            minimum_matching_threshold=0.8,
            frame_rate=30
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
        
        # Performance optimization settings
        self.batch_size = 4  # Optimal for RTX 3050
        self.confidence_threshold = 0.25  # Lowered for better detection
        self.iou_threshold = 0.5
        
    def _configure_optimal_device(self) -> str:
        """Configure optimal device settings for RTX 3050"""
        if torch.cuda.is_available():
            # RTX 3050 specific optimizations
            torch.cuda.set_per_process_memory_fraction(0.8)  # Leave some memory for system
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
            
            # Load custom basketball-specific model
            if Path(custom_path).exists():
                try:
                    logger.info(f"Loading custom basketball model from: {custom_path}")
                    self.yolo_basketball = YOLO(custom_path)
                    self.yolo_basketball.to(self.device)
                    logger.info("Custom basketball model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load custom model: {e}")
                    logger.info("Falling back to general YOLO model")
                    self.yolo_basketball = self.yolo_general
            else:
                logger.warning(f"Custom basketball model not found at: {custom_path}")
                logger.info("Using general YOLO model")
                self.yolo_basketball = self.yolo_general
            
            # Load SAM2 for precise segmentation
            logger.info(f"Loading SAM model from: {sam_path}")
            self.sam_model = SAM(sam_path)
            self.sam_model.to(self.device)
            
            # Optimize models for inference
            if self.device == 'cuda':
                try:
                    self.yolo_general.model.half()  # Use FP16 for better performance
                    if self.yolo_basketball != self.yolo_general:
                        self.yolo_basketball.model.half()
                    logger.info("Enabled FP16 optimization for CUDA")
                except Exception as e:
                    logger.warning(f"Could not enable FP16: {e}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _setup_annotators(self):
        """Setup visualization components"""
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.5,
            text_color=sv.Color.WHITE
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
        """Enhanced object detection with basketball-specific optimizations"""
        try:
            # Use custom basketball model for better detection
            results = self.yolo_basketball(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                half=True if self.device == 'cuda' else False
            )[0]
            
            detections = sv.Detections.from_ultralytics(results)
            
            # Debug: Print detection info
            if len(detections) > 0:
                logger.info(f"Raw detections: {len(detections)} objects found")
                if hasattr(detections, 'class_id') and detections.class_id is not None:
                    unique_classes = np.unique(detections.class_id)
                    logger.info(f"Detected classes: {unique_classes}")
            
            # For custom basketball model, use specific classes from data.yaml
            if self.yolo_basketball != self.yolo_general:
                # Based on your data.yaml: 0='unknown', 1='basketball', 2='people', 3='rim'
                relevant_classes = [1, 2, 3]  # basketball, people, rim
                if hasattr(detections, 'class_id') and detections.class_id is not None:
                    mask = np.isin(detections.class_id, relevant_classes)
                    detections = detections[mask]
                    logger.info(f"After filtering: {len(detections)} relevant objects")
            else:
                # For general YOLO model, use person class (0) and try to detect all objects
                # Don't filter too aggressively - let the tracker handle it
                logger.info(f"Using general model - keeping all {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            # Return empty detections to prevent crash
            return sv.Detections.empty()
    
    def track_players(self, detections: sv.Detections) -> sv.Detections:
        """Advanced player tracking with state management"""
        if len(detections) == 0:
            return detections
            
        try:
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Debug tracking
            if len(tracked_detections) > 0:
                logger.debug(f"Tracking: {len(tracked_detections)} objects tracked")
            
            # Update player statistics
            for i in range(len(tracked_detections)):
                if hasattr(tracked_detections, 'tracker_id') and tracked_detections.tracker_id is not None:
                    if i < len(tracked_detections.tracker_id):
                        player_id = tracked_detections.tracker_id[i]
                        
                        if player_id not in self.player_stats:
                            self.player_stats[player_id] = PlayerStats(player_id=player_id)
                            logger.info(f"New player detected: {player_id}")
                        
                        # Extract position
                        if i < len(tracked_detections.xyxy):
                            x1, y1, x2, y2 = tracked_detections.xyxy[i]
                            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                            
                            # Update position history
                            self.player_stats[player_id].positions.append((center_x, center_y))
                            
                            # Calculate speed if we have previous positions
                            if len(self.player_stats[player_id].positions) >= 2:
                                prev_pos = self.player_stats[player_id].positions[-2]
                                current_pos = self.player_stats[player_id].positions[-1]
                                distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + 
                                                 (current_pos[1] - prev_pos[1])**2)
                                speed = distance * self.fps  # pixels per second
                                self.player_stats[player_id].speed_data.append(speed)
                            
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
                elif class_id == 2:  # people
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
                    self.player_stats[detection.tracker_id].ball_touches += 1
    
    def generate_player_analysis(self, player_id: int) -> Dict[str, Any]:
        """Generate comprehensive player analysis"""
        if player_id not in self.player_stats:
            return {}
        
        stats = self.player_stats[player_id]
        
        analysis = {
            "player_id": player_id,
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
        logger.info(f"Starting video analysis: {video_path}")
        
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
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Write frame if output specified
                if output_path:
                    out.write(annotated_frame)
                
                # Progress logging
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / self.frame_count) * (total_frames - self.frame_count)
                    logger.info(f"Progress: {progress:.1f}% - ETA: {eta:.1f}s")
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        # Generate final analysis
        analysis_results = self.generate_game_analysis()
        
        logger.info("Video analysis completed successfully")
        return analysis_results
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with all analysis components"""
        # Object detection
        detections = self.detect_objects(frame)
        
        # Debug first frame
        if self.frame_count == 1:
            self.debug_detection(frame)
        
        # Player tracking
        tracked_detections = self.track_players(detections)
        
        # Basketball event analysis
        self.analyze_basketball_events(frame, tracked_detections)
        
        # Create annotations
        annotated_frame = self._annotate_frame(frame, tracked_detections)
        
        return annotated_frame
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Create comprehensive frame annotations"""
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels
        if len(detections) > 0:
            # Create labels for each detection
            labels = []
            for i in range(len(detections)):
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    if i < len(detections.tracker_id):
                        player_id = detections.tracker_id[i]
                        if player_id in self.player_stats:
                            stats = self.player_stats[player_id]
                            label = f"P{player_id} | Speed: {stats.average_speed:.1f}"
                        else:
                            label = f"P{player_id}"
                    else:
                        label = "Player"
                else:
                    label = "Player"
                labels.append(label)
            
            # Apply annotations
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
            
            # Add traces for player movement
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        
        # Add court zones overlay
        annotated_frame = self._draw_court_zones(annotated_frame)
        
        # Add statistics overlay
        annotated_frame = self._add_stats_overlay(annotated_frame)
        
        return annotated_frame
    
    def _draw_court_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw basketball court zones"""
        overlay = frame.copy()
        
        for zone_name, (x1, y1, x2, y2) in self.court_zones.items():
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, zone_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Blend with original frame
        return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    
    def _add_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add real-time statistics overlay"""
        y_offset = 30
        
        # Game statistics
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Players Tracked: {len(self.player_stats)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Top performers
        if self.player_stats:
            top_scorer = max(self.player_stats.items(), 
                           key=lambda x: x[1].shots_made, default=(None, None))
            if top_scorer[0]:
                cv2.putText(frame, f"Top Scorer: P{top_scorer[0]} ({top_scorer[1].shots_made})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def generate_game_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive game analysis report"""
        analysis = {
            "game_summary": {
                "total_frames_processed": self.frame_count,
                "players_detected": len(self.player_stats),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "player_analyses": {},
            "team_statistics": self._generate_team_stats(),
            "game_flow": self._analyze_game_flow(),
            "key_insights": self._generate_insights()
        }
        
        # Generate individual player analyses
        for player_id in self.player_stats:
            analysis["player_analyses"][player_id] = self.generate_player_analysis(player_id)
        
        return analysis
    
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
            "average_player_speed": np.mean([stats.average_speed for stats in self.player_stats.values()]) if self.player_stats else 0
        }
    
    def _analyze_game_flow(self) -> Dict[str, Any]:
        """Analyze game flow and momentum"""
        return {
            "possession_changes": self.game_analytics.total_possessions,
            "fast_breaks": self.game_analytics.fast_breaks,
            "turnovers": self.game_analytics.turnovers,
            "game_pace": "High" if len(self.player_stats) > 8 else "Moderate"
        }
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        if self.player_stats:
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
                insights.append(f"Most contested area: {most_used_zone[0]}")
        
        return insights
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to file"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Analysis saved to: {output_path}")
    
    def export_player_stats_csv(self, output_path: str):
        """Export player statistics to CSV"""
        data = []
        for player_id, stats in self.player_stats.items():
            row = {
                'player_id': player_id,
                'shot_attempts': stats.shot_attempts,
                'shots_made': stats.shots_made,
                'shooting_percentage': stats.shooting_percentage,
                'ball_touches': stats.ball_touches,
                'distance_covered': stats.distance_covered,
                'average_speed': stats.average_speed,
                'dominant_zone': max(stats.time_in_zones.items(), key=lambda x: x[1])[0] if stats.time_in_zones else 'unknown'
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Player statistics exported to: {output_path}")
    
    def debug_detection(self, frame: np.ndarray) -> None:
        """Debug method to test detection on a single frame"""
        logger.info("=== DEBUG DETECTION ===")
        logger.info(f"Frame shape: {frame.shape}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Using model: {'custom' if self.yolo_basketball != self.yolo_general else 'general'}")
        
        # Test detection without filtering
        results = self.yolo_basketball(frame, conf=0.1, verbose=True)  # Very low confidence for debugging
        raw_detections = sv.Detections.from_ultralytics(results[0])
        
        logger.info(f"Raw detections (conf=0.1): {len(raw_detections)}")
        if hasattr(raw_detections, 'class_id') and raw_detections.class_id is not None:
            for i, class_id in enumerate(raw_detections.class_id):
                conf = raw_detections.confidence[i] if hasattr(raw_detections, 'confidence') else 'N/A'
                logger.info(f"  Detection {i}: class={class_id}, confidence={conf}")
        
        # Test with normal confidence
        normal_detections = self.detect_objects(frame)
        logger.info(f"Normal detections (conf={self.confidence_threshold}): {len(normal_detections)}")
        logger.info("=== END DEBUG ===")

# Example usage and configuration
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = BasketballAnalyzer(
        custom_model_path="./runs/detect/train9/weights/best.pt",
        device="auto"  # Will auto-configure for RTX 3050
    )
    
    # Process video
    video_path = "./media/video_1.mp4"
    output_video_path = "./media/analyzed_video.mp4"
    
    try:
        analysis_results = analyzer.process_video(video_path, output_video_path)
        
        # Save detailed analysis
        analyzer.save_analysis(analysis_results, "./analysis_results.json")
        
        # Export player stats
        analyzer.export_player_stats_csv("./player_statistics.csv")
        
        print("Analysis completed successfully!")
        print(f"Players detected: {len(analyzer.player_stats)}")
        print(f"Total frames processed: {analyzer.frame_count}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
