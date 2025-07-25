# Basketball Game Analysis System

A comprehensive basketball game analysis system using YOLO11 and SAM2 for player tracking, performance analysis, and game insights.

## 🏀 Features

### Player Analysis
- **Individual Performance Tracking**: Shot attempts, accuracy, ball touches, movement patterns
- **Play Style Identification**: Automatic classification (Post Player, Perimeter Shooter, Point Guard, etc.)
- **Strengths & Weaknesses**: AI-powered analysis of player capabilities
- **Movement Analytics**: Speed, distance covered, court coverage

### Team Analytics
- **Zone Analysis**: Time spent in different court areas (paint, three-point, mid-court)
- **Game Flow**: Possession tracking, fast breaks, turnovers
- **Heat Maps**: Visual representation of player positioning
- **Team Performance Metrics**: Composite scoring for offensive/defensive capabilities

### Advanced Visualizations
- **Interactive Dashboards**: Comprehensive player performance overview
- **Court Heat Maps**: Player position density analysis
- **Trajectory Plots**: Individual player movement patterns
- **Statistical Charts**: Shooting analysis, zone utilization, team comparisons

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (RTX 3050 or better recommended)
- CUDA Toolkit 11.8+

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download required models**:
   - Place `yolo11n.pt` in `./models/` directory
   - Place `sam2.1_b.pt` in `./models/` directory
   - Your custom basketball model should be at `./runs/detect/train9/weights/best.pt`

4. **Prepare video files**:
   - Place basketball game videos in `./media/` directory
   - Supported formats: MP4, AVI, MOV, MKV, WMV

### Simple Usage

**Option 1: Interactive Quick Start**
```bash
python quick_start.py
```

**Option 2: Command Line**
```bash
# Basic analysis
python script.py --video ./media/video_1.mp4

# Full analysis with visualizations
python script.py --video ./media/video_1.mp4 --visualize

# Complete analysis with video output
python script.py --video ./media/video_1.mp4 --visualize --save-video
```

## 📊 Output Files

After analysis, you'll find results in the `./output/` directory:

### Analysis Results
- `analysis/analysis_results.json` - Complete analysis data
- `statistics/player_statistics.csv` - Player stats in CSV format

### Visualizations (when `--visualize` is used)
- `visualizations/player_dashboard.html` - Interactive player dashboard
- `visualizations/court_heatmap.png` - Court heat map
- `visualizations/player_trajectories.png` - Movement trajectories
- `visualizations/team_performance.png` - Team performance charts
- `visualizations/shooting_analysis.png` - Detailed shooting stats
- `visualizations/zone_analysis.png` - Court zone utilization
- `visualizations/analysis_report.html` - Comprehensive HTML report

### Video Output (when `--save-video` is used)
- `videos/analyzed_[filename].mp4` - Annotated video with overlays

## ⚡ Performance Optimization

### RTX 3050 Specific Settings
- **GPU Memory**: Configured for 80% usage (20% reserved for system)
- **Batch Processing**: Optimized batch size of 4
- **Half Precision**: FP16 enabled for better performance
- **CUDA Optimization**: Benchmark mode enabled

### Performance Tips
1. **Close unnecessary applications** while running analysis
2. **Use SSD storage** for video files if possible
3. **Monitor GPU temperature** during long analyses
4. **Consider lower resolution videos** for faster processing

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Court zones (modify for different court layouts)
COURT_ZONES = {
    "paint_left": (0.078, 0.278, 0.313, 0.722),
    # ... other zones
}

# Performance settings
BATCH_SIZE = 4  # Increase for more powerful GPUs
GPU_MEMORY_FRACTION = 0.8  # Adjust based on available memory
```

## 📈 Understanding the Analysis

### Player Metrics
- **Shooting Percentage**: Made shots / Attempted shots × 100
- **Ball Touches**: Number of times player had ball possession
- **Distance Covered**: Total movement distance in pixels
- **Average Speed**: Movement speed in pixels per second
- **Zone Time**: Seconds spent in each court area

### Play Style Classification
- **Post Player/Center**: High paint area usage (>40%)
- **Perimeter Shooter**: High three-point area usage (>30%)
- **Point Guard/Playmaker**: High mobility (speed >200 px/s)
- **Mid-Range Specialist**: High mid-court usage (>40%)
- **Versatile Player**: Balanced across multiple areas

### Team Analytics
- **Possession Changes**: Estimated ball possession transfers
- **Fast Breaks**: Quick transition plays detected
- **Zone Popularity**: Most contested court areas
- **Game Pace**: High/Moderate based on player activity

## 🎯 Advanced Usage

### Custom Model Training
If you have your own basketball dataset:

```bash
# Train custom YOLO model
yolo detect train data=./dataset/basketball/data.yaml model=yolo11n.pt epochs=100 imgsz=640

# Use trained model
python script.py --video video.mp4 --model ./runs/detect/train/weights/best.pt
```

### Batch Processing
Process multiple videos:

```python
import glob
from basketball_analyzer import BasketballAnalyzer

analyzer = BasketballAnalyzer()
for video_path in glob.glob("./media/*.mp4"):
    results = analyzer.process_video(video_path)
    print(f"Processed: {video_path}")
```

### Custom Analysis
Extend the system with custom metrics:

```python
class CustomAnalyzer(BasketballAnalyzer):
    def custom_metric(self, player_stats):
        # Your custom analysis logic
        return analysis_results
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in `config.py`
   - Reduce `GPU_MEMORY_FRACTION`
   - Close other GPU-using applications

2. **Model Not Found**
   - Verify model paths in `config.py`
   - Download models to correct directories
   - Check file permissions

3. **Slow Performance**
   - Ensure CUDA is properly installed
   - Check GPU utilization with `nvidia-smi`
   - Consider using smaller video resolution

4. **No Players Detected**
   - Lower `CONFIDENCE_THRESHOLD`
   - Check video quality and lighting
   - Verify custom model is basketball-specific

### Performance Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check Python process
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 Technical Details

### Architecture
- **Detection**: YOLO11 for object detection (players, ball, rim)
- **Tracking**: ByteTracker for multi-object tracking
- **Segmentation**: SAM2 for precise object boundaries
- **Analytics**: Custom algorithms for basketball-specific metrics

### Dependencies
- **ultralytics**: YOLO11 and SAM2 models
- **supervision**: Computer vision utilities
- **opencv-python**: Video processing
- **torch**: Deep learning framework
- **pandas**: Data analysis
- **matplotlib/plotly**: Visualizations

### Hardware Requirements
- **Minimum**: GTX 1060 6GB, 8GB RAM
- **Recommended**: RTX 3050+ 8GB, 16GB RAM
- **Optimal**: RTX 3070+ 12GB, 32GB RAM

## 🤝 Contributing

Feel free to extend the system with:
- Additional player metrics
- New visualization types
- Performance optimizations
- Support for different sports

## 📄 License

This project is designed for educational and research purposes. Please ensure you have appropriate rights for any video content you analyze.

## 🆘 Support

For issues or questions:
1. Check this README for common solutions
2. Review the error logs in `basketball_analysis.log`
3. Verify your setup matches the requirements
4. Consider hardware limitations for performance issues

---

**Happy Analyzing! 🏀📊**
