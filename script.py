"""
Main execution script for Basketball Game Analysis System
Comprehensive analysis using YOLO11 and SAM2
Optimized for RTX 3050 GPU
"""

import argparse
import sys
from pathlib import Path
import logging
import time

# Import our custom modules
from basketball_analyzer import BasketballAnalyzer
from visualizer import BasketballVisualizer
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basketball_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Basketball Game Analysis System')
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--model', '-m', type=str, 
                       default='./runs/detect/train9/weights/best.pt',
                       help='Path to custom YOLO model')
    parser.add_argument('--confidence', '-c', type=float, default=0.4,
                       help='Detection confidence threshold')
    parser.add_argument('--device', '-d', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization charts')
    parser.add_argument('--save-video', action='store_true',
                       help='Save annotated video output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    # Create output directories
    Config.OUTPUT_DIR = Path(args.output)
    Config.create_directories()
    
    # Configure GPU optimization
    Config.optimize_gpu()
    
    logger.info("=" * 60)
    logger.info("Basketball Game Analysis System Starting")
    logger.info("=" * 60)
    logger.info(f"Video: {args.video}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.model}")
    
    try:
        # Initialize analyzer
        logger.info("Initializing Basketball Analyzer...")
        analyzer = BasketballAnalyzer(
            custom_model_path=args.model,
            device=args.device
        )
        
        # Set confidence threshold
        analyzer.confidence_threshold = args.confidence
        
        # Process video
        start_time = time.time()
        
        output_video_path = None
        if args.save_video:
            output_video_path = Config.VIDEO_DIR / f"analyzed_{Path(args.video).stem}.mp4"
        
        logger.info("Starting video processing...")
        analysis_results = analyzer.process_video(
            video_path=args.video,
            output_path=str(output_video_path) if output_video_path else None
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        # Save analysis results
        analysis_file = Config.ANALYSIS_DIR / "analysis_results.json"
        analyzer.save_analysis(analysis_results, str(analysis_file))
        
        # Export player statistics
        stats_file = Config.STATS_DIR / "player_statistics.csv"
        analyzer.export_player_stats_csv(str(stats_file))
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            visualizer = BasketballVisualizer(str(Config.OUTPUT_DIR / "visualizations"))
            
            # Create all visualization types
            dashboard_path = visualizer.create_player_dashboard(
                analyzer.player_stats, analyzer.game_analytics
            )
            
            heatmap_path = visualizer.create_court_heatmap(
                analyzer.player_stats, (720, 1280)  # Assuming standard frame size
            )
            
            trajectory_path = visualizer.create_player_trajectory_plot(
                analyzer.player_stats
            )
            
            team_chart_path = visualizer.create_team_performance_chart(
                analyzer.player_stats
            )
            
            shooting_analysis_path = visualizer.create_shooting_analysis(
                analyzer.player_stats
            )
            
            zone_analysis_path = visualizer.create_zone_analysis(
                analyzer.player_stats
            )
            
            # Generate comprehensive report
            report_path = visualizer.generate_analysis_report(analysis_results)
            
            logger.info("Visualizations generated:")
            logger.info(f"  - Dashboard: {dashboard_path}")
            logger.info(f"  - Heatmap: {heatmap_path}")
            logger.info(f"  - Trajectories: {trajectory_path}")
            logger.info(f"  - Team Performance: {team_chart_path}")
            logger.info(f"  - Shooting Analysis: {shooting_analysis_path}")
            logger.info(f"  - Zone Analysis: {zone_analysis_path}")
            logger.info(f"  - Full Report: {report_path}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Players detected: {len(analyzer.player_stats)}")
        logger.info(f"Total frames processed: {analyzer.frame_count}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Average FPS: {analyzer.frame_count / processing_time:.1f}")
        
        # Player statistics summary
        if analyzer.player_stats:
            logger.info("\nTOP PERFORMERS:")
            
            # Best shooter
            best_shooter = max(analyzer.player_stats.items(), 
                             key=lambda x: x[1].shooting_percentage if x[1].shot_attempts > 2 else 0)
            if best_shooter[1].shot_attempts > 2:
                logger.info(f"  Best Shooter: Player {best_shooter[0]} "
                          f"({best_shooter[1].shooting_percentage:.1f}%)")
            
            # Most active
            most_active = max(analyzer.player_stats.items(), 
                            key=lambda x: x[1].distance_covered)
            logger.info(f"  Most Active: Player {most_active[0]} "
                       f"(Distance: {most_active[1].distance_covered:.1f})")
            
            # Ball handler
            ball_handler = max(analyzer.player_stats.items(), 
                             key=lambda x: x[1].ball_touches)
            logger.info(f"  Primary Ball Handler: Player {ball_handler[0]} "
                       f"({ball_handler[1].ball_touches} touches)")
        
        logger.info(f"\nResults saved to: {Config.OUTPUT_DIR}")
        logger.info("Analysis completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.exception("Full error details:")
        return 1

if __name__ == "__main__":
    exit_code = main()

