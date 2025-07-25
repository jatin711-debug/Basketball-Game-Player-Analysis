"""
Advanced Visualization Module for Basketball Analysis
Creates comprehensive charts, graphs and visual reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import cv2
from basketball_analyzer import PlayerStats, GameAnalytics

class BasketballVisualizer:
    """Advanced visualization system for basketball analytics"""
    
    def __init__(self, output_dir: str = "./output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_player_dashboard(self, player_stats: Dict[int, PlayerStats], 
                              game_analytics: GameAnalytics) -> str:
        """Create comprehensive player performance dashboard"""
        
        # Prepare data
        players_data = []
        for player_id, stats in player_stats.items():
            players_data.append({
                'Player ID': f'P{player_id}',
                'Shot Attempts': stats.shot_attempts,
                'Shots Made': stats.shots_made,
                'Shooting %': stats.shooting_percentage,
                'Ball Touches': stats.ball_touches,
                'Distance': stats.distance_covered,
                'Avg Speed': stats.average_speed,
                'Zones Played': len(stats.time_in_zones)
            })
        
        df = pd.DataFrame(players_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Shooting Performance', 'Ball Involvement', 'Movement Analysis',
                'Zone Coverage', 'Speed Distribution', 'Shot Accuracy',
                'Player Comparison', 'Distance vs Touches', 'Performance Radar'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatterpolar"}]
            ]
        )
        
        # 1. Shooting Performance
        fig.add_trace(
            go.Bar(x=df['Player ID'], y=df['Shot Attempts'], 
                   name='Attempts', marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=df['Player ID'], y=df['Shots Made'], 
                   name='Made', marker_color='green'),
            row=1, col=1
        )
        
        # 2. Ball Involvement
        fig.add_trace(
            go.Bar(x=df['Player ID'], y=df['Ball Touches'], 
                   marker_color='orange'),
            row=1, col=2
        )
        
        # 3. Movement Analysis
        fig.add_trace(
            go.Scatter(x=df['Distance'], y=df['Avg Speed'],
                      text=df['Player ID'], mode='markers+text',
                      marker=dict(size=12, color='red')),
            row=1, col=3
        )
        
        # 4. Zone Coverage
        fig.add_trace(
            go.Bar(x=df['Player ID'], y=df['Zones Played'],
                   marker_color='purple'),
            row=2, col=1
        )
        
        # 5. Speed Distribution
        fig.add_trace(
            go.Histogram(x=df['Avg Speed'], nbinsx=10,
                        marker_color='cyan'),
            row=2, col=2
        )
        
        # 6. Shot Accuracy
        fig.add_trace(
            go.Bar(x=df['Player ID'], y=df['Shooting %'],
                   marker_color='gold'),
            row=2, col=3
        )
        
        # 7. Player Comparison (normalized metrics)
        normalized_df = df.copy()
        for col in ['Shot Attempts', 'Ball Touches', 'Distance', 'Avg Speed']:
            max_val = normalized_df[col].max()
            if max_val > 0:
                normalized_df[col] = normalized_df[col] / max_val * 100
        
        fig.add_trace(
            go.Bar(x=normalized_df['Player ID'], y=normalized_df['Shot Attempts'],
                   name='Shots (norm)', marker_color='lightcoral'),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=normalized_df['Player ID'], y=normalized_df['Ball Touches'],
                   name='Touches (norm)', marker_color='lightgreen'),
            row=3, col=1
        )
        
        # 8. Distance vs Touches correlation
        fig.add_trace(
            go.Scatter(x=df['Distance'], y=df['Ball Touches'],
                      text=df['Player ID'], mode='markers+text',
                      marker=dict(size=10, color='blue')),
            row=3, col=2
        )
        
        # 9. Performance Radar (for top player)
        if len(df) > 0:
            top_player = df.loc[df['Shooting %'].idxmax()]
            categories = ['Shot Attempts', 'Shots Made', 'Ball Touches', 'Distance', 'Avg Speed']
            values = [top_player[cat] for cat in categories]
            max_values = [df[cat].max() for cat in categories]
            normalized_values = [v/m*100 if m > 0 else 0 for v, m in zip(values, max_values)]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=normalized_values,
                    theta=categories,
                    fill='toself',
                    name=f'Top Player: {top_player["Player ID"]}'
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Basketball Game Analysis Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        output_path = self.output_dir / "player_dashboard.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def create_court_heatmap(self, player_stats: Dict[int, PlayerStats], 
                           frame_shape: Tuple[int, int]) -> str:
        """Create court heatmap showing player positions"""
        
        # Create court background
        court_img = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        court_img[:] = (34, 139, 34)  # Forest green background
        
        # Draw court lines (simplified)
        self._draw_court_lines(court_img, frame_shape)
        
        # Create heatmap data
        heatmap_data = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        
        for player_id, stats in player_stats.items():
            for pos in stats.positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                    # Add Gaussian distribution around each position
                    for i in range(max(0, y-25), min(frame_shape[0], y+25)):
                        for j in range(max(0, x-25), min(frame_shape[1], x+25)):
                            distance = np.sqrt((i-y)**2 + (j-x)**2)
                            if distance <= 25:
                                heatmap_data[i, j] += np.exp(-distance**2 / 300)
        
        # Normalize and apply colormap
        heatmap_data = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_data.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with court background
        alpha = 0.6
        final_image = cv2.addWeighted(court_img, 1-alpha, heatmap_colored, alpha, 0)
        
        # Save heatmap
        output_path = self.output_dir / "court_heatmap.png"
        cv2.imwrite(str(output_path), final_image)
        
        return str(output_path)
    
    def _draw_court_lines(self, img: np.ndarray, frame_shape: Tuple[int, int]):
        """Draw simplified basketball court lines"""
        height, width = frame_shape
        
        # Court outline
        cv2.rectangle(img, (50, 50), (width-50, height-50), (255, 255, 255), 3)
        
        # Center line
        cv2.line(img, (width//2, 50), (width//2, height-50), (255, 255, 255), 2)
        
        # Center circle
        cv2.circle(img, (width//2, height//2), 80, (255, 255, 255), 2)
        
        # Three-point lines (simplified)
        cv2.ellipse(img, (150, height//2), (200, 250), 0, -90, 90, (255, 255, 255), 2)
        cv2.ellipse(img, (width-150, height//2), (200, 250), 0, 90, 270, (255, 255, 255), 2)
        
        # Paint areas
        cv2.rectangle(img, (50, height//2-100), (200, height//2+100), (255, 255, 255), 2)
        cv2.rectangle(img, (width-200, height//2-100), (width-50, height//2+100), (255, 255, 255), 2)
    
    def create_player_trajectory_plot(self, player_stats: Dict[int, PlayerStats]) -> str:
        """Create individual player trajectory visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Player Movement Trajectories', fontsize=16)
        
        # Select top 6 players by activity
        sorted_players = sorted(player_stats.items(), 
                              key=lambda x: len(x[1].positions), reverse=True)[:6]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_players)))
        
        for idx, (player_id, stats) in enumerate(sorted_players):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            if stats.positions:
                positions = np.array(stats.positions)
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]
                
                # Plot trajectory
                ax.plot(x_coords, y_coords, color=colors[idx], alpha=0.7, linewidth=2)
                ax.scatter(x_coords[::10], y_coords[::10], 
                          color=colors[idx], s=20, alpha=0.8)
                
                # Start and end points
                ax.scatter(x_coords[0], y_coords[0], color='green', s=100, marker='o', label='Start')
                ax.scatter(x_coords[-1], y_coords[-1], color='red', s=100, marker='X', label='End')
                
                ax.set_title(f'Player {player_id} Trajectory\n'
                           f'Distance: {stats.distance_covered:.1f} | '
                           f'Avg Speed: {stats.average_speed:.1f}')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()  # Invert Y to match image coordinates
        
        # Remove empty subplots
        for idx in range(len(sorted_players), 6):
            row, col = idx // 3, idx % 3
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        output_path = self.output_dir / "player_trajectories.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_team_performance_chart(self, player_stats: Dict[int, PlayerStats]) -> str:
        """Create team performance analysis charts"""
        
        # Prepare data
        metrics = {
            'Offensive': [],
            'Defensive': [],
            'Playmaking': [],
            'Athleticism': []
        }
        
        player_ids = []
        
        for player_id, stats in player_stats.items():
            player_ids.append(f'P{player_id}')
            
            # Calculate composite metrics
            offensive = (stats.shot_attempts * 2 + stats.shots_made * 3) / 5 if stats.shot_attempts > 0 else 0
            defensive = min(stats.distance_covered / 100, 10)  # Capped at 10
            playmaking = min(stats.ball_touches / 10, 10)  # Capped at 10
            athleticism = min(stats.average_speed / 30, 10)  # Capped at 10
            
            metrics['Offensive'].append(offensive)
            metrics['Defensive'].append(defensive)
            metrics['Playmaking'].append(playmaking)
            metrics['Athleticism'].append(athleticism)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(player_ids))
        width = 0.6
        
        bottom = np.zeros(len(player_ids))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (metric, values) in enumerate(metrics.items()):
            ax.bar(x, values, width, label=metric, bottom=bottom, color=colors[i])
            bottom += values
        
        ax.set_xlabel('Players')
        ax.set_ylabel('Performance Score')
        ax.set_title('Team Performance Analysis - Composite Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(player_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "team_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_shooting_analysis(self, player_stats: Dict[int, PlayerStats]) -> str:
        """Create detailed shooting analysis visualization"""
        
        # Filter players with shot attempts
        shooting_data = []
        for player_id, stats in player_stats.items():
            if stats.shot_attempts > 0:
                shooting_data.append({
                    'Player': f'P{player_id}',
                    'Attempts': stats.shot_attempts,
                    'Made': stats.shots_made,
                    'Percentage': stats.shooting_percentage,
                    'Efficiency': stats.shots_made / max(stats.shot_attempts, 1) * 100
                })
        
        if not shooting_data:
            return "No shooting data available"
        
        df = pd.DataFrame(shooting_data)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Shooting Analysis Dashboard', fontsize=16)
        
        # 1. Shot attempts vs makes
        x_pos = range(len(df))
        ax1.bar([x - 0.2 for x in x_pos], df['Attempts'], 0.4, label='Attempts', alpha=0.8)
        ax1.bar([x + 0.2 for x in x_pos], df['Made'], 0.4, label='Made', alpha=0.8)
        ax1.set_xlabel('Players')
        ax1.set_ylabel('Shots')
        ax1.set_title('Shot Attempts vs Made')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df['Player'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Shooting percentage
        bars = ax2.bar(df['Player'], df['Percentage'], color='green', alpha=0.7)
        ax2.set_xlabel('Players')
        ax2.set_ylabel('Shooting Percentage (%)')
        ax2.set_title('Shooting Percentage by Player')
        ax2.set_xticklabels(df['Player'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, df['Percentage']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # 3. Efficiency scatter plot
        ax3.scatter(df['Attempts'], df['Percentage'], s=100, alpha=0.7, color='red')
        for i, player in enumerate(df['Player']):
            ax3.annotate(player, (df['Attempts'].iloc[i], df['Percentage'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax3.set_xlabel('Shot Attempts')
        ax3.set_ylabel('Shooting Percentage (%)')
        ax3.set_title('Volume vs Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Player ranking
        df_sorted = df.sort_values('Percentage', ascending=True)
        bars = ax4.barh(df_sorted['Player'], df_sorted['Percentage'], color='orange', alpha=0.7)
        ax4.set_xlabel('Shooting Percentage (%)')
        ax4.set_title('Player Shooting Rankings')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "shooting_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_zone_analysis(self, player_stats: Dict[int, PlayerStats]) -> str:
        """Create court zone utilization analysis"""
        
        # Aggregate zone data
        zone_totals = {}
        player_zone_data = {}
        
        for player_id, stats in player_stats.items():
            player_zone_data[player_id] = stats.time_in_zones
            for zone, time in stats.time_in_zones.items():
                zone_totals[zone] = zone_totals.get(zone, 0) + time
        
        if not zone_totals:
            return "No zone data available"
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Court Zone Analysis', fontsize=16)
        
        # 1. Overall zone utilization
        zones = list(zone_totals.keys())
        times = list(zone_totals.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        
        wedges, texts, autotexts = ax1.pie(times, labels=zones, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Overall Zone Utilization')
        
        # 2. Zone utilization bar chart
        ax2.bar(zones, times, color=colors)
        ax2.set_xlabel('Court Zones')
        ax2.set_ylabel('Total Time (seconds)')
        ax2.set_title('Time Spent in Each Zone')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Player zone preferences heatmap
        if len(player_zone_data) > 0:
            # Create matrix for heatmap
            all_zones = set()
            for zones in player_zone_data.values():
                all_zones.update(zones.keys())
            all_zones = sorted(list(all_zones))
            
            matrix = []
            player_labels = []
            
            for player_id, zones in player_zone_data.items():
                if zones:  # Only include players with zone data
                    row = [zones.get(zone, 0) for zone in all_zones]
                    matrix.append(row)
                    player_labels.append(f'P{player_id}')
            
            if matrix:
                im = ax3.imshow(matrix, cmap='YlOrRd', aspect='auto')
                ax3.set_xticks(range(len(all_zones)))
                ax3.set_xticklabels(all_zones, rotation=45, ha='right')
                ax3.set_yticks(range(len(player_labels)))
                ax3.set_yticklabels(player_labels)
                ax3.set_title('Player Zone Preferences')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label('Time (seconds)')
        
        # 4. Zone activity levels
        zone_activity = {zone: len([p for p, zones in player_zone_data.items() 
                                  if zone in zones and zones[zone] > 1])
                        for zone in zone_totals.keys()}
        
        ax4.bar(zone_activity.keys(), zone_activity.values(), color='purple', alpha=0.7)
        ax4.set_xlabel('Court Zones')
        ax4.set_ylabel('Number of Active Players')
        ax4.set_title('Zone Popularity (Players per Zone)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "zone_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_analysis_report(self, analysis_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML analysis report"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Basketball Game Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #1f4e79; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
                .player-card {{ border: 1px solid #ccc; margin: 10px; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .insight {{ background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Basketball Game Analysis Report</h1>
                <p>Generated on: {analysis_data.get('game_summary', {}).get('analysis_timestamp', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Game Summary</h2>
                <div class="metric">
                    <strong>Total Frames:</strong> {analysis_data.get('game_summary', {}).get('total_frames_processed', 0)}
                </div>
                <div class="metric">
                    <strong>Players Detected:</strong> {analysis_data.get('game_summary', {}).get('players_detected', 0)}
                </div>
            </div>
            
            <div class="section">
                <h2>Team Statistics</h2>
                <div class="metric">
                    <strong>Total Shots:</strong> {analysis_data.get('team_statistics', {}).get('total_shot_attempts', 0)}
                </div>
                <div class="metric">
                    <strong>Team Shooting %:</strong> {analysis_data.get('team_statistics', {}).get('team_shooting_percentage', 0):.1f}%
                </div>
                <div class="metric">
                    <strong>Ball Touches:</strong> {analysis_data.get('team_statistics', {}).get('total_ball_touches', 0)}
                </div>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                {''.join([f'<div class="insight">• {insight}</div>' for insight in analysis_data.get('key_insights', [])])}
            </div>
            
            <div class="section">
                <h2>Player Performance</h2>
                {self._generate_player_cards_html(analysis_data.get('player_analyses', {}))}
            </div>
            
        </body>
        </html>
        """
        
        output_path = self.output_dir / "analysis_report.html"
        with open(output_path, 'w') as f:
            f.write(html_template)
        
        return str(output_path)
    
    def _generate_player_cards_html(self, player_analyses: Dict[str, Any]) -> str:
        """Generate HTML for player performance cards"""
        cards_html = ""
        
        for player_id, analysis in player_analyses.items():
            metrics = analysis.get('performance_metrics', {})
            strengths = analysis.get('strengths', [])
            weaknesses = analysis.get('weaknesses', [])
            play_style = analysis.get('play_style', 'Unknown')
            
            card_html = f"""
            <div class="player-card">
                <h3>Player {player_id}</h3>
                <p><strong>Play Style:</strong> {play_style}</p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Shot Attempts</td><td>{metrics.get('shots_attempted', 0)}</td></tr>
                    <tr><td>Shooting %</td><td>{metrics.get('shooting_percentage', 0):.1f}%</td></tr>
                    <tr><td>Ball Touches</td><td>{metrics.get('ball_touches', 0)}</td></tr>
                    <tr><td>Distance Covered</td><td>{metrics.get('distance_covered', 0):.1f}</td></tr>
                    <tr><td>Average Speed</td><td>{metrics.get('average_speed', 0):.1f}</td></tr>
                </table>
                <p><strong>Strengths:</strong> {', '.join(strengths)}</p>
                <p><strong>Areas for Improvement:</strong> {', '.join(weaknesses)}</p>
            </div>
            """
            cards_html += card_html
        
        return cards_html
