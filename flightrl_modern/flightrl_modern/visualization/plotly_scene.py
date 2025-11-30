#!/usr/bin/env python3
"""
Plotly-based 3D Trajectory Visualizer for Flightmare

Creates interactive HTML visualizations of drone trajectories with:
- 3D trajectory traces with color-coded rewards
- Animated drone marker with playback controls
- Ground plane and arena bounds
- Episode metrics sidebar
- Start/end/target point markers

Usage:
    from flightrl_modern.visualization.plotly_scene import create_trajectory_visualization
    
    html_path = create_trajectory_visualization(
        csv_path="./rollout_ep00.csv",
        output_path="./trajectory.html"
    )
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for trajectory visualization."""
    # Arena bounds
    arena_size: float = 20.0  # meters, half-extent
    arena_height: float = 10.0  # meters
    
    # Visual settings
    trajectory_width: float = 3.0
    marker_size: float = 8.0
    drone_marker_size: float = 12.0
    
    # Animation
    frame_duration: int = 50  # milliseconds per frame
    transition_duration: int = 0  # smooth transitions
    
    # Colors
    color_start: str = "#00ff00"  # Green
    color_end: str = "#ff6600"  # Orange
    color_failure: str = "#ff0000"  # Red
    color_trajectory: str = "Viridis"  # Colorscale for reward
    color_ground: str = "rgba(100, 100, 100, 0.3)"
    color_arena: str = "rgba(50, 50, 200, 0.1)"
    
    # Theme
    background_color: str = "#1a1a2e"
    paper_color: str = "#16213e"
    font_color: str = "#e8e8e8"
    grid_color: str = "rgba(255, 255, 255, 0.1)"


class TrajectoryVisualizer:
    """
    Creates interactive 3D visualizations of drone trajectories.
    
    Features:
    - Trajectory line with reward-based coloring
    - Animated drone marker
    - Ground plane and optional arena bounds
    - Metrics annotations
    - Playback controls
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
    
    def load_rollout(self, csv_path: str) -> pd.DataFrame:
        """Load rollout data from CSV."""
        df = pd.read_csv(csv_path)
        return df
    
    def create_visualization(
        self,
        df: pd.DataFrame,
        title: str = "Drone Trajectory",
        show_animation: bool = True,
        show_arena: bool = True,
        episode_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create an interactive 3D visualization.
        
        Args:
            df: DataFrame with rollout data
            title: Plot title
            show_animation: Include animation controls
            show_arena: Show arena bounds
            episode_summary: Optional episode summary dict
            
        Returns:
            HTML string of the visualization
        """
        import plotly.graph_objects as go
        
        cfg = self.config
        
        # Extract trajectory data
        x = df["pos_x"].values
        y = df["pos_y"].values
        z = df["pos_z"].values
        rewards = df["reward"].values
        times = df["time"].values
        steps = df["step"].values
        
        # Normalize rewards for coloring
        reward_min = rewards.min()
        reward_max = rewards.max()
        if reward_max - reward_min > 1e-6:
            reward_norm = (rewards - reward_min) / (reward_max - reward_min)
        else:
            reward_norm = np.ones_like(rewards) * 0.5
        
        # Create figure
        fig = go.Figure()
        
        # Add ground plane
        ground_size = cfg.arena_size
        
        fig.add_trace(go.Mesh3d(
            x=[-ground_size, ground_size, ground_size, -ground_size],
            y=[-ground_size, -ground_size, ground_size, ground_size],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=cfg.color_ground,
            opacity=0.3,
            name="Ground",
            showlegend=True,
            hoverinfo="skip",
        ))
        
        # Add arena bounds (translucent box) if enabled
        if show_arena:
            # Vertical edges
            for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                fig.add_trace(go.Scatter3d(
                    x=[dx * ground_size, dx * ground_size],
                    y=[dy * ground_size, dy * ground_size],
                    z=[0, cfg.arena_height],
                    mode="lines",
                    line=dict(color="rgba(100, 100, 255, 0.3)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            
            # Top edges
            corners = [
                (-ground_size, -ground_size),
                (ground_size, -ground_size),
                (ground_size, ground_size),
                (-ground_size, ground_size),
                (-ground_size, -ground_size),
            ]
            fig.add_trace(go.Scatter3d(
                x=[c[0] for c in corners],
                y=[c[1] for c in corners],
                z=[cfg.arena_height] * len(corners),
                mode="lines",
                line=dict(color="rgba(100, 100, 255, 0.3)", width=2),
                showlegend=False,
                hoverinfo="skip",
            ))
        
        # Add trajectory line with reward coloring
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(
                color=rewards,
                colorscale=cfg.color_trajectory,
                width=cfg.trajectory_width,
                colorbar=dict(
                    title="Reward",
                    x=1.02,
                    len=0.5,
                    tickfont=dict(color=cfg.font_color),
                    title_font=dict(color=cfg.font_color),
                ),
            ),
            name="Trajectory",
            hovertemplate=(
                "Step: %{customdata[0]}<br>"
                "Time: %{customdata[1]:.2f}s<br>"
                "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>"
                "Reward: %{customdata[2]:.3f}<br>"
                "<extra></extra>"
            ),
            customdata=np.column_stack([steps, times, rewards]),
        ))
        
        # Add start marker
        fig.add_trace(go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode="markers+text",
            marker=dict(
                size=cfg.marker_size,
                color=cfg.color_start,
                symbol="diamond",
            ),
            text=["START"],
            textposition="top center",
            textfont=dict(color=cfg.color_start, size=12),
            name="Start",
            hovertemplate="Start<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        ))
        
        # Add end marker
        end_color = cfg.color_failure if df["terminated"].iloc[-1] else cfg.color_end
        end_symbol = "x" if df["terminated"].iloc[-1] else "circle"
        end_label = "CRASH" if df["terminated"].iloc[-1] else "END"
        
        fig.add_trace(go.Scatter3d(
            x=[x[-1]],
            y=[y[-1]],
            z=[z[-1]],
            mode="markers+text",
            marker=dict(
                size=cfg.marker_size,
                color=end_color,
                symbol=end_symbol,
            ),
            text=[end_label],
            textposition="top center",
            textfont=dict(color=end_color, size=12),
            name="End",
            hovertemplate=f"{end_label}<br>Position: (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>",
        ))
        
        # Add target marker if target position is available
        has_target = "target_x" in df.columns and "target_y" in df.columns and "target_z" in df.columns
        if has_target:
            target_x = df["target_x"].iloc[0]
            target_y = df["target_y"].iloc[0]
            target_z = df["target_z"].iloc[0]
            
            # Calculate final distance to target
            final_dist = np.sqrt((x[-1] - target_x)**2 + (y[-1] - target_y)**2 + (z[-1] - target_z)**2)
            
            # Target marker (star shape, bright color)
            fig.add_trace(go.Scatter3d(
                x=[target_x],
                y=[target_y],
                z=[target_z],
                mode="markers+text",
                marker=dict(
                    size=cfg.marker_size + 4,
                    color="#ffff00",  # Bright yellow
                    symbol="diamond",
                    line=dict(color="#ff8800", width=2),
                ),
                text=["TARGET"],
                textposition="top center",
                textfont=dict(color="#ffff00", size=12, family="Arial Black"),
                name="Target",
                hovertemplate=(
                    "TARGET<br>"
                    f"Position: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})<br>"
                    f"Final Distance: {final_dist:.3f}m<extra></extra>"
                ),
            ))
            
            # Add dashed line from end position to target (shows error)
            fig.add_trace(go.Scatter3d(
                x=[x[-1], target_x],
                y=[y[-1], target_y],
                z=[z[-1], target_z],
                mode="lines",
                line=dict(
                    color="#ffff00",
                    width=2,
                    dash="dash",
                ),
                name=f"Error ({final_dist:.2f}m)",
                hoverinfo="skip",
            ))
        
        # Add animated drone marker if animation enabled
        if show_animation and len(x) > 1:
            # Initial drone position
            fig.add_trace(go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                marker=dict(
                    size=cfg.drone_marker_size,
                    color="#00ffff",
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                name="Drone",
                hovertemplate="Drone<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            ))
            
            # Create animation frames
            frames = []
            for i in range(len(x)):
                frame = go.Frame(
                    data=[
                        go.Scatter3d(
                            x=[x[i]],
                            y=[y[i]],
                            z=[z[i]],
                        )
                    ],
                    traces=[len(fig.data) - 1],  # Update only the drone marker
                    name=str(i),
                )
                frames.append(frame)
            
            fig.frames = frames
            
            # Add animation controls
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        y=0.0,
                        x=0.1,
                        xanchor="right",
                        yanchor="top",
                        buttons=[
                            dict(
                                label="▶ Play",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=cfg.frame_duration, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=cfg.transition_duration),
                                    ),
                                ],
                            ),
                            dict(
                                label="⏸ Pause",
                                method="animate",
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0),
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        yanchor="top",
                        xanchor="left",
                        currentvalue=dict(
                            prefix="Step: ",
                            visible=True,
                            xanchor="right",
                            font=dict(color=cfg.font_color),
                        ),
                        transition=dict(duration=cfg.transition_duration),
                        pad=dict(b=10, t=50),
                        len=0.9,
                        x=0.1,
                        y=0.0,
                        steps=[
                            dict(
                                args=[
                                    [str(i)],
                                    dict(
                                        frame=dict(duration=cfg.transition_duration, redraw=True),
                                        mode="immediate",
                                        transition=dict(duration=cfg.transition_duration),
                                    ),
                                ],
                                label=str(i),
                                method="animate",
                            )
                            for i in range(0, len(x), max(1, len(x) // 50))  # Limit slider steps
                        ],
                    )
                ],
            )
        
        # Build annotations for metrics
        annotations = []
        if episode_summary:
            metrics_text = (
                f"<b>Episode Metrics</b><br>"
                f"Total Reward: {episode_summary.get('total_reward', 0):.2f}<br>"
                f"Episode Length: {episode_summary.get('episode_length', 0)}<br>"
                f"Max Alt. Dev: {episode_summary.get('max_altitude_deviation', 0):.3f}m<br>"
                f"Max Horiz. Dev: {episode_summary.get('max_horizontal_deviation', 0):.3f}m<br>"
                f"Outcome: {episode_summary.get('terminal_reason', 'unknown')}"
            )
            
            # Add target metrics if available
            if episode_summary.get('target_position') is not None:
                target_pos = episode_summary.get('target_position')
                final_dist = episode_summary.get('final_distance_to_target', 0)
                min_dist = episode_summary.get('min_distance_to_target', 0)
                metrics_text += (
                    f"<br><br><b>Target Metrics</b><br>"
                    f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})<br>"
                    f"Final Distance: {final_dist:.3f}m<br>"
                    f"Min Distance: {min_dist:.3f}m"
                )
            annotations.append(dict(
                text=metrics_text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                font=dict(size=11, color=cfg.font_color),
                bgcolor="rgba(30, 30, 60, 0.8)",
                bordercolor="rgba(100, 100, 200, 0.5)",
                borderwidth=1,
                borderpad=8,
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color=cfg.font_color),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(
                    title="X (m)",
                    range=[-cfg.arena_size, cfg.arena_size],
                    backgroundcolor=cfg.background_color,
                    gridcolor=cfg.grid_color,
                    showbackground=True,
                    title_font=dict(color=cfg.font_color),
                    tickfont=dict(color=cfg.font_color),
                ),
                yaxis=dict(
                    title="Y (m)",
                    range=[-cfg.arena_size, cfg.arena_size],
                    backgroundcolor=cfg.background_color,
                    gridcolor=cfg.grid_color,
                    showbackground=True,
                    title_font=dict(color=cfg.font_color),
                    tickfont=dict(color=cfg.font_color),
                ),
                zaxis=dict(
                    title="Z (m)",
                    range=[0, cfg.arena_height],
                    backgroundcolor=cfg.background_color,
                    gridcolor=cfg.grid_color,
                    showbackground=True,
                    title_font=dict(color=cfg.font_color),
                    tickfont=dict(color=cfg.font_color),
                ),
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            paper_bgcolor=cfg.paper_color,
            plot_bgcolor=cfg.background_color,
            font=dict(color=cfg.font_color),
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(30, 30, 60, 0.8)",
                bordercolor="rgba(100, 100, 200, 0.5)",
                font=dict(color=cfg.font_color),
            ),
            annotations=annotations,
            margin=dict(l=0, r=0, t=50, b=100 if show_animation else 0),
        )
        
        return fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    def create_multi_episode_view(
        self,
        dfs: List[pd.DataFrame],
        summaries: List[Dict[str, Any]],
        title: str = "Multi-Episode Trajectories",
    ) -> str:
        """
        Create a visualization showing multiple episodes.
        
        Args:
            dfs: List of DataFrames with rollout data
            summaries: List of episode summary dicts
            title: Plot title
            
        Returns:
            HTML string of the visualization
        """
        import plotly.graph_objects as go
        
        cfg = self.config
        
        # Color palette for different episodes
        colors = [
            "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", 
            "#ffeaa7", "#dfe6e9", "#fd79a8", "#a29bfe"
        ]
        
        fig = go.Figure()
        
        # Add ground plane
        ground_size = cfg.arena_size
        fig.add_trace(go.Mesh3d(
            x=[-ground_size, ground_size, ground_size, -ground_size],
            y=[-ground_size, -ground_size, ground_size, ground_size],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=cfg.color_ground,
            opacity=0.3,
            name="Ground",
            showlegend=True,
            hoverinfo="skip",
        ))
        
        # Add target marker (same for all hovering episodes)
        target_added = False
        for df in dfs:
            if "target_x" in df.columns and not target_added:
                target_x = df["target_x"].iloc[0]
                target_y = df["target_y"].iloc[0]
                target_z = df["target_z"].iloc[0]
                
                fig.add_trace(go.Scatter3d(
                    x=[target_x],
                    y=[target_y],
                    z=[target_z],
                    mode="markers+text",
                    marker=dict(
                        size=12,
                        color="#ffff00",
                        symbol="diamond",
                        line=dict(color="#ff8800", width=2),
                    ),
                    text=["TARGET"],
                    textposition="top center",
                    textfont=dict(color="#ffff00", size=14, family="Arial Black"),
                    name=f"Target ({target_x:.0f},{target_y:.0f},{target_z:.0f})",
                    hovertemplate=f"TARGET<br>({target_x:.1f}, {target_y:.1f}, {target_z:.1f})<extra></extra>",
                ))
                target_added = True
                break
        
        # Add each episode trajectory
        for i, (df, summary) in enumerate(zip(dfs, summaries)):
            color = colors[i % len(colors)]
            
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            
            ep_num = summary.get("episode", i)
            reward = summary.get("total_reward", 0)
            
            # Trajectory line
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=cfg.trajectory_width),
                name=f"Ep {ep_num} (R={reward:.1f})",
                hovertemplate=f"Episode {ep_num}<br>Position: (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>",
            ))
            
            # Start marker
            fig.add_trace(go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                marker=dict(size=6, color=color, symbol="diamond"),
                showlegend=False,
                hovertemplate=f"Ep {ep_num} Start<extra></extra>",
            ))
            
            # End marker
            end_symbol = "x" if df["terminated"].iloc[-1] else "circle"
            fig.add_trace(go.Scatter3d(
                x=[x[-1]],
                y=[y[-1]],
                z=[z[-1]],
                mode="markers",
                marker=dict(size=6, color=color, symbol=end_symbol),
                showlegend=False,
                hovertemplate=f"Ep {ep_num} End<extra></extra>",
            ))
        
        # Build summary table
        summary_rows = []
        for i, s in enumerate(summaries):
            row = f"Ep {s.get('episode', i)}: R={s.get('total_reward', 0):.1f}, L={s.get('episode_length', 0)}"
            
            # Add target distance if available
            final_dist = s.get('final_distance_to_target')
            if final_dist is not None:
                min_dist = s.get('min_distance_to_target', final_dist)
                row += f", d={final_dist:.2f}m (min={min_dist:.2f}m)"
            else:
                row += f", {s.get('terminal_reason', '?')}"
            
            summary_rows.append(row)
        summary_text = "<b>Episodes</b><br>" + "<br>".join(summary_rows)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color=cfg.font_color),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(
                    title="X (m)",
                    range=[-cfg.arena_size, cfg.arena_size],
                    backgroundcolor=cfg.background_color,
                    gridcolor=cfg.grid_color,
                    showbackground=True,
                    title_font=dict(color=cfg.font_color),
                    tickfont=dict(color=cfg.font_color),
                ),
                yaxis=dict(
                    title="Y (m)",
                    range=[-cfg.arena_size, cfg.arena_size],
                    backgroundcolor=cfg.background_color,
                    gridcolor=cfg.grid_color,
                    showbackground=True,
                    title_font=dict(color=cfg.font_color),
                    tickfont=dict(color=cfg.font_color),
                ),
                zaxis=dict(
                    title="Z (m)",
                    range=[0, cfg.arena_height],
                    backgroundcolor=cfg.background_color,
                    gridcolor=cfg.grid_color,
                    showbackground=True,
                    title_font=dict(color=cfg.font_color),
                    tickfont=dict(color=cfg.font_color),
                ),
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            paper_bgcolor=cfg.paper_color,
            plot_bgcolor=cfg.background_color,
            font=dict(color=cfg.font_color),
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(30, 30, 60, 0.8)",
                bordercolor="rgba(100, 100, 200, 0.5)",
                font=dict(color=cfg.font_color),
            ),
            annotations=[
                dict(
                    text=summary_text,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=10, color=cfg.font_color),
                    bgcolor="rgba(30, 30, 60, 0.8)",
                    bordercolor="rgba(100, 100, 200, 0.5)",
                    borderwidth=1,
                    borderpad=8,
                )
            ],
            margin=dict(l=0, r=0, t=50, b=0),
        )
        
        return fig.to_html(include_plotlyjs="cdn", full_html=True)


def create_trajectory_visualization(
    csv_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    episode_summary: Optional[Dict[str, Any]] = None,
    show_animation: bool = True,
    show_arena: bool = True,
) -> Path:
    """
    Create a 3D trajectory visualization from a rollout CSV.
    
    Args:
        csv_path: Path to rollout CSV file
        output_path: Output HTML path (default: same dir as CSV)
        title: Plot title
        episode_summary: Optional episode summary dict
        show_animation: Include animation controls
        show_arena: Show arena bounds
        
    Returns:
        Path to generated HTML file
    """
    csv_path = Path(csv_path)
    
    if output_path is None:
        output_path = csv_path.with_suffix(".html")
    else:
        output_path = Path(output_path)
    
    if title is None:
        title = f"Trajectory: {csv_path.stem}"
    
    visualizer = TrajectoryVisualizer()
    df = visualizer.load_rollout(str(csv_path))
    
    html = visualizer.create_visualization(
        df=df,
        title=title,
        show_animation=show_animation,
        show_arena=show_arena,
        episode_summary=episode_summary,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Visualization saved: {output_path}")
    return output_path


def create_multi_episode_visualization(
    session_dir: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Create a multi-episode visualization from a session directory.
    
    Args:
        session_dir: Path to session directory with rollout CSVs
        output_path: Output HTML path (default: index.html in session_dir)
        title: Plot title
        
    Returns:
        Path to generated HTML file
    """
    session_dir = Path(session_dir)
    
    if output_path is None:
        output_path = session_dir / "index.html"
    else:
        output_path = Path(output_path)
    
    # Load summary
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            session_summary = json.load(f)
        episode_summaries = session_summary.get("episodes", [])
        run_id = session_summary.get("run_id", "unknown")
    else:
        episode_summaries = []
        run_id = "unknown"
    
    if title is None:
        title = f"Model: {run_id}"
    
    # Load all rollout CSVs
    csv_files = sorted(session_dir.glob("rollout_ep*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No rollout CSVs found in {session_dir}")
    
    visualizer = TrajectoryVisualizer()
    dfs = [visualizer.load_rollout(str(f)) for f in csv_files]
    
    # Match summaries to CSVs
    summaries = []
    for i, csv_file in enumerate(csv_files):
        # Find matching summary
        matching = [s for s in episode_summaries if s.get("csv_file") == csv_file.name]
        if matching:
            summaries.append(matching[0])
        else:
            summaries.append({"episode": i, "csv_file": csv_file.name})
    
    # Create multi-episode view
    html = visualizer.create_multi_episode_view(
        dfs=dfs,
        summaries=summaries,
        title=title,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Multi-episode visualization saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create 3D trajectory visualization from rollout CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to rollout CSV or session directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title"
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable animation controls"
    )
    parser.add_argument(
        "--no-arena",
        action="store_true",
        help="Hide arena bounds"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Multi-episode visualization
        create_multi_episode_visualization(
            session_dir=str(input_path),
            output_path=args.output,
            title=args.title,
        )
    else:
        # Single trajectory visualization
        create_trajectory_visualization(
            csv_path=str(input_path),
            output_path=args.output,
            title=args.title,
            show_animation=not args.no_animation,
            show_arena=not args.no_arena,
        )

