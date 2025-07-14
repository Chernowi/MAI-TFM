import os
import numpy as np
import io # For in-memory image saving

# Conditional imports for matplotlib and imageio
VISUALIZATION_ENABLED = False
plt = None
imageio = None

try:
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend for saving to file/buffer
    import matplotlib.pyplot as plt
    import imageio
    VISUALIZATION_ENABLED = True
    print("Matplotlib and imageio imported successfully for visualization.")
except ImportError:
    print("Warning: Matplotlib or imageio not found. Visualization will be disabled.")
    print("Install them with: pip install matplotlib imageio")


class EpisodeVisualizer:
    def __init__(self, grid_size_r, grid_size_c, num_agents, num_headings, cell_size_m, enabled=True):
        self.grid_r = grid_size_r
        self.grid_c = grid_size_c
        self.num_agents = num_agents
        self.num_headings = num_headings # Added for correct arrow plotting
        self.cell_size_m = cell_size_m
        self.enabled = enabled and VISUALIZATION_ENABLED # Global and instance enable flags

        self.frames = []
        self.current_episode_number = -1
        self.output_gif_path_template = ""

        if self.enabled:
            # Agent colors
            self.agent_colors = plt.cm.get_cmap('gist_rainbow', num_agents) \
                                if num_agents > 0 else ['blue']
            
            # Colors for individual agent beliefs
            # Generate a list of RGBA colors from the agent colormap
            self.agent_belief_colors = [self.agent_colors(i) for i in range(num_agents)]
            
            # Heading symbols/arrows
            # Grid: +r is Down (South), +c is Right (East)
            # Headings: 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW (for 8 headings)
            # Arrow dx, dy in plot coordinates (matplotlib default: +y is Up, +x is Right)
            self.heading_arrows_plot_coords_8 = { 
                0: (0, 1),   # N (Up)
                1: (1, 1),   # NE (Up-Right)
                2: (1, 0),   # E  (Right)
                3: (1, -1),  # SE (Down-Right)
                4: (0, -1),  # S  (Down)
                5: (-1, -1), # SW (Down-Left)
                6: (-1, 0),  # W  (Left)
                7: (-1, 1)   # NW (Up-Left)
            }
            self.heading_arrows_plot_coords_4 = {
                0: (0, 1),   # N
                1: (1, 0),   # E
                2: (0, -1),  # S
                3: (-1, 0)   # W
            }


    def start_episode_recording(self, episode_number, output_gif_path_template="episode_{ep_num}.gif"):
        if not self.enabled: return
        self.frames = []
        self.current_episode_number = episode_number
        self.output_gif_path_template = output_gif_path_template
        print(f"Starting GIF recording for episode {self.current_episode_number}...")

    def add_frame(self, ground_truth_grid, agent_belief_maps_dict, shared_consensus_map, 
                  agent_positions_rc_dict, agent_headings_dict, 
                  current_vector_m_per_step, timestep_info_string):
        """Adds a single frame to the current recording."""
        if not self.enabled:
            return

        # Create a figure with two subplots: one for the main map, one for the shared map
        fig, (ax, ax_shared) = plt.subplots(1, 2, figsize=(13, 10 * self.grid_r / self.grid_c), 
                                            gridspec_kw={'width_ratios': [10, 3]})
        
        fig.suptitle(f"Oil Spill Response - {timestep_info_string}", fontsize=16)

        # --- Plot Ground Truth Oil Spill (as a heatmap) on the main axis ---
        # Plot extents (in grid cells)
        plot_extent = [-0.5, self.grid_c - 0.5, self.grid_r - 0.5, -0.5] # left, right, bottom, top for imshow with origin 'upper'

        # 1. Ground Truth Oil (e.g., light gray for oil, white for clean)
        gt_display = np.zeros_like(ground_truth_grid, dtype=float)
        gt_display[ground_truth_grid == 1] = 0.8 # Light gray for true oil
        gt_display[ground_truth_grid == 0] = 1.0 # White for true clean
        ax.imshow(gt_display, cmap='gray_r', vmin=0, vmax=1, extent=plot_extent, origin='upper', alpha=0.5, zorder=0)

        # 2. Individual Agent Belief Maps
        for i, agent_id in enumerate(agent_positions_rc_dict.keys()):
            # FIX: Access the 'belief' key from the agent's belief dictionary
            belief_map_data = agent_belief_maps_dict.get(agent_id)
            if belief_map_data is None: continue
            belief_map = belief_map_data['belief'] # Extract the numpy array

            agent_belief_display = np.zeros((self.grid_r, self.grid_c, 4)) # RGBA
            
            # Get base color for the agent
            base_color = self.agent_belief_colors[i]

            # Set color for cells believed to be oil
            oil_mask = (belief_map == 1)
            agent_belief_display[oil_mask] = [*base_color[:3], 0.5] # Color with 50% alpha

            # Set color for cells believed to be clean
            clean_mask = (belief_map == 0)
            agent_belief_display[clean_mask] = [*base_color[:3], 0.15] # Same color, but more transparent

            ax.imshow(agent_belief_display, extent=plot_extent, origin='upper', zorder=1)


        # 3. Agent Positions and Headings
        arrow_scale = 0.4 # Scale for heading arrows relative to cell size
        for i, agent_id in enumerate(agent_positions_rc_dict.keys()):
            r, c = agent_positions_rc_dict[agent_id]
            h = agent_headings_dict[agent_id]
            
            # Agent marker (circle)
            # For imshow with origin 'upper', y-coordinates are flipped relative to array indexing
            # (r,c) grid cell needs to be mapped to plot coordinates.
            # Cell (0,0) center is at plot (0,0) if extent is [-0.5, C-0.5, R-0.5, -0.5] for imshow
            # So, agent at grid (r,c) has plot center (c, r)
            plot_c, plot_r = c, r 
            ax.scatter(plot_c, plot_r, s=100, color=self.agent_colors(i) if self.num_agents > 0 else 'blue', 
                       edgecolors='black', zorder=3, marker='o', label=f"Agent {i}" if i==0 else "_nolegend_")
            
            # Heading arrow
            if self.num_agents > 0: # Ensure there are agents
                # FIX: Check self.num_headings, not the length of the dictionary
                if self.num_headings == 8 :
                    arrow_dx, arrow_dy = self.heading_arrows_plot_coords_8.get(h, (0,0))
                elif self.num_headings == 4:
                    arrow_dx, arrow_dy = self.heading_arrows_plot_coords_4.get(h, (0,0))
                else:
                    arrow_dx, arrow_dy = (0,0) # No arrows for other heading counts
                
                # Plot arrows with y-axis inverted (standard matplotlib) from grid's +r = South
                # Our heading_arrows_plot_coords are already for standard plot coords (+y is Up)
                ax.arrow(plot_c, plot_r, arrow_dx * arrow_scale, arrow_dy * arrow_scale,
                         head_width=0.2, head_length=0.2, fc=self.agent_colors(i), ec='black', zorder=4)

        # 4. Environmental Current Vector (Global Arrow)
        # current_vector_m_per_step is [dx_m_per_step, dy_m_per_step]
        # Convert to grid steps per env_step
        curr_dc_per_step = current_vector_m_per_step[0] / self.cell_size_m
        curr_dr_per_step = current_vector_m_per_step[1] / self.cell_size_m 
        
        # Current arrow origin (e.g., top-left corner or center of grid)
        # Plot arrow: dx is curr_dc, dy is -curr_dr (because plot +y is up, grid +r is down)
        current_arrow_origin_c, current_arrow_origin_r = self.grid_c * 0.1, self.grid_r * 0.1
        ax.arrow(current_arrow_origin_c, current_arrow_origin_r, 
                 curr_dc_per_step * 5,  # Scale for visibility
                 -curr_dr_per_step * 5, # Scale and invert dy for plot
                 head_width=0.5, head_length=0.5, fc='purple', ec='purple', zorder=2, alpha=0.7, label="Current")

        # Setup main plot
        ax.set_xlim(-0.5, self.grid_c - 0.5)
        ax.set_ylim(self.grid_r - 0.5, -0.5) # Flipped y-axis for 'upper' origin
        ax.set_xticks(np.arange(0, self.grid_c, max(1, self.grid_c//10)))
        ax.set_yticks(np.arange(0, self.grid_r, max(1, self.grid_r//10)))
        ax.set_xlabel("Grid Column (X)")
        ax.set_ylabel("Grid Row (Y)")
        ax.set_title("Main View (Agent Beliefs Overlay)")
        ax.set_aspect('equal', adjustable='box')
        # Set grid to align with cells
        ax.set_xticks(np.arange(-0.5, self.grid_c, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_r, 1), minor=True)
        ax.grid(which='minor', color='k', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))

        # --- Plot Shared Consensus Map on the side axis ---
        consensus_display = np.ones((self.grid_r, self.grid_c, 4)) # RGBA
        consensus_display[shared_consensus_map == 1] = [0, 0, 1, 0.7]  # Blue for predicted oil
        consensus_display[shared_consensus_map == 0] = [1, 0, 0, 0.3]  # Red for predicted clean
        consensus_display[shared_consensus_map == -1, 3] = 0 # Transparent for unknown
        ax_shared.imshow(consensus_display, extent=plot_extent, origin='upper')
        ax_shared.set_title("Shared Belief Map")
        ax_shared.set_xticks([])
        ax_shared.set_yticks([])
        ax_shared.set_aspect('equal', adjustable='box')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

        # Save to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100) # Adjust DPI as needed
        buf.seek(0)
        self.frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig) # Close the figure to free memory

    def save_recording(self, duration_per_frame_ms=200):
        if not self.enabled or not self.frames or imageio is None:
            if self.frames: print("Visualizer: imageio not available, cannot save GIF.")
            return

        filename = self.output_gif_path_template.format(ep_num=self.current_episode_number)
        # Ensure output directory exists if template includes path
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            imageio.mimsave(filename, self.frames, duration=duration_per_frame_ms / 1000.0) # duration in seconds
            print(f"Visualizer: Saved GIF: {filename}")
        except Exception as e:
            print(f"Visualizer: Error saving GIF {filename}: {e}")
        self.frames = [] # Clear frames after saving

    def close(self):
        if self.enabled and plt is not None:
            plt.close('all') # Close any stray figures
        self.frames = []


if __name__ == '__main__':
    if not VISUALIZATION_ENABLED:
        print("Skipping visualization test as matplotlib/imageio are not installed.")
    else:
        print("Running visualization test...")
        g_r, g_c = 10, 12
        n_a = 2
        n_h = 8
        cell_m = 10

        vis = EpisodeVisualizer(g_r, g_c, n_a, n_h, cell_m)
        vis.start_episode_recording(episode_number=1, output_gif_path_template="test_vis_ep{ep_num}.gif")

        for t_step in range(5):
            gt_grid = np.random.randint(0, 2, (g_r, g_c))
            
            # Generate individual belief maps for each agent
            # The visualizer expects the full dict with 'belief' key
            agent_beliefs = {
                f"agent_{i}": {
                    'belief': np.random.randint(-1, 2, (g_r, g_c)),
                    'timestamp': np.random.randint(0, 10, (g_r, g_c))
                } for i in range(n_a)
            }
            
            # The shared map would be derived from these in a real scenario
            consensus = np.random.randint(-1, 2, (g_r, g_c))
            
            agent_pos = {f"agent_{i}": (np.random.randint(0,g_r), np.random.randint(0,g_c)) for i in range(n_a)}
            agent_h = {f"agent_{i}": np.random.randint(0,n_h) for i in range(n_a)} # Use n_h
            
            current_vec = np.array([0.5, -0.2]) * cell_m # m/step

            info_str = f"Timestep {t_step}, IoU: {np.random.rand():.2f}"
            vis.add_frame(gt_grid, agent_beliefs, consensus, agent_pos, agent_h, current_vec, info_str)
        
        vis.save_recording(duration_per_frame_ms=500)
        vis.close()
        print("Visualization test finished. Check for 'test_vis_ep1.gif'.")
        # Clean up test GIF
        if os.path.exists("test_vis_ep1.gif"):
            os.remove("test_vis_ep1.gif")