# marl_framework/utils/visualization_utils.py

import os
import numpy as np
import io

VISUALIZATION_ENABLED = False
plt = None
imageio = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio
    VISUALIZATION_ENABLED = True
    print("Matplotlib and imageio imported successfully for visualization.")
except ImportError:
    print("Warning: Matplotlib or imageio not found. Visualization will be disabled.")
    print("Install them with: pip install matplotlib imageio")


class EpisodeVisualizer:
    """
    Handles the creation of visualization frames and saving them as GIFs for episode rollouts.
    This version is compatible with a codebase that uses agent headings, but does NOT visualize them.
    """
    def __init__(self, grid_size_r, grid_size_c, num_agents, num_headings, cell_size_m, enabled=True):
        """
        Initializes the EpisodeVisualizer.

        Args:
            grid_size_r (int): Number of rows in the grid.
            grid_size_c (int): Number of columns in the grid.
            num_agents (int): Number of agents in the environment.
            num_headings (int): Number of discrete headings for agents (accepted for compatibility, but not used).
            cell_size_m (float): The size of a grid cell in meters.
            enabled (bool): Whether the visualizer is active.
        """
        self.grid_r = grid_size_r
        self.grid_c = grid_size_c
        self.num_agents = num_agents
        # num_headings is accepted for compatibility but not used in plotting
        self.cell_size_m = cell_size_m
        self.enabled = enabled and VISUALIZATION_ENABLED

        self.frames = []
        self.current_episode_number = -1
        self.output_gif_path = ""

        if self.enabled:
            self.agent_colors = plt.cm.get_cmap('gist_rainbow', num_agents) if num_agents > 0 else ['blue']
            self.agent_belief_colors = [self.agent_colors(i) for i in range(num_agents)]

    def start_episode_recording(self, episode_number, output_gif_path):
        """
        Prepares the visualizer for a new episode recording.

        Args:
            episode_number (str or int): Identifier for the current episode.
            output_gif_path (str): The file path where the resulting GIF should be saved.
        """
        if not self.enabled: return
        self.frames = []
        self.current_episode_number = episode_number
        self.output_gif_path = output_gif_path
        print(f"Starting GIF recording for episode {self.current_episode_number}...")

    def add_frame(self, ground_truth_grid, agent_belief_maps_dict, shared_consensus_map, 
                  agent_positions_rc_dict, agent_headings_dict, 
                  current_vector_m_per_step, timestep_info_string):
        """
        Creates and adds a single visualization frame to the current recording.

        Args:
            ground_truth_grid (np.ndarray): The ground truth grid of the oil spill.
            agent_belief_maps_dict (dict): Dict of individual agent belief maps.
            shared_consensus_map (np.ndarray): The team's shared consensus map.
            agent_positions_rc_dict (dict): Dict of agent positions (row, col).
            agent_headings_dict (dict): Dict of agent headings (accepted for compatibility, but not used).
            current_vector_m_per_step (np.ndarray): The environmental current vector [dx, dy] in meters.
            timestep_info_string (str): A string with info about the current step to display in the title.
        """
        if not self.enabled:
            return

        fig, (ax, ax_shared) = plt.subplots(1, 2, figsize=(13, 10 * self.grid_r / self.grid_c), 
                                            gridspec_kw={'width_ratios': [10, 3]})
        fig.suptitle(f"Oil Spill Response - {timestep_info_string}", fontsize=16)

        plot_extent = [-0.5, self.grid_c - 0.5, self.grid_r - 0.5, -0.5]

        gt_display = np.zeros_like(ground_truth_grid, dtype=float)
        gt_display[ground_truth_grid == 1] = 0.8
        gt_display[ground_truth_grid == 0] = 1.0
        ax.imshow(gt_display, cmap='gray_r', vmin=0, vmax=1, extent=plot_extent, origin='upper', alpha=0.5, zorder=0)

        for i, agent_id in enumerate(agent_positions_rc_dict.keys()):
            belief_map_data = agent_belief_maps_dict.get(agent_id)
            if belief_map_data is None: continue
            belief_map = belief_map_data['belief']
            agent_belief_display = np.zeros((self.grid_r, self.grid_c, 4))
            base_color = self.agent_belief_colors[i]
            oil_mask = (belief_map == 1)
            agent_belief_display[oil_mask] = [*base_color[:3], 0.5]
            clean_mask = (belief_map == 0)
            agent_belief_display[clean_mask] = [*base_color[:3], 0.15]
            ax.imshow(agent_belief_display, extent=plot_extent, origin='upper', zorder=1)

        # Agent Positions
        for i, agent_id in enumerate(agent_positions_rc_dict.keys()):
            r, c = agent_positions_rc_dict[agent_id]
            plot_c, plot_r = c, r 
            
            # Agent marker (circle) with correct legend for all agents
            ax.scatter(plot_c, plot_r, s=100, color=self.agent_colors(i) if self.num_agents > 0 else 'blue', 
                       edgecolors='black', zorder=3, marker='o', label=f"Agent {i}")
            
            # The heading arrow plotting logic has been removed from this section.
            # The `agent_headings_dict` argument is ignored.

        # Corrected Environmental Current Vector
        curr_dc_per_step = current_vector_m_per_step[0] / self.cell_size_m
        curr_dr_per_step = current_vector_m_per_step[1] / self.cell_size_m 
        
        plot_dx = curr_dc_per_step
        plot_dy = curr_dr_per_step # Invert vertical component for plotting
        
        current_arrow_origin_c, current_arrow_origin_r = self.grid_c * 0.1, self.grid_r * 0.1
        ax.arrow(current_arrow_origin_c, current_arrow_origin_r, 
                 plot_dx * 5,  # Scale for visibility
                 plot_dy * 5,  # Use the correctly-signed plot_dy and scale
                 head_width=0.5, head_length=0.5, fc='purple', ec='purple', zorder=2, alpha=0.7, label="Current")

        ax.set_xlim(-0.5, self.grid_c - 0.5)
        ax.set_ylim(self.grid_r - 0.5, -0.5)
        ax.set_xticks(np.arange(0, self.grid_c, max(1, self.grid_c//10)))
        ax.set_yticks(np.arange(0, self.grid_r, max(1, self.grid_r//10)))
        ax.set_xlabel("Grid Column (X)")
        ax.set_ylabel("Grid Row (Y)")
        ax.set_title("Main View (Agent Beliefs Overlay)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(-0.5, self.grid_c, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_r, 1), minor=True)
        ax.grid(which='minor', color='k', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))

        consensus_display = np.ones((self.grid_r, self.grid_c, 4))
        consensus_display[shared_consensus_map == 1] = [0, 0, 1, 0.7]
        consensus_display[shared_consensus_map == 0] = [1, 0, 0, 0.3]
        consensus_display[shared_consensus_map == -1, 3] = 0
        ax_shared.imshow(consensus_display, extent=plot_extent, origin='upper')
        ax_shared.set_title("Shared Belief Map")
        ax_shared.set_xticks([])
        ax_shared.set_yticks([])
        ax_shared.set_aspect('equal', adjustable='box')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        self.frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    def save_recording(self, duration_per_frame_ms=200):
        """
        Saves all collected frames to a GIF file.

        Args:
            duration_per_frame_ms (int): The duration each frame should be displayed for, in milliseconds.
        """
        if not self.enabled or not self.frames or imageio is None:
            if self.frames: print("Visualizer: imageio not available, cannot save GIF.")
            return

        filename = self.output_gif_path
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            imageio.mimsave(filename, self.frames, duration=duration_per_frame_ms / 1000.0)
            print(f"Visualizer: Saved GIF: {filename}")
        except Exception as e:
            print(f"Visualizer: Error saving GIF {filename}: {e}")
        self.frames = []

    def close(self):
        """
        Cleans up resources used by the visualizer.
        """
        if self.enabled and plt is not None:
            plt.close('all')
        self.frames = []