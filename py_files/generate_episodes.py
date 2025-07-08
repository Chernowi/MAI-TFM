import numpy as np
import argparse
import os
import json
from .simulation_core import OilSpillSimulatorCore # Relative import

def generate_episodes(num_episodes, steps_per_episode, grid_size_r, grid_size_c,
                      output_dir, sim_config_file_path):
    """
    Generates episode data using the OilSpillSimulatorCore and saves it to .npz files.
    Calculates cell size to fit the simulation domain onto the specified grid.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the base simulation config to get its details for reproducibility
    try:
        with open(sim_config_file_path, 'r') as f:
            sim_config_details_json = f.read()
            sim_config_data = json.loads(sim_config_details_json)
    except Exception as e:
        print(f"Error loading simulation config file {sim_config_file_path}: {e}")
        return

    # Calculate cell_size_meters to fit the domain to the grid resolution.
    domain_size_m = sim_config_data["simulation_setup"]["domain_size_meters"]
    domain_width_m, domain_height_m = domain_size_m[0], domain_size_m[1]

    # Use two separate cell sizes for mapping to handle non-square aspect ratios correctly.
    cell_size_meters_c = domain_width_m / grid_size_c
    cell_size_meters_r = domain_height_m / grid_size_r
    
    domain_aspect_ratio = domain_width_m / domain_height_m
    grid_aspect_ratio = grid_size_c / grid_size_r
    if not np.isclose(domain_aspect_ratio, grid_aspect_ratio, rtol=0.05):
        print(f"WARNING: Aspect ratio mismatch! Domain: {domain_aspect_ratio:.2f}, Grid: {grid_aspect_ratio:.2f}. Visualization may appear stretched.")

    # We save a single averaged value for the environment to use.
    cell_size_meters = (cell_size_meters_c + cell_size_meters_r) / 2.0
    print(f"Simulation Domain: {domain_width_m}m x {domain_height_m}m")
    print(f"Environment Grid: {grid_size_c} x {grid_size_r} cells")
    print(f"Calculated effective cell size for discretization: {cell_size_meters:.2f} meters/cell")
    
    sim_config_filename_no_ext = os.path.splitext(os.path.basename(sim_config_file_path))[0]

    for episode_idx in range(num_episodes):
        print(f"Generating episode {episode_idx + 1}/{num_episodes}...")
        
        simulator = OilSpillSimulatorCore(sim_config_file_path) # Simulator runs on its original config
        env_time_step_hours = simulator.get_time_step_hours() 

        ground_truth_grids_for_episode = []
        current_vectors_for_episode_m_per_step = []

        for step_num in range(steps_per_episode):
            if simulator.is_finished():
                last_grid = ground_truth_grids_for_episode[-1] if ground_truth_grids_for_episode else np.zeros((grid_size_r, grid_size_c), dtype=np.uint8)
                last_current = current_vectors_for_episode_m_per_step[-1] if current_vectors_for_episode_m_per_step else np.array([0.0, 0.0], dtype=np.float32)
                for _ in range(steps_per_episode - step_num):
                    ground_truth_grids_for_episode.append(last_grid.copy())
                    current_vectors_for_episode_m_per_step.append(last_current.copy())
                break 

            simulator.step()
            particle_pos_meters = simulator.get_active_particle_positions()
            current_vec_m_per_hr = simulator.get_total_effective_current() 
            current_vec_m_per_step = current_vec_m_per_hr * env_time_step_hours 
            
            binary_grid = np.zeros((grid_size_r, grid_size_c), dtype=np.uint8)
            for x_m, y_m in particle_pos_meters:
                r = int(y_m / cell_size_meters_r)
                c = int(x_m / cell_size_meters_c)
                
                if 0 <= r < grid_size_r and 0 <= c < grid_size_c:
                    binary_grid[r, c] = 1
            
            ground_truth_grids_for_episode.append(binary_grid)
            current_vectors_for_episode_m_per_step.append(current_vec_m_per_step.astype(np.float32))

        while len(ground_truth_grids_for_episode) < steps_per_episode:
            last_grid = ground_truth_grids_for_episode[-1] if ground_truth_grids_for_episode else np.zeros((grid_size_r, grid_size_c), dtype=np.uint8)
            last_current = current_vectors_for_episode_m_per_step[-1] if current_vectors_for_episode_m_per_step else np.array([0.0, 0.0], dtype=np.float32)
            ground_truth_grids_for_episode.append(last_grid.copy())
            current_vectors_for_episode_m_per_step.append(last_current.copy())

        episode_filename = f"ep_SIM_{sim_config_filename_no_ext}_R{grid_size_r}_C{grid_size_c}_S{steps_per_episode}_ID{episode_idx:04d}.npz"
        episode_filepath = os.path.join(output_dir, episode_filename)
        
        generation_params = {
            'num_episodes_generated_in_run': num_episodes,
            'steps_per_episode': steps_per_episode,
            'grid_size_r': grid_size_r,
            'grid_size_c': grid_size_c,
            'cell_size_meters': cell_size_meters, # Save the calculated value
            'original_sim_config_file': sim_config_file_path,
            'env_time_step_hours': env_time_step_hours
        }

        np.savez_compressed(
            episode_filepath,
            ground_truth_grids=np.array(ground_truth_grids_for_episode, dtype=np.uint8),
            current_vectors_m_per_step=np.array(current_vectors_for_episode_m_per_step, dtype=np.float32),
            simulation_config_details_json=sim_config_details_json,
            generation_params_json=json.dumps(generation_params)
        )
        print(f"  Saved: {episode_filepath}")

    print("Episode generation complete.")


if __name__ == "__main__":
    # --- MODIFICATION START ---
    # Set up argument parser with defaults that align with the main experiment config.
    parser = argparse.ArgumentParser(description="Generate oil spill episode data for MARL training.")
    
    parser.add_argument("--num_episodes", type=int, default=1, 
                        help="Number of distinct episode scenarios to generate. Default: 200")
    
    parser.add_argument("--steps_per_episode", type=int, default=500, 
                        help="Number of environment time steps for each generated episode. Default: 500")
    
    parser.add_argument("--grid_size_r", type=int, default=64, 
                        help="Number of rows for the discretized environment grid. Default: 64")
    
    parser.add_argument("--grid_size_c", type=int, default=64, 
                        help="Number of columns for the discretized environment grid. Default: 64")
    
    parser.add_argument("--output_dir", type=str, default="marl_framework/episode_data", 
                        help="Directory to save the generated .npz files.")
    
    parser.add_argument("--sim_config_file", type=str, default="marl_framework/configs/all_features.json", 
                        help="Path to the JSON configuration file for the OilSpillSimulatorCore.")
    # --- MODIFICATION END ---
    
    args = parser.parse_args()

    # Create a dummy sim_config.json if a user specifies a dummy path for testing
    if not os.path.exists(args.sim_config_file) and "dummy" in args.sim_config_file:
        # For a dummy config, we create a domain that matches the grid aspect ratio to avoid warnings.
        # Here we arbitrarily pick 100.0 m/cell for the dummy simulation.
        dummy_cell_size = 100.0
        dummy_domain_size_m = [args.grid_size_c * dummy_cell_size, args.grid_size_r * dummy_cell_size]
        dummy_config = {
            "simulation_setup": {
                "simulation_time_hours": args.steps_per_episode * 0.1, 
                "time_step_hours": 0.1, 
                "domain_size_meters": dummy_domain_size_m,
            },
            "initial_spill": {
                "num_particles": 50, 
                "location_xy_meters": [dummy_domain_size_m[0] / 2, dummy_domain_size_m[1] / 2], 
                "radius_meters": dummy_domain_size_m[0] * 0.1
            },
            "particle_properties": {"mean_lifetime_hours": 100, "lifetime_stddev_hours": 10, "diffusion_strength_m_per_sqrt_hr": 100.0},
            "initial_environment": {"current_velocity_xy_m_per_hr": [5.0, 2.5], "wind_speed_m_per_hr": 25.0, "wind_direction_degrees_from_north": 90, "wind_effect_factor": 0.03},
        }
        with open(args.sim_config_file, 'w') as f:
            json.dump(dummy_config, f, indent=4)
        print(f"Created dummy simulation config for generation: {args.sim_config_file}")

    # Call the main generation function with the parsed arguments
    generate_episodes(
        args.num_episodes,
        args.steps_per_episode,
        args.grid_size_r,
        args.grid_size_c,
        args.output_dir,
        args.sim_config_file,
    )

    # Example test load of a generated file
    if args.num_episodes > 0:
        sim_config_filename_no_ext = os.path.splitext(os.path.basename(args.sim_config_file))[0]
        test_file = os.path.join(args.output_dir, f"ep_SIM_{sim_config_filename_no_ext}_R{args.grid_size_r}_C{args.grid_size_c}_S{args.steps_per_episode}_ID0000.npz")
        if os.path.exists(test_file):
            print(f"\n--- Testing load of generated file: {test_file} ---")
            data = np.load(test_file, allow_pickle=True)
            print("Keys in loaded file:", data.files)
            print("Shape of ground_truth_grids:", data['ground_truth_grids'].shape)
            print("Shape of current_vectors_m_per_step:", data['current_vectors_m_per_step'].shape)
            
            gen_params = json.loads(data['generation_params_json'].item())
            print(f"Loaded Generation Params:")
            print(f"  - Cell Size (meters): {gen_params.get('cell_size_meters'):.2f}")
            print(f"  - Grid Size: {gen_params.get('grid_size_c')}x{gen_params.get('grid_size_r')}")
            
            sim_details = json.loads(data['simulation_config_details_json'].item())
            domain = sim_details.get("simulation_setup", {}).get("domain_size_meters", "[Not Found]")
            print(f"  - Original Sim Domain: {domain}")
            print("--- Load test successful ---")
        else:
            print(f"Test file not found: {test_file}")