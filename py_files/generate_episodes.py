import numpy as np
import argparse
import os
import json
from .simulation_core import OilSpillSimulatorCore # Relative import

def generate_episodes(num_episodes, steps_per_episode, grid_size_r, grid_size_c,
                      output_dir, sim_config_file_path, cell_size_meters):
    """
    Generates episode data using the OilSpillSimulatorCore and saves it to .npz files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the base simulation config to get its details for reproducibility
    try:
        with open(sim_config_file_path, 'r') as f:
            sim_config_details_json = f.read()
            # sim_config_data = json.loads(sim_config_details_json) # Not strictly needed here, but good for validation
    except Exception as e:
        print(f"Error loading simulation config file {sim_config_file_path}: {e}")
        return

    sim_config_filename_no_ext = os.path.splitext(os.path.basename(sim_config_file_path))[0]

    for episode_idx in range(num_episodes):
        print(f"Generating episode {episode_idx + 1}/{num_episodes}...")
        
        simulator = OilSpillSimulatorCore(sim_config_file_path)
        # Ensure sim core's time step is what we expect for env steps, or adjust conversion
        # For now, assume MARL env step = 1 sim core step
        env_time_step_hours = simulator.get_time_step_hours() 

        ground_truth_grids_for_episode = []
        current_vectors_for_episode_m_per_step = [] # Store as m/env_step

        for step_num in range(steps_per_episode):
            if simulator.is_finished():
                print(f"  Simulator finished early at step {step_num} (sim time {simulator.get_current_sim_time_hours()}h). Padding remaining steps.")
                # Pad with last known state if simulation finishes before steps_per_episode
                # This could happen if sim_config_file_path defines a shorter sim_time_hours
                # than steps_per_episode * env_time_step_hours
                last_grid = ground_truth_grids_for_episode[-1] if ground_truth_grids_for_episode else np.zeros((grid_size_r, grid_size_c), dtype=np.uint8)
                last_current = current_vectors_for_episode_m_per_step[-1] if current_vectors_for_episode_m_per_step else np.array([0.0, 0.0], dtype=np.float32)
                for _ in range(steps_per_episode - step_num):
                    ground_truth_grids_for_episode.append(last_grid.copy())
                    current_vectors_for_episode_m_per_step.append(last_current.copy())
                break # Exit the inner loop

            simulator.step()
            
            particle_pos_meters = simulator.get_active_particle_positions()
            
            # Get total effective current (ocean + wind) in m/hr
            # Convert to m/env_step (assuming 1 env_step = 1 sim_core_step)
            current_vec_m_per_hr = simulator.get_total_effective_current() 
            current_vec_m_per_step = current_vec_m_per_hr * env_time_step_hours 
            
            binary_grid = np.zeros((grid_size_r, grid_size_c), dtype=np.uint8)
            for x_m, y_m in particle_pos_meters:
                r = int(y_m / cell_size_meters) # y is typically rows
                c = int(x_m / cell_size_meters) # x is typically columns
                
                # Ensure within grid bounds
                if 0 <= r < grid_size_r and 0 <= c < grid_size_c:
                    binary_grid[r, c] = 1
            
            ground_truth_grids_for_episode.append(binary_grid)
            current_vectors_for_episode_m_per_step.append(current_vec_m_per_step.astype(np.float32))

        # Ensure we have `steps_per_episode` entries (padding if necessary)
        while len(ground_truth_grids_for_episode) < steps_per_episode:
            print(f"  Padding episode {episode_idx+1} as simulator duration was shorter than requested steps.")
            last_grid = ground_truth_grids_for_episode[-1] if ground_truth_grids_for_episode else np.zeros((grid_size_r, grid_size_c), dtype=np.uint8)
            last_current = current_vectors_for_episode_m_per_step[-1] if current_vectors_for_episode_m_per_step else np.array([0.0, 0.0], dtype=np.float32)
            ground_truth_grids_for_episode.append(last_grid.copy())
            current_vectors_for_episode_m_per_step.append(last_current.copy())


        # Save the episode data
        episode_filename = f"ep_SIM_{sim_config_filename_no_ext}_R{grid_size_r}_C{grid_size_c}_S{steps_per_episode}_ID{episode_idx:04d}.npz"
        episode_filepath = os.path.join(output_dir, episode_filename)
        
        generation_params = {
            'num_episodes_generated_in_run': num_episodes,
            'steps_per_episode': steps_per_episode,
            'grid_size_r': grid_size_r,
            'grid_size_c': grid_size_c,
            'cell_size_meters': cell_size_meters,
            'original_sim_config_file': sim_config_file_path,
            'env_time_step_hours': env_time_step_hours # time duration of one MARL env step
        }

        np.savez_compressed(
            episode_filepath,
            ground_truth_grids=np.array(ground_truth_grids_for_episode, dtype=np.uint8),
            current_vectors_m_per_step=np.array(current_vectors_for_episode_m_per_step, dtype=np.float32),
            simulation_config_details_json=sim_config_details_json,
            generation_params_json=json.dumps(generation_params) # Store as JSON string
        )
        print(f"  Saved: {episode_filepath}")

    print("Episode generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate oil spill episode data for MARL training.")
    parser.add_argument("--num_episodes", type=int, required=True, help="Number of distinct episode scenarios to generate.")
    parser.add_argument("--steps_per_episode", type=int, required=True, help="Number of environment time steps for each generated episode.")
    parser.add_argument("--grid_size_r", type=int, required=True, help="Number of rows for the discretized environment grid.")
    parser.add_argument("--grid_size_c", type=int, required=True, help="Number of columns for the discretized environment grid.")
    parser.add_argument("--output_dir", type=str, default="marl_framework/episode_data", help="Directory to save the generated .npz files.")
    parser.add_argument("--sim_config_file", type=str, required=True, help="Path to the JSON configuration file for the OilSpillSimulatorCore.")
    parser.add_argument("--cell_size_meters", type=float, required=True, help="The size of each grid cell in meters.")
    
    args = parser.parse_args()

    # Create a dummy sim_config.json if it doesn't exist for testing
    if not os.path.exists(args.sim_config_file) and args.sim_config_file == "dummy_sim_core_config.json":
        dummy_config = {
            "simulation_setup": {
                "simulation_time_hours": args.steps_per_episode * 0.1, # Make sure sim runs long enough for all steps
                "time_step_hours": 0.1, # This will be env_time_step_hours
                "domain_size_meters": [args.grid_size_c * args.cell_size_meters, args.grid_size_r * args.cell_size_meters],
            },
            "initial_spill": {"num_particles": 50, "location_xy_meters": [args.grid_size_c * args.cell_size_meters / 2, args.grid_size_r * args.cell_size_meters / 2], "radius_meters": args.cell_size_meters * 5},
            "particle_properties": {"mean_lifetime_hours": 100, "lifetime_stddev_hours": 10, "diffusion_strength_m_per_sqrt_hr": 2.0 * args.cell_size_meters},
            "initial_environment": {"current_velocity_xy_m_per_hr": [0.1 * args.cell_size_meters, 0.05 * args.cell_size_meters], "wind_speed_m_per_hr": 0.5 * args.cell_size_meters, "wind_direction_degrees_from_north": 90, "wind_effect_factor": 0.03},
            "environmental_changes": [{"time_hours": args.steps_per_episode * 0.1 / 2, "current_velocity_xy_m_per_hr": [-0.1 * args.cell_size_meters, 0.0]}]
        }
        with open(args.sim_config_file, 'w') as f:
            json.dump(dummy_config, f, indent=4)
        print(f"Created dummy simulation config for generation: {args.sim_config_file}")


    generate_episodes(
        args.num_episodes,
        args.steps_per_episode,
        args.grid_size_r,
        args.grid_size_c,
        args.output_dir,
        args.sim_config_file,
        args.cell_size_meters
    )

    # Example test load of a generated file
    if args.num_episodes > 0:
        sim_config_filename_no_ext = os.path.splitext(os.path.basename(args.sim_config_file))[0]
        test_file = os.path.join(args.output_dir, f"ep_SIM_{sim_config_filename_no_ext}_R{args.grid_size_r}_C{args.grid_size_c}_S{args.steps_per_episode}_ID0000.npz")
        if os.path.exists(test_file):
            print(f"\nTesting load of generated file: {test_file}")
            data = np.load(test_file, allow_pickle=True)
            print("Keys in loaded file:", data.files)
            print("Shape of ground_truth_grids:", data['ground_truth_grids'].shape)
            print("Shape of current_vectors_m_per_step:", data['current_vectors_m_per_step'].shape)
            print("First current vector:", data['current_vectors_m_per_step'][0])
            gen_params = json.loads(data['generation_params_json'].item()) # .item() if saved as 0-d array
            print("Generation params env_time_step_hours:", gen_params.get('env_time_step_hours'))
            print("Simulation config details (first 100 chars):", data['simulation_config_details_json'].item()[:100])
        else:
            print(f"Test file not found: {test_file}")