# evaluate_and_visualize.py

import torch
import yaml
import argparse
import os
import numpy as np

from marl_framework.environments import OilSpillEnv
from marl_framework.agents import TransfQMixAgentNN
from marl_framework.utils import load_checkpoint, EpisodeVisualizer, VISUALIZATION_ENABLED

def evaluate_and_visualize(config, checkpoint_path, num_episodes_to_run, output_dir):
    """
    Loads a trained agent model and runs it in the environment to generate visualization GIFs.

    Args:
        config (dict): Configuration dictionary from the YAML file.
        checkpoint_path (str): Path to the saved model checkpoint (.pth.tar file).
        num_episodes_to_run (int): The number of evaluation episodes to run and visualize.
        output_dir (str): The directory where the output GIFs will be saved.
    """
    if not VISUALIZATION_ENABLED:
        print("Visualization is disabled because required packages (matplotlib, imageio) are not installed.")
        return

    # --- 1. Setup Device ---
    device = torch.device("cuda" if config['use_cuda'] and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Initialize Environment ---
    eval_env = OilSpillEnv(config['environment'],
                           config['environment']['episode_data_directory'],
                           specific_episode_file=config['environment'].get('specific_episode_file'))
    eval_env.reset()

    # Get environment specifications
    obs_spec = eval_env.get_observation_spec()
    action_space_size = eval_env.get_action_space_size()
    num_agents = eval_env.get_num_agents()
    agent_ids = eval_env.get_agent_ids()

    # --- 3. Initialize Agent Policy Network ---
    agent_nn_global_config_ext = config['environment'].copy()
    agent_nn_global_config_ext["ACTION_SPACE_SIZE"] = action_space_size
    agent_policy_nn = TransfQMixAgentNN(obs_spec, config['agent_nn'], agent_nn_global_config_ext).to(device)

    # --- 4. Load Checkpoint ---
    checkpoint = load_checkpoint(checkpoint_path, agent_nn=agent_policy_nn, mixer_nn=None, device=device)
    if checkpoint is None:
        print(f"Failed to load checkpoint from {checkpoint_path}. Exiting.")
        return
    agent_policy_nn.eval()
    print(f"Successfully loaded model weights from training episode {checkpoint.get('episode', 'N/A')}.")

    # --- 5. Initialize Visualizer ---
    os.makedirs(output_dir, exist_ok=True)
    gif_output_path_template = os.path.join(output_dir, "evaluation_run_ep{eval_ep_num}.gif")

    visualizer = EpisodeVisualizer(
        grid_size_r=config['environment']['GRID_SIZE_R'],
        grid_size_c=config['environment']['GRID_SIZE_C'],
        num_agents=num_agents,
        num_headings=config['environment']['NUM_HEADINGS'],
        cell_size_m=eval_env.cell_size_meters,
        enabled=True  # Force enable for this script
    )

    print(f"\nStarting visualization run for {num_episodes_to_run} episode(s)...")

    # --- 6. Evaluation & Visualization Loop ---
    for eval_ep_num in range(num_episodes_to_run):
        print(f"--- Running evaluation episode {eval_ep_num + 1}/{num_episodes_to_run} ---")

        # Reset environment and hidden states
        obs_eval, _ = eval_env.reset()
        eval_agent_h_states = {aid: torch.zeros(1, config['agent_nn']['AGENT_TRANSFORMER_EMBED_DIM']).to(device) for aid in agent_ids}

        # Start GIF recording
        current_gif_path = gif_output_path_template.format(eval_ep_num=eval_ep_num)
        visualizer.start_episode_recording(f"eval_{eval_ep_num}", current_gif_path)

        # Add initial frame (Step 0)
        initial_iou = eval_env.iou_oil_previous_step
        vis_info_str = f"Eval Ep {eval_ep_num+1}, Step 0 (Initial), IoU: {initial_iou:.3f}"
        visualizer.add_frame(eval_env._get_ground_truth_grid(),
                             eval_env.agent_belief_maps,
                             eval_env.shared_consensus_map,
                             eval_env.agent_positions_rc,
                             eval_env.agent_headings,
                             eval_env._get_current_vector_m_per_step(),
                             timestep_info_string=vis_info_str)

        # Run the episode
        for step in range(config['environment']['MAX_STEPS_PER_EPISODE']):
            actions_eval = {}
            current_h_eval = {}
            eval_belief_maps_np_dict = {aid: eval_env.agent_belief_maps[aid]['belief'] for aid in agent_ids}

            with torch.no_grad():
                for agent_id in agent_ids:
                    # Prepare inputs for the agent network
                    belief_map_t = torch.from_numpy(eval_belief_maps_np_dict[agent_id]).float().unsqueeze(0).to(device)

                    obs_entities_list = [obs_eval[agent_id]]
                    max_entities = len(obs_entities_list[0])
                    obs_tensor_eval = torch.zeros(1, max_entities, obs_spec['agent_observation']['entity_feature_dim'], device=device)
                    if max_entities > 0:
                        obs_tensor_eval[0, :max_entities, :] = torch.from_numpy(np.stack(obs_entities_list[0])).float()

                    pad_mask_eval = torch.ones(1, max_entities + 1, dtype=torch.bool, device=device)
                    pad_mask_eval[0, :(max_entities + 1)] = False

                    h_in_t = eval_agent_h_states[agent_id]

                    # Get deterministic action from the policy network
                    q_vals_eval, h_out_eval, _ = agent_policy_nn(belief_map_t, obs_tensor_eval, h_in_t, pad_mask_eval)
                    actions_eval[agent_id] = q_vals_eval.argmax(dim=1).item()
                    current_h_eval[agent_id] = h_out_eval

            # Step the environment
            next_obs_eval, _, _, dones_eval, infos_eval = eval_env.step(actions_eval)

            # Add frame to GIF
            vis_info_str = f"Eval Ep {eval_ep_num+1}, Step {step+1}, IoU: {infos_eval[agent_ids[0]]['iou']:.3f}"
            visualizer.add_frame(eval_env._get_ground_truth_grid(),
                                 eval_env.agent_belief_maps,
                                 eval_env.shared_consensus_map,
                                 eval_env.agent_positions_rc,
                                 eval_env.agent_headings,
                                 eval_env._get_current_vector_m_per_step(),
                                 timestep_info_string=vis_info_str)

            # Update state for next step
            obs_eval = next_obs_eval
            eval_agent_h_states = current_h_eval

            if dones_eval["__all__"]:
                break
        
        # Save the GIF for the completed episode
        if visualizer.frames:
            visualizer.save_recording(duration_per_frame_ms=config['logging']['visualization_duration_per_frame_ms'])

    visualizer.close()
    print("--- Visualization run finished. ---")

if __name__ == "__main__":
    """
    Main entry point for the evaluation and visualization script.

    Parses command-line arguments to specify the configuration file,
    model checkpoint, number of episodes to run, and output directory for GIFs.
    """
    parser = argparse.ArgumentParser(description="Generate GIFs from a trained TransfQMix model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the experiment configuration YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pth.tar file).")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes to visualize.")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save the generated GIFs.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_params = yaml.safe_load(f)

    evaluate_and_visualize(config_params, args.checkpoint, args.episodes, args.output_dir)