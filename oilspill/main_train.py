import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import os
import random
from collections import deque

from marl_framework.environments import OilSpillEnv
from marl_framework.agents import TransfQMixAgentNN
from marl_framework.mixers import TransfQMixMixer
from marl_framework.replay_buffer import ReplayBuffer
from marl_framework.utils import (
    setup_logger, get_tensorboard_writer, log_hyperparameters,
    save_checkpoint, load_checkpoint, EpisodeVisualizer, VISUALIZATION_ENABLED
)

def _soft_update_target_networks(policy_net, target_net, tau):
    """Soft update target networks using Polyak averaging."""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def _prepare_global_state_for_mixer(base_global_state_entities_batch, # List (batch_size) of List of raw entities (current, etc.)
                                   agent_f_cnn_features_batch,      # Tensor (batch_size, num_agents, cnn_dim)
                                   global_state_spec, config_env, device):
    """
    Constructs the full global state for the mixer by integrating agent-specific
    CNN features with other global entities (like environmental current).
    """
    batch_size = agent_f_cnn_features_batch.shape[0]
    num_agents = agent_f_cnn_features_batch.shape[1]
    final_global_state_batch_for_mixer = []

    for b_idx in range(batch_size):
        current_b_global_entities_raw = base_global_state_entities_batch[b_idx] # List of np.arrays
        num_global_entities_from_env = len(current_b_global_entities_raw)
        
        temp_entities_for_b = []
        for entity_idx in range(num_global_entities_from_env):
            raw_entity_np = current_b_global_entities_raw[entity_idx]
            entity_tensor = torch.from_numpy(raw_entity_np).float().to(device)

            if entity_idx < num_agents: # This is an agent's global state entity
                start_idx_fcnn = 2 + config_env.get("NUM_HEADINGS", 8) 
                end_idx_fcnn = start_idx_fcnn + agent_f_cnn_features_batch.shape[2] # cnn_dim
                
                if end_idx_fcnn <= len(entity_tensor):
                     entity_tensor[start_idx_fcnn:end_idx_fcnn] = agent_f_cnn_features_batch[b_idx, entity_idx, :]
                else:
                    pass # Keep original placeholder if indices are problematic
            
            temp_entities_for_b.append(entity_tensor)
        
        final_global_state_batch_for_mixer.append(temp_entities_for_b)
        
    return final_global_state_batch_for_mixer


def train(config):
    if config['use_cuda'] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

    logger, log_subdir = setup_logger(config['experiment_name'], 
                                      log_dir=os.path.join(config['logging']['log_dir'], config['experiment_name']),
                                      level=config['logging']['log_level'].upper())
    
    tb_writer_main_log_dir = os.path.join(config['logging']['tb_log_dir'], config['experiment_name'])
    tb_writer, tb_writer_subdir = get_tensorboard_writer(config['experiment_name'], tb_log_dir=tb_writer_main_log_dir)
    log_hyperparameters(tb_writer, config)

    # --- Initialize Environment ---
    env = OilSpillEnv(config['environment'], 
                      config['environment']['episode_data_directory'],
                      specific_episode_file=None)
    eval_env = OilSpillEnv(config['environment'], 
                           config['environment']['episode_data_directory'],
                           specific_episode_file=config['environment'].get('specific_episode_file'))

    # Perform a preliminary reset() to load cell_size_meters for the visualizer
    print("Performing initial environment reset to load parameters...")
    env.reset()
    eval_env.reset() # Also reset the eval env
    print(f"Environment configured with cell size: {env.cell_size_meters:.2f} meters")
    
    obs_spec = env.get_observation_spec()
    global_state_spec_env = env.get_observation_spec()
    action_space_size = env.get_action_space_size()
    num_agents = env.get_num_agents()
    agent_ids = env.get_agent_ids()

    # --- Initialize Networks ---
    agent_nn_global_config_ext = config['environment'].copy()
    agent_nn_global_config_ext["ACTION_SPACE_SIZE"] = action_space_size
    agent_policy_nn = TransfQMixAgentNN(obs_spec, config['agent_nn'], agent_nn_global_config_ext).to(device)
    agent_target_nn = TransfQMixAgentNN(obs_spec, config['agent_nn'], agent_nn_global_config_ext).to(device)
    agent_target_nn.load_state_dict(agent_policy_nn.state_dict())
    agent_target_nn.eval()
    mixer_policy_nn = TransfQMixMixer(num_agents, global_state_spec_env, 
                                     config['agent_nn']['AGENT_TRANSFORMER_EMBED_DIM'], 
                                     config['mixer_nn']).to(device)
    mixer_target_nn = TransfQMixMixer(num_agents, global_state_spec_env,
                                     config['agent_nn']['AGENT_TRANSFORMER_EMBED_DIM'],
                                     config['mixer_nn']).to(device)
    mixer_target_nn.load_state_dict(mixer_policy_nn.state_dict())
    mixer_target_nn.eval()

    params_to_optimize = list(agent_policy_nn.parameters()) + list(mixer_policy_nn.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config['training']['learning_rate'])

    replay_buffer = ReplayBuffer(config['training']['replay_buffer_capacity'], num_agents, agent_ids, obs_spec, global_state_spec_env, device)
    
    gif_output_path_template = os.path.join(
        config['logging']['visualization_gif_output_dir'], config['experiment_name'], tb_writer_subdir.split('/')[-1], "eval_ep{ep_num}_sub{eval_sub_ep}.gif"
    )
    if not os.path.exists(os.path.dirname(gif_output_path_template)):
        os.makedirs(os.path.dirname(gif_output_path_template))
    
    visualizer = EpisodeVisualizer(
        grid_size_r=config['environment']['GRID_SIZE_R'],
        grid_size_c=config['environment']['GRID_SIZE_C'],
        num_agents=num_agents,
        cell_size_m=env.cell_size_meters, # Use the discovered value
        enabled=config['logging']['visualization_enabled'] and VISUALIZATION_ENABLED
    )

    logger.info("Starting training...")
    total_env_steps = 0
    best_eval_iou = -float('inf')
    agent_h_states_prev = {aid: torch.zeros(1, config['agent_nn']['AGENT_TRANSFORMER_EMBED_DIM']).to(device) for aid in agent_ids}

    for episode_num in range(1, config['training']['num_training_episodes'] + 1):
        ep_reward = 0; ep_iou_sum = 0; ep_steps = 0
        agent_obs_dict, global_state_entities_np = env.reset()
        for aid in agent_ids: agent_h_states_prev[aid].zero_()

        for step_in_ep in range(config['environment']['MAX_STEPS_PER_EPISODE']):
            total_env_steps += 1; ep_steps += 1
            actions_dict = {}; agent_h_states_current = {}
            current_agent_belief_maps_dict_np = {aid: env.agent_belief_maps[aid]['belief'].copy() for aid in agent_ids}

            with torch.no_grad():
                for i, agent_id in enumerate(agent_ids):
                    belief_map_tensor = torch.from_numpy(current_agent_belief_maps_dict_np[agent_id]).float().unsqueeze(0).to(device)
                    obs_entities_list_for_agent = [agent_obs_dict[agent_id]] 
                    h_in_tensor = agent_h_states_prev[agent_id]
                    q_values_agent, h_out_agent, _ = agent_policy_nn(belief_map_tensor, obs_entities_list_for_agent, h_in_tensor)
                    agent_h_states_current[agent_id] = h_out_agent
                    epsilon = np.interp(total_env_steps, [0, config['training']['epsilon_anneal_time']], [config['training']['epsilon_start'], config['training']['epsilon_finish']])
                    actions_dict[agent_id] = random.randint(0, action_space_size - 1) if random.random() < epsilon else q_values_agent.argmax(dim=1).item()
            
            next_agent_obs_dict, next_global_state_entities_np, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)
            next_agent_belief_maps_dict_np = {aid: env.agent_belief_maps[aid]['belief'].copy() for aid in agent_ids}

            team_reward = rewards_dict[agent_ids[0]]; team_done = dones_dict["__all__"]
            ep_reward += team_reward; ep_iou_sum += infos_dict[agent_ids[0]]['iou']

            replay_buffer.push(agent_obs_dict, current_agent_belief_maps_dict_np, global_state_entities_np,
                               actions_dict, rewards_dict, 
                               next_agent_obs_dict, next_agent_belief_maps_dict_np, next_global_state_entities_np,
                               dones_dict, agent_h_states_prev, agent_h_states_current)
            
            agent_obs_dict = next_agent_obs_dict
            global_state_entities_np = next_global_state_entities_np
            agent_h_states_prev = agent_h_states_current

            if episode_num > config['training']['learning_starts_episodes'] and len(replay_buffer) >= config['training']['batch_size']:
                batch = replay_buffer.sample(config['training']['batch_size'])
                if batch is None: continue
                
                agent_obs_b, agent_belief_maps_b, global_state_b_raw_from_buffer, actions_b, reward_b, \
                next_agent_obs_b, next_agent_belief_maps_b, next_global_state_b_raw_from_buffer, done_b, \
                h_in_prev_b, h_out_curr_b = (batch['agent_obs'], batch['agent_belief_maps'], batch['global_state'], 
                                           batch['actions'], batch['reward'], batch['next_agent_obs'], 
                                           batch['next_agent_belief_maps'], batch['next_global_state'],
                                           batch['done'], batch['h_in_list'], batch['h_out_list'])

                chosen_action_q_vals_batch = torch.zeros(config['training']['batch_size'], num_agents, device=device)
                f_cnn_batch_policy = torch.zeros(config['training']['batch_size'], num_agents, config['agent_nn']['CNN_OUTPUT_FEATURE_DIM'], device=device)
                for agent_idx in range(num_agents):
                    belief_map_tensor_agent = agent_belief_maps_b[:, agent_idx, :, :]
                    h_in_tensor_agent = h_in_prev_b[:, agent_idx, :]
                    current_agent_obs_entities_for_batch = [agent_obs_b[b_idx][agent_idx] for b_idx in range(config['training']['batch_size'])]
                    q_all_actions_agent, _, f_cnn_agent = agent_policy_nn(belief_map_tensor_agent, current_agent_obs_entities_for_batch, h_in_tensor_agent)
                    action_taken_by_agent = actions_b[:, agent_idx].unsqueeze(1)
                    chosen_action_q_vals_batch[:, agent_idx] = q_all_actions_agent.gather(1, action_taken_by_agent).squeeze(1)
                    f_cnn_batch_policy[:, agent_idx, :] = f_cnn_agent
                
                global_state_for_policy_mixer = _prepare_global_state_for_mixer(global_state_b_raw_from_buffer, f_cnn_batch_policy, global_state_spec_env, config['environment'], device)
                q_total_policy = mixer_policy_nn(chosen_action_q_vals_batch, global_state_for_policy_mixer, h_out_curr_b)

                with torch.no_grad():
                    max_next_q_vals_target_batch = torch.zeros_like(chosen_action_q_vals_batch)
                    f_cnn_batch_target = torch.zeros_like(f_cnn_batch_policy)
                    h_next_target_batch = torch.zeros_like(h_out_curr_b)
                    for agent_idx in range(num_agents):
                        next_belief_map_tensor_agent = next_agent_belief_maps_b[:, agent_idx, :, :]
                        h_in_for_next_tensor_agent = h_out_curr_b[:, agent_idx, :]
                        current_next_agent_obs_entities_for_batch = [next_agent_obs_b[b_idx][agent_idx] for b_idx in range(config['training']['batch_size'])]
                        q_all_next_actions_target, h_out_target_agent, f_cnn_next_target_agent = agent_target_nn(next_belief_map_tensor_agent, current_next_agent_obs_entities_for_batch, h_in_for_next_tensor_agent)
                        max_next_q_vals_target_batch[:, agent_idx] = q_all_next_actions_target.max(dim=1)[0]
                        f_cnn_batch_target[:, agent_idx, :] = f_cnn_next_target_agent
                        h_next_target_batch[:, agent_idx, :] = h_out_target_agent
                    
                    global_state_for_target_mixer = _prepare_global_state_for_mixer(next_global_state_b_raw_from_buffer, f_cnn_batch_target, global_state_spec_env, config['environment'], device)
                    q_total_target = mixer_target_nn(max_next_q_vals_target_batch, global_state_for_target_mixer, h_next_target_batch)
                
                y_target = reward_b + config['training']['gamma'] * (1 - done_b.float()) * q_total_target.detach()
                loss = F.huber_loss(q_total_policy, y_target)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_optimize, config['training']['grad_norm_clip'])
                optimizer.step()
                tb_writer.add_scalar("Training/Loss", loss.item(), total_env_steps)

            if team_done: break
        
        avg_iou_ep = ep_iou_sum / ep_steps if ep_steps > 0 else 0
        logger.info(f"Ep: {episode_num}/{config['training']['num_training_episodes']} | Steps: {ep_steps} | Reward: {ep_reward:.2f} | Avg IoU: {avg_iou_ep:.3f}")
        tb_writer.add_scalar("Episode/Reward", ep_reward, episode_num)
        tb_writer.add_scalar("Episode/Avg_IoU", avg_iou_ep, episode_num)

        if episode_num % config['training']['target_update_interval_episodes'] == 0:
            _soft_update_target_networks(agent_policy_nn, agent_target_nn, config['training']['tau'])
            _soft_update_target_networks(mixer_policy_nn, mixer_target_nn, config['training']['tau'])

        # --- CORRECTED EVALUATION AND VISUALIZATION LOOP ---
        if episode_num % config['training']['evaluation_interval_episodes'] == 0:
            logger.info(f"--- Running Evaluation after Episode {episode_num} ---")
            total_eval_reward = 0; total_eval_iou = 0;
            eval_agent_h_states = {aid: torch.zeros(1, config['agent_nn']['AGENT_TRANSFORMER_EMBED_DIM']).to(device) for aid in agent_ids}

            for eval_ep_num in range(config['training']['num_evaluation_episodes']):
                eval_ep_reward = 0; eval_ep_iou_sum = 0; eval_ep_steps = 0
                obs_eval, gs_eval_np = eval_env.reset() # This sets the initial state and IoU
                for aid in agent_ids: eval_agent_h_states[aid].zero_()

                vis_this_eval_ep = visualizer.enabled and (eval_ep_num % config['logging']['visualization_interval_eval_episodes'] == 0)
                if vis_this_eval_ep:
                    current_gif_path = gif_output_path_template.format(ep_num=episode_num, eval_sub_ep=eval_ep_num)
                    visualizer.start_episode_recording(f"{episode_num}_eval{eval_ep_num}", current_gif_path)
                    
                    # Add a frame for the initial state (Step 0)
                    initial_iou = eval_env.iou_oil_previous_step
                    vis_info_str = f"Eval Ep {eval_ep_num+1}, Step 0 (Initial), IoU: {initial_iou:.3f}"
                    visualizer.add_frame(eval_env._get_ground_truth_grid(), 
                                         eval_env.agent_belief_maps,  # Make sure this is the correct dict
                                         eval_env.shared_consensus_map,
                                         eval_env.agent_positions_rc,
                                         eval_env.agent_headings,
                                         eval_env._get_current_vector_m_per_step(),
                                         timestep_info_string=vis_info_str)

                for step_in_eval_ep in range(config['environment']['MAX_STEPS_PER_EPISODE']):
                    eval_ep_steps += 1
                    actions_eval = {}
                    current_h_eval = {}
                    eval_belief_maps_np_dict = {aid: eval_env.agent_belief_maps[aid]['belief'] for aid in agent_ids}
                    
                    with torch.no_grad():
                        for agent_id in agent_ids:
                            belief_map_t = torch.from_numpy(eval_belief_maps_np_dict[agent_id]).float().unsqueeze(0).to(device)
                            obs_entities_list = [obs_eval[agent_id]]
                            h_in_t = eval_agent_h_states[agent_id]
                            q_vals_eval, h_out_eval, _ = agent_policy_nn(belief_map_t, obs_entities_list, h_in_t)
                            actions_eval[agent_id] = q_vals_eval.argmax(dim=1).item()
                            current_h_eval[agent_id] = h_out_eval
                    
                    # 1. Take a step in the environment
                    next_obs_eval, next_gs_eval_np, rewards_eval, dones_eval, infos_eval = eval_env.step(actions_eval)
                    
                    # 2. Add a frame for the NEW state, using the info from the step that led to it
                    if vis_this_eval_ep:
                         vis_info_str = f"Eval Ep {eval_ep_num+1}, Step {eval_ep_steps}, IoU: {infos_eval[agent_ids[0]]['iou']:.3f}"
                         visualizer.add_frame(eval_env._get_ground_truth_grid(), 
                                             eval_env.agent_belief_maps,  # Make sure this is the correct dict
                                             eval_env.shared_consensus_map,
                                             eval_env.agent_positions_rc,
                                             eval_env.agent_headings,
                                             eval_env._get_current_vector_m_per_step(),
                                             timestep_info_string=vis_info_str)

                    # 3. Update metrics and states for the next loop iteration
                    eval_ep_reward += rewards_eval[agent_ids[0]]
                    eval_ep_iou_sum += infos_eval[agent_ids[0]]['iou']
                    obs_eval, gs_eval_np = next_obs_eval, next_gs_eval_np
                    eval_agent_h_states = current_h_eval
                    
                    if dones_eval["__all__"]:
                        break
                
                if vis_this_eval_ep and visualizer.frames:
                    visualizer.save_recording(duration_per_frame_ms=config['logging']['visualization_duration_per_frame_ms'])

                total_eval_reward += eval_ep_reward
                total_eval_iou += (eval_ep_iou_sum / eval_ep_steps if eval_ep_steps > 0 else 0)
            
            avg_eval_reward = total_eval_reward / config['training']['num_evaluation_episodes']
            avg_eval_iou = total_eval_iou / config['training']['num_evaluation_episodes']
            logger.info(f"Evaluation complete: Avg Reward: {avg_eval_reward:.2f}, Avg IoU: {avg_eval_iou:.3f}")
            tb_writer.add_scalar("Evaluation/Avg_Reward", avg_eval_reward, total_env_steps)
            tb_writer.add_scalar("Evaluation/Avg_IoU", avg_eval_iou, total_env_steps)

            model_save_dir = os.path.join(config['logging']['model_save_dir'], config['experiment_name'], tb_writer_subdir.split('/')[-1])
            if avg_eval_iou > best_eval_iou:
                best_eval_iou = avg_eval_iou
                logger.info(f"New best evaluation IoU: {best_eval_iou:.3f}. Saving best model.")
                save_checkpoint({ 'episode': episode_num, 'best_eval_iou': best_eval_iou,
                    'agent_nn_state_dict': agent_policy_nn.state_dict(), 'mixer_nn_state_dict': mixer_policy_nn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() }, model_save_dir, is_best=True)

        if episode_num % config['training']['save_model_interval_episodes'] == 0:
            model_save_dir = os.path.join(config['logging']['model_save_dir'], config['experiment_name'], tb_writer_subdir.split('/')[-1])
            save_checkpoint({ 'episode': episode_num, 
                'agent_nn_state_dict': agent_policy_nn.state_dict(), 'mixer_nn_state_dict': mixer_policy_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() }, model_save_dir, episode=episode_num)

    logger.info("Training finished.")
    tb_writer.close()
    if visualizer.enabled: visualizer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TransfQMix MARL training for Oil Spill Response.")
    parser.add_argument("--config", type=str, default="marl_framework/configs/default_exp_config.yaml",
                        help="Path to the experiment configuration YAML file.")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config_params = yaml.safe_load(f)
    
    os.makedirs(os.path.join(config_params['logging']['log_dir'], config_params['experiment_name']), exist_ok=True)
    os.makedirs(os.path.join(config_params['logging']['tb_log_dir'], config_params['experiment_name']), exist_ok=True)
    os.makedirs(os.path.join(config_params['logging']['model_save_dir'], config_params['experiment_name']), exist_ok=True)
    os.makedirs(config_params['environment']['episode_data_directory'], exist_ok=True)
    os.makedirs(os.path.join(config_params['logging']['visualization_gif_output_dir'], config_params['experiment_name']), exist_ok=True)

    train(config_params)