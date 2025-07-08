import numpy as np
import os
import random
import json
from collections import OrderedDict # For ordered dict in obs/state spec

class OilSpillEnv:
    def __init__(self, experiment_hyperparams, episode_data_directory, specific_episode_file=None):
        self.config = experiment_hyperparams
        self.episode_data_dir = episode_data_directory
        self.specific_episode_file = specific_episode_file

        # Environment Grid and Agent Params from config
        self.grid_size_r = self.config.get("GRID_SIZE_R", 64)
        self.grid_size_c = self.config.get("GRID_SIZE_C", 64)
        
        # --- MODIFICATION START ---
        # cell_size_meters is now loaded from the episode data itself.
        # Initialize to None and set it in _load_episode.
        self.cell_size_meters = None 
        # --- MODIFICATION END ---
        
        self.num_agents = self.config.get("NUM_AGENTS", 3)
        self.num_headings = self.config.get("NUM_HEADINGS", 8)

        # Observation and Communication
        self.obs_radius_agents = self.config.get("OBSERVATION_RADIUS_AGENTS", 10)
        self.comm_radius_cells = self.config.get("COMMUNICATION_RADIUS_CELLS", 5)
        self.direct_sensing_mode = self.config.get("DIRECT_SENSING_MODE", "surrounding_cells")
        
        self.cnn_output_feature_dim = self.config.get("CNN_OUTPUT_FEATURE_DIM", 128)
        self.max_expected_current_mps = self.config.get("MAX_EXPECTED_CURRENT_MPS", 2.0)

        # Rewards
        self.reward_scaling_factor = self.config.get("REWARD_SCALING_FACTOR", 100.0)
        self.penalty_per_step = self.config.get("PENALTY_PER_STEP", -0.01)

        # Episode termination
        self.max_steps_per_episode = self.config.get("MAX_STEPS_PER_EPISODE", 400)

        # Action space (fixed)
        self.action_space_size = 6 

        # Internal state
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_positions_rc = {}
        self.agent_headings = {}
        self.agent_belief_maps = {} 
        self.shared_consensus_map = np.full((self.grid_size_r, self.grid_size_c), -1, dtype=np.int8)
        self.current_env_step = 0
        self.iou_oil_previous_step = 0.0

        # Load episode file list
        self.episode_files = []
        if self.specific_episode_file:
            if os.path.exists(os.path.join(self.episode_data_dir, self.specific_episode_file)):
                self.episode_files = [self.specific_episode_file]
            else:
                raise FileNotFoundError(f"Specific episode file not found: {os.path.join(self.episode_data_dir, self.specific_episode_file)}")
        else:
            for f_name in os.listdir(self.episode_data_dir):
                if f_name.endswith(".npz"):
                    self.episode_files.append(f_name)
        if not self.episode_files:
            raise ValueError(f"No episode (.npz) files found in {self.episode_data_dir}")
        
        self.current_episode_data = None
        self.current_episode_idx = -1
        self.env_time_step_hours = self.config.get("FALLBACK_ENV_TIME_STEP_HOURS", 0.1)

        self._max_entity_features = self._calculate_max_entity_feature_length()
        self._max_global_state_entity_features = self._calculate_max_global_state_entity_feature_length()


    def _calculate_max_entity_feature_length(self):
        # This method is safe as it doesn't depend on cell_size_meters
        self_len = 2 + self.num_headings + 4 
        other_agent_len = 2 + self.num_headings + 4
        belief_cnn_len = self.cnn_output_feature_dim + 4
        if self.direct_sensing_mode == "surrounding_cells":
            sensor_len = 9 + 4
        elif self.direct_sensing_mode == "current_cell":
            sensor_len = 1 + 4
        else:
            sensor_len = 0 + 4
        return max(self_len, other_agent_len, belief_cnn_len, sensor_len)

    def _calculate_max_global_state_entity_feature_length(self):
        # This method is safe as it doesn't depend on cell_size_meters
        agent_global_len = 2 + self.num_headings + self.cnn_output_feature_dim + 3
        env_current_len = 2 + 3
        pooled_map_dim = self.config.get("GLOBAL_BELIEF_POOLED_DIM", 8) 
        global_belief_len = (pooled_map_dim * pooled_map_dim) + 3
        return max(agent_global_len, env_current_len, global_belief_len)

    def _pad_features(self, features, target_len):
        padding_needed = target_len - len(features)
        if padding_needed < 0:
            raise ValueError(f"Feature length {len(features)} exceeds target {target_len}. Features: {features}")
        return np.pad(features, (0, padding_needed), 'constant', constant_values=0.0).astype(np.float32)

    def _load_episode(self):
        if self.specific_episode_file:
            episode_file_to_load = self.episode_files[0]
        else:
            self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episode_files)
            episode_file_to_load = self.episode_files[self.current_episode_idx]
        
        filepath = os.path.join(self.episode_data_dir, episode_file_to_load)
        try:
            self.current_episode_data = np.load(filepath, allow_pickle=True)
            
            gen_params_json = self.current_episode_data.get('generation_params_json')
            if gen_params_json is not None:
                gen_params = json.loads(gen_params_json.item())
                self.env_time_step_hours = gen_params.get('env_time_step_hours', self.env_time_step_hours)
                
                # --- MODIFICATION START ---
                # Load cell_size_meters from the episode data
                loaded_cell_size = gen_params.get('cell_size_meters')
                if loaded_cell_size is None:
                    raise ValueError(f"Episode file {episode_file_to_load} is missing 'cell_size_meters' in its generation parameters.")
                self.cell_size_meters = loaded_cell_size
                # --- MODIFICATION END ---
            else:
                 raise ValueError(f"Episode file {episode_file_to_load} is missing 'generation_params_json'. It might be an old format.")

        except Exception as e:
            raise IOError(f"Error loading episode file {filepath}: {e}")

    def reset(self):
        self._load_episode()
        # The rest of the reset logic is fine as it runs after _load_episode sets cell_size_meters
        self.current_env_step = 0
        self.iou_oil_previous_step = 0.0

        occupied_cells = set()
        for agent_id in self.agent_ids:
            while True:
                r = random.randint(0, self.grid_size_r - 1)
                c = random.randint(0, self.grid_size_c - 1)
                if (r,c) not in occupied_cells:
                    self.agent_positions_rc[agent_id] = np.array([r,c])
                    occupied_cells.add((r,c))
                    break
            self.agent_headings[agent_id] = random.randint(0, self.num_headings - 1)
            self.agent_belief_maps[agent_id] = {
                'belief': np.full((self.grid_size_r, self.grid_size_c), -1, dtype=np.int8),
                'timestamp': np.full((self.grid_size_r, self.grid_size_c), -1, dtype=np.int32)
            }
        
        self.shared_consensus_map.fill(-1)
        self._perform_sensing()
        self._perform_communication()
        self._update_shared_consensus_map()
        initial_iou = self._calculate_iou(self.shared_consensus_map, self._get_ground_truth_grid())
        self.iou_oil_previous_step = initial_iou

        obs_dict, global_state_entities = self._get_observations_and_state()
        return obs_dict, global_state_entities
    
    # ... The rest of the file (from _get_ground_truth_grid onwards) is unchanged ...
    # ... The step method correctly uses self.cell_size_meters, which will be populated by reset()
    def _get_ground_truth_grid(self):
        # Handle cases where current_env_step might exceed available data due to episode length vs max_steps
        idx = min(self.current_env_step, self.current_episode_data['ground_truth_grids'].shape[0] - 1)
        return self.current_episode_data['ground_truth_grids'][idx]

    def _get_current_vector_m_per_step(self):
        idx = min(self.current_env_step, self.current_episode_data['current_vectors_m_per_step'].shape[0] - 1)
        return self.current_episode_data['current_vectors_m_per_step'][idx]
        
    def _perform_sensing(self):
        gt_grid = self._get_ground_truth_grid()
        for agent_id in self.agent_ids:
            r_a, c_a = self.agent_positions_rc[agent_id]
            
            cells_to_sense = []
            if self.direct_sensing_mode == "current_cell":
                cells_to_sense.append((r_a, c_a))
            elif self.direct_sensing_mode == "surrounding_cells":
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        r_s, c_s = r_a + dr, c_a + dc
                        if 0 <= r_s < self.grid_size_r and 0 <= c_s < self.grid_size_c:
                            cells_to_sense.append((r_s, c_s))
            
            for r_s, c_s in cells_to_sense:
                sensed_value = gt_grid[r_s, c_s]
                self.agent_belief_maps[agent_id]['belief'][r_s, c_s] = sensed_value
                self.agent_belief_maps[agent_id]['timestamp'][r_s, c_s] = self.current_env_step

    def _perform_communication(self):
        if self.comm_radius_cells <= 0:
            return

        agent_ids_list = list(self.agent_ids)
        for i in range(len(agent_ids_list)):
            for j in range(i + 1, len(agent_ids_list)):
                id_a, id_b = agent_ids_list[i], agent_ids_list[j]
                pos_a, pos_b = self.agent_positions_rc[id_a], self.agent_positions_rc[id_b]
                
                dist = np.max(np.abs(pos_a - pos_b))
                
                if dist <= self.comm_radius_cells:
                    map_a_belief = self.agent_belief_maps[id_a]['belief']
                    map_a_ts = self.agent_belief_maps[id_a]['timestamp']
                    map_b_belief = self.agent_belief_maps[id_b]['belief']
                    map_b_ts = self.agent_belief_maps[id_b]['timestamp']

                    newer_from_b = map_b_ts > map_a_ts
                    map_a_belief[newer_from_b] = map_b_belief[newer_from_b]
                    map_a_ts[newer_from_b] = map_b_ts[newer_from_b]

                    newer_from_a = map_a_ts > map_b_ts
                    map_b_belief[newer_from_a] = map_a_belief[newer_from_a]
                    map_b_ts[newer_from_a] = map_a_ts[newer_from_a]


    def _update_shared_consensus_map(self):
        self.shared_consensus_map.fill(-1)
        
        for agent_id in self.agent_ids:
            belief_map = self.agent_belief_maps[agent_id]['belief']
            is_oil = (belief_map == 1)
            self.shared_consensus_map[is_oil] = 1
            
        for agent_id in self.agent_ids:
            belief_map = self.agent_belief_maps[agent_id]['belief']
            is_clean = (belief_map == 0)
            is_unknown_in_shared = (self.shared_consensus_map == -1)
            
            update_to_clean = is_clean & is_unknown_in_shared
            self.shared_consensus_map[update_to_clean] = 0

    def _calculate_iou(self, consensus_map, gt_grid):
        pred_oil = (consensus_map == 1)
        true_oil = (gt_grid == 1)

        tp = np.sum(pred_oil & true_oil)
        fp = np.sum(pred_oil & (~true_oil))
        fn = np.sum((~pred_oil) & true_oil)
        
        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0 if np.sum(true_oil) == 0 else 0.0
        return tp / denominator

    def _get_observations_and_state(self):
        observations_dict = {}
        global_state_entities = []

        for agent_id in self.agent_ids:
            r, c = self.agent_positions_rc[agent_id]
            heading = self.agent_headings[agent_id]
            f_cnn_a = np.random.rand(self.cnn_output_feature_dim).astype(np.float32)
            
            norm_r = r / self.grid_size_r
            norm_c = c / self.grid_size_c
            one_hot_h = np.zeros(self.num_headings, dtype=np.float32)
            one_hot_h[heading] = 1.0
            
            raw_features = np.concatenate([
                [norm_r, norm_c], one_hot_h, f_cnn_a, [1, 0, 0] 
            ]).astype(np.float32)
            global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))

        current_vec_m_per_step = self._get_current_vector_m_per_step()
        max_current_m_per_step_component = self.max_expected_current_mps * self.env_time_step_hours
        
        norm_curr_x = current_vec_m_per_step[0] / max_current_m_per_step_component if max_current_m_per_step_component else 0
        norm_curr_y = current_vec_m_per_step[1] / max_current_m_per_step_component if max_current_m_per_step_component else 0
        norm_curr_x = np.clip(norm_curr_x, -1, 1)
        norm_curr_y = np.clip(norm_curr_y, -1, 1)

        raw_features = np.array([norm_curr_x, norm_curr_y, 0, 1, 0], dtype=np.float32)
        global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))

        if self.config.get("INCLUDE_GLOBAL_BELIEF_IN_STATE", False):
            pooled_map_dim = self.config.get("GLOBAL_BELIEF_POOLED_DIM", 8)
            block_r = self.grid_size_r // pooled_map_dim
            block_c = self.grid_size_c // pooled_map_dim
            if block_r > 0 and block_c > 0 :
                pooled_map = self.shared_consensus_map.reshape(
                                pooled_map_dim, block_r, 
                                pooled_map_dim, block_c).max(axis=(1,3))
                flattened_pooled_map = pooled_map.flatten().astype(np.float32)
                flattened_pooled_map = (flattened_pooled_map + 1.0) / 2.0 
                raw_features = np.concatenate([flattened_pooled_map, [0,0,1]]).astype(np.float32)
                global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))

        for agent_id in self.agent_ids:
            agent_obs_entities = []
            r_a, c_a = self.agent_positions_rc[agent_id]
            h_a = self.agent_headings[agent_id]

            norm_r_a = r_a / self.grid_size_r
            norm_c_a = c_a / self.grid_size_c
            one_hot_h_a = np.zeros(self.num_headings, dtype=np.float32)
            one_hot_h_a[h_a] = 1.0
            raw_self_feats = np.concatenate([[norm_r_a, norm_c_a], one_hot_h_a, [1,1,0,0]]).astype(np.float32)
            agent_obs_entities.append(self._pad_features(raw_self_feats, self._max_entity_features))

            for other_id in self.agent_ids:
                if other_id == agent_id: continue
                r_b, c_b = self.agent_positions_rc[other_id]
                h_b = self.agent_headings[other_id]
                dist = np.max(np.abs(np.array([r_a,c_a]) - np.array([r_b,c_b])))
                if dist <= self.obs_radius_agents:
                    rel_r = (r_b - r_a) / (2 * self.obs_radius_agents) if self.obs_radius_agents > 0 else 0
                    rel_c = (c_b - c_a) / (2 * self.obs_radius_agents) if self.obs_radius_agents > 0 else 0
                    one_hot_h_b = np.zeros(self.num_headings, dtype=np.float32)
                    one_hot_h_b[h_b] = 1.0
                    raw_other_feats = np.concatenate([[rel_r, rel_c], one_hot_h_b, [0,1,0,0]]).astype(np.float32)
                    agent_obs_entities.append(self._pad_features(raw_other_feats, self._max_entity_features))
            
            f_cnn_a_dummy_for_obs_structure = np.random.rand(self.cnn_output_feature_dim).astype(np.float32)
            raw_cnn_feats = np.concatenate([f_cnn_a_dummy_for_obs_structure, [0,0,1,0]]).astype(np.float32)
            agent_obs_entities.append(self._pad_features(raw_cnn_feats, self._max_entity_features))

            if self.direct_sensing_mode != "none":
                sensed_features = []
                if self.direct_sensing_mode == "current_cell":
                    sensed_features.append(self.agent_belief_maps[agent_id]['belief'][r_a,c_a])
                elif self.direct_sensing_mode == "surrounding_cells":
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            r_s, c_s = r_a + dr, c_a + dc
                            if 0 <= r_s < self.grid_size_r and 0 <= c_s < self.grid_size_c:
                                sensed_features.append(self.agent_belief_maps[agent_id]['belief'][r_s,c_s])
                            else:
                                sensed_features.append(-1)
                sensed_features_normalized = (np.array(sensed_features, dtype=np.float32) + 1.0) / 2.0
                raw_sensor_feats = np.concatenate([sensed_features_normalized, [0,0,0,1]]).astype(np.float32)
                agent_obs_entities.append(self._pad_features(raw_sensor_feats, self._max_entity_features))

            observations_dict[agent_id] = agent_obs_entities
        
        return observations_dict, global_state_entities

    def _get_heading_vector(self, heading_int):
        if self.num_headings == 8:
            if heading_int == 0: return np.array([-1, 0])
            if heading_int == 1: return np.array([-1, 1])
            if heading_int == 2: return np.array([0, 1])
            if heading_int == 3: return np.array([1, 1])
            if heading_int == 4: return np.array([1, 0])
            if heading_int == 5: return np.array([1, -1])
            if heading_int == 6: return np.array([0, -1])
            if heading_int == 7: return np.array([-1, -1])
        elif self.num_headings == 4:
            if heading_int == 0: return np.array([-1, 0])
            if heading_int == 1: return np.array([0, 1])
            if heading_int == 2: return np.array([1, 0])
            if heading_int == 3: return np.array([0, -1])
        raise ValueError(f"Unsupported num_headings: {self.num_headings}")

    def step(self, actions_dict):
        """Execute one time step within the environment."""
        self.current_step += 1
        
        # Save the previous state for reward calculation
        previous_positions = self.agent_positions_rc.copy()
        
        # Process actions for each agent
        boundary_violation = False
        violating_agent_id = None
        
        for agent_id, action in actions_dict.items():
            # Get current position and calculate intended next position (before clipping)
            curr_pos = self.agent_positions_rc[agent_id]
            curr_heading = self.agent_headings[agent_id]
            
            # Calculate the intended next position based on action
            intended_pos = self._calculate_next_position(curr_pos, curr_heading, action)
            
            # Check if the intended position would be outside grid boundaries
            if (intended_pos[0] < 0 or intended_pos[0] >= self.grid_size_r or 
                intended_pos[1] < 0 or intended_pos[1] >= self.grid_size_c):
                boundary_violation = True
                violating_agent_id = agent_id
                break
        # If there's a boundary violation, terminate episode with penalty
        if boundary_violation:
            # Apply penalty to all agents (or specifically to the violating agent)
            penalty = self.config.get('BOUNDARY_VIOLATION_PENALTY', -10.0)  # Default penalty value if not specified
            rewards_dict = {agent_id: penalty for agent_id in self.agent_ids}
            
            # Set all agents as done
            dones_dict = {agent_id: True for agent_id in self.agent_ids}
            dones_dict["__all__"] = True
            
            # Log the boundary violation
            self.logger.info(f"Episode terminated: Agent {violating_agent_id} attempted to move outside grid boundaries.")
            
            # Return current observation since episode is terminating
            agent_obs = self._get_observations()
            global_state = self._get_global_state()
            infos = {agent_id: {'boundary_violation': True, 'violating_agent': violating_agent_id, 'iou': self.iou_oil_previous_step} 
                     for agent_id in self.agent_ids}
            
            return agent_obs, global_state, rewards_dict, dones_dict, infos
        
        # Continue with normal step execution if no boundary violation
        current_vec_m_per_step = self._get_current_vector_m_per_step()

        final_agent_positions_rc = {}
        for agent_id in self.agent_ids:
            r_curr, c_curr = self.agent_positions_rc[agent_id]
            x_m_curr = (c_curr + 0.5) * self.cell_size_meters
            y_m_curr = (r_curr + 0.5) * self.cell_size_meters
            dx_intent_m, dy_intent_m = intended_displacements_meters[agent_id]
            dx_total_m = dx_intent_m + current_vec_m_per_step[0] 
            dy_total_m = dy_intent_m + current_vec_m_per_step[1]
            x_m_new = x_m_curr + dx_total_m
            y_m_new = y_m_curr + dy_total_m
            domain_width_m = self.grid_size_c * self.cell_size_meters
            domain_height_m = self.grid_size_r * self.cell_size_meters
            x_m_new_clipped = np.clip(x_m_new, 0, domain_width_m - 1e-6)
            y_m_new_clipped = np.clip(y_m_new, 0, domain_height_m - 1e-6)
            c_final = int(x_m_new_clipped / self.cell_size_meters)
            r_final = int(y_m_new_clipped / self.cell_size_meters)
            final_agent_positions_rc[agent_id] = np.array([r_final, c_final])

        collision_detected = False
        final_occupied_cells = {}
        for agent_id in self.agent_ids:
            r_f, c_f = final_agent_positions_rc[agent_id]
            if (r_f, c_f) in final_occupied_cells:
                collision_detected = True
                break
            final_occupied_cells[(r_f, c_f)] = agent_id
        
        if not collision_detected:
            for agent_id in self.agent_ids:
                self.agent_positions_rc[agent_id] = final_agent_positions_rc[agent_id]
                self.agent_headings[agent_id] = intended_new_headings[agent_id]
        
        self.current_env_step += 1
        self._perform_sensing()
        self._perform_communication()
        self._update_shared_consensus_map()

        current_gt_grid = self._get_ground_truth_grid()
        iou_current = self._calculate_iou(self.shared_consensus_map, current_gt_grid)
        delta_iou = iou_current - self.iou_oil_previous_step
        reward_iou = delta_iou * self.reward_scaling_factor
        team_reward = reward_iou + self.penalty_per_step
        self.iou_oil_previous_step = iou_current

        done = collision_detected or (self.current_env_step >= self.max_steps_per_episode)
        
        next_obs_dict, next_global_state_entities = self._get_observations_and_state()

        rewards_dict = {agent_id: team_reward for agent_id in self.agent_ids}
        dones_dict = {agent_id: done for agent_id in self.agent_ids}
        dones_dict["__all__"] = done 
        
        infos_dict = {agent_id: {
            'iou': iou_current, 
            'delta_iou': delta_iou,
            'collision': collision_detected,
            'current_vec_m_per_step': self._get_current_vector_m_per_step().tolist(),
            'current_env_step': self.current_env_step
            } for agent_id in self.agent_ids}

        return next_obs_dict, next_global_state_entities, rewards_dict, dones_dict, infos_dict

    def _calculate_next_position(self, current_pos, heading, action):
        """Calculate the intended next position based on current position, heading, and action.
        This is called before position clipping to detect boundary violations."""
        # Convert the action to movement delta
        dr, dc = self._action_to_movement_delta(action, heading)
        
        # Calculate intended next position (without clipping)
        intended_r = current_pos[0] + dr
        intended_c = current_pos[1] + dc
        
        return (intended_r, intended_c)

    def _action_to_movement_delta(self, action, heading):
        """Convert action and heading to row and column movement deltas."""
        # If the existing implementation already has this method, use that instead
        # This is a simplified example assuming discrete actions
        if action == 0:  # Forward
            if heading == 0:  # North
                return -1, 0
            elif heading == 1:  # East
                return 0, 1
            elif heading == 2:  # South
                return 1, 0
            elif heading == 3:  # West
                return 0, -1
        elif action == 1:  # No movement
            return 0, 0
        # Add other actions as needed
        
        return 0, 0  # Default to no movement

    def get_num_agents(self):
        return self.num_agents

    def get_action_space_size(self):
        return self.action_space_size

    def get_agent_ids(self):
        return self.agent_ids

    def get_observation_spec(self):
        max_obs_entities = 1 + (self.num_agents -1) + 1 + 1 
        max_global_state_entities = self.num_agents + 1 + (1 if self.config.get("INCLUDE_GLOBAL_BELIEF_IN_STATE", False) else 0)
        spec = {
            "agent_observation": {
                "type": "list_of_variable_length_entities",
                "entity_feature_dim": self._max_entity_features,
            },
            "global_state": {
                "type": "list_of_variable_length_entities",
                "entity_feature_dim": self._max_global_state_entity_features,
            },
            "belief_map_shape": (self.grid_size_r, self.grid_size_c, 1)
        }
        return spec

    def render(self, mode='human'):
        # This method is also fine and will work correctly once cell_size_meters is set.
        pass