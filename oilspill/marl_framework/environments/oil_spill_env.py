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
        
        self.cell_size_meters = None 
        
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
        self.collision_penalty = self.config.get("COLLISION_PENALTY", -1.0)
        # --- MODIFICATION: Add boundary penalty to config ---
        self.boundary_violation_penalty = self.config.get("BOUNDARY_VIOLATION_PENALTY", -10.0)

        # Episode termination
        self.max_steps_per_episode = self.config.get("MAX_STEPS_PER_EPISODE", 400)

        self.action_space_size = 9

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

    # ... (no changes to other methods from __init__ up to step) ...
    def _calculate_max_entity_feature_length(self):
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
        agent_global_len = 2 + self.num_headings + self.cnn_output_feature_dim + 3
        env_current_len = 2 + 3
        
        pooled_map_dim = self.config.get("GLOBAL_BELIEF_POOLED_DIM", 8) 
        global_map_len = (pooled_map_dim * pooled_map_dim) + 3 
        
        return max(agent_global_len, env_current_len, global_map_len)

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
                
                loaded_cell_size = gen_params.get('cell_size_meters')
                if loaded_cell_size is None:
                    raise ValueError(f"Episode file {episode_file_to_load} is missing 'cell_size_meters' in its generation parameters.")
                self.cell_size_meters = loaded_cell_size
            else:
                 raise ValueError(f"Episode file {episode_file_to_load} is missing 'generation_params_json'. It might be an old format.")

        except Exception as e:
            raise IOError(f"Error loading episode file {filepath}: {e}")

    def reset(self):
        self._load_episode()
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
    
    def _get_ground_truth_grid(self):
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
        if self.comm_radius_cells <= 0: return
        agent_ids_list = list(self.agent_ids)
        for i in range(len(agent_ids_list)):
            for j in range(i + 1, len(agent_ids_list)):
                id_a, id_b = agent_ids_list[i], agent_ids_list[j]
                pos_a, pos_b = self.agent_positions_rc[id_a], self.agent_positions_rc[id_b]
                dist = np.max(np.abs(pos_a - pos_b))
                if dist <= self.comm_radius_cells:
                    map_a_belief, map_a_ts = self.agent_belief_maps[id_a]['belief'], self.agent_belief_maps[id_a]['timestamp']
                    map_b_belief, map_b_ts = self.agent_belief_maps[id_b]['belief'], self.agent_belief_maps[id_b]['timestamp']
                    newer_from_b = map_b_ts > map_a_ts
                    map_a_belief[newer_from_b], map_a_ts[newer_from_b] = map_b_belief[newer_from_b], map_b_ts[newer_from_b]
                    newer_from_a = map_a_ts > map_b_ts
                    map_b_belief[newer_from_a], map_b_ts[newer_from_a] = map_a_belief[newer_from_a], map_a_ts[newer_from_a]

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
        intersection = np.sum(pred_oil & true_oil)
        union = np.sum(pred_oil | true_oil)
        return 1.0 if union == 0 else intersection / union

    def _get_observations_and_state(self):
        observations_dict, global_state_entities = {}, []
        # --- Global State Entities ---
        for agent_id in self.agent_ids:
            r, c = self.agent_positions_rc[agent_id]
            h = self.agent_headings[agent_id]
            f_cnn_placeholder = np.zeros(self.cnn_output_feature_dim, dtype=np.float32)
            norm_r, norm_c = r / self.grid_size_r, c / self.grid_size_c
            one_hot_h = np.zeros(self.num_headings, dtype=np.float32); one_hot_h[h] = 1.0
            raw_features = np.concatenate([[norm_r, norm_c], one_hot_h, f_cnn_placeholder, [1, 0, 0]]).astype(np.float32)
            global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))
        current_vec_m_per_step = self._get_current_vector_m_per_step()
        max_curr_m_per_step = self.max_expected_current_mps * (self.env_time_step_hours * 3600) / self.cell_size_meters
        norm_curr_x = np.clip(current_vec_m_per_step[0] / self.cell_size_meters / max_curr_m_per_step if max_curr_m_per_step else 0, -1, 1)
        norm_curr_y = np.clip(current_vec_m_per_step[1] / self.cell_size_meters / max_curr_m_per_step if max_curr_m_per_step else 0, -1, 1)
        raw_features = np.array([norm_curr_x, norm_curr_y, 0, 1, 0], dtype=np.float32)
        global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))
        pooled_map_dim = self.config.get("GLOBAL_BELIEF_POOLED_DIM", 8)
        block_r, block_c = self.grid_size_r // pooled_map_dim, self.grid_size_c // pooled_map_dim
        def pool_map(grid_map):
            if block_r > 0 and block_c > 0:
                pooled = grid_map.reshape(pooled_map_dim, block_r, pooled_map_dim, block_c).max(axis=(1,3))
                return (pooled.flatten().astype(np.float32) + 1.0) / 2.0 
            return np.array([], dtype=np.float32)
        if self.config.get("INCLUDE_GLOBAL_BELIEF_IN_STATE", True):
            pooled_belief_features = pool_map(self.shared_consensus_map)
            raw_features = np.concatenate([pooled_belief_features, [0, 0, 1]]).astype(np.float32)
            global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))
        if self.config.get("INCLUDE_GROUND_TRUTH_IN_STATE", True):
            gt_as_belief = (self._get_ground_truth_grid() * 2) - 1 
            pooled_gt_features = pool_map(gt_as_belief)
            raw_features = np.concatenate([pooled_gt_features, [0, 0, 1]]).astype(np.float32)
            global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))
        # --- Agent-Specific Observations ---
        for agent_id in self.agent_ids:
            agent_obs_entities = []
            r_a, c_a = self.agent_positions_rc[agent_id]
            h_a = self.agent_headings[agent_id]
            norm_r_a, norm_c_a = r_a / self.grid_size_r, c_a / self.grid_size_c
            one_hot_h_a = np.zeros(self.num_headings, dtype=np.float32); one_hot_h_a[h_a] = 1.0
            raw_self_feats = np.concatenate([[norm_r_a, norm_c_a], one_hot_h_a, [1,1,0,0]]).astype(np.float32)
            agent_obs_entities.append(self._pad_features(raw_self_feats, self._max_entity_features))
            for other_id in self.agent_ids:
                if other_id == agent_id: continue
                r_b, c_b = self.agent_positions_rc[other_id]
                h_b = self.agent_headings[other_id]
                if np.max(np.abs(np.array([r_a,c_a]) - np.array([r_b,c_b]))) <= self.obs_radius_agents:
                    rel_r = (r_b - r_a) / (2 * self.obs_radius_agents) if self.obs_radius_agents > 0 else 0
                    rel_c = (c_b - c_a) / (2 * self.obs_radius_agents) if self.obs_radius_agents > 0 else 0
                    one_hot_h_b = np.zeros(self.num_headings, dtype=np.float32); one_hot_h_b[h_b] = 1.0
                    raw_other_feats = np.concatenate([[rel_r, rel_c], one_hot_h_b, [0,1,0,0]]).astype(np.float32)
                    agent_obs_entities.append(self._pad_features(raw_other_feats, self._max_entity_features))
            f_cnn_placeholder = np.zeros(self.cnn_output_feature_dim).astype(np.float32)
            raw_cnn_feats = np.concatenate([f_cnn_placeholder, [0,0,1,0]]).astype(np.float32)
            agent_obs_entities.append(self._pad_features(raw_cnn_feats, self._max_entity_features))
            if self.direct_sensing_mode != "none":
                sensed_features = []
                if self.direct_sensing_mode == "current_cell":
                    sensed_features.append(self.agent_belief_maps[agent_id]['belief'][r_a,c_a])
                elif self.direct_sensing_mode == "surrounding_cells":
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            r_s, c_s = r_a+dr, c_a+dc
                            sensed_features.append(self.agent_belief_maps[agent_id]['belief'][r_s,c_s] if 0<=r_s<self.grid_size_r and 0<=c_s<self.grid_size_c else -1)
                sensed_norm = (np.array(sensed_features, dtype=np.float32) + 1.0) / 2.0
                raw_sensor_feats = np.concatenate([sensed_norm, [0,0,0,1]]).astype(np.float32)
                agent_obs_entities.append(self._pad_features(raw_sensor_feats, self._max_entity_features))
            observations_dict[agent_id] = agent_obs_entities
        return observations_dict, global_state_entities

    def _action_to_delta(self, action):
        """Converts one of the 9 actions to a (dr, dc) grid delta."""
        deltas = {0: (0, 0), 1: (-1, 0), 2: (-1, 1), 3: (0, 1), 4: (1, 1),
                  5: (1, 0), 6: (1, -1), 7: (0, -1), 8: (-1, -1)}
        return deltas.get(action, (0,0))

    def step(self, actions_dict):
        """Execute one time step within the environment."""
        self.current_env_step += 1
        
        # --- MODIFICATION START: Boundary Violation Check ---
        boundary_violation = False
        violating_agent_id = None
        
        for agent_id, action in actions_dict.items():
            curr_pos = self.agent_positions_rc[agent_id]
            dr, dc = self._action_to_delta(action)
            intended_pos_r, intended_pos_c = curr_pos[0] + dr, curr_pos[1] + dc
            
            # Check if the intended grid position is out of bounds
            if not (0 <= intended_pos_r < self.grid_size_r and 0 <= intended_pos_c < self.grid_size_c):
                boundary_violation = True
                violating_agent_id = agent_id
                break # One violation is enough to terminate

        if boundary_violation:
            # Episode terminates immediately with a penalty
            rewards_dict = {aid: self.boundary_violation_penalty for aid in self.agent_ids}
            dones_dict = {aid: True for aid in self.agent_ids}
            dones_dict["__all__"] = True
            
            # The "next" observation is the same as the current one since the state doesn't change
            obs_dict, global_state = self._get_observations_and_state()

            infos_dict = {aid: {
                'iou': self.iou_oil_previous_step, 
                'delta_iou': 0,
                'collision': False,
                'boundary_violation': True, # Add flag to info
                'violating_agent': violating_agent_id,
                'current_vec_m_per_step': self._get_current_vector_m_per_step().tolist(),
                'current_env_step': self.current_env_step
            } for aid in self.agent_ids}
            
            return obs_dict, global_state, rewards_dict, dones_dict, infos_dict
        # --- MODIFICATION END: Boundary Violation Check ---

        # --- If no boundary violation, proceed with normal step logic ---
        
        # 1. Calculate intended agent displacements from actions
        intended_displacements_meters = {}
        for agent_id, action in actions_dict.items():
            dr, dc = self._action_to_delta(action)
            intended_displacements_meters[agent_id] = np.array([
                dc * self.cell_size_meters, 
                dr * self.cell_size_meters
            ])
        
        # 2. Calculate final positions considering currents
        current_vec_m_per_step = self._get_current_vector_m_per_step()
        final_agent_positions_rc = {}
        for agent_id in self.agent_ids:
            r_curr, c_curr = self.agent_positions_rc[agent_id]
            x_m_curr, y_m_curr = (c_curr + 0.5) * self.cell_size_meters, (r_curr + 0.5) * self.cell_size_meters
            dx_intent_m, dy_intent_m = intended_displacements_meters[agent_id]
            dx_total_m, dy_total_m = dx_intent_m + current_vec_m_per_step[0], dy_intent_m + current_vec_m_per_step[1]
            x_m_new, y_m_new = x_m_curr + dx_total_m, y_m_curr + dy_total_m
            domain_w_m, domain_h_m = self.grid_size_c * self.cell_size_meters, self.grid_size_r * self.cell_size_meters
            x_m_clip, y_m_clip = np.clip(x_m_new, 0, domain_w_m - 1e-6), np.clip(y_m_new, 0, domain_h_m - 1e-6)
            final_agent_positions_rc[agent_id] = np.array([int(y_m_clip / self.cell_size_meters), int(x_m_clip / self.cell_size_meters)])

        # 3. Handle collisions
        collision_detected = False
        final_occupied_cells = {}
        sorted_agent_ids = sorted(self.agent_ids) # Process in a fixed order
        for agent_id in sorted_agent_ids:
            r_f, c_f = final_agent_positions_rc[agent_id]
            if (r_f, c_f) in final_occupied_cells.values():
                collision_detected = True
                # Agent that attempts to move into an occupied cell stays put
                final_agent_positions_rc[agent_id] = self.agent_positions_rc[agent_id]
            else:
                final_occupied_cells[agent_id] = (r_f, c_f)
        self.agent_positions_rc = final_agent_positions_rc
        
        # 4. Update environment state
        self._perform_sensing()
        self._perform_communication()
        self._update_shared_consensus_map()

        # 5. Calculate reward and done
        iou_current = self._calculate_iou(self.shared_consensus_map, self._get_ground_truth_grid())
        delta_iou = iou_current - self.iou_oil_previous_step
        team_reward = (delta_iou * self.reward_scaling_factor) + self.penalty_per_step
        if collision_detected: team_reward += self.collision_penalty
        self.iou_oil_previous_step = iou_current
        done = (self.current_env_step >= self.max_steps_per_episode)
        
        # 6. Get next observations and global state
        next_obs_dict, next_global_state_entities = self._get_observations_and_state()
        rewards_dict = {aid: team_reward for aid in self.agent_ids}
        dones_dict = {aid: done for aid in self.agent_ids}; dones_dict["__all__"] = done
        infos_dict = {aid: {'iou': iou_current, 'delta_iou': delta_iou, 'collision': collision_detected, 'boundary_violation': False,
                             'current_vec_m_per_step': current_vec_m_per_step.tolist(), 'current_env_step': self.current_env_step
                            } for aid in self.agent_ids}
        return next_obs_dict, next_global_state_entities, rewards_dict, dones_dict, infos_dict

    # ... (no changes to the remaining methods get_num_agents, get_action_space_size, etc.) ...
    def get_num_agents(self):
        return self.num_agents

    def get_action_space_size(self):
        return self.action_space_size

    def get_agent_ids(self):
        return self.agent_ids

    def get_observation_spec(self):
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
        pass