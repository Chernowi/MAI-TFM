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
        self.cell_size_meters = self.config.get("CELL_SIZE_METERS", 100.0)
        self.num_agents = self.config.get("NUM_AGENTS", 3)
        self.num_headings = self.config.get("NUM_HEADINGS", 8) # 0:N, 1:NE, ..., 7:NW

        # Observation and Communication
        self.obs_radius_agents = self.config.get("OBSERVATION_RADIUS_AGENTS", 10) # in grid cells
        self.comm_radius_cells = self.config.get("COMMUNICATION_RADIUS_CELLS", 5) # Chebyshev distance
        self.direct_sensing_mode = self.config.get("DIRECT_SENSING_MODE", "surrounding_cells") # "current_cell" or "surrounding_cells"
        
        # CNN output feature dimension (needed for observation construction)
        self.cnn_output_feature_dim = self.config.get("CNN_OUTPUT_FEATURE_DIM", 128)

        # Max Current Normalization (for state representation)
        self.max_expected_current_mps = self.config.get("MAX_EXPECTED_CURRENT_MPS", 2.0) # meters per second

        # Rewards
        self.reward_scaling_factor = self.config.get("REWARD_SCALING_FACTOR", 100.0)
        self.penalty_per_step = self.config.get("PENALTY_PER_STEP", -0.01)

        # Episode termination
        self.max_steps_per_episode = self.config.get("MAX_STEPS_PER_EPISODE", 400)

        # Action space (fixed)
        self.action_space_size = 6 
        # 0: STAY, 1: MOVE_FORWARD, 2: MOVE_DIAG_LEFT, 3: MOVE_DIAG_RIGHT,
        # 4: TURN_LEFT_90_AND_MOVE, 5: TURN_RIGHT_90_AND_MOVE

        # Internal state
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_positions_rc = {} # {agent_id: [r, c]}
        self.agent_headings = {}    # {agent_id: int_heading}
        self.agent_belief_maps = {} # {agent_id: {'belief': np.array, 'timestamp': np.array}}
        
        self.shared_consensus_map = np.full((self.grid_size_r, self.grid_size_c), -1, dtype=np.int8) # -1 unknown, 0 clean, 1 oil
        
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
        self.current_episode_idx = -1 # for sequential loading if specific_episode_file is None

        # Calculate max possible features for entity embedder consistency
        # This needs careful calculation based on all entity types
        self._max_entity_features = self._calculate_max_entity_feature_length()
        self._max_global_state_entity_features = self._calculate_max_global_state_entity_feature_length()


    def _calculate_max_entity_feature_length(self):
        # Self: norm_r, norm_c, one_hot_heading, IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR
        self_len = 2 + self.num_headings + 4 
        
        # Other Agent: rel_norm_r, rel_norm_c, one_hot_heading, IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR
        other_agent_len = 2 + self.num_headings + 4
        
        # Belief Map CNN: cnn_features_vector, IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR
        belief_cnn_len = self.cnn_output_feature_dim + 4
        
        # Sensor: 
        if self.direct_sensing_mode == "surrounding_cells":
            sensor_len = 9 + 4 # 9 cells + 4 flags
        elif self.direct_sensing_mode == "current_cell":
            sensor_len = 1 + 4 # 1 cell + 4 flags
        else: # No explicit sensor entity
            sensor_len = 0 + 4 # Only flags if an empty sensor entity is needed for padding
            
        return max(self_len, other_agent_len, belief_cnn_len, sensor_len)

    def _calculate_max_global_state_entity_feature_length(self):
        # Agent in Global State: norm_r, norm_c, one_hot_heading, cnn_features, IS_AGENT, IS_ENV, IS_GLOBAL_BELIEF
        agent_global_len = 2 + self.num_headings + self.cnn_output_feature_dim + 3

        # Env Current: norm_curr_x, norm_curr_y, IS_AGENT, IS_ENV, IS_GLOBAL_BELIEF
        env_current_len = 2 + 3

        # Global Belief Summary (Optional, assuming max-pool to fixed size, e.g., 8x8 from 64x64)
        # For simplicity, let's assume a fixed max-pooled size for now
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
            # Validate shapes
            expected_steps = self.current_episode_data['ground_truth_grids'].shape[0]
            if expected_steps < self.max_steps_per_episode:
                print(f"Warning: Loaded episode {episode_file_to_load} has {expected_steps} steps, less than env max {self.max_steps_per_episode}. Training might be shorter.")
            # self.max_steps_per_episode = min(self.max_steps_per_episode, expected_steps) # Option: cap env steps to episode length
            
            # Load generation params to get env_time_step_hours for current conversion
            gen_params_json = self.current_episode_data.get('generation_params_json')
            if gen_params_json is not None:
                gen_params = json.loads(gen_params_json.item())
                self.env_time_step_hours = gen_params.get('env_time_step_hours', 0.1) # Default if not found
            else: # Fallback if older data format
                self.env_time_step_hours = self.config.get("FALLBACK_ENV_TIME_STEP_HOURS", 0.1)


        except Exception as e:
            raise IOError(f"Error loading episode file {filepath}: {e}")

    def reset(self):
        self._load_episode()
        self.current_env_step = 0
        self.iou_oil_previous_step = 0.0

        # Initialize agent positions (e.g., random, or fixed starting for debugging)
        # For now, random non-overlapping start
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

        # Initial sensing and communication before first step's observations
        self._perform_sensing()
        self._perform_communication()
        self._update_shared_consensus_map()
        
        # Calculate initial IoU
        initial_iou = self._calculate_iou(self.shared_consensus_map, self._get_ground_truth_grid())
        self.iou_oil_previous_step = initial_iou

        obs_dict, global_state_entities = self._get_observations_and_state()
        return obs_dict, global_state_entities

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
                
                # Chebyshev distance
                dist = np.max(np.abs(pos_a - pos_b))
                
                if dist <= self.comm_radius_cells:
                    # Exchange and update belief maps
                    map_a_belief = self.agent_belief_maps[id_a]['belief']
                    map_a_ts = self.agent_belief_maps[id_a]['timestamp']
                    map_b_belief = self.agent_belief_maps[id_b]['belief']
                    map_b_ts = self.agent_belief_maps[id_b]['timestamp']

                    # A updates from B
                    newer_from_b = map_b_ts > map_a_ts
                    map_a_belief[newer_from_b] = map_b_belief[newer_from_b]
                    map_a_ts[newer_from_b] = map_b_ts[newer_from_b]

                    # B updates from A (use original A's map before it was updated by B in this same comm step)
                    # To be fully symmetric, one could use copies or a two-pass approach.
                    # For simplicity here, A updates first, then B updates from potentially A's newly acquired info from B.
                    # Or, more robustly, copy B's original map before A updates. Let's do a slightly simpler version.
                    # This implies information can propagate faster.
                    # A more correct pair-wise exchange would be:
                    # 1. Identify cells where B is newer than A. Store these updates for A.
                    # 2. Identify cells where A is newer than B. Store these updates for B.
                    # 3. Apply stored updates.
                    # Current simpler:
                    newer_from_a = map_a_ts > map_b_ts # map_a_ts might have been updated by B
                    map_b_belief[newer_from_a] = map_a_belief[newer_from_a]
                    map_b_ts[newer_from_a] = map_a_ts[newer_from_a]


    def _update_shared_consensus_map(self):
        self.shared_consensus_map.fill(-1) # Reset to unknown
        
        # Pass 1: Mark all known 'oil' cells
        for agent_id in self.agent_ids:
            belief_map = self.agent_belief_maps[agent_id]['belief']
            is_oil = (belief_map == 1)
            self.shared_consensus_map[is_oil] = 1
            
        # Pass 2: Mark 'clean' cells if still unknown
        for agent_id in self.agent_ids:
            belief_map = self.agent_belief_maps[agent_id]['belief']
            is_clean = (belief_map == 0)
            is_unknown_in_shared = (self.shared_consensus_map == -1)
            
            update_to_clean = is_clean & is_unknown_in_shared
            self.shared_consensus_map[update_to_clean] = 0

    def _calculate_iou(self, consensus_map, gt_grid):
        # Ensure -1 (unknown) in consensus_map is treated as not predicting oil (e.g. map to 0 for IoU)
        # Or, define TP, FP, FN based on positive class (oil=1)
        
        pred_oil = (consensus_map == 1)
        true_oil = (gt_grid == 1)

        tp = np.sum(pred_oil & true_oil)
        fp = np.sum(pred_oil & (~true_oil))
        fn = np.sum((~pred_oil) & true_oil)
        
        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0 if np.sum(true_oil) == 0 else 0.0 # Perfect if no oil and none predicted, 0 if oil exists but not found
        return tp / denominator

    def _get_observations_and_state(self):
        observations_dict = {}
        global_state_entities = [] # For the mixer

        # --- Populate Global State Entities ---
        # 1. Agent entities for global state
        for agent_id in self.agent_ids:
            r, c = self.agent_positions_rc[agent_id]
            heading = self.agent_headings[agent_id]
            # Assuming agent's BeliefMapCNN output is pre-calculated and stored
            # For now, let's use a placeholder if the CNN part is not yet integrated into this step
            # This F_cnn_a should be the output of agent_a's BeliefMapCNN on its agent_a_belief_map
            # This requires a call to the CNN model from within the env, or the trainer passes it in.
            # For env purity, the trainer should compute CNN features. Here, we'll assume they are passed/available.
            # Simplified: Let's assume for now the observation construction has access to F_cnn_a.
            # In a real TransfQMix, the agent policy network would compute F_cnn_a.
            # For the *global state*, F_cnn_a needs to be part of what's collected.
            # This implies the environment might need access to these if they are generated by agent policies.
            # Let's create dummy F_cnn_a for now for structure.
            f_cnn_a = np.random.rand(self.cnn_output_feature_dim).astype(np.float32) # DUMMY
            
            norm_r = r / self.grid_size_r
            norm_c = c / self.grid_size_c
            one_hot_h = np.zeros(self.num_headings, dtype=np.float32)
            one_hot_h[heading] = 1.0
            
            # Features: norm_r, norm_c, one_hot_h, f_cnn_a, IS_AGENT=1, IS_ENV=0, IS_GLOBAL_BELIEF=0
            raw_features = np.concatenate([
                [norm_r, norm_c], one_hot_h, f_cnn_a, [1, 0, 0] 
            ]).astype(np.float32)
            global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))

        # 2. Environmental current entity for global state
        current_vec_m_per_step = self._get_current_vector_m_per_step()
        # Normalize current. Max_expected_current_mps needs conversion to m/step
        max_current_m_per_step_component = self.max_expected_current_mps * self.env_time_step_hours
        
        norm_curr_x = current_vec_m_per_step[0] / max_current_m_per_step_component if max_current_m_per_step_component else 0
        norm_curr_y = current_vec_m_per_step[1] / max_current_m_per_step_component if max_current_m_per_step_component else 0
        norm_curr_x = np.clip(norm_curr_x, -1, 1)
        norm_curr_y = np.clip(norm_curr_y, -1, 1)

        # Features: norm_curr_x, norm_curr_y, IS_AGENT=0, IS_ENV=1, IS_GLOBAL_BELIEF=0
        raw_features = np.array([norm_curr_x, norm_curr_y, 0, 1, 0], dtype=np.float32)
        global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))

        # 3. (Optional) Global shared belief summary entity
        if self.config.get("INCLUDE_GLOBAL_BELIEF_IN_STATE", False):
            pooled_map_dim = self.config.get("GLOBAL_BELIEF_POOLED_DIM", 8)
            # Simplified max pooling
            block_r = self.grid_size_r // pooled_map_dim
            block_c = self.grid_size_c // pooled_map_dim
            if block_r > 0 and block_c > 0 : # only if pooling is meaningful
                pooled_map = self.shared_consensus_map.reshape(
                                pooled_map_dim, block_r, 
                                pooled_map_dim, block_c).max(axis=(1,3))
                flattened_pooled_map = pooled_map.flatten().astype(np.float32)
                # Normalize -1,0,1 to 0,0.5,1 for features
                flattened_pooled_map = (flattened_pooled_map + 1.0) / 2.0 
                # Features: flattened_map, IS_AGENT=0, IS_ENV=0, IS_GLOBAL_BELIEF=1
                raw_features = np.concatenate([flattened_pooled_map, [0,0,1]]).astype(np.float32)
                global_state_entities.append(self._pad_features(raw_features, self._max_global_state_entity_features))


        # --- Populate Agent Observations ---
        for agent_id in self.agent_ids:
            agent_obs_entities = []
            r_a, c_a = self.agent_positions_rc[agent_id]
            h_a = self.agent_headings[agent_id]

            # 1. Self-entity
            norm_r_a = r_a / self.grid_size_r
            norm_c_a = c_a / self.grid_size_c
            one_hot_h_a = np.zeros(self.num_headings, dtype=np.float32)
            one_hot_h_a[h_a] = 1.0
            # Features: norm_r, norm_c, one_hot_heading, IS_SELF=1, IS_AGENT=1, IS_MAP=0, IS_SENSOR=0
            raw_self_feats = np.concatenate([[norm_r_a, norm_c_a], one_hot_h_a, [1,1,0,0]]).astype(np.float32)
            agent_obs_entities.append(self._pad_features(raw_self_feats, self._max_entity_features))

            # 2. Other nearby agent entities
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
                    # Features: rel_norm_r, rel_norm_c, one_hot_h, IS_SELF=0, IS_AGENT=1, IS_MAP=0, IS_SENSOR=0
                    raw_other_feats = np.concatenate([[rel_r, rel_c], one_hot_h_b, [0,1,0,0]]).astype(np.float32)
                    agent_obs_entities.append(self._pad_features(raw_other_feats, self._max_entity_features))
            
            # 3. Individual Belief Map CNN Feature Entity
            # This F_cnn_a should be generated by the agent's policy network.
            # The environment is just structuring how it would be included.
            # For environment's observation construction, we pass a DUMMY one.
            # The actual F_cnn_a will be computed by the AgentNN and used in its transformer.
            f_cnn_a_dummy_for_obs_structure = np.random.rand(self.cnn_output_feature_dim).astype(np.float32) # DUMMY
            # Features: F_cnn_a, IS_SELF=0, IS_AGENT=0, IS_MAP=1, IS_SENSOR=0
            raw_cnn_feats = np.concatenate([f_cnn_a_dummy_for_obs_structure, [0,0,1,0]]).astype(np.float32)
            agent_obs_entities.append(self._pad_features(raw_cnn_feats, self._max_entity_features))

            # 4. (Optional) Explicit Local Sensor Reading Entity
            if self.direct_sensing_mode != "none": # Assuming "none" means no explicit sensor entity
                sensed_features = []
                if self.direct_sensing_mode == "current_cell":
                    sensed_features.append(self.agent_belief_maps[agent_id]['belief'][r_a,c_a])
                elif self.direct_sensing_mode == "surrounding_cells":
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]: # N, NE, E, SE, S, SW, W, NW, Current (order matters for consistency)
                            r_s, c_s = r_a + dr, c_a + dc
                            if 0 <= r_s < self.grid_size_r and 0 <= c_s < self.grid_size_c:
                                sensed_features.append(self.agent_belief_maps[agent_id]['belief'][r_s,c_s])
                            else:
                                sensed_features.append(-1) # Pad for out-of-bounds with "unknown"
                
                # Normalize -1,0,1 to 0,0.5,1
                sensed_features_normalized = (np.array(sensed_features, dtype=np.float32) + 1.0) / 2.0
                # Features: sensed_values, IS_SELF=0, IS_AGENT=0, IS_MAP=0, IS_SENSOR=1
                raw_sensor_feats = np.concatenate([sensed_features_normalized, [0,0,0,1]]).astype(np.float32)
                agent_obs_entities.append(self._pad_features(raw_sensor_feats, self._max_entity_features))

            observations_dict[agent_id] = agent_obs_entities
        
        return observations_dict, global_state_entities

    def _get_heading_vector(self, heading_int):
        # 0:N (0,1), 1:NE (1,1), 2:E (1,0), 3:SE (1,-1), 4:S (0,-1), 5:SW (-1,-1), 6:W (-1,0), 7:NW (-1,1)
        # Assuming grid: +r is Down (South), +c is Right (East)
        # So, N is (-1,0) in [dr,dc] grid steps
        if self.num_headings == 8:
            if heading_int == 0: return np.array([-1, 0])  # N
            if heading_int == 1: return np.array([-1, 1])  # NE
            if heading_int == 2: return np.array([0, 1])   # E
            if heading_int == 3: return np.array([1, 1])   # SE
            if heading_int == 4: return np.array([1, 0])   # S
            if heading_int == 5: return np.array([1, -1])  # SW
            if heading_int == 6: return np.array([0, -1])  # W
            if heading_int == 7: return np.array([-1, -1]) # NW
        elif self.num_headings == 4: # N, E, S, W
            if heading_int == 0: return np.array([-1, 0]) # N
            if heading_int == 1: return np.array([0, 1])  # E
            if heading_int == 2: return np.array([1, 0])  # S
            if heading_int == 3: return np.array([0, -1]) # W
        raise ValueError(f"Unsupported num_headings: {self.num_headings}")


    def step(self, actions_dict):
        # 1. Determine intended new headings and target cells based on actions
        intended_new_headings = {}
        intended_target_cells_rc = {} # grid coords
        intended_displacements_meters = {} # meters

        for agent_id, action in actions_dict.items():
            r_curr, c_curr = self.agent_positions_rc[agent_id]
            h_curr = self.agent_headings[agent_id]
            h_new_intent = h_curr # Default: heading maintained

            dr_intent, dc_intent = 0, 0 # Grid steps

            if action == 0: # STAY
                pass 
            elif action == 1: # MOVE_FORWARD
                dh_vec = self._get_heading_vector(h_curr)
                dr_intent, dc_intent = dh_vec[0], dh_vec[1]
            elif action == 2: # MOVE_DIAG_LEFT (-45 deg relative)
                h_new_intent = (h_curr - 1 + self.num_headings) % self.num_headings # -45 deg for 8 headings
                if self.num_headings == 4: h_new_intent = (h_curr -1 + self.num_headings) % self.num_headings # effectively -90 for 4 headings
                dh_vec = self._get_heading_vector(h_new_intent)
                dr_intent, dc_intent = dh_vec[0], dh_vec[1]
            elif action == 3: # MOVE_DIAG_RIGHT (+45 deg relative)
                h_new_intent = (h_curr + 1) % self.num_headings # +45 deg for 8 headings
                if self.num_headings == 4: h_new_intent = (h_curr + 1) % self.num_headings # effectively +90 for 4 headings
                dh_vec = self._get_heading_vector(h_new_intent)
                dr_intent, dc_intent = dh_vec[0], dh_vec[1]
            elif action == 4: # TURN_LEFT_90_AND_MOVE
                h_new_intent = (h_curr - (self.num_headings // 4) + self.num_headings) % self.num_headings # -90 deg
                dh_vec = self._get_heading_vector(h_new_intent)
                dr_intent, dc_intent = dh_vec[0], dh_vec[1]
            elif action == 5: # TURN_RIGHT_90_AND_MOVE
                h_new_intent = (h_curr + (self.num_headings // 4)) % self.num_headings # +90 deg
                dh_vec = self._get_heading_vector(h_new_intent)
                dr_intent, dc_intent = dh_vec[0], dh_vec[1]
            
            intended_new_headings[agent_id] = h_new_intent
            intended_target_cells_rc[agent_id] = np.array([r_curr + dr_intent, c_curr + dc_intent])
            
            # Convert intended grid displacement to meters (center to center)
            # dx_intent_m = dc_intent * self.cell_size_meters
            # dy_intent_m = dr_intent * self.cell_size_meters 
            # This is displacement from current cell center to target cell center.
            # A more precise way:
            x_m_curr = (c_curr + 0.5) * self.cell_size_meters
            y_m_curr = (r_curr + 0.5) * self.cell_size_meters
            x_m_target = (intended_target_cells_rc[agent_id][1] + 0.5) * self.cell_size_meters
            y_m_target = (intended_target_cells_rc[agent_id][0] + 0.5) * self.cell_size_meters
            intended_displacements_meters[agent_id] = np.array([x_m_target - x_m_curr, y_m_target - y_m_curr])


        # 2. Get environmental current
        current_vec_m_per_step = self._get_current_vector_m_per_step() # This is [dx, dy] in meters for this step

        # 3. Calculate final positions in meters and then grid cells
        final_agent_positions_rc = {}
        for agent_id in self.agent_ids:
            r_curr, c_curr = self.agent_positions_rc[agent_id]
            x_m_curr = (c_curr + 0.5) * self.cell_size_meters
            y_m_curr = (r_curr + 0.5) * self.cell_size_meters

            dx_intent_m, dy_intent_m = intended_displacements_meters[agent_id]
            
            # Total displacement in meters
            dx_total_m = dx_intent_m + current_vec_m_per_step[0] 
            dy_total_m = dy_intent_m + current_vec_m_per_step[1]

            x_m_new = x_m_curr + dx_total_m
            y_m_new = y_m_curr + dy_total_m

            # Clip to domain boundaries (meter coordinates)
            domain_width_m = self.grid_size_c * self.cell_size_meters
            domain_height_m = self.grid_size_r * self.cell_size_meters
            x_m_new_clipped = np.clip(x_m_new, 0, domain_width_m - 1e-6) # Epsilon to avoid edge cases with int()
            y_m_new_clipped = np.clip(y_m_new, 0, domain_height_m - 1e-6)

            # Convert back to grid cell (integer truncation)
            c_final = int(x_m_new_clipped / self.cell_size_meters)
            r_final = int(y_m_new_clipped / self.cell_size_meters)
            final_agent_positions_rc[agent_id] = np.array([r_final, c_final])

        # 4. Collision check
        collision_detected = False
        final_occupied_cells = {} # { (r,c) : agent_id }
        for agent_id in self.agent_ids:
            r_f, c_f = final_agent_positions_rc[agent_id]
            if (r_f, c_f) in final_occupied_cells:
                collision_detected = True
                # print(f"Collision at ({r_f},{c_f}) between {agent_id} and {final_occupied_cells[(r_f,c_f)]}")
                break
            final_occupied_cells[(r_f, c_f)] = agent_id
        
        # 5. Update agent states if no collision
        if not collision_detected:
            for agent_id in self.agent_ids:
                self.agent_positions_rc[agent_id] = final_agent_positions_rc[agent_id]
                self.agent_headings[agent_id] = intended_new_headings[agent_id]
        
        # 6. Perform sensing, communication, update shared map
        self.current_env_step += 1 # Update step first for timestamps
        self._perform_sensing()
        self._perform_communication()
        self._update_shared_consensus_map()

        # 7. Calculate reward
        current_gt_grid = self._get_ground_truth_grid()
        iou_current = self._calculate_iou(self.shared_consensus_map, current_gt_grid)
        delta_iou = iou_current - self.iou_oil_previous_step
        reward_iou = delta_iou * self.reward_scaling_factor
        team_reward = reward_iou + self.penalty_per_step
        self.iou_oil_previous_step = iou_current

        # 8. Determine done
        done = collision_detected or (self.current_env_step >= self.max_steps_per_episode)
        
        # 9. Get next observations and global state
        next_obs_dict, next_global_state_entities = self._get_observations_and_state()

        rewards_dict = {agent_id: team_reward for agent_id in self.agent_ids}
        dones_dict = {agent_id: done for agent_id in self.agent_ids}
        # Add __all__ key for some MARL runners
        dones_dict["__all__"] = done 
        
        infos_dict = {agent_id: {
            'iou': iou_current, 
            'delta_iou': delta_iou,
            'collision': collision_detected,
            'current_vec_m_per_step': self._get_current_vector_m_per_step().tolist(),
            'current_env_step': self.current_env_step
            } for agent_id in self.agent_ids}

        return next_obs_dict, next_global_state_entities, rewards_dict, dones_dict, infos_dict

    def get_num_agents(self):
        return self.num_agents

    def get_action_space_size(self):
        return self.action_space_size

    def get_agent_ids(self):
        return self.agent_ids

    def get_observation_spec(self):
        # Provides the structure of observation entities for one agent
        # And the structure of global state entities
        # This is useful for initializing networks correctly.
        
        # Example: One agent's observation will be a list of entities.
        # Each entity has self._max_entity_features features.
        # The number of entities can vary (self, other agents, cnn, sensor).
        # Max possible entities: 1(self) + (N-1)(others) + 1(CNN) + 1(Sensor)
        max_obs_entities = 1 + (self.num_agents -1) + 1 + 1 
        
        # Global state: N_agents + 1_current + 1_global_belief (if enabled)
        max_global_state_entities = self.num_agents + 1 + (1 if self.config.get("INCLUDE_GLOBAL_BELIEF_IN_STATE", False) else 0)

        spec = {
            "agent_observation": {
                "type": "list_of_variable_length_entities",
                "entity_feature_dim": self._max_entity_features,
                "max_num_entities_approx": max_obs_entities,
                "entity_types_description": {
                    "self": f"norm_r, norm_c, one_hot_heading ({self.num_headings}), IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR + padding",
                    "other_agent": f"rel_norm_r, rel_norm_c, one_hot_heading ({self.num_headings}), IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR + padding",
                    "belief_map_cnn": f"cnn_features ({self.cnn_output_feature_dim}), IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR + padding",
                    "sensor": f"{self.direct_sensing_mode} data + IS_SELF, IS_AGENT, IS_MAP, IS_SENSOR + padding"
                }
            },
            "global_state": {
                "type": "list_of_variable_length_entities",
                "entity_feature_dim": self._max_global_state_entity_features,
                "max_num_entities_approx": max_global_state_entities,
                 "entity_types_description": {
                    "agent_in_global": f"norm_r, norm_c, one_hot_h, cnn_features ({self.cnn_output_feature_dim}), IS_AGENT, IS_ENV, IS_GLOBAL_BELIEF + padding",
                    "env_current": "norm_curr_x, norm_curr_y, IS_AGENT, IS_ENV, IS_GLOBAL_BELIEF + padding",
                    "global_belief_summary": "(optional) flattened_pooled_map, IS_AGENT, IS_ENV, IS_GLOBAL_BELIEF + padding"
                }
            },
            "belief_map_shape": (self.grid_size_r, self.grid_size_c, 1) # For CNN input
        }
        return spec

    def render(self, mode='human'):
        # Basic text render for now
        print(f"--- Timestep {self.current_env_step} ---")
        grid_render = np.full((self.grid_size_r, self.grid_size_c), '.', dtype=str)
        
        # Mark oil from shared consensus
        grid_render[self.shared_consensus_map == 1] = 'O' # Oil
        grid_render[self.shared_consensus_map == 0] = '~' # Clean
        
        # Mark agents
        heading_symbols = ['^', '↗', '>', '↘', 'v', '↙', '<', '↖'] # For 8 headings
        if self.num_headings == 4: heading_symbols = ['^', '>', 'v', '<']

        for i, agent_id in enumerate(self.agent_ids):
            r, c = self.agent_positions_rc[agent_id]
            h = self.agent_headings[agent_id]
            grid_render[r,c] = heading_symbols[h] if len(heading_symbols) > h else str(i)
        
        for r_idx in range(self.grid_size_r):
            print("".join(grid_render[r_idx, :]))
        
        print(f"Current IoU: {self.iou_oil_previous_step:.3f}")
        current_m_ps = self._get_current_vector_m_per_step() / self.env_time_step_hours
        print(f"Env Current (m/s): [{current_m_ps[0]:.2f}, {current_m_ps[1]:.2f}] (m/s from m/step)")


if __name__ == '__main__':
    # --- Example Usage & Test ---
    # 1. Create dummy episode data (simplified from generate_episodes.py)
    dummy_episode_dir = "dummy_episode_data_envtest"
    if not os.path.exists(dummy_episode_dir):
        os.makedirs(dummy_episode_dir)

    test_grid_r, test_grid_c = 10, 10
    test_steps = 50
    test_cell_m = 10.0

    dummy_gt_grids = np.random.randint(0, 2, size=(test_steps, test_grid_r, test_grid_c), dtype=np.uint8)
    dummy_currents = np.random.randn(test_steps, 2).astype(np.float32) * 0.1 # m/step
    
    gen_params = {'env_time_step_hours': 0.1} # Assume 0.1 hr per env step for dummy current conversion
    
    np.savez_compressed(
        os.path.join(dummy_episode_dir, "dummy_ep_00.npz"),
        ground_truth_grids=dummy_gt_grids,
        current_vectors_m_per_step=dummy_currents,
        simulation_config_details_json=json.dumps({"info":"dummy_sim_config"}),
        generation_params_json = json.dumps(gen_params)
    )
    print(f"Created dummy episode in {dummy_episode_dir}")

    # 2. Setup dummy hyperparameters for the environment
    hyperparams = {
        "GRID_SIZE_R": test_grid_r,
        "GRID_SIZE_C": test_grid_c,
        "CELL_SIZE_METERS": test_cell_m,
        "NUM_AGENTS": 2,
        "NUM_HEADINGS": 8,
        "OBSERVATION_RADIUS_AGENTS": 3,
        "COMMUNICATION_RADIUS_CELLS": 2,
        "DIRECT_SENSING_MODE": "surrounding_cells", # "current_cell"
        "CNN_OUTPUT_FEATURE_DIM": 32, # Smaller for test
        "REWARD_SCALING_FACTOR": 100.0,
        "PENALTY_PER_STEP": -0.01,
        "MAX_STEPS_PER_EPISODE": test_steps - 5, # Test early termination
        "MAX_EXPECTED_CURRENT_MPS": 1.0, # m/s
        "FALLBACK_ENV_TIME_STEP_HOURS": gen_params['env_time_step_hours'],
        "INCLUDE_GLOBAL_BELIEF_IN_STATE": True,
        "GLOBAL_BELIEF_POOLED_DIM": 2 # 10x10 -> 2x2
    }

    # 3. Initialize and run the environment
    env = OilSpillEnv(hyperparams, dummy_episode_dir)
    obs_spec = env.get_observation_spec()
    print("\nObservation Spec:", json.dumps(obs_spec, indent=2))
    
    for ep in range(2): # Run a couple of episodes
        print(f"\n--- Starting Episode {ep + 1} ---")
        obs_dict, global_state = env.reset()
        env.render()
        
        print("Initial Obs (agent_0 num entities):", len(obs_dict['agent_0']))
        if obs_dict['agent_0']:
            print("Initial Obs (agent_0, first entity shape):", obs_dict['agent_0'][0].shape)
        print("Initial Global State (num entities):", len(global_state))
        if global_state:
            print("Initial Global State (first entity shape):", global_state[0].shape)

        total_reward_ep = 0
        done = False
        step_count = 0
        while not done:
            actions = {agent_id: random.randint(0, env.get_action_space_size() - 1) 
                       for agent_id in env.get_agent_ids()}
            
            next_obs_dict, next_global_state, rewards_dict, dones_dict, infos_dict = env.step(actions)
            
            env.render()
            print(f"Step {env.current_env_step}: Actions: {actions}, Reward: {rewards_dict['agent_0']:.3f}, Done: {dones_dict['__all__']}")
            print(f"  Info (agent_0): {infos_dict['agent_0']}")

            obs_dict = next_obs_dict
            global_state = next_global_state
            total_reward_ep += rewards_dict['agent_0']
            done = dones_dict["__all__"]
            step_count +=1
            if step_count > hyperparams["MAX_STEPS_PER_EPISODE"] + 5 : # Safety break for test
                print("Test safety break")
                break
        print(f"Episode {ep+1} finished. Total reward: {total_reward_ep:.2f}, Steps: {env.current_env_step}")

    # Clean up dummy data
    os.remove(os.path.join(dummy_episode_dir, "dummy_ep_00.npz"))
    os.rmdir(dummy_episode_dir)
    print(f"\nCleaned up {dummy_episode_dir}")