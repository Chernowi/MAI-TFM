**MARL Framework Design for Oil Spill Response using TransfQMix**

**Report Version:** 1.6
**Date:** October 26, 2023

**1. Introduction**

This document outlines the design and architecture for a Multi-Agent Reinforcement Learning (MARL) framework aimed at solving a cooperative oil spill detection and mapping task. The core learning algorithm will be based on the principles of TransfQMix (refer to the provided TransfQMix paper), leveraging transformers for enhanced coordination. Agents will operate in a grid-world environment, attempting to collaboratively build an accurate map of an oil spill. This report details the environment, agent architecture, learning paradigm, data generation, supporting functionalities like logging, visualization, model management, and hyperparameter experimentation. The reader is expected to have access to the TransfQMix paper and the `simulation.py` script (for understanding underlying dynamics and data generation). This report aims to be sufficiently detailed for direct implementation.

**2. Environment Design**

**2.1. Environment Map & State:**
*   **Grid World:** A 2D grid of `GRID_SIZE_R` rows and `GRID_SIZE_C` columns. Each cell has dimensions `CELL_SIZE_METERS x CELL_SIZE_METERS`. These are configurable hyperparameters.
*   **Cell State (Ground Truth):** Each cell `(r,c)` in the ground truth map has a binary state: `1` (oil present) or `0` (no oil). This map, `ground_truth_grid_t`, is pre-generated for each episode and changes at each time step `t`.
*   **Agent State:** Each agent `a` maintains:
    *   Position: `(pos_r_a, pos_c_a)` representing the grid cell it occupies.
    *   Absolute Heading: An integer representing one of a fixed number of discrete directions (e.g., 8 directions: 0 for North, 1 for North-East, ..., 7 for North-West).
*   **Environmental Currents:** At each time step `t`, a global environmental current vector `(current_vel_x_t, current_vel_y_t)` (in meters per environment time step) is active. This data is pre-generated per episode.

**2.2. Agent Observations (`O_t^a`):**
At each time step `t`, agent `a` receives a local observation `O_t^a`, which is a list (or set) of entity feature vectors. Each feature vector describes an entity perceived by agent `a`.
    *   **Self-Entity:**
        *   Raw Features: `[pos_r_a, pos_c_a, heading_a]`
        *   Processed Features for Embedder: `[pos_r_a / GRID_SIZE_R, pos_c_a / GRID_SIZE_C, one_hot_encoded(heading_a), IS_SELF_FLAG=1, IS_AGENT_FLAG=1, IS_MAP_SUMMARY_FLAG=0, IS_SENSOR_FLAG=0]` (Flags help the embedder distinguish entity types). The number of elements in `one_hot_encoded(heading_a)` depends on the number of discrete heading directions.
    *   **Other Nearby Agent Entities:**
        *   For each other agent `b` within `OBSERVATION_RADIUS_AGENTS` (hyperparameter, in grid cells) of agent `a`.
        *   Raw Features: `[pos_r_b, pos_c_b, heading_b]`
        *   Processed Features for Embedder: `[(pos_r_b - pos_r_a) / (2*OBSERVATION_RADIUS_AGENTS), (pos_c_b - pos_c_a) / (2*OBSERVATION_RADIUS_AGENTS), one_hot_encoded(heading_b), IS_SELF_FLAG=0, IS_AGENT_FLAG=1, IS_MAP_SUMMARY_FLAG=0, IS_SENSOR_FLAG=0]` (Relative positions normalized by observation diameter).
    *   **Individual Belief Map CNN Feature Entity:**
        *   A fixed-size feature vector `F_cnn_a` (e.g., 128-dim) produced by agent `a`'s `BeliefMapCNN` processing its full `agent_a_belief_map`.
        *   Processed Features for Embedder: `[F_cnn_a (vector), IS_SELF_FLAG=0, IS_AGENT_FLAG=0, IS_MAP_SUMMARY_FLAG=1, IS_SENSOR_FLAG=0]`.
    *   **(Optional) Explicit Local Sensor Reading Entities:**
        *   If `DIRECT_SENSING_MODE` is `"surrounding_cells"`: One entity could represent the collective sensor readings.
            *   Raw Features: A vector of binary values `[sensed_current_cell, sensed_N, sensed_NE, ..., sensed_NW]` for the current cell and its 8 neighbors (9 values total).
            *   Processed Features for Embedder: `[sensed_current_cell, sensed_N, ..., sensed_NW, IS_SELF_FLAG=0, IS_AGENT_FLAG=0, IS_MAP_SUMMARY_FLAG=0, IS_SENSOR_FLAG=1]`.
        *   If `DIRECT_SENSING_MODE` is `"current_cell"`, this entity would represent only the `sensed_current_cell`.
        *   The length of these feature vectors must be consistent or padded for the shared Entity Embedder.

**2.3. Agent Individual Belief Map (`agent_a_belief_map`):**
*   **Structure:** Each agent `a` maintains its own **full individual belief map**, with dimensions `GRID_SIZE_R x GRID_SIZE_C`.
*   **Initialization:** At the start of an episode, all cells in `agent_a_belief_map` are initialized to `-1` (unknown), and all `timestamp_of_last_update` are initialized to `-1` or `0`.
*   **Cell Content:** Each cell `(r,c)` stores:
    1.  `belief_value`: `1` (believed oil), `0` (believed clean), `-1` (unknown).
    2.  `timestamp_of_last_update`: The environment time step `t` at which this belief was last updated.
*   **Direct Sensing Logic:**
    *   Controlled by hyperparameter `DIRECT_SENSING_MODE`:
        *   `"current_cell"`: Agent `a` at `(r_a, c_a)` gets the true state `S_true` of cell `(r_a, c_a)` from `ground_truth_grid_t`. It updates `agent_a_belief_map[r_a, c_a].belief_value = S_true` and `agent_a_belief_map[r_a, c_a].timestamp_of_last_update = current_env_step`.
        *   `"surrounding_cells"`: Agent `a` gets true states for `(r_a, c_a)` and its 8 Moore neighbors. For each of these 9 cells `(r_s, c_s)`, it updates `agent_a_belief_map[r_s, c_s].belief_value` with the true state and its timestamp with `current_env_step`.
*   **Communication Logic (Limited & Localized):**
    *   Hyperparameter: `COMMUNICATION_RADIUS_CELLS`.
    *   After the direct sensing phase within an environment step, for every pair of agents (`A`, `B`):
        *   If `distance(A, B) <= COMMUNICATION_RADIUS_CELLS` (e.g., Chebyshev distance on the grid):
            *   They exchange their full `agent_belief_map`s (both `belief_value` and `timestamp_of_last_update` for all cells).
            *   For each cell `(r,c)`:
                *   Agent `A` updates its `agent_A_belief_map[r,c]` if `agent_B_belief_map[r,c].timestamp > agent_A_belief_map[r,c].timestamp`. In this case, `A` adopts `B`'s belief value and timestamp for that cell.
                *   Symmetrically, agent `B` updates from `A`'s more recent or newly known information if `A`'s timestamp for the cell is greater.
            *   This ensures propagation of the latest information. Simultaneous observations are inherently consistent.
*   **CNN Input:** The `belief_value` part of `agent_a_belief_map` (a 2D grid of -1, 0, or 1) is fed into agent `a`'s `BeliefMapCNN`.

**2.4. Shared Consensus Map (for Reward Calculation & Visualization ONLY):**
*   **Structure:** `GRID_SIZE_R x GRID_SIZE_C`.
*   **Update:** At the end of each environment step (after sensing and communication), for each cell `(r,c)`:
    *   Initialize `SharedConsensusMap[r,c]` to "unknown" (e.g., -1).
    *   Iterate through all agents `a`. If `agent_a_belief_map[r,c].belief_value` is not "unknown":
        *   If `agent_a_belief_map[r,c].belief_value == 1` (oil), set `SharedConsensusMap[r,c] = 1`. Prioritize oil detection.
        *   Else if `agent_a_belief_map[r,c].belief_value == 0` (clean) and `SharedConsensusMap[r,c]` is still "unknown", set `SharedConsensusMap[r,c] = 0`.
*   This map reflects the team's combined knowledge, prioritizing oil detection. It is not directly used by agents for policy decisions.

**2.5. Agent Actions & Movement:**
*   **Action Space (6 discrete actions):** Each agent `a` maintains an absolute heading (e.g., 0=N, 1=NE, ..., 7=NW).
    1.  **STAY:** Agent intends to remain in its current cell. Current heading is maintained.
    2.  **MOVE_FORWARD:** Agent intends to move one cell in its current absolute heading. Current heading is maintained.
    3.  **MOVE_DIAG_LEFT:** Agent intends to move one cell in the direction -45° relative to its current absolute heading. The agent's absolute heading is updated to this new direction of movement.
    4.  **MOVE_DIAG_RIGHT:** Agent intends to move one cell in the direction +45° relative to its current absolute heading. The agent's absolute heading is updated to this new direction of movement.
    5.  **TURN_LEFT_90_AND_MOVE:** Agent's absolute heading is updated by -90° (to its left). It then intends to move one cell in this new absolute heading.
    6.  **TURN_RIGHT_90_AND_MOVE:** Agent's absolute heading is updated by +90° (to its right). It then intends to move one cell in this new absolute heading.
*   **Movement Execution Logic:**
    1.  Agent `a` selects an action `act_a`. Its current grid position is `(r_curr_a, c_curr_a)` and current heading `h_curr_a`.
    2.  Determine the *new intended absolute heading* `h_new_intent_a` based on `act_a` and `h_curr_a`. If `act_a` is `STAY` or `MOVE_FORWARD`, `h_new_intent_a = h_curr_a`.
    3.  Determine the *intended target grid cell* `(r_target_a, c_target_a)` based on `act_a` (if it involves movement) and `h_new_intent_a`. If `STAY`, target is `(r_curr_a, c_curr_a)`.
    4.  Convert current `(r_curr_a, c_curr_a)` and target `(r_target_a, c_target_a)` grid cells to meter coordinates (cell centers): `(x_m_curr, y_m_curr)` and `(x_m_target, y_m_target)`.
    5.  Calculate agent's *intended displacement vector* in meters: `(dx_intent, dy_intent) = (x_m_target - x_m_curr, y_m_target - y_m_curr)`.
    6.  Get the environmental current vector for current time step `t`: `(current_vel_x_t, current_vel_y_t)` in meters per time_step (from pre-loaded episode data).
    7.  Calculate *total displacement vector*: `(dx_total, dy_total) = (dx_intent + current_vel_x_t, dy_intent + current_vel_y_t)`.
    8.  Calculate agent's new potential position in meters: `(x_m_new, y_m_new) = (x_m_curr + dx_total, y_m_curr + dy_total)`.
    9.  Clip `(x_m_new, y_m_new)` to be within the domain boundaries (0 to `GRID_SIZE_C * CELL_SIZE_METERS - epsilon`, 0 to `GRID_SIZE_R * CELL_SIZE_METERS - epsilon`).
    10. Convert the clipped `(x_m_new_clipped, y_m_new_clipped)` back to a final grid cell `(r_final_a, c_final_a)` by integer truncation or rounding.
    11. Update agent's state: `agent_a.position = (r_final_a, c_final_a)` and `agent_a.heading = h_new_intent_a`.

**2.6. Reward Structure:**
*   **Team Reward:** All agents receive the same `team_reward_t`.
*   **Primary Reward Source (IoU Change):**
    *   `IoU_oil_t = TruePositives_oil_t / (TruePositives_oil_t + FalsePositives_oil_t + FalseNegatives_oil_t)` calculated using `SharedConsensusMap_t` vs. `ground_truth_grid_t`.
    *   `delta_IoU_t = IoU_oil_t - IoU_oil_{t-1}` (where `IoU_oil_0` is initialized to 0 or the IoU of the initial belief state).
    *   `reward_iou_t = delta_IoU_t * REWARD_SCALING_FACTOR`.
    *   **Default `REWARD_SCALING_FACTOR`:** `100.0`. Tunable hyperparameter.
*   **Auxiliary Rewards/Penalties:**
    *   **Time Penalty:** `PENALTY_PER_STEP` (Default: `-0.01`). Applied per agent per step. Tunable hyperparameter.
    *   `team_reward_t = reward_iou_t + PENALTY_PER_STEP`.
*   The `rewards_dict` returned by `env.step()` will have `rewards_dict[agent_id] = team_reward_t`.

**2.7. Episode Termination:**
*   An agent attempts to move into a cell that will be occupied by another agent at the end of the current step (collision). The collision check happens after all agents' final positions for the step are determined.
*   `current_env_step >= MAX_STEPS_PER_EPISODE` (hyperparameter).

**2.8. Global State for Mixer (`S_t`):**
A list (or set) of entity feature vectors provided to the mixer's transformer.
    *   **All Agent Entities:**
        *   For each agent `a`: Raw features `[pos_r_a, pos_c_a, heading_a, F_cnn_a (vector)]`.
        *   Processed features for embedder: `[pos_r_a / GRID_SIZE_R, pos_c_a / GRID_SIZE_C, one_hot_encoded(heading_a), F_cnn_a (vector properly scaled/normalized if needed), IS_AGENT_FLAG=1, IS_ENV_FLAG=0, ...]`. The `F_cnn_a` vector from the agent's belief map CNN is part of its state representation for the mixer.
    *   **Environmental Current Entity:**
        *   Raw features: `[current_vel_x_t, current_vel_y_t]` (meters/step).
        *   Processed features for embedder: `[normalized_current_vel_x_t, normalized_current_vel_y_t, IS_AGENT_FLAG=0, IS_ENV_FLAG=1, ...]`. Normalization based on expected max current speeds.
    *   **(Optional) Global Shared Belief Summary Entity:**
        *   Raw features: Flattened max-pooled `SharedConsensusMap_t`.
        *   Processed features for embedder: `[flattened_max_pooled_map_vector, IS_AGENT_FLAG=0, IS_ENV_FLAG=0, IS_GLOBAL_BELIEF_FLAG=1, ...]`.

**3. Agent Architecture (TransfQMix based)**

**3.1. Shared `BeliefMapCNN` Architecture:**
*   Input: Agent `a`'s full `agent_a_belief_map.belief_value` (a `GRID_SIZE_R x GRID_SIZE_C x 1` tensor). Values should be normalized if not already in a suitable range for CNNs (e.g., map -1,0,1 to 0,0.5,1 or use an initial embedding layer for discrete values if treated as categorical).
*   **Default Architecture (Example for a 64x64 input grid):**
    1.  Conv1: 16 filters, kernel size 5x5, stride 1, padding 2, ReLU activation. Output: (batch, 16, 64, 64).
    2.  MaxPool1: Kernel size 2x2, stride 2. Output: (batch, 16, 32, 32).
    3.  Conv2: 32 filters, kernel size 3x3, stride 1, padding 1, ReLU activation. Output: (batch, 32, 32, 32).
    4.  MaxPool2: Kernel size 2x2, stride 2. Output: (batch, 32, 16, 16).
    5.  Conv3: 64 filters, kernel size 3x3, stride 1, padding 1, ReLU activation. Output: (batch, 64, 16, 16).
    6.  MaxPool3: Kernel size 2x2, stride 2. Output: (batch, 64, 8, 8).
    7.  Flatten layer. Output: (batch, `8*8*64 = 4096`).
    8.  Dense1 (Output Layer): `nn.Linear(4096, CNN_OUTPUT_FEATURE_DIM)`. ReLU activation. (Default `CNN_OUTPUT_FEATURE_DIM`: `128`). Output: (batch, 128).
*   Output: `F_cnn_a` (a fixed-size feature vector, e.g., 128-dim).
*   Tunable hyperparameters: Number of convolutional/pooling layers, number of filters, kernel sizes, strides, padding strategies, activation functions, final `CNN_OUTPUT_FEATURE_DIM`.
*   All weights of this CNN architecture are shared across all agents.

**3.2. Shared Entity Embedder:**
*   A single `nn.Linear` layer that maps the concatenated raw (or pre-processed to a fixed length) features of an entity to `TRANSFORMER_EMBED_DIM`. The input dimension to this embedder must be fixed. This requires consistent feature vector length for all entity types (achieved by defining a superset of features and using type flags/padding where features are not applicable).

**3.3. Shared Agent Transformer Architecture:**
*   Input Sequence: `[h_{t-1}^a, entity_embedding_1, entity_embedding_2, ...]` where `h_{t-1}^a` is the agent's previous transformer hidden state (dimension `TRANSFORMER_EMBED_DIM`), and other embeddings are the `TRANSFORMER_EMBED_DIM`-dimensional outputs from the Entity Embedder for self, CNN map summary, other agents, etc.
*   **Default Hyperparameters:**
    *   `TRANSFORMER_EMBED_DIM` (d_model): Default: `64`. Tunable.
    *   `TRANSFORMER_NUM_HEADS` (for multi-head attention): Default: `4`. Tunable.
    *   `TRANSFORMER_NUM_BLOCKS` (number of transformer encoder layers): Default: `2`. Tunable.
    *   `TRANSFORMER_FFN_DIM_MULTIPLIER` (multiplier for hidden dim of feed-forward layers within blocks): Default: `4` (so FFN hidden dim = `EMBED_DIM * 4`). Tunable.
    *   `TRANSFORMER_DROPOUT_RATE`: Default: `0.1`. Tunable.
*   Output: `h_t^a` (new agent hidden state, dimension `TRANSFORMER_EMBED_DIM`).
*   All weights of this transformer architecture are shared across all agents.

**3.4. Shared Q-Value Head:**
*   A single `nn.Linear(TRANSFORMER_EMBED_DIM, NUM_ACTIONS)` where `NUM_ACTIONS` is 6.
*   Weights are shared across all agents.

**3.5. Action Selection:** Epsilon-greedy based on the output Q-values. Epsilon decay schedule is a tunable hyperparameter.

**4. Mixer Architecture (TransfQMix based)**

*   The central mixer adheres to the TransfQMix paper's design: a transformer processes global state entities (including agent positions and their `F_cnn_a` features) and all agent hidden states `h_t^a`. The outputs of this mixer transformer parameterize a monotonic Multi-Layer Perceptron (MLP) which mixes the individual agent Q-values.
*   Hyperparameters for the mixer's transformer (e.g., `MIXER_TRANSFORMER_EMBED_DIM`, `MIXER_TRANSFORMER_NUM_HEADS`, etc.) can be set independently or mirror agent transformer defaults. **Default: Mirror agent transformer defaults.** These are tunable.
*   The MLP part of the mixer will also have tunable hyperparameters (e.g., number of hidden layers, units per layer).

**5. Training Procedure**

*   **CTDE (Centralized Training, Decentralized Execution) paradigm.**
*   **Replay Buffer:** Stores transitions: `(O_t_all_agents, S_t, U_t_joint, team_R_t, O_{t+1}_all_agents, S_{t+1}, D_t, h_t_all_agents, h_{t+1}_all_agents)`.
    *   `O_t_all_agents`: A list/tuple where each element `O_t^a` is the list of entity feature vectors for agent `a`.
    *   `S_t`: The global state, as a list of entity feature vectors.
    *   `U_t_joint`: A tuple of actions `(u_t^1, ..., u_t^N)`.
    *   `team_R_t`: The scalar team reward.
    *   `D_t`: The boolean done flag (same for all agents).
    *   `h_t_all_agents`: A list/tuple of the `h_t^a` vectors (agent transformer hidden states) from each agent at time `t`. `h_{t+1}` for the next states. The `h_t` used to generate `Q_a(O_t^a, u_t^a)` is `h_t^a` itself, while `h_{t-1}^a` was the input to the agent transformer to produce `h_t^a`. The buffer needs to store the hidden states that were *inputs* to the agent transformers for `O_t` and `O_{t+1}` to ensure correct gradient flow for the recurrent part.
*   **Learning Step (Simplified Flow):**
    1.  Sample a batch of transitions from the replay buffer.
    2.  For each agent `a` in the batch and for the current state `O_t^a` (using `h_{t-1}^a` from buffer):
        *   Pass `O_t^a` and `h_{t-1}^a` through agent `a`'s policy network (CNN -> Embedder -> Transformer -> Q-head) to get all Q-values and the current hidden state `h_t^a`.
        *   Extract `Q_a(O_t^a, u_t^a)` (the Q-value for the action `u_t^a` actually taken, from buffer).
    3.  Feed all `Q_a(O_t^a, u_t^a)` values, `S_t`, and all `h_t^a` (the *output* hidden states from step 2) into the policy mixer network to get `Q_tot`.
    4.  For next states `O_{t+1}^a` (using `h_t^a` from buffer as input hidden state `h'_{t}` for next step calculation):
        *   Pass `O_{t+1}^a` and `h_t^a` through agent `a`'s *target* network to get Q-values for all next actions and the next hidden state `h_{t+1}^a`.
        *   Select greedy actions `u'_{t+1}^a` based on these target Q-values.
        *   Extract `Q_a_target(O_{t+1}^a, u'_{t+1}^a)`.
    5.  Feed all `Q_a_target(O_{t+1}^a, u'_{t+1}^a)`, `S_{t+1}`, and all `h_{t+1}^a` (output hidden states from step 4) into the *target* mixer network to get `Q_tot_target`.
    6.  Calculate the Bellman target: `Y = team_R_t + \gamma * (1-D_t) * Q_tot_target`. (`gamma` is discount factor).
    7.  Calculate loss: `L = HuberLoss(Q_tot, Y)`.
    8.  Perform backpropagation of `L` through all policy networks (mixer, agent Q-heads, agent transformers, entity embedders, and agent `BeliefMapCNN`s).
    9.  Update policy network parameters using an optimizer (e.g., Adam).
    10. Periodically update target network weights (e.g., soft Polyak averaging or hard copy).
*   **Parameter Sharing:** As stated, `BeliefMapCNN`, `EntityEmbedder`, `AgentTransformer`, and `Q-Value Head` weights are shared among all agents. Each agent instance maintains its own recurrent hidden state `h_t^a`.

**6. Framework Functionality & API**

**6.1. Environment API:**
*   **Class `OilSpillEnv`:**
    *   `__init__(self, experiment_hyperparams, episode_data_directory, specific_episode_file=None)`:
        *   `experiment_hyperparams`: A dictionary containing all tunable parameters like `GRID_SIZE_R`, `GRID_SIZE_C`, `CELL_SIZE_METERS`, `OBSERVATION_RADIUS_AGENTS`, `COMMUNICATION_RADIUS_CELLS`, `DIRECT_SENSING_MODE`, `MAX_STEPS_PER_EPISODE`, etc.
        *   `episode_data_directory`: Path to the folder containing `.npz` episode files.
        *   `specific_episode_file`: Optional. If provided, only this episode is loaded for sequential runs (e.g., for evaluation).
    *   `reset(self)`:
        *   Randomly selects an `.npz` file from `episode_data_directory` (if `specific_episode_file` is None) or loads the specific one.
        *   Loads ground truth grids and current vectors for the episode.
        *   Resets agent positions, headings, individual belief maps (all cells to "unknown", timestamps to -1).
        *   Resets `current_env_step = 0`.
        *   Calculates initial agent observations `O_0^a` for all agents and initial global state `S_0`.
        *   Returns `(initial_observations_dict, initial_global_state_entities)`.
            *   `initial_observations_dict`: `{agent_id: O_0^a_entity_list}`.
            *   `initial_global_state_entities`: List of global state entity feature vectors.
    *   `step(self, actions_dict)`:
        *   `actions_dict`: `{agent_id: action_id}`.
        *   Executes agent movements considering actions and environmental currents.
        *   Updates agent headings if actions involve turns.
        *   Performs collision checks.
        *   Agents perform direct sensing based on `DIRECT_SENSING_MODE`, updating their `agent_a_belief_map`.
        *   Agents perform communication, updating their `agent_a_belief_map`.
        *   Constructs the `SharedConsensusMap`.
        *   Calculates team reward based on IoU change.
        *   Determines if the episode is done (collision, max steps).
        *   Generates next observations `O_{t+1}^a` for all agents and next global state `S_{t+1}`.
        *   Increments `current_env_step`.
        *   Returns `(next_observations_dict, next_global_state_entities, rewards_dict, dones_dict, infos_dict)`.
            *   `rewards_dict`: `{agent_id: team_reward_t}`.
            *   `dones_dict`: `{agent_id: boolean_done_flag}`.
            *   `infos_dict`: `{agent_id: {'iou': current_iou, 'current_vec': current_step_current_vector, ...}}`.
    *   `get_num_agents(self)`
    *   `get_action_space_size(self)`
    *   `get_observation_spec(self)` (details structure of entities and feature vector lengths).

**6.2. Logging:**
*   **Terminal/File Logging:** Use Python's built-in `logging` module.
    *   Configure with different log levels (DEBUG, INFO, WARNING, ERROR).
    *   Log experiment hyperparameters at the start of a run.
    *   Log training progress (e.g., every N episodes: episode number, average reward over last M episodes, average IoU, current loss, epsilon).
    *   Log evaluation results.
    *   Log significant events or errors.
*   **TensorBoard Compatibility:** Use `torch.utils.tensorboard.SummaryWriter`.
    *   Create a unique log directory for each experiment run (e.g., `runs/experiment_name_timestamp/`).
    *   Log scalars: `training/episode_reward`, `training/avg_iou_per_step`, `training/loss`, `training/epsilon`, `training/episode_length`. During evaluation: `evaluation/avg_episode_reward`, `evaluation/final_iou`.
    *   Log hyperparameters: `writer.add_hparams(hyperparam_dict, metric_dict)` to track performance against hyperparameter settings.
    *   (Optional) `writer.add_graph()` for model architecture, `writer.add_histogram()` for weights/biases/gradients.

**6.3. Visualization Tool:**
*   **Class `EpisodeVisualizer`:**
    *   `__init__(self, grid_size_r, grid_size_c, num_agents, cell_size_m)`
    *   `start_episode_recording(self, episode_number, output_gif_path_template)`: Initializes frame list for a new GIF.
    *   `add_frame(self, ground_truth_grid, shared_consensus_map, agent_positions_list, agent_headings_list, current_vector, timestep_info_string)`:
        *   Uses `matplotlib.pyplot` to generate a plot showing:
            *   Ground truth oil spill.
            *   Shared consensus map.
            *   Agent positions (e.g., colored circles).
            *   Agent headings (e.g., small arrows or orientation of agent markers).
            *   Environmental current vector (e.g., a global arrow).
            *   Timestep information text.
        *   Saves the plot to an in-memory buffer (e.g., `io.BytesIO`).
        *   Appends the image data from the buffer to an internal list of frames.
    *   `save_recording(self)`: Compiles the list of frames into a GIF using `imageio.mimsave()`.
    *   `close(self)`: Closes matplotlib figures.
*   **Conditional Imports:**
    ```python
    try:
        import matplotlib.pyplot as plt
        import imageio
        VISUALIZATION_ENABLED = True
    except ImportError:
        VISUALIZATION_ENABLED = False
        # Log a warning that visualization is disabled
    ```
*   Visualization calls within the training/evaluation loop will be guarded by `if VISUALIZATION_ENABLED and config.ENABLE_VISUALIZATION_FLAG:`.

**6.4. Model Saving and Loading:**
*   **Directory Structure:** `saved_models/experiment_name_or_id/`.
*   **Components to Save (State Dictionaries):**
    1.  Shared `BeliefMapCNN`.
    2.  Shared `EntityEmbedder`.
    3.  Shared `AgentTransformer` (encoder part).
    4.  Shared `QValueHead`.
    5.  Policy `Mixer` network.
    6.  Target networks for all the above.
    7.  (Optional) Optimizer state dictionaries.
*   **Saving Function:** A utility function `save_checkpoint(state, experiment_dir, filename_prefix, episode, metric_value)` will save all necessary components. Filenames will include prefix, episode, and metric (e.g., `iou0.75_ep5000_cnn.pth`).
*   **Loading Function:** A utility function `load_checkpoint(experiment_dir, filename_prefix, map_location)` will load weights into newly instantiated network objects. This will be used for resuming training, evaluation, or inference.
*   Logic to save the model achieving the best evaluation metric (e.g., highest average IoU over N evaluation episodes).

**6.5. Hyperparameter Experimentation Mechanism:**
*   **Configuration Files (JSON or YAML):** Each experiment run will be driven by a configuration file. This file will specify values for all tunable hyperparameters listed in previous sections (Environment, Rewards, CNN, Transformer, Training, Optimizer).
    ```yaml
    # Example experiment_config.yaml
    experiment_name: "TransfQMix_CNN_LargeBuffer_HighLR"
    # Environment params
    GRID_SIZE_R: 64
    MAX_STEPS_PER_EPISODE: 400
    COMMUNICATION_RADIUS_CELLS: 5
    DIRECT_SENSING_MODE: "surrounding_cells"
    # Reward params
    REWARD_SCALING_FACTOR: 100.0
    PENALTY_PER_STEP: -0.01
    # CNN params
    CNN_OUTPUT_FEATURE_DIM: 128
    CNN_FILTERS: [16, 32, 64] 
    # ... other CNN params (kernels, strides)
    # Transformer params (Agent & Mixer can have separate sections if needed)
    AGENT_TRANSFORMER_EMBED_DIM: 64
    AGENT_TRANSFORMER_NUM_HEADS: 4
    # ... other transformer params
    # Training params
    LEARNING_RATE: 0.0005
    BATCH_SIZE: 32
    # ...
    ```
*   **Experiment Runner Script (`run_experiment.py`):**
    *   Parses command-line arguments (e.g., `--config path/to/experiment_config.yaml`).
    *   Loads hyperparameters from the specified config file.
    *   Initializes all framework components (Environment, Agent policies, Mixer, Replay Buffer, Loggers, Visualizer) using these loaded hyperparameters.
    *   Executes the main training loop.
    *   All outputs (TensorBoard logs, saved models, generated GIFs, text logs) for this run are saved into a unique directory, e.g., `results/experiment_name_or_timestamp/`.
*   **Comparison:** TensorBoard's HParams dashboard can be used to compare metrics across different experiment runs by logging hyperparameters. Manual comparison of log files and output plots is also possible due to structured output directories.
*   **Automated Sweeps (Future Extension):** For systematic hyperparameter optimization, tools like Weights & Biases Sweeps, Optuna, or Ray Tune can be integrated to wrap the `run_experiment.py` script and manage multiple runs with different hyperparameter sets.

**7. Environment Data Generation (Offline)**

**7.1. Simulation Script (`generate_episodes.py`):**
*   This script will utilize the (refactored) `OilSpillSimulatorCore` class derived from `simulation.py`.
*   **Command Line Arguments:**
    *   `--num_episodes <int>`: Number of distinct episode scenarios to generate.
    *   `--steps_per_episode <int>`: Maximum number of simulation time steps for each generated episode.
    *   `--grid_size_r <int>`: Number of rows for the discretized environment grid.
    *   `--grid_size_c <int>`: Number of columns for the discretized environment grid.
    *   `--output_dir <path>` (default: `episode_grids`): Directory to save the generated `.npz` files.
    *   `--sim_config_file <path>`: Path to the JSON configuration file for the `OilSpillSimulatorCore` (controlling spill dynamics, currents, etc.). This allows generating varied scenarios.
    *   `--cell_size_meters <float>`: The size of each grid cell in meters, needed for discretization.
*   **Process for Each Episode to Generate:**
    1.  Create an instance of `OilSpillSimulatorCore`, initializing it with parameters from `sim_config_file`.
    2.  Create empty lists/arrays to store `ground_truth_grids_for_episode` and `current_vectors_for_episode`.
    3.  Loop for `steps_per_episode`:
        a.  Call `simulator_core.step()`.
        b.  Get active particle positions: `particle_pos_meters = simulator_core.get_active_particle_positions()`.
        c.  Get current environmental current vector: `current_vec_meters_per_hour = simulator_core.get_current_environmental_current()`. Convert this to meters per environment time step based on the environment's `TIME_STEP_HOURS`.
        d.  Create an empty binary grid of `grid_size_r x grid_size_c`.
        e.  For each particle in `particle_pos_meters`:
            i.  Convert its meter position `(x_m, y_m)` to grid cell `(r, c)` using `CELL_SIZE_METERS`.
            ii. If `(r, c)` is within grid bounds, mark `binary_grid[r, c] = 1`.
        f.  Append the `binary_grid` to `ground_truth_grids_for_episode`.
        g.  Append the converted `current_vector_meters_per_step` to `current_vectors_for_episode`.
    4.  After the loop, save the collected data.
*   **Output Format:**
    *   One `.npz` file per generated episode.
    *   Naming convention: `episode_SIMCFG_<sim_config_filename_no_ext>_R<rows>_C<cols>_S<steps>_ID<unique_id>.npz`.
    *   `.npz` File Contents:
        *   `ground_truth_grids`: A NumPy array of shape (`steps_per_episode`, `grid_size_r`, `grid_size_c`) containing the binary oil maps. Stored as `uint8` for efficiency.
        *   `current_vectors`: A NumPy array of shape (`steps_per_episode`, `2`) containing the `(current_x_mps, current_y_mps)` in meters per environment step. Stored as `float32`.
        *   `simulation_config_details_json`: A string containing the JSON content of the `sim_config_file` used for generating this episode, for full reproducibility of the scenario.
        *   `generation_params`: A dictionary storing `{'steps_per_episode': val, 'grid_size_r': val, ...}` used for this specific generation.

**7.2. Loading Episode Data in `OilSpillEnv`:**
*   The `OilSpillEnv.__init__` method will scan the `episode_data_directory` for matching `.npz` files.
*   During `env.reset()`:
    *   It selects an episode (either randomly or sequentially if iterating through a dataset for evaluation).
    *   It loads the `ground_truth_grids` and `current_vectors` arrays from the chosen `.npz` file.
    *   The `current_env_step` is reset to 0.
*   During `env.step()`:
    *   The environment provides `ground_truth_grids[current_env_step]` as the current true oil map.
    *   It uses `current_vectors[current_env_step]` for affecting agent movement.

**8. Refactoring `simulation.py` (Crucial Prerequisite):**
    The existing `simulation.py` script relies on global variables. For the `generate_episodes.py` script to function correctly (especially when generating multiple distinct episodes with potentially different simulation parameters), `simulation.py` **must** be refactored into a class-based structure, for example, `OilSpillSimulatorCore`.
    *   **Class `OilSpillSimulatorCore`:**
        *   `__init__(self, sim_config_filepath)`: Loads JSON config, initializes all internal state variables (particles, current schedules, domain, time, etc.) as instance attributes.
        *   `reset(self)`: Re-initializes the simulation to its starting state based on the loaded config (useful if running multiple simulations with the same instance).
        *   `step(self)`: Advances the simulation by one internal time step, updating particle positions, environmental conditions according to its schedule, and internal simulation time.
        *   `get_active_particle_positions(self)`: Returns a list of `[x,y]` meter coordinates for active particles.
        *   `get_current_environmental_current(self)`: Returns the `(vx, vy)` current vector in meters/hour (or per internal simulation time step if more convenient) for the current simulation time.
        *   `is_finished(self)`: Returns true if simulation duration is reached.
        *   `get_current_sim_time_hours(self)`.
    This encapsulation is vital for creating independent simulation instances for each episode generation run.
