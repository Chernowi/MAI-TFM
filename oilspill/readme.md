# MARL Framework for Oil Spill Response using TransfQMix

This project implements a Multi-Agent Reinforcement Learning (MARL) framework to train cooperative agents for an oil spill detection and mapping task. The core learning algorithm is based on TransfQMix, leveraging Transformers for enhanced coordination among agents. Agents operate in a 2D grid-world, sensing the environment, communicating locally, and collaboratively building a belief map of an oil spill.

## Table of Contents

- [MARL Framework for Oil Spill Response using TransfQMix](#marl-framework-for-oil-spill-response-using-transfqmix)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
  - [Data Generation](#data-generation)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Evaluation and Visualization](#evaluation-and-visualization)
  - [Core Components](#core-components)
    - [Environment (`OilSpillEnv`)](#environment-oilspillenv)
    - [Agent Network (`TransfQMixAgentNN`)](#agent-network-transfqmixagentnn)
    - [Mixer Network (`TransfQMixMixer`)](#mixer-network-transfqmixmixer)
    - [Replay Buffer](#replay-buffer)
  - [Future Work and Potential Improvements](#future-work-and-potential-improvements)

## Features

-   **TransfQMix Implementation:** Utilizes Transformer-based architectures for both individual agent policies and the central mixing network.
-   **Cooperative Task:** Agents learn to collaboratively map an evolving oil spill.
-   **Grid-World Environment:** A configurable 2D grid where oil spill dynamics are simulated.
-   **Agent Perception:** Agents have local observations including self-state, nearby agents, and a summary of their individual belief map processed by a CNN.
-   **Individual Belief Maps:** Each agent maintains and updates its own map of the spill.
-   **Localized Communication:** Agents can exchange belief map information with nearby peers.
-   **Environmental Dynamics:** Simulated environmental currents affect agent movement.
-   **CTDE Paradigm:** Centralized Training with Decentralized Execution.
-   **Offline Data Generation:** Oil spill scenarios (ground truth maps and current data) are pre-generated using a modified version of the `simulation.py` script.
-   **Configurable Experiments:** Hyperparameters for the environment, networks, and training are managed via YAML configuration files.
-   **Logging & Visualization:**
    -   Comprehensive logging to console and file.
    -   TensorBoard integration for tracking metrics and hyperparameters.
    -   Episode visualization (GIF generation) during evaluation.
-   **Model Management:** Saving and loading of model checkpoints.

## Project Structure

```
marl_framework/
├── agents/                 # Agent neural network architectures (TransfQMixAgentNN)
│   ├── __init__.py
│   └── transfqmix_agent.py
├── configs/                # Experiment configuration files (YAML)
│   └── default_exp_config.yaml
├── data_generation/        # Scripts and core logic for generating episode data
│   ├── __init__.py
│   ├── generate_episodes.py  # Main script to generate .npz episode files
│   └── simulation_core.py    # Refactored oil spill simulator
├── environments/           # MARL environment definition
│   ├── __init__.py
│   └── oil_spill_env.py
├── mixers/                 # Mixer neural network architecture (TransfQMixMixer)
│   ├── __init__.py
│   └── transfqmix_mixer.py
├── replay_buffer/          # Replay buffer implementation
│   ├── __init__.py
│   └── buffer.py
├── utils/                  # Utility functions for logging, model I/O, visualization
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── model_io_utils.py
│   └── visualization_utils.py
├── main_train.py           # Main script to run training experiments
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure you have PyTorch installed according to your CUDA version if you plan to use a GPU. Visit [pytorch.org](https://pytorch.org/) for instructions.

## Data Generation

Before training, you need to generate episode data. This data consists of sequences of ground truth oil spill maps and corresponding environmental current vectors.

1.  **Prepare a Simulation Configuration:**
    Create a JSON configuration file (e.g., `configs/sim_configs/base_spill_config.json`) for the `OilSpillSimulatorCore`. This file controls the dynamics of the oil spill (initial spill, continuous sources, current changes, wind, etc.). An example based on the original `simulation.py` defaults should be created.
    *(Self-correction: The `simulation_core.py` example includes a way to generate a dummy config if one isn't provided, and `generate_episodes.py` can use it.)*

2.  **Run the generation script:**
    ```bash
    python -m marl_framework.data_generation.generate_episodes \
        --num_episodes 100 \
        --steps_per_episode 200 \
        --grid_size_r 32 \
        --grid_size_c 32 \
        --output_dir episode_data \
        --sim_config_file path/to/your/sim_config.json \
        --cell_size_meters 50.0
    ```
    -   Adjust parameters as needed.
    -   The `--output_dir` (default: `episode_data`) should match the `episode_data_directory` in your experiment configuration YAML.
    -   If you use the `dummy_sim_core_config.json` name for `--sim_config_file` and it doesn't exist, the script will generate a basic one for you.

    This will create `.npz` files in the specified output directory, each containing data for one episode.

## Configuration

Experiments are controlled by YAML configuration files located in the `configs/` directory. `configs/default_exp_config.yaml` provides a template with all available parameters.

**Key sections in the configuration file:**
-   `experiment_name`: Used for naming output directories.
-   `environment`: Parameters for `OilSpillEnv` (grid size, number of agents, observation/communication radii, episode length, etc.).
-   `rewards`: Reward scaling factors and penalties.
-   `agent_nn`: Hyperparameters for the shared `TransfQMixAgentNN` (CNN output dimension, Transformer embedding dimension, heads, blocks).
-   `mixer_nn`: Hyperparameters for the `TransfQMixMixer`.
-   `training`: Training loop parameters (batch size, learning rate, gamma, epsilon schedule, target update frequency, buffer capacity).
-   `logging`: Directories for logs, TensorBoard runs, saved models, and visualization settings.

To run an experiment with a specific configuration:
```bash
python main_train.py --config configs/your_experiment_config.yaml
```

## Training

The main training script `main_train.py` orchestrates the training process:

```bash
python main_train.py --config configs/default_exp_config.yaml
```

-   The script will:
    -   Initialize the environment, agent/mixer networks, optimizer, and replay buffer based on the specified configuration.
    -   Set up logging (console, file) and a TensorBoard writer.
    -   Run the main training loop:
        -   Agents interact with the environment.
        -   Transitions (observations, actions, rewards, hidden states, belief maps) are stored in the replay buffer.
        -   Periodically, batches are sampled from the buffer to update network weights.
    -   Log training progress to the console and TensorBoard.
    -   Periodically evaluate the model on separate episodes.
    -   Save model checkpoints (periodically and the best model based on evaluation metrics).

**Monitoring Training:**
-   **Console Output:** Provides information on episode rewards, IoU, epsilon, etc.
-   **Log Files:** Detailed logs are saved in `logs/<experiment_name>/<experiment_name_timestamp>/experiment.log`.
-   **TensorBoard:** Launch TensorBoard to visualize metrics:
    ```bash
    tensorboard --logdir runs
    ```
    Navigate to `http://localhost:6006` in your browser. Metrics will be under `runs/<experiment_name>/<experiment_name_timestamp>`.

## Evaluation and Visualization

-   **Evaluation:** During training, the model is periodically evaluated on a set of episodes (defined by `num_evaluation_episodes`). Average reward and IoU are reported.
-   **Visualization:** If `logging:visualization_enabled` is `true` in the config, GIFs of evaluation episodes can be generated.
    -   GIFs are saved in `episode_gifs/<experiment_name>/<run_timestamp_subdir>/`.
    -   The frequency is controlled by `visualization_interval_eval_episodes`.

To run a saved model in evaluation-only mode, you would typically need a separate script that loads the checkpoint and runs `eval_env.step()` without training updates. (This script is not explicitly provided in the current generation but would be a straightforward extension.)

## Core Components

### Environment (`OilSpillEnv`)

-   Located in `environments/oil_spill_env.py`.
-   Manages the grid world, agent states (position, heading, individual belief maps), and oil spill ground truth (loaded from pre-generated episode data).
-   Handles agent actions, movement (including current effects), collisions, direct sensing, and local communication.
-   Calculates team rewards based on the change in Intersection over Union (IoU) of the team's shared consensus map against the ground truth.
-   Provides observations (`O_t^a`) as a list of entity feature vectors for each agent and a global state (`S_t`) for the mixer.

### Agent Network (`TransfQMixAgentNN`)

-   Located in `agents/transfqmix_agent.py`.
-   Shared network for all agents.
-   Comprises:
    -   `BeliefMapCNN`: Processes the agent's individual belief map into a feature vector (`F_cnn_a`).
    -   `EntityEmbedder`: Embeds raw features of observed entities (self, other agents, `F_cnn_a`, sensor readings) into a common dimension.
    -   `AgentTransformer`: Processes the sequence of embedded entities (including the agent's previous hidden state `h_{t-1}^a`) to produce a new hidden state `h_t^a`.
    -   `QValueHead`: Maps `h_t^a` to Q-values for each action.

### Mixer Network (`TransfQMixMixer`)

-   Located in `mixers/transfqmix_mixer.py`.
-   Centralized network used during training.
-   Comprises:
    -   `EntityEmbedder`: Embeds entities from the global state `S_t`.
    -   `MixerTransformer`: Processes a sequence of projected agent hidden states (`h_t^a`) and embedded global state entities.
    -   **Hypernetwork:** Linear layers that take the `MixerTransformer`'s output to generate the weights and biases for a monotonic Multi-Layer Perceptron (MLP).
    -   **Monotonic MLP:** Mixes the individual agent Q-values (for actions taken) into a total team Q-value (`Q_tot`), ensuring that `dQ_tot / dQ_a >= 0`.

### Replay Buffer

-   Located in `replay_buffer/buffer.py`.
-   Stores transitions experienced by the agents:
    -   Agent observations (list of entity lists)
    -   Agent belief maps (current and next)
    -   Global state entities (current and next)
    -   Joint actions
    -   Team reward
    -   Done flags
    -   Agent hidden states (`h_in` for the current step, `h_out` from the current step).
-   Provides a `sample()` method to retrieve batches of transitions for training.

## Future Work and Potential Improvements

-   **More Sophisticated Global State for Mixer:** The `_prepare_global_state_for_mixer` helper in `main_train.py` could be made more robust by using explicit flags or entity types within the global state list to identify and update agent-specific `F_cnn` features, rather than relying on positional assumptions.
-   **Adaptive CNN:** Use `nn.AdaptiveAvgPool2d` in `BeliefMapCNN` to handle varying input grid sizes more gracefully.
-   **Attention Analysis:** Visualize attention weights in the agent and mixer transformers to understand what entities the models are focusing on.
-   **Advanced Exploration Strategies:** Implement more sophisticated exploration techniques beyond epsilon-greedy (e.g., prioritized experience replay, noisy networks).
-   **Curriculum Learning:** Start with simpler scenarios (smaller grids, fewer agents, less dynamic spills) and gradually increase complexity.
-   **Parameter Sharing Schemes:** Experiment with different levels of parameter sharing (e.g., partially shared layers).
-   **Hyperparameter Optimization:** Use tools like Optuna, Ray Tune, or Weights & Biases Sweeps for systematic hyperparameter tuning.
-   **Decentralized Execution Evaluation Script:** A dedicated script to load a trained model and run it in a purely decentralized manner for evaluation and demonstration.
-   **More Complex Communication:** Explore learnable communication protocols or different types of information exchange.
-   **Heterogeneous Agents:** Extend to support agents with different capabilities or observation spaces.
