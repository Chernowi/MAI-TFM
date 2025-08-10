# MARL Framework for Oil Spill Response using TransfQMix

This project provides a Multi-Agent Reinforcement Learning (MARL) framework for training cooperative agents to detect and map oil spills. The learning algorithm is based on **TransfQMix**, which uses Transformers to enhance coordination between agents. Agents operate in a 2D grid-world, sensing the environment, communicating locally, and collaboratively building a belief map of an oil spill.

Most of the project's functionality is located inside the `marl_framework` folder.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Data Generation](#data-generation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Results](#results)
- [Future Work](#future-work)

## Getting Started

### 1. Prerequisites

- Python 3.8+
- A Python virtual environment (recommended)
- PyTorch (see [pytorch.org](https://pytorch.org/) for installation instructions)

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Generate Episode Data

Before training, you need to generate the oil spill scenarios.

```bash
python -m marl_framework.data_generation.generate_episodes --num_episodes 100
```

This will create 100 episode files in the `episode_data/` directory.

### 4. Train the Model

Start the training process using the default configuration:

```bash
python main_train.py
```

### 5. Evaluate and Visualize

After training, you can evaluate the model and generate GIFs of the agents' behavior.

```bash
python evaluate_and_visualize.py --config <path-to-your-config> --checkpoint <path-to-your-model>
```

## Project Structure

```
oilspill/
├── marl_framework/         # Core MARL framework
│   ├── agents/             # Agent neural network architectures
│   ├── configs/            # Experiment configuration files
│   ├── data_generation/    # Scripts for generating episode data
│   ├── environments/       # MARL environment definition
│   ├── mixers/             # Mixer neural network architecture
│   ├── replay_buffer/      # Replay buffer implementation
│   └── utils/              # Utility functions
├── main_train.py           # Main script to run training experiments
├── evaluate_and_visualize.py # Script to evaluate and visualize trained models
├── requirements.txt        # Python dependencies
└── readme.md               # This file
```

## Data Generation

The `generate_episodes.py` script creates the dataset for training and evaluation. You can customize the data generation process by modifying the script's arguments.

```bash
python -m marl_framework.data_generation.generate_episodes \
    --num_episodes 100 \
    --steps_per_episode 200 \
    --grid_size_r 32 \
    --grid_size_c 32 \
    --output_dir episode_data
```

## Configuration

Experiments are configured using YAML files in the `marl_framework/configs/` directory. The `default_exp_config.yaml` file contains all available parameters.

To run an experiment with a specific configuration, use the `--config` flag:

```bash
python main_train.py --config marl_framework/configs/your_experiment_config.yaml
```

## Training

The `main_train.py` script handles the training process. It initializes the environment, networks, and replay buffer, then runs the main training loop.

- **Monitoring:** Track progress using the console output, log files (`logs/`), and TensorBoard (`runs/`).
- **Checkpoints:** Models are saved in the `saved_models/` directory.

To resume training from a checkpoint, use the `--load_checkpoint_path` flag:

```bash
python main_train.py --load_checkpoint_path <path-to-your-model>
```

## Evaluation and Visualization

The `evaluate_and_visualize.py` script is used to evaluate a trained model and generate GIFs of the episodes.

```bash
python evaluate_and_visualize.py \
    --config <path-to-your-config> \
    --checkpoint <path-to-your-model> \
    --episodes 5 \
    --output_dir visualizations
```

## Results

Here are some examples of the agents' behavior during evaluation:

*(Add your own GIFs here)*

![Evaluation Episode 1](visualizations/transfqmix_long_explore_v1_20250709-152956/eval_ep200_sub0.gif)
![Evaluation Episode 2](visualizations/transfqmix_long_explore_v1_20250709-152956/eval_ep400_sub0.gif)

## Future Work

-   **More Sophisticated Global State for Mixer:** Improve how the global state is constructed for the mixer network.
-   **Adaptive CNN:** Use `nn.AdaptiveAvgPool2d` to handle varying grid sizes.
-   **Attention Analysis:** Visualize attention weights to understand the models' focus.
-   **Advanced Exploration Strategies:** Implement techniques like prioritized experience replay or noisy networks.
-   **Curriculum Learning:** Start with simpler scenarios and gradually increase complexity.
-   **Hyperparameter Optimization:** Use tools like Optuna or Ray Tune for systematic tuning.
-   **Decentralized Execution Evaluation Script:** Create a dedicated script for decentralized evaluation.
-   **More Complex Communication:** Explore learnable communication protocols.
-   **Heterogeneous Agents:** Extend the framework to support agents with different capabilities.