import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, num_agents, agent_ids, obs_spec, global_state_spec, device='cpu'):
        self.capacity = capacity
        self.num_agents = num_agents
        self.agent_ids = agent_ids # list of agent_id strings
        self.device = device
        
        self.memory = deque(maxlen=capacity)
        
        self.obs_entity_feature_dim = obs_spec["agent_observation"]["entity_feature_dim"]
        self.global_entity_feature_dim = global_state_spec["global_state"]["entity_feature_dim"]
        self.belief_map_shape = obs_spec["belief_map_shape"] # (grid_r, grid_c, 1)

    def push(self, agent_obs_dict, agent_belief_maps_dict, global_state_entities, 
             joint_actions_dict, rewards_dict, 
             next_agent_obs_dict, next_agent_belief_maps_dict, next_global_state_entities,
             dones_dict, agent_h_in_dict, agent_h_out_dict):
        """
        Stores a transition in the buffer.
        All inputs are expected to be dictionaries keyed by agent_id or a single team value.
        Entities are lists of np.arrays (raw features). Hidden states are torch tensors.
        Belief maps are dictionaries of np.arrays.

        Args:
            agent_obs_dict (dict): {agent_id: list_of_np_entity_features}
            agent_belief_maps_dict (dict): {agent_id: np.array (grid_r, grid_c)}
            global_state_entities (list): list_of_np_entity_features for global state
            joint_actions_dict (dict): {agent_id: action_int}
            rewards_dict (dict): {agent_id: reward_float} (should be team reward)
            next_agent_obs_dict (dict): {agent_id: list_of_np_entity_features}
            next_agent_belief_maps_dict (dict): {agent_id: np.array (grid_r, grid_c)}
            next_global_state_entities (list): list_of_np_entity_features
            dones_dict (dict): {agent_id: done_bool} (should be team done)
            agent_h_in_dict (dict): {agent_id: h_in_tensor (1, embed_dim)} - h_in for current obs
            agent_h_out_dict (dict): {agent_id: h_out_tensor (1, embed_dim)} - h_out from current obs (h_in for next)
        """
        
        ordered_agent_obs = [agent_obs_dict[aid] for aid in self.agent_ids]
        # Store belief maps as a list of np.arrays in agent order
        ordered_agent_belief_maps = [agent_belief_maps_dict[aid].astype(np.int8) for aid in self.agent_ids] # Save space with int8

        ordered_joint_actions = np.array([joint_actions_dict[aid] for aid in self.agent_ids], dtype=np.int64)
        
        team_reward = rewards_dict.get(self.agent_ids[0], rewards_dict.get("__all__", 0.0))
        team_done = dones_dict.get(self.agent_ids[0], dones_dict.get("__all__", False))

        ordered_next_agent_obs = [next_agent_obs_dict[aid] for aid in self.agent_ids]
        ordered_next_agent_belief_maps = [next_agent_belief_maps_dict[aid].astype(np.int8) for aid in self.agent_ids]
        
        ordered_agent_h_in = [agent_h_in_dict[aid].detach().cpu() for aid in self.agent_ids]
        ordered_agent_h_out = [agent_h_out_dict[aid].detach().cpu() for aid in self.agent_ids]

        transition = {
            'agent_obs': ordered_agent_obs,
            'agent_belief_maps': ordered_agent_belief_maps, # NEW
            'global_state': global_state_entities, 
            'actions': ordered_joint_actions, 
            'reward': np.array([team_reward], dtype=np.float32),
            'next_agent_obs': ordered_next_agent_obs,
            'next_agent_belief_maps': ordered_next_agent_belief_maps, # NEW
            'next_global_state': next_global_state_entities,
            'done': np.array([team_done], dtype=np.bool_),
            'h_in_list': ordered_agent_h_in,
            'h_out_list': ordered_agent_h_out,
        }
        self.memory.append(transition)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None 
        
        transitions = random.sample(self.memory, batch_size)
        
        batch = {}
        keys = transitions[0].keys()

        for key in keys:
            if key in ['agent_obs', 'next_agent_obs', 'global_state', 'next_global_state']:
                batch[key] = [t[key] for t in transitions] # List (batch_size) of list_of_entities
            elif key in ['agent_belief_maps', 'next_agent_belief_maps']:
                # List (batch_size) of list (num_agents) of np.arrays (grid_r, grid_c)
                # Stack to (batch_size, num_agents, grid_r, grid_c)
                list_of_belief_map_lists = [t[key] for t in transitions] # Each item is a list of num_agent maps
                
                # Check if all inner lists have the same number of agents
                if not all(len(agent_maps) == self.num_agents for agent_maps in list_of_belief_map_lists):
                    raise ValueError(f"Inconsistent number of agent belief maps in a transition for key {key}.")

                # Stack belief maps for each agent across the batch, then stack agents
                # (num_agents, batch_size, grid_r, grid_c)
                stacked_per_agent = [
                    np.stack([list_of_belief_map_lists[b_idx][agent_idx] for b_idx in range(batch_size)])
                    for agent_idx in range(self.num_agents)
                ]
                # Transpose to (batch_size, num_agents, grid_r, grid_c)
                batch[key] = torch.from_numpy(np.stack(stacked_per_agent, axis=1)).float().to(self.device)
            elif key in ['h_in_list', 'h_out_list']:
                batched_h_states_per_agent = [[] for _ in range(self.num_agents)]
                for trans in transitions:
                    for i, h_state_tensor in enumerate(trans[key]):
                        batched_h_states_per_agent[i].append(h_state_tensor)
                
                stacked_h_per_agent = [torch.cat(h_list, dim=0) for h_list in batched_h_states_per_agent]
                batch[key] = torch.stack(stacked_h_per_agent, dim=1).to(self.device)
            else:
                batch[key] = torch.from_numpy(np.stack([t[key] for t in transitions])).to(self.device)
        
        return batch

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    # --- Example Usage & Test ---
    grid_r_test, grid_c_test = 10,10
    mock_obs_spec = {
        "agent_observation": {"entity_feature_dim": 10},
        "belief_map_shape": (grid_r_test, grid_c_test, 1) # Added for test
    }
    mock_global_spec = {"global_state": {"entity_feature_dim": 12}}
    num_a = 2
    agent_ids_test = [f"agent_{i}" for i in range(num_a)]
    buffer_capacity = 100
    h_dim = 16

    buffer = ReplayBuffer(buffer_capacity, num_a, agent_ids_test, mock_obs_spec, mock_global_spec)
    print(f"Buffer initialized with capacity {buffer.capacity}")

    for i in range(5):
        dummy_agent_obs = { aid: [np.random.rand(10).astype(np.float32) for _ in range(random.randint(1,3))] for aid in agent_ids_test }
        dummy_belief_maps = { aid: np.random.randint(-1, 2, (grid_r_test, grid_c_test)) for aid in agent_ids_test } # Added
        dummy_global_state = [np.random.rand(12).astype(np.float32) for _ in range(random.randint(1,2))]
        dummy_actions = {aid: random.randint(0,5) for aid in agent_ids_test}
        dummy_rewards = {aid: random.random(), "__all__": random.random()}
        dummy_dones = {aid: (random.random() > 0.8), "__all__": (random.random() > 0.8)}
        dummy_h_in = {aid: torch.randn(1, h_dim) for aid in agent_ids_test}
        dummy_h_out = {aid: torch.randn(1, h_dim) for aid in agent_ids_test}

        buffer.push(dummy_agent_obs, dummy_belief_maps, dummy_global_state, dummy_actions, dummy_rewards,
                    dummy_agent_obs, dummy_belief_maps, dummy_global_state, # Using same for next_obs/maps for simplicity
                    dummy_dones, dummy_h_in, dummy_h_out)
    
    print(f"Buffer size: {len(buffer)}")
    batch_size = 3
    sampled_batch = buffer.sample(batch_size)

    if sampled_batch:
        print("\nSampled batch keys:", sampled_batch.keys())
        print("Shape of batched agent_belief_maps:", sampled_batch['agent_belief_maps'].shape) # Expected: (batch_size, num_agents, grid_r, grid_c)
        print("Shape of batched next_agent_belief_maps:", sampled_batch['next_agent_belief_maps'].shape)
        # ... other prints from previous version ...
        print("Shape of batched actions:", sampled_batch['actions'].shape) 
        print("Device of agent_belief_maps:", sampled_batch['agent_belief_maps'].device)
    else:
        print("Not enough samples for batching.")