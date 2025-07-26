import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, num_agents, agent_ids, obs_spec, global_state_spec, device='cpu'):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            num_agents (int): Number of agents in the environment.
            agent_ids (list): List of agent IDs (strings).
            obs_spec (dict): Observation specifications, including entity feature dimensions and belief map shape.
            global_state_spec (dict): Global state specifications, including entity feature dimensions.
            device (str): Device to store tensors ('cpu' or 'cuda').
        """
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

        Args:
            agent_obs_dict (dict): {agent_id: list_of_np_entity_features}.
            agent_belief_maps_dict (dict): {agent_id: np.array (grid_r, grid_c)}.
            global_state_entities (list): List of np.arrays representing global state entities.
            joint_actions_dict (dict): {agent_id: action_int}.
            rewards_dict (dict): {agent_id: reward_float} (team reward).
            next_agent_obs_dict (dict): {agent_id: list_of_np_entity_features}.
            next_agent_belief_maps_dict (dict): {agent_id: np.array (grid_r, grid_c)}.
            next_global_state_entities (list): List of np.arrays representing next global state entities.
            dones_dict (dict): {agent_id: done_bool} (team done).
            agent_h_in_dict (dict): {agent_id: h_in_tensor (1, embed_dim)}.
            agent_h_out_dict (dict): {agent_id: h_out_tensor (1, embed_dim)}.
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
        """
        Samples a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A batch of transitions with keys corresponding to transition components.
        """
        if len(self.memory) < batch_size:
            return None 
        
        transitions = random.sample(self.memory, batch_size)
        
        batch = {}
        keys = transitions[0].keys()

        for key in keys:
            if key in ['agent_obs', 'next_agent_obs']:
                list_of_agent_obs_lists = [t[key] for t in transitions]

                max_entities = 0
                for agent_obs_list in list_of_agent_obs_lists:
                    for single_agent_entities in agent_obs_list:
                        max_entities = max(max_entities, len(single_agent_entities))

                padded_tensor = torch.zeros((batch_size, self.num_agents, max_entities, self.obs_entity_feature_dim), dtype=torch.float32)
                pad_mask = torch.ones((batch_size, self.num_agents, max_entities + 1), dtype=torch.bool)

                for b_idx, agent_obs_list in enumerate(list_of_agent_obs_lists):
                    for a_idx, single_agent_entities in enumerate(agent_obs_list):
                        num_entities = len(single_agent_entities)
                        if num_entities > 0:
                            stacked_entities = np.stack(single_agent_entities)
                            padded_tensor[b_idx, a_idx, :num_entities] = torch.from_numpy(stacked_entities)
                        pad_mask[b_idx, a_idx, :(num_entities + 1)] = False
                
                batch[key] = padded_tensor.to(self.device)
                batch[key + '_pad_mask'] = pad_mask.to(self.device)

            elif key in ['global_state', 'next_global_state']:
                list_of_entity_lists = [t[key] for t in transitions]
                max_entities = max(len(entities) for entities in list_of_entity_lists) if list_of_entity_lists else 0

                padded_tensor = torch.zeros((batch_size, max_entities, self.global_entity_feature_dim), dtype=torch.float32)
                pad_mask = torch.ones((batch_size, self.num_agents + max_entities), dtype=torch.bool)

                for b_idx, entities in enumerate(list_of_entity_lists):
                    num_entities = len(entities)
                    if num_entities > 0:
                        stacked_entities = np.stack(entities)
                        padded_tensor[b_idx, :num_entities] = torch.from_numpy(stacked_entities)
                    
                    pad_mask[b_idx, :self.num_agents + num_entities] = False

                batch[key] = padded_tensor.to(self.device)
                batch[key + '_pad_mask'] = pad_mask.to(self.device)
            elif key in ['agent_belief_maps', 'next_agent_belief_maps']:
                list_of_belief_map_lists = [t[key] for t in transitions]
                
                if not all(len(agent_maps) == self.num_agents for agent_maps in list_of_belief_map_lists):
                    raise ValueError(f"Inconsistent number of agent belief maps in a transition for key {key}.")

                stacked_per_agent = [
                    np.stack([list_of_belief_map_lists[b_idx][agent_idx] for b_idx in range(batch_size)])
                    for agent_idx in range(self.num_agents)
                ]
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
        """
        Returns the current size of the buffer.

        Returns:
            int: Number of transitions currently stored in the buffer.
        """
        return len(self.memory)