import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BeliefMapCNN(nn.Module):
    def __init__(self, grid_r, grid_c, cnn_output_feature_dim, config):
        super(BeliefMapCNN, self).__init__()
        # Configurable CNN architecture
        # Example for a 64x64 input, adjust based on grid_r, grid_c
        # Input: (batch, 1, grid_r, grid_c) - assuming 1 channel for belief value
        
        # Default architecture from spec (adjust based on actual grid_r, grid_c to ensure output size for flatten)
        # For dynamic sizing, adaptive pooling could be used before flatten, or calculate flatten_dim
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2) # Output: (batch, 16, grid_r, grid_c)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch, 16, grid_r/2, grid_c/2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: (batch, 32, grid_r/2, grid_c/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch, 32, grid_r/4, grid_c/4)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Output: (batch, 64, grid_r/4, grid_c/4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch, 64, grid_r/8, grid_c/8)

        # Calculate flattened dimension
        # This needs to be robust if grid_r/grid_c are not perfectly divisible by 8
        conv_out_r = grid_r // 8
        conv_out_c = grid_c // 8
        self.flattened_dim = 64 * conv_out_r * conv_out_c
        if self.flattened_dim == 0:
            raise ValueError(f"CNN flattened_dim is 0. Grid size {grid_r}x{grid_c} too small for 3 pool layers.")

        self.fc_out = nn.Linear(self.flattened_dim, cnn_output_feature_dim)
        self.cnn_output_feature_dim = cnn_output_feature_dim

    def forward(self, belief_map_batch):
        # belief_map_batch: (batch_size, grid_r, grid_c)
        # Add channel dimension: (batch_size, 1, grid_r, grid_c)
        x = belief_map_batch.unsqueeze(1)
        
        # Normalize belief values from -1,0,1 to a suitable range for CNN e.g. 0, 0.5, 1
        x = (x + 1.0) / 2.0 

        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        
        x = x.reshape(-1, self.flattened_dim) # Flatten
        f_cnn = F.relu(self.fc_out(x)) # (batch_size, cnn_output_feature_dim)
        return f_cnn


class EntityEmbedder(nn.Module):
    def __init__(self, raw_feature_dim, embed_dim):
        super(EntityEmbedder, self).__init__()
        self.linear = nn.Linear(raw_feature_dim, embed_dim)

    def forward(self, entity_features_batch):
        # entity_features_batch: (batch_size, num_entities, raw_feature_dim)
        # or (total_entities_in_batch, raw_feature_dim) if processing entities flatly
        return F.relu(self.linear(entity_features_batch))


class AgentTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, ffn_dim_multiplier, dropout_rate):
        super(AgentTransformer, self).__init__()
        self.embed_dim = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_dim_multiplier,
            dropout=dropout_rate,
            batch_first=True # Important: expects (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    def forward(self, embedded_entities_seq, src_key_padding_mask=None):
        # embedded_entities_seq: (batch_size, seq_len, embed_dim)
        # src_key_padding_mask: (batch_size, seq_len) - True for padded items
        transformer_output = self.transformer_encoder(embedded_entities_seq, src_key_padding_mask=src_key_padding_mask)
        return transformer_output


class QValueHead(nn.Module):
    def __init__(self, embed_dim, num_actions):
        super(QValueHead, self).__init__()
        self.linear = nn.Linear(embed_dim, num_actions)

    def forward(self, agent_hidden_state):
        # agent_hidden_state: (batch_size, embed_dim)
        return self.linear(agent_hidden_state)


class TransfQMixAgentNN(nn.Module):
    """
    Combines BeliefMapCNN, EntityEmbedder, AgentTransformer, and QValueHead for a single agent's policy.
    This network is SHARED among all agents.
    """
    def __init__(self, env_obs_spec, agent_config, global_config):
        super(TransfQMixAgentNN, self).__init__()
        
        grid_r, grid_c, _ = env_obs_spec["belief_map_shape"]
        self.cnn_output_dim = agent_config.get("CNN_OUTPUT_FEATURE_DIM", 128)
        self.belief_map_cnn = BeliefMapCNN(grid_r, grid_c, self.cnn_output_dim, agent_config)

        self.entity_raw_feature_dim = env_obs_spec["agent_observation"]["entity_feature_dim"]
        self.transformer_embed_dim = agent_config.get("AGENT_TRANSFORMER_EMBED_DIM", 64)
        self.entity_embedder = EntityEmbedder(self.entity_raw_feature_dim, self.transformer_embed_dim)

        self.transformer = AgentTransformer(
            embed_dim=self.transformer_embed_dim,
            num_heads=agent_config.get("AGENT_TRANSFORMER_NUM_HEADS", 4),
            num_blocks=agent_config.get("AGENT_TRANSFORMER_NUM_BLOCKS", 2),
            ffn_dim_multiplier=agent_config.get("AGENT_TRANSFORMER_FFN_DIM_MULTIPLIER", 4),
            dropout_rate=agent_config.get("AGENT_TRANSFORMER_DROPOUT_RATE", 0.1)
        )
        
        self.q_value_head = QValueHead(self.transformer_embed_dim, global_config.get("ACTION_SPACE_SIZE", 6))
        
        # For explicit local sensor reading entity construction if needed
        self.direct_sensing_mode = global_config.get("DIRECT_SENSING_MODE", "surrounding_cells")
        self.num_headings = global_config.get("NUM_HEADINGS", 8)
        
        # Store for obs processing if CNN features are part of the direct entity list
        # This helps reconstruct observation entities correctly within the forward pass
        # if the env provides raw belief map and agent needs to insert its CNN output
        self.agent_config = agent_config
        self.global_config = global_config


    def forward(self, agent_belief_map_batch, agent_raw_obs_entity_list_batch, h_in_batch, src_key_padding_mask=None):
        """
        Processes a batch of observations for multiple agents (or one agent over time).
        Args:
            agent_belief_map_batch: (batch_size, grid_r, grid_c) - Agent's own belief map.
            agent_raw_obs_entity_list_batch: List of lists of raw entity feature np.arrays.
                                             Outer list is batch, inner list is entities for one agent obs.
                                             [[entity1_agent1, entity2_agent1,...], [entity1_agent2, ...]]
                                             These raw features are padded by the environment.
            h_in_batch: (batch_size, transformer_embed_dim) - Previous agent transformer hidden state.
            src_key_padding_mask: (batch_size, max_seq_len) for transformer.
        Returns:
            q_values: (batch_size, num_actions)
            h_out: (batch_size, transformer_embed_dim) - New agent transformer hidden state.
            f_cnn_out: (batch_size, cnn_output_dim) - Output from CNN (for global state).
        """
        batch_size = agent_belief_map_batch.shape[0]

        # 1. Process belief map through CNN
        f_cnn_out = self.belief_map_cnn(agent_belief_map_batch) # (batch_size, cnn_output_dim)

        # 2. Prepare entity sequence for transformer
        #    The environment provides raw_obs_entity_list_batch where one entity might be a placeholder for F_cnn.
        #    We need to replace that placeholder with the actual f_cnn_out.
        
        processed_entity_sequences = []
        max_entities_in_batch = 0

        for i in range(batch_size):
            current_agent_entities_raw = agent_raw_obs_entity_list_batch[i]
            current_agent_entities_processed = []
            
            for raw_entity_features_np in current_agent_entities_raw:
                # Convert numpy to tensor if not already
                raw_entity_features = torch.tensor(raw_entity_features_np, dtype=torch.float32, device=f_cnn_out.device)
                
                # Identify the CNN placeholder entity by its flags/structure
                # Assuming IS_MAP_SUMMARY_FLAG (idx -2) is 1 for the CNN entity placeholder
                # And its feature part (before flags) matches cnn_output_dim
                # This logic depends on how env.py structures the placeholder
                is_map_summary_flag_idx = -2 # Example: IS_MAP_SUMMARY_FLAG
                is_sensor_flag_idx = -1      # Example: IS_SENSOR_FLAG
                
                # Assuming flags are [IS_SELF, IS_AGENT, IS_MAP_SUMMARY, IS_SENSOR]
                # Placeholder for CNN might have [0,0,1,0] as last 4 flags
                # This needs to be robust:
                flags_part = raw_entity_features[-4:]
                is_cnn_placeholder = (flags_part[0]==0 and flags_part[1]==0 and flags_part[2]==1 and flags_part[3]==0)

                if is_cnn_placeholder and len(raw_entity_features[:-4]) == self.cnn_output_dim :
                    # Replace placeholder features with actual f_cnn output for this batch item
                    # Keep the flags part
                    updated_entity_features = torch.cat((f_cnn_out[i], flags_part), dim=0)
                    current_agent_entities_processed.append(updated_entity_features)
                else:
                    current_agent_entities_processed.append(raw_entity_features)
            
            processed_entity_sequences.append(torch.stack(current_agent_entities_processed)) # (num_entities_for_this_agent, raw_feat_dim)
            if len(current_agent_entities_processed) > max_entities_in_batch:
                max_entities_in_batch = len(current_agent_entities_processed)

        # Pad sequences to max_entities_in_batch for batching into entity_embedder and transformer
        # And create the src_key_padding_mask
        padded_entity_sequences_for_embedding = torch.zeros(
            (batch_size, max_entities_in_batch, self.entity_raw_feature_dim), 
            dtype=torch.float32, device=f_cnn_out.device
        )
        if src_key_padding_mask is None: # if not provided (e.g. during single forward pass)
            # Add 1 for h_in when creating mask
            src_key_padding_mask_for_transformer = torch.ones(
                (batch_size, max_entities_in_batch + 1), dtype=torch.bool, device=f_cnn_out.device 
            ) # True means ignore

        for i in range(batch_size):
            seq = processed_entity_sequences[i] # (num_actual_entities, raw_feat_dim)
            padded_entity_sequences_for_embedding[i, :seq.shape[0], :] = seq
            if src_key_padding_mask is None: # Create mask if not passed
                 src_key_padding_mask_for_transformer[i, :(seq.shape[0] + 1)] = False # +1 for h_in

        # 3. Embed all entities
        # (batch_size * max_entities_in_batch, raw_feature_dim) -> (batch_size * max_entities_in_batch, transformer_embed_dim)
        # Or, pass as (batch_size, max_entities, raw_feature_dim) if embedder handles it
        embedded_entities = self.entity_embedder(padded_entity_sequences_for_embedding) # (batch_size, max_entities_in_batch, transformer_embed_dim)

        # 4. Prepend h_in to the sequence of embedded entities
        # h_in_batch: (batch_size, transformer_embed_dim) -> (batch_size, 1, transformer_embed_dim)
        h_in_reshaped = h_in_batch.unsqueeze(1) 
        transformer_input_seq = torch.cat((h_in_reshaped, embedded_entities), dim=1) # (batch_size, max_entities_in_batch + 1, embed_dim)

        # 5. Pass through Transformer
        # The src_key_padding_mask should correspond to transformer_input_seq shape
        transformer_output_seq = self.transformer(transformer_input_seq, src_key_padding_mask=src_key_padding_mask_for_transformer)
        
        # 6. The first element of the transformer output sequence is the new hidden state h_out
        h_out = transformer_output_seq[:, 0, :] # (batch_size, transformer_embed_dim)

        # 7. Pass h_out through Q-Value Head
        q_values = self.q_value_head(h_out) # (batch_size, num_actions)

        return q_values, h_out, f_cnn_out


if __name__ == '__main__':
    # --- Example Usage & Test ---
    # Mock env_obs_spec and configs
    mock_env_obs_spec = {
        "belief_map_shape": (10, 10, 1), # grid_r, grid_c, channels
        "agent_observation": {
            "entity_feature_dim": 32 + 4, # Example: 32 cnn_dim + 4 flags for one entity type
            "max_num_entities_approx": 5
        },
        "global_state": { # Not directly used by agent NN but good for context
            "entity_feature_dim": 32 + 2 + 3, # agent_global + current + global_belief flags
            "max_num_entities_approx": 3 + 1 + 1 
        }
    }
    mock_agent_config = {
        "CNN_OUTPUT_FEATURE_DIM": 32,
        "AGENT_TRANSFORMER_EMBED_DIM": 24, # Smaller for test
        "AGENT_TRANSFORMER_NUM_HEADS": 2,
        "AGENT_TRANSFORMER_NUM_BLOCKS": 1,
        "AGENT_TRANSFORMER_FFN_DIM_MULTIPLIER": 2,
        "AGENT_TRANSFORMER_DROPOUT_RATE": 0.0
    }
    mock_global_config = {
        "ACTION_SPACE_SIZE": 6,
        "NUM_HEADINGS": 8,
        "DIRECT_SENSING_MODE": "surrounding_cells"
    }

    # Instantiate the agent network
    agent_nn = TransfQMixAgentNN(mock_env_obs_spec, mock_agent_config, mock_global_config)
    print("Agent NN instantiated.")
    print(f"CNN output dim: {agent_nn.cnn_output_dim}")
    print(f"Entity raw feature dim for embedder: {agent_nn.entity_raw_feature_dim}")
    print(f"Transformer embed dim: {agent_nn.transformer_embed_dim}")

    # Create dummy input data for a batch size of 2
    batch_s = 2
    grid_r, grid_c, _ = mock_env_obs_spec["belief_map_shape"]
    
    dummy_belief_maps = torch.randn(batch_s, grid_r, grid_c) # (batch, r, c)
    
    # Dummy raw obs entity lists (list of lists of np arrays)
    # Entity structure: cnn_placeholder_features (32) + 4 flags
    dummy_cnn_placeholder_features = np.zeros(mock_agent_config["CNN_OUTPUT_FEATURE_DIM"])
    dummy_flags_cnn = np.array([0,0,1,0]) # IS_MAP_SUMMARY
    cnn_entity_placeholder_np = np.concatenate([dummy_cnn_placeholder_features, dummy_flags_cnn])
    
    dummy_self_features = np.random.rand(2 + mock_global_config["NUM_HEADINGS"])
    dummy_flags_self = np.array([1,1,0,0]) # IS_SELF, IS_AGENT
    self_entity_np = np.concatenate([dummy_self_features, dummy_flags_self])
    # Pad to entity_raw_feature_dim
    self_entity_np = np.pad(self_entity_np, (0, mock_env_obs_spec["agent_observation"]["entity_feature_dim"] - len(self_entity_np)))


    dummy_raw_obs_batch = [
        [self_entity_np, cnn_entity_placeholder_np, self_entity_np[:mock_env_obs_spec["agent_observation"]["entity_feature_dim"]].copy()], # Agent 1: self, cnn_placeholder, other_agent_like
        [self_entity_np, cnn_entity_placeholder_np]  # Agent 2: self, cnn_placeholder
    ]
    
    dummy_h_in = torch.randn(batch_s, mock_agent_config["AGENT_TRANSFORMER_EMBED_DIM"])

    # Create src_key_padding_mask for transformer
    # Max entities in this dummy batch is 3. Transformer input seq len = max_entities + 1 (for h_in) = 4
    # Batch item 1 has 3 entities (+h_in = 4 inputs to transformer), item 2 has 2 entities (+h_in = 3 inputs)
    # Mask should be True for padded/invalid items.
    # Seq len for transformer: max_entities + 1 (for h_in)
    max_entities_test = 0
    for obs_list in dummy_raw_obs_batch:
        if len(obs_list) > max_entities_test:
            max_entities_test = len(obs_list)
    
    transformer_seq_len = max_entities_test + 1
    test_src_key_padding_mask = torch.ones((batch_s, transformer_seq_len), dtype=torch.bool)
    for i in range(batch_s):
        num_actual_entities = len(dummy_raw_obs_batch[i])
        test_src_key_padding_mask[i, :(num_actual_entities + 1)] = False # +1 for h_in

    print(f"Dummy belief maps shape: {dummy_belief_maps.shape}")
    print(f"Dummy h_in shape: {dummy_h_in.shape}")
    print(f"Test src_key_padding_mask (True means ignore):\n{test_src_key_padding_mask}")


    # Forward pass
    q_vals, h_out, f_cnn = agent_nn(dummy_belief_maps, dummy_raw_obs_batch, dummy_h_in, src_key_padding_mask=test_src_key_padding_mask)

    print(f"\nOutput Q-values shape: {q_vals.shape}") # Expected: (batch_s, num_actions)
    print(f"Output h_out shape: {h_out.shape}")     # Expected: (batch_s, transformer_embed_dim)
    print(f"Output f_cnn shape: {f_cnn.shape}")     # Expected: (batch_s, cnn_output_dim)
    print("Test successful if shapes are correct.")