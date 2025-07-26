import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BeliefMapCNN(nn.Module):
    """
    Convolutional Neural Network for processing belief maps.
    
    Processes 2D belief maps through a series of convolutional and pooling layers
    to extract spatial features for agent decision making.
    
    Args:
        grid_r (int): Grid height (number of rows)
        grid_c (int): Grid width (number of columns)
        cnn_output_feature_dim (int): Dimension of the output feature vector
        config (dict): Configuration dictionary with CNN parameters
    """
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
        """
        Forward pass through the CNN.
        
        Args:
            belief_map_batch (torch.Tensor): Batch of belief maps with shape (batch_size, grid_r, grid_c)
            
        Returns:
            torch.Tensor: Processed features with shape (batch_size, cnn_output_feature_dim)
        """
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

class BeliefMapMaxPooler(nn.Module):
    """
    Simplified belief map processor using adaptive max pooling.
    
    A lightweight alternative to the full CNN that uses adaptive pooling
    to handle variable grid sizes and produce fixed-size outputs.
    
    Args:
        output_dim (int): Dimension of the output feature vector
        config (dict): Configuration dictionary with pooling parameters
    """
    def __init__(self, output_dim, config):
        super(BeliefMapMaxPooler, self).__init__()
        # Use adaptive pooling to handle any grid size and produce a fixed-size output
        self.pool_dim = config.get("MAXPOOL_DIM", 8) # e.g., 8x8 output
        self.pool = nn.AdaptiveMaxPool2d((self.pool_dim, self.pool_dim))
        self.flattened_dim = self.pool_dim * self.pool_dim
        self.fc_out = nn.Linear(self.flattened_dim, output_dim)

    def forward(self, belief_map_batch):
        """
        Forward pass through the max pooler.
        
        Args:
            belief_map_batch (torch.Tensor): Batch of belief maps with shape (batch_size, grid_r, grid_c)
            
        Returns:
            torch.Tensor: Processed features with shape (batch_size, output_dim)
        """
        # belief_map_batch: (batch_size, grid_r, grid_c)
        # Add channel dimension: (batch_size, 1, grid_r, grid_c)
        x = belief_map_batch.unsqueeze(1)
        
        # Normalize belief values from -1,0,1 to a suitable range e.g. 0, 0.5, 1
        x = (x + 1.0) / 2.0

        x = self.pool(x)
        x = x.reshape(-1, self.flattened_dim) # Flatten
        f_out = F.relu(self.fc_out(x)) # (batch_size, output_dim)
        return f_out

class EntityEmbedder(nn.Module):
    """
    Neural network module for embedding entity features.
    
    Transforms raw entity feature vectors into embeddings suitable for
    transformer processing.
    
    Args:
        raw_feature_dim (int): Dimension of input raw features
        embed_dim (int): Dimension of output embeddings
    """
    def __init__(self, raw_feature_dim, embed_dim):
        super(EntityEmbedder, self).__init__()
        self.linear = nn.Linear(raw_feature_dim, embed_dim)

    def forward(self, entity_features_batch):
        """
        Forward pass through the embedder.
        
        Args:
            entity_features_batch (torch.Tensor): Raw entity features with shape 
                (batch_size, num_entities, raw_feature_dim) or (total_entities_in_batch, raw_feature_dim)
                
        Returns:
            torch.Tensor: Embedded features with same batch dimensions but embed_dim as last dimension
        """
        # entity_features_batch: (batch_size, num_entities, raw_feature_dim)
        # or (total_entities_in_batch, raw_feature_dim) if processing entities flatly
        return F.relu(self.linear(entity_features_batch))


class AgentTransformer(nn.Module):
    """
    Transformer encoder for processing sequences of entity embeddings.
    
    Uses multi-head self-attention to model relationships between entities
    and the agent's hidden state.
    
    Args:
        embed_dim (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        num_blocks (int): Number of transformer encoder layers
        ffn_dim_multiplier (int): Multiplier for feedforward network dimension
        dropout_rate (float): Dropout rate for regularization
    """
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
        """
        Forward pass through the transformer.
        
        Args:
            embedded_entities_seq (torch.Tensor): Embedded entity sequence with shape (batch_size, seq_len, embed_dim)
            src_key_padding_mask (torch.Tensor, optional): Padding mask with shape (batch_size, seq_len).
                True values indicate positions to ignore.
                
        Returns:
            torch.Tensor: Transformer output with shape (batch_size, seq_len, embed_dim)
        """
        transformer_output = self.transformer_encoder(embedded_entities_seq, src_key_padding_mask=src_key_padding_mask)
        return transformer_output


class QValueHead(nn.Module):
    """
    Linear layer for computing Q-values from agent hidden states.
    
    Args:
        embed_dim (int): Dimension of input hidden state
        num_actions (int): Number of possible actions
    """
    def __init__(self, embed_dim, num_actions):
        super(QValueHead, self).__init__()
        self.linear = nn.Linear(embed_dim, num_actions)

    def forward(self, agent_hidden_state):
        """
        Forward pass to compute Q-values.
        
        Args:
            agent_hidden_state (torch.Tensor): Agent hidden state with shape (batch_size, embed_dim)
            
        Returns:
            torch.Tensor: Q-values with shape (batch_size, num_actions)
        """
        return self.linear(agent_hidden_state)


class TransfQMixAgentNN(nn.Module):
    """
    Complete agent neural network combining belief map processing, entity embedding,
    transformer attention, and Q-value computation.
    
    This network is shared among all agents and processes belief maps and entity
    observations to produce Q-values for action selection.
    
    Args:
        env_obs_spec (dict): Environment observation specification
        agent_config (dict): Agent-specific configuration parameters
        global_config (dict): Global configuration parameters
    """
    def __init__(self, env_obs_spec, agent_config, global_config):
        super(TransfQMixAgentNN, self).__init__()
        
        grid_r, grid_c, _ = env_obs_spec["belief_map_shape"]
        self.cnn_output_dim = agent_config.get("CNN_OUTPUT_FEATURE_DIM", 128)
        
        self.belief_processor_type = agent_config.get("BELIEF_MAP_PROCESSOR", "cnn")
        if self.belief_processor_type == "cnn":
            self.belief_map_processor = BeliefMapCNN(grid_r, grid_c, self.cnn_output_dim, agent_config)
        elif self.belief_processor_type == "maxpool":
            self.belief_map_processor = BeliefMapMaxPooler(self.cnn_output_dim, agent_config)
        else:
            raise ValueError(f"Unknown BELIEF_MAP_PROCESSOR type: {self.belief_processor_type}")

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


    def forward(self, agent_belief_map_batch, agent_obs_batch, h_in_batch, src_key_padding_mask):
        """
        Forward pass processing agent observations to produce Q-values.
        
        Processes belief maps through CNN/MaxPooler, embeds entity observations,
        applies transformer attention, and computes Q-values.
        
        Args:
            agent_belief_map_batch (torch.Tensor): Belief maps with shape (batch_size, grid_r, grid_c)
            agent_obs_batch (torch.Tensor): Padded entity observations with shape 
                (batch_size, max_entities, raw_feature_dim)
            h_in_batch (torch.Tensor): Previous hidden states with shape (batch_size, transformer_embed_dim)
            src_key_padding_mask (torch.Tensor): Transformer padding mask with shape 
                (batch_size, max_entities + 1)
        
        Returns:
            tuple: A tuple containing:
                - q_values (torch.Tensor): Q-values with shape (batch_size, num_actions)
                - h_out (torch.Tensor): New hidden states with shape (batch_size, transformer_embed_dim)  
                - f_cnn_out (torch.Tensor): CNN features with shape (batch_size, cnn_output_dim)
        """
        batch_size, max_entities, _ = agent_obs_batch.shape

        # 1. Process belief map through the selected processor (CNN or MaxPooler)
        f_cnn_out = self.belief_map_processor(agent_belief_map_batch) # (batch_size, cnn_output_dim)

        # 2. Prepare entity sequence for transformer by replacing CNN placeholder features.
        # This is now a fully batched tensor operation, no Python loops.
        flags_part = agent_obs_batch[:, :, -4:]
        
        # Create a mask to find the placeholder entities: shape (batch_size, max_entities)
        # Placeholder flags are [0,0,1,0]
        is_cnn_placeholder_mask = (flags_part[:, :, 0] == 0) & (flags_part[:, :, 1] == 0) & (flags_part[:, :, 2] == 1) & (flags_part[:, :, 3] == 0)

        # Prepare f_cnn_out for broadcasting by adding a sequence dimension
        f_cnn_expanded = f_cnn_out.unsqueeze(1).expand(-1, max_entities, -1) # -> (batch_size, max_entities, cnn_dim)
        
        # Get the feature part of the original tensor (all features except the last 4 flags)
        features_part = agent_obs_batch[:, :, :-4]
        
        # Use the mask to select between original features and the new cnn features
        # The mask needs to be expanded to match the feature dimension for torch.where
        updated_features = torch.where(
            is_cnn_placeholder_mask.unsqueeze(-1), 
            f_cnn_expanded, 
            features_part
        )
        
        # Recombine the (now updated) features with the original flags
        processed_entities = torch.cat((updated_features, flags_part), dim=-1)

        # 3. Embed all entities in a single batched call
        embedded_entities = self.entity_embedder(processed_entities) # (batch_size, max_entities, transformer_embed_dim)

        # 4. Prepend h_in to the sequence of embedded entities
        h_in_reshaped = h_in_batch.unsqueeze(1) 
        transformer_input_seq = torch.cat((h_in_reshaped, embedded_entities), dim=1) # (batch_size, max_entities + 1, embed_dim)

        # 5. Pass through Transformer
        # The src_key_padding_mask is provided by the replay buffer and corresponds to transformer_input_seq shape
        transformer_output_seq = self.transformer(transformer_input_seq, src_key_padding_mask=src_key_padding_mask)
        
        # 6. The first element of the transformer output sequence is the new hidden state h_out
        h_out = transformer_output_seq[:, 0, :] # (batch_size, transformer_embed_dim)

        # 7. Pass h_out through Q-Value Head
        q_values = self.q_value_head(h_out) # (batch_size, num_actions)

        return q_values, h_out, f_cnn_out