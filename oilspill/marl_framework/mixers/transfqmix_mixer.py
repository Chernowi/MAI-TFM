import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# The import below is fine, but make sure the agents/__init__.py is also correct.
from marl_framework.agents.transfqmix_agent import EntityEmbedder

class MixerTransformer(nn.Module): # Similar to AgentTransformer, but for global state
    def __init__(self, embed_dim, num_heads, num_blocks, ffn_dim_multiplier, dropout_rate):
        """
        Initialize the MixerTransformer module.
        
        Args:
            embed_dim (int): Embedding dimension for the transformer
            num_heads (int): Number of attention heads
            num_blocks (int): Number of transformer encoder layers
            ffn_dim_multiplier (int): Multiplier for feedforward network dimension
            dropout_rate (float): Dropout rate for regularization
        """
        super(MixerTransformer, self).__init__()
        self.embed_dim = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_dim_multiplier,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    def forward(self, embedded_entities_seq, src_key_padding_mask=None):
        """
        Forward pass through the mixer transformer.
        
        Args:
            embedded_entities_seq (torch.Tensor): Embedded entity sequence of shape 
                (batch_size, seq_len, embed_dim)
            src_key_padding_mask (torch.Tensor, optional): Padding mask of shape 
                (batch_size, seq_len) where True indicates positions to ignore
                
        Returns:
            torch.Tensor: Transformer output of shape (batch_size, seq_len, embed_dim)
        """
        # embedded_entities_seq: (batch_size, seq_len, embed_dim)
        # src_key_padding_mask: (batch_size, seq_len)
        transformer_output = self.transformer_encoder(embedded_entities_seq, src_key_padding_mask=src_key_padding_mask)
        return transformer_output


class TransfQMixMixer(nn.Module):
    def __init__(self, num_agents, env_global_state_spec, agent_transformer_embed_dim, mixer_config):
        """
        Initialize the TransfQMix mixer network.
        
        Args:
            num_agents (int): Number of agents in the environment
            env_global_state_spec (dict): Specification of the global state structure
            agent_transformer_embed_dim (int): Embedding dimension from agent transformers
            mixer_config (dict): Configuration parameters for the mixer network
        """
        super(TransfQMixMixer, self).__init__()
        self.num_agents = num_agents
        self.mixer_config = mixer_config
        
        self.global_entity_raw_feature_dim = env_global_state_spec["global_state"]["entity_feature_dim"]
        self.mixer_embed_dim = mixer_config.get("MIXER_TRANSFORMER_EMBED_DIM", 64)

        # Embedder for global state entities
        self.global_state_entity_embedder = EntityEmbedder(
            self.global_entity_raw_feature_dim, 
            self.mixer_embed_dim
        )

        # Mixer's Transformer
        self.mixer_transformer = MixerTransformer(
            embed_dim=self.mixer_embed_dim,
            num_heads=mixer_config.get("MIXER_TRANSFORMER_NUM_HEADS", 4),
            num_blocks=mixer_config.get("MIXER_TRANSFORMER_NUM_BLOCKS", 2),
            ffn_dim_multiplier=mixer_config.get("MIXER_TRANSFORMER_FFN_DIM_MULTIPLIER", 4),
            dropout_rate=mixer_config.get("MIXER_TRANSFORMER_DROPOUT_RATE", 0.1)
        )

        self.agent_hidden_state_dim = agent_transformer_embed_dim
        if self.agent_hidden_state_dim != self.mixer_embed_dim:
            self.agent_h_projector = nn.Linear(self.agent_hidden_state_dim, self.mixer_embed_dim)
        else:
            self.agent_h_projector = nn.Identity()

        self.mlp_hidden_dim = mixer_config.get("MIXER_MLP_HIDDEN_DIM", 64)
        
        self.hyper_w1_b1_head = nn.Linear(self.mixer_embed_dim, 
                                          (self.num_agents * self.mlp_hidden_dim) + self.mlp_hidden_dim)
        self.hyper_w2_b2_head = nn.Linear(self.mixer_embed_dim, 
                                          self.mlp_hidden_dim + 1)


    def forward(self, agent_q_values_batch, global_state_batch, agent_h_states_batch, src_key_padding_mask):
        """
        Forward pass through the TransfQMix mixer network.
        
        This method combines individual agent Q-values using a mixing network that processes
        both agent hidden states and global state entities through a transformer architecture.
        The mixer uses hypernetworks to generate mixing weights dynamically based on the
        global state context.
        
        Args:
            agent_q_values_batch (torch.Tensor): Q-values for chosen actions of shape 
                (batch_size, num_agents)
            global_state_batch (torch.Tensor): Padded tensor of raw global entities of shape
                (batch_size, max_global_entities, raw_feat_dim)
            agent_h_states_batch (torch.Tensor): Agent hidden states of shape
                (batch_size, num_agents, agent_hidden_state_dim)
            src_key_padding_mask (torch.Tensor): Padding mask for mixer transformer of shape
                (batch_size, num_agents + max_global_entities) where True indicates positions to ignore
                
        Returns:
            torch.Tensor: Mixed Q-total values of shape (batch_size, 1)
        """
        batch_size = agent_q_values_batch.shape[0]

        # 1. Embed global state entities (already a padded tensor)
        embedded_global_entities = self.global_state_entity_embedder(global_state_batch)

        # 2. Project agent hidden states if dimensions differ
        projected_agent_h_states = self.agent_h_projector(agent_h_states_batch)

        # 3. Concatenate for mixer transformer input
        mixer_transformer_input_seq = torch.cat((projected_agent_h_states, embedded_global_entities), dim=1)

        # 4. Pass through Mixer Transformer
        mixer_transformer_output_seq = self.mixer_transformer(mixer_transformer_input_seq, src_key_padding_mask=src_key_padding_mask)
        
        # 5. Use average pooling over all valid output tokens to generate hypernetwork parameters.
        # Mask is inverted (~), converted to float, and unsqueezed for broadcasting.
        valid_outputs_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        masked_outputs = mixer_transformer_output_seq * valid_outputs_mask
        sum_outputs = masked_outputs.sum(dim=1)
        num_valid = valid_outputs_mask.sum(dim=1).clamp(min=1) # Avoid division by zero
        hypernet_input_features = sum_outputs / num_valid

        # 6. Generate MLP parameters via Hypernetwork
        w1_b1_params = self.hyper_w1_b1_head(hypernet_input_features)
        w2_b2_params = self.hyper_w2_b2_head(hypernet_input_features)

        w1_size = self.num_agents * self.mlp_hidden_dim
        b1_size = self.mlp_hidden_dim
        w2_size = self.mlp_hidden_dim

        W1 = w1_b1_params[:, :w1_size].reshape(batch_size, self.num_agents, self.mlp_hidden_dim)
        b1 = w1_b1_params[:, w1_size:(w1_size + b1_size)].reshape(batch_size, 1, self.mlp_hidden_dim)
        
        W2 = w2_b2_params[:, :w2_size].reshape(batch_size, self.mlp_hidden_dim, 1)
        b2 = w2_b2_params[:, w2_size:].reshape(batch_size, 1, 1)
        
        # Enforce positivity constraint
        W1 = torch.abs(W1)
        W2 = torch.abs(W2)

        # 7. Mix agent Q-values
        q_vals_reshaped = agent_q_values_batch.unsqueeze(1) 
        hidden_layer = F.elu(torch.bmm(q_vals_reshaped, W1) + b1)
        q_total_batch = torch.bmm(hidden_layer, W2) + b2
        
        return q_total_batch.squeeze(-1)