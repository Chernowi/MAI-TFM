import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from marl_framework.agents.transfqmix_agent import EntityEmbedder # Re-use if compatible

class MixerTransformer(nn.Module): # Similar to AgentTransformer, but for global state
    def __init__(self, embed_dim, num_heads, num_blocks, ffn_dim_multiplier, dropout_rate):
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
        # embedded_entities_seq: (batch_size, seq_len, embed_dim)
        # src_key_padding_mask: (batch_size, seq_len)
        transformer_output = self.transformer_encoder(embedded_entities_seq, src_key_padding_mask=src_key_padding_mask)
        return transformer_output


class TransfQMixMixer(nn.Module):
    def __init__(self, num_agents, env_global_state_spec, agent_transformer_embed_dim, mixer_config):
        super(TransfQMixMixer, self).__init__()
        self.num_agents = num_agents
        self.mixer_config = mixer_config
        
        self.global_entity_raw_feature_dim = env_global_state_spec["global_state"]["entity_feature_dim"]
        self.mixer_embed_dim = mixer_config.get("MIXER_TRANSFORMER_EMBED_DIM", 64) # Can be same as agent_transformer_embed_dim

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

        # The input to the transformer includes:
        # - N agent hidden states (h_t^a from agent transformers) -> these are already `agent_transformer_embed_dim`
        # - Embedded global state entities -> these will be `mixer_embed_dim`
        # If agent_transformer_embed_dim != mixer_embed_dim, agent hidden states need projection.
        self.agent_hidden_state_dim = agent_transformer_embed_dim
        if self.agent_hidden_state_dim != self.mixer_embed_dim:
            self.agent_h_projector = nn.Linear(self.agent_hidden_state_dim, self.mixer_embed_dim)
        else:
            self.agent_h_projector = nn.Identity()

        # Hypernetwork to generate MLP weights and biases for mixing
        # The mixer transformer's output needs to be processed to generate these.
        # Output of mixer transformer: (batch_size, num_mixer_transformer_outputs, mixer_embed_dim)
        # num_mixer_transformer_outputs = num_agents (for projected h_a) + num_global_entities
        
        # MLP Mixer structure (e.g., QMix style)
        self.mlp_hidden_dim = mixer_config.get("MIXER_MLP_HIDDEN_DIM", 64)
        
        # W1: (num_agents * mlp_hidden_dim) weights + b1: (mlp_hidden_dim) biases
        # W2: (mlp_hidden_dim * 1) weights + b2: (1) bias
        
        # The transformer needs to output enough features to parameterize these.
        # Let's assume the *first output token* of the mixer transformer (like a [CLS] token)
        # is used to generate the hypernetwork parameters.
        # Or, average pool all output tokens. Let's use first token for simplicity.
        
        # Size of params for W1: num_agents * self.mlp_hidden_dim
        # Size of params for b1: self.mlp_hidden_dim
        # Size of params for W2: self.mlp_hidden_dim * 1
        # Size of params for b2: 1
        
        self.hyper_w1_b1_head = nn.Linear(self.mixer_embed_dim, 
                                          (self.num_agents * self.mlp_hidden_dim) + self.mlp_hidden_dim)
        self.hyper_w2_b2_head = nn.Linear(self.mixer_embed_dim, 
                                          self.mlp_hidden_dim + 1)


    def forward(self, agent_q_values_batch, global_raw_state_entity_list_batch, agent_h_states_batch):
        """
        Args:
            agent_q_values_batch: (batch_size, num_agents) - Q-value for the chosen action for each agent.
            global_raw_state_entity_list_batch: List of lists of raw global state entity np.arrays.
                                                Outer list is batch, inner list is entities for global state.
                                                Padded by the environment.
            agent_h_states_batch: (batch_size, num_agents, agent_hidden_state_dim) - Hidden states from agent transformers.
        Returns:
            q_total_batch: (batch_size, 1)
        """
        batch_size = agent_q_values_batch.shape[0]
        device = agent_q_values_batch.device

        # 1. Prepare global state entities for embedding
        max_global_entities_in_batch = 0
        processed_global_entity_sequences = []
        for i in range(batch_size):
            current_global_entities_raw = global_raw_state_entity_list_batch[i]
            current_global_entities_processed = [
                torch.tensor(raw_entity_features_np, dtype=torch.float32, device=device)
                for raw_entity_features_np in current_global_entities_raw
            ]
            processed_global_entity_sequences.append(torch.stack(current_global_entities_processed) if current_global_entities_processed else torch.empty(0, self.global_entity_raw_feature_dim, device=device))
            if len(current_global_entities_processed) > max_global_entities_in_batch:
                max_global_entities_in_batch = len(current_global_entities_processed)
        
        padded_global_entity_sequences_for_embedding = torch.zeros(
            (batch_size, max_global_entities_in_batch, self.global_entity_raw_feature_dim),
            dtype=torch.float32, device=device
        )
        # Create src_key_padding_mask for mixer_transformer input
        # Input seq len = num_agents (for h_states) + max_global_entities_in_batch
        mixer_transformer_input_seq_len = self.num_agents + max_global_entities_in_batch
        mixer_src_key_padding_mask = torch.ones(
            (batch_size, mixer_transformer_input_seq_len), dtype=torch.bool, device=device
        )

        for i in range(batch_size):
            seq = processed_global_entity_sequences[i] # (num_actual_global_entities, raw_feat_dim)
            if seq.nelement() > 0: # Check if tensor is not empty
                 padded_global_entity_sequences_for_embedding[i, :seq.shape[0], :] = seq
            # Mask for agent h_states (always present)
            mixer_src_key_padding_mask[i, :self.num_agents] = False 
            # Mask for global entities
            if seq.nelement() > 0:
                mixer_src_key_padding_mask[i, self.num_agents:(self.num_agents + seq.shape[0])] = False


        # 2. Embed global state entities
        embedded_global_entities = self.global_state_entity_embedder(padded_global_entity_sequences_for_embedding)
        # Shape: (batch_size, max_global_entities_in_batch, mixer_embed_dim)

        # 3. Project agent hidden states if dimensions differ
        projected_agent_h_states = self.agent_h_projector(agent_h_states_batch)
        # Shape: (batch_size, num_agents, mixer_embed_dim)

        # 4. Concatenate for mixer transformer input
        mixer_transformer_input_seq = torch.cat((projected_agent_h_states, embedded_global_entities), dim=1)
        # Shape: (batch_size, num_agents + max_global_entities_in_batch, mixer_embed_dim)

        # 5. Pass through Mixer Transformer
        mixer_transformer_output_seq = self.mixer_transformer(mixer_transformer_input_seq, src_key_padding_mask=mixer_src_key_padding_mask)
        
        # 6. Use average pooling over all valid output tokens to generate hypernetwork parameters.
        valid_outputs_mask = (~mixer_src_key_padding_mask).unsqueeze(-1).float() # (batch, seq_len, 1)
        masked_outputs = mixer_transformer_output_seq * valid_outputs_mask
        sum_outputs = masked_outputs.sum(dim=1) # (batch, embed_dim)
        num_valid = valid_outputs_mask.sum(dim=1).clamp(min=1) # (batch, 1)
        hypernet_input_features = sum_outputs / num_valid


        # 7. Generate MLP parameters via Hypernetwork
        w1_b1_params = self.hyper_w1_b1_head(hypernet_input_features)
        w2_b2_params = self.hyper_w2_b2_head(hypernet_input_features)

        # Extract W1, b1, W2, b2
        w1_size = self.num_agents * self.mlp_hidden_dim
        b1_size = self.mlp_hidden_dim
        w2_size = self.mlp_hidden_dim # * 1 implicitly

        W1 = w1_b1_params[:, :w1_size].reshape(batch_size, self.num_agents, self.mlp_hidden_dim)
        b1 = w1_b1_params[:, w1_size:(w1_size + b1_size)].reshape(batch_size, 1, self.mlp_hidden_dim)
        
        W2 = w2_b2_params[:, :w2_size].reshape(batch_size, self.mlp_hidden_dim, 1)
        b2 = w2_b2_params[:, w2_size:].reshape(batch_size, 1, 1)
        
        # Monotonicity: abs for weights, relu for biases (or ensure positive during generation)
        W1 = torch.abs(W1)
        W2 = torch.abs(W2)
        # b1 can be anything (or relu for standard QMix)
        # b2 can be anything

        # 8. Mix agent Q-values
        # agent_q_values_batch: (batch_size, num_agents) -> (batch_size, 1, num_agents) for bmm
        q_vals_reshaped = agent_q_values_batch.unsqueeze(1) 
        
        # Layer 1
        hidden_layer = F.elu(torch.bmm(q_vals_reshaped, W1) + b1) # (batch_size, 1, mlp_hidden_dim)
        
        # Layer 2 (Output)
        q_total_batch = torch.bmm(hidden_layer, W2) + b2 # (batch_size, 1, 1)
        
        return q_total_batch.squeeze(-1) # (batch_size, 1)


if __name__ == '__main__':
    # --- Example Usage & Test ---
    num_a = 3
    agent_h_dim = 24 # From agent_nn test
    
    mock_mixer_config = {
        "MIXER_TRANSFORMER_EMBED_DIM": 32, # Can be different from agent_h_dim
        "MIXER_TRANSFORMER_NUM_HEADS": 2,
        "MIXER_TRANSFORMER_NUM_BLOCKS": 1,
        "MIXER_TRANSFORMER_FFN_DIM_MULTIPLIER": 2,
        "MIXER_TRANSFORMER_DROPOUT_RATE": 0.0,
        "MIXER_MLP_HIDDEN_DIM": 20 # Smaller for test
    }
    
    # Mock global state spec (simplified)
    mock_global_state_spec = {
        "global_state": {
            "entity_feature_dim": 32 + 2 + 3, # Example as in agent_nn test
            "max_num_entities_approx": num_a + 1 + 1 # num_agents + current + global_belief_summary
        }
    }

    mixer_nn = TransfQMixMixer(num_a, mock_global_state_spec, agent_h_dim, mock_mixer_config)
    print("Mixer NN instantiated.")

    batch_s = 2
    dummy_agent_q_vals = torch.randn(batch_s, num_a)
    dummy_agent_h_states = torch.randn(batch_s, num_a, agent_h_dim)
    
    # Dummy global state entity lists (list of lists of np arrays)
    global_entity_feat_dim = mock_global_state_spec["global_state"]["entity_feature_dim"]
    dummy_global_entity1_np = np.random.rand(global_entity_feat_dim)
    dummy_global_entity2_np = np.random.rand(global_entity_feat_dim)
    
    dummy_global_raw_state_batch = [
        [dummy_global_entity1_np, dummy_global_entity2_np], # State 1: 2 global entities
        [dummy_global_entity1_np]                          # State 2: 1 global entity
    ]

    print(f"Dummy agent_q_vals shape: {dummy_agent_q_vals.shape}")
    print(f"Dummy agent_h_states shape: {dummy_agent_h_states.shape}")
    
    # Forward pass
    q_total = mixer_nn(dummy_agent_q_vals, dummy_global_raw_state_batch, dummy_agent_h_states)
    
    print(f"\nOutput Q_total shape: {q_total.shape}") # Expected: (batch_s, 1)
    print("Mixer test successful if shapes are correct.")
    print("Q_total example:", q_total)