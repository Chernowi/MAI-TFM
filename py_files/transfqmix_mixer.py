import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# The import below is fine, but make sure the agents/__init__.py is also correct.
from marl_framework.agents.transfqmix_agent import EntityEmbedder

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


    def forward(self, agent_q_values_batch, global_raw_state_entity_list_batch, agent_h_states_batch):
        """
        Args:
            agent_q_values_batch: (batch_size, num_agents) - Q-value for the chosen action for each agent.
            global_raw_state_entity_list_batch: List of lists of raw global state entity np.arrays or tensors.
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
            # Handle both np.arrays from env and tensors from buffer
            current_global_entities_processed = [
                torch.tensor(e, dtype=torch.float32, device=device) if isinstance(e, np.ndarray) else e.to(device)
                for e in current_global_entities_raw
            ]
            if current_global_entities_processed:
                processed_global_entity_sequences.append(torch.stack(current_global_entities_processed))
                if len(current_global_entities_processed) > max_global_entities_in_batch:
                    max_global_entities_in_batch = len(current_global_entities_processed)
            else:
                 processed_global_entity_sequences.append(torch.empty(0, self.global_entity_raw_feature_dim, device=device))
        
        padded_global_entity_sequences_for_embedding = torch.zeros(
            (batch_size, max_global_entities_in_batch, self.global_entity_raw_feature_dim),
            dtype=torch.float32, device=device
        )
        
        mixer_transformer_input_seq_len = self.num_agents + max_global_entities_in_batch
        mixer_src_key_padding_mask = torch.ones(
            (batch_size, mixer_transformer_input_seq_len), dtype=torch.bool, device=device
        )

        for i in range(batch_size):
            seq = processed_global_entity_sequences[i]
            if seq.nelement() > 0:
                 padded_global_entity_sequences_for_embedding[i, :seq.shape[0], :] = seq
            
            mixer_src_key_padding_mask[i, :self.num_agents] = False 
            if seq.nelement() > 0:
                mixer_src_key_padding_mask[i, self.num_agents:(self.num_agents + seq.shape[0])] = False


        # 2. Embed global state entities
        embedded_global_entities = self.global_state_entity_embedder(padded_global_entity_sequences_for_embedding)

        # 3. Project agent hidden states if dimensions differ
        projected_agent_h_states = self.agent_h_projector(agent_h_states_batch)

        # 4. Concatenate for mixer transformer input
        mixer_transformer_input_seq = torch.cat((projected_agent_h_states, embedded_global_entities), dim=1)

        # 5. Pass through Mixer Transformer
        mixer_transformer_output_seq = self.mixer_transformer(mixer_transformer_input_seq, src_key_padding_mask=mixer_src_key_padding_mask)
        
        # 6. Use average pooling over all valid output tokens to generate hypernetwork parameters.
        valid_outputs_mask = (~mixer_src_key_padding_mask).unsqueeze(-1).float()
        masked_outputs = mixer_transformer_output_seq * valid_outputs_mask
        sum_outputs = masked_outputs.sum(dim=1)
        num_valid = valid_outputs_mask.sum(dim=1).clamp(min=1)
        hypernet_input_features = sum_outputs / num_valid

        # 7. Generate MLP parameters via Hypernetwork
        w1_b1_params = self.hyper_w1_b1_head(hypernet_input_features)
        w2_b2_params = self.hyper_w2_b2_head(hypernet_input_features)

        w1_size = self.num_agents * self.mlp_hidden_dim
        b1_size = self.mlp_hidden_dim
        w2_size = self.mlp_hidden_dim

        W1 = w1_b1_params[:, :w1_size].reshape(batch_size, self.num_agents, self.mlp_hidden_dim)
        b1 = w1_b1_params[:, w1_size:(w1_size + b1_size)].reshape(batch_size, 1, self.mlp_hidden_dim)
        
        W2 = w2_b2_params[:, :w2_size].reshape(batch_size, self.mlp_hidden_dim, 1)
        b2 = w2_b2_params[:, w2_size:].reshape(batch_size, 1, 1)
        
        W1 = torch.abs(W1)
        W2 = torch.abs(W2)

        # 8. Mix agent Q-values
        q_vals_reshaped = agent_q_values_batch.unsqueeze(1) 
        hidden_layer = F.elu(torch.bmm(q_vals_reshaped, W1) + b1)
        q_total_batch = torch.bmm(hidden_layer, W2) + b2
        
        return q_total_batch.squeeze(-1)


if __name__ == '__main__':
    # ... (rest of the file is unchanged) ...
    num_a = 3
    agent_h_dim = 24 # From agent_nn test
    
    mock_mixer_config = {
        "MIXER_TRANSFORMER_EMBED_DIM": 32,
        "MIXER_TRANSFORMER_NUM_HEADS": 2,
        "MIXER_TRANSFORMER_NUM_BLOCKS": 1,
        "MIXER_TRANSFORMER_FFN_DIM_MULTIPLIER": 2,
        "MIXER_TRANSFORMER_DROPOUT_RATE": 0.0,
        "MIXER_MLP_HIDDEN_DIM": 20
    }
    
    mock_global_state_spec = {
        "global_state": {
            "entity_feature_dim": 32 + 2 + 3,
            "max_num_entities_approx": num_a + 1 + 1
        }
    }

    mixer_nn = TransfQMixMixer(num_a, mock_global_state_spec, agent_h_dim, mock_mixer_config)
    print("Mixer NN instantiated.")

    batch_s = 2
    dummy_agent_q_vals = torch.randn(batch_s, num_a)
    dummy_agent_h_states = torch.randn(batch_s, num_a, agent_h_dim)
    
    global_entity_feat_dim = mock_global_state_spec["global_state"]["entity_feature_dim"]
    dummy_global_entity1_np = np.random.rand(global_entity_feat_dim)
    dummy_global_entity2_np = np.random.rand(global_entity_feat_dim)
    
    dummy_global_raw_state_batch = [
        [dummy_global_entity1_np, dummy_global_entity2_np],
        [dummy_global_entity1_np]
    ]

    print(f"Dummy agent_q_vals shape: {dummy_agent_q_vals.shape}")
    print(f"Dummy agent_h_states shape: {dummy_agent_h_states.shape}")
    
    q_total = mixer_nn(dummy_agent_q_vals, dummy_global_raw_state_batch, dummy_agent_h_states)
    
    print(f"\nOutput Q_total shape: {q_total.shape}")
    print("Mixer test successful if shapes are correct.")
    print("Q_total example:", q_total)