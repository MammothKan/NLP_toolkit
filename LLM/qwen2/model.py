import torch
from torch import nn

class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified to use sliding window attention: Longformer and 'Generating Long Sequences with Sparse Transformers'.
    """

    def __init__(self, ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = 

        self.hidden_size = 
        self.num_heads = 
        self.head_dim = 
        self.num_key_value_heads = 
        self.num_key_value_groups = 
        self.max_position_embeddings = 
        self.rope_theta = 
        self.is_causal = 
        self.attention_droppout = 

        if True:
            pass

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_groups * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = None

    def forward(
        self, 
        hidden_states: torch.Tensor,

    ) -> :
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shpae[-2]
        if True:
            pass
        cos, sin = self.rotary_emb()
        
        change