import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class GeneratorGPT2(nn.Module):
    def __init__(self, graph, num_layers, embedding_dim, hidden_dim):
        super(GeneratorGPT2, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim # GPT-2 d_model is the hidden dim in GPT-2
        # hidden_dim is effectively d_model in GPT-2
        
        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.d_model = embedding_dim
        # Ensure n_head divides d_model. Defaulting to 4 or 8 if possible.
        self.n_head = 8 if self.d_model % 8 == 0 else 4
        if self.d_model % self.n_head != 0:
             self.n_head = 1 
        
        # GPT-2 Configuration
        self.config = GPT2Config(
            vocab_size=self.vocab_size,
            n_embd=self.d_model,
            n_layer=self.num_layers,
            n_head=self.n_head,
            n_inner=self.d_model * 4,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            output_attentions=True, # Critical for visualization
            use_cache=True          # Critical for generation
        )
        
        # GPT2Model handles embedding
        self.transformer = GPT2Model(self.config)
        
        # Output projection
        self.linear = nn.Linear(self.d_model, self.label_size)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, relation, hidden=None):
        """
        inputs: (batch, seq_len) - Token indices
        relation: (batch,) - Query relation indices
        hidden: previous 'past_key_values' (tuple)
        """
        
        # Handling the "relation" context
        # If hidden is None (start of sequence), we prepend the relation.
        
        if hidden is None:
            # First step: Prepend relation to inputs
            relation_seq = relation.unsqueeze(1)
            model_input = torch.cat([relation_seq, inputs], dim=1)
        else:
            # Subsequent steps: Just use inputs
            model_input = inputs
            
        # forward pass
        # GPT2Model returns: last_hidden_state, past_key_values, (hidden_states), (attentions)
        outputs = self.transformer(input_ids=model_input, past_key_values=hidden)
        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        
        # Handle logit slicing for initial step
        if hidden is None:
             # Input was [Rel, T1, T2...]. Output has len N+1.
             # We want to predict [T1, T2...].
             # So we use h_Rel to predict T1, h_T1 to predict T2.
             # We drop the last one? No.
             # Target is [T1, T2].
             # Logits should come from [Rel, T1].
             # So we drop the LAST hidden state which corresponds to T2 (predicting T3).
             logits = self.linear(last_hidden_state[:, :-1, :])
        else:
             logits = self.linear(last_hidden_state)
        
        attentions = outputs.attentions
        
        # Return only logits and hidden state (past_key_values) to be compatible with TrainerGenerator
        return logits, past_key_values

    def loss(self, inputs, target, mask, weight, hidden=None):
        # inputs: (batch, seq_len)
        # target: (batch, seq_len)
        logits, _ = self.forward(inputs, inputs[:, 0], hidden=hidden)
        
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss
