import torch
import torch.nn as nn
from transformers import TransfoXLConfig, TransfoXLModel

class GeneratorTransformerXL(nn.Module):
    def __init__(self, graph, num_layers, embedding_dim, hidden_dim):
        super(GeneratorTransformerXL, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim # Not strictly used as TransfoXL uses d_model
        
        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.d_model = embedding_dim
        # Ensure n_head divides d_model. Defaulting to 4 or 8 if possible.
        self.n_head = 8 if self.d_model % 8 == 0 else 4
        if self.d_model % self.n_head != 0:
             self.n_head = 1 # Fallback to 1 head if weird dim
        
        # TransfoXL Configuration
        self.config = TransfoXLConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            d_embed=self.d_model,
            n_head=self.n_head,
            d_head=self.d_model // self.n_head,
            d_inner=self.d_model * 4,
            n_layer=self.num_layers,
            mem_len=10, # Short rules, small memory sufficient
            clamp_len=1000,
            same_length=False,
            dropout=0.1,
            dropatt=0.0,
            dropout=0.1,
            dropatt=0.0,
            output_attentions=True
        )
        
        # We use the base model to handle embeddings and layers ourselves if needed, 
        # but TransfoXLModel handles embedding. We just need to ensure correct vocab size.
        self.transformer = TransfoXLModel(self.config)
        
        # Output projection
        self.linear = nn.Linear(self.d_model, self.label_size)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, relation, hidden=None):
        """
        inputs: (batch, seq_len) - Token indices
        relation: (batch,) - Query relation indices
        hidden: previous 'mems' (list of tensors) from TransformerXL
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Embed relation to prepended context? 
        # TransfoXL expects inputs as embeddings or indices.
        # If we just pass indices, we can't easily prepend the *embedding* of the relation 
        # unless we modify the input_ids sequence.
        # Strategy: We will rely on the fact that relation is just another token index.
        # We prepend the relation index to the input sequence.
        
        # However, inputs usually start empty or with start token.
        # If 'inputs' contains the rule body so far.
        
        # Let's verify how Trainer calls this. 
        # Trainer passes `inputs` as current sequence.
        # Ideally we want to Condition on `relation`.
        
        # Standard hack: Prepend relation as a token.
        # But `relation` indices overlap with `inputs` indices (relations).
        # This is fine, since it's just "start with this relation".
        
        # If seq_len is 1 (step-by-step generation):
        # We need to ensure the FIRST step sees the relation.
        # If hidden is None (start of seq), we prepend relation.
        
        if hidden is None:
            # First step: Prepend relation to inputs
            # inputs: (batch, seq_len) 
            # relation: (batch,) -> (batch, 1)
            relation_seq = relation.unsqueeze(1)
            # Combine: [relation, inputs]
            model_input = torch.cat([relation_seq, inputs], dim=1)
        else:
            # Subsequent steps: Just use inputs
            # The relation context is in 'mems' (hidden)
            model_input = inputs
            
        # forward pass
        # TransfoXLModel returns: last_hidden_state, mems
        outputs = self.transformer(input_ids=model_input, mems=hidden)
        last_hidden_state = outputs.last_hidden_state
        mems = outputs.mems
        
        # If we prepended relation (hidden is None), the output length is seq_len + 1.
        # We likely want to discard the position corresponding to the relation strictly for logits 
        # matching 'inputs'. 
        
        if hidden is None:
            # Create logits for the whole sequence except the 'relation' token itself?
            # Or does the trainer expect logits for 'inputs'?
            # Trainer expects logits matching 'inputs' length usually.
            
            # If we pass [rel, token1, token2], output is [h_rel, h_token1, h_token2]
            # Predicting next token:
            # h_rel -> predicts token1
            # h_token1 -> predicts token2
            
            # So actual logits for 'inputs' correspond to positions 0..end.
            # wait, if input is [rel, t1], we want to predict [t1, t2].
            # h_rel predicts t1. h_t1 predicts t2.
            # So we keep all outputs?
            
            # Let's align with standard RNN usage:
            # rnn(inputs) -> output corresponding to each input step.
            # Here, we provided an "extra" start token. 
            pass

        # Project
        logits = self.linear(last_hidden_state)
        
        # If we expanded input, we should probably truncate output to match input length requested?
        # If hidden is None (start), input became length N+1. Output is N+1.
        # Logic: 
        # Input: [Rel, A, B]
        # Logits should predict: [A, B, C]
        # h_Rel -> A
        # h_A -> B
        # h_B -> C
        
        # Original inputs: [A, B] meant we wanted to predict [B, C]? 
        # In `loss` function: inputs, target.
        # Target usually shifted.
        
        # If we just return logits matching `model_input`, let the caller handle slicing?
        # But Trainer expects `logits` to match `inputs`.
        
        if hidden is None:
             # We added 1 token at start.
             # We want logits for the 'inputs' part.
             # h_Rel predicts inputs[0].
             # h_inputs[0] predicts inputs[1].
             # So we actually want the FULL sequence of logits?
             
             # Actually, if we look at `loss` in existing code:
             # logits, _ = self.forward(inputs, inputs[:, 0], hidden=None)
             # inputs usually is full sequence for training.
             
             # If we return logits of shape (batch, seq_len+1, vocab), 
             # and target is (batch, seq_len), we have a mismatch.
             
             # We should probably return logits[:, :-1, :]?
             # No, if input is [Rel, A, B], output is prediction for [A, B, C].
             # We want predictions for [A, B].
             # So we take logits corresponding to [Rel, A].
             # i.e., logits[:, :-1, :]
             
             logits = logits[:, :-1, :]
        
        # attentions tuple (one per layer)
        # TransfoXL returns (last_hidden, mems) usually, or (last_hidden, mems, hidden_states, attentions) if config enabled.
        # Let's check typical HuggingFace output structure.
        # TransfoXLModelOutput: last_hidden_state, mems, hidden_states, attentions
        attentions = outputs.attentions
        
        return logits, mems, attentions

    def loss(self, inputs, target, mask, weight, hidden=None):
        # inputs: (batch, seq_len)
        # target: (batch, seq_len)
        logits, _, _ = self.forward(inputs, inputs[:, 0], hidden=hidden)
        
        # Check shapes
        # forward adds relation -> input is (batch, seq_len+1)
        # returns logits[:, :-1] -> (batch, seq_len)
        # matches target shape (batch, seq_len)
        
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss
