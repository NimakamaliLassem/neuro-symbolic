import torch
from trm import TRM
from recursive_trm import RecursiveTRM, HierarchicalRecursiveTRM

class Generator(torch.nn.Module):
    def __init__(self, graph, num_layers, embedding_dim, hidden_dim, rnn_type='gru',
                 num_refinement_steps=3, H_cycles=2, L_cycles=3):
        super(Generator, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()

        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

        # Select RNN type
        input_size = self.embedding_dim * 2
        if self.rnn_type == 'recursive_trm':
            # TRUE recursive reasoning - refines predictions multiple times per timestep
            # Best for compositional logical rules (e.g., grandfather ← father, father)
            self.rnn = RecursiveTRM(input_size, self.hidden_dim, self.num_layers,
                                   num_refinement_steps=num_refinement_steps, batch_first=True)
        elif self.rnn_type == 'hierarchical_trm':
            # Two-level recursive reasoning (H and L cycles like TRM paper)
            self.rnn = HierarchicalRecursiveTRM(input_size, self.hidden_dim, self.num_layers,
                                                H_cycles=H_cycles, L_cycles=L_cycles, batch_first=True)
        elif self.rnn_type == 'trm':
            # Simple parameter-efficient RNN (minimal gating, no recursive reasoning)
            self.rnn = TRM(input_size, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = torch.nn.GRU(input_size, self.hidden_dim, self.num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}. Choose from 'recursive_trm', 'hierarchical_trm', 'trm', 'lstm', or 'gru'.")

        self.linear = torch.nn.Linear(self.hidden_dim, self.label_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, relation, hidden):
        # Handle tuple (h, c) from LSTM-style trainer
        # Only LSTM needs (h, c), all others just need h
        if isinstance(hidden, tuple) and self.rnn_type != 'lstm':
            hidden = hidden[0]
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)
        outputs, hidden = self.rnn(embedding, hidden)
        logits = self.linear(outputs)
        return logits, hidden

    def loss(self, inputs, target, mask, weight, hidden):
        logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss

    