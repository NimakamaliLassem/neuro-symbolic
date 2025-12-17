import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Mamba block with residual connection and layer norm."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm residual connection
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class Generator(torch.nn.Module):
    def __init__(self, graph, num_layers, embedding_dim, hidden_dim):
        super(Generator, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim  # Not used by Mamba, kept for config compatibility

        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        # Input dimension for Mamba (embedding + relation embedding concatenated)
        self.d_model = self.embedding_dim * 2

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

        # Stack of Mamba blocks with residual connections and layer norm
        self.mamba_layers = torch.nn.ModuleList([
            MambaBlock(
                d_model=self.d_model,
                d_state=24,
                d_conv=4,
                expand=2,
                dropout=0.02,  # Further reduced for deeper model
            )
            for _ in range(self.num_layers)
        ])

        # Final layer norm before output projection
        self.final_norm = nn.LayerNorm(self.d_model)

        # Output projection from d_model to label_size
        self.linear = torch.nn.Linear(self.d_model, self.label_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, relation, hidden=None):
        # hidden is ignored - Mamba doesn't use explicit hidden state
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        x = torch.cat([embedding, embedding_r], dim=-1)

        # Pass through each Mamba block (with residual + norm)
        for mamba_block in self.mamba_layers:
            x = mamba_block(x)

        # Final norm before projection
        x = self.final_norm(x)

        # Project to output logits
        logits = self.linear(x)

        # Return None for hidden state (Mamba handles state internally)
        return logits, None

    def loss(self, inputs, target, mask, weight, hidden=None):
        logits, _ = self.forward(inputs, inputs[:, 0], hidden=None)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss
