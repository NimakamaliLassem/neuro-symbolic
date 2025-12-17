"""
Value Network for RL-Guided Rule Generation.

Predicts the quality (H-score) of a rule without running full predictor training.
This enables fast rule quality estimation for guided search.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class ValueNetwork(nn.Module):
    """
    Predicts H-score (rule quality) from rule representation.

    Architecture:
    - Embedding layer for relations
    - Bidirectional GRU to encode rule sequence
    - MLP to predict scalar value (H-score proxy)
    """

    def __init__(self, num_relations, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.vocab_size = num_relations + 2  # +1 for END, +1 for PAD
        self.padding_idx = num_relations + 1
        self.ending_idx = num_relations

        # Embedding for relations
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.padding_idx)

        # Bidirectional GRU to encode rule sequence
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # MLP head for value prediction
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, rules):
        """
        Forward pass to predict value for each rule.

        Args:
            rules: Tensor of shape (batch_size, max_rule_len) containing relation indices

        Returns:
            values: Tensor of shape (batch_size,) containing predicted H-scores
        """
        # Embed rules
        embedded = self.embedding(rules)  # (batch, seq_len, embed_dim)

        # Encode with GRU
        # Get lengths for packing (exclude padding)
        mask = (rules != self.padding_idx)
        lengths = mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)  # Ensure at least 1

        # Pack, encode, unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)  # hidden: (num_layers*2, batch, hidden_dim)

        # Concatenate final hidden states from both directions
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        # Take the last layer's hidden states
        hidden_fwd = hidden[-2]  # Forward direction
        hidden_bwd = hidden[-1]  # Backward direction
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=-1)  # (batch, hidden_dim * 2)

        # Predict value
        values = self.value_head(hidden_cat).squeeze(-1)  # (batch,)

        return values

    def encode_rules(self, rules):
        """
        Get rule embeddings (for similarity-based operations).

        Args:
            rules: Tensor of shape (batch_size, max_rule_len)

        Returns:
            embeddings: Tensor of shape (batch_size, hidden_dim * 2)
        """
        embedded = self.embedding(rules)
        mask = (rules != self.padding_idx)
        lengths = mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)

        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=-1)

        return hidden_cat


class ValueNetworkTrainer:
    """
    Trainer for the value network.
    """

    def __init__(self, value_network, device, lr=1e-3, weight_decay=1e-5):
        self.model = value_network
        self.device = device
        self.optimizer = torch.optim.Adam(
            value_network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.MSELoss()

    def rules_to_tensor(self, rules, max_len=None):
        """
        Convert list of rules to padded tensor.

        Args:
            rules: List of rules, each rule is [head_rel, rel1, rel2, ...]
            max_len: Maximum rule length (for padding)

        Returns:
            Tensor of shape (num_rules, max_len)
        """
        if max_len is None:
            max_len = max(len(r) for r in rules)

        padding_idx = self.model.padding_idx

        padded_rules = []
        for rule in rules:
            padded = list(rule) + [padding_idx] * (max_len - len(rule))
            padded_rules.append(padded[:max_len])

        return torch.tensor(padded_rules, dtype=torch.long, device=self.device)

    def train_epoch(self, rules, h_scores, batch_size=256):
        """
        Train for one epoch.

        Args:
            rules: List of rules
            h_scores: List of H-scores (ground truth)
            batch_size: Batch size

        Returns:
            Average loss for the epoch
        """
        self.model.train()

        # Convert to tensors
        rules_tensor = self.rules_to_tensor(rules)
        scores_tensor = torch.tensor(h_scores, dtype=torch.float32, device=self.device)

        # Normalize scores for stable training
        scores_mean = scores_tensor.mean()
        scores_std = scores_tensor.std() + 1e-8
        scores_normalized = (scores_tensor - scores_mean) / scores_std

        # Shuffle
        indices = torch.randperm(len(rules))

        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(rules), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_rules = rules_tensor[batch_idx]
            batch_scores = scores_normalized[batch_idx]

            self.optimizer.zero_grad()

            predictions = self.model(batch_rules)
            loss = self.criterion(predictions, batch_scores)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.scheduler.step(avg_loss)

        return avg_loss

    def train(self, rules, h_scores, num_epochs=50, batch_size=256, patience=10, verbose=True):
        """
        Full training loop with early stopping.
        """
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            loss = self.train_epoch(rules, h_scores, batch_size)

            if verbose and (epoch + 1) % 10 == 0:
                logging.info(f'Value Network Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}')

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logging.info(f'Early stopping at epoch {epoch+1}')
                    break

        return best_loss

    @torch.no_grad()
    def predict(self, rules):
        """
        Predict H-scores for rules.

        Args:
            rules: List of rules

        Returns:
            List of predicted H-scores
        """
        self.model.eval()

        if len(rules) == 0:
            return []

        rules_tensor = self.rules_to_tensor(rules)
        predictions = self.model(rules_tensor)

        return predictions.cpu().numpy().tolist()

    @torch.no_grad()
    def rank_rules(self, rules, top_k=None):
        """
        Rank rules by predicted value and return top-k.

        Args:
            rules: List of rules
            top_k: Number of top rules to return (None = all)

        Returns:
            Sorted list of (rule, predicted_score) tuples
        """
        self.model.eval()

        if len(rules) == 0:
            return []

        predictions = self.predict(rules)

        # Sort by predicted score (descending)
        rule_scores = list(zip(rules, predictions))
        rule_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            rule_scores = rule_scores[:top_k]

        return rule_scores
