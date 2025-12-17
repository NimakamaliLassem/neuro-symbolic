"""
RL-Guided Rule Generator.

Replaces EM loop with value-guided search:
1. Sample many candidate rules from generator
2. Use value network to quickly estimate rule quality
3. Select top rules based on value predictions
4. Train predictor only with high-value rules
5. Update value network with actual H-scores
6. Repeat for fewer iterations than EM
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from collections import defaultdict

from value_network import ValueNetwork, ValueNetworkTrainer
from data import RuleDataset
from trainer import TrainerPredictor


class RLGuidedRuleGenerator:
    """
    RL-guided rule generation that replaces EM iterations.

    Key insight: Use a learned value function to quickly estimate
    rule quality, avoiding expensive predictor training on bad rules.
    """

    def __init__(
        self,
        generator,
        graph,
        num_relations,
        embedding_dim,
        hidden_dim,
        value_hidden_dim=256,
        device=None
    ):
        self.generator = generator
        self.graph = graph
        self.num_relations = num_relations
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize value network
        self.value_network = ValueNetwork(
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            hidden_dim=value_hidden_dim,
            num_layers=2,
            dropout=0.1
        ).to(self.device)

        self.value_trainer = ValueNetworkTrainer(
            self.value_network,
            device=self.device,
            lr=1e-3
        )

        # Configuration
        self.num_candidate_rules = 4000  # Sample many candidates
        self.num_selected_rules = 1500   # Keep more rules (similar to EM)
        self.value_train_epochs = 30    # Epochs to train value network

        # Generator training configuration (M-step)
        # If <= 1.0: fraction (e.g., 1.0 = all, 0.5 = top 50%)
        # If > 1.0: absolute count (e.g., 2000 = top 2000 rules)
        self.generator_rule_fraction = 1.0  # CHANGED: was implicitly 0.5, now matches EM

        # Value network filtering
        # Set to True to use value network for rule selection (RL behavior)
        # Set to False to disable value network and match EM exactly
        self.use_value_filtering = True

        # Value network target
        # Set to True to predict posterior (H + prior), False to predict H-score only
        # Posterior prediction may better align with actual optimization objective
        self.predict_posterior = False  # Default: predict H-score (original behavior)

        # History for replay
        self.rule_history = []
        self.score_history = []
        self.prior_history = []  # Store priors when predict_posterior=True

    def initialize_value_network(self, initial_rules, predictor_class, graph_data, train_set, cfg):
        """
        Warm-start the value network using initial rules.

        Args:
            initial_rules: Pre-sampled rules from generator
            predictor_class: Predictor class to instantiate
            graph_data: Knowledge graph
            train_set: Training dataset
            cfg: Configuration object
        """
        logging.info('Initializing value network with warm-start...')

        # Train predictor briefly to get initial H-scores
        predictor = predictor_class(graph_data, **cfg.predictor.model)
        predictor.set_rules(initial_rules)
        optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)

        # Use a dummy validation/test set for initialization
        solver_p = TrainerPredictor(
            predictor, train_set, train_set, train_set,
            optim, gpus=cfg.predictor.gpus
        )
        solver_p.train(**cfg.predictor.train)

        # Get H-scores
        h_scores = solver_p.compute_H(**cfg.predictor.H_score)

        # Get priors for posterior computation
        # Note: initial_rules don't have priors attached, so we approximate with 0
        # This is just for warm-start; real training will use actual priors
        initial_priors = [0.0] * len(initial_rules)  # Placeholder priors

        # Compute target scores based on mode
        if self.predict_posterior:
            target_scores = [h + p * cfg.EM.prior_weight for h, p in zip(h_scores, initial_priors)]
            logging.info('Value network will predict POSTERIOR scores (H + prior)')
        else:
            target_scores = h_scores
            logging.info('Value network will predict H-SCORES only')

        # Store in history
        self.rule_history.extend(initial_rules)
        self.score_history.extend(target_scores)
        if self.predict_posterior:
            self.prior_history.extend(initial_priors)

        # Train value network
        logging.info(f'Training value network on {len(initial_rules)} rules...')
        self.value_trainer.train(
            initial_rules, target_scores,
            num_epochs=self.value_train_epochs,
            batch_size=256,
            verbose=True
        )

        logging.info('Value network initialization complete.')

    def sample_candidates(self, trainer_generator, num_samples, max_length):
        """
        Sample candidate rules from the generator.

        Returns:
            List of rules (without scores)
        """
        sampled = trainer_generator.sample(num_samples, max_length)
        # Remove the log probability score at the end
        rules = [rule[:-1] for rule in sampled]
        prior_scores = [rule[-1] for rule in sampled]
        return rules, prior_scores

    def select_rules_by_value(self, rules, prior_scores, top_k, prior_weight=0.0):
        """
        Select top-k rules based on value network predictions.

        Args:
            rules: List of candidate rules
            prior_scores: Generator's prior scores for each rule
            top_k: Number of rules to select
            prior_weight: Weight for prior contribution (default: 0.0)

        Returns:
            Selected rules, their value predictions, and prior scores
        """
        # Get value predictions
        value_predictions = self.value_trainer.predict(rules)

        # Combine value prediction with prior (weighted)
        if self.predict_posterior:
            # Value network already predicts posterior, use directly
            # (priors were already incorporated during training)
            combined_scores = value_predictions
        else:
            # Value network predicts H-score, add prior manually
            combined_scores = [
                v + prior_weight * p
                for v, p in zip(value_predictions, prior_scores)
            ]

        # Sort and select top-k
        indexed_scores = list(enumerate(combined_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed_scores[:top_k]]

        selected_rules = [rules[i] for i in top_indices]
        selected_values = [value_predictions[i] for i in top_indices]
        selected_priors = [prior_scores[i] for i in top_indices]

        return selected_rules, selected_values, selected_priors

    def update_value_network(self, rules, h_scores, priors=None, prior_weight=0.0):
        """
        Update value network with new rule-score pairs.

        Args:
            rules: List of rules
            h_scores: H-scores for each rule
            priors: Prior scores (required if predict_posterior=True)
            prior_weight: Weight for prior in posterior computation
        """
        # Compute target scores based on mode
        if self.predict_posterior:
            if priors is None:
                raise ValueError("priors required when predict_posterior=True")
            target_scores = [h + p * prior_weight for h, p in zip(h_scores, priors)]
        else:
            target_scores = h_scores

        # Add to history
        self.rule_history.extend(rules)
        self.score_history.extend(target_scores)
        if self.predict_posterior:
            self.prior_history.extend(priors)

        # Use recent history for training (avoid stale data)
        max_history = 10000
        if len(self.rule_history) > max_history:
            self.rule_history = self.rule_history[-max_history:]
            self.score_history = self.score_history[-max_history:]
            if self.predict_posterior:
                self.prior_history = self.prior_history[-max_history:]

        # Retrain value network
        self.value_trainer.train(
            self.rule_history, self.score_history,
            num_epochs=self.value_train_epochs,
            batch_size=256,
            verbose=False
        )

    def train_generator_on_rules(self, trainer_generator, rules, scores, cfg):
        """
        Update generator on high-quality rules (M-step equivalent).
        """
        # Attach scores to rules
        rules_with_scores = [rule + [score] for rule, score in zip(rules, scores)]

        # Create dataset
        dataset = RuleDataset(self.num_relations, rules_with_scores)

        # Train generator
        trainer_generator.train(dataset, **cfg.generator.train)

    def train(
        self,
        train_set,
        valid_set,
        test_set,
        predictor_class,
        cfg,
        trainer_generator,
        num_iterations=3,
        log_dir=None
    ):
        """
        Main RL-guided training loop.

        Args:
            train_set, valid_set, test_set: Data splits
            predictor_class: Predictor class
            cfg: Configuration
            trainer_generator: TrainerGenerator instance
            num_iterations: Number of RL iterations (typically 2-3)
            log_dir: Directory for logs

        Returns:
            Final rules, all evaluated rules, all H-scores
        """
        all_evaluated_rules = []
        all_h_scores = []
        all_posterior_scores = []  # ADDED: Store posterior scores like EM does
        best_test_mrr = 0.0

        for iteration in range(num_iterations):
            logging.info('=' * 60)
            logging.info(f'| RL-Guided Iteration {iteration + 1}/{num_iterations}')
            logging.info('=' * 60)

            # Step 1: Sample many candidate rules
            # Sample more per relation to get enough diversity
            samples_per_relation = max(100, self.num_candidate_rules // self.num_relations)
            logging.info(f'Sampling ~{samples_per_relation * self.num_relations} candidate rules...')
            candidate_rules, prior_scores = self.sample_candidates(
                trainer_generator,
                samples_per_relation,
                cfg.EM.max_length
            )
            logging.info(f'Generated {len(candidate_rules)} candidate rules')

            # Step 2: Select rules (with or without value network)
            if self.use_value_filtering and iteration > 0:
                # RL mode: Use value network to select top rules
                logging.info(f'Selecting top {self.num_selected_rules} rules by value prediction...')
                selected_rules, selected_values, selected_priors = self.select_rules_by_value(
                    candidate_rules, prior_scores, self.num_selected_rules, prior_weight=cfg.EM.prior_weight
                )
                logging.info(f'Selected {len(selected_rules)} rules (value-filtered)')
            else:
                # EM mode: Use ALL sampled rules (no filtering, matches EM exactly)
                if not self.use_value_filtering:
                    # Pure EM: use all sampled rules
                    selected_rules = candidate_rules
                    selected_priors = prior_scores
                    logging.info(f'Using ALL {len(selected_rules)} rules (EM mode - no value filtering)')
                else:
                    # First iteration: select top N since value network not ready yet
                    selected_rules = candidate_rules[:self.num_selected_rules]
                    selected_priors = prior_scores[:self.num_selected_rules]
                    logging.info(f'Using {len(selected_rules)} rules (first iteration - value network not ready)')

            # Step 3: Train predictor with selected rules
            logging.info('Training predictor with selected rules...')
            predictor = predictor_class(self.graph, **cfg.predictor.model)
            predictor.set_rules(selected_rules)
            optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)

            solver_p = TrainerPredictor(
                predictor, train_set, valid_set, test_set,
                optim, gpus=cfg.predictor.gpus
            )
            solver_p.train(**cfg.predictor.train)

            # Evaluate
            valid_mrr = solver_p.evaluate('valid', expectation=cfg.predictor.eval.expectation)['mrr']
            test_mrr = solver_p.evaluate('test', expectation=cfg.predictor.eval.expectation)['mrr']

            if test_mrr > best_test_mrr:
                best_test_mrr = test_mrr

            # Step 4: Compute actual H-scores
            logging.info('Computing H-scores...')
            h_scores = solver_p.compute_H(**cfg.predictor.H_score)

            # Compute posterior (combine H-score with prior)
            posterior_scores = [
                h + p * cfg.EM.prior_weight
                for h, p in zip(h_scores, selected_priors)
            ]

            # Store rules with their scores (like EM does)
            all_evaluated_rules.extend(selected_rules)
            all_h_scores.extend(h_scores)
            all_posterior_scores.extend(posterior_scores)  # ADDED: Store posterior like EM

            # Step 5: Update value network with actual H-scores (or posteriors)
            logging.info('Updating value network...')
            self.update_value_network(
                selected_rules, h_scores,
                priors=selected_priors,
                prior_weight=cfg.EM.prior_weight
            )

            # Step 6: Update generator (M-step)
            logging.info('Updating generator on top rules...')

            # Sort by posterior and filter by configured fraction/count
            rule_posteriors = list(zip(selected_rules, posterior_scores))
            rule_posteriors.sort(key=lambda x: x[1], reverse=True)

            # Determine number of rules: fraction if <= 1.0, absolute count if > 1.0
            if self.generator_rule_fraction <= 1.0:
                num_rules_for_generator = max(1, int(len(rule_posteriors) * self.generator_rule_fraction))
                mode = f"fraction={self.generator_rule_fraction}"
            else:
                num_rules_for_generator = min(int(self.generator_rule_fraction), len(rule_posteriors))
                mode = f"top {int(self.generator_rule_fraction)}"

            top_rules = [r for r, _ in rule_posteriors[:num_rules_for_generator]]
            top_scores = [s for _, s in rule_posteriors[:num_rules_for_generator]]

            logging.info(f'Training generator on {len(top_rules)}/{len(selected_rules)} rules ({mode})')
            self.train_generator_on_rules(trainer_generator, top_rules, top_scores, cfg)

            logging.info(f'Iteration {iteration + 1} complete. Best Test MRR: {best_test_mrr:.6f}')

        # Final: Return accumulated rules with their scores
        # FIXED: Use posterior scores (H + prior) like EM does, not just H-scores
        final_rules = []
        for rule, score in zip(all_evaluated_rules, all_posterior_scores):
            final_rules.append(rule + [score])

        return final_rules, all_evaluated_rules, all_h_scores


def create_rl_guide(generator, graph, cfg, device, generator_rule_fraction=1.0, use_value_filtering=True, predict_posterior=False):
    """
    Factory function to create RLGuidedRuleGenerator.

    Args:
        generator_rule_fraction: Rules to use for generator training (M-step)
                                 If <= 1.0: treated as fraction (1.0 = all, 0.5 = top 50%)
                                 If > 1.0: treated as absolute count (2000 = top 2000 rules)
                                 Examples:
                                   1.0 = use all rules (matches EM behavior)
                                   0.5 = use top 50% only
                                   2000 = use top 2000 rules
        use_value_filtering: Whether to use value network for rule selection
                            True = RL mode (use value network)
                            False = EM mode (no filtering, just sample)
        predict_posterior: Whether value network predicts posterior (H + prior) vs H-score only
                          True = predict posterior (better alignment with optimization)
                          False = predict H-score (original behavior)
    """
    rl_guide = RLGuidedRuleGenerator(
        generator=generator,
        graph=graph,
        num_relations=graph.relation_size,
        embedding_dim=cfg.generator.model.embedding_dim,
        hidden_dim=cfg.generator.model.hidden_dim,
        value_hidden_dim=256,
        device=device
    )
    rl_guide.generator_rule_fraction = generator_rule_fraction
    rl_guide.use_value_filtering = use_value_filtering
    rl_guide.predict_posterior = predict_posterior
    return rl_guide
