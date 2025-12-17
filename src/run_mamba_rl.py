"""
Mamba with RL-Guided Rule Generation.

Replaces EM iterations with value-function guided search for the Mamba generator.
"""
import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset
from predictors import Predictor, PredictorPlus
from generators_mamba import Generator
from utils import load_config, save_config, set_logger, set_seed
from trainer_mamba import TrainerPredictor, TrainerGenerator
from value_network import ValueNetwork, ValueNetworkTrainer
from rl_guided_generator import RLGuidedRuleGenerator, create_rl_guide
import comm


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Mamba with RL-Guided Search',
        usage='run_rnnlogic_rl.py [<args>] [-h | --help]'
    )
    parser.add_argument('--config', default='../config/kinship.yaml', type=str)
    parser.add_argument("--local_rank", type=int, default=0)

    # RL-guided parameters
    parser.add_argument('--num_rl_iterations', type=int, default=5,
                        help='Number of RL-guided iterations (default: 5, same as EM)')
    parser.add_argument('--num_candidates', type=int, default=4000,
                        help='Number of candidate rules to sample per iteration')
    parser.add_argument('--num_selected', type=int, default=1500,
                        help='Number of top rules to select by value prediction')
    parser.add_argument('--use_em_fallback', action='store_true',
                        help='Fall back to original EM if RL fails')

    return parser.parse_args(args)


def main(args):
    cfgs = load_config(args.config)
    cfg = cfgs[0]

    if cfg.save_path is None:
        cfg.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))

    if cfg.save_path and not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)

    save_config(cfg, cfg.save_path)
    set_logger(cfg.save_path)
    set_seed(cfg.seed)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cfg.generator.gpu}')
    else:
        device = torch.device('cpu')

    graph = KnowledgeGraph(cfg.data.data_path)
    train_set = TrainDataset(graph, cfg.data.batch_size)
    valid_set = ValidDataset(graph, cfg.data.batch_size)
    test_set = TestDataset(graph, cfg.data.batch_size)

    dataset = RuleDataset(graph.relation_size, cfg.data.rule_file)

    # =========================================
    # Step 1: Pre-train Generator (Mamba)
    # =========================================
    if comm.get_rank() == 0:
        logging.info('=' * 60)
        logging.info('| RL-GUIDED RULE GENERATION (MAMBA)')
        logging.info('=' * 60)
        logging.info('| Step 1: Pre-train Generator')
        logging.info('=' * 60)

    generator = Generator(graph, **cfg.generator.model)
    solver_g = TrainerGenerator(generator, gpu=cfg.generator.gpu)
    solver_g.train(dataset, **cfg.generator.pre_train)

    # =========================================
    # Step 2: Initialize RL-Guided Generator
    # =========================================
    if comm.get_rank() == 0:
        logging.info('=' * 60)
        logging.info('| Step 2: Initializing RL-Guided Generator')
        logging.info('=' * 60)

    rl_guide = create_rl_guide(generator, graph, cfg, device)

    # Configure RL guide parameters
    rl_guide.num_candidate_rules = args.num_candidates
    rl_guide.num_selected_rules = args.num_selected

    # Warm-start value network with initial samples
    if comm.get_rank() == 0:
        logging.info('Sampling initial rules for warm-start...')

    initial_rules = solver_g.sample(cfg.EM.num_rules, cfg.EM.max_length)
    initial_rules_clean = [rule[:-1] for rule in initial_rules]

    if comm.get_rank() == 0:
        logging.info(f'Warm-starting value network with {len(initial_rules_clean)} rules...')

    rl_guide.initialize_value_network(
        initial_rules=initial_rules_clean,
        predictor_class=Predictor,
        graph_data=graph,
        train_set=train_set,
        cfg=cfg
    )

    # =========================================
    # Step 3: RL-Guided Training Loop
    # =========================================
    if comm.get_rank() == 0:
        logging.info('=' * 60)
        logging.info('| Step 3: RL-Guided Training')
        logging.info('=' * 60)

    final_rules, all_eval_rules, all_h_scores = rl_guide.train(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        predictor_class=Predictor,
        cfg=cfg,
        trainer_generator=solver_g,
        num_iterations=args.num_rl_iterations,
        log_dir=cfg.save_path
    )

    # =========================================
    # Step 4: Post-train Generator (optional)
    # =========================================
    if final_rules:
        if comm.get_rank() == 0:
            logging.info('=' * 60)
            logging.info('| Step 4: Post-train Generator')
            logging.info('=' * 60)

        # Use accumulated rules for post-training
        post_dataset = RuleDataset(graph.relation_size, final_rules)
        solver_g.train(post_dataset, **cfg.generator.post_train)

    # =========================================
    # Step 5: Beam Search for Final Rules
    # =========================================
    if comm.get_rank() == 0:
        logging.info('=' * 60)
        logging.info('| Step 5: Beam Search Best Rules')
        logging.info('=' * 60)

    sampled_rules = []
    for num_rules, max_length in zip(cfg.final_prediction.num_rules, cfg.final_prediction.max_length):
        sampled_rules_ = solver_g.beam_search(num_rules, max_length)
        sampled_rules += sampled_rules_

    prior = [rule[-1] for rule in sampled_rules]
    rules = [rule[0:-1] for rule in sampled_rules]

    if comm.get_rank() == 0:
        logging.info(f'Generated {len(rules)} final rules via beam search')

    # =========================================
    # Step 6: Train Final Predictor+
    # =========================================
    if comm.get_rank() == 0:
        logging.info('=' * 60)
        logging.info('| Step 6: Train Final Predictor+')
        logging.info('=' * 60)

    predictor = PredictorPlus(graph, **cfg.predictorplus.model)
    predictor.set_rules(rules)
    optim = torch.optim.Adam(predictor.parameters(), **cfg.predictorplus.optimizer)

    solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, gpus=cfg.predictorplus.gpus)
    best_valid_mrr = 0.0
    test_mrr = {'mrr': 0.0, 'mr': 0.0, 'hit1': 0.0, 'hit3': 0.0, 'hit10': 0.0}

    for k in range(cfg.final_prediction.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info(f'| Final Iteration: {k + 1}/{cfg.final_prediction.num_iters}')
            logging.info('-------------------------')

        solver_p.train(**cfg.predictorplus.train)
        valid_metrics = solver_p.evaluate('valid', expectation=cfg.predictorplus.eval.expectation)
        test_metrics = solver_p.evaluate('test', expectation=cfg.predictorplus.eval.expectation)

        if valid_metrics['mrr'] > best_valid_mrr:
            best_valid_mrr = valid_metrics['mrr']
            test_mrr = test_metrics
            solver_p.save(os.path.join(cfg.save_path, 'predictor.pt'))

    if comm.get_rank() == 0:
        logging.info('=' * 60)
        logging.info('| FINAL TEST METRICS')
        logging.info('=' * 60)
        logging.info('| MRR:  {:.6f}'.format(test_mrr['mrr']))
        logging.info('| MR:   {:.6f}'.format(test_mrr['mr']))
        logging.info('| H@1:  {:.6f}'.format(test_mrr['hit1']))
        logging.info('| H@3:  {:.6f}'.format(test_mrr['hit3']))
        logging.info('| H@10: {:.6f}'.format(test_mrr['hit10']))
        logging.info('=' * 60)

        # Log summary
        logging.info('')
        logging.info('Summary:')
        logging.info(f'  - (MAMBA RL)')
        logging.info(f'  - RL iterations: {args.num_rl_iterations}')
        logging.info(f'  - Final MRR: {test_mrr["mrr"]:.6f}')


if __name__ == '__main__':
    main(parse_args())
