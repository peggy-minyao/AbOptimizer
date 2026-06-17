import os
from typing import List
from deeplog import Config, DeepLog
import yaml
import random
import numpy as np
import torch
import argparse

from Abopt.agent import Agent, AgentEnv, TransformerNet
from Abopt.env import Environment, Interpreter, Evaluator
from Abopt.env import (
    Affinity, Specificity, Her2Specificity, MHCIIAffinity, FvCSP, FvNetCharge, HISum
)
from Abopt.data import Ab4000SeqUtils, SeqDataLoader, CDR3SeqUtils
from Abopt import PPO



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

set_seed(2024)

class PPOOptimizer:
    def __init__(self, antibody, task='co_optimization'):
        self.antibody = antibody
        self.task = task
    
    def optimize_sequence(self, sequence: str, config_name: str, output_dir: str, device: str) -> str:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config = self.load_config(config_name)
        config.seq = sequence
        logger = DeepLog(save_path = output_dir)

        if self.antibody == 'emi':
            seq_utils = Ab4000SeqUtils()
        elif  self.antibody == 'tras':
            seq_utils = CDR3SeqUtils()
        else:
            print("error: invalid antibody type")

        data_loader = SeqDataLoader([sequence] * config.batch_size, config.batch_size)
        agent = Agent(
            policy=TransformerNet(n_action=config.n_action),
            use_GPU=config.use_GPU,
            device=device,
        )
        agent_env = AgentEnv(
            policy=TransformerNet(n_action=config.n_action),
            use_GPU=config.use_GPU,
            device=device,
        )

        interpreter = Interpreter(
            seq_utils=seq_utils,
            device=device,
            interpretive_method=config.interpretive_method,
        )

        if self.antibody == 'emi':
            affinity = Affinity(model_path="./Abopt/env/hgfr_affinity.pt", device=device, seq_utils=seq_utils)
            specificity = Specificity(model_path="./Abopt/env/hgfr_specificity.pt", device=device, seq_utils=seq_utils)
        elif  self.antibody == 'tras':
            affinity = Her2Specificity(model_path="./Abopt/env/variable_her2_spec.pt", device=device, seq_utils=seq_utils)
            specificity = None
        else:
            print("error: invalid antibody type")

        mhc = MHCIIAffinity(seq_utils=seq_utils)
        fvnet = FvNetCharge(seq_utils=seq_utils)
        fvcsp = FvCSP(seq_utils=seq_utils)
        hisum = HISum(seq_utils=seq_utils)

        if self.antibody == 'emi':
            task_map = {
                "affinity_opt": (
                    {"affinity": affinity},
                    [1.0]
                ),
                "co_opt": (
                    {"affinity": affinity, "specificity": specificity},
                    [1.0, 1.0]
                ),
                "multiple_opt": (
                    {
                        "affinity": affinity,
                        "specificity": specificity,
                        "mhc": mhc,
                        "fvnet": fvnet,
                        "fvcsp": fvcsp,
                        "hisum": hisum,
                    },
                    [1.0] * 6
                ),
            }

        elif self.antibody == 'tras':
            task_map = {
                "affinity_opt": (
                    {"affinity": affinity},
                    [1.0]
                ),
                "multiple_opt": (
                    {
                        "affinity": affinity,
                        "mhc": mhc,
                        "fvnet": fvnet,
                        "fvcsp": fvcsp,
                        "hisum": hisum,
                    },
                    [1.0] * 5
                ),
            }

        if self.task not in task_map:
            raise ValueError(f"Unknown task type: {self.task}")

        score_funcs, weights = task_map[self.task]

        evaluator = Evaluator(
            score_functions=score_funcs,
            function_weights=weights,
            use_dynamic_weights=config.use_dynamic_weights,
            use_non_dominated_sort=config.use_non_dominated_sort,
        )

        
        environment = Environment(interpreter=interpreter, evaluator=evaluator)

        ppo = PPO(
            agent=agent,
            agent_env=agent_env,
            environment=environment,
            logger=logger,
        )
        ppo.train_agent(
            data_loader=data_loader,
            epoch=config.epoch,
            lr=config.lr,
            batch_size=config.batch_size,
            scheduler_step=config.scheduler_step,
            scheduler_gamma=config.scheduler_gamma,
            epsilon=config.epsilon,
            mini_epoch=config.mini_epoch,
            n_data_sample=config.n_data_sample,
            weighted_samples=config.weighted_samples,
            threshold=config.threshold,
        )


        actions_probs, properties = ppo.inference(sequence, 1)

        properties.to_csv(output_dir + config.seq + ".csv")
        print("actions_probs:", actions_probs)
        print(properties.head(10))
        optimal_sequence = properties["opt_sequence"][0]

        logger.log("actions_probs", actions_probs.tolist())
        logger.log_file_name = config.seq + ".txt"
        logger.save(config)
        save_model_path = output_dir + 'log/' + config.seq + config.save_model_path + ".pth"
        ppo.save_model(save_model_path)        

        return optimal_sequence

    def optimize_sequences(self, sequences: List[str], config_name: str) -> List[str]:
        optimal_sequences = [
            self.optimize_sequence(sequence, config_name) for sequence in sequences
        ]

        return optimal_sequences

    @staticmethod
    def load_config(file_name: str) -> Config:
        config = Config()
        with open(file_name, "r") as fp:
            config_d = yaml.load(fp, yaml.FullLoader)
        config.from_dict(config_d)
        config.print_parameters()
        return config



import csv

def main(input_sequence, task, output_dir='./output_files_tras_multi_properties/', config_path = "config.yaml", antibody= 'emi', device='cuda:0'):
    ppo_optimizer = PPOOptimizer(antibody, task=task)
    opt_seqs = ppo_optimizer.optimize_sequence(input_sequence, config_path, output_dir, device)
    print(f"Optimized sequence for {input_sequence}: {opt_seqs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AbOptimizer with specified antibody.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--antibody", type=str, choices=['emi', 'tras'], default='emi', help="Specify the antibody.")
    parser.add_argument("--index", type=str, default='0', help="Which sequence to run (number or 'all'). Default is 0 (first row).")
    parser.add_argument("--output_dir", type=str, default='./output_files_emi_co_properties/', help="The output data dir ")
    parser.add_argument("--input_dir", type=str, default='./starting_data/co-optimization_emi_data.csv', help="The input data dir ")
    parser.add_argument("--task", type=str, choices=['co_opt', 'affinity_opt','multiple_opt'], default='co_opt', help="The task type.")
    args = parser.parse_args()
    
    device = 'cuda:0'
    with open(args.input_dir, mode='r') as file:
        csv_reader = list(csv.DictReader(file))
        total_rows = len(csv_reader)

        if args.index.lower() == 'all':
            rows_to_run = range(total_rows)
        else:
            try:
                idx = int(args.index)
                if idx < 0 or idx >= total_rows:
                    raise ValueError(f"Index {idx} is out of range (0 to {total_rows-1}).")
                rows_to_run = [idx]
            except ValueError:
                raise ValueError(f"Invalid index '{args.index}'. It should be an integer or 'all'.")

        for i in rows_to_run:
            input_sequence = csv_reader[i]['AASeq']
            print(f"Running index {i}: {input_sequence[:10]}...")
            main(input_sequence, args.task, output_dir=args.output_dir, config_path=args.config, antibody= args.antibody, device=device)
