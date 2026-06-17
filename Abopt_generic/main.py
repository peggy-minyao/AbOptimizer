import os
from typing import List
from deeplog import Config, DeepLog
import yaml
import random
import numpy as np
import torch
import pandas as pd
import argparse
from pathlib import Path

from Abopt.agent import Agent, AgentEnv, TransformerNet
from Abopt.env import (
    Environment,
    Interpreter,
    Evaluator,
    MHCIIAffinity,
    FvCSP,
    FvNetCharge,
    HISum,
    FitnessScorer,
)
from Abopt.data import SeqDataLoader, GeneralSeqUtils
from Abopt import PPO


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mutations(wt_seq, opt_seq, chain_tag="H"):
    mutations = []
    for i, (wt, opt) in enumerate(zip(wt_seq, opt_seq), start=1):
        if wt != opt:
            mutations.append(f"{wt}{i}{opt}")
    return "_".join(mutations) if mutations else "WT"


class PPOOptimizer:
    def optimize_sequence(
        self,
        H_chain: str,
        L_chain: str,
        Ag_chian: str,
        chain_type: str,
        config_name: str,
        region: str,
        output_dir: str,
        pdb_path: str,
        cache_path: str,
        device: str,
        task: str = "fitness",
    ) -> str:

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        config = self.load_config(config_name)
        seq_utils = GeneralSeqUtils()
        logger = DeepLog(save_path=output_dir)

        seq_utils.ag_sequence = Ag_chian
        seq_utils.heavy_sequence = H_chain
        seq_utils.light_sequence = L_chain

        sequence = seq_utils.get_edit_sequences([H_chain], [L_chain], region=region)

        logger.log("Hchain", H_chain)
        logger.log("Lchain", L_chain)

        config.seq = sequence

        data_loader = SeqDataLoader(sequence * config.batch_size, config.batch_size)

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
            region=region,
        )

        if region in ["CDRH", "H3", "H2", "H1"]:
            inputchain = H_chain
        else:
            inputchain = L_chain

        fitness = FitnessScorer(
            pdb_path=pdb_path,
            inputchain=inputchain,
            chain=chain_type,
            region=region,
            cache_path=cache_path,
            device=device,
        )

        if task == "fitness":
            score_functions = {
                "fitness": fitness,
            }
            function_weights = [1.0]

        elif task == "multi":
            mhc = MHCIIAffinity(seq_utils=seq_utils)
            fvnet = FvNetCharge(seq_utils=seq_utils)
            fvcsp = FvCSP(seq_utils=seq_utils)
            hisum = HISum(seq_utils=seq_utils)

            score_functions = {
                "fitness": fitness,
                "mhc": mhc,
                "fvnet": fvnet,
                "fvcsp": fvcsp,
                "hisum": hisum,
            }
            function_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        else:
            raise ValueError(
                f"Unsupported task: {task}. "
                "Please choose from ['fitness', 'multi']."
            )

        evaluator = Evaluator(
            score_functions=score_functions,
            function_weights=function_weights,
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

        actions_probs, properties = ppo.inference([sequence], 1)

        if region in ["CDRH", "H3", "H2", "H1"]:
            properties = properties.drop_duplicates("opt_Hsequence", ignore_index=True)
            properties["H_mutation"] = properties["opt_Hsequence"].apply(
                lambda seq: get_mutations(H_chain, seq, chain_tag="H")
            )
        else:
            properties = properties.drop_duplicates("opt_Lsequence", ignore_index=True)
            properties["L_mutation"] = properties["opt_Lsequence"].apply(
                lambda seq: get_mutations(L_chain, seq, chain_tag="L")
            )

        _, wt_df = evaluator([H_chain], [L_chain])
        wt_df["action"] = "WT"
        wt_df["reward"] = "WT"
        wt_df["opt_Hsequence"] = H_chain
        wt_df["opt_Lsequence"] = L_chain

        properties = pd.concat([wt_df, properties], axis=0)

        output_csv_path = os.path.join(output_dir, config.seq[0] + ".csv")
        properties.to_csv(output_csv_path, index=False)

        print("actions_probs:", actions_probs)
        print(properties.head(10))

        if region in ["CDRH", "H3", "H2", "H1"]:
            optimal_sequence = properties["opt_Hsequence"].iloc[0]
        else:
            optimal_sequence = properties["opt_Lsequence"].iloc[0]

        logger.log("actions_probs", actions_probs.tolist())
        logger.log_file_name = config.seq[0] + ".txt"
        logger.save(config)

        save_model_path = os.path.join(
            output_dir,
            "log",
            config.seq[0] + config.save_model_path + ".pth",
        )
        ppo.save_model(save_model_path)

        return optimal_sequence

    def optimize_sequences(self, sequences: List[str], config_name: str) -> List[str]:
        raise NotImplementedError(
            "optimize_sequences is not implemented for the current H/L/Ag-chain workflow."
        )

    @staticmethod
    def load_config(file_name: str) -> Config:
        config = Config()
        with open(file_name, "r") as fp:
            config_d = yaml.load(fp, yaml.FullLoader)
        config.from_dict(config_d)
        config.print_parameters()
        return config


def main():
    parser = argparse.ArgumentParser(
        description="Antibody optimization with generic-specificity surrogate"
    )

    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./pdb/siltuximab.pdb",
        help="Path to the input PDB file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_files_siltuximab/",
        help="Directory to save optimization results.",
    )

    parser.add_argument(
        "--region",
        type=str,
        choices=["CDRH", "H3", "H2", "H1", "CDRL", "L3", "L2", "L1"],
        default="CDRH",
        help="Region to optimize.",
    )

    parser.add_argument(
        "--chain_type",
        type=str,
        default="B",
        help="PDB chain ID used for fitness scoring. Usually A for heavy chain, B for light chain.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run optimization on.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["fitness", "multi"],
        default="fitness",
        help="Task type: fitness for single-property training, multi for multi-property training.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    ppo_optimizer = PPOOptimizer()

    pdb_path = Path(args.pdb_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_name = pdb_path.stem
    cache_path = output_dir / f"{pdb_name}_cache.pkl"

    H_chain = (
        "EVQLVESGGKLLKPGGSLKLSCAASGFTFSSFAMSWFRQSPEKRLEWVAEISSGGSYTYYPDTVTGRFTISRDNAKNTLYLEMSSLRSEDTAMYYCARHLWGYYALDYWGQGTSVTVSS"
    )

    L_chain = (
        "QIVLIQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDTSNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSGYPYTFGGGTKLEIK"
    )

    Ag_chain = (
        "PHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQM"
    )

    print("\n=== PPO Optimization Configuration ===")
    print(f"PDB file:        {pdb_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Cache file:      {cache_path}")
    print(f"Region:          {args.region}")
    print(f"Chain type:      {args.chain_type}")
    print(f"Task:            {args.task}")
    print(f"Config file:     {args.config}")
    print(f"Device:          {args.device}")
    print("======================================\n")

    optimized_sequence = ppo_optimizer.optimize_sequence(
        H_chain,
        L_chain,
        Ag_chain,
        args.chain_type,
        args.config,
        args.region,
        str(output_dir),
        str(pdb_path),
        str(cache_path),
        args.device,
        args.task,
    )

    print("Optimized sequence:")
    print(optimized_sequence)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")