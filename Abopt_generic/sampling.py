import os
from pathlib import Path
import argparse
import random
import re
import yaml
import numpy as np
import torch
import pandas as pd

from deeplog import Config, DeepLog
from Abopt.agent import Agent, TransformerNet
from Abopt.env import (
    Environment,
    Interpreter,
    Evaluator,
    FitnessScorer,
    MHCIIAffinity,
    FvNetCharge,
    FvCSP,
    HISum,
)
from Abopt.data import GeneralSeqUtils


HEAVY_REGIONS = ["CDRH", "H3", "H2", "H1"]
LIGHT_REGIONS = ["CDRL", "L3", "L2", "L1"]


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str):
    device_arg = str(device_arg)

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")

        n_gpu = torch.cuda.device_count()
        idx = 0

        if ":" in device_arg:
            try:
                idx = int(device_arg.split(":")[1])
            except ValueError:
                idx = 0

        if idx >= n_gpu:
            print(
                f"[WARNING] Requested {device_arg}, but only {n_gpu} CUDA device(s) are available. "
                "Using cuda:0 instead."
            )
            idx = 0

        return torch.device(f"cuda:{idx}")

    return torch.device("cpu")


def list2str(li):
    return "".join([str(i) for i in li])


def rewards_to_numpy(rewards):
    if torch.is_tensor(rewards):
        rewards = rewards.detach().cpu().numpy()
    rewards = np.asarray(rewards)
    return rewards.reshape(-1)


def safe_filename(name: str):
    name = str(name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def get_mutations(wt_seq, opt_seq, chain_tag="H"):
    mutations = []

    for i, (wt, opt) in enumerate(zip(wt_seq, opt_seq), start=1):
        if wt != opt:
            mutations.append(f"{wt}{i}{opt}")

    return "_".join(mutations) if mutations else "WT"


def build_evaluator(
    task,
    seq_utils,
    config,
    pdb_path,
    inputchain,
    chain_type,
    region,
    cache_path,
    device,
):
    """
    task='fitness':
        FitnessScorer only.

    task='multi':
        FitnessScorer + MHCIIAffinity + FvNetCharge + FvCSP + HISum.
    """

    task = str(task).lower()
    if task == "multi_property":
        task = "multi"

    fitness = FitnessScorer(
        pdb_path=str(pdb_path),
        inputchain=inputchain,
        chain=chain_type,
        region=region,
        cache_path=str(cache_path),
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
        raise ValueError(f"Invalid task: {task}. Available tasks are: fitness, multi")

    evaluator = Evaluator(
        score_functions=score_functions,
        function_weights=function_weights,
        use_dynamic_weights=config.use_dynamic_weights,
        use_non_dominated_sort=config.use_non_dominated_sort,
    )

    return evaluator


class PPOOptimizer:
    @staticmethod
    def load_config(file_name: str) -> Config:
        config = Config()

        with open(file_name, "r") as fp:
            config_dict = yaml.load(fp, yaml.FullLoader)

        config.from_dict(config_dict)
        config.print_parameters()

        return config

    def inference_sequences(
        self,
        H_chain: str,
        L_chain: str,
        Ag_chain: str,
        chain_type: str,
        config_name: str,
        region: str,
        output_dir: str,
        pdb_path: str,
        save_model_path: str,
        cache_path: str,
        device: str,
        num: int,
        task: str = "fitness",
    ) -> pd.DataFrame:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Keep two device representations:
        # 1) torch_device: used by PyTorch APIs, e.g. .to(), torch.load(map_location=...)
        # 2) device_str: used by Abopt/Cython classes that require a Python str
        torch_device = resolve_device(device)
        device_str = str(torch_device)

        config = self.load_config(config_name)

        seq_utils = GeneralSeqUtils()
        logger = DeepLog(save_path=str(output_dir))

        seq_utils.ag_sequence = Ag_chain
        seq_utils.heavy_sequence = H_chain
        seq_utils.light_sequence = L_chain

        sequence = seq_utils.get_edit_sequences(
            [H_chain],
            [L_chain],
            region=region,
        )

        logger.log("Hchain", H_chain)
        logger.log("Lchain", L_chain)
        logger.log("Agchain", Ag_chain)
        logger.log("region", region)
        logger.log("task", task)

        config.seq = sequence

        agent = Agent(
            policy=TransformerNet(n_action=config.n_action),
            use_GPU=(torch_device.type == "cuda"),
            device=device_str,
        )

        print(f"Loading model from: {save_model_path}")
        state_dict = torch.load(save_model_path, map_location=torch_device)
        result = agent.policy.load_state_dict(state_dict, strict=True)
        print(result)

        agent.policy.to(torch_device)
        agent.policy.eval()

        interpreter = Interpreter(
            seq_utils=seq_utils,
            device=device_str,
            interpretive_method=config.interpretive_method,
            region=region,
        )

        if region in HEAVY_REGIONS:
            inputchain = H_chain
        elif region in LIGHT_REGIONS:
            inputchain = L_chain
        else:
            raise ValueError(
                f"Invalid region: {region}. Available regions: {HEAVY_REGIONS + LIGHT_REGIONS}"
            )

        evaluator = build_evaluator(
            task=task,
            seq_utils=seq_utils,
            config=config,
            pdb_path=pdb_path,
            inputchain=inputchain,
            chain_type=chain_type,
            region=region,
            cache_path=cache_path,
            device=device_str,
        )

        environment = Environment(
            interpreter=interpreter,
            evaluator=evaluator,
        )

        seq_tensor = seq_utils.sequence_to_tensor(sequence).to(torch_device)

        # One input sequence is sampled once per iteration.
        # Therefore, num controls the number of stochastic inference attempts.
        states = seq_tensor.repeat(1, 1)

        result_list = []
        actions_probs = None

        for i in range(num):
            with torch.no_grad():
                actions, actions_probs = agent(states)
                interpreted_sequences_H, interpreted_sequences_L, rewards, properties = environment(
                    states,
                    actions,
                )

            properties = properties.copy()

            reward_values = rewards_to_numpy(rewards)
            if len(reward_values) == len(properties):
                properties["reward"] = reward_values
            else:
                properties["reward"] = reward_values[0]

            properties["opt_Hsequence"] = interpreted_sequences_H
            properties["opt_Lsequence"] = interpreted_sequences_L

            actions = actions.detach().cpu().tolist() if torch.is_tensor(actions) else actions
            properties["action"] = [list2str(action) for action in actions]

            result_list.append(properties)

        properties = pd.concat(result_list, ignore_index=True)

        if region in HEAVY_REGIONS:
            properties = properties.drop_duplicates("opt_Hsequence", ignore_index=True)
            properties["H_mutation"] = properties["opt_Hsequence"].apply(
                lambda seq: get_mutations(H_chain, seq, chain_tag="H")
            )
        else:
            properties = properties.drop_duplicates("opt_Lsequence", ignore_index=True)
            properties["L_mutation"] = properties["opt_Lsequence"].apply(
                lambda seq: get_mutations(L_chain, seq, chain_tag="L")
            )

        properties = properties.sort_values(
            "reward",
            ascending=False,
            ignore_index=True,
        )

        # Add WT properties as the first row for reference.
        _, wt_df = evaluator([H_chain], [L_chain])
        wt_df = wt_df.copy()
        wt_df["action"] = "WT"
        wt_df["reward"] = "WT"
        wt_df["opt_Hsequence"] = H_chain
        wt_df["opt_Lsequence"] = L_chain

        if region in HEAVY_REGIONS:
            wt_df["H_mutation"] = "WT"
        else:
            wt_df["L_mutation"] = "WT"

        properties = pd.concat([wt_df, properties], axis=0, ignore_index=True)

        output_csv = output_dir / f"{safe_filename(config.seq[0])}_{task}.csv"
        properties.to_csv(output_csv, index=False)

        print(f"\nSaved results to: {output_csv}")
        print(properties.head(10))

        if actions_probs is not None:
            if torch.is_tensor(actions_probs):
                actions_probs_to_log = actions_probs.detach().cpu().tolist()
            else:
                actions_probs_to_log = actions_probs
            logger.log("actions_probs", actions_probs_to_log)

        logger.log_file_name = f"{safe_filename(config.seq[0])}_{task}.txt"

        return properties


def main(args):
    set_seed(args.seed)

    ppo_optimizer = PPOOptimizer()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "siltuximab_cache.pkl"
    save_model_path = Path(args.save_model_path)

    H_chain = (
        "EVQLVESGGKLLKPGGSLKLSCAASGFTFSSFAMSWFRQSPEKRLEWVAEISSGGSYTYYPDTVTGRFTISRDNAKNTLYLEMSSLRSEDTAMYYCARGLWGYYALDYWGQGTSVTVSS"
    )
    L_chain = (
        "QIVLIQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDTSNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSGYPYTFGGGTKLEIK"
    )
    Ag_chain = (
        "PHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQM"
    )

    ppo_optimizer.inference_sequences(
        H_chain=H_chain,
        L_chain=L_chain,
        Ag_chain=Ag_chain,
        chain_type=args.chain_type,
        config_name=args.config,
        region=args.region,
        output_dir=str(output_dir),
        pdb_path=args.pdb_path,
        save_model_path=str(save_model_path),
        cache_path=str(cache_path),
        device=args.device,
        num=args.num_sequences,
        task=args.task,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antibody optimization inference")

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
        default="CDRH",
        choices=["CDRH", "H3", "H2", "H1", "CDRL", "L3", "L2", "L1"],
        help="Region to optimize.",
    )

    parser.add_argument(
        "--chain_type",
        type=str,
        default="B",
        help="Chain identifier used by FitnessScorer, e.g., antigen or target chain in the PDB.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run optimization on, e.g., cuda:0 or cpu.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file.",
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        default="./saved_models/output_files_siltuximab_single_2024/log/GFTFSSFSSGGSYHLWGYYALDYhgfr_model.pth",
        help="Path to the saved model checkpoint.",
    )

    parser.add_argument(
        "--num_sequences",
        type=int,
        default=128,
        help="Number of stochastic inference attempts.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="fitness",
        choices=["fitness", "multi"],
        help="Optimization task. Use 'fitness' for fitness-only optimization and 'multi' for multi-property optimization.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed used for stochastic inference.",
    )

    args = parser.parse_args()

    import time

    start_time = time.time()
    main(args)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
