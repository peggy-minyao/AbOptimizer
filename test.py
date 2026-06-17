import argparse
import pandas as pd
import torch
import random
import numpy as np
import yaml

from deeplog import Config
from Abopt.agent import Agent, TransformerNet
from Abopt.env import (
    Environment,
    Interpreter,
    Evaluator,
    Affinity,
    Specificity,
    Her2Specificity,
    MHCIIAffinity,
    FvNetCharge,
    FvCSP,
    HISum,
)
from Abopt.data import Ab4000SeqUtils, CDR3SeqUtils


def load_config(file_name: str) -> Config:
    config = Config()
    with open(file_name, "r") as fp:
        config_d = yaml.load(fp, yaml.FullLoader)
    config.from_dict(config_d)
    config.print_parameters()
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Improve reproducibility. This still may not guarantee bitwise-identical results
    # if stochastic sampling is used in inference.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config):
    if getattr(config, "use_GPU", False) and torch.cuda.is_available():
        return torch.device(config.device)
    return torch.device("cpu")


def build_seq_utils(antibody: str):
    """
    Build sequence utility according to antibody type.
    """
    if antibody == "emi":
        return Ab4000SeqUtils()
    elif antibody == "tras":
        return CDR3SeqUtils()
    else:
        raise ValueError(f"Invalid antibody type: {antibody}")


def build_environment(args, config, seq_utils, device):
    """
    Build interpreter, evaluator, and environment according to antibody type and task.

    For emibetuzumab (emi):
        - affinity_opt: affinity
        - co_opt: affinity + specificity
        - multiple_opt: affinity + specificity + MHC + FvNetCharge + FvCSP + HISum

    For trastuzumab (tras):
        - affinity_opt: affinity
        - multiple_opt: affinity + MHC + FvNetCharge + FvCSP + HISum
        - co_opt is not available because no trastuzumab-specific specificity predictor is defined.
    """

    interpreter = Interpreter(
        seq_utils=seq_utils,
        device=device,
        interpretive_method=config.interpretive_method,
    )

    # ---------- Antibody-specific affinity / specificity predictors ----------
    if args.antibody == "emi":
        affinity = Affinity(
            model_path="./Abopt/env/hgfr_affinity.pt",
            device=device,
            seq_utils=seq_utils,
        )
        specificity = Specificity(
            model_path="./Abopt/env/hgfr_specificity.pt",
            device=device,
            seq_utils=seq_utils,
        )

    elif args.antibody == "tras":
        # Her2Specificity is used here as the HER2-binding / affinity predictor.
        affinity = Her2Specificity(
            model_path="./Abopt/env/variable_her2_spec.pt",
            device=device,
            seq_utils=seq_utils,
        )
        specificity = None

    else:
        raise ValueError(f"Invalid antibody type: {args.antibody}")

    # ---------- Common biophysical property predictors ----------
    mhc = MHCIIAffinity(seq_utils=seq_utils)
    fvnet = FvNetCharge(seq_utils=seq_utils)
    fvcsp = FvCSP(seq_utils=seq_utils)
    hisum = HISum(seq_utils=seq_utils)

    # ---------- Antibody-specific task maps ----------
    if args.antibody == "emi":
        task_map = {
            "affinity_opt": (
                {"affinity": affinity},
                [1.0],
            ),
            "co_opt": (
                {"affinity": affinity, "specificity": specificity},
                [1.0, 1.0],
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
                [1.0] * 6,
            ),
        }

    elif args.antibody == "tras":
        task_map = {
            "affinity_opt": (
                {"affinity": affinity},
                [1.0],
            ),
            "multiple_opt": (
                {
                    "affinity": affinity,
                    "mhc": mhc,
                    "fvnet": fvnet,
                    "fvcsp": fvcsp,
                    "hisum": hisum,
                },
                [1.0] * 5,
            ),
        }

    if args.task not in task_map:
        raise ValueError(
            f"Task '{args.task}' is not available for antibody '{args.antibody}'. "
            f"Available tasks: {list(task_map.keys())}"
        )

    score_functions, function_weights = task_map[args.task]

    evaluator = Evaluator(
        score_functions=score_functions,
        function_weights=function_weights,
        use_dynamic_weights=config.use_dynamic_weights,
        use_non_dominated_sort=config.use_non_dominated_sort,
    )

    environment = Environment(
        interpreter=interpreter,
        evaluator=evaluator,
    )

    return environment

def rewards_to_numpy(rewards):
    if torch.is_tensor(rewards):
        rewards = rewards.detach().cpu().numpy()
    rewards = np.asarray(rewards)
    return rewards.reshape(-1)


def inference(sequence: str, num: int, agent, environment, seq_utils, device):
    seq_tensor = seq_utils.sequence_to_tensor([sequence]).to(device)

    # Keep the same batch size as the original implementation.
    states = seq_tensor.repeat(128, 1)

    result = []

    agent.policy.eval()

    for _ in range(num):
        with torch.no_grad():
            actions, actions_probs = agent(states)
            interpreted_seqs, rewards, properties = environment(states, actions)

        properties = properties.copy()
        properties["reward"] = rewards_to_numpy(rewards)
        properties["opt_sequence"] = interpreted_seqs

        result.append(properties)

    properties = pd.concat(result, ignore_index=True)
    properties = properties.drop_duplicates("opt_sequence", ignore_index=True)
    properties = properties.sort_values("reward", ascending=False, ignore_index=True)

    print(f"\n=== Top 10 optimized sequences for {sequence} ===")
    print(properties.head(10))
    print(f"Total unique optimized sequences: {len(properties)}")

    return properties


def get_model_suffix(antibody: str):
    """
    Adjust this mapping according to your actual saved model naming rule.
    """
    suffix_map = {
        "emi": "hgfr_model.pth",
        "tras": "her2_model.pth",
    }

    if antibody not in suffix_map:
        raise ValueError(f"No model suffix is defined for antibody={antibody}")

    return suffix_map[antibody]


def get_model_path(args, sequence: str):
    suffix = get_model_suffix(args.antibody)

    return (
        f"./saved_model/{args.antibody}/{args.task}/"
        f"output_files_abopt_seed_{args.seed}/log/"
        f"{sequence}{suffix}"
    )


def load_policy(agent, model_path, device):
    print(f"Loading model from: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    result = agent.policy.load_state_dict(state_dict, strict=True)
    print(result)

    agent.policy.to(device)
    agent.policy.eval()


def get_csv_path(args):
    """
    Adjust these file paths according to your actual starting data files.
    """

    csv_map = {
        "emi": {
            "co_opt": "./starting_data/co-optimization_emi_data.csv",
            "affinity_opt": "./starting_data/affinity_maturation_emi_data.csv",
            "multiple_opt": "./starting_data/multiple_optimization_emi_data.csv",
        },
        "tras": {
            "affinity_opt": "./starting_data/affinity_maturation_tras_data.csv",
            "multiple_opt": "./starting_data/multiple_optimization_tras_data.csv",
            # co_opt is not available for trastuzumab because no specificity predictor is defined.
        },
    }

    if args.antibody not in csv_map:
        raise ValueError(f"No CSV map is defined for antibody={args.antibody}")

    if args.task not in csv_map[args.antibody]:
        raise ValueError(
            f"No CSV file is defined for antibody={args.antibody}, task={args.task}. "
            f"Please provide --sequence directly or add the corresponding CSV path in get_csv_path()."
        )

    return csv_map[args.antibody][args.task]


def main():
    parser = argparse.ArgumentParser(
        description="Run antibody sequence optimization inference."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--antibody",
        type=str,
        default="emi",
        choices=["emi","tras"],
        help="Antibody type. Options: emi, tras.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Input amino acid sequence. If not provided, sequences will be read from the corresponding CSV file.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="affinity_opt",
        choices=["co_opt", "affinity_opt", "multiple_opt"],
        help="Optimization task type.",
    )

    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of stochastic inference iterations to perform for each input sequence.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used to initialize stochastic inference.",
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="Path to the saved model (.pth). If not provided, it is automatically generated.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    config = load_config(args.config)
    device = str(get_device(config))

    print(f"Using device: {device}")

    seq_utils = build_seq_utils(args.antibody)

    agent = Agent(
        policy=TransformerNet(n_action=config.n_action),
        use_GPU=config.use_GPU,
        device=device,
    )

    environment = build_environment(
        args=args,
        config=config,
        seq_utils=seq_utils,
        device=device,
    )

    # ---------- Single-sequence inference ----------
    if args.sequence:
        model_path = args.save_model_path
        if model_path is None:
            model_path = get_model_path(args, args.sequence)

        load_policy(agent, model_path, device)
        inference(args.sequence, args.num, agent, environment, seq_utils, device)

    # ---------- Batch inference from CSV ----------
    else:
        csv_path = get_csv_path(args)

        print(f"Reading sequences from {csv_path}")
        df = pd.read_csv(csv_path)

        if "AASeq" not in df.columns:
            raise ValueError(f"CSV file {csv_path} must contain a column named 'AASeq'.")

        for seq in df["AASeq"]:
            model_path = get_model_path(args, seq)

            print(f"\nLoading model for sequence: {seq}")

            try:
                load_policy(agent, model_path, device)
                inference(seq, args.num, agent, environment, seq_utils, device)

            except FileNotFoundError:
                print(f"Model not found for {seq}: {model_path}")


if __name__ == "__main__":
    main()