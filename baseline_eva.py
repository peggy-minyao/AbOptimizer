import os
import random
import numpy as np
import torch
import argparse
import csv
from deeplog import Config, DeepLog
from Abopt.agent import Agent, AgentEnv, TransformerNet
from Abopt.env import (
    Environment, Interpreter, Evaluator,
    Affinity, Her2Specificity, MHCIIAffinity, FvCSP, FvNetCharge, HISum, Specificity
)
from Abopt.data import CDR3SeqUtils, Ab4000SeqUtils
from Abopt.baseline import (
    BoOptimizer, LMOptimizer, LMmaxOptimizer,
    VirtualScreen, GeneticAlgorithm, GradientAscent
)



def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_evaluator(antibody, task, device: str = "cuda:0"):

    if antibody == 'emi':
        seq_utils = Ab4000SeqUtils()
        affinity = Affinity(model_path="./Abopt/env/hgfr_affinity.pt", device=device, seq_utils=seq_utils)
    elif antibody == 'tras':
        seq_utils = CDR3SeqUtils()
        affinity = Her2Specificity(model_path="./Abopt/env/variable_her2_spec.pt", device=device, seq_utils=seq_utils)
    else:
        print("error: invalid antibody type")

    specificity = Specificity(model_path="./Abopt/env/hgfr_specificity.pt", device=device, seq_utils=seq_utils)
    mhc = MHCIIAffinity(seq_utils=seq_utils)
    fvnet = FvNetCharge(seq_utils=seq_utils)
    fvcsp = FvCSP(seq_utils=seq_utils)
    hisum = HISum(seq_utils=seq_utils)

    if antibody == "emi":
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

    elif antibody == "tras":
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

    if task not in task_map:
        raise ValueError(f"Unknown task type: {task}")

    score_funcs, weights = task_map[task]

    evaluator = Evaluator(
        score_functions=score_funcs,
        function_weights=weights,
        use_dynamic_weights=False,
        use_non_dominated_sort=True,
    )
    return evaluator, seq_utils


def run_baseline(baseline: str, input_sequence: str, evaluator, seq_utils, device="cuda:0", output_dir="./output_files/"):
    baseline = baseline.lower()
    baseline_map = {
        "bo": BoOptimizer(evaluator),
        "ga": GeneticAlgorithm(128, 10, evaluator),
        "lm": LMOptimizer(evaluator, seq_utils, device=device),
        "lmmax": LMmaxOptimizer(evaluator, seq_utils, device=device),
        "vs": VirtualScreen(evaluator),
        "grad": GradientAscent(seq_utils),
    }

    if baseline not in baseline_map:
        raise ValueError(f"Unknown baseline method: {baseline}. Choose from {list(baseline_map.keys())}")

    optimizer = baseline_map[baseline]

    print(f"Running {baseline.upper()} optimization for sequence: {input_sequence}")
    result = optimizer.optimize_sequence(input_sequence)

    if isinstance(result, tuple):
        optimal_sequence, properties_all = result
    else:
        optimal_sequence, properties_all = result, None

    os.makedirs(output_dir, exist_ok=True)
    if properties_all is not None:
        save_path = os.path.join(output_dir, f"{input_sequence}.csv")
        properties_all.to_csv(save_path)
        print(properties_all.head(10))
        print(f"Results saved to: {save_path}")

    return optimal_sequence


def main():
    parser = argparse.ArgumentParser(description="Run antibody optimization with different baselines.")
    parser.add_argument("--baseline", type=str, default="vs",
                        choices=["bo", "ga", "lm", "lmmax", "vs", "grad"],
                        help="Select the optimization baseline.")
    parser.add_argument("--input_csv", type=str,
                        default="./starting_data/co-optimization_emi_data.csv",
                        help="CSV file containing sequences (with 'AASeq' column).")
    parser.add_argument("--output_dir", type=str, default="./output_files_emi_co_properties_baselines/",
                        help="Directory to save output CSVs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g., cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--index", type=str, default="0",
                        help="Which sequence index to run (integer or 'all').")
    parser.add_argument("--task", type=str, choices=['co_opt', 'affinity_opt','multiple_opt'], default='co_opt', help="The task type.")
    parser.add_argument("--antibody", type=str, choices=['emi', 'tras'], default='emi', help="Specify the antibody.")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.baseline}_results")
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    evaluator, seq_utils = create_evaluator(args.antibody, args.task, args.device)

    with open(args.input_csv, mode="r") as file:
        csv_reader = list(csv.DictReader(file))
        total_rows = len(csv_reader)

        if args.index.lower() == "all":
            indices = range(total_rows)
        else:
            idx = int(args.index)
            if idx < 0 or idx >= total_rows:
                raise ValueError(f"Index {idx} out of range (0–{total_rows-1}).")
            indices = [idx]

        for i in indices:
            input_sequence = csv_reader[i]["AASeq"]
            print(f"\n=== Running index {i}: {input_sequence}... ===")
            run_baseline(args.baseline, input_sequence, evaluator, seq_utils,
                         device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
