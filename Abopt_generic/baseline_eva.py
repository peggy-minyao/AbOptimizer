import os
import random
import numpy as np
import torch
import argparse
import csv
from deeplog import Config, DeepLog
from Abopt.agent import Agent, AgentEnv, TransformerNet
from Abopt.env import (
    Environment, Interpreter, Evaluator, MHCIIAffinity, FvCSP, FvNetCharge, HISum, FitnessScorer
)
from Abopt.data import CDR3SeqUtils, Ab4000SeqUtils, GeneralSeqUtils
from Abopt.baseline import (
    BoOptimizer, LMOptimizer, LMmaxOptimizer,
    VirtualScreen, GeneticAlgorithm, GradientAscent
)



def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_evaluator(task, device: str = "cuda:0", pdb_path:str = "./pdb/siltuximab.pdb", chain_type:str = "A", region:str = "CDRH", cache_path:str = "./output_files_siltuximab/siltuximab_cache.pkl"):

    H_chain_sil = (
        "EVQLVESGGKLLKPGGSLKLSCAASGFTFSSFAMSWFRQSPEKRLEWVAEISSGGSYTYYPDTVTGRFTISRDNAKNTLYLEMSSLRSEDTAMYYCARGLWGYYALDYWGQGTSVTVSS"
    )
    L_chain_sil  = (
        "QIVLIQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDTSNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSGYPYTFGGGTKLEIK"
    )
    Ag_chain_sil  = (
        "PHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQM"
    )    

    seq_utils = GeneralSeqUtils()
    seq_utils.ag_sequence = Ag_chain_sil
    seq_utils.heavy_sequence = H_chain_sil
    seq_utils.light_sequence = L_chain_sil

    fitness = FitnessScorer(pdb_path = pdb_path, inputchain = H_chain_sil, chain=chain_type, region = region, cache_path = cache_path, device=device)

    # if region == "CDRH" or region == "H3" or region == "H2" or region == "H1":
    #     inputchain = H_chain_sil
    # else:
    #     inputchain = L_chain_sil

    mhc = MHCIIAffinity(seq_utils=seq_utils)
    fvnet = FvNetCharge(seq_utils=seq_utils)
    fvcsp = FvCSP(seq_utils=seq_utils)
    hisum = HISum(seq_utils=seq_utils)


    task_map = {
        "affinity_opt": ({"affinity": fitness}, [1.0]),    
        "multiple_opt_sil": (
            {
                "fitness": fitness,
                "mhc": mhc,
                "fvnet": fvnet,
                "fvcsp": fvcsp,
                "hisum": hisum,
            },
            [1.0] * 5,
        ),    
    }

    score_funcs, weights = task_map[task]

    evaluator = Evaluator(
        score_functions=score_funcs,
        function_weights=weights,
        use_dynamic_weights=False,
        use_non_dominated_sort=True,
    )
    return evaluator, seq_utils


def run_baseline(baseline: str, evaluator, seq_utils, device="cuda:0", output_dir="./output_files/", region="CDRH"):
    input_H_sequence = seq_utils.heavy_sequence
    edit_sequence = seq_utils.get_edit_sequences([seq_utils.heavy_sequence], [seq_utils.light_sequence], region=region)
    
    baseline = baseline.lower()
    baseline_map = {
        "bo": BoOptimizer(evaluator, seq_utils),
        "ga": GeneticAlgorithm(128, 10, evaluator, seq_utils),
        "lm": LMOptimizer(evaluator, seq_utils, device=device),
        "lmmax": LMmaxOptimizer(evaluator, seq_utils, device=device),
        "vs": VirtualScreen(evaluator,seq_utils),
        # "grad": GradientAscent(seq_utils),
    }

    if baseline not in baseline_map:
        raise ValueError(f"Unknown baseline method: {baseline}. Choose from {list(baseline_map.keys())}")

    optimizer = baseline_map[baseline]


    
    print(f"Running {baseline.upper()} optimization for sequence: {edit_sequence}")
    result = optimizer.optimize_sequence(edit_sequence)

    if isinstance(result, tuple):
        optimal_sequence, properties_all = result
    else:
        optimal_sequence, properties_all = result, None

    os.makedirs(output_dir, exist_ok=True)
    if properties_all is not None:
        save_path = os.path.join(output_dir, f"{edit_sequence}.csv")
        properties_all.to_csv(save_path)
        print(properties_all.head(10))
        print(f"Results saved to: {save_path}")

    return optimal_sequence


def main():
    parser = argparse.ArgumentParser(description="Run antibody optimization with different baselines.")
    parser.add_argument("--baseline", type=str, default="bo",
                        choices=["bo", "ga", "lm", "lmmax", "vs", "grad"],
                        help="Select the optimization baseline.")
    # parser.add_argument("--input_csv", type=str,
    #                     default="./starting_data/co-optimization_emi_data.csv",
    #                     help="CSV file containing sequences (with 'AASeq' column).")
    parser.add_argument("--output_dir", type=str, default="./output_files_sil_multi_properties_baselines_2025/",
                        help="Directory to save output CSVs.")
    parser.add_argument("--device", type=str, default="cuda:1", help="CUDA device, e.g., cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--index", type=str, default="0",
                        help="Which sequence index to run (integer or 'all').")
    parser.add_argument("--task", type=str, choices=['affinity_opt', 'multiple_opt_sil'], default='multiple_opt_sil', help="The task type.")
    # parser.add_argument("--antibody", type=str, choices=['emi', 'tras', 'sil'], default='sil', help="Specify the antibody.")
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./pdb/siltuximab.pdb",
        help="Path to the input PDB file (default: ./framework_general/pdb/siltuximab.pdb)",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=['CDRH', 'H3','H2','H1'],
        default="CDRH",
        help="Region to optimize"
    )
    parser.add_argument(
        "--chain_type",
        type=str,
        default="B",
        help="Antibody chain type, e.g., A (heavy) or B (light) (default: A), must allign with pdb file",
    )

    
    args = parser.parse_args()
    cache_path = '/mnt/qmy/aboptimizer/framework_general/output_files_siltuximab/siltuximab_cache.pkl'
    args.output_dir = os.path.join(args.output_dir, f"{args.baseline}_results")
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    evaluator, seq_utils = create_evaluator(args.task, args.device, pdb_path=args.pdb_path, chain_type=args.chain_type, region=args.region, cache_path=cache_path)

    run_baseline(args.baseline, evaluator, seq_utils,
                    device=args.device, output_dir=args.output_dir, region = args.region)


if __name__ == "__main__":
    main()
