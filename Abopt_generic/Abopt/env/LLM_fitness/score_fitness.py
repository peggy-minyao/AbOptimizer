import pandas as pd
import torch
import numpy as np
# torch.hub.set_dir('/home/u/data/whz/qmy') 
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
import esm
from multichain_util import extract_coords_from_complex, _concatenate_coords, _concatenate_seqs, score_sequence_in_complex
from util import get_sequence_loss, load_structure, load_coords, score_sequence, extract_coords_from_structure

def score_seqs_from_csv(model, alphabet, csv_path, pdb_path, chain='A', device='cuda:0'):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get sequence column (support multiple names)
    if 'seq' in df.columns:
        sequences = df['seq']
    elif 'sequence' in df.columns:
        sequences = df['sequence']
    else:
        raise ValueError("CSV must contain a column named 'seq' or 'sequence'")
    
    # Load structure coords
    structure = load_structure(pdb_path)
    coords, native_seqs = extract_coords_from_complex(structure)
    #native_seq = native_seqs[chain]
    
    # Move model to device
    model = model.to(device)
    model = model.eval()

    scores = []
    for seq in sequences:
        ll, _ = score_sequence_in_complex(model, alphabet, coords, native_seqs, chain, str(seq), order = None)
        scores.append(ll)
    
    return scores

csv_path = "/root/autodl-tmp/framework_general/AbEditor/env/LLM_fitness/7mmo_secround_A.csv"
pdb_path = "/root/autodl-tmp/framework_general/AbEditor/env/LLM_fitness/structural-evolution-main/examples/7mmo_abc_fvar.pdb"
chain = "A"

# 预加载模型
model_checkpoint_path = "/root/autodl-tmp/framework_general/AbEditor/env/LLM_fitness/esm_if1_20220410.pt"
model, alphabet = esm.pretrained.load_model_and_alphabet(model_checkpoint_path)

# 获取分数列表
scores = score_seqs_from_csv(model, alphabet, csv_path, pdb_path, chain)

# 输出示例
for s in scores:
    print(f"log_likelihood: {s:.5f}")
    

# import pandas as pd
# import torch
# import esm
# from LLM_fitness.multichain_util import extract_coords_from_complex, score_sequence_in_complex
# from LLM_fitness.util import load_structure

# class StructuralScorer:
#     def __init__(self, model_path, pdb_path, chain="A", device="cuda:0"):
#         """
#         初始化结构评分模型
#         :param model_path: ESM-IF1 模型路径
#         :param pdb_path: 三维结构 PDB 文件路径
#         :param chain: 分析的链 ID
#         :param device: 使用的设备
#         """
#         self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
#         self.model = self.model.eval().to(device)
#         self.device = device

#         self.chain = chain
#         self.pdb_path = pdb_path
#         structure = load_structure(pdb_path)
#         self.coords, self.native_seqs = extract_coords_from_complex(structure)

#     def __call__(self, sequences):
#         """
#         对输入序列列表打分
#         :param sequences: list of str，蛋白质序列
#         :return: list of float，对应 log likelihood 分数
#         """
#         scores = []
#         for seq in sequences:
#             ll, _ = score_sequence_in_complex(
#                 self.model,
#                 self.alphabet,
#                 self.coords,
#                 self.native_seqs,
#                 self.chain,
#                 str(seq),
#                 order=None
#             )
#             scores.append(ll)
#         return scores

