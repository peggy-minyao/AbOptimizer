# Multi-property Optimization of Therapeutic Antibodies through Sequence Editing Guided by Large Language Model and Reinforcement Learning

Discovering antibodies with multiple beneficial properties is essential for therapeutic drug development. However, antibodies identified through experimental screening often show suboptimal properties, necessitating further optimization. Despite computational advancements, joint optimization of multiple properties remains challenging because antibody sequence space is vast, experimental annotations are sparse, and property trade-offs are common. Herein, we propose AbOptimizer, a novel reinforcement learning framework for efficient multi-property antibody optimization through parallel multi-site sequence editing. AbOptimizer simulates natural evolution, where an agent learns to adjust mutations from property-based feedback. The agent performs site-specific editing actions, including ‘reservation' and ‘substitution', while allowing ‘insertion' and ‘deletion' as optional action types. A protein language model proposes plausible mutations, while property surrogate models evaluate the optimized variants. Pareto-based rewards then guide balanced optimization across multiple properties. AbOptimizer was evaluated on three therapeutic antibodies across complementary optimization settings: multi-property optimization of emibetuzumab, generic fitness surrogate-guided optimization of siltuximab without antibody-specific mutant data, and indel-inclusive optimization of trastuzumab with insertion and deletion edits.  \textit{In silico}, AbOptimizer consistently improved optimized properties and outperformed baseline methods. \textit{In vitro}, the selected variants showed improved affinity and specificity for emibetuzumab and siltuximab, largely maintained thermal stability for siltuximab, and enhanced affinity for trastuzumab indel variants. These results demonstrate that AbOptimizer enables multi-property, data-limited and indel-inclusive optimization, providing a practical framework for discovering therapeutic antibody candidates.![Figure 1](Figure1.jpg)

## AbOptimizer

AbOptimizer is a therapeutic antibody optimization framework based on reinforcement learning and large language models. When provided with reliable user-defined surrogate models, it supports the optimization of diverse antibody properties across different antibody sequences. 


## ⚙️Requirement

* python=3.10.15

* numpy=2.0.2

* torch=2.1.2+cu118

  Detail in **Requirement.txt**



## 🚀Getting started

### Step1: Clone this repository and cd into it

```
mkdir your_workspace
cd your_workspace
git clone https://github.com/xxxx/
```



### Step2: Create the Conda environment

```
conda create -n abopt python=3.10.15 numpy=2.0.2

conda activate abopt

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

pip install -r requirements.txt
```

⚠️ **Note:** The CUDA version (`cu118`) in the installation command should be adjusted according to your local CUDA environment.
 You can check your CUDA version by running `nvidia-smi` and select the corresponding PyTorch wheel (e.g., `cu117`, `cu121`, or `cpu`).

### Step3: Required packages

#### 1.Deeplog

```
cd DeepLog
python setup.py install
```

#### 2.Optional: netMHCIIpan-4.1 

This step is only required for multiple-property optimization in **Antibody optimization with antibody-specificity surrogate**. First, download **NetMHCIIpan-4.1** from [DTU Health Tech](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.1/). Then, place the downloaded compressed file in `./AbOpt/env/`, extract it, and follow the installation instructions in `netMHCIIpan-4.1.readme`.  Finally, ensure that **NetMHCIIpan** passes all test runs successfully.

Please ensure that `tcsh` is installed and available at `/bin/tcsh` before running NetMHCIIpan-dependent modules, for example by using `sudo apt-get install -y tcsh && sudo ln -sf /usr/bin/tcsh /bin/tcsh`.

```
# Temporarily add the directory to system path
export PATH="/root/autodl-tmp/framework_LLM_4acitions/AbOpt/env/netMHCIIpan-4.1:$PATH"
```

#### 3.Optioal: ANARCI

This step is necessary only when performing **antibody optimization with the generic-specificity surrogate**. The [ANARCI repository](https://github.com/oxpig/ANARCI) is provided under `./AbOpt/env/ANARCI`. Carefully follow the README instructions to complete installation and verify that the tests run successfully.

Then,

```
pip install numpy==1.26.4 --force-reinstall
```



### Step4: Antibody optimization with antibody-specificity surrogate

#### 🧪Training

```python
# Take co-optimization of emibetuzumab for example
python main.py
```

| Parameter      | Type  | Default                                        | Description                                                  |
| -------------- | :---- | :--------------------------------------------- | :----------------------------------------------------------- |
| `--config`     | `str` | `config.yaml`                                  | Path to the configuration file.                              |
| `--antibody`   | `str` | `emi`                                          | Specify the antibody type. Options: `emi`, `tras`.           |
| `--index`      | `str` | `0`                                            | Specify which sequence index to run, or use `'all'` to run all sequences. |
| `--output_dir` | `str` | `./output_files_emi_co_properties/`            | Directory to save the output results.                        |
| `--input_dir`  | `str` | `./starting_data/co-optimization_emi_data.csv` | Input CSV file path, must contain an `AASeq` column.         |
| `--task`       | `str` | `co_opt`                                       | Task type. Options: `co_opt` (co-optimization), `affinity_opt` (affinity only), `multiple_opt` (multi-property optimization). |

⚠️ **Note:** The first run downloads the Ablang model (including amodel.pt, hparams.json,vocab.json ) and may take some time.



#### 🧪Running inference with saved models

```python
# Take co-optimization of emibetuzumab for example
python test.py
```

| Parameter           | Type  | Default       | Description                                                  |
| ------------------- | ----- | ------------- | ------------------------------------------------------------ |
| `--config`          | `str` | `config.yaml` | Path to the configuration YAML file containing model and environment settings. |
| `--antibody`        | `str` | `emi`         | Specify antibody type. Options: `emi`, `tras`.               |
| `--sequence`        | `str` | `None`        | Input amino acid sequence. If not provided, sequences will be read from the corresponding CSV file. |
| `--task`            | `str` | `co_opt`      | Type of optimization task. Options:• `co_opt` – Co-optimization (affinity + specificity)• `affinity_opt` – Affinity optimization only• `multiple_opt` – Multi-property optimization |
| `--num`             | `int` | `1`           | Number of inference iterations to perform for each sequence. |
| `--seed`            | `int` | `2024`        | Random seed                                                  |
| `--save_model_path` | `str` | `None`        | Path to the saved model file (`.pth`).If not provided, it is automatically generated as:`./saved_model/{antibody}/{task}/output_files_abopt_seed_{seed}/log/{sequence}hgfr_model.pth` |



#### 🧪Baselines evaluation

```python
# Take co-optimization of emibetuzumab by Bayesian optimizations as an example
python baseline_eva.py
```

| Parameter      | Type  | Default                                        | Description                                                  |
| -------------- | ----- | ---------------------------------------------- | ------------------------------------------------------------ |
| `--baseline`   | `str` | `bo`                                           | Select which optimization baseline to run. Options:• `bo` – Bayesian Optimization• `ga` – Genetic Algorithm• `lm` – Language Model sampling• `lmmax` – LM with max sampling• `vs` – Virtual Screening• `grad` – Gradient-based optimization |
| `--input_csv`  | `str` | `./starting_data/co-optimization_emi_data.csv` | Path to input CSV file containing sequences. Must include an `AASeq` column. |
| `--output_dir` | `str` | `./output_files_emi_co_properties_baselines/`  | Directory to save output results. Subfolder automatically named by baseline (e.g., `bo_results`). |
| `--device`     | `str` | `cuda:0`                                       | Device for model inference, e.g., `cuda:0` or `cpu`.         |
| `--seed`       | `int` | `2024`                                         | Random seed                                                  |
| `--index`      | `str` | `"0"`                                          | Sequence index to run. Can be:• Integer (e.g., `0`, `5`) – run specific row from CSV• `"all"` – run all sequences in the CSV |
| `--task`       | `str` | `co_opt`                                       | Type of optimization task. Options:• `co_opt` – Co-optimization (affinity + specificity)• `affinity_opt` – Affinity optimization only• `multiple_opt` – Multi-property optimization |
| `--antibody`   | `str` | `emi`                                          | Antibody type. Options: `emi`, `tras`.                       |



#### 📝Surrogates

The trained surrogate predictors and the corresponding compiled files are provided in: `./Abopt/env/`

| Antibody                | Affinity Predictor             | Specificity Predictor               | Fitness Predictor                     |
| ----------------------- | ------------------------------ | ----------------------------------- | ------------------------------------- |
| **Emibetuzumab (HGFR)** | `./Abopt/env/hgfr_affinity.pt` | `./Abopt/env/hgfr_specificity.pt`   | (N/A)                                 |
| **Trastuzumab (HER2)**  | *(N/A)*                        | `./Abopt/env/variable_her2_spec.pt` | (N/A)                                 |
| **Siltuximab (IL-6)**   | (N/A)                          | (N/A)                               | ./Abopt/abopt_generic/env/LLM_fitness |



#### 📝Datasets

The datasets of emibetuzumab and trastuzumab are sourced from GitHub: [emibetuzumab ](https://github.com/Tessier-Lab-UMich/Emi_Pareto_Opt_ML) and [trastuzumab](https://github.com/dahjan/DMS_opt/tree/master).

| **Antibody**     | **Subset** | **Type** | **Length** | **Binding** | **Non-binding** | **High Non-specific** | **Low Non-specific** | **KD** | **Total**  |
| ---------------- | ---------- | -------- | ---------- | ----------- | --------------- | --------------------- | -------------------- | ------ | ---------- |
| **Emibetuzumab** | Train      | Binary   | Equal      | 1,516       | 2,484           | 2,000                 | 2,000                | —      | **4,000*** |
|                  | Test       | Numeric  | Equal      | —           | —               | —                     | —                    | 125    | **125**    |
| **Trastuzumab**  | Train      | Binary   | Equal      | 11,300      | 27,539          | —                     | —                    | —      | **38,839** |
|                  | Train      | Binary   | Variable   | 337         | 1,097           | —                     | —                    | —      | **1,434**  |
|                  | Test       | Numeric  | Variable   | —           | —               | —                     | —                    | 421    | **421**    |

🧠*: The Emibetuzumab training set contains a total of **4,000 sequences**, each annotated as either binding or non-specific binding.



### Step5: Antibody optimization with generic-specificity surrogate

 Here, we employed a [general-purpose fitness](https://www.science.org/doi/10.1126/science.adk8946) predictor within AbOptimizer as a surrogate. 

#### 🧪Training with generic-specificity surrogate

```
# Take fitness optimization of siltuximab as an example
cd Abopt_generic
python main.py 
```

| Parameter      | Type  | Default                      | Description                                                  |
| -------------- | ----- | ---------------------------- | ------------------------------------------------------------ |
| `--pdb_path`   | `str` | `./pdb/siltuximab.pdb`       | Path to the input PDB file.Example: `--pdb_path ./pdb/siltuximab.pdb` |
| `--output_dir` | `str` | `./output_files_siltuximab/` | Directory to save optimization results.Example: `--output_dir ./results/` |
| `--region`     | `str` | `CDRH`                       | Antibody region to optimize. Options: CDRH，H3，H2，H1，CDRL，L3，L2，L1 |
| `--chain_type` | `str` | `AB                          | PDB chain ID (e.g., heavy = B, light = A in PDB user defined.); set to A when optimizing CDRH. |
| `--device`     | `str` | `cuda:0`                     | Device for running optimization (e.g., `cuda:0` or `cpu`).   |
| `--config`     | `str` | `config.yaml`                | Path to the configuration YAML file.Testing with saved models |
| `--seed`       | int   | 2026                         | Random seed                                                  |
| --task         | str   | fitness                      | Task type. Options: `fitness` (fitness-optimization), `multi` (multi-property optimization). |

💡 Typical time: < 2 min under cached files. New antibodies may require more time.



#### 🧪Running inference with saved models

```
# Take fitness optimization of siltuximab as an example
python sampling.py
```

| Parameter         | Type  | Default                                   | Description                                                  |
| ----------------- | ----- | ----------------------------------------- | ------------------------------------------------------------ |
| `--pdb_path`      | `str` | `./pdb/siltuximab.pdb`                    | Path to the input PDB file.Example: `--pdb_path ./pdb/siltuximab.pdb` |
| `--output_dir`    | `str` | `./output_files_siltuximab/`              | Directory to save optimization results.Example: `--output_dir ./results/` |
| `--region`        | `str` | `H3`                                      | Antibody region to optimize. Options: CDRH，H3，H2，H1，CDRL，L3，L2，L1 |
| `--chain_type`    | `str` | `B`                                       | PDB chain ID (e.g., heavy = A, light = B in PDB user defined.); set to A when optimizing CDRH. |
| `--device`        | `str` | `cuda:0`                                  | Device for running optimization (e.g., `cuda:0` or `cpu`).   |
| `--config`        | `str` | `config.yaml`                             | Path to the configuration YAML file.Testing with saved models |
| --save_model_path | str   | `/GFTFSSFSSGGSYHLWGYYALDYhgfr_model.pth'` | Path to save trained model.                                  |
| --num_sequences   | int   | 128                                       | Number of sequences to generate.                             |
| --task            | str   | fitness                                   | Task type. Options: `fitness` (fitness-optimization), `multi` (multi-property optimization). |
| --seed            | int   | 2026                                      | Random seed                                                  |


## 📜 License

This software is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).
You may view a copy of this license at:
👉 https://creativecommons.org/licenses/by-nc-nd/4.0/

Under this license, you are free to share (copy and redistribute) the material in any medium or format, under the following terms:

Attribution — You must give appropriate credit and provide a link to the license.

NonCommercial — You may not use the material for commercial purposes.

NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.


## 🧾 Citation

If you use AbOptimizer in your research, please cite it as:

Multi-property Optimization of Therapeutic Antibodies through Sequence Editing Guided by Large Language Model and Reinforcement Learning
[Name] et al., [Institution], [Year].
DOI or GitHub: https://github.com/Repo/AbOptimizer
