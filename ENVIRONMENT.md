# KuaiSim Environment Setup

## 1. Recommended baseline

This repository is easiest to run with a Python 3.8 Conda environment.

Recommended baseline:

- OS: Linux or macOS
- Python: 3.8
- PyTorch: 1.12.x or 1.13.x
- CUDA: optional, only needed if you want GPU training

The codebase contains three dependency layers:

1. Core KuaiSim / KuaiRand training and simulation
2. Optional `VirTB` / `Recogym` extras
3. Optional `recsim` TensorFlow-based experiments

If your goal is to run the main KuaiRand user model and RL training, you only need the core environment.

## 2. Core environment

Create the environment:

```bash
conda create -n kuaisim python=3.8 -y
conda activate kuaisim
```

Install PyTorch:

- CPU:

```bash
pip install torch==1.13.1 torchvision==0.14.1
```

- CUDA 11.7:

```bash
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

Install core Python packages:

```bash
pip install -r requirements-core.txt
```

Optional Jupyter support:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name kuaisim --display-name "KuaiSim"
```

## 3. Optional packages

For `VirTB` and `Recogym`:

```bash
pip install -r requirements-optional.txt
```

For the vendored `recsim` folder:

```bash
pip install -r requirements-recsim.txt
```

Notes:

- `requirements-recsim.txt` is only needed for the TensorFlow-based `code/recsim/` experiments.
- `pystan` in `requirements-optional.txt` is only used by one `recogym` baseline and is often the hardest dependency to build.

## 4. Data layout

The code assumes data lives under the repository, typically in:

```text
dataset/kuairand/kuairand-Pure/data/
```

Expected KuaiRand Pure files include:

- `log_standard_4_08_to_4_21_pure.csv`
- `log_standard_4_22_to_5_08_pure.csv`
- `user_features_pure.csv`
- `video_features_basic_pure.csv`
- `video_features_statistic_pure.csv`

`dataset/kuairand/kuairand-Pure/load_data_pure.py` also references:

- `log_random_4_22_to_5_08_pure.csv`

If that file is missing, that demo script will fail even if the main training code works.

## 5. Main workflow

### Step 1: train the immediate user response model

The core supervised training entrypoint is:

```text
code/train_multibehavior.py
```

This script requires at least:

- `--reader`
- `--model`
- `--train_file`
- `--user_meta_file`
- `--item_meta_file`

Typical run pattern:

```bash
cd code
python train_multibehavior.py \
  --reader KRMBSeqReader \
  --model KRMBUserResponse \
  --train_file ../dataset/kuairand/kuairand-Pure/data/log_standard_4_08_to_5_08_pure.csv \
  --user_meta_file ../dataset/kuairand/kuairand-Pure/data/user_features_pure.csv \
  --item_meta_file ../dataset/kuairand/kuairand-Pure/data/video_features_basic_pure.csv \
  --cuda 0
```

Important:

- The repository currently does not include a merged `log_standard_4_08_to_5_08_pure.csv`.
- You either need to create that file yourself or point `--train_file` to the exact log file you want to use.

### Step 2: generate session data

After the user response model is trained, `code/generate_session_data.py` consumes the saved model log via:

- `--behavior_model_log_file`

The shell helper `code/generate_session_data.sh` expects the trained model log under:

```text
code/output/Kuairand_Pure/env/log/
```

There is a path mismatch in the current repository:

- the script writes output to `dataset/Kuairand-Pure/...`
- the dataset folder in this repo is `dataset/kuairand/kuairand-Pure/...`

You should fix that path before running the script.

### Step 3: train RL agents

Whole-session RL entrypoint:

```text
code/train_actor_critic.py
```

Cross-session TD3 entrypoint:

```text
code/train_td3.py
```

These RL scripts need a pretrained user response model log through:

- `--uirm_log_path`

The environment classes read the first lines of that log file and reconstruct the reader/model from it.

## 6. What each requirements file covers

`requirements-core.txt`

- enough for the main KuaiRand reader, user response model, environment simulation, and RL training

`requirements-optional.txt`

- extras used by `code/VirTB/` and `code/recogym/`

`requirements-recsim.txt`

- TensorFlow stack for `code/recsim/`

## 7. Known pitfalls

1. Python newer than 3.9 may break older dependencies and some dynamic imports.
2. `code/recsim/` needs TensorFlow 1.x style compatibility APIs and extra packages not needed by the main project.
3. Some shell scripts assume GPU usage with `--cuda 0`; if you want CPU-only, set `--cuda -1`.
4. Several helper scripts hardcode paths under `output/Kuairand_Pure/` and expect model log files to already exist.
5. The repository contains multiple dataset path naming styles:
   - `dataset/kuairand/kuairand-Pure/...`
   - `dataset/Kuairand-Pure/...`
   Check and normalize these before running long jobs.

## 8. Minimal recommendation

If you just want the project runnable with the least friction:

1. Create the Conda env with Python 3.8
2. Install `requirements-core.txt`
3. Put KuaiRand data under `dataset/kuairand/kuairand-Pure/data/`
4. Run only the core `code/train_multibehavior.py`, `code/generate_session_data.py`, `code/train_actor_critic.py`, and `code/train_td3.py` pipeline first
5. Leave `VirTB`, `Recogym`, and `recsim` for a second environment if needed
