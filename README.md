# continuous-control-sac-mbpo-mlpdreamer

This is a class project for CS5180 (Reinforcement Learning), 2026 Spring, from Professor Christopher Amato. It focuses on performance and sample efficiency for three setups: off-policy (SAC), model-based (MBPO), and a latent world model / Dreamer-style agent **without** a VAE encoder. We run on relatively simple environments on purpose.

---

## Dependencies

Python **3.10+**. In the project root:

```bash
pip install -e .
```

That pulls `gymnasium`, `numpy`, `torch` from `pyproject.toml`. Optional logging:

```bash
pip install -e ".[wandb]"
```

MuJoCo / Box2D / extra gymnasium envs are **not** in this package; install those separately.

---

## CLI

### `sac.sac_runner`

| flag | default | meaning |
|------|---------|---------|
| `--env` | Pendulum-v1 | gymnasium env id |
| `--seed` | 42 | rng |
| `--max_steps` | 100000 | total env steps |
| `--prefill_steps` | 5000 | random actions before learning |
| `--eval_freq` | 2000 | eval every N steps |
| `--batch_size` | 256 | SAC minibatch |
| `--buffer_size` | 1000000 | replay size |
| `--lr` | 3e-4 | actor & critic Adam lr |
| `--gamma` | 0.99 | discount |
| `--tau` | 0.005 | target soft-update |
| `--alpha` | 0.2 | entropy coef (if not `--auto_alpha`) |
| `--mlp_depth` | 2 | MLP layers actor/critic |
| `--auto_alpha` | off | flag: learn temperature |
| `--wandb` | off | log to Weights & Biases |
| `--wandb_project` | sac-student-version | wandb project name |

### `mbpo.mbpo_runner`

| flag | default | meaning |
|------|---------|---------|
| `--env` | Pendulum-v1 | env id |
| `--seed` | 42 | rng |
| `--max_steps` | 100000 | total steps |
| `--prefill_steps` | 5000 | random prefill |
| `--eval_freq` | 1000 | eval every N steps |
| `--batch_size` | 256 | batch for SAC + model |
| `--horizon` | 5 | imagined rollout length |
| `--real_ratio` | 0.05 | fraction of real transitions in mixed SAC update |
| `--model_train_freq` | 250 | train dynamics ensemble every N steps |
| `--wandb` | off | |
| `--wandb_project` | mbpo-student-version | |

(SAC inside MBPO uses `SACConfig` defaults for lr/gamma/tau/alpha except action scale/bias from the env; not all SAC flags are exposed on this CLI.)

### `mlp_dreamer.mlp_dreamer_runner`

| flag | default | meaning |
|------|---------|---------|
| `--env` | Pendulum-v1 | env id |
| `--seed` | 42 | rng |
| `--max_steps` | 100000 | total steps |
| `--prefill_steps` | 1000 | random prefill |
| `--eval_freq` | 2000 | eval every N steps |
| `--batch_size` | 256 | sequence batch size |
| `--seq_len` | 50 | time length sampled from replay for WM + AC |
| `--wandb` | off | |
| `--wandb_project` | dreamer-student-version | |
| `--horizon` | *(DreamerConfig)* | imagination steps; only if passed |
| `--gamma` | *(DreamerConfig)* | discount |
| `--lambda_` | *(DreamerConfig)* | TD(λ) |
| `--kl_beta` | *(DreamerConfig)* | KL weight |
| `--wm_lr` | *(DreamerConfig)* | world model Adam lr |
| `--actor_lr` | *(DreamerConfig)* | |
| `--critic_lr` | *(DreamerConfig)* | |

---

## How to run

### Local (terminal)

`cd` into the folder that contains `pyproject.toml` (this repo root). Optional: `python3 -m venv .venv` then `source .venv/bin/activate` (Mac/Linux). Install deps (see Dependencies), then:

```bash
python -m sac.sac_runner --env Pendulum-v1
python -m mbpo.mbpo_runner --env Pendulum-v1
python -m mlp_dreamer.mlp_dreamer_runner --env Pendulum-v1
```

Add flags from the CLI section above. Same shell, same working directory each time.

### Google Colab

1. Put this project on Colab: upload a zip and unzip, or `git clone` if you have a remote, or mount Drive and `%cd` into the folder that has `pyproject.toml`.
2. In a cell:

```python
%cd /content/continuous_vector_lowdim_sac_mbpo_mlpdreamer   # change path to your folder
!pip install -e .
```

3. Run a runner in the next cell (use `!` because it is a subprocess):

```text
!python -m sac.sac_runner --env Pendulum-v1 --max_steps 5000
```

Colab usually has CPU or one GPU; `torch` may already be there—if `pip install -e .` upgrades it, that is fine. For short smoke tests lower `--max_steps` / `--prefill_steps`. If an env is missing, install the matching gymnasium extra (e.g. classic control) with `pip` in a cell.

## Directory structure

```text
.
├── pyproject.toml
├── README.md
├── report.pdf
├── .gitignore
├── sac/
│   ├── __init__.py
│   ├── sac_model.py
│   ├── sac_agent.py
│   └── sac_runner.py
├── mbpo/
│   ├── __init__.py
│   ├── mbpo_model.py
│   ├── mbpo_agent.py
│   └── mbpo_runner.py
└── mlp_dreamer/
    ├── __init__.py
    ├── mlp_dreamer_model.py
    ├── mlp_dreamer_agent.py
    └── mlp_dreamer_runner.py
```

(Local `.venv/` if you make one is gitignored.)
