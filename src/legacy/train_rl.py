"""
TD3 Training Script — Phase 5, Section 4 (Phase5_RL_Spec.md)

Trains a TD3 agent on NPIDTuningEnv in either static (drift=False) or
drift (drift=True) mode, then saves the trained model.

Usage:
  python src/legacy/train_rl.py                    # Week 6: static case
  python src/legacy/train_rl.py --drift            # Week 7: drift case
  python src/legacy/train_rl.py --drift --finetune # Week 7: fine-tune from static agent

Hyperparameters are spec §4.2 defaults — adjust total_timesteps if
the reward curve hasn't converged (see diagnostics).
"""

import argparse
import os
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.legacy.rl_env import NPIDTuningEnv


# ---------------------------------------------------------------------------
# Reproducibility seeds (spec §6.3)
# ---------------------------------------------------------------------------
SEED = 42


class RewardLoggerCallback(BaseCallback):
    """
    Logs episode reward and rolling mean to a list for post-training plotting.
    Saved alongside the model so rl_diagnostics.py can reload it.
    """

    def __init__(self, log_path: str, rolling_window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.rolling_window = rolling_window
        self._episode_rewards: list = []
        self._episode_timesteps: list = []
        self._current_ep_reward: float = 0.0
        self._n_episodes: int = 0

    def _on_step(self) -> bool:
        # SB3 Monitor wrapper populates 'episode' key in info when episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self._episode_rewards.append(ep_reward)
                self._episode_timesteps.append(self.num_timesteps)
                self._n_episodes += 1

                if self.verbose >= 1 and self._n_episodes % 100 == 0:
                    recent = self._episode_rewards[-self.rolling_window:]
                    print(f"  [ep {self._n_episodes:5d}] "
                          f"timestep={self.num_timesteps:7d}  "
                          f"ep_reward={ep_reward:.3f}  "
                          f"rolling_mean({self.rolling_window})={np.mean(recent):.3f}")
        return True

    def _on_training_end(self) -> None:
        np.savez(
            self.log_path,
            episode_rewards=np.array(self._episode_rewards),
            episode_timesteps=np.array(self._episode_timesteps),
        )
        print(f"\n  Training log saved to: {self.log_path}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    drift: bool = False,
    finetune_from: str | None = None,
    total_timesteps: int = 300_000,
    episode_length: float | None = None,
):
    """
    Train (or fine-tune) a TD3 agent on NPIDTuningEnv.

    Args:
        drift:          Whether to enable thermal resistance drift (Week 7).
        finetune_from:  Path to an existing .zip model to fine-tune from.
        total_timesteps: Number of environment steps to train for.
        episode_length:  Override episode duration (auto-set based on drift if None).
    """
    if episode_length is None:
        episode_length = 20.0 if drift else 4.0

    tag = "drift" if drift else "static"
    model_save_path = f"models/legacy/td3_npid_{tag}"
    log_path = f"results/legacy/tables/td3_{tag}_training_log"

    os.makedirs("results/legacy/tables", exist_ok=True)
    os.makedirs("results/legacy/figures", exist_ok=True)

    print("=" * 65)
    print(f"  TD3 Training -- {tag.upper()} case")
    print(f"  drift={drift}, episode_length={episode_length}s")
    print(f"  total_timesteps={total_timesteps:,}")
    print(f"  seed={SEED}")
    print(f"  save_path: {model_save_path}.zip")
    print("=" * 65)

    # ── Environment setup ────────────────────────────────────────────────────
    env = NPIDTuningEnv(
        drift=drift,
        k_heat=0.05,
        episode_length=episode_length,
        seed=SEED,
    )
    env = Monitor(env)   # wraps env so SB3 logs episode reward/length

    # Evaluation env — fixed setpoint 20 rad/s, no drift
    eval_env = NPIDTuningEnv(
        drift=False,
        episode_length=episode_length,
        seed=SEED + 1,
        omega_ref_range=(20.0, 20.0),
    )
    eval_env = Monitor(eval_env)

    # ── Action noise (spec §4.2) ──────────────────────────────────────────────
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    reward_logger = RewardLoggerCallback(log_path=log_path, rolling_window=50, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/legacy/",
        log_path=f"results/legacy/tables/td3_{tag}_eval_log",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    # ── Model setup ───────────────────────────────────────────────────────────
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if finetune_from and os.path.exists(finetune_from + ".zip"):
        print(f"\n  Fine-tuning from: {finetune_from}.zip\n")
        model = TD3.load(
            finetune_from,
            env=env,
            # Keep existing buffer but re-attach new env
        )
        model.set_env(env)
    else:
        model = TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=SEED,
            verbose=1,
            tensorboard_log="./results/legacy/tb_logs/",
        )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n  Starting training ({total_timesteps:,} timesteps)...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, eval_callback],
        reset_num_timesteps=True,
    )

    model.save(model_save_path)
    print(f"\n  Model saved to: {model_save_path}.zip")
    print(f"  Training log: {log_path}.npz")
    return model, log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TD3 agent for N-PID gain tuning (Phase 5)"
    )
    parser.add_argument(
        "--drift", action="store_true",
        help="Enable drift training (Week 7). Default: static case (Week 6)."
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help="Fine-tune from the static case agent (td3_npid_static.zip)."
    )
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="Total training timesteps (default: 300000)."
    )
    parser.add_argument(
        "--episode-length", type=float, default=None,
        help="Override episode length (s). Auto: 4.0 static, 20.0 drift."
    )
    args = parser.parse_args()

    finetune_path = "models/legacy/td3_npid_static" if args.finetune else None

    train(
        drift=args.drift,
        finetune_from=finetune_path,
        total_timesteps=args.timesteps,
        episode_length=args.episode_length,
    )
