"""Double DQN training loop with checkpointing and validation."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import RouterConfig, TrainingConfig
from ddqn_router.dataset.dataset import Task, load_tasks
from ddqn_router.env.routing_env import RoutingEnv
from ddqn_router.eval.evaluator import evaluate_routing
from ddqn_router.rl.q_network import QNetwork
from ddqn_router.rl.replay_buffer import ReplayBuffer, Transition
from ddqn_router.rl.state_encoder import StateEncoder


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_epsilon(step: int, cfg: TrainingConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def _select_action(
    q_net: QNetwork,
    state: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        valid = np.where(action_mask)[0]
        return int(np.random.choice(valid))

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_t).squeeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
        q_values[~mask_t] = float("-inf")
        return int(q_values.argmax().item())


def _run_episode_for_eval(
    q_net: QNetwork,
    task: Task,
    encoder: StateEncoder,
    env: RoutingEnv,
    device: torch.device,
) -> set[int]:
    """Run a greedy episode (epsilon=0) and return selected agents."""
    tfidf_vec = encoder.transform(task["text"])
    state = env.reset(tfidf_vec, task["required_agents"])
    done = False
    while not done:
        action_mask = env.get_action_mask()
        action = _select_action(q_net, state, action_mask, 0.0, env.num_actions, device)
        state, _reward, done = env.step(action)
    return env.selected_agents


def train(config: RouterConfig) -> dict:
    cfg = config.training
    _set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = AgentRegistry(config.agents)
    num_agents = registry.num_agents

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config.dataset.input).parent
    train_tasks = load_tasks(data_dir / "train.jsonl")
    val_tasks = load_tasks(data_dir / "val.jsonl")
    test_tasks = load_tasks(data_dir / "test.jsonl")

    if not train_tasks:
        raise ValueError("Training set is empty")

    encoder = StateEncoder(max_features=cfg.tfidf_max_features)
    encoder.fit([t["text"] for t in train_tasks])

    tfidf_dim = encoder.dim
    q_online = QNetwork(tfidf_dim, num_agents, cfg.hidden_layers).to(device)
    q_target = QNetwork(tfidf_dim, num_agents, cfg.hidden_layers).to(device)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_online.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    replay = ReplayBuffer(cfg.replay_buffer_size)

    env = RoutingEnv(
        num_agents=num_agents,
        reward_mode=cfg.reward_mode,
        step_cost=cfg.step_cost,
        max_steps=cfg.max_steps_per_episode,
        action_masking=cfg.action_masking,
    )

    best_val_jaccard = -1.0
    best_val_metrics: dict = {}
    training_log: list[dict] = []
    global_step = 0
    total_loss = 0.0
    loss_count = 0

    t_start = time.time()
    print(f"\n  Training DDQN router ({cfg.total_steps} steps, device={device})")
    print(f"  {'─' * 60}")

    while global_step < cfg.total_steps:
        task = random.choice(train_tasks)
        tfidf_vec = encoder.transform(task["text"])
        state = env.reset(tfidf_vec, task["required_agents"])
        done = False

        while not done and global_step < cfg.total_steps:
            epsilon = _get_epsilon(global_step, cfg)
            action_mask = env.get_action_mask()
            action = _select_action(
                q_online, state, action_mask, epsilon, env.num_actions, device
            )
            next_state, reward, done = env.step(action)
            next_mask = env.get_action_mask()

            replay.add(
                Transition(state, action, reward, next_state, done, next_mask)
            )
            state = next_state
            global_step += 1

            # Training step
            if len(replay) >= cfg.min_replay_size:
                batch = replay.sample(cfg.batch_size)
                states_b = torch.tensor(
                    np.array([t.state for t in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                actions_b = torch.tensor(
                    [t.action for t in batch], dtype=torch.long, device=device
                )
                rewards_b = torch.tensor(
                    [t.reward for t in batch], dtype=torch.float32, device=device
                )
                next_states_b = torch.tensor(
                    np.array([t.next_state for t in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                dones_b = torch.tensor(
                    [t.done for t in batch], dtype=torch.float32, device=device
                )
                next_masks_b = torch.tensor(
                    np.array([t.mask for t in batch]),
                    dtype=torch.bool,
                    device=device,
                )

                # Double DQN: action selection from online, value from target
                with torch.no_grad():
                    q_online_next = q_online(next_states_b)
                    q_online_next[~next_masks_b] = float("-inf")
                    best_actions = q_online_next.argmax(dim=1)

                    q_target_next = q_target(next_states_b)
                    target_vals = q_target_next.gather(
                        1, best_actions.unsqueeze(1)
                    ).squeeze(1)
                    td_target = rewards_b + cfg.gamma * target_vals * (1 - dones_b)

                q_current = q_online(states_b).gather(
                    1, actions_b.unsqueeze(1)
                ).squeeze(1)
                loss = loss_fn(q_current, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loss_count += 1

            # Target network sync
            if global_step % cfg.target_update_freq == 0:
                q_target.load_state_dict(q_online.state_dict())

            # Validation eval
            if global_step % cfg.val_eval_freq == 0 and val_tasks:
                val_preds = []
                val_targets = []
                for vt in val_tasks:
                    pred = _run_episode_for_eval(q_online, vt, encoder, env, device)
                    val_preds.append(pred)
                    val_targets.append(set(vt["required_agents"]))

                val_metrics = evaluate_routing(val_preds, val_targets)
                avg_loss = total_loss / loss_count if loss_count else 0.0
                elapsed = time.time() - t_start

                log_entry = {
                    "step": global_step,
                    "epsilon": round(epsilon, 4),
                    "avg_loss": round(avg_loss, 6),
                    "val_jaccard": round(val_metrics["mean_jaccard"], 4),
                    "val_f1": round(val_metrics["mean_f1"], 4),
                    "elapsed_s": round(elapsed, 1),
                }
                training_log.append(log_entry)

                sys.stdout.write(
                    f"\r  step {global_step:>7d}/{cfg.total_steps}"
                    f"  eps={epsilon:.3f}"
                    f"  loss={avg_loss:.5f}"
                    f"  val_jacc={val_metrics['mean_jaccard']:.4f}"
                    f"  val_f1={val_metrics['mean_f1']:.4f}"
                    f"  [{elapsed:.0f}s]"
                )
                sys.stdout.flush()

                total_loss = 0.0
                loss_count = 0

                if cfg.save_best and val_metrics["mean_jaccard"] > best_val_jaccard:
                    best_val_jaccard = val_metrics["mean_jaccard"]
                    best_val_metrics = val_metrics
                    torch.save(q_online.state_dict(), output_dir / "model.pt")
                    encoder.save(output_dir / "encoder.joblib")

    print()

    # Save final artifacts if no best-save happened
    if not (output_dir / "model.pt").exists():
        torch.save(q_online.state_dict(), output_dir / "model.pt")
        encoder.save(output_dir / "encoder.joblib")

    # Final test evaluation
    if test_tasks:
        q_eval = QNetwork(tfidf_dim, num_agents, cfg.hidden_layers).to(device)
        if (output_dir / "model.pt").exists():
            q_eval.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
        else:
            q_eval.load_state_dict(q_online.state_dict())
        q_eval.eval()

        test_preds = []
        test_targets = []
        for tt in test_tasks:
            pred = _run_episode_for_eval(q_eval, tt, encoder, env, device)
            test_preds.append(pred)
            test_targets.append(set(tt["required_agents"]))
        test_metrics = evaluate_routing(test_preds, test_targets)
    else:
        test_metrics = {}

    # Save config snapshot
    with open(output_dir / "config_used.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    with open(output_dir / "metrics_val_best.json", "w") as f:
        json.dump(best_val_metrics, f, indent=2)

    with open(output_dir / "metrics_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(output_dir / "training_log.jsonl", "w") as f:
        for entry in training_log:
            f.write(json.dumps(entry) + "\n")

    print(f"\n  Artifacts saved to {output_dir}/")
    if test_metrics:
        print(f"  Test Jaccard: {test_metrics.get('mean_jaccard', 0):.4f}")
        print(f"  Test F1:      {test_metrics.get('mean_f1', 0):.4f}")

    return test_metrics
