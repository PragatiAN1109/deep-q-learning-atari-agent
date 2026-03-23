"""
scripts/train_and_plot.py
--------------------------
Runs REAL DQN training on LunarLander-v3 for 500 episodes and
saves a reward vs episode plot from actual training data.

All logic is self-contained — no imports from src/ needed.
Outputs:
  experiments/metrics/real_training_log.csv
  docs/figures/reward_vs_episode.png

Usage:
    python scripts/train_and_plot.py
    python scripts/train_and_plot.py --episodes 300
"""

import os, sys, csv, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Hyperparameters ───────────────────────────────────────────
LR              = 0.0005
GAMMA           = 0.99
EPS_START       = 1.0
EPS_END         = 0.01
DECAY_RATE      = 0.005      # exponential: eps = max(end, start*exp(-rate*ep))
BATCH_SIZE      = 64
BUFFER_CAPACITY = 50000
HIDDEN          = 128
TARGET_SYNC     = 10
SEED            = 42


# ── Q-Network ─────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,    hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x): return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))
    def __len__(self): return len(self.buf)


# ── Training loop ─────────────────────────────────────────────
def train(n_episodes=500, max_steps=1000):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cpu")

    env        = gym.make("LunarLander-v3", render_mode=None)
    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.n               # 4

    online = DQN(state_dim, action_dim, HIDDEN).to(device)
    target = DQN(state_dim, action_dim, HIDDEN).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer  = optim.Adam(online.parameters(), lr=LR)
    loss_fn    = nn.MSELoss()
    buf        = ReplayBuffer(BUFFER_CAPACITY)

    rewards_log, steps_log, eps_log = [], [], []
    recent = deque(maxlen=100)

    # CSV setup
    csv_dir  = os.path.join(ROOT, "experiments", "metrics")
    csv_path = os.path.join(csv_dir, "real_training_log.csv")
    os.makedirs(csv_dir, exist_ok=True)
    csvf   = open(csv_path, "w", newline="")
    writer = csv.writer(csvf)
    writer.writerow(["episode","reward","avg50","avg100","epsilon","steps"])

    print(f"\n{'='*55}")
    print(f"  Real DQN Training — LunarLander-v3")
    print(f"  Episodes: {n_episodes} | LR: {LR} | γ: {GAMMA}")
    print(f"  Device  : {device}")
    print(f"{'='*55}")

    for ep in range(n_episodes):
        state, _ = env.reset(seed=SEED + ep)
        state    = np.array(state, dtype=np.float32)
        epsilon  = max(EPS_END, EPS_START * math.exp(-DECAY_RATE * ep))
        ep_reward, steps = 0.0, 0

        for _ in range(max_steps):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0)
                    action = int(online(st).argmax(1).item())

            ns, reward, term, trunc, _ = env.step(action)
            ns   = np.array(ns, dtype=np.float32)
            done = term or trunc
            buf.push(state, action, reward, ns, done)

            # Learn once buffer is ready
            if len(buf) >= BATCH_SIZE:
                s,a,r,ns_,d_ = buf.sample(BATCH_SIZE)
                st_  = torch.FloatTensor(s)
                at_  = torch.LongTensor(a)
                rt_  = torch.FloatTensor(r)
                nst_ = torch.FloatTensor(ns_)
                dt_  = torch.FloatTensor(d_)

                q_cur = online(st_).gather(1, at_.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_tgt = rt_ + GAMMA * target(nst_).max(1)[0] * (1 - dt_)

                loss = loss_fn(q_cur, q_tgt)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                optimizer.step()

            ep_reward += reward
            state = ns
            steps += 1
            if done: break

        # Sync target network
        if (ep + 1) % TARGET_SYNC == 0:
            target.load_state_dict(online.state_dict())

        rewards_log.append(ep_reward)
        steps_log.append(steps)
        eps_log.append(epsilon)
        recent.append(ep_reward)

        avg50  = float(np.mean(rewards_log[-50:]))
        avg100 = float(np.mean(rewards_log[-100:]))
        writer.writerow([ep+1, round(ep_reward,2), round(avg50,2),
                         round(avg100,2), round(epsilon,5), steps])
        csvf.flush()

        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep+1:>4} | reward {ep_reward:>8.1f} | "
                  f"avg50 {avg50:>7.1f} | avg100 {avg100:>7.1f} | "
                  f"ε {epsilon:.4f}")

        if avg100 >= 200 and len(recent) == 100:
            print(f"\n  🎉 Solved at episode {ep+1}! avg100={avg100:.1f}\n")
            break

    env.close(); csvf.close()
    print(f"\n  CSV saved → {csv_path}")
    return rewards_log, steps_log, eps_log, csv_path


# ── Plot from real CSV data ───────────────────────────────────
def plot_from_csv(csv_path, out_path):
    episodes, rewards, avg50s, avg100s, epsilons = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            avg50s.append(float(row["avg50"]))
            avg100s.append(float(row["avg100"]))
            epsilons.append(float(row["epsilon"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios":[3,1]},
                                    facecolor="#fafafa")
    fig.subplots_adjust(hspace=0.38, top=0.91, bottom=0.08,
                        left=0.08, right=0.97)

    # ── Rewards panel ─────────────────────────────────────────
    ax1.set_facecolor("#f7f9fc")
    ax1.plot(episodes, rewards, color="steelblue", alpha=0.35,
             linewidth=0.7, label="Episode reward")
    ax1.plot(episodes, avg50s,  color="darkorange", linewidth=2.2,
             label="Moving avg (50 ep)")
    ax1.plot(episodes, avg100s, color="crimson", linewidth=1.8,
             linestyle="--", label="Moving avg (100 ep)")
    ax1.axhline(200, color="green", linestyle=":", linewidth=1.8,
                alpha=0.9, label="Solve threshold (200)")
    ax1.fill_between(episodes, 200, max(rewards)+20,
                     alpha=0.05, color="green")

    # Annotate if solved
    solved = [e for e, a in zip(episodes, avg100s) if a >= 200]
    if solved:
        se = solved[0]
        ax1.annotate(f"Solved at ep {se}",
                     xy=(se, 200), xytext=(se - len(episodes)*0.18, 240),
                     fontsize=9, color="green", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="green"))

    ax1.set_xlim(0, episodes[-1] + 5)
    ax1.set_ylabel("Total Reward", fontsize=11)
    ax1.set_title(
        f"Deep Q-Network — LunarLander-v3\n"
        f"Real Training Run  ({episodes[-1]} Episodes, Seed={SEED})",
        fontsize=13, fontweight="bold", pad=8, color="#222222")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.tick_params(labelbottom=False)

    # ── Epsilon panel ─────────────────────────────────────────
    ax2.set_facecolor("#f7f9fc")
    ax2.fill_between(episodes, epsilons, alpha=0.15, color="purple")
    ax2.plot(episodes, epsilons, color="purple", linewidth=1.8,
             label="Epsilon (ε)")
    ax2.axhline(EPS_END, color="#888", linestyle=":", linewidth=1.1,
                label=f"ε_min = {EPS_END}")
    ax2.set_xlim(0, episodes[-1] + 5)
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Epsilon", fontsize=10)
    ax2.set_title("Exploration Rate (ε) Decay", fontsize=10,
                  color="#555", pad=3)
    ax2.legend(fontsize=8.5, framealpha=0.85)
    ax2.grid(True, alpha=0.25, linestyle="--")

    # Footer
    final_avg = avg100s[-1]
    peak_avg  = max(avg100s)
    fig.text(0.08, 0.005,
             f"Final avg100: {final_avg:.1f}  |  "
             f"Peak avg100: {peak_avg:.1f}  |  "
             f"LR={LR}  γ={GAMMA}  decay={DECAY_RATE}  "
             f"buffer={BUFFER_CAPACITY}  batch={BATCH_SIZE}",
             fontsize=8, color="#666", ha="left")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {out_path}")
    print(f"  Peak avg100  : {peak_avg:.1f}")
    print(f"  Final avg100 : {final_avg:.1f}")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args()

    rewards, steps, epsilons, csv_path = train(args.episodes)

    out = os.path.join(ROOT, "docs", "figures", "reward_vs_episode.png")
    plot_from_csv(csv_path, out)

    print(f"\n{'='*55}")
    print("  ✅  Done.")
    print(f"  CSV  → {csv_path}")
    print(f"  Plot → {out}")
    print(f"{'='*55}\n")
