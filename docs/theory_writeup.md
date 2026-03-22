# Theory & Design Writeup — Deep Q-Learning Agent

**Repository:** https://github.com/PragatiAN1109/deep-q-learning-atari-agent
**Environment:** LunarLander-v3 (Gymnasium)
**Author:** Pragati Narote | **Date:** March 2026

---

# 1. Q-Learning Theory

## 1.1 Value-Based vs Policy-Based Methods

Reinforcement learning algorithms fall into two families based on what they learn to represent.

**Policy-based methods** (e.g., REINFORCE, PPO) directly learn a function
π(a|s) that maps states to action probabilities. The agent learns *how to act*
without reasoning about the value of each state.

**Value-based methods** (e.g., Q-Learning, DQN) learn a value function — a
score estimating the cumulative reward reachable from a given situation. The
agent then acts greedily with respect to that score. **This project is entirely
value-based.**

**Where Q-Learning fits:** Q-Learning learns an *action-value function*
Q(s, a): the expected discounted reward for taking action `a` in state `s` and
following the optimal policy thereafter. Rather than learning "what to do", it
learns "how good each action is", then picks the best.

---

## 1.2 The Bellman Equation

### Intuitive explanation

Imagine the lander hovering above the pad. The value of your current position
is not only the reward you receive right now — it is that reward *plus* the best
possible reward achievable from all future steps. The Bellman equation captures
this recursive relationship mathematically.

### Mathematical form

The optimal Q-function satisfies:

```
Q*(s, a) = E[ r  +  γ · max_{a'} Q*(s', a') ]
```

| Symbol | Meaning in this repo |
|---|---|
| `s` | 8-dim observation vector (position, velocity, angle, leg contacts) |
| `a` | Action integer 0–3 (engine command) |
| `r` | Scalar reward from `env.step(action)` |
| `s'` | Next observation returned by the environment |
| `γ` | Discount factor — `0.99` in `config.yaml` |
| `max_{a'} Q*(s', a')` | Best Q-value achievable from the next state |

### Implementation in this repo (`src/agent/dqn_agent.py`)

```python
with torch.no_grad():
    q_next   = self.target_net(next_states_t).max(dim=1)[0]   # max over actions
    q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)
```

The `(1 - done)` term masks the future reward when the episode ends — no next
state exists after a crash or landing. The online network trains toward this
target using MSE loss. This single equation is the entire learning signal.

---

# 2. RL Agent vs LLM: Decision-Making Paradigms

## 2.1 Core Difference

| Dimension | RL Agent (this project) | Large Language Model |
|---|---|---|
| Input | 8-float state vector | Token sequence (text prompt) |
| Output | Integer action 0–3 | Next token / text continuation |
| Feedback | Scalar reward from physics engine | None at inference |
| Goal | Maximise cumulative discounted reward | Produce fluent, helpful text |
| Memory | Replay buffer (50,000 transitions) | Context window only |
| Learning | Online, via gradient updates during training | Offline, during pretraining / fine-tuning |

## 2.2 Concrete Example

**RL agent step:**
```
State:  [0.01, 0.85, -0.02, -0.31, 0.00, 0.01, 0.0, 0.0]
Q-vals: [12.1, 8.3, 18.4, 9.2]   → pick action 2 (fire main engine)
Reward: +0.3 received from environment
Stored: (s, 2, 0.3, s', False) in replay buffer
```

**LLM equivalent:**
```
Prompt:   "What should a lunar lander do when falling too fast?"
Response: "It should fire its main engine to decelerate..."
Effect:   None — no consequence, no weight update, no feedback
```

The RL agent **acts and learns**. The LLM **describes and generates**.
This distinction is fundamental.

---

# 3. Planning Differences

## 3.1 How the DQN Agent Plans

The DQN agent in this project does not plan explicitly — it does not simulate
future trajectories at inference time. Instead, it encodes an **implicit plan**
inside its Q-network weights.

After training, picking `argmax Q(s, ·)` asks: *"Given everything I have
learned, which action leads to the best long-term outcome?"* The plan is
compressed into 18,308 parameters (the MLP in `src/model.py`).

Target network synchronisation (`target_update_freq: 10` in `config.yaml`)
prevents the Bellman target from shifting every step — stabilising the implicit
plan as it converges.

## 3.2 How LLMs "Plan"

LLMs approximate planning through **chain-of-thought reasoning**: generating
intermediate reasoning steps as tokens before a final answer. Each token
conditions all subsequent ones, enabling step-by-step decomposition.

This is not true planning — the LLM cannot execute actions, observe real
outcomes, or revise decisions based on real-world feedback. It reasons
*in-context only*.

## 3.3 Comparison

| Dimension | DQN Agent | LLM |
|---|---|---|
| Planning horizon | Implicit via γ=0.99 (~200+ steps) | Bounded by context window |
| Feedback | Real environment rewards every step | None at inference |
| Revision | Every episode via gradient updates | Frozen at inference |
| Uncertainty | ε-greedy exploration (ε: 1.0 → 0.01) | Token probability distributions |
| Commitment | Must pick one action per timestep | Generates distributions, no real consequence |

---

# 4. Expected Lifetime Value and Discounted Rewards

## 4.1 Why Discount?

An agent that weights all future rewards equally faces two problems:
1. An infinite sum of rewards may **diverge mathematically**
2. Rewards far in the future are **more uncertain** — the episode may end,
   the environment may change

Discounting solves both. The discount factor γ ∈ [0, 1] exponentially reduces
the weight of distant rewards:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^∞  γ^k · r_{t+k}
```

## 4.2 Long-Term vs Short-Term Tradeoff

| γ value | Behaviour | Problem |
|---|---|---|
| 0.0 | Fully myopic — only cares about immediate reward | Never saves fuel to prevent later crash |
| 0.8 (original assignment) | 0.8^200 ≈ 0 — terminal rewards invisible | Landing bonus has zero influence on early steps |
| 0.99 (this repo) | 0.99^200 ≈ 0.13 — terminal rewards still meaningful | Balanced: fuel penalty and landing bonus both matter |
| 1.0 | All future rewards equal weight | May diverge in long episodes |

## 4.3 The γ = 0.99 Justification

LunarLander episodes average 200–400 steps. The landing bonus (+100 to +140)
arrives at the very end. With γ = 0.99, that bonus retains ~13% of its value
when viewed from step 0 — still strong enough to propagate back through 200
Bellman updates and influence early-episode decisions.

With γ = 0.8 (the assignment's original value), the landing bonus is
effectively invisible to early actions, and the agent cannot learn to plan a
safe approach trajectory. This is why the baseline config uses γ = 0.99.

---

# 5. RL for LLM Agents — RLHF

## 5.1 What is RLHF?

Reinforcement Learning from Human Feedback (RLHF) is the technique that
aligns large language models (GPT-4, Claude, Gemini) with human preferences.
It bridges the gap between the supervised pretraining objective (predict next
token) and the actual goal (produce helpful, safe, accurate responses).

**Three-stage pipeline:**

**Stage 1 — Supervised Fine-Tuning (SFT):**
Train the LLM on high-quality human-written demonstrations of desired behaviour.

**Stage 2 — Reward Model Training:**
Collect human preference comparisons between pairs of LLM outputs. Train a
separate neural network (the reward model) to predict which response a human
would prefer. This model replaces the environment in standard RL.

**Stage 3 — RL Fine-Tuning (PPO):**
Use the reward model as the reward signal. Fine-tune the LLM using Proximal
Policy Optimisation (PPO) to produce responses that score highly on human
preferences.

## 5.2 Connection to This Project

| Concept | DQN (this repo) | RLHF |
|---|---|---|
| Agent | DQN online network | LLM policy |
| Environment | LunarLander-v3 physics | Human preference reward model |
| Reward | Physics-based scalar | Reward model score |
| Experience storage | Replay buffer (50k transitions) | Rollout buffer |
| Update algorithm | Bellman MSE + Adam | PPO |
| Exploration | ε-greedy | Temperature sampling |

Both use the same conceptual loop: **generate behaviour → receive reward →
store experience → update policy**. The difference is scale and reward source.

## 5.3 Modern Feedback Loops in LLM Training

**Constitutional AI (Anthropic):** The model critiques and revises its own
outputs using a set of principles — analogous to self-play in RL.

**Process Reward Models:** Reward correct intermediate reasoning steps, not
just final answers — closer to the per-step dense reward in this DQN project
than to sparse terminal rewards.

**RLAIF (RL from AI Feedback):** A stronger AI model replaces human labellers
as the reward source. More scalable but introduces distribution shift risks
analogous to replay buffer staleness in DQN.

---

# 6. Architecture Design: DQN + LLM Integration

## 6.1 Motivation

This project's DQN agent is highly effective at low-level control: deciding
which engine to fire at each timestep. But it cannot explain its decisions,
understand high-level goals expressed in natural language, or adapt to novel
instructions without retraining.

An LLM can reason about goals, interpret language, and generate
explanations — but cannot control a physical system step-by-step. Combining
them creates a system where each component does what it does best.

## 6.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LAYER 1 — Language Interface            │
│                           (LLM)                             │
│  Input : Natural language goal from user                    │
│  Output: Reward shaping parameters + sub-goal constraints   │
│  e.g.  : "Land gently near the left flag"                   │
│         → leg_bonus_weight=3.0, target_x=-0.3               │
└───────────────────────┬─────────────────────────────────────┘
                        │  reward config
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 2 — Decision Engine                 │
│                  (DQN — this project's core)                │
│  Input : 8-dim state vector every timestep                  │
│  Output: Discrete action (0–3) every timestep               │
│  Runs  : src/train.py training loop                         │
│  Uses  : src/env/reward_shaping.py for shaped rewards       │
└───────────────────────┬─────────────────────────────────────┘
                        │  episode trajectory
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3 — Explanation & Feedback               │
│                           (LLM)                             │
│  Input : Episode trajectory (states, actions, rewards)      │
│  Output: Natural language post-mortem + reward adjustments  │
│  e.g.  : "Agent fired main engine too late — increase       │
│           velocity penalty below y=0.3"                     │
└─────────────────────────────────────────────────────────────┘
                        │  updated goal / reward params
                        └──────────────► back to Layer 1
```

## 6.3 Data Flow

1. User provides natural language goal (Layer 1 LLM)
2. LLM produces reward shaping parameters → updates `ShapedLunarLander` wrapper
3. DQN trains for N episodes using the shaped reward (Layer 2)
4. Episode trajectories logged to `experiments/baseline_run/training_metrics.csv`
5. LLM analyses trajectories and generates explanation + parameter suggestions (Layer 3)
6. Human reviews; updated goal sent back to Layer 1

At inference (per timestep): State (8 floats) → DQN → Action (integer 0–3).
The LLM operates only at episode/session boundaries — not in the per-step
inference path, avoiding latency overhead.

## 6.4 Use Cases

**Adaptive game AI:** An LLM interprets player feedback ("the landing was too
rough") and automatically adjusts `src/env/reward_shaping.py` parameters between
episodes — no manual reward engineering needed.

**Explainable autonomous systems:** DQN controls drone stabilisation at 50Hz.
After each flight, an LLM analyses Q-value trajectories and explains which
flight phases were risky or suboptimal in plain English.

**Natural language RL interface:** Non-technical users specify landing
constraints in English. The LLM translates to CLI flags and re-triggers
`python src/experiments/run_all.py` automatically.

## 6.5 Why This Architecture Matters

This design separates concerns cleanly:
- **DQN** handles real-time, high-frequency, reward-driven optimisation
- **LLM** handles language understanding, goal interpretation, and explanation

Neither replaces the other. This is the design principle behind SayCan
(Google DeepMind), VOYAGER (Minecraft LLM agent with skill library), and
ReAct — all use LLMs for high-level reasoning and learned controllers for
low-level execution.

---

# Summary

This project implements a Deep Q-Network agent that learns to land a spacecraft
using only an 8-dimensional sensor vector and a dense reward signal. Every
design choice — the MLP architecture, γ=0.99, ε-greedy exploration, experience
replay, the target network — is grounded in the theoretical framework described
above.

The **Bellman equation** is the mathematical heart: it defines what the
Q-network is trained to approximate. **Experience replay** breaks temporal
correlation between training samples. The **target network** provides stable
Bellman targets. Together, these innovations — first described in Mnih et al.
(2015) — make value-based deep RL tractable on real problems.

The connection to LLMs is not superficial. Both DQN and RLHF use the same
fundamental loop: generate behaviour, receive a reward signal, store experience,
update the policy. The difference is scale, action space, and reward source.
Understanding DQN deeply is foundational to understanding how the most capable
AI systems in the world are trained today.

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533.
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS.
- Ahn et al. (2022). *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan).* Google DeepMind.
- Wang et al. (2023). *VOYAGER: An Open-Ended Embodied Agent with Large Language Models.*
- Gymnasium Documentation: https://gymnasium.farama.org/environments/box2d/lunar_lander/
