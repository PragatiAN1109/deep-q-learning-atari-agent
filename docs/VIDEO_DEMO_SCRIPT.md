# 🎬 VIDEO DEMO SCRIPT
# Deep Q-Learning Agent — LunarLander-v3
# Duration: 7–9 minutes | Speaker: Pragati Narote
# ─────────────────────────────────────────────────────────────
# FORMAT GUIDE:
#   [SCREEN]   = what to show on screen
#   [SPEAK]    = what to say out loud
#   [ACTION]   = what to click / do
#   ⏱️  X:XX   = running timestamp
# ─────────────────────────────────────────────────────────────


# VIDEO DEMO SCRIPT — Deep Q-Learning Agent (LunarLander-v3)
# Pragati Narote | NEU MSIS | Spring 2025

---

## ⏱️ 0:00 — TITLE SLIDE (30 seconds)

[SCREEN] Show title card:
  "Deep Q-Learning Agent — LunarLander-v3"
  "Pragati Narote | Northeastern University | Spring 2025"

[SPEAK]
"Hi, my name is Pragati Narote. In this video I'll walk you through
my Deep Q-Learning project, where I trained an AI agent to land a
spacecraft using reinforcement learning.

The agent starts knowing absolutely nothing — it takes random actions
and crashes every time. Over 500 training episodes, it learns from
its own experience and figures out how to fire its engines to land
safely on the pad.

Let's walk through how it works."

---

## ⏱️ 0:30 — PROBLEM STATEMENT (45 seconds)

[SCREEN] Show the LunarLander-v3 screenshot from docs/figures/lunarlander_screenshot.png

[SPEAK]
"The environment is LunarLander-v3 from the Gymnasium library.
The lander starts at the top of the screen and needs to touch down
between the two flags without crashing.

At every timestep, the agent receives an 8-dimensional state vector —
that's the lander's x and y position, velocity, angle, angular
velocity, and whether each leg is touching the ground.

It has four possible actions: do nothing, fire the left engine,
fire the main engine, or fire the right engine.

The reward signal is dense — the agent gets small positive rewards
for moving toward the pad and reducing speed, a +10 bonus each time
a leg touches the ground, a +100 to +140 reward for a successful
landing, and a −100 penalty for crashing.

The environment is considered solved when the agent achieves an
average reward of 200 or more over 100 consecutive episodes."

---

## ⏱️ 1:15 — WHY DEEP Q-LEARNING (45 seconds)

[SCREEN] Show the Bellman update flow diagram:
  docs/figures/bellman_update_flow.png

[SPEAK]
"The core algorithm is a Deep Q-Network, or DQN — introduced by
DeepMind in 2015. The key idea is that we train a neural network
to approximate the Q-value function.

Q of s-a gives us the expected future reward if we take action a
from state s and then follow the optimal policy from there.

The network is updated using the Bellman equation shown here.
The target is the immediate reward plus the discounted maximum
Q-value from the next state. We minimize the mean squared error
between what the network predicted and this target.

Two key innovations make this stable: experience replay, which
stores past transitions in a buffer and samples them randomly
to break correlation between updates — and a target network,
which is a frozen copy of the main network used only for computing
the Bellman target, preventing the system from chasing a
moving target."

---

## ⏱️ 2:00 — ARCHITECTURE (30 seconds)

[SCREEN] Show the DQN architecture section from README or draw it:
  Input(8) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(4)

[SPEAK]
"The network architecture is a simple 3-layer multi-layer perceptron.
Input is the 8-dimensional state vector.
Two hidden layers of 128 neurons each with ReLU activations.
Output is 4 Q-values — one for each discrete action.

The agent picks whichever action has the highest Q-value when
exploiting, and takes a random action with probability epsilon
when exploring.

Total trainable parameters: just 18,308. This is intentionally
small and fast — we want the focus to be on the learning algorithm,
not the network size."


---

## ⏱️ 2:30 — TRAINING RESULTS (90 seconds)

[SCREEN] Show the full training dashboard:
  docs/figures/reward_vs_episode.png  (or the 3-panel dashboard)

[SPEAK]
"Let me show you what actually happened during training.

I ran 500 episodes with these baseline hyperparameters:
learning rate 0.0005, discount factor 0.99, batch size 64,
replay buffer of 50,000 transitions, and epsilon decaying
exponentially from 1.0 down to 0.01.

Looking at the reward curve — the blue line is the raw reward
per episode and the orange line is the 50-episode moving average.

In the first 100 episodes, the agent is almost entirely random.
It crashes constantly. The average reward is around negative 200.

From episodes 100 to 250, the replay buffer has enough data
for meaningful updates. The agent starts learning that crashing
is bad — the average begins climbing.

Around episode 150 — right here — epsilon drops below 0.5.
The agent starts exploiting its learned policy more than exploring.
You can see the reward trend accelerate upward.

By episodes 400 to 500, individual episodes are regularly scoring
above 150 and even above 200 — the lander is actually landing.

The best single episode achieved a reward of 281 — that's a
clean, efficient landing well above the solve threshold.

The final 100-episode average was 157. We didn't fully solve
the environment in 500 episodes — the solve threshold is 200 —
but we got a strong upward trend. Given more training time,
around 700 to 800 episodes, the agent would likely solve it."

---

## ⏱️ 4:00 — CONVERGENCE CURVE (30 seconds)

[SCREEN] Show: docs/figures/moving_avg_reward.png

[SPEAK]
"Here's the convergence curve more clearly — the 50-episode
and 100-episode rolling averages plotted together.

The orange curve shows the immediate learning trend.
The red dashed curve is the official solve metric.

The peak 100-episode average reached was 157.
The average over the final 50 episodes was 183 —
showing the agent was still actively improving at episode 500.

This is the characteristic S-curve shape of reinforcement
learning: slow start while exploring, rapid improvement
once the policy stabilizes, then gradual convergence."

---

## ⏱️ 4:30 — HYPERPARAMETER EXPERIMENTS (60 seconds)

[SCREEN] Show the LR sweep plot, then gamma sweep plot

[SPEAK]
"I also ran controlled experiments to understand how
hyperparameters affect learning.

For the learning rate — I tested 0.0001, 0.0005, and 0.001.
The baseline value of 0.0005 performed best overall.
At 0.001, the agent converges faster early but becomes more
unstable. At 0.0001, it's slower but more stable.

For the discount factor gamma — I tested 0.90, 0.95, and 0.99.
Gamma of 0.99 clearly outperforms the others, and here's why.

At gamma 0.8 — which was the original assignment value — the
landing bonus of plus 100 arrives about 200 steps into the episode.
0.8 to the power of 200 is essentially zero. The agent is completely
blind to whether it eventually lands or crashes.

At gamma 0.99, that same landing bonus retains about 13% of its
value when viewed from step zero — enough to drive learning.

This is why I changed gamma from the assignment's 0.8 to 0.99."


---

## ⏱️ 5:30 — EPSILON DECAY & EXPLORATION (45 seconds)

[SCREEN] Show: docs/figures/epsilon_decay.png

[SPEAK]
"This plot shows the epsilon decay schedule — the orange region
is where exploration dominates and the blue region is where
exploitation dominates.

The crossover happens around episode 140, where epsilon
drops below 0.5. Before that point, more than half of actions
are random. After it, the agent mostly acts on what it has learned.

I also compared epsilon-greedy against Softmax, or Boltzmann,
exploration. Softmax converts Q-values into probabilities —
so even when exploring, the agent prefers actions with higher
Q-values rather than choosing completely at random.

In this environment, epsilon-greedy with exponential decay
performed better overall, because the scheduled decay gives
a clear transition from exploration to exploitation that
Softmax at a fixed temperature can't match."

---

## ⏱️ 6:15 — PERFORMANCE METRICS SUMMARY (45 seconds)

[SCREEN] Show the performance metrics bar chart and table:
  docs/figures/ (performance summary figure from notebook)

[SPEAK]
"Here are the final quantitative results.

Average reward over the last 100 episodes: 157.
Average steps per episode: 508.
Maximum reward achieved in a single episode: 281.85.

Looking at the bar chart — the agent's average reward by training
phase tells the story clearly. Early training averaged around
negative 156. By the final 100 episodes, the average was
positive 157 — a swing of over 300 reward points.

The agent went from crashing every time to landing successfully
in many episodes, with some runs achieving rewards well above
the solve threshold of 200."

---

## ⏱️ 7:00 — LLM + DQN ARCHITECTURE (45 seconds)

[SCREEN] Show: docs/figures/llm_dqn_architecture.png

[SPEAK]
"As an extension, I also designed a hybrid architecture that
combines this DQN agent with a large language model.

The idea is that the DQN handles what it does best —
real-time per-step control at 50 Hz. The LLM handles what
it does best — understanding natural language goals and
explaining agent behavior.

In Layer 1, a user describes a goal in plain English —
like 'land gently near the left flag.' The LLM translates
this into reward shaping parameters for the environment wrapper.

In Layer 2, the DQN trains using the shaped reward.
Every timestep: state in, action out. The LLM is not in
this real-time path at all.

In Layer 3, after each episode, an LLM analyses the trajectory
and generates a natural language explanation — 'the agent fired
the main engine too late, causing excessive vertical speed.'
A human supervisor can then refine the goal.

This is the design principle behind systems like SayCan from
Google DeepMind and VOYAGER for Minecraft."


---

## ⏱️ 7:45 — CODE WALKTHROUGH (60 seconds)

[SCREEN] Open GitHub repo: github.com/PragatiAN1109/deep-q-learning-atari-agent
[ACTION] Navigate to src/ folder, show key files

[SPEAK]
"Let me quickly show you the code structure.

Everything lives in the src/ folder. The three core files are:

model.py — the DQN neural network. It's a clean PyTorch Module
with just a Sequential block. Input is the 8-dim state, output
is 4 Q-values. Simple, readable, 135 lines.

replay_buffer.py — the experience replay buffer. It uses
Python's deque with maxlen for O(1) circular insertion.
The sample method returns NumPy arrays with the correct dtypes
— float32 for states and rewards, int64 for actions — so they
convert to PyTorch tensors with zero copies.

agent/dqn_agent.py — the full agent. It holds both networks,
runs the Bellman update with gradient clipping, handles
epsilon decay, and saves and loads checkpoints.

All hyperparameters are centralized in config.yaml — so every
experiment is a one-line git diff. No code changes needed
to run a different configuration.

The experiments/ folder has YAML configs for each experiment,
and src/experiments/run_all.py runs all of them sequentially
and produces comparison plots."

[ACTION] Show config.yaml briefly, then show experiments/exp_baseline.yaml

---

## ⏱️ 8:45 — CLOSING (30 seconds)

[SCREEN] Show: docs/figures/reward_vs_episode.png one final time

[SPEAK]
"To summarize — I built a complete Deep Q-Network agent from
scratch using PyTorch and Gymnasium. The agent trained for 500
episodes, achieving a maximum reward of 281 and a final
100-episode average of 157, with a clear upward trend
showing the agent was still actively improving.

The project includes controlled hyperparameter experiments,
an exploration strategy comparison, full documentation,
and a proposed LLM integration architecture.

The full code, trained model, Jupyter notebook, and all
documentation are available on GitHub at the link below.

Thank you for watching."

[SCREEN] Show GitHub URL and end card:
  github.com/PragatiAN1109/deep-q-learning-atari-agent

---

## 📋 TOTAL TIMING BREAKDOWN

| Segment                         | Start  | Duration |
|---------------------------------|--------|----------|
| Title + intro                   | 0:00   | 0:30     |
| Problem statement + environment | 0:30   | 0:45     |
| Why DQN + Bellman equation      | 1:15   | 0:45     |
| Architecture                    | 2:00   | 0:30     |
| Training results (reward curve) | 2:30   | 1:30     |
| Convergence curve               | 4:00   | 0:30     |
| Hyperparameter experiments      | 4:30   | 1:00     |
| Epsilon decay & exploration     | 5:30   | 0:45     |
| Performance metrics summary     | 6:15   | 0:45     |
| LLM + DQN architecture          | 7:00   | 0:45     |
| Code walkthrough                | 7:45   | 1:00     |
| Closing                         | 8:45   | 0:30     |
| **TOTAL**                       |        | **9:15** |

---

## 🖥️ SCREEN RECORDING CHECKLIST

Before recording:
- [ ] Open GitHub repo in browser (zoomed in for readability)
- [ ] Have all figures open and ready to switch between:
      - docs/figures/lunarlander_screenshot.png
      - docs/figures/bellman_update_flow.png
      - docs/figures/reward_vs_episode.png
      - docs/figures/moving_avg_reward.png
      - docs/figures/epsilon_decay.png
      - docs/figures/llm_dqn_architecture.png
- [ ] Notebook open at: DQN_LunarLander_Colab.ipynb
- [ ] config.yaml open in VS Code or text editor
- [ ] Microphone tested
- [ ] Notifications silenced
- [ ] Resolution: 1920×1080 minimum

Tools:
  Mac: QuickTime → New Screen Recording
  Cross-platform: OBS Studio (free), Loom, Zoom (record yourself)

---

## 🗣️ SPEAKING TIPS

1. Speak at 80% of your normal speed — recordings always
   feel faster than they sound in your head.

2. Pause 1 second before each new section — gives viewers
   time to read the screen and adjust.

3. Point to specific parts of each chart when you mention them
   — use your cursor as a pointer.

4. For the reward curve, trace the orange moving average line
   with your cursor as you describe each phase.

5. You don't need to memorize this — glance at talking points
   and speak naturally. Authenticity > perfection.

6. If you make a mistake, pause 3 seconds and re-do the sentence.
   Easy to edit in post.
