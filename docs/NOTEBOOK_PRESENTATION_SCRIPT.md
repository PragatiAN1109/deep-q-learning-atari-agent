# 🎬 NOTEBOOK PRESENTATION SCRIPT
## Deep Q-Learning Agent — LunarLander-v3
### Pragati Narote | NEU MSIS | Spring 2025

---

**HOW TO USE THIS SCRIPT**
- Open your Colab notebook on one screen, this script on the other
- Each section tells you exactly which cell to scroll to and what to say
- `[SCROLL TO]` = scroll notebook to that cell
- `[POINT TO]` = hover cursor over that part of the screen
- `[SPEAK]` = say this out loud
- ⏱️ = suggested running timestamp
- Total runtime: **8–9 minutes**

---

## ⏱️ 0:00 — INTRO (30 seconds)

[SCROLL TO] Cell 1 — Title cell at the very top of the notebook

[SPEAK]
"Hi, I'm Pragati Narote. This is my Deep Q-Learning project for my
Machine Learning course at Northeastern.

I'm going to walk you through this Jupyter notebook, which implements
a complete Deep Q-Network agent that teaches itself to land a spacecraft.

The notebook has 13 sections — everything from the environment setup,
to the training loop, to hyperparameter experiments. All the outputs
and plots you'll see are already pre-executed from a real training run.

Let's start from the top."

---

## ⏱️ 0:30 — SECTION 1: INSTALLATION (20 seconds)

[SCROLL TO] Cell 2–3 — Installation section

[POINT TO] The install cell

[SPEAK]
"Section 1 is the Colab install cell. If you're running this on Google Colab,
you run this cell first. It installs swig, box2d-py, and gymnasium in the
correct order — swig has to come before box2d, otherwise the C extension
fails to compile.

It also auto-restarts the runtime when done, which is necessary to avoid
a NumPy binary incompatibility.

I'm not running this now because we're already set up, but this is the
first thing to run if you clone the repo."

---

## ⏱️ 0:50 — SECTION 2: IMPORTS (20 seconds)

[SCROLL TO] Cell 4–5 — Imports section

[POINT TO] The output of Cell 5 showing versions

[SPEAK]
"Section 2 is imports and setup. You can see here the output already shows
the library versions we're running — Gymnasium 1.1.1, PyTorch 2.2.2,
NumPy 1.26.4, running on CPU.

The global seed is set to 42 across Python, NumPy, and PyTorch —
that's what makes all results in this notebook reproducible."

---

## ⏱️ 1:10 — SECTION 3: ENVIRONMENT (75 seconds)

[SCROLL TO] Cell 6–7 — Environment Exploration section

[POINT TO] The text output showing observation space details

[SPEAK]
"Section 3 is the environment. We're using LunarLander-v3 from Gymnasium.

Look at this output — the observation space is a Box with 8 dimensions,
all continuous floats. The action space is Discrete with 4 actions.

Let me read through the 8 state dimensions:
x and y position, x and y velocity, the lander's angle,
its angular velocity, and two binary flags for whether
the left and right legs are touching the ground."

[SCROLL TO] Cell 8 — Environment screenshot

[POINT TO] The rendered frame image

[SPEAK]
"This is what the environment looks like at the start of an episode.
The lander spawns near the top and must reach the landing pad
between the two flags. You can see at the bottom the full initial
state vector — it starts centered with near-zero velocity."

[SCROLL TO] Cell 9 — Random episode rewards

[POINT TO] The plot showing step-by-step rewards

[SPEAK]
"This is what happens when a completely untrained agent takes
random actions. The rewards bounce between small negatives and positives
each step, and the episode ends with a large crash penalty.
Total reward: around negative 250. This is our baseline — what we're
trying to improve on."

---

## ⏱️ 2:25 — SECTION 4: DQN MODEL (40 seconds)

[SCROLL TO] Cell 10–11 — DQN Model section

[POINT TO] The class definition, then the output showing shapes

[SPEAK]
"Section 4 defines the Q-Network. Here's the architecture — it's a
simple 3-layer fully-connected network built in PyTorch.

Input is the 8-dimensional state vector.
Two hidden layers of 128 neurons each, with ReLU activations.
Output is 4 Q-values — one per action.

Look at the output here: input shape is batch-size 4 by 8,
output is 4 by 4. That's correct — 4 Q-values for 4 actions.
Total trainable parameters: 18,308. Very lightweight.

Below that is the ReplayBuffer class — a circular buffer using
Python's deque. It stores up to 50,000 transitions and samples
them randomly during training to break temporal correlation."

---

## ⏱️ 3:05 — SECTION 5: DQN AGENT (45 seconds)

[SCROLL TO] Cell 12–13 — DQN Agent section

[POINT TO] The class definition, highlight the learn() method

[SPEAK]
"Section 5 is the full DQN Agent class.

The key method is learn(). Let me walk through what it does.

First, it samples a random mini-batch of 64 transitions from the buffer.
Then it converts everything to PyTorch tensors.

For the current Q-values — it calls the online network on the states,
then uses gather to pick out only the Q-value for the action that
was actually taken. That's this line here.

For the Bellman target — it calls the target network on the next states,
takes the maximum Q-value, and computes:
reward plus gamma times that max, masked by done.
When done equals 1 — meaning the episode ended — the future term is
zero, because there's no next state.

Then MSE loss, zero gradients, backprop, gradient clip to max-norm 1,
and optimizer step.

The decay method updates epsilon exponentially.
The sync method hard-copies online weights to the target network."


---

## ⏱️ 3:50 — SECTION 6: BASELINE TRAINING (60 seconds)

[SCROLL TO] Cell 14–15 — Baseline Training section

[POINT TO] The hyperparameters dict first, then the training output

[SPEAK]
"Section 6 is the baseline training run. Here are the hyperparameters —
I'll point out the key ones.

Learning rate: 0.0005 with Adam optimizer.
Gamma: 0.99 — this is the discount factor, and I'll explain why
this specific value matters in a moment.
Epsilon starts at 1.0 — fully random — and decays exponentially
toward 0.01 using a decay rate of 0.005.
Batch size 64, replay buffer capacity 50,000 transitions.
Target network synced every 10 episodes.

Now look at the training output. This is real — these are the actual
results from running 500 episodes on this machine.

At episode 50 — reward is negative 220, average 100 is negative 168.
The agent is still mostly crashing.

By episode 200 — reward is starting to climb. Average 100 is around
negative 60.

By episode 400 — individual episodes are positive. Average 100 is 47.

By episode 500 — individual episodes are consistently positive,
some above 200. The final average over 100 episodes: 157.03.

The solve threshold is 200. We didn't quite reach it in 500 episodes —
we'd need around 700 to 800 episodes to fully solve it — but the
strong upward trend shows the agent is clearly learning."

---

## ⏱️ 4:50 — SECTION 7: VISUALIZATIONS (75 seconds)

[SCROLL TO] Cell 16–17 — Training Dashboard (3-panel plot)

[POINT TO] Top panel first, then step through each panel

[SPEAK]
"Section 7 is where it gets visual. This three-panel dashboard shows
everything that happened during training.

Top panel — rewards. The blue line is the raw per-episode reward —
very noisy. The orange line is the 50-episode moving average.
The red dashed line is the 100-episode rolling average.
The green dotted line is the solve threshold at 200.

Trace the orange line with me — it starts around negative 200,
stays flat through the first 100 episodes while the buffer fills,
then you see a steady climb from episode 150 onward as epsilon
decays and the policy improves. By episode 400, we're consistently
in positive territory. That's the learning signal working.

Middle panel — steps per episode. Early on, the agent crashes in
under 200 steps. Later, it's surviving full 1000-step episodes —
the agent has learned to stay alive and maneuver.

Bottom panel — epsilon decay. Watch how it drops from 1.0 toward
0.01. At episode 140, epsilon crosses below 0.5 — that's when
exploitation starts to dominate over exploration."

[SCROLL TO] Cell 18 — Moving average convergence

[POINT TO] The peak annotation and the two average lines

[SPEAK]
"Here's the convergence curve more clearly — just the moving averages
without the noisy raw rewards.

The peak 100-episode average reached was 157 — right here.
The orange 50-episode average gets as high as 183 by the end,
showing the agent was still actively improving at episode 500."

[SCROLL TO] Cell 19 — Epsilon decay

[POINT TO] The explore/exploit shaded zones

[SPEAK]
"And the epsilon curve in detail. Orange zone — exploration dominated.
Blue zone — exploitation dominated. The crossover at episode 140.
The formula in the top right corner: epsilon equals max of 0.01 and
e to the power of negative 0.005 times the episode number."

---

## ⏱️ 6:05 — SECTIONS 8–9: HYPERPARAMETER EXPERIMENTS (60 seconds)

[SCROLL TO] Cell 20–21 — Learning Rate Sweep

[POINT TO] The three curves in the plot

[SPEAK]
"Section 8 is hyperparameter experiments. First, the learning rate sweep.

Three values: 0.0001 in blue, 0.0005 in orange — our baseline —
and 0.001 in green. Each run is 100 episodes.

At a higher learning rate of 0.001, you see faster initial improvement
but more instability. At the lower rate, slower but smoother.
The baseline 0.0005 hits the right balance for this environment."

[SCROLL TO] Cell 22 — Gamma sweep

[POINT TO] The three gamma curves, especially the γ=0.90 line

[SPEAK]
"Now gamma — the discount factor. This is the most important parameter.

Three values: 0.90 in red, 0.95 in orange, 0.99 in green.

Look at gamma 0.90 — it performs the worst. Here's the math:
the landing bonus of plus 100 arrives about 200 steps in.
0.9 to the power of 200 is essentially zero — the agent is
completely blind to whether it eventually lands.

At 0.99, that same bonus retains 13% of its value at step zero.
That's enough signal to drive the policy toward landing.

This is why I changed the assignment's original value of 0.8 to 0.99."

[SCROLL TO] Cell 23–24 — Epsilon decay experiments

[POINT TO] The two panels — reward curves and epsilon curves

[SPEAK]
"Section 9 compares three epsilon decay rates.

Fast decay at 0.015 — the agent commits to exploitation by episode 150.
It improves quickly but can get stuck in a suboptimal policy.

Medium at 0.005 — our baseline — reaches a good balance.

Slow at 0.002 — the agent keeps exploring almost the entire run.
You can see in the bottom panel how slowly epsilon falls.

The bottom panel maps the epsilon schedule for each rate — fast drops
steeply, slow barely moves over 100 episodes."

---

## ⏱️ 7:05 — SECTION 10: EXPLORATION COMPARISON (30 seconds)

[SCROLL TO] Cell 25–26 — Exploration Strategy

[POINT TO] The three curves in the comparison plot

[SPEAK]
"Section 10 compares exploration strategies — epsilon-greedy versus
Softmax, also called Boltzmann exploration.

Softmax converts Q-values into probabilities. Higher temperature
means more exploration. Lower temperature is greedier.

For this environment, epsilon-greedy with scheduled decay outperforms
both Softmax variants because the exponential decay gives a clear
transition from exploration to exploitation that a fixed temperature
Softmax cannot match."

---

## ⏱️ 7:35 — SECTION 11: PERFORMANCE METRICS (40 seconds)

[SCROLL TO] Cell 27–28 — Performance Metrics output

[POINT TO] The printed numbers first

[SPEAK]
"Section 11 is the quantitative results — the numbers for Section 5.1.

Average reward over the last 100 episodes: 157.03.
Average steps per episode: 507.8.
Maximum reward in a single episode: 281.85, at episode 466."

[SCROLL TO] Cell 29 — Performance metrics chart

[POINT TO] The bar chart on the left, then the table on the right

[SPEAK]
"The bar chart shows average reward by training phase.
Early training averaged negative 156. Final 100 episodes averaged
positive 157. That's a swing of over 300 reward points.

The table on the right collects all the key metrics in one place —
useful for the report."


---

## ⏱️ 8:15 — SECTION 12: EVALUATION (30 seconds)

[SCROLL TO] Cell 30–31 — Greedy Evaluation output

[POINT TO] The printed episode-by-episode results

[SPEAK]
"Section 12 evaluates the trained agent with epsilon set to zero —
pure exploitation, no random actions at all.

Ten evaluation episodes. You can see the individual rewards here.
Some episodes reach over 100, one hits 183. The mean is 0.27 for
this particular model — this is a 300-episode trained model used
for the eval demo, not the full 500-episode model.

With the full 500-episode model loaded, these numbers improve
significantly — as we saw from the training output earlier where
individual episodes were regularly hitting 150 to 200 plus."

[SCROLL TO] Cell 32 — Evaluation bar chart

[POINT TO] The colored bars — green for above 200, blue for positive, red for negative

[SPEAK]
"The bar chart color-codes each episode — green means the agent
solved it that run, blue means positive but below threshold,
red means crash. The orange dashed line is the mean.

The agent isn't consistent yet — it needs more training — but
the trend is clearly moving in the right direction."

---

## ⏱️ 8:45 — SECTION 13: FINAL SUMMARY + CLOSE (30 seconds)

[SCROLL TO] Cell 33–34 — Final Summary

[POINT TO] The printed summary output

[SPEAK]
"Section 13 is the final summary. All the key numbers in one place.

Environment: LunarLander-v3. Architecture: 8 to 128 to 128 to 4 MLP.
18,308 parameters.

Section 5.1 results: average reward last 100 episodes 157.03,
average steps 507.8, maximum reward 281.85.

Section 9 metrics: the agent converges around episode 430,
transitions from exploration to exploitation at episode 140,
and was still actively improving at episode 500.

The agent didn't fully solve the environment in 500 episodes —
the solve threshold is 200 — but the strong trend shows it would
get there with more training.

The full code, trained model, and all documentation are on GitHub.
Thank you for watching."

[ACTION] Pan camera to show GitHub URL if recording yourself,
         or end screen recording.

---

## 📋 FULL TIMING SUMMARY

| ⏱️ Time | Section | Notebook Cells | Duration |
|---------|---------|----------------|----------|
| 0:00 | Intro | Cell 1 (title) | 0:30 |
| 0:30 | Installation | Cell 2–3 | 0:20 |
| 0:50 | Imports & setup | Cell 4–5 | 0:20 |
| 1:10 | Environment | Cell 6–9 | 1:15 |
| 2:25 | DQN model + buffer | Cell 10–11 | 0:40 |
| 3:05 | DQN agent | Cell 12–13 | 0:45 |
| 3:50 | Baseline training | Cell 14–15 | 1:00 |
| 4:50 | Training visualizations | Cell 16–19 | 1:15 |
| 6:05 | HP + epsilon experiments | Cell 20–24 | 1:00 |
| 7:05 | Exploration comparison | Cell 25–26 | 0:30 |
| 7:35 | Performance metrics | Cell 27–29 | 0:40 |
| 8:15 | Evaluation | Cell 30–32 | 0:30 |
| 8:45 | Summary + close | Cell 33–34 | 0:30 |
| **TOTAL** | | | **~9:15** |

---

## 🎥 SETUP CHECKLIST BEFORE RECORDING

- [ ] Colab notebook open — `DQN_LunarLander_Colab.ipynb` uploaded
- [ ] All cells collapsed (View → Collapse all sections) so you start clean
- [ ] Browser zoom: 100–110% so text is readable
- [ ] Script open in a second window or on a phone/tablet beside you
- [ ] Screen resolution: 1920×1080 minimum
- [ ] Microphone tested — check audio levels
- [ ] Notifications off (Do Not Disturb on Mac: top right menu)
- [ ] Close all unrelated tabs

**Recording tools:**
- Mac built-in: QuickTime → File → New Screen Recording
- Free cross-platform: OBS Studio
- Quick share: Loom (records + uploads automatically)

**Pro tip:** Record in sections, not one take.
Do the intro, stop, check it, then do the next section.
Easier to re-do 90 seconds than 9 minutes.

---

## 🗣️ KEY THINGS TO SAY FOR EACH PLOT

**3-panel dashboard (Cell 17):**
*"Trace the orange line — flat early, then steady climb after episode 150."*

**Moving avg convergence (Cell 18):**
*"Peak 100-ep average: 157. Still trending up at episode 500."*

**Epsilon decay (Cell 19):**
*"Crossover at episode 140 — more exploitation than exploration from here."*

**LR sweep (Cell 21):**
*"0.0005 hits the right balance — 0.001 is faster but noisier."*

**Gamma sweep (Cell 22):**
*"0.9 to the power of 200 is essentially zero — that's why gamma 0.99 wins."*

**Metrics table (Cell 29):**
*"157.03 average reward, 507.8 average steps, 281.85 maximum."*
