# Experiment log

What was trained, how it was measured, and what we learned. Unless said otherwise, a head-to-head
score is measured with `eval/head_to_head.py --real-game` over 400 games: both sides get the noisy
input of the PC kit and a 0.25 s action delay, and both play at temperature 0.5 ("inner" mode). With
400 games the 95 % margin is about ±4.9 points.

## 1. Data

- IL_Replay has 252,238 real games. 14,040 of them (5.6 %) have at least one exact 2.6 deck, 12,492 of
  those are Path of Legends games and 265 are mirrors.
- 822 games are dropped because the opponent plays a card missing from the name table, which leaves
  13,218 games.
- What a replay gives: card levels, evolutions and heroes, slot order, and every card played with its
  tile and tick (1/20 s). What it does not give: seed, starting hand, game version, starting tower HP,
  and whether a card was played in its evolved form.
- In 2026 the 2.6 almost always runs evolutions and a hero. The most played variant (8,552 decks) is
  evolved Cannon + hero Musketeer + evolved Skeletons. This is the one the bot plays, at level 16.
- Meta: one file of 5,000 Path of Legends games has 2,054 different decks for 10,000 players. The most
  played deck is only 2.5 % of them.

## 2. Replaying the games in crforge

A replay only has the moves, not the board. To imitate "in this situation the player did this", the
situation has to be rebuilt, and it has to use the same observation format as RL. So every game is
replayed in crforge (`sim/il_to_crforge.py`):

- the starting hand and the cycle order are inferred from the order of the cards played;
- the deck is ordered so that crforge's shuffle deals exactly that hand, so any seed works;
- moves of both players are sent as actions every 5 ticks. A refused move is retried for 1 s, then
  dropped.

Result on the 2.6 games: 13,218 games, 1.7 M rows, 598,000 plays on the 2.6 side, 796 MB, one simulator
error. About 85 % of the 2.6 plays go through, about 1 % late and about 15 % dropped. Most drops come
from games that the simulator ends earlier than the real one (the state drifts because the combat RNG
is not the real one). On a Mac with 6 threads this runs at about 7 games per second.

On all decks and both players: 84 % of the plays go through, 15 games per second with 6 threads.

## 3. First baseline: BC then PPO

**Behavior cloning.** 1.19 M parameters (d_model 128, 4 heads, 3 layers). Only about 4 % of the
decision steps contain a play, so the converter keeps 10 % of the "wait" steps and a log(10) bias is
added back to the wait logit when saving. 6 epochs took 14 minutes. The loss was still going down
(train 2.38 to 1.905, validation 2.19 to 1.920), so this model was under-trained.

At temperature 1 the BC bot plays 52 cards per game (humans: 54) and beats crforge's scripted bot
75 % of the time. Greedy play does not work: the most likely action is almost always "wait", and at
temperature 0.5 applied to everything the bot plays one card per game. This is why temperature is
only applied to the card and tile choice ("inner" mode).

**PPO self-play.** Half the games are pure self-play (the bot plays both seats), half are against a
pool of past versions. Reward: +1/-1 at the end plus a tower-damage bonus that decays to 0 over 3,000
iterations. KL penalty towards the BC policy, decaying over 2,000 iterations with a floor at 0.01.
The critic alone is trained for the first 20 iterations. About 41 M decisions per hour on one cluster
node.

The in-training check against the starting model went up to 97 % (checkpoint it1450, 84 % against the
scripted bot), then dropped to about 80 % by the end of the run, after the shaping bonus reached 0.

## 4. Training for the real game

The PC kit does not see the game like the simulator does. So PPO was restarted from it1450 with:
random card and tower levels per game, evolutions/heroes present or not, a corrupted observation
(20 % of units missed, 0.3 tile position noise, 15 % of enemy plays missed) and a one-step action
delay.

Run of 2.9 h, 2,010 iterations:

| Match | Score |
|---|---|
| reel_it0400 vs it1450 | 61 % |
| reel_it0400 vs reel_it2010 | 61 % |
| reel_it2010 vs it1450 | 55 % |

The same gaps show up with perfect input. The best checkpoint is iteration 400 (about 35 to 45
minutes in), then the run stalls.

Temperature: 0.5 beats 1.0 (58 %). Going lower gives nothing (0.25 and 0.1 score 48.5 % and 51 %
against 0.5).

Against the BC bot, the most "human" opponent we have, PPO does not help:

| Model | Score vs BC bot |
|---|---|
| it1450 | 81.5 % (800 games) |
| reel_it0400 | 79.5 % (800 games) |
| reel_it2010 | 77.2 % (400 games) |

PPO learns to beat its own past versions, not to play better against a human style.

## 5. Why PPO went backwards

A third run (moving KL anchor, pool of 20 versions) did 2,860 iterations and 141 M decisions in
3 h 56. The anchor never moved. Every checkpoint lost to reel_it0400: the best (it300) scored 49 %,
the others 30 to 46 %, with the same number of cards played per minute on both sides.

In all three runs, the drop starts when the tower-damage bonus reaches 0. The reason is simple: with
gamma 0.995 and one decision every 0.25 s, the final reward only weighs half 35 s before the end, and
about 3 % at the start of a 3-minute game. Without the bonus, the early game learns almost nothing
and drifts. Projects that work keep a permanent tower bonus (FirstLight, Daybreak) and use a gamma of
0.999 (ClashAI) or 0.9997 (FirstLight).

Changes: `--shaping-floor` (the bonus never goes below this value) and gamma 0.999.

## 6. A bigger BC model on all the data

| Setting | Before | After |
|---|---|---|
| Parameters | 1.19 M | 4.56 M (d_model 256, 8 heads, 4 layers, trunk 512) |
| Data | 2.6 players of 13,218 games | both players of all IL_Replay games (52 files, 0.83 GB, about 260,000 games, 85 % readable) |
| Training | 2.6 only | pre-training on all decks, then fine-tuning on 2.6 |

The learning-rate schedule now follows whichever is further along, the steps or the wall-clock time
(`--max-hours`). Before, a run cut by the time limit ended with a learning rate that was still high.

Results, all at equal cards per minute (11.4 on both sides):

| Match | Score | W / L |
|---|---|---|
| bc_tout_26 (all decks, then 2.6) vs old BC | **82.8 %** | 331 / 69 |
| bc_grand (bigger net, 2.6 data only) vs old BC | 76.2 % | 305 / 95 |
| bc_tout_26 vs reel_it0400 (best PPO so far) | 54.5 % | 218 / 182 |

Model size alone gives most of the gain, the extra data adds about 6 points. The new BC model is level
with the best PPO model without any RL, so the next PPO runs start from it.

## 7. PPO from the big BC model

Settings of the run (read from its checkpoint): 512 parallel games, rollout 64, gamma 0.999,
lambda 0.95, lr 5e-5, KL coefficient 0.025 with a floor at 0.025, permanent tower bonus 0.2, PFSP
opponent sampling over a pool of 20 versions (one every 100 iterations), privileged critic, card
levels drawn from 9 to 16, random evolutions/heroes, noisy input and a one-step action delay. It ran
5,217 iterations, about 256 M decisions.

The model shipped in the PC kit is a weight average of 15 PPO checkpoints from this setup, exported
to ONNX (no gap between the torch and ONNX outputs).

## Lessons

- Imitation does most of the work. A bigger network trained longer on more data beat every PPO run.
- Never trust a single win rate. Use a fixed panel, an Elo league (`eval/league.py`) and the worst
  score against older versions, and always count cards per minute: a high score against an opponent
  that barely plays means nothing.
- Reward shaping that decays to 0 with a short horizon made every run regress. Keep a small permanent
  bonus or raise gamma.
- The simulator gap matters as much as the policy: see `bridge/README.md` for what noisy vision
  costs.
