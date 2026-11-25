# Ideas and roadmap

This list comes from reading what other projects did: Rocket League bots (Necto, Nexto, Ripple,
Seer), large game-AI projects (AlphaStar, OpenAI Five, AlphaGo Zero, DeepNash, Honor of Kings) and
previous Clash Royale work (crforge, RoyaleGym, ClashAI, Hasty-CR, KataCR, FirstLight, Daybreak).
Ideas are tested one at a time, each against the same reference run. Sources are at the end.

## What others learned the hard way

- **RL can make an imitation bot worse.** In Hasty-CR, a BC bot at 75 % fell to 35 % at the start of
  PPO because the critic was not warmed up. In ClashAI, RL only added 3 points over imitation.
- **Self-play from scratch does not go far.** 30 minutes of PPO without imitation gave 0 wins out of
  128 against crforge's scripted bot.
- **Never play the most likely action.** It is almost always "wait", so the bot becomes passive.
  Sample instead, and only sharpen the card and tile choice.
- **Too much entropy kills the Hog plan** (Hasty-CR). Keep the coefficient under 0.01.
- **Too much pure self-play overfits** (Hasty-CR at 70 %). Good projects use 35 to 80 %, the rest
  against past versions.
- **A win rate against the BC bot can lie** (FirstLight): one of their checkpoints showed 80 %, but
  296 of its 306 wins came from games where the opponent played 3 cards or fewer. We always count the
  opponent's cards during checks.
- **crforge's scripted bot is weak.** Against itself it gets 131 wins, 150 losses and 19 draws. It
  catches bugs, it does not measure strength.

## Done

| Idea | Where |
|---|---|
| BC warm start, KL penalty towards BC with a floor, critic warm-up | `train/ppo.py --value-warmup --kl-floor` |
| Pool of past versions + pure self-play | `train/ppo.py --pool-size --self-frac` |
| PFSP opponent sampling (AlphaStar) | `train/ppo.py --pfsp` |
| Privileged critic: the value head sees the opponent's hand and elixir during training only | `train/ppo.py --privileged-critic` |
| Elo league and worst score against older versions | `eval/league.py` |
| Real-game conditions: random levels and forms, noisy input, action delay | `train/ppo.py --levels --random-slots --obs-noise --action-delay` |
| Moving KL anchor and larger pool | `train/ppo.py --moving-anchor` (the anchor never moved in practice) |
| Fixed BC opponent group | `train/ppo.py --human-opponent` |
| Permanent tower bonus and longer horizon | `train/ppo.py --shaping-floor 0.2 --gamma 0.999` |
| BC on all IL_Replay games with a 4.6 M network, then 2.6 fine-tuning | `train/bc.py --init` |
| Tile lookahead in the digital twin on the real game (+21 points in simulation) | `bridge/pc/` |

## Next, in order

1. **A fixed panel every time**: the BC bot, it1450 and reel_it0400, at temperature 0.5, 400 games,
   counting cards per minute. The score that matters is the worst one, not the last one.
2. **A short bug audit**: tiles used per card, ties between tiles, refused moves.
3. **A decaying learning rate (5e-5 to 5e-6) and 2 to 4 times bigger batches**, like OpenAI Five and
   the RLGym guide.
4. **A small fixed KL towards BC next to the moving anchor** (+380 Elo in AlphaStar).
5. **A realistic random delay** (0.25 to 0.5 s), with the bot seeing its card on the way.
6. **Some BC on the 2.6 games won during PPO** (FirstLight's recipe).
7. **An exploiter** trained from the BC bot against the frozen main bot, first as a diagnostic: a bot
   that an exploiter beats in 2 hours has a weakness (AlphaStar: 1,540 to 1,824 Elo with exploiters).
8. **The opponent's cycle as a computed input**: in the mirror match the opponent's hand is known
   after 4 plays.
9. **Search distillation**: train the policy to play directly what the lookahead would pick. Search
   is the biggest known gain (AlphaGo Zero, Boney et al. on Clash Royale), but distilling it in one
   pass only keeps 4 to 14 % of the gain, so this needs care.
10. **Several decks**: one teacher per deck, then one student that imitates all of them (Honor of
    Kings).

## Not now

An LSTM or GRU, a bigger actor alone, resetting layers, more epochs on the same data, a search that
can pick "wait" (it made the bot passive: 91 % of its moves were cancelled in simulation), a
temperature under 0.5 (measured, no effect).

## Sources

- Rocket League: [Necto](https://github.com/Rolv-Arild/Necto),
  [rocket-learn](https://github.com/Rolv-Arild/rocket-learn),
  [Ripple](https://wandb.ai/rolv-arild/ripple/reports/Ripple--VmlldzozNDE5NzE4),
  [RLGym PPO guide](https://github.com/ZealanL/RLGym-PPO-Guide),
  [Lucy-SKG](https://arxiv.org/abs/2305.15801)
- Large projects:
  [AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf),
  [OpenAI Five](https://arxiv.org/abs/1912.06680), [AlphaZero](https://arxiv.org/abs/1712.01815),
  [DeepNash](https://arxiv.org/abs/2206.15378), [Honor of Kings](https://arxiv.org/abs/2011.12692)
- Clash Royale: [crforge issue 746](https://github.com/voonhous/crforge/issues/746),
  [crforge PR 708](https://github.com/voonhous/crforge/pull/708),
  [RoyaleLearn](https://github.com/RoyaleGym/RoyaleLearn),
  [ClashAI results](https://github.com/vegetableleaf/ClashAI/blob/main/RESULTS.md),
  [Hasty-CR RL decisions](https://github.com/hastylmao/Hasty-CR/blob/main/docs/RL_SPRINT4_DECISIONS.md),
  [Boney et al.](https://arxiv.org/abs/2012.12186),
  [FirstLight training history](https://github.com/Jaasssoooonnnnn/FirstLight_CR/blob/main/docs/training_history.md),
  [Daybreak training notes](https://github.com/Jason-XII/daybreak-cr/blob/main/docs/TRAINING.md)
