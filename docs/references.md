# References

## Simulators and environments

- [crforge](https://github.com/voonhous/crforge): open-source Clash Royale simulator (Java engine,
  Python env), tick-for-tick fidelity checked on real battles. Main simulator of this project.
  Its [environment docs](https://github.com/voonhous/crforge/blob/main/docs/environment.md) describe
  the replay format, `ReplayLabeller` and `label_replays`.
- [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym): Python env with a Rust engine (RoyaleSim) on the
  current card table. It can download, filter and replay IL_Replay. Used for the second pipeline in
  `rg/`.
- [Hasty-CR](https://github.com/hastylmao/Hasty-CR): simulator and RL agent for the 2.6. Shows that
  imitation + PPO learns on this deck, but results are only inside its own simulator.

## Data

- [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay): about 252,000 real games, mostly
  Path of Legends, with both decks and every card played (card, tile, tick). Used for imitation.
- [RoyaleGym guide "Cloning a bot"](https://royalegym.github.io/RoyaleGym/clash-royale/cloning-a-bot/):
  describes IL_Replay and the script that downloads, filters and replays it.
- [RoyaleImitate](https://github.com/RoyaleGym/RoyaleImitate): source of the IL_Replay to game card
  name table (`sim/il_card_names.json`), MIT.
- [Clash-Royale-Replay-Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset) and the replay
  downloader in [ClashAI](https://github.com/vegetableleaf/ClashAI) (`tools/`).

## Playing the real game

- [Playing Non-Embedded Card-Based Games with RL](https://arxiv.org/abs/2504.04783) (arXiv 2504.04783):
  plays on a real phone with YOLOv8, tracking and offline RL.
- [KataCR](https://github.com/wty-yy/KataCR): the code of that paper (MIT). Its YOLOv8 detectors are
  used by the PC kit.
- [Google Play Games for PC](https://support.google.com/googleplay/answer/11358071): runs Clash Royale
  officially on Windows. The kit targets it because the game refuses to run on Android emulators.

## Other Clash Royale bots

- [FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) (Apache-2.0): same method as ours
  (imitation on IL_Replay, then PPO self-play), with a general model specialized into two 2.6 bots.
  Plays on a private server, not on the real game.
- [Daybreak CR](https://github.com/Jason-XII/daybreak-cr): same method, with
  [training notes](https://github.com/Jason-XII/daybreak-cr/blob/main/docs/TRAINING.md) on temperature
  and tile search. Its author writes that simulator gains don't clearly show on the ladder.
- [ClashAI](https://github.com/vegetableleaf/ClashAI): X-Bow bot,
  [results](https://github.com/vegetableleaf/ClashAI/blob/main/RESULTS.md).
- [RoyaleFish](https://github.com/fabiospalla9-tech/RoyaleFish): 17.5 M parameter 2.6 bot with human
  "ghost" opponents.
- [ClashMaster](https://github.com/rajerthat1/ClashMaster/tree/main/crforge_bridge): runs FirstLight
  models inside crforge.
- [itzik123/ClashRoyaleAi](https://github.com/itzik123/ClashRoyaleAi/blob/main/docs/DECISIONS.md).

## Self-play, RL and imitation

- [Necto](https://github.com/Rolv-Arild/Necto) (Rocket League): PPO self-play with an opponent pool
  and an entity attention encoder, the model for our policy network.
- [RLGym PPO guide](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/learner_settings.md),
  [Ripple](https://wandb.ai/rolv-arild/ripple/reports/Ripple--VmlldzozNDE5NzE4),
  [Seer thesis](https://nevillewalo.ch/assets/docs/MA_Neville_Walo_Seer_RLRL.pdf).
- [AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
  (league training, PFSP, privileged critic),
  [AlphaStar Unplugged](https://arxiv.org/abs/2308.03526), [OpenAI Five](https://arxiv.org/abs/1912.06680),
  [Honor of Kings](https://arxiv.org/abs/2011.12692).
- PPO stability: [Moalla et al.](https://arxiv.org/abs/2405.00662) (collapse),
  [Juliani and Ash](https://arxiv.org/abs/2405.19153) (plasticity),
  [McCandlish et al.](https://arxiv.org/abs/1812.06162) (useful batch size).
- Delays and memory: [Bouteiller et al.](https://arxiv.org/abs/2010.02966),
  [POPGym](https://arxiv.org/abs/2303.01859).
- Imitation at scale: [Tuyls et al.](https://arxiv.org/abs/2307.09423),
  [Metamon](https://arxiv.org/abs/2504.04395).
