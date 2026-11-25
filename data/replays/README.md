# data/replays/

Play-by-play replays for imitation.

Source: the IL_Replay dataset (Hugging Face). `scripts/download_il_replay.py` gets the raw files into `il_replay_all/`, and `scripts/filter_26.py` keeps the games where one of the two decks is exactly the 2.6 (`il_replay_26/`). These replays are then replayed in crforge (`sim/il_to_crforge.py`) to produce the imitation data.

Each play: card played, position on the grid, timing.

This folder is ignored by git (except this README). Never commit replays.
